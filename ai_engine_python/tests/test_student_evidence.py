import copy
import json
import re
import unittest
from unittest.mock import Mock

from app.services.student_evidence import evidence_chunks, grade_materials, analysis_parts


def materials(text='核心源码', role='code'):
    return {'schema':'student-materials.v1', 'reading':{'status':'COMPLETE','warnings':[]},
            'files':[{'id':'F0001','path':'src/main.ts','role':role,'status':'READ','text':text}]}


def score(refs, value=80):
    return {'total_score':value,'overall_feedback':'依据实际材料评价',
            'score_details':[{'criterion':'正确性','score':value,'max_score':100,'feedback':'有可核查依据','evidence_refs':refs}]}


class StudentEvidenceTests(unittest.TestCase):
    def test_artifacts_cannot_evict_the_code_and_report(self):
        data=materials('Promise.all(tasks)')
        data['files'].insert(0, {'id':'F0002','path':'downloads/big.txt','role':'artifact','status':'INVENTORY_ONLY','text':'x'*(7<<20)})
        calls=[]
        def call(prompt,*args):
            calls.append(prompt)
            return score(['F0001.C0001'])
        result=grade_materials(data,'并发下载',{'正确性':100},call)
        self.assertEqual(result['total_score'],80)
        self.assertEqual(len(calls),1)
        self.assertIn('Promise.all(tasks)',calls[0])
        self.assertLess(len(calls[0]),10000)
        self.assertEqual(analysis_parts(data)[1],'Promise.all(tasks)')

    def test_long_material_reads_every_chunk_including_the_tail(self):
        data=materials('a'*20000+'\n尾部核心证据-END')
        expected=[c['id'] for c in evidence_chunks(data)]
        seen=[]
        prompts=[]
        def call(prompt,*args):
            prompts.append(prompt)
            if '不打分' in prompt:
                cid=re.search(r'\[证据 (F\d+\.C\d+)',prompt).group(1)
                seen.append(cid)
                return {'chunk_id':cid,'facts':['包含尾部核心证据-END' if '尾部核心证据-END' in prompt else '包含前部实现'],'uncertainties':[]}
            return score(expected)
        result=grade_materials(data,'题目',{'正确性':100},call,budget=16000)
        self.assertEqual(seen,expected)
        self.assertEqual(result['reading']['chunks_read'],expected)
        self.assertIn('尾部核心证据-END',prompts[-1])
        self.assertEqual(result['score_details'][0]['evidence'][-1]['line_end'],2)

    def test_unreadable_material_never_becomes_a_zero_or_calls_the_model(self):
        data=materials();data['reading']['status']='NEEDS_REVIEW'
        call=Mock()
        result=grade_materials(data,'题目',{'正确性':100},call)
        self.assertEqual(result['total_score'],-1)
        self.assertEqual(result['reading']['status'],'NEEDS_REVIEW')
        call.assert_not_called()

    def test_invalid_scores_or_invented_references_require_review(self):
        for result in [score(['F9999.C0001']),score(['F0001.C0001'],101),score([])]:
            with self.subTest(result=result):
                actual=grade_materials(materials(),'题目',{'正确性':100},Mock(return_value=result))
                self.assertEqual(actual['total_score'],-1)
                self.assertEqual(actual['reading']['status'],'NEEDS_REVIEW')

    def test_total_is_computed_from_all_validated_criteria(self):
        rubric={'功能':45,'设计':{'score':20},'异常':15,'验证':10,'说明':10}
        details=[{'criterion':name,'score':value,'max_score':maximum['score'] if isinstance(maximum,dict) else maximum,
                  'feedback':'逐项证据','evidence_refs':['F0001.C0001']}
                 for (name,maximum),value in zip(rubric.items(),[40,18,12,8,9])]
        for model_total in [90,None,'90',True,float('nan'),float('inf')]:
            with self.subTest(model_total=model_total):
                response={'total_score':model_total,'overall_feedback':'按各项评价','score_details':copy.deepcopy(details)}
                call=Mock(return_value=response)
                result=grade_materials(materials(),'题目',rubric,call)
                self.assertEqual(result['total_score'],87)
                self.assertEqual(result['reading']['status'],'COMPLETE')
                self.assertEqual([d['score'] for d in result['score_details']],[40,18,12,8,9])
                call.assert_called_once()
        response={'overall_feedback':'没有模型总分','score_details':details}
        self.assertEqual(grade_materials(materials(),'题目',rubric,Mock(return_value=response))['total_score'],87)

    def test_fractional_items_are_summed_without_binary_roundoff(self):
        response={'overall_feedback':'保留小数','score_details':[
            {'criterion':name,'score':value,'max_score':1,'feedback':'证据','evidence_refs':['F0001.C0001']}
            for name,value in [('第一项',.1),('第二项',.2)]]}
        result=grade_materials(materials(),'题目',{'第一项':1,'第二项':1},Mock(return_value=response))
        self.assertEqual(result['total_score'],.3)

    def test_invalid_items_still_require_review_after_one_correction(self):
        for value in [-1,101,True,'80',float('nan'),float('inf')]:
            with self.subTest(value=value):
                call=Mock(return_value=score(['F0001.C0001'],value))
                result=grade_materials(materials(),'题目',{'正确性':100},call)
                self.assertEqual(result['total_score'],-1)
                self.assertEqual(result['reading']['status'],'NEEDS_REVIEW')
                self.assertEqual(call.call_count,2)

    def test_missing_item_gets_one_model_correction_before_summing(self):
        invalid={'overall_feedback':'首次漏项','score_details':[]}
        corrected=score(['F0001.C0001']);corrected.pop('total_score')
        call=Mock(side_effect=[invalid,corrected])
        result=grade_materials(materials(),'题目',{'正确性':100},call)
        self.assertEqual(result['total_score'],80)
        self.assertEqual(result['reading']['status'],'COMPLETE')
        self.assertEqual(call.call_count,2)
        self.assertIn('评分项缺失或重复',call.call_args.args[0])

    def test_empty_model_response_does_not_add_a_correction_retry(self):
        call=Mock(return_value=None)
        result=grade_materials(materials(),'题目',{'正确性':100},call)
        self.assertEqual(result['total_score'],-1)
        call.assert_called_once()

    def test_invalid_criteria_cannot_be_hidden_by_a_valid_model_total(self):
        rubric={'功能':60,'质量':40}
        details=[{'criterion':name,'score':value,'max_score':maximum,'feedback':'证据','evidence_refs':['F0001.C0001']}
                 for name,value,maximum in [('功能',50,60),('质量',30,40)]]
        duplicate=copy.deepcopy(details);duplicate[1]=copy.deepcopy(duplicate[0])
        unknown=copy.deepcopy(details);unknown[1]['criterion']='额外项目'
        wrong_max=copy.deepcopy(details);wrong_max[1]['max_score']=100
        for items in [details[:1],duplicate,unknown,wrong_max]:
            call=Mock(return_value={'total_score':80,'overall_feedback':'总评','score_details':items})
            result=grade_materials(materials(),'题目',rubric,call)
            self.assertEqual(result['total_score'],-1)
            self.assertEqual(result['reading']['status'],'NEEDS_REVIEW')
            self.assertEqual(call.call_count,2)

    def test_actual_zero_is_allowed_with_readable_evidence(self):
        data=materials('与编程题无关的证书','evidence')
        result=grade_materials(data,'编程题',{'正确性':100},Mock(return_value=score(['F0001.C0001'],0)))
        self.assertEqual(result['total_score'],0)
        self.assertEqual(result['reading']['status'],'COMPLETE')

    def test_duplicate_refs_or_hidden_partial_files_cannot_claim_complete(self):
        data=materials();data['files'].append(copy.deepcopy(data['files'][0]))
        call=Mock()
        result=grade_materials(data,'题目',{'正确性':100},call)
        self.assertEqual(result['reading']['status'],'NEEDS_REVIEW')
        call.assert_not_called()

    def test_pdf_page_locations_survive_grading(self):
        data=materials('[第4页]\n运行截图转写\n[第5页]\n下载完成', 'report')
        data['files'][0]['path']='report.pdf'
        result=grade_materials(data,'题目',{'正确性':100},Mock(return_value=score(['F0001.C0001'])))
        evidence=result['score_details'][0]['evidence'][0]
        self.assertEqual(evidence['pages'],[4,5])
        self.assertEqual(evidence['path'],'report.pdf')

    def test_oversized_question_is_reviewed_without_silent_truncation(self):
        call=Mock()
        result=grade_materials(materials(),'很长的题目'*10000,{'正确性':100},call,token_budget=16000)
        self.assertEqual(result['total_score'],-1)
        self.assertEqual(result['reading']['review_stage'],'SCORING')
        call.assert_not_called()


if __name__=='__main__':
    unittest.main()
