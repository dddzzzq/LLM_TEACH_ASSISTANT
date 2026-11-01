import json
import re
import unittest

from app.services.student_evidence import evidence_chunks, grade_materials


def materials(text):
    return {'schema':'student-materials.v1','reading':{'status':'COMPLETE','warnings':[]},
            'files':[{'id':'F0001','path':'src/main.ts','role':'code','status':'READ','text':text}],
            'submission_archives':[{'id':'a','name':'20260001_张三.zip','format':'zip','student_id':'20260001','sha256':'original'}]}


def scored(refs):
    return {'overall_feedback':'依据完整材料','score_details':[{'criterion':'正确性','score':80,'max_score':100,
            'feedback':'依据实际证据','evidence_refs':refs}]}


class EvidenceSummaryTests(unittest.TestCase):
    def test_oversized_leaf_is_resummarized_without_losing_tail(self):
        data=materials('x'*25000+'\nTAIL-END')
        refs=[c['id'] for c in evidence_chunks(data)]
        seen=[];prompts=[]
        def call(prompt,*args):
            prompts.append(prompt)
            match=re.search(r'\[证据 (F\d+\.C\d+)',prompt)
            if match:
                cid=match.group(1);seen.append(cid)
                if len(seen)==1:return {'chunk_id':cid,'facts':['过长事实'*1000],'uncertainties':[]}
                return {'chunk_id':cid,'facts':['TAIL-END' if 'TAIL-END' in prompt else '实现事实'],'uncertainties':[]}
            self.assertIn('20260001_张三.zip',prompt)
            self.assertIn('不能据目录形式',prompt)
            self.assertIn('TAIL-END',prompt)
            return scored(refs)
        result=grade_materials(data,'题目',{'正确性':100},call,budget=16000)
        self.assertEqual(result['total_score'],80)
        self.assertEqual(set(seen),set(refs))
        self.assertEqual(result['reading']['summary']['leaf_requests'],len(refs)+1)
        self.assertTrue(any('[摘要纠正]' in p for p in prompts))
        self.assertEqual(result['reading']['chunks_read'],refs)

    def _hierarchical(self,drop_coverage=False):
        data=materials('x'*180000+'\nTAIL-END')
        refs=[c['id'] for c in evidence_chunks(data)]
        leaves=[];merges=[];final=[]
        def call(prompt,*args):
            self.assertLessEqual(len(prompt.encode('utf-8')),16000)
            if prompt.startswith('[分层证据汇总]'):
                group=json.loads(prompt.split('输入证据：',1)[1].split('\n[摘要纠正]',1)[0])
                covered=[cid for node in group for cid in node['covered_chunks']]
                merges.append(covered)
                return {'covered_chunks':covered[:-1] if drop_coverage else covered,
                        'facts':[{'evidence_refs':[covered[-1]],'text':'TAIL-END' if 'TAIL-END' in json.dumps(group) else '汇总实现证据'}],
                        'uncertainties':[]}
            match=re.search(r'\[证据 (F\d+\.C\d+)',prompt)
            if match:
                cid=match.group(1);leaves.append(cid)
                return {'chunk_id':cid,'facts':['TAIL-END' if 'TAIL-END' in prompt else '实现事实'+'x'*1800],'uncertainties':[]}
            final.append(prompt);return scored(refs)
        result=grade_materials(data,'题目',{'正确性':100},call,budget=16000,token_budget=16000)
        self.assertEqual(leaves,refs)
        self.assertTrue(merges)
        return result,refs,final

    def test_large_notes_use_hierarchical_merge_with_complete_coverage(self):
        result,refs,final=self._hierarchical()
        self.assertEqual(result['total_score'],80)
        self.assertGreater(result['reading']['summary']['levels'],0)
        self.assertEqual(result['reading']['chunks_read'],refs)
        self.assertIn('TAIL-END',final[0])

    def test_merge_cannot_drop_a_chunk_or_generate_a_final_grade(self):
        result,refs,final=self._hierarchical(drop_coverage=True)
        self.assertEqual(result['total_score'],-1)
        self.assertEqual(result['reading']['status'],'NEEDS_REVIEW')
        self.assertIn('覆盖',result['overall_feedback'])
        self.assertFalse(final)


if __name__=='__main__':unittest.main()
