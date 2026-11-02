"""Real gRPC serialization/dispatch with fake AI services; no model downloads or GPU."""

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import grpc

from app.rpc.compute import ComputeServices, ComputeServicer
from app.rpc.protocol import pb2, pb2_grpc
from app.rpc.limits import GRPC_OPTIONS
from app.services.deepseek_service import DeepSeekService


class ComputeRPCContractTests(unittest.TestCase):
    def setUp(self):
        self.ocr_paths = []

        def extract(paths, **kwargs):
            self.ocr_paths.extend(paths)
            self.assertEqual(Path(paths[0]).read_bytes(), b"test image")
            return "识别结果"

        self.plagiarism = Mock()
        self.plagiarism.check_plagiarism_in_batch.return_value = {
            "suspicious_text_pairs": [("s1", "s2", 0.9)],
            "separated_contents": {"s1": {"text": "甲"}, "s2": {"text": "乙"}},
        }
        self.aigc = Mock()
        self.aigc.detect.return_value = {"ai_probability": 0.25}
        self.aigc._separate_content.return_value = ("说明", "code")
        self.deepseek = SimpleNamespace(
            analyze_plagiarism=Mock(return_value=({"similarity_score": 90}, None)),
            analyze_code_doc_match=Mock(return_value=({"score": 80, "reasoning": "一致"}, None)),
            grade_homework=Mock(return_value={"total_score": 85, "overall_feedback": "良好"}),
            identify_question_number=AsyncMock(return_value=2),
            grade_exam_question=AsyncMock(
                return_value={"score": 8, "feedback": "正确", "student_answer_extracted": "答案"}
            ),
            summarize_exam_performance=AsyncMock(return_value="整卷建议"),
            pool_scores=AsyncMock(return_value=[{"student_id": "s1", "score": 86}]),
        )
        services = ComputeServices(
            self.plagiarism,
            self.aigc,
            self.deepseek,
            SimpleNamespace(get_concatenated_text=extract),
        )
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.server = grpc.server(self.executor, options=GRPC_OPTIONS)
        pb2_grpc.add_ComputeServiceServicer_to_server(ComputeServicer(services), self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.assertNotEqual(port, 0)
        self.server.start()
        self.channel = grpc.insecure_channel(f"127.0.0.1:{port}", options=GRPC_OPTIONS)
        grpc.channel_ready_future(self.channel).result(timeout=5)
        self.stub = pb2_grpc.ComputeServiceStub(self.channel)
        self.addCleanup(self.close_server)

    def close_server(self):
        self.channel.close()
        self.server.stop(0).wait(timeout=5)
        self.executor.shutdown(wait=True)

    def test_large_compute_request_and_response_do_not_hit_old_four_mib_limit(self):
        body = 'x' * (6 << 20)
        response = self.stub.GradeHomework(pb2.GradeRequest(student_text=body, rubric_json='{"正确性":100}'), timeout=10)
        self.assertEqual(response.total_score, 85)
        self.assertEqual(len(response.merged_content), len(body))

    def test_structured_reading_and_scoring_evidence_round_trip(self):
        material = {'schema':'student-materials.v1', 'reading':{'status':'COMPLETE','warnings':[]},
                    'files':[{'id':'F0001','path':'src/main.ts','role':'code','status':'READ','text':'Promise.all(tasks)'}]}
        self.deepseek.grade_homework.return_value = {'total_score':85,'overall_feedback':'有依据',
            'score_details':[{'criterion':'正确性','score':85,'max_score':100,'evidence_refs':['F0001.C0001']}], 'reading':material['reading']}
        response = self.stub.GradeHomework(pb2.GradeRequest(student_text=json.dumps(material), rubric_json='{"正确性":100}'), timeout=5)
        result = json.loads(response.merged_content)
        self.assertEqual(result['schema'], 'grading-result.v1')
        self.assertEqual(result['reading']['status'], 'COMPLETE')
        self.assertIn('src/main.ts', result['text'])
        self.assertEqual(result['score_details'][0]['evidence_refs'], ['F0001.C0001'])

    def test_partial_reading_and_missing_citations_still_return_a_numeric_score(self):
        service=DeepSeekService()
        service._call_api_json=Mock(side_effect=lambda *args: {'overall_feedback':'依据可读代码',
            'score_details':[{'criterion':'正确性','score':86,'max_score':100,'feedback':'已实现主要功能','evidence_refs':[]}]})
        self.deepseek.grade_homework=service.grade_homework
        material={'schema':'student-materials.v1','reading':{'status':'NEEDS_REVIEW','warnings':['截图识别失败']},
            'files':[{'id':'F0001','path':'main.ts','role':'code','status':'READ','text':'actual source'},
                     {'id':'F0002','path':'report.docx','role':'report','status':'PARTIAL','text':'','warnings':['OCR失败']}]}
        response=self.stub.GradeHomework(pb2.GradeRequest(student_text=json.dumps(material),rubric_json='{"正确性":100}'),timeout=5)
        result=json.loads(response.merged_content)
        self.assertEqual(response.total_score,86)
        self.assertEqual(result['reading']['status'],'PARTIAL')
        self.assertIn('截图识别失败',result['reading']['warnings'])
        self.assertEqual(result['score_details'][0]['evidence_refs'],[])
        self.assertTrue(result['score_details'][0]['evidence_warning'])

    def test_program_computed_total_reaches_both_rpc_response_formats(self):
        service=DeepSeekService()
        model_response={'total_score':90,'overall_feedback':'逐项有依据','score_details':[
            {'criterion':name,'score':value,'max_score':maximum,'feedback':'实际材料依据','evidence_refs':['F0001.C0001']}
            for name,value,maximum in [('功能',40,45),('质量',47,55)]]}
        service._call_api_json=Mock(return_value=model_response)
        self.deepseek.grade_homework=service.grade_homework
        self.aigc._separate_content.return_value=('', 'code')
        material={'schema':'student-materials.v1','reading':{'status':'COMPLETE','warnings':[]},
                  'files':[{'id':'F0001','path':'main.ts','role':'code','status':'READ','text':'actual source'}]}
        for student_text in [json.dumps(material), 'legacy source']:
            with self.subTest(structured=student_text.startswith('{')):
                model_response['total_score']=90
                service._call_api_json.reset_mock()
                response=self.stub.GradeHomework(pb2.GradeRequest(student_text=student_text,
                    question='题目',rubric_json='{"功能":45,"质量":55}'),timeout=5)
                self.assertEqual(response.total_score,87)
                service._call_api_json.assert_called_once()
                if student_text.startswith('{'):
                    result=json.loads(response.merged_content)
                    self.assertEqual(result['reading']['status'],'COMPLETE')
                    self.assertEqual(sum(item['score'] for item in result['score_details']),87)
                else:
                    self.assertEqual(response.merged_content,student_text)

    def test_all_public_methods_round_trip(self):
        text = self.stub.ExtractText(
            pb2.ExtractRequest(filename="scan.png", file_content=b"test image"), timeout=5
        )
        self.assertEqual(text.text_content, "识别结果")
        self.assertFalse(Path(self.ocr_paths[0]).exists())
        plag = self.stub.CheckPlagiarism(
            pb2.PlagiarismRequest(student_texts={"s1": "甲", "s2": "乙"}), timeout=5
        )
        reports = json.loads(plag.plagiarism_results_json)
        self.assertEqual(reports["s1"][0]["similar_to"], "s2")
        self.assertEqual(reports["s2"][0]["similar_to"], "s1")
        aigc = self.stub.DetectAIGC(pb2.AIGCRequest(text_content="正文"), timeout=5)
        self.assertEqual(json.loads(aigc.aigc_report_json), {"ai_probability": 0.25})
        grade = self.stub.GradeHomework(
            pb2.GradeRequest(
                student_id="s1", question="题目", student_text="原文", rubric_json='{"max":100}'
            ),
            timeout=5,
        )
        self.assertEqual(
            (grade.total_score, grade.feedback, grade.merged_content), (85, "良好", "原文")
        )
        self.assertEqual(json.loads(grade.code_doc_match_report_json)["score"], 80)
        self.assertEqual(self.deepseek.grade_homework.call_args.kwargs["rubric"], {"max": 100})
        pool = self.stub.PoolScores(
            pb2.PoolScoresRequest(assignment_id="42", scores_data_json="[]"), timeout=5
        )
        self.assertTrue(pool.success)
        self.assertEqual(json.loads(pool.pooled_results_json), [{"student_id": "s1", "score": 86}])
        with patch(
            "app.rpc.compute.fetch_homework_sync", return_value=["/data/homework.zip"]
        ) as fetch:
            result = self.stub.FetchPortalHomework(
                pb2.FetchRequest(
                    username="test", password="fake", course_name="课程", assignment_name="作业"
                ),
                timeout=5,
            )
        self.assertTrue(result.success)
        self.assertEqual(list(result.file_paths), ["/data/homework.zip"])
        fetch.assert_called_once_with(
            username="test", password="fake", course_name="课程", assignment_name="作业"
        )

    def test_non_code_and_code_only_submissions_skip_code_document_matching(self):
        for prose, code in [("论文或答题正文", ""), ("", "print(1)"), ("汇报内容", "  ")]:
            with self.subTest(prose=prose, code=code):
                self.aigc._separate_content.return_value = (prose, code)
                self.deepseek.analyze_code_doc_match.reset_mock()
                grade = self.stub.GradeHomework(
                    pb2.GradeRequest(student_id="s1", question="按提交内容评价", student_text=prose or code, rubric_json='{"正确性":100}'),
                    timeout=5,
                )
                self.assertEqual(grade.total_score, 85)
                self.deepseek.analyze_code_doc_match.assert_not_called()
                self.assertIsNone(self.deepseek.grade_homework.call_args.kwargs["code_doc_match_report"])
                self.assertEqual(json.loads(grade.code_doc_match_report_json), {})

    def test_legacy_fallback_responses_remain_compatible(self):
        self.plagiarism.check_plagiarism_in_batch.side_effect = ValueError("unavailable")
        self.aigc.detect.side_effect = ValueError("unavailable")
        self.assertEqual(
            self.stub.CheckPlagiarism(pb2.PlagiarismRequest(), timeout=5).plagiarism_results_json,
            "{}",
        )
        self.assertEqual(self.stub.DetectAIGC(pb2.AIGCRequest(), timeout=5).aigc_report_json, "{}")
        invalid = self.stub.GradeHomework(pb2.GradeRequest(rubric_json="invalid"), timeout=5)
        self.assertEqual(invalid.total_score, -1)
        self.assertTrue(invalid.feedback.startswith("评分失败:"))
        self.deepseek.pool_scores.return_value = None
        pool = self.stub.PoolScores(pb2.PoolScoresRequest(), timeout=5)
        self.assertFalse(pool.success)
        self.assertEqual(pool.pooled_results_json, "{}")
        with patch("app.rpc.compute.fetch_homework_sync", return_value="旧入口已停用"):
            result = self.stub.FetchPortalHomework(pb2.FetchRequest(), timeout=5)
        self.assertFalse(result.success)
        self.assertEqual(result.message, "旧入口已停用")

    def test_removed_exam_rpcs_cannot_call_ai_services(self):
        for method, request in [
            (self.stub.IdentifyQuestionNumber, pb2.IdentifyQuestionRequest()),
            (self.stub.GradeExamQuestion, pb2.GradeExamQuestionRequest()),
            (self.stub.SummarizeExam, pb2.SummarizeExamRequest()),
        ]:
            with self.assertRaises(grpc.RpcError) as failure:
                method(request, timeout=5)
            self.assertEqual(failure.exception.code(), grpc.StatusCode.UNIMPLEMENTED)
        self.deepseek.identify_question_number.assert_not_awaited()
        self.deepseek.grade_exam_question.assert_not_awaited()
        self.deepseek.summarize_exam_performance.assert_not_awaited()
