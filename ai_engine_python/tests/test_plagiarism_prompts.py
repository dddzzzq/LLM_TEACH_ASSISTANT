import json
import unittest
from unittest.mock import Mock, patch

from app.core.config import settings
from app.schemas.models import LLMPlagiarismAnalysis, PlagiarismReport
from app.services.deepseek_service import DeepSeekService


class PlagiarismPromptContractTests(unittest.TestCase):
    def test_both_prompt_types_preserve_payload_and_response_contract(self):
        first = '我的推导："边界"\nprint("自己的实现")'
        second = '结果讨论\\测试：不同的取舍与错误处理'
        answer = {'similarity_score': 35, 'reasoning': '存在局部复用，但有独立分析。', 'suspicious_parts': []}
        usage = {'prompt_tokens': 100, 'completion_tokens': 30, 'total_tokens': 130}
        service = DeepSeekService()
        for kind, label in [('text', '报告'), ('code', '代码')]:
            with self.subTest(kind=kind):
                service._call_api_with_usage = Mock(return_value={
                    'choices': [{'message': {'content': json.dumps(answer, ensure_ascii=False)}}],
                    'usage': usage,
                })
                result, actual_usage = service.analyze_plagiarism(first, second, kind)
                self.assertEqual(LLMPlagiarismAnalysis.model_validate(result).similarity_score, 35)
                self.assertEqual(actual_usage, usage)
                args, kwargs = service._call_api_with_usage.call_args
                prompt = args[0]
                self.assertIn(f'[{label} A]:', prompt)
                self.assertIn(f'[{label} B]:', prompt)
                self.assertIn(json.dumps(first, ensure_ascii=False), prompt)
                self.assertIn(json.dumps(second, ensure_ascii=False), prompt)
                self.assertEqual(kwargs['response_format'], {'type': 'json_object'})

    def test_input_limits_still_apply_to_each_student(self):
        service = DeepSeekService()
        with patch.object(settings, 'MAX_PLAGIARISM_TEXT_LENGTH', 8):
            for build in [service._get_text_plagiarism_prompt, service._get_code_plagiarism_prompt]:
                prompt = build('甲' * 8 + 'FIRST_SECRET_SUFFIX', '乙' * 8 + 'SECOND_SECRET_SUFFIX')
                self.assertIn(json.dumps('甲' * 8, ensure_ascii=False), prompt)
                self.assertIn(json.dumps('乙' * 8, ensure_ascii=False), prompt)
                self.assertNotIn('SECRET_SUFFIX', prompt)

    def test_extreme_risk_is_review_context_without_automatic_deduction(self):
        service = DeepSeekService()
        service._call_api_json = Mock(return_value={'total_score': 88, 'overall_feedback': '按作业质量评价', 'score_details': []})
        report = PlagiarismReport(similar_to='fixture', content_type='code', initial_score=.98,
                                  llm_analysis=LLMPlagiarismAnalysis(similarity_score=96, reasoning='大范围重复且独立贡献很少'))
        result = service.grade_homework('测试题目', {}, '测试答案', plagiarism_reports=[report])
        self.assertEqual(result['total_score'], 88)
        prompt = service._call_api_json.call_args.args[0]
        self.assertIn('不能仅凭此分数直接认定抄袭或自动扣分', prompt)
        self.assertIn('已考虑合理复用和独立贡献', prompt)


if __name__ == '__main__':
    unittest.main()
