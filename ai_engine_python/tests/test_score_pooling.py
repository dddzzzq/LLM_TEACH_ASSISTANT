import json
import unittest
from unittest.mock import Mock

from app.services.deepseek_service import DeepSeekService


class ScorePoolingContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_object_response_becomes_rpc_array(self):
        service = DeepSeekService()
        rows = [{"student_id": "s1", "original_score": 85, "pooled_score": 100, "pool_reason": "最高分锚定"}]
        service._call_api_json = Mock(return_value={"pooled_results": rows})
        result = await service.pool_scores("1", json.dumps([{"student_id": "s1", "original_score": 85}]))
        self.assertEqual(result, rows)
        self.assertIn('"pooled_results"', service._call_api_json.call_args.args[0])

    async def test_invalid_response_cannot_be_reported_as_success(self):
        service = DeepSeekService()
        base = {"student_id": "s1", "original_score": 85, "pooled_score": 100, "pool_reason": "最高分锚定"}
        invalid = [
            {},
            {"pooled_results": []},
            {"pooled_results": [dict(base, student_id="someone_else")]},
            {"pooled_results": [base, base]},
            {"pooled_results": [dict(base, pooled_score=float("nan"))]},
            {"pooled_results": [dict(base, pooled_score=101)]},
        ]
        for response in invalid:
            with self.subTest(response=response):
                service._call_api_json = Mock(return_value=response)
                result = await service.pool_scores("1", json.dumps([{"student_id": "s1", "original_score": 85}]))
                self.assertIsNone(result)
