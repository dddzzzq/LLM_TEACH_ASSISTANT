"""Tool contracts remain executable policy when Skill instructions change."""
import json
import unittest

from app.rpa.task_controller import Task
from app.rpa.tools import tool_instructions, validate_action


class BrowserToolContractTests(unittest.TestCase):
    def test_exported_protocol_and_validator_agree(self):
        actions = [
            {'tool': 'click', 'element_ref': 'p0f0e1'},
            {'tool': 'fill', 'element_ref': 'p0f0e2', 'value': '操作系统'},
            {'tool': 'scroll', 'delta_y': 600},
            {'tool': 'switch_page', 'page_id': '1'},
            {'tool': 'open_export_settings'},
            {'tool': 'request_human', 'reason': '同名课程需要选择'},
        ]
        schemas = json.loads(tool_instructions())['oneOf']
        by_name = {s['properties']['tool']['enum'][0]: s for s in schemas}
        for action in actions:
            with self.subTest(action=action):
                self.assertEqual(validate_action(action), action)
                self.assertEqual(set(action), set(by_name[action['tool']]['required']))

    def test_invalid_actions_fail_before_browser_execution(self):
        for action in [None, [], {'tool': []}, {'tool': 'run_python'},
                       {'tool': 'click'}, {'tool': 'click', 'element_ref': 2},
                       {'tool': 'click', 'element_ref': ' '},
                       {'tool': 'scroll', 'delta_y': True},
                       {'tool': 'switch_page', 'page_id': '-1'},
                       {'tool': 'open_export_settings', 'classes': 'all'}]:
            with self.subTest(action=action), self.assertRaises(ValueError):
                validate_action(action)

    def test_worker_contract_has_no_model_or_skill_dependency(self):
        import sys
        self.assertNotIn('app.rpa.agent', sys.modules)
        with self.assertRaises(ValueError):
            validate_action({'tool': 'run_python', 'code': 'anything'})


class BrowserExecutionBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_unknown_action_cannot_fall_through_to_click(self):
        # A valid-looking element reference must not turn an unknown action into a click.
        task = Task(None, {})
        with self.assertRaises(ValueError):
            await task.execute({'tool': 'delete', 'element_ref': 'p0f0e1'})
