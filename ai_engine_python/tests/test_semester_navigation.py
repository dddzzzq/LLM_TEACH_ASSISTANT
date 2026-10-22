"""Semester controls are manual; observation focus preserves original navigation."""
import unittest

from app.rpa.tools import tool_instructions
from tests import test_target_confirmation as fixtures

CURRENT = '2026-2027第一学期'
SECOND = '2025-2026第二学期'

class SemesterNavigationTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.TargetConfirmationTests.asyncSetUp
    asyncTearDown = fixtures.TargetConfirmationTests.asyncTearDown
    bare_task = fixtures.TargetConfirmationTests.bare_task
    task_page = fixtures.TargetConfirmationTests.task_page
    action = fixtures.TargetConfirmationTests.action
    click = fixtures.TargetConfirmationTests.click

    async def test_native_semester_is_manual_and_teacher_can_resume_from_current_page(self):
        task=await self.task_page(f'''<select aria-label=学期 onchange="document.querySelector('h3').innerText='操作系统'">
            <option>{CURRENT}</option><option>{SECOND}</option></select>
            <article role=listitem><h3>其他课程</h3><button onclick="window.entered=true">进入课程</button></article>''')
        task.state['target']['term']=CURRENT
        await self.action(task,self.click('学期'))
        self.assertEqual(task.state['control_mode'],'HUMAN')
        self.assertEqual(task.state['waiting_reason'],'COURSE_OR_TERM')
        self.assertIn('课程名',task.state['message'])
        self.assertIn('学期',task.state['message'])
        self.assertEqual(await task.page.locator('select').input_value(),CURRENT)
        await task.page.locator('select').select_option(label=SECOND)  # Teacher's manual input.
        await task.resume()
        self.assertEqual(task.state['target']['term'],SECOND)
        self.assertEqual(task.state['requested_target']['term'],CURRENT)
        await self.action(task,self.click('进入课程'))
        self.assertTrue(await task.page.evaluate('window.entered'))
        self.assertEqual(task.state['control_mode'],'AUTO')
        self.assertNotIn('lookup_progress',(await task.observe())['observation'])

    async def test_custom_semester_trigger_and_open_options_cannot_bypass_manual_control(self):
        task=await self.task_page(f'''<button id=current onclick="document.querySelector('ul').hidden=false">{CURRENT}</button>
            <ul hidden><li>{CURRENT}</li><li onclick="document.querySelector('#current').innerText=this.innerText">{SECOND}</li></ul>''')
        await self.action(task,self.click(CURRENT))
        self.assertFalse(await task.page.locator('ul').is_visible())
        self.assertEqual(task.state['control_mode'],'HUMAN')
        await task.page.locator('#current').click()  # Teacher opens menu.
        await task.resume()
        await self.action(task,self.click(SECOND))
        self.assertEqual(task.state['control_mode'],'HUMAN')
        self.assertEqual(await task.page.locator('#current').inner_text(),CURRENT)

    async def test_removed_tool_and_old_lookup_cannot_restart_semester_automation(self):
        self.assertNotIn('select_semester',tool_instructions())
        task=await self.task_page('<article role=listitem><h3>操作系统</h3><button onclick="window.entered=true">进入课程</button></article>')
        task.state['semester_lookup']={'requested_term':SECOND,'switch_status':'unverified'}
        with self.assertRaises(ValueError):
            await task.execute(dict(tool='select_semester',element_ref='old',term=SECOND,purpose='fallback'))
        await self.action(task,self.click('进入课程'))
        self.assertTrue(await task.page.evaluate('window.entered'))
        self.assertEqual(task.state['control_mode'],'AUTO')

    async def test_observation_focus_ignores_old_tabs_and_preserves_open_options_under_budget(self):
        task = await self.task_page('<a>个人空间</a><article><h3>操作系统</h3><button>进入课程</button></article>')
        for _ in range(5):
            page=await task.context.new_page()
            await page.set_content('<a>个人空间</a><article><h3>操作系统</h3><button>进入课程</button></article>')
        task.page=await task.context.new_page()
        clutter=''.join(f'<button>普通项目{i}</button>' for i in range(500))
        await task.page.set_content(f'''<h2>当前课程页</h2>{clutter}
            <input aria-label=学期 readonly value="{CURRENT}" style="position:fixed;top:0">
            <div style="position:fixed;top:40px"><div>{CURRENT}</div><div>{SECOND}</div></div>''')
        view=(await task.observe())['observation']
        self.assertEqual(view['active_page_id'],'6')
        self.assertEqual(len(view['pages']),7)
        self.assertTrue(all(e['page_id']=='6' for e in view['elements']))
        self.assertFalse(any(e['name']=='个人空间' for e in view['elements']))
        self.assertEqual(view['target_candidates']['course']['group_count'],0)
        self.assertTrue(view['coverage']['truncated'])
        self.assertEqual([e['name'] for e in view['elements'][:2]],[CURRENT,SECOND])
        await self.action(task,lambda _:dict(tool='switch_page',page_id='0'))
        previous=(await task.observe())['observation']
        self.assertEqual(previous['active_page_id'],'0')
        self.assertTrue(any(e['name']=='个人空间' for e in previous['elements']))

    async def test_delayed_personal_space_popup_becomes_active_without_reentering_portal(self):
        task=await self.task_page('''<a href=# onclick="setTimeout(()=>window.open('/space','_blank'),650);return false">个人空间</a>''')
        async def space(route):
            await route.fulfill(content_type='text/html; charset=utf-8',body='<h2>我的课程</h2><button>选择课程</button>')
        await task.context.route('**/space',space)
        await self.action(task,self.click('个人空间'))
        self.assertEqual(len(task.context.pages),2)
        self.assertEqual(task.page,task.context.pages[1])
        obs=(await task.observe())['observation']
        self.assertEqual(obs['active_page_id'],'1')
        self.assertTrue(any(e['name']=='选择课程' for e in obs['elements']))
        self.assertFalse(any(e['name']=='个人空间' for e in obs['elements']))
