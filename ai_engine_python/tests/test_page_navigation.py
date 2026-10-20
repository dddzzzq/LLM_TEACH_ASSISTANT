import json
import unittest

from playwright.async_api import async_playwright

from app.rpa.portal_adapter import PortalAdapter
from app.rpa.page_snapshot import public_url


class PublicURLTests(unittest.TestCase):
    def test_url_inventory_omits_credentials_and_tolerates_malformed_links(self):
        self.assertEqual(public_url('https://user:secret@example.com/work?token=secret#secret'), 'https://example.com/work')
        self.assertEqual(public_url('https://example.com:broken/work'), '')
        self.assertEqual(public_url('javascript:void(0)'), '')


class SemanticPageNavigationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(headless=True, args=['--no-sandbox'])
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        self.target = {'course_name':'操作系统','assignment_name':'第一次作业'}
        self.adapter = PortalAdapter(self.context, self.target)

    async def asyncTearDown(self):
        await self.browser.close()
        await self.pw.stop()

    async def test_inventory_covers_frames_roles_headings_without_credentials(self):
        async def route(request):
            body = '''<main><h2>作业区域</h2><div role=button aria-label="查看提交">查看</div>
                <a href="/work?signature=link-private-secret">作业详情</a></main>'''
            if '/portal' in request.request.url:
                body = '''<header><h1>教学平台</h1><nav aria-label="课程导航"><div role=tab>我的课程</div></nav></header>
                    <input type=search aria-label="搜索课程" value="typed-private-secret">
                    <form><input type=password value="password-private-secret"><button>登录</button></form>
                    <input type=hidden value="hidden-private-secret">
                    <iframe name=course-area src="https://mooc1.chaoxing.com/work?token=frame-private-secret"></iframe>'''
            await request.fulfill(content_type='text/html; charset=utf-8', body=body)
        await self.page.route('**/*', route)
        await self.page.goto('https://learning.xidian.edu.cn/portal?token=page-private-secret')
        await self.page.frames[1].wait_for_selector('[role=button]')
        view = await self.adapter.observe()
        serialized = json.dumps(view, ensure_ascii=False)
        self.assertNotIn('private-secret', serialized)
        self.assertEqual(len(view['frames']), 2)
        self.assertIn('作业区域', serialized)
        self.assertIn('课程导航', serialized)
        self.assertTrue(any(e['role']=='tab' for e in view['elements']))
        self.assertTrue(any(e['name']=='查看提交' and e['frame_id']=='p0f1' for e in view['elements']))

    async def test_redesigned_cards_and_review_controls_reach_export_without_legacy_selectors(self):
        await self.page.set_content('''<main><section role=list><article role=listitem>
            <h3>操作系统</h3><span>2026秋季</span><button aria-label="进入课程">进入</button>
            </article></section></main>''')
        view = await self.adapter.observe()
        entry = next(e for e in view['elements'] if e['name']=='进入课程')
        page, loc, item = self.adapter.refs[entry['ref']]
        await self.adapter.record_navigation(loc, item, page)
        await loc.click()
        self.assertTrue(self.adapter.course_confirmed)
        await self.page.set_content('''<header><div role=heading aria-level=1>操作系统</div></header>
            <section role=list><article role=listitem><h3>第一次作业</h3><div role=button>查看提交</div></article>
            <article role=listitem><h3>第二次作业</h3><div role=button>查看提交</div></article></section>''')
        view = await self.adapter.observe()
        entry = next(e for e in view['elements'] if e['name']=='查看提交' and '第一次作业' in e['context_labels'])
        page, loc, item = self.adapter.refs[entry['ref']]
        await self.adapter.record_navigation(loc, item, page)
        await loc.click()
        self.assertTrue(self.adapter.assignment_confirmed)
        await self.page.set_content('''<main><h1>第一次作业</h1>
            <button aria-label="导出附件" onclick="document.querySelector('[role=dialog]').hidden=false">导出附件</button>
            <section role=dialog hidden>导出设置<label><input type=checkbox>班级A</label></section></main>''')
        view = await self.adapter.observe()
        self.assertTrue(view['can_open_export'])
        await self.adapter.open_export()
        self.assertTrue(await self.page.locator('[role=dialog]').is_visible())

    async def test_duplicate_courses_require_unique_term_evidence(self):
        for term, semesters, accepted in (
            ('', ['2025秋季','2026秋季'], False),
            ('2026秋季', ['2025秋季','2026秋季'], True),
            ('2026秋季', ['2026秋季','2026秋季'], False),
        ):
            with self.subTest(term=term, semesters=semesters):
                self.adapter = PortalAdapter(self.context, {**self.target,'term':term})
                await self.page.set_content('<section role=list>' + ''.join(
                    f'<article role=listitem><h3>操作系统</h3><span>{semester}</span><button>进入</button></article>'
                    for semester in semesters) + '</section>')
                view = await self.adapter.observe()
                entry = next(e for e in reversed(view['elements']) if e['name']=='进入')
                page, loc, item = self.adapter.refs[entry['ref']]
                if accepted:
                    await self.adapter.record_navigation(loc, item, page)
                    self.assertTrue(self.adapter.course_confirmed)
                else:
                    with self.assertRaises(ValueError): await self.adapter.record_navigation(loc, item, page)

    async def test_nested_title_and_review_controls_count_as_one_assignment_record(self):
        for container in ('li', 'div class=work-row'):
            with self.subTest(container=container):
                tag = container.split()[0]
                await self.page.set_content(f'''<h1>操作系统</h1><section>
                  <{container}><div class=title><h3><a href=#>第一次作业</a></h3></div>
                  <div class=actions><button>查看</button><button>批阅</button></div></{tag}>
                  </section>''')
                self.adapter = PortalAdapter(self.context, self.target)
                self.adapter.course_confirmed = True
                view = await self.adapter.observe()
                self.assertEqual(view['target_candidates']['assignment']['group_count'], 1)
                matching = [e for e in view['elements'] if e['name'] in ('第一次作业','查看','批阅')]
                self.assertEqual(len({e['record_id'] for e in matching}), 1)
                entry = next(e for e in matching if e['name']=='批阅')
                page, loc, item = self.adapter.refs[entry['ref']]
                await self.adapter.record_navigation(loc, item, page)
                self.assertTrue(self.adapter.assignment_confirmed)

    async def test_actual_duplicate_nested_assignment_records_still_require_human(self):
        for container in ('li', 'div class=work-row'):
            with self.subTest(container=container):
                tag = container.split()[0]
                row = f'''<{container}><div><h3><a href=#>第一次作业</a></h3></div>
                  <div><button>查看</button><button>批阅</button></div></{tag}>'''
                await self.page.set_content('<article>' + row + row + '</article>')
                self.adapter = PortalAdapter(self.context, self.target)
                self.adapter.course_confirmed = True
                view = await self.adapter.observe()
                self.assertEqual(view['target_candidates']['assignment']['group_count'], 2)
                entry = next(e for e in view['elements'] if e['name']=='批阅')
                page, loc, item = self.adapter.refs[entry['ref']]
                with self.assertRaisesRegex(ValueError, '多个同名作业'):
                    await self.adapter.record_navigation(loc, item, page)
                self.assertFalse(self.adapter.assignment_confirmed)

    async def test_plain_card_titles_keep_review_buttons_in_their_own_record(self):
        await self.page.set_content('''<section><div class=work-row><div class=title><a>第一次作业</a></div>
          <div class=actions><button>查看</button><button>批阅</button></div></div>
          <div class=work-row><div class=title><a>第二次作业</a></div>
          <div class=actions><button>查看</button><button>批阅</button></div></div></section>''')
        self.adapter.course_confirmed = True
        view = await self.adapter.observe()
        self.assertEqual(view['target_candidates']['assignment']['group_count'], 1)
        buttons = [e for e in view['elements'] if e['name']=='批阅']
        self.assertIn('第一次作业', buttons[0]['context_labels'])
        self.assertNotIn('第一次作业', buttons[1]['context_labels'])
        page, loc, item = self.adapter.refs[buttons[0]['ref']]
        await self.adapter.record_navigation(loc, item, page)
        self.assertTrue(self.adapter.assignment_confirmed)

    async def test_collection_article_does_not_merge_differently_shaped_records(self):
        await self.page.set_content('''<article>
          <div class=work-row><div><h3><a>第一次作业</a></h3></div><button>批阅</button></div>
          <section class=another-row><header><h3><a>第一次作业</a></h3></header>
          <div><button>批阅</button></div></section></article>''')
        self.adapter.course_confirmed = True
        view = await self.adapter.observe()
        evidence = view['target_candidates']['assignment']
        self.assertEqual(evidence['group_count'], 2)
        self.assertEqual(evidence['count_basis'], 'record')
        self.assertEqual(len(evidence['record_ids']), 2)
        entry = next(e for e in view['elements'] if e['name']=='批阅')
        page, loc, item = self.adapter.refs[entry['ref']]
        with self.assertRaises(ValueError) as failure:
            await self.adapter.record_navigation(loc, item, page)
        self.assertEqual(failure.exception.evidence['group_count'], 2)

    async def test_nested_duplicate_courses_keep_term_disambiguation_on_record_identity(self):
        self.adapter = PortalAdapter(self.context, {**self.target, 'term':'2026秋季'})
        await self.page.set_content('''<section><article><div><h3><a>操作系统</a></h3></div>
          <span>2025秋季</span><div><button>进入课程</button></div></article>
          <article><div><h3><a>操作系统</a></h3></div><span>2026秋季</span>
          <div><button>进入课程</button></div></article></section>''')
        view = await self.adapter.observe()
        self.assertEqual(view['target_candidates']['course']['group_count'], 2)
        entry = next(e for e in reversed(view['elements']) if e['name']=='进入课程')
        page, loc, item = self.adapter.refs[entry['ref']]
        await self.adapter.record_navigation(loc, item, page)
        self.assertTrue(self.adapter.course_confirmed)

    async def test_large_inventory_keeps_target_and_reports_truncation(self):
        await self.page.set_content(''.join(f'<button>其他入口{i}</button>' for i in range(450)) +
            '<article role=listitem><h3>操作系统</h3><button>进入课程</button></article>')
        view = await self.adapter.observe()
        self.assertTrue(view['coverage']['truncated'])
        self.assertEqual(view['coverage']['total_elements'], 451)
        self.assertTrue(any(e['name']=='进入课程' for e in view['elements']))
        self.assertEqual(set(self.adapter.refs), {e['ref'] for e in view['elements']})
        self.assertLess(len(json.dumps(view, ensure_ascii=False)), 65000)

    async def test_shadow_component_inventory_and_reference_point_to_same_control(self):
        await self.page.set_content('''<portal-card></portal-card><button>其他入口</button>
            <script>document.querySelector('portal-card').attachShadow({mode:'open'}).innerHTML=
            '<article><h3>操作系统</h3><button aria-label="进入课程" onclick="window.targetClicked=true">进入</button></article>';</script>''')
        view = await self.adapter.observe()
        self.assertTrue(any(h['text']=='操作系统' for f in view['frames'] for h in f['headings']))
        entry = next(e for e in view['elements'] if e['name']=='进入课程')
        page, loc, item = self.adapter.refs[entry['ref']]
        await self.adapter.record_navigation(loc, item, page)
        await loc.click()
        self.assertTrue(await self.page.evaluate('window.targetClicked'))

    async def test_list_title_does_not_confirm_unrelated_export_and_duplicate_menus_are_rejected(self):
        await self.page.set_content('''<h1>操作系统</h1><ul><li><h2>第一次作业</h2></li></ul>
            <button>导出附件</button>''')
        await self.adapter.reconcile()
        self.assertTrue(self.adapter.course_confirmed)
        self.assertFalse(self.adapter.assignment_confirmed)
        self.assertFalse((await self.adapter.observe())['can_open_export'])
        self.adapter.assignment_confirmed = True
        await self.page.set_content('<h1>第一次作业</h1><button>导出附件</button><button>导出附件</button>')
        self.assertFalse((await self.adapter.observe())['can_open_export'])
        with self.assertRaises(ValueError): await self.adapter.open_export()

    async def test_hidden_review_section_does_not_authorize_export_from_assignment_list(self):
        await self.page.set_content('''<h1>操作系统</h1><section id=works>
          <div class=work-row><div><h3><a>第一次作业</a></h3></div>
          <div><button onclick="document.querySelector('#works').hidden=true;document.querySelector('#review').hidden=false">批阅</button></div></div>
          <div class=work-row><h3>第二次作业</h3><button>批阅</button></div></section>
          <section id=review hidden><h2>第一次作业</h2><ul class=morePop style="display:none">
          <a onclick="window.exportOpened=true">导出作业附件</a></ul></section>''')
        await self.adapter.reconcile()
        self.assertTrue(self.adapter.course_confirmed)
        self.assertFalse(self.adapter.assignment_confirmed)
        view = await self.adapter.observe()
        self.assertFalse(view['can_open_export'])
        entry = next(e for e in view['elements'] if e['name']=='批阅' and '第一次作业' in e['context_labels'])
        page, loc, item = self.adapter.refs[entry['ref']]
        await self.adapter.record_navigation(loc, item, page)
        await loc.click()
        self.assertTrue((await self.adapter.observe())['can_open_export'])
        await self.adapter.open_export()
        self.assertTrue(await self.page.evaluate('window.exportOpened'))
