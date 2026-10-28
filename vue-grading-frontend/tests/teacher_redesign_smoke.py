"""Real Vue router regression for class isolation, review, imports and floating AI.

All business APIs are intercepted. No real credentials, grading or messages are used.
Run with the existing WSL environment and the Vite server on port 6017.
"""
import asyncio
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright, expect


async def main():
    base = os.environ.get('RPA_UI_BASE_URL', 'http://127.0.0.1:6017')
    diagnostics = Path(__file__).resolve().parents[2] / '.runtime/diagnostics/redesign'
    diagnostics.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + '/src/views/PortalSettingsView.vue', timeout=10) as response:
        module = response.read().decode()
    vue = next(path for path in re.findall(r'from "([^"]+)"', module) if '/vue.js' in path)
    rubric = {'功能实现': {'score': 60, 'description': '功能正确，测试充分'}, '代码质量': {'score': 40, 'description': '结构清晰'}}
    assignments = [dict(id=11, course_name='移动互联网导论', task_name='编程小作业 1', class_name='全部班级', question='实现移动端待办应用，提交源代码与测试报告。', rubric=rubric, portal_homework_id='h1', requirements_status='READY', summary=dict(status='GRADED', graded_count=2, reviewed_count=0, issue_count=0, batch_count=2)), dict(id=12, course_name='软件工程', task_name='本地报告', class_name='全部班级', question='提交设计报告。', rubric=rubric, summary=dict(status='EMPTY', graded_count=0, reviewed_count=0, issue_count=0, batch_count=0))]
    classes = [dict(id='class-a', name='软件工程一班', source='portal', can_grade=True, can_import=True), dict(id='class-b', name='软件工程二班', source='portal', can_grade=True, can_import=True), dict(id='__unassigned__', name='历史未分班', source='unassigned', can_grade=False, can_import=False)]
    results = {key: [dict(id=index, student_id='20260001', student_name=name, score=90, feedback='功能完整，建议补充边界测试。', plagiarism_reports=[], aigc_report={'ai_probability': 0.15, 'detection_source': 'fixture'}, code_doc_match_report={'score': 88, 'reasoning': '报告和代码基本一致'}, is_human_reviewed=False)] for index, (key, name) in enumerate([('class-a', '一班学生'), ('class-b', '二班学生')], 1)}
    captures = dict(uploads=[], directories=[], grades=[], exports=[], saves=[])
    errors, unexpected = [], []
    fail_review, delay_a = True, False
    workflow = '22222222-2222-4222-8222-222222222222'
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=['--no-sandbox'])
        context = await browser.new_context(viewport={'width': 1440, 'height': 1000}, accept_downloads=True)

        async def api(route):
            nonlocal fail_review, delay_a
            request = route.request
            parts = urllib.parse.urlsplit(request.url)
            path, query = parts.path, urllib.parse.parse_qs(parts.query)
            data = []
            if path == '/api/profile':
                data = {'id': 7, 'username': 'teacher-demo', 'name': '陈老师', 'role': 'teacher'}
            elif path == '/api/portal/connection':
                data = {'status': 'READY', 'connection': {'active_snapshot_id': 's1', 'last_synced_at': '2026-10-27T10:00:00+08:00'}, 'import': {'total': 2, 'ready': 2}}
            elif path == '/api/portal/assignments/11':
                data = {'linked': True, 'available': True, 'snapshot_id': 's1', 'requirements_status': 'READY', 'question': assignments[0]['question'], 'homework': {'id': 'h1'}, 'classes': classes[:2]}
            elif path == '/api/portal/homeworks/h1/detail':
                data = {'snapshot_id': 's1', 'operation_id': 'read-fixture', 'detail': {'question': assignments[0]['question'], 'complete': True, 'warnings': [], 'sources': [{'kind': 'text', 'text': assignments[0]['question']}]}}
            elif path == '/api/portal/assignments/11/rubric':
                assignments[0]['rubric'] = request.post_data_json['rubric']
                data = {'message': '已保存'}
            elif path == '/api/portal/assignments/11/grade':
                captures['grades'].append(request.post_data_json)
                data = {'id': workflow}
            elif path == '/api/grading/workflows/' + workflow:
                data = {'workflow': {'id': workflow, 'status': 'RUNNING'}, 'items': [{'id': 'i1', 'assignment_id': 11, 'class_id': 'class-a', 'class_name': '软件工程一班', 'status': 'RUNNING', 'message': '正在批改'}]}
            elif path.endswith('/attention') or path == '/api/rpa/jobs':
                data = []
            elif path == '/api/materials':
                if request.method == 'POST':
                    captures['directories'].append(request.post_data_json)
                    data = {'id': 'batch-fixture'}
                else:
                    class_id = query.get('class_id', [''])[0]
                    data = {'items': [{'id': 'batch-fixture', 'assignment_id': 11, 'class_name': next((item['name'] for item in classes if item['id'] == class_id), '未分班'), 'created_at': '2026-10-27T10:00:00+08:00', 'status': 'READY', 'student_count': 1, 'issue_count': 0, 'output_dir': '/fixture/materials', 'auto_grade': False}], 'grading_jobs': []}
            elif path == '/api/materials/batch-fixture':
                data = {'batch': {'id': 'batch-fixture', 'output_dir': '/fixture/materials'}, 'manifest': {'students': [{'student_id': '20260001', 'student_name': '二班学生', 'warnings': [], 'files': [{'path': 'report.pdf', 'kind': 'report'}]}], 'issues': []}}
            elif path == '/api/sessions':
                data = {'id': 'session-fixture'} if request.method == 'POST' else [{'id': 'session-fixture', 'title': '测试会话'}]
            elif path.startswith('/api/sessions/'):
                data = []
            elif path == '/assignments/':
                if request.method == 'POST':
                    data = dict(id=13, **request.post_data_json)
                    assignments.append(data)
                else:
                    data = assignments
            elif re.fullmatch(r'/assignments/\d+/classes', path):
                data = classes if path.split('/')[2] == '11' else [dict(id='__unassigned__', name='未分班', source='unassigned', can_import=False)]
            elif path.endswith('/submit'):
                captures['uploads'].append(request.post_data_buffer.decode('utf-8', errors='replace'))
                data = {'message': '材料已受理'}
            elif path.endswith('/results'):
                key = query.get('class_id', [''])[0]
                if key == 'class-a' and delay_a:
                    delay_a = False
                    await asyncio.sleep(0.5)
                data = results.get(key, [])
            elif path.endswith('/export'):
                captures['exports'].append(query)
                await route.fulfill(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', body=b'fixture-workbook')
                return
            elif path.startswith('/submissions/'):
                if request.method == 'PUT':
                    captures['saves'].append(request.post_data_json)
                    if fail_review:
                        fail_review = False
                        await route.fulfill(status=500, content_type='application/json', body=json.dumps({'detail': '模拟保存失败'}))
                        return
                    data = next(row for rows in results.values() for row in rows if row['id'] == int(path.rsplit('/', 1)[1]))
                    data.update(request.post_data_json)
                else:
                    data = {}
            elif re.fullmatch(r'/assignments/\d+', path):
                data = next(item for item in assignments if item['id'] == int(path.rsplit('/', 1)[1]))
            else:
                unexpected.append((request.method, path))
            await route.fulfill(content_type='application/json', body=json.dumps(data, ensure_ascii=False))

        harness = f'''<html><head><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/src/assets/main.css"></head><body><div id="app"></div><script type="module">
        import {{createApp}} from '{vue}'; import App from '/src/App.vue'; import router from '/src/router/index.js';
        localStorage.setItem('access_token','fixture');localStorage.setItem('user_info',JSON.stringify({{user_id:7,username:'teacher-demo',role:'teacher'}}));
        window.fixtureRouter=router; await router.replace('/assignments'); createApp(App).use(router).mount('#app');
        </script></body></html>'''
        page = await context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('dialog', lambda dialog: dialog.accept())
        await page.route('**/api/**', api)
        await page.route('**/assignments/**', api)
        await page.route('**/submissions/**', api)
        await page.route(base + '/redesign-fixture', lambda route: route.fulfill(content_type='text/html; charset=utf-8', body=harness))
        await page.goto(base + '/redesign-fixture')
        await expect(page.locator('.assignment-table tbody tr')).to_have_count(2)
        await expect(page.get_by_text('工具与技能', exact=True)).to_have_count(0)
        await page.screenshot(path=str(diagnostics / 'workbench.png'), full_page=True)
        await page.get_by_label('筛选课程').select_option('移动互联网导论')
        await page.get_by_role('link', name='编程小作业 1', exact=True).click()
        await expect(page.get_by_role('heading', name='题目要求', exact=True)).to_be_visible()
        await page.get_by_role('button', name='编辑标准', exact=True).click()
        await page.get_by_label('评分项1分值', exact=True).fill('65')
        await page.get_by_role('button', name='保存评分标准', exact=True).click()
        await expect(page.get_by_text('评分标准已保存，后续批改使用此标准。', exact=True)).to_be_visible()
        await page.get_by_role('button', name='成绩与复核', exact=True).click()
        await expect(page.get_by_role('navigation', name='作业班级').get_by_role('button')).to_have_count(3)
        await expect(page.get_by_text('一班学生', exact=True)).to_be_visible()
        await page.get_by_role('button', name='软件工程二班', exact=False).click()
        await expect(page.get_by_text('二班学生', exact=True)).to_be_visible()
        await expect(page.get_by_text('一班学生', exact=True)).to_have_count(0)
        delay_a = True
        await page.get_by_role('button', name='软件工程一班', exact=False).click()
        await page.get_by_role('button', name='软件工程二班', exact=False).click()
        await page.wait_for_timeout(650)
        await expect(page.get_by_text('二班学生', exact=True)).to_be_visible()
        await expect(page.get_by_text('一班学生', exact=True)).to_have_count(0)
        await page.get_by_role('button', name='查看并复核', exact=True).click()
        await page.get_by_label('最终分数', exact=True).fill('0')
        await page.get_by_role('button', name='保存复核', exact=True).click()
        await expect(page.get_by_text('模拟保存失败', exact=True)).to_be_visible()
        await expect(page.get_by_label('最终分数', exact=True)).to_have_value('0')
        await page.get_by_role('button', name='保存复核', exact=True).click()
        await expect(page.locator('.score')).to_have_text('0.0')
        await expect(page.locator('.review-state.reviewed')).to_have_text('已复核')
        assert captures['saves'][-1]['human_score'] == 0
        await page.get_by_role('button', name='导出本班成绩', exact=True).click()
        assert captures['exports'][-1]['class_id'] == ['class-b']
        await page.screenshot(path=str(diagnostics / 'class-grades.png'), full_page=True)
        await page.get_by_role('button', name='学生作业与批改', exact=True).click()
        await page.get_by_role('button', name='下载并批改本班作业', exact=True).click()
        assert captures['grades'][-1] == {'class_ids': ['class-b']}
        await page.get_by_role('button', name='上传压缩包', exact=True).click()
        await expect(page.get_by_role('checkbox', name='材料整理完成后自动批改')).not_to_be_checked()
        await page.locator('#file-upload').set_input_files({'name': 'fixture.zip', 'mimeType': 'application/zip', 'buffer': b'fixture'})
        await page.get_by_role('button', name='上传并整理材料', exact=True).click()
        await expect(page.get_by_text('材料已受理', exact=True)).to_be_visible()
        assert 'name="auto_grade"\r\n\r\nfalse' in captures['uploads'][-1]
        assert 'name="class_id"\r\n\r\nclass-b' in captures['uploads'][-1]
        await page.get_by_role('button', name='读取目录', exact=True).click()
        await page.get_by_label('作业所在目录', exact=True).fill('/fixture/homework')
        await page.get_by_role('button', name='仅导入并整理', exact=True).click()
        await expect(page.get_by_text('已创建材料整理任务，完成后等待你主动开始批改。', exact=True)).to_be_visible()
        assert captures['directories'][-1]['class_id'] == 'class-b'
        await page.screenshot(path=str(diagnostics / 'class-materials.png'), full_page=True)
        await page.locator('.batch-list').scroll_into_view_if_needed()
        await page.screenshot(path=str(diagnostics / 'class-materials-records.png'), full_page=True)
        await page.get_by_role('button', name='查看学生与文件清单', exact=True).click()
        await expect(page.get_by_role('dialog')).to_be_visible()
        await page.screenshot(path=str(diagnostics / 'material-manifest.png'), full_page=True)
        await page.get_by_role('button', name='关闭', exact=True).click()
        launcher = page.get_by_role('button', name='AI 教学助手', exact=True)
        box = await launcher.bounding_box()
        await page.mouse.move(box['x'] + 26, box['y'] + 26)
        await page.mouse.down()
        await page.mouse.move(box['x'] - 100, box['y'] - 100, steps=10)
        await page.mouse.up()
        await expect(page.locator('#floating-assistant-panel')).not_to_be_visible()
        await launcher.click()
        await expect(page.locator('#floating-assistant-panel')).to_be_visible()
        textarea = page.locator('#floating-assistant-panel').get_by_label('消息内容', exact=True)
        await textarea.fill('保留这条未发送的草稿')
        await page.get_by_role('button', name='收起 AI 助手', exact=True).click()
        await launcher.click()
        await expect(textarea).to_have_value('保留这条未发送的草稿')
        await page.screenshot(path=str(diagnostics / 'floating-assistant.png'), full_page=True)
        await page.get_by_role('button', name='收起 AI 助手', exact=True).click()
        await page.get_by_role('link', name='返回作业列表', exact=True).click()
        await expect(page.get_by_label('筛选课程')).to_have_value('移动互联网导论')
        await page.evaluate("window.fixtureRouter.push('/ai-assistant')")
        await expect(page.locator('#floating-assistant-panel')).to_be_visible()
        await expect(page.locator('.agent-chat-widget')).to_have_count(1)
        await expect(textarea).to_have_value('保留这条未发送的草稿')
        await page.get_by_role('button', name='收起 AI 助手', exact=True).click()
        await page.evaluate("window.fixtureRouter.push('/tools-admin')")
        await expect(page.get_by_role('heading', name='作业工作台', exact=True)).to_be_visible()
        await page.evaluate("window.fixtureRouter.push('/assignments/11?tab=results&class_id=class-b')")
        await expect(page.get_by_text('二班学生', exact=True)).to_be_visible()
        await page.set_viewport_size({'width': 390, 'height': 844})
        await page.screenshot(path=str(diagnostics / 'mobile-grades.png'), full_page=True)
        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        await page.locator('.results-table').scroll_into_view_if_needed()
        await page.screenshot(path=str(diagnostics / 'mobile-grades-table.png'), full_page=True)
        await page.locator('.result-table-scroll').evaluate('(el) => el.scrollLeft = el.scrollWidth')
        await page.screenshot(path=str(diagnostics / 'mobile-grade-actions.png'), full_page=True)
        await page.get_by_role('button', name='查看并复核', exact=True).click()
        await expect(page.get_by_label('最终分数', exact=True)).to_have_value('0')
        await page.screenshot(path=str(diagnostics / 'mobile-review.png'), full_page=True)
        await page.get_by_role('button', name='取消', exact=True).click()
        await launcher.click()
        await expect(page.locator('#floating-assistant-panel')).to_be_visible()
        panel_box = await page.locator('#floating-assistant-panel').bounding_box()
        assert panel_box['x'] >= 0 and panel_box['x'] + panel_box['width'] <= 391
        await page.screenshot(path=str(diagnostics / 'mobile-assistant.png'), full_page=True)
        assert not errors, errors
        assert not unexpected, unexpected
        await browser.close()
    print('PASS: class isolation + slow response, rubric edit, manual zero/retry, class export, three material sources, draggable persistent AI, old routes, filters and mobile.')


if __name__ == '__main__':
    asyncio.run(main())
