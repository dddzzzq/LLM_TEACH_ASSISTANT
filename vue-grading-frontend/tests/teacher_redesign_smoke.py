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
    classes = [dict(id='class-a', name='软件工程一班', source='portal', can_grade=True, can_import=True, material_status='READY', grading_status='GRADED', graded_count=1, reviewed_count=0, issue_count=0), dict(id='class-b', name='软件工程二班', source='portal', can_grade=True, can_import=True, material_status='READY', grading_status='GRADED', graded_count=3, reviewed_count=0, issue_count=0), dict(id='__unassigned__', name='历史未分班', source='unassigned', can_grade=False, can_import=False, material_status='EMPTY', grading_status='NOT_STARTED', graded_count=0, reviewed_count=0, issue_count=0)]
    results = {key: [dict(id=index, student_id='20260001', student_name=name, score=90, feedback='功能完整，建议补充边界测试。', plagiarism_reports=[], aigc_report={'ai_probability': 0.15, 'detection_source': 'fixture'}, code_doc_match_report={'score': 88, 'reasoning': '报告和代码基本一致'}, is_human_reviewed=False)] for index, (key, name) in enumerate([('class-a', '一班学生'), ('class-b', '二班学生')], 1)}
    for index, name in [(3, '二班学生二'), (4, '二班学生三')]:
        results['class-b'].append(dict(id=index, student_id=f'2026000{index}', student_name=name, score=80 + index, feedback='待教师复核。', plagiarism_reports=[], aigc_report=None, code_doc_match_report=None, is_human_reviewed=False))
    results['class-b'][0].update(material_batch_id='batch-fixture', reading_status='PARTIAL',
        reading_report={'status':'PARTIAL','warnings':['截图2识别失败，其余材料正常给分'], 'submission_archives':[{'id':'zip-fixture','name':'20260001_二班学生.zip','format':'zip','student_id':'20260001'}], 'files':[{'id':'F0001','path':'students/20260001/src/main.ts','status':'READ'}]},
        score_details=[{'criterion':'功能实现','score':54,'max_score':60,'feedback':'源码与截图均已核对',
                        'evidence_warning':'部分引用未核验，保留基于可读内容的评分',
                        'evidence':[{'id':'F0001.C0001','path':'students/20260001/src/main.ts','line_start':1,'line_end':8}]}])
    captures = dict(uploads=[], directories=[], grades=[], exports=[], saves=[], rubrics=[], deletes=[], regrades=[])
    material_state={'enabled':False,'id':'old-completed-job','status':'SUCCESS'}
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
                captures['rubrics'].append(request.post_data_json['rubric'])
                assignments[0]['rubric'] = request.post_data_json['rubric']
                data = {'message': '已保存'}
            elif path == '/api/portal/assignments/11/grade':
                captures['grades'].append(request.post_data_json)
                data = {'id': workflow}
            elif path == '/api/grading/workflows/' + workflow:
                chosen = classes[:2] if captures['grades'][-1].get('all_classes') else [item for item in classes if item['id'] in captures['grades'][-1]['class_ids']]
                data = {'workflow': {'id': workflow, 'status': 'SUCCEEDED'}, 'items': [{'id': item['id'], 'assignment_id': 11, 'class_id': item['id'], 'class_name': item['name'], 'status': 'SUCCEEDED', 'message': '已完成'} for item in chosen]}
            elif path.endswith('/attention') or path == '/api/rpa/jobs':
                data = []
            elif path == '/api/materials':
                if request.method == 'POST':
                    captures['directories'].append(request.post_data_json)
                    data = {'id': 'batch-fixture'}
                else:
                    class_id = query.get('class_id', [''])[0]
                    data = {'items': [{'id': 'batch-fixture', 'assignment_id': 11, 'class_name': next((item['name'] for item in classes if item['id'] == class_id), '未分班'), 'created_at': '2026-10-27T10:00:00+08:00', 'status': 'READY', 'student_count': 1, 'issue_count': 0, 'output_dir': '/fixture/materials', 'auto_grade': False,
                                       'grading_job_id':material_state['id'] if material_state['enabled'] else ''}],
                            'grading_jobs': [{'id':material_state['id'],'status':material_state['status']}] if material_state['enabled'] else []}
            elif path == '/api/materials/batch-fixture/grade':
                captures['regrades'].append((request.post_data_json,request.headers.get('idempotency-key')))
                material_state.update(id='new-fixture-job',status='PENDING')
                data={'message':'重新批改已受理'}
            elif path == '/api/materials/batch-fixture':
                data = {'batch': {'id': 'batch-fixture', 'output_dir': '/fixture/materials'}, 'manifest': {'students': [{'student_id': '20260001', 'student_name': '二班学生', 'warnings': [], 'files': [{'path': 'report.pdf', 'kind': 'report'}]}], 'issues': []}}
            elif path == '/api/materials/batch-fixture/file':
                assert query['path'] == ['students/20260001/src/main.ts']
                await route.fulfill(content_type='text/plain; charset=utf-8', body='Promise.all(tasks); // source evidence')
                return
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
                if request.method=='DELETE':
                    captures['deletes'].append(query)
                    assert key=='class-b','class delete lost its scope'
                    results[key]=[]
                    next(item for item in classes if item['id']==key).update(graded_count=0,reviewed_count=0)
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
        import {{rubricTemplates, createRubricFromTemplate}} from '/src/services/rubricTemplates.js';
        window.fixtureRubrics=Object.fromEntries(rubricTemplates.map(item=>[item.id,JSON.parse(createRubricFromTemplate(item.id))]));
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
        sidebar = page.get_by_role('complementary', name='主导航')
        await expect(sidebar.get_by_role('navigation', name='工作区').get_by_role('link')).to_have_count(1)
        for category in ['需要处理', '处理中', '已有成绩']:
            await expect(sidebar.get_by_text(category, exact=True)).to_have_count(0)
        await sidebar.get_by_role('button', name='任务控制台', exact=True).click()
        await expect(page.get_by_role('dialog', name='作业拉取与材料')).to_be_visible()
        await page.get_by_role('button', name='收起作业控制台', exact=True).click()
        await expect(sidebar.get_by_role('button', name='任务控制台', exact=True)).to_be_focused()
        await expect(page.get_by_text('工具与技能', exact=True)).to_have_count(0)
        await page.screenshot(path=str(diagnostics / 'workbench.png'), full_page=True)
        await page.get_by_label('筛选课程').select_option('移动互联网导论')
        await page.get_by_role('link', name='编程小作业 1', exact=True).click()
        await expect(page.get_by_role('heading', name='题目要求', exact=True)).to_be_visible()
        await page.get_by_role('button', name='编辑标准', exact=True).click()
        picker = page.get_by_label('评分模板', exact=True)
        await expect(picker.locator('option:not([disabled])')).to_have_text(['答题解题类', '工程代码类', '论文研究类', '展示汇报类'])
        await expect(page.get_by_label('评分项1名称', exact=True)).to_have_value('功能实现')
        # Opening keeps the saved rubric. Selecting fills all criteria at once.
        assert not captures['rubrics'] and assignments[0]['rubric'] == rubric
        expected_rubrics = await page.evaluate('window.fixtureRubrics')
        for template, first_name, count in [('problem_solving', '答案正确性', 4), ('research_paper', '研究问题与任务契合', 5), ('presentation', '内容准确与任务完成', 4), ('engineering_code', '功能与需求完成度', 5)]:
            await picker.select_option(template)
            await expect(page.get_by_label('评分项1名称', exact=True)).to_have_value(first_name)
            await expect(page.locator('.criterion')).to_have_count(count)
            await expect(page.get_by_text('当前合计 100 分。', exact=True)).to_be_visible()
            for index, (name, criterion) in enumerate(expected_rubrics[template].items(), 1):
                await expect(page.get_by_label(f'评分项{index}名称', exact=True)).to_have_value(name)
                await expect(page.get_by_label(f'评分项{index}分值', exact=True)).to_have_value(str(criterion['score']))
                await expect(page.get_by_label(f'评分项{index}要求', exact=True)).to_have_value(criterion['description'])
            assert not captures['rubrics'] and assignments[0]['rubric'] == rubric
        await page.get_by_label('评分项1要求', exact=True).fill('未保存的临时修改')
        await page.get_by_role('button', name='恢复模板初始标准', exact=True).click()
        await expect(page.get_by_label('评分项1要求', exact=True)).to_have_value(expected_rubrics['engineering_code']['功能与需求完成度']['description'])
        await page.screenshot(path=str(diagnostics / 'rubric-templates.png'), full_page=True)
        # Advanced edits stay in JSON mode until the teacher requests the table.
        await page.get_by_role('button', name='编辑 JSON（高级）', exact=True).click()
        json_editor = page.get_by_label('评分标准（JSON）', exact=True)
        draft = json.loads(await json_editor.input_value())
        draft['功能与需求完成度']['description'] = '教师补充：核对主要功能与测试结果'
        await json_editor.fill(json.dumps(draft, ensure_ascii=False))
        await expect(json_editor).to_be_visible()
        await page.get_by_role('button', name='返回表格编辑', exact=True).click()
        await expect(page.get_by_label('评分项1要求', exact=True)).to_have_value(draft['功能与需求完成度']['description'])
        await page.get_by_label('评分项1分值', exact=True).fill('50')
        await page.get_by_role('button', name='保存评分标准', exact=True).click()
        await expect(page.get_by_text('评分标准已保存，后续批改使用此标准。', exact=True)).to_be_visible()
        assert captures['rubrics'][-1]['功能与需求完成度']['score'] == 50
        assert sum(item['score'] for item in captures['rubrics'][-1].values()) == 105
        await page.get_by_role('button', name='编辑标准', exact=True).click()
        await expect(page.get_by_label('评分项1分值', exact=True)).to_have_value('50')
        await expect(page.get_by_label('评分项1要求', exact=True)).to_have_value(draft['功能与需求完成度']['description'])
        await page.get_by_role('button', name='编辑 JSON（高级）', exact=True).click()
        await picker.select_option('research_paper')
        await expect(page.get_by_label('评分项1名称', exact=True)).to_have_value('研究问题与任务契合')
        await page.get_by_role('button', name='取消', exact=True).click()
        await page.get_by_role('button', name='编辑标准', exact=True).click()
        await expect(page.get_by_label('评分项1分值', exact=True)).to_have_value('50')
        await expect(page.get_by_label('评分项1要求', exact=True)).to_have_value(draft['功能与需求完成度']['description'])
        assert len(captures['rubrics']) == 1
        await page.get_by_role('button', name='取消', exact=True).click()
        # The new-assignment form uses the same templates and saves their full
        # editable criteria through the original assignment API.
        assignment_location = await page.evaluate('window.fixtureRouter.currentRoute.value.fullPath')
        await page.evaluate("window.fixtureRouter.push('/assignments/new')")
        await expect(page.get_by_role('heading', name='新建本地作业', exact=True)).to_be_visible()
        await expect(picker).to_have_value('problem_solving')
        await expect(page.get_by_label('评分项1名称', exact=True)).to_have_value('答案正确性')
        create_form = page.locator('.create-form')
        await create_form.get_by_label('课程名称', exact=True).fill('模板验证课程')
        await create_form.get_by_label('作业名称', exact=True).fill('展示汇报作业')
        await create_form.get_by_label('题目要求', exact=True).fill('提交项目汇报 PPT，说明背景、成果与结论。')
        await picker.select_option('presentation')
        await page.screenshot(path=str(diagnostics / 'new-assignment-template.png'), full_page=True)
        await page.get_by_role('button', name='仅创建作业', exact=True).click()
        await expect(page.get_by_role('heading', name='展示汇报作业', exact=True)).to_be_visible()
        assert len(assignments) == 3 and sum(item['score'] for item in assignments[-1]['rubric'].values()) == 100
        assert '内容准确与任务完成' in assignments[-1]['rubric'] and '视觉呈现与规范' in assignments[-1]['rubric']
        await page.evaluate('(path) => window.fixtureRouter.push(path)', assignment_location)
        await expect(page.get_by_role('heading', name='编程小作业 1', exact=True)).to_be_visible()
        await page.get_by_role('button', name='成绩与复核', exact=True).click()
        await expect(page.get_by_role('navigation', name='作业班级').get_by_role('button')).to_have_count(3)
        await expect(page.locator('.results-table').get_by_text('一班学生', exact=True)).to_be_visible()
        await page.get_by_role('button', name='软件工程二班', exact=False).click()
        await expect(page.locator('.results-table').get_by_text('二班学生', exact=True)).to_be_visible()
        await expect(page.get_by_text('一班学生', exact=True)).to_have_count(0)
        delay_a = True
        await page.get_by_role('button', name='软件工程一班', exact=False).click()
        await page.get_by_role('button', name='软件工程二班', exact=False).click()
        await page.wait_for_timeout(650)
        await expect(page.locator('.results-table').get_by_text('二班学生', exact=True)).to_be_visible()
        await expect(page.get_by_text('一班学生', exact=True)).to_have_count(0)
        score_input = page.get_by_label(re.compile(r'^最终分数（满分 105 分）$'), exact=True)
        await page.locator('.results-table tbody tr').filter(has_text='二班学生三').get_by_role('button', name='查看并复核', exact=True).click()
        await expect(page.get_by_role('dialog')).to_contain_text('二班学生三')
        await expect(score_input).to_have_value('84')
        await expect(score_input).to_have_attribute('max', '105')
        await score_input.fill('0')
        await page.get_by_role('button', name='仅保存', exact=True).click()
        await expect(page.get_by_text('模拟保存失败', exact=True)).to_be_visible()
        await expect(score_input).to_have_value('0')
        await page.get_by_role('button', name='仅保存', exact=True).click()
        reviewed_row = page.locator('.results-table tbody tr').filter(has_text='二班学生三')
        await expect(reviewed_row.locator('.score')).to_have_text('0.0')
        await expect(reviewed_row.locator('.review-state.reviewed')).to_have_text('已复核')
        assert captures['saves'][-1]['human_score'] == 0
        await page.locator('.results-table tbody tr').filter(has_text='二班学生二').get_by_role('button', name='查看并复核', exact=True).click()
        await expect(page.get_by_role('dialog')).to_contain_text('二班学生二')
        await expect(score_input).to_have_value('83')
        await page.get_by_role('button', name='保存并复核下一位', exact=True).click()
        await expect(page.get_by_role('heading', name='学生复核 / 二班学生', exact=True)).to_be_visible()
        await expect(score_input).to_have_value('90')
        await expect(page.get_by_text('逐项评分与材料依据', exact=True)).to_be_visible()
        await expect(page.get_by_text('源码与截图均已核对', exact=True)).to_be_visible()
        await expect(page.locator('.original-review strong')).to_have_text('90.0')
        await expect(page.get_by_text('截图2识别失败，其余材料正常给分',exact=True)).to_be_visible()
        await expect(page.get_by_text('部分引用未核验，保留基于可读内容的评分',exact=True)).to_be_visible()
        await expect(page.get_by_text('20260001_二班学生.zip（ZIP）',exact=True)).to_be_visible()
        async with context.expect_page() as evidence_page:
            await page.locator('.criterion-evidence').get_by_role('button', name='查看原文件', exact=True).click()
        source_page = await evidence_page.value
        await expect(source_page.locator('body')).to_contain_text('source evidence')
        await source_page.close()
        await page.screenshot(path=str(diagnostics / 'review-reading-evidence.png'), full_page=True)
        await page.get_by_role('button', name='取消', exact=True).click()
        async with page.expect_download() as class_export:
            await page.get_by_role('button', name='导出本班成绩', exact=True).click()
        await class_export.value
        assert captures['exports'][-1]['class_id'] == ['class-b']
        await page.screenshot(path=str(diagnostics / 'class-grades.png'), full_page=True)
        await page.get_by_role('button', name='学生作业与批改', exact=True).click()
        await expect(page.get_by_role('button', name='下载并批改全部 2 个班级', exact=True)).to_be_visible()
        await page.get_by_role('button', name='当前班级', exact=True).click()
        await page.get_by_role('button', name='下载并批改本班作业', exact=True).click()
        assert captures['grades'][-1] == {'class_ids': ['class-b']}
        await expect(page.get_by_role('button', name='全部班级（2）', exact=True)).to_be_enabled()
        await page.get_by_role('button', name='全部班级（2）', exact=True).click()
        await page.get_by_role('button', name='下载并批改全部 2 个班级', exact=True).click()
        assert captures['grades'][-1] == {'class_ids': [], 'all_classes': True}
        await expect(page.locator('.class-progress-row')).to_have_count(2)
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
        await expect(page.locator('.results-table').get_by_text('二班学生', exact=True)).to_be_visible()
        await page.set_viewport_size({'width': 390, 'height': 844})
        await page.screenshot(path=str(diagnostics / 'mobile-grades.png'), full_page=True)
        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        await expect(page.locator('.result-card')).to_have_count(3)
        await expect(page.locator('.result-table-scroll')).to_be_hidden()
        await page.locator('.result-card').first.scroll_into_view_if_needed()
        await page.screenshot(path=str(diagnostics / 'mobile-grade-cards.png'), full_page=True)
        await page.locator('.result-card').first.get_by_role('button', name='查看并复核', exact=True).click()
        await expect(score_input).to_have_value('90')
        await expect(page.locator('dialog.app-modal .modal-footer')).to_be_visible()
        await page.screenshot(path=str(diagnostics / 'mobile-review.png'), full_page=True)
        await page.get_by_role('button', name='取消', exact=True).click()
        await launcher.click()
        await expect(page.locator('#floating-assistant-panel')).to_be_visible()
        panel_box = await page.locator('#floating-assistant-panel').bounding_box()
        assert panel_box['x'] >= 0 and panel_box['x'] + panel_box['width'] <= 391
        await page.screenshot(path=str(diagnostics / 'mobile-assistant.png'), full_page=True)
        await page.get_by_role('button', name='收起 AI 助手', exact=True).click()
        await page.get_by_role('button', name='题目与评分标准', exact=True).click()
        await page.get_by_role('button', name='编辑标准', exact=True).click()
        await picker.select_option('presentation')
        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        await page.locator('.template-picker').scroll_into_view_if_needed()
        await page.screenshot(path=str(diagnostics / 'mobile-rubric-template.png'), full_page=True)
        await page.get_by_role('button', name='取消', exact=True).click()
        await page.set_viewport_size({'width':1440,'height':1000})
        await page.get_by_role('button',name='成绩与复核',exact=True).click()
        await page.get_by_label('搜索本班学生',exact=True).fill('二班学生三')
        await expect(page.locator('.results-table tbody tr')).to_have_count(1)
        await page.get_by_role('button',name='删除当前班级作业',exact=True).click()
        await expect(page.locator('.inline-notice[role="status"]')).to_contain_text('已删除当前班级的评分记录')
        assert captures['deletes']==[{'class_id':['class-b']}]
        assert len(results['class-a'])==1 and results['class-b']==[]
        empty_icon=await page.locator('.workspace-empty > svg').bounding_box()
        assert empty_icon and empty_icon['width']<=48 and empty_icon['height']<=48,'Empty results icon fills the page'
        await page.screenshot(path=str(diagnostics/'class-delete-empty-state.png'),full_page=True)
        material_state['enabled']=True
        await page.get_by_role('button',name='使用已有材料批改',exact=True).click()
        await expect(page.get_by_role('button',name='重新批改',exact=True)).to_be_enabled()
        await page.locator('.batch-list').scroll_into_view_if_needed()
        await page.screenshot(path=str(diagnostics/'regrade-existing-materials.png'),full_page=True)
        await page.get_by_role('button',name='重新批改',exact=True).click()
        await expect(page.get_by_role('button',name='正在批改，请等待',exact=True)).to_be_disabled()
        assert captures['regrades'][0][0]=={'allow_partial':False,'regrade':True,'previous_job_id':'old-completed-job'}
        assert captures['regrades'][0][1]
        assert not errors, errors
        assert not unexpected, unexpected
        await browser.close()
    print('PASS: four complete rubric drafts on selection, template reset/save/cancel isolation, custom and JSON edits, class isolation + slow response, manual zero/retry, class export, three material sources, draggable persistent AI, old routes, filters and mobile.')


if __name__ == '__main__':
    asyncio.run(main())
