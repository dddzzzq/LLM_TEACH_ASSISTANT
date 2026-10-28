"""Requirement review loop with fixture APIs only; never acknowledges real tasks."""
import asyncio
import json
import os
import re
import urllib.request
from pathlib import Path
from playwright.async_api import async_playwright, expect


async def main():
    base = os.environ.get('RPA_UI_BASE_URL', 'http://127.0.0.1:6017')
    root = Path(__file__).resolve().parents[2]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + '/src/components/HomeworkRequirementReview.vue', timeout=10) as response:
        source = response.read().decode()
    vue = next(path for path in re.findall(r'from "([^"]+)"', source) if '/vue.js' in path)
    warning = '要求待核对：引用课件示例5.5和示例5.6，未提供其正文。'
    detail = {'question': '提交示例5.5和示例5.6的运行截图，不要压缩包。', 'raw_question': '官网原始文字：提交运行截图，不要压缩包。', 'complete': False, 'sources': [{'id': '1', 'kind': 'text', 'text': '官网原始文字'}], 'warnings': [warning]}
    state = {'snapshot_id': 'snapshot-fixture', 'operation_id': 'read-v1', 'operation': {'status': 'SUCCEEDED'}, 'detail': detail}
    calls, errors = [], []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=['--no-sandbox'])
        context = await browser.new_context(viewport={'width': 1100, 'height': 1000})
        async def api(route):
            if route.request.method == 'GET':
                await route.fulfill(content_type='application/json', body=json.dumps(state, ensure_ascii=False))
                return
            calls.append(route.request.post_data_json)
            if len(calls) == 1:
                await route.fulfill(status=500, content_type='application/json', body=json.dumps({'detail': '模拟保存失败'}))
            elif len(calls) == 2:
                state['operation_id'] = 'read-v2'
                state['detail']['question'] = '新版本要求：分别提交两个实验截图。'
                await route.fulfill(status=409, content_type='application/json', body='{}')
            else:
                detail.update(question=calls[-1]['question'], complete=True, review={'reviewed_at': '2026-10-28T10:00:00+08:00'})
                await route.fulfill(content_type='application/json', body='{"requirements_status":"READY"}')
        harness = f'''<html><head><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/src/assets/main.css"></head><body><div id="app" style="max-width:760px;margin:24px auto;padding:24px"></div><script type="module">
        import {{createApp,h}} from '{vue}';import Review from '/src/components/HomeworkRequirementReview.vue';
        localStorage.setItem('access_token','fixture');
        createApp({{render:()=>h(Review,{{homeworkId:'fixture-homework',requirementsStatus:'REVIEW_REQUIRED',onConfirmed:q=>window.confirmedQuestion=q}})}}).mount('#app');
        </script></body></html>'''
        page = await context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        await page.route('**/api/portal/**', api)
        await page.route(base + '/requirement-review-fixture', lambda route: route.fulfill(content_type='text/html; charset=utf-8', body=harness))
        await page.goto(base + '/requirement-review-fixture')
        await expect(page.get_by_text(warning, exact=True)).to_be_visible()
        await expect(page.get_by_text(detail['question'], exact=True)).to_be_visible()
        button = page.get_by_role('button', name='保存并确认要求', exact=True)
        await expect(button).to_be_disabled()
        assert not calls
        await page.get_by_text('原始文字与来源', exact=True).click()
        await expect(page.get_by_text(detail['raw_question'], exact=True)).to_be_visible()
        await expect(page.get_by_text('本次来源为文字，没有可下载的图片或附件。', exact=True)).to_be_visible()
        editor = page.get_by_label('核对后的作业要求', exact=True)
        await editor.fill('已核对课件：分别提交两个实验的运行截图，不要压缩包。')
        await page.get_by_role('checkbox').check()
        out = root / '.runtime/diagnostics'
        out.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(out / 'requirement-review.png'), full_page=True)
        await button.click()
        await expect(page.locator('.workspace-error')).to_contain_text('模拟保存失败')
        await expect(editor).to_have_value('已核对课件：分别提交两个实验的运行截图，不要压缩包。')
        await button.click()
        await expect(page.locator('.workspace-error')).to_contain_text('这份要求已更新或正在重新读取。请重新加载最新结果再核对；当前编辑内容已保留。')
        await page.get_by_role('button', name='重新加载', exact=True).click()
        await expect(editor).to_have_value('新版本要求：分别提交两个实验截图。')
        await expect(page.get_by_role('checkbox')).not_to_be_checked()
        await page.get_by_role('checkbox').check()
        await button.click()
        await expect(page.get_by_text('已由教师核对确认。已有成绩不会重新计算。', exact=True)).to_be_visible()
        assert calls[-1]['operation_id'] == 'read-v2' and calls[-1]['confirmed'] is True
        assert await page.evaluate('window.confirmedQuestion') == calls[-1]['question']
        assert not errors, errors
        await browser.close()
    print('PASS: cached candidate/warnings/original text visible, explicit confirmation, failure draft retained, stale revision reload, teacher-reviewed state; fixture APIs only.')


if __name__ == '__main__':
    asyncio.run(main())
