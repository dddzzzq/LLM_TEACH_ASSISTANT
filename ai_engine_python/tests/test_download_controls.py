"""Downloads remain tied to verified exports even behind a floating panel."""
import time
import unittest
import zipfile
from pathlib import Path

from playwright.async_api import TimeoutError as BrowserTimeout

from app.rpa.download_manager import verify_zip
from app.rpa.errors import RpaIssue
from tests import test_rpa_worker as fixtures


class DownloadControlTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.BrowserTaskTests.asyncSetUp
    asyncTearDown = fixtures.BrowserTaskTests.asyncTearDown
    bare_task = fixtures.BrowserTaskTests.bare_task

    async def covered_downloads(self, in_frame=False, text_button=False):
        task = await self.bare_task()
        zip_path=Path(self.temp.name)/'student.zip'
        with zipfile.ZipFile(zip_path,'w') as archive:
            archive.writestr('20260001-test/report.txt','student answer')
        downloaded=[]
        rows='''<div id=downloadcenter><table><tbody>
          <tr data-id=upper><td>第一次作业 班级A</td><td id=status>导出中</td>
            <td><a class=download_ic href=/download/upper>下载</a></td></tr>
          <tr data-id=lower><td>第一次作业 班级B 导出成功</td>
            <td><a class=download_ic href=/download/lower
              onclick="top.document.querySelector('#bubble').hidden=false">下载</a></td></tr>
        </tbody></table></div>'''
        if text_button:
            rows=rows.replace('<a class=download_ic href=/download/upper>下载</a>',
                '<button onclick="location.href=\'/download/upper\'">下载</button>')
        bubble='''<aside id=bubble hidden style="position:fixed;inset:0;background:#ddd;z-index:99999"
            onclick="window.bubbleHits=(window.bubbleHits||0)+1">下载悬浮窗</aside>'''
        html=('<iframe src=/exports width=900 height=400></iframe>' if in_frame else rows)+bubble
        async def route(request):
            url=request.request.url
            if '/download/' in url:
                name=url.rsplit('/',1)[-1]
                downloaded.append(name)
                await request.fulfill(body=zip_path.read_bytes(),content_type='application/zip',
                    headers={'Content-Disposition':f'attachment; filename={name}.zip'})
            else:
                await request.fulfill(content_type='text/html; charset=utf-8',
                    body=rows if url.endswith('/exports') else html)
        await task.context.route('**/*',route)
        await task.page.goto('https://mooc1.chaoxing.com/course')
        frame=task.page.frames[1] if in_frame else task.page.main_frame
        await frame.wait_for_selector('#downloadcenter')
        exports=await task.adapter.export_rows()
        self.assertEqual(len(exports),2)
        task.update(exports=exports,export_attempted=True,export_baseline=[],
                    export_started_at=time.time(),stage='WAIT_EXPORT',
                    class_scope={'mode':'all','classes':[]})
        task.gate.set()
        await task.process_exports()
        self.assertEqual(downloaded,['lower'])
        self.assertTrue(await task.page.locator('#bubble').is_visible())
        await frame.locator('#status').evaluate("e=>e.innerText='导出成功'")
        # Prove this fixture reproduces the actual pointer-interception bug.
        upper=frame.locator('[data-id=upper]').get_by_text('下载',exact=True)
        with self.assertRaises(BrowserTimeout): await upper.click(trial=True,timeout=200)
        await task.process_exports()
        await task.process_exports()
        self.assertEqual(task.state['status'],'SUCCEEDED')
        self.assertEqual(task.state['control_mode'],'AUTO')
        self.assertEqual(downloaded,['lower','upper'])  # One request per associated record.
        self.assertEqual(len(task.state['files']),2)
        for file in task.state['files']:
            self.assertEqual(verify_zip(file['path'])['status'],'VERIFIED')
        self.assertFalse(await task.page.evaluate('!!window.bubbleHits'))
        self.assertTrue(await task.page.locator('#bubble').is_visible())

    async def test_second_file_downloads_behind_floating_panel(self):
        await self.covered_downloads()

    async def test_download_inside_iframe_survives_parent_page_overlay(self):
        await self.covered_downloads(in_frame=True)

    async def test_text_download_button_keeps_its_platform_handler(self):
        await self.covered_downloads(text_button=True)

    async def test_invalid_or_changed_download_controls_never_activate(self):
        cases=[
            ('disabled',"document.querySelector('a').setAttribute('aria-disabled','true')"),
            ('hidden',"document.querySelector('a').hidden=true"),
            ('ambiguous',"document.querySelector('td').append(document.querySelector('a').cloneNode(true))"),
            ('identity',"document.querySelector('tr').dataset.id='different-export'"),
            ('status',"document.querySelector('span').innerText='导出失败'"),
            ('reordered',"document.querySelector('tbody').insertAdjacentHTML('afterbegin','<tr data-id=foreign><td>其他作业 导出成功 <a class=download_ic onclick=window.hits++>下载</a></td></tr>')"),
        ]
        for name,change in cases:
            with self.subTest(name=name):
                task=await self.bare_task()
                await task.page.set_content('''<script>window.hits=0</script><div id=downloadcenter>
                    <table><tbody><tr data-id=ours><td>第一次作业 <span>导出成功</span>
                    <a class=download_ic onclick="window.hits++">下载</a></td></tr></tbody></table></div>''')
                rows=await task.adapter.export_rows()
                await task.page.evaluate(change)
                with self.assertRaises(RpaIssue): await task.adapter.click_export_download(rows[0]['ref'])
                self.assertEqual(await task.page.evaluate('window.hits'),0)
