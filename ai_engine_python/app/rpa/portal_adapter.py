"""Grounded observations and export evidence, using the working portal script as a baseline."""
import hashlib
import re
from urllib.parse import urlsplit


TARGET_URL = 'https://learning.xidian.edu.cn/portal'
FORBIDDEN = re.compile(r'删除|发布|评分|保存成绩|提交成绩|修改|编辑|新建|创建|退课|退出登录|logout|delete|publish', re.I)
EXPORT_CONFIRM = re.compile(r'确定|确认|confirm|提交|submit', re.I)


def fingerprint(row):
    # Download hrefs often appear only after packaging; using them changes identity mid-task.
    stable = row.get('platform_id')
    if not stable:
        stable = re.sub(r'重新下载|导出中|等待|导出成功|下载|失败|处理中|已完成|重新导出|\s+', '', row['text'])
    return hashlib.sha256(stable.strip().encode()).hexdigest()[:24]


def match_new_exports(rows, baseline, assignment, scope):
    """Never associate by list position or by a transient 'exporting' status."""
    fresh = [r for r in rows if r['ref'] not in baseline and not r.get('ambiguous')]
    names = [c['name'] for c in scope.get('classes', [])] if scope.get('mode') == 'selected' else []
    matched = [r for r in fresh if assignment in r['text'] and
               (not names or any(name in r['text'] for name in names))]
    # A single row per selected class is required; otherwise ask a human to bind records.
    if names and len(matched) == len(names) and all(
            sum(name in r['text'] for r in matched) == 1 for name in names):
        return matched
    return []


class PortalAdapter:
    def __init__(self, context, target):
        self.context = context
        self.target = target
        self.refs = {}
        self.export_locators = {}
        self.class_locators = {}
        self.course_confirmed = False
        self.assignment_confirmed = False
        self.course_pages = set()
        self.assignment_pages = set()

    async def login_required(self):
        for page in self.context.pages:
            if page.is_closed():
                continue
            for frame in page.frames:
                challenge = frame.locator('input[type=password]:visible,#sliderDiv:visible,input#captcha:visible')
                if await challenge.count():
                    return True
        return False

    async def authenticated(self):
        if await self.login_required():
            return False
        for page in self.context.pages:
            if page.is_closed():
                continue
            if 'learning.xidian.edu.cn' in page.url:
                if await page.get_by_text('个人空间', exact=True).count():
                    return not await page.locator('.denglu').is_visible() if await page.locator('.denglu').count() else True
            if 'chaoxing.com' in page.url and await page.locator('a[title="作业"]').count():
                return True
        return False

    async def observe(self):
        self.refs = {}
        pages, elements = [], []
        for pi, page in enumerate(self.context.pages):
            if page.is_closed():
                continue
            page_id = str(pi)
            parsed = urlsplit(page.url)
            pages.append({'page_id': page_id, 'url': parsed.scheme + '://' + parsed.netloc + parsed.path,
                          'title': (await page.title())[:100]})
            for fi, frame in enumerate(page.frames):
                controls = frame.locator('a,button,input:not([type=hidden]):not([type=password]),select,[role=button],.grade_check')
                # Extract no input values, cookies, hidden tokens or whole student submission content.
                data = await controls.evaluate_all('''els => els.map((e,i) => {
                  const box=e.getBoundingClientRect();
                  const visible=box.width>0 && box.height>0 && getComputedStyle(e).visibility!=='hidden';
                  const row=e.closest('.myde_course_item,li,tr,.export-range,label');
                  return {i, visible, tag:e.tagName.toLowerCase(), type:e.getAttribute('type')||'',
                    name:(e.getAttribute('aria-label')||e.getAttribute('title')||e.innerText||e.getAttribute('placeholder')||'').trim().slice(0,100),
                    context:(row?.getAttribute('cname')||row?.innerText||'').trim().slice(0,180)};
                })''')
                for item in data:
                    if not item.pop('visible') or item['type'] in ('password', 'email'):
                        continue
                    if item['tag'] == 'input' and re.search('用户名|密码|验证码|username|password|captcha', item['name'], re.I):
                        continue
                    if not item['name'] and not item['context']:
                        continue
                    ref = f'p{pi}f{fi}e{item.pop("i")}'
                    idx = int(ref.split('e')[-1])
                    self.refs[ref] = (page, controls.nth(idx), item)
                    elements.append({'ref': ref, 'page_id': page_id, **item})
        export_entry = await self.export_entry()
        return {'pages': pages, 'elements': elements[:250],
                'can_open_export': bool(self.assignment_confirmed and export_entry)}

    async def export_entry(self):
        for page in self.context.pages:
            if page.is_closed():
                continue
            for frame in page.frames:
                entry = frame.locator('ul.morePop a').filter(has_text='导出作业附件')
                if await entry.count() == 1:
                    return page, entry
        return None

    async def open_export(self):
        entry = await self.export_entry()
        if not self.assignment_confirmed or not entry:
            raise ValueError('作业导出入口尚未核验')
        page, link = entry
        if await link.is_visible():
            await link.click(timeout=8000)
        else:
            # Same site-owned action as the previously working script, never model-supplied JS.
            await link.evaluate('element => element.click()')
        return page

    async def record_navigation(self, locator, item, page):
        course = self.target['course_name']
        assignment = self.target['assignment_name']
        course_card = locator.locator('xpath=ancestor-or-self::*[@cname][1]')
        is_course = await course_card.count() and (await course_card.get_attribute('cname')) == course
        # A course name must identify the card itself, not an arbitrary surrounding list.
        if is_course or item['name'] == course:
            matches = 0
            for candidate in self.context.pages:
                for frame in candidate.frames:
                    names = await frame.locator('[cname]').evaluate_all('els => els.map(e=>e.getAttribute("cname"))')
                    matches += sum(name == course for name in names)
            if matches > 1 and not (self.target.get('term') and self.target['term'] in item['context']):
                raise ValueError('存在同名课程，需要人工选择或提供学期')
            self.course_confirmed = True
            self.course_pages.add(page)
        row = locator.locator('xpath=ancestor::li[.//h2][1]')
        if self.course_confirmed and await row.count():
            titles = await row.locator('h2').all_text_contents()
            if any(t.strip() == assignment for t in titles) and '批阅' in item['name']:
                self.assignment_confirmed = True
                self.assignment_pages.add(page)

    async def reconcile(self):
        # After human navigation, accept an explicit course title and assignment heading as evidence.
        for page in self.context.pages:
            if page.is_closed():
                continue
            for frame in page.frames:
                course_titles = frame.locator('h1,h2,.courseName,.course-name,.coursename')
                if self.target['course_name'] in [s.strip() for s in await course_titles.all_text_contents()]:
                    self.course_confirmed = True
                    self.course_pages.add(page)
                headings = frame.locator('h1,h2,.work-title,.workTitle,.mark_title')
                visible_titles = await headings.evaluate_all('els => els.filter(e=>e.getClientRects().length).map(e=>e.innerText.trim())')
                if self.course_confirmed and self.target['assignment_name'] in visible_titles:
                    # A list title alone is insufficient; the export menu must belong to this page.
                    entry = frame.locator('ul.morePop a').filter(has_text='导出作业附件')
                    if await entry.count() and await entry.first.evaluate('e => !!e.closest("ul.morePop")?.parentElement?.getClientRects().length'):
                        self.assignment_confirmed = True
                        self.assignment_pages.add(page)

    async def export_area(self):
        for page in self.context.pages:
            if page.is_closed():
                continue
            for frame in page.frames:
                modal = frame.locator('.popDiv.centerPop,[role=dialog]').filter(has_text='导出')
                for i in range(await modal.count()):
                    if await modal.nth(i).is_visible():
                        return page, frame, modal.nth(i)
        return None

    async def classes(self, modal):
        self.class_locators = {}
        options = []
        controls = modal.locator('.export-range .grade_check,input[type=checkbox],[role=checkbox]')
        for i in range(await controls.count()):
            control = controls.nth(i)
            if not await control.is_visible():
                continue
            info = await control.evaluate('''e => {
              const label=e.closest('label,li,.export-range')||e.parentElement;
              return {name:(label?.innerText||e.getAttribute('aria-label')||'').trim(),
                all:!!e.closest('.all') || /所有班级|全部班级/.test(label?.innerText||'')};
            }''')
            if not info['name'] or len(info['name']) > 120:
                continue
            ref = 'class_' + str(i)
            options.append({'id': ref, 'name': info['name'], 'all': info['all']})
            self.class_locators[ref] = control
        return options

    @staticmethod
    async def checked(control):
        return await control.evaluate('''e => {
          if(e.matches('input')) return e.checked;
          if(e.getAttribute('aria-checked')!==null) return e.getAttribute('aria-checked')==='true';
          const input=e.closest('label,li,.export-range')?.querySelector('input[type=checkbox],input[type=radio]');
          if(input) return input.checked;
          return /(^|\s)(checked|active|selected|check_on|on)(\s|$)/.test(e.className);
        }''')

    async def apply_scope(self, modal, scope, options):
        requested = set(scope.get('class_ids', []))
        available = {o['id']: o for o in options}
        if scope.get('mode') == 'all':
            selected = {o['id'] for o in options if o['all']}
        else:
            if not requested or not requested <= available.keys() or any(available[x]['all'] for x in requested):
                raise ValueError('请选择当前列表中的具体班级')
            selected = requested
        if not selected:
            raise ValueError('当前页面没有可验证的班级范围，请人工打开正确的导出设置')
        for option in options:
            control = self.class_locators[option['id']]
            desired = option['id'] in selected
            if await self.checked(control) != desired:
                await control.click(timeout=5000)
        actual = {o['id'] for o in options if await self.checked(self.class_locators[o['id']])}
        if actual != selected:
            raise ValueError('官网当前班级选择与确认范围不一致，请人工核对')
        attachment = modal.locator('div.out').filter(has_text='导出提交附件').locator('.grade_check')
        if await attachment.count():
            if not await self.checked(attachment.first):
                await attachment.first.click(timeout=5000)
            if not await self.checked(attachment.first):
                raise ValueError('无法核验“导出提交附件”选项')
        return {'mode': scope['mode'], 'classes': [available[x] for x in sorted(selected)]}

    async def export_rows(self):
        rows = []
        self.export_locators = {}
        for page in self.context.pages:
            if page.is_closed():
                continue
            for frame in page.frames:
                center = frame.locator('#downloadcenter,.downloadCenter')
                if not await center.count():
                    continue
                locators = center.locator('tbody tr')
                if not await locators.count():
                    locators = center.locator('.dataBody_td')
                for i in range(await locators.count()):
                    loc = locators.nth(i)
                    info = await loc.evaluate('''e => ({text:(e.innerText||'').trim(),
                      platform_id:e.getAttribute('data-id')||e.getAttribute('data-taskid')||e.id||'',
                      href:e.querySelector('a.download_ic,a[download]')?.getAttribute('href')||''})''')
                    if not info['text']:
                        continue
                    if info['href'] in ('#', 'javascript:;', 'javascript:void(0)'):
                        info['href'] = ''
                    info['ref'] = fingerprint(info)
                    text = info['text']
                    info['status'] = 'FAILED' if re.search('失败|错误', text) else ('READY' if re.search('导出成功|已完成', text) else 'PENDING')
                    # Only opaque refs and display text are exposed outside the executor.
                    rows.append({k: v for k, v in info.items() if k not in ('href', 'platform_id')})
                    self.export_locators[info['ref']] = (page, loc)
        counts = {}
        for row in rows:
            counts[row['ref']] = counts.get(row['ref'], 0) + 1
        for row in rows:
            row['ambiguous'] = counts[row['ref']] > 1
            if row['ambiguous']:
                self.export_locators.pop(row['ref'], None)
        return rows
