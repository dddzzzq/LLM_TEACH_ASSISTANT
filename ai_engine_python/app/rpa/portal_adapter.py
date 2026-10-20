"""Grounded observations and export evidence, using the working portal script as a baseline."""
import hashlib
import re
import asyncio
from .errors import RpaIssue
from .page_snapshot import HEADINGS, EXPORT_LABEL, inventory


TARGET_URL = 'https://learning.xidian.edu.cn/portal'
FORBIDDEN = re.compile(r'删除|发布|评分|保存成绩|提交成绩|修改|编辑|新建|创建|退课|退出登录|logout|delete|publish', re.I)
EXPORT_CONFIRM = re.compile(r'确定|确认|confirm|提交|submit', re.I)
CHOICES = ('label:has(input[type=checkbox]),label:has(input[type=radio]),'
           'input[type=checkbox],input[type=radio],[role=checkbox],[role=radio],'
           '.grade_check,.grade_checked')


EXPORT_TIME = re.compile(r'(?<!\d)(?:\d{4}[-/])?\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?(?!\d)')
EXPORT_STATE = r'重新下载|导出中|等待|导出成功|下载|失败|处理中|已完成|重新导出|已过期|排队中'


def export_identity(row):
    # The portal replaces the displayed creation time with the completion time.
    # Neither that timestamp nor a signed download URL identifies an export.
    text = EXPORT_TIME.sub('', row['text'])
    text = re.sub(r'(?:(?:' + EXPORT_STATE + r')|\s)+$', '', text)
    return re.sub(r'\s+', '', text)


def fingerprint(row):
    stable = row.get('platform_id') or export_identity(row)
    return hashlib.sha256(stable.strip().encode()).hexdigest()[:24]


def reconcile_exports(exports, rows, baseline):
    """Recover old timestamp-based references without changing saved file identities."""
    resolved, claimed = [], set()
    identities = [export_identity(e) for e in exports]
    for export in exports:
        refs = {export['ref'], export.get('row_ref')}
        matches = [r for r in rows if r['ref'] in refs and not r.get('ambiguous')]
        if not matches:
            identity = export_identity(export)
            matches = [r for r in rows if identity and identities.count(identity) == 1 and export_identity(r) == identity
                       and not r.get('ambiguous') and r['ref'] not in baseline
                       and r.get('legacy_ref') not in baseline]
        if len(matches) != 1 or matches[0]['ref'] in claimed:
            resolved.append({**export, 'available': False})
            continue
        row = matches[0]
        claimed.add(row['ref'])
        resolved.append({**export, **row, 'ref': export['ref'], 'row_ref': row['ref'], 'available': True})
    return resolved


def match_new_exports(rows, baseline, assignment, scope):
    """Never associate by list position or by a transient 'exporting' status."""
    fresh = [r for r in rows if r['ref'] not in baseline and not r.get('ambiguous')]
    names = [c['name'] for c in scope.get('classes', [])] if scope.get('mode') == 'selected' else []
    matched = [r for r in fresh if assignment in r['text'] and
               (not names or any(name in r['text'] for name in names))]
    # A unique fresh record for an explicitly selected all-class export can be
    # associated without asking for an otherwise redundant second confirmation.
    if scope.get('mode') == 'all':
        if not scope.get('baseline_observed'):
            return []  # An unopened download center cannot prove which records are new.
        title = re.compile(r'(?:^|[\s_\-/【\[(（:：])' + re.escape(assignment) +
                           r'(?=$|[\s_.\-/】\])）:：])')
        exact = [r for r in rows if r['ref'] not in baseline and title.search(r['text'])]
        return exact if len(exact) == 1 and not exact[0].get('ambiguous') else []
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
        self.target_candidates = {}

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
        view, self.refs = await inventory(self.context, self.target)
        self.target_candidates = view['target_candidates']
        export_entry = await self.export_entry()
        return {**view, 'target_evidence': {'course_confirmed':self.course_confirmed,
                                          'assignment_confirmed':self.assignment_confirmed},
                'can_open_export': bool(self.assignment_confirmed and export_entry)}

    async def export_entry(self):
        candidates = []
        for page in self.context.pages:
            if page.is_closed():
                continue
            for frame in page.frames:
                for role in ('link', 'button', 'menuitem'):
                    entry = frame.get_by_role(role, name=re.compile(EXPORT_LABEL))
                    for i in range(await entry.count()):
                        candidates.append((page, entry.nth(i)))
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            return None  # Never guess between multiple live export menus.
        for page in self.context.pages:
            if page.is_closed(): continue
            for frame in page.frames:
                entries = frame.locator('a,button,[onclick],[role=menuitem]').filter(has_text=re.compile(EXPORT_LABEL))
                for i in range(await entries.count()):
                    if await entries.nth(i).is_visible(): candidates.append((page, entries.nth(i)))
        if candidates:
            return candidates[0] if len(candidates) == 1 else None
        # Compatibility with the old portal's hidden, site-owned menu.
        for page in self.context.pages:
            if page.is_closed(): continue
            for frame in page.frames:
                entry = frame.locator('ul.morePop a').filter(has_text='导出作业附件')
                active_view = await entry.evaluate('''e => {
                  const menu=e.closest('ul.morePop');
                  for(let node=menu?.parentElement; node; node=node.parentElement) {
                    const style=getComputedStyle(node);
                    if(node.hidden || node.getAttribute('aria-hidden')==='true' ||
                       style.display==='none' || style.visibility==='hidden') return false;
                  }
                  return !!menu;
                }''') if await entry.count() == 1 else False
                if active_view:
                    candidates.append((page, entry))
        return candidates[0] if len(candidates) == 1 else None

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
        def matches(info, title):
            return info['name'] == title or title in info.get('context_labels', [])
        if matches(item, course):
            evidence = self.target_candidates.get('course', {})
            group = item.get('record_id') or item.get('context_id') or item.get('ref')
            if evidence.get('group_count', 0) > 1 and evidence.get('term_matches') != [group]:
                raise RpaIssue('TARGET_AMBIGUOUS', '存在同名课程，需要人工选择或提供学期',
                               {'kind':'course', **evidence})
            self.course_confirmed = True
            self.course_pages.add(page)
        if self.course_confirmed and matches(item, assignment):
            if item['name'] == assignment or re.search('批阅|评阅|查看提交|查看作业|作业详情', item['name']):
                evidence = self.target_candidates.get('assignment', {})
                if evidence.get('group_count', 0) > 1:
                    raise RpaIssue('TARGET_AMBIGUOUS', '存在多个同名作业，需要人工核对',
                                   {'kind':'assignment', **evidence})
                self.assignment_confirmed = True
                self.assignment_pages.add(page)

    async def reconcile(self):
        # After human navigation, accept an explicit course title and assignment heading as evidence.
        for page in self.context.pages:
            if page.is_closed():
                continue
            for frame in page.frames:
                course_titles = frame.locator(HEADINGS + ',.courseName,.course-name,.coursename')
                title_reader = '''els => els.filter(e => e.getClientRects().length &&
                  !e.closest('[hidden],[aria-hidden="true"],li,tr,[role=listitem],[role=row],[role=list]'))
                  .map(e => (e.innerText||'').trim())'''
                if self.target['course_name'] in await course_titles.evaluate_all(title_reader):
                    self.course_confirmed = True
                    self.course_pages.add(page)
                headings = frame.locator(HEADINGS + ',.work-title,.workTitle,.mark_title')
                visible_titles = await headings.evaluate_all(title_reader)
                if self.course_confirmed and self.target['assignment_name'] in visible_titles:
                    # A list title alone is insufficient; the export menu must belong to this page.
                    entry = await self.export_entry()
                    if entry and entry[0] == page:
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
        for control, info in await self.export_choices(modal):
            if info['group'] != 'scope' or not info['name'] or len(info['name']) > 120:
                continue
            ref = 'class_' + str(len(options))
            options.append({'id': ref, 'name': info['name'], 'all': info['all']})
            self.class_locators[ref] = control
        return options

    async def export_choices(self, modal):
        """Classify options by their section; descriptions are not class names."""
        controls = modal.locator(CHOICES)
        infos = await controls.evaluate_all(r'''els => {
          const clean=value=>String(value||'').replace(/\s+/g,' ').trim();
          const visible=e=>!!e.getClientRects().length && getComputedStyle(e).visibility!=='hidden' &&
            !e.closest('[hidden],[aria-hidden=true]');
          const labelled=e=>clean(e.getAttribute('aria-label') ||
            (e.getAttribute('aria-labelledby')||'').split(/\s+/).map(id=>document.getElementById(id)?.textContent||'').join(' '));
          const sectionName=e=>labelled(e) || clean(e.querySelector(':scope > legend')?.textContent);
          return els.flatMap((e,i)=>{
            if(!visible(e)) return [];
            if(!e.matches('label') && e.closest('label')?.querySelector('input[type=checkbox],input[type=radio]')) return [];
            const roleOwner=e.closest('[role=checkbox],[role=radio]');
            if(roleOwner && roleOwner!==e) return [];
            const owner=e.closest('label,li,.export-range,.out,.nameItem,[role=radio],[role=checkbox]')||e.parentElement;
            const name=labelled(e) || clean(owner?.innerText);
            let group='';
            for(let node=e; node && !node.matches('[role=dialog],.popDiv.centerPop'); node=node.parentElement) {
              if(node.matches('.nameItem')) {group='naming';break;}
              if(node.matches('.out')) {group='format';break;}
              if(node.matches('.export-range')) {group='scope';break;}
              if(node.matches('fieldset,[role=group],[role=radiogroup]')) {
                const heading=sectionName(node);
                if(/命名|文件名/.test(heading)) group='naming';
                else if(/导出格式|导出内容|附件格式/.test(heading)) group='format';
                else if(/导出范围|班级范围|导出班级|选择班级/.test(heading)) group='scope';
                if(group) break;
              }
            }
            // Compatibility with simple labels without fieldsets. Recognize
            // actual option labels, never an incidental "班级" in a description.
            if(!group && /默认格式|文件原名|命名|学号.*姓名/.test(name)) group='naming';
            if(!group && /^(导出完整答题记录|仅导出文档|导出(?:学生)?(?:提交|作业)?附件)/.test(name)) group='format';
            if(!group && /^(?:导出(?:作业下)?)?(?:本班级|当前班级|所有班级|全部班级)(?:$|\s|作业|附件|[（(])/.test(name)) group='scope';
            if(!group && /^(班级|教学班|行政班)[A-Za-z0-9一二三四五六七八九十（( -]/.test(name)) group='scope';
            return [{i,name,group,all:group==='scope' &&
              (!!e.closest('.all') || /(?:所有|全部)班级/.test(name) || /^(全部|所有)$/.test(name))}];
          });
        }''')
        return [(controls.nth(info['i']), info) for info in infos]

    @staticmethod
    async def checked(control):
        return await control.evaluate('''e => {
          if(e.matches('input[type=checkbox],input[type=radio]')) return e.checked;
          const parent=e.closest('label,li,.export-range,.out,[role=radio],[role=checkbox]');
          const inputs=(parent||e).querySelectorAll('input[type=checkbox],input[type=radio]');
          if(inputs.length===1) return inputs[0].checked;
          for(const node of [e,parent].filter(Boolean)) {
            const state=node.getAttribute('aria-checked') ?? node.getAttribute('data-checked');
            if(state!==null) return state==='true' || state==='1';
          }
          const selected=/(^|\s)(checked|active|selected|check_on|on|grade_checked|check_answer)(\s|$)/;
          return [e,parent].filter(Boolean).some(node=>selected.test(String(node.className)));
        }''')

    async def wait_checked(self, control, desired):
        for _ in range(16):
            if await self.checked(control) == desired:
                return True
            await asyncio.sleep(.1)
        return False

    async def select_attachments(self, modal):
        candidates = []
        other_formats = []
        for control, info in await self.export_choices(modal):
            if info['group'] != 'format':
                continue
            if re.match(r'^导出(?:学生)?(?:提交|作业)?附件', info['name']):
                candidates.append(control)
            elif re.match(r'^导出完整答题记录|^仅导出文档', info['name']):
                other_formats.append(control)
        if len(candidates) > 1:
            raise RpaIssue('ATTACHMENT_AMBIGUOUS', '发现多个附件导出选项，请在官网选择“导出提交附件”后交还自动化')
        if not candidates:
            if await modal.get_by_text(re.compile('导出完整答题记录|仅导出文档')).count():
                raise RpaIssue('ATTACHMENT_MISSING', '未识别到“导出提交附件”控件，请在官网选中该项后交还自动化')
            return  # Some attachment-only dialogs have no separate content choice.
        control = candidates[0]
        if not await self.checked(control):
            await control.click(timeout=5000)
        if not await self.wait_checked(control, True):
            raise RpaIssue('ATTACHMENT_UNVERIFIED', '未能核验“导出提交附件”的选中状态，请检查该选项后交还自动化')
        if any([await self.checked(other) for other in other_formats]):
            raise RpaIssue('ATTACHMENT_MISMATCH', '导出格式仍选中了其他内容，请在官网只选“导出提交附件”后交还自动化')

    async def scope_diagnostics(self, modal):
        # Selected-state metadata only, intentionally omitting labels, attributes
        # such as value/onclick/href, and any text from student submissions.
        return await modal.locator(CHOICES).evaluate_all('''els => els.slice(0,30).map(e=>({
          tag:e.tagName.toLowerCase(),classes:String(e.className).slice(0,100),
          parent_classes:String(e.parentElement?.className||'').slice(0,100),
          role:e.getAttribute('role')||'',aria_checked:e.getAttribute('aria-checked'),
          checked:e.matches('input[type=checkbox],input[type=radio]')?e.checked:null
        }))''')

    async def apply_scope(self, modal, scope, options):
        requested = set(scope.get('class_ids', []))
        available = {o['id']: o for o in options}
        if scope.get('mode') == 'all':
            selected = {o['id'] for o in options if o['all']}
            if len(selected) > 1:
                raise RpaIssue('SCOPE_AMBIGUOUS', '识别到多个全部班级控件，请检查导出范围分组后交还自动化')
        else:
            if not requested or not requested <= available.keys() or any(available[x]['all'] for x in requested):
                raise RpaIssue('SCOPE_CHANGED', '班级列表已变化，请重新选择当前列表中的具体班级')
            selected = requested
        if not selected:
            raise RpaIssue('SCOPE_MISSING', '当前页面没有可验证的班级范围，请打开正确的导出设置')
        # Select requested radios first: their group deselects its previous choice.
        # Clicking an already selected radio cannot uncheck it.
        for option in sorted(options, key=lambda o: o['id'] not in selected):
            control = self.class_locators[option['id']]
            desired = option['id'] in selected
            if await self.checked(control) != desired:
                await control.click(timeout=5000)
                await self.wait_checked(control, desired)
        actual = {o['id'] for o in options if await self.checked(self.class_locators[o['id']])}
        if actual != selected:
            raise RpaIssue('SCOPE_MISMATCH', '官网班级选择未能与已确认范围一致，请检查班级选项后交还自动化；无需再次确认作业')
        await self.select_attachments(modal)
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
                    if not await loc.is_visible():
                        continue
                    info = await loc.evaluate('''e => ({text:(e.innerText||'').trim(),
                      platform_id:e.getAttribute('data-id')||e.getAttribute('data-taskid')||e.id||'',
                      href:e.querySelector('a.download_ic,a[download]')?.getAttribute('href')||''})''')
                    if not info['text']:
                        continue
                    if info['href'] in ('#', 'javascript:;', 'javascript:void(0)'):
                        info['href'] = ''
                    info['ref'] = fingerprint(info)
                    text = info['text']
                    legacy = info.get('platform_id') or re.sub(
                        r'重新下载|导出中|等待|导出成功|下载|失败|处理中|已完成|重新导出|\s+', '', text)
                    info['legacy_ref'] = hashlib.sha256(legacy.strip().encode()).hexdigest()[:24]
                    info['status'] = 'FAILED' if re.search('失败|错误|已过期', text) else ('READY' if re.search('导出成功|已完成', text) else 'PENDING')
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

    async def export_center_visible(self):
        for page in self.context.pages:
            if page.is_closed():
                continue
            for frame in page.frames:
                centers = frame.locator('#downloadcenter,.downloadCenter')
                for i in range(await centers.count()):
                    if await centers.nth(i).is_visible():
                        return True
        return False
