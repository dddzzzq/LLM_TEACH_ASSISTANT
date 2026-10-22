"""Bounded semantic page inventory; never read form values or serialize HTML."""
import json
from urllib.parse import urljoin, urlsplit


CONTROLS = ('a,button,input:not([type=hidden]):not([type=password]),select,summary,'
            '[role=button],[role=link],[role=tab],[role=menuitem],[role=checkbox],'
            '[role=combobox],[role=treeitem],[role=option],li,dd,[onclick],[tabindex]:not([tabindex="-1"]),.grade_check,'
            r':is(div,span,p):text-matches("^\\s*[0-9]{4}\\s*[-—–/]\\s*[0-9]{4}\\s*第[一二三123]学期\\s*$")')
HEADINGS = 'h1,h2,h3,h4,h5,h6,[role=heading]'
EXPORT_LABEL = r'^导出(?:作业|提交)?附件(?:\s*[（(].*[）)])?$'
MAX_ELEMENTS = 400
MAX_SNAPSHOT_CHARS = 60000


def public_url(value):
    try:
        parsed = urlsplit(value)
        host, port_number = parsed.hostname, parsed.port
    except ValueError:
        return ''
    if parsed.scheme not in ('http', 'https') or not host:
        return ''
    # Omit user info, query parameters and fragments, including signed links.
    port = f':{port_number}' if port_number else ''
    host = f'[{host}]' if ':' in host else host
    return f'{parsed.scheme}://{host}{port}{parsed.path}'


# DOM names, labels and local semantic containers provide evidence independently
# of the portal's CSS classes. Legacy cname remains an optional hint.
FRAME_SNAPSHOT = r'''els => {
  const clean = value => String(value || '').replace(/\s+/g, ' ').trim();
  const visible = e => !!e.getClientRects().length && e.getBoundingClientRect().width > 0 &&
    e.getBoundingClientRect().height > 0 && getComputedStyle(e).visibility !== 'hidden' &&
    !e.closest('[hidden],[aria-hidden="true"]');
  const headingsSelector = 'h1,h2,h3,h4,h5,h6,[role=heading]';
  const roots = new Set([document]);
  for (const e of els) {
    for (let root=e.getRootNode(); root && !roots.has(root); root=root.host?.getRootNode()) roots.add(root);
  }
  const all = selector => Array.from(roots).flatMap(root => Array.from(root.querySelectorAll(selector)));
  const sensitive = e => e.matches('input[type=password],input[type=hidden],input[type=email]') ||
    e.isContentEditable || /密码|验证码|用户名|账号|password|captcha|username|account/i.test(
      [e.getAttribute('name'), e.id, e.getAttribute('autocomplete'), e.getAttribute('placeholder'),
       e.getAttribute('aria-label')].join(' '));
  const text = e => clean(e.innerText || e.textContent);
  const term = /^\d{4}\s*[-—–/]\s*\d{4}\s*第[一二三123]学期$/;
  const menus = new Map();
  const menuFor = e => {
    const option = e.closest('li,dd,[role=option]');
    if (option && term.test(text(option))) return {option,root:option.closest('ul,dl,[role=listbox]')||option.parentElement};
    // Custom selects may render plain div/span/p options without ARIA roles.
    for(let node=e,depth=0;node?.parentElement && depth<3;node=node.parentElement,depth++) {
      const siblings=[...node.parentElement.children].filter(c=>visible(c)&&term.test(text(c)));
      if(siblings.length>1 && siblings.includes(node)) return {option:node,root:node.parentElement};
    }
    return null;
  };
  const name = e => {
    const labelled = (e.getAttribute('aria-labelledby') || '').split(/\s+/)
      .map(id => e.getRootNode().getElementById?.(id) || document.getElementById(id)).filter(Boolean).map(text).join(' ');
    const labels = Array.from(e.labels || []).filter(visible).map(text).join(' ');
    const icon = e.querySelector('img[alt],svg title');
    return clean(e.getAttribute('aria-label') || labelled || labels || e.getAttribute('title') ||
      (e.matches('input,textarea') ? '' : e.innerText) || e.getAttribute('placeholder') ||
      icon?.getAttribute('alt') || icon?.textContent).slice(0, 120);
  };
  const groups = new Map();
  const records = new Map();
  const recordBoundary = 'li,tr,[role=listitem],[role=row],[cname],' +
    '[data-workid],[data-work-id],[data-assignment-id],[data-courseid],[data-course-id]';
  const interactive = 'a,button,input,select,textarea';
  const headingCounts = new WeakMap();
  const siblingCounts = new WeakMap();
  const signature = node => [node.tagName,node.getAttribute('role')||'',
    [...node.classList].sort().join(' '),
    [...node.children].slice(0,6).map(child=>child.tagName+':'+(child.getAttribute('role')||'')).join(',')].join('|');
  const repeatedRow = node => {
    const parent = node.parentElement;
    if (!parent || node.matches(interactive)) return false;
    if (!siblingCounts.has(parent)) {
      const counts = new Map();
      for (const child of parent.children) {
        if (!visible(child)) continue;
        const key = signature(child);
        counts.set(key,(counts.get(key)||0)+1);
      }
      siblingCounts.set(parent,counts);
    }
    return (siblingCounts.get(parent).get(signature(node))||0)>1 &&
      !!node.querySelector('a,button,[role=button],[role=link],[onclick]');
  };
  const headingCount = node => {
    if (!headingCounts.has(node)) {
      const headings = [...node.querySelectorAll(headingsSelector)];
      if (node.matches(headingsSelector)) headings.push(node);
      // Nested ARIA/native headings describe one title, not two records.
      headingCounts.set(node,headings.filter(h=>visible(h) &&
        !headings.some(other=>other!==h && h.contains(other))).length);
    }
    return headingCounts.get(node);
  };
  const recordFor = e => {
    // A record's title link and its action buttons can have very different
    // immediate parents. Use their common business row as identity.
    const boundary = e.closest(recordBoundary);
    if (boundary && !boundary.matches(interactive)) return boundary;
    let titled = null;
    for (let node=e,depth=0; node && node!==document.body && node!==document.documentElement && depth<9;
         node=node.parentElement,depth++) {
      const count = headingCount(node);
      if (repeatedRow(node) || (node.matches('article') && count<=1)) return node;
      if (count>1) break; // Never merge separate cards via their list container.
      if (count===1 && text(node).length<=600) titled=node;
    }
    return titled;
  };
  const contextFor = e => {
    let fallback = null;
    for (let parent=e.parentElement, depth=0; parent && parent!==document.body && depth<6;
         parent=parent.parentElement, depth++) {
      if (parent.matches('form') && parent.querySelector('input[type=password]')) return null;
      const value = text(parent);
      if (!value || value.length > 600) continue;
      if (!fallback) fallback = parent;
      if (parent.hasAttribute('cname') || parent.matches('li,tr,article,[role=listitem],[role=row],[role=group]') ||
          parent.querySelector(headingsSelector)) return parent;
    }
    return fallback;
  };
  const controls = els.flatMap((e,i) => {
    if (!visible(e) || sensitive(e) || e.closest('form')?.querySelector('input[type=password]')) return [];
    const record = recordFor(e);
    const group = record || contextFor(e);
    if (group && !groups.has(group)) groups.set(group, groups.size);
    if (record && !records.has(record)) records.set(record,records.size);
    const labels = group ? Array.from(group.querySelectorAll(headingsSelector+',span,strong,label,a'))
      .filter(x=>visible(x) && !sensitive(x) && !x.querySelector('input,textarea,[contenteditable=true]'))
      .map(text).filter(x=>x && x.length<=120).slice(0,12) : [];
    if (group?.getAttribute('cname')) labels.unshift(clean(group.getAttribute('cname')));
    const box = e.getBoundingClientRect();
    const nativeRole = e.matches('input') ?
      ({checkbox:'checkbox',radio:'radio',search:'searchbox',button:'button',submit:'button',image:'button'}[e.type]||'textbox') :
      ({A:'link',BUTTON:'button',SELECT:'combobox',SUMMARY:'button'}[e.tagName]||'');
    const label = name(e);
    const menu = term.test(text(e)) ? menuFor(e) : null;
    if(menu && menu.option!==e) return []; // One reference per actual option, not its nested caption.
    if(menu && !menus.has(menu.root)) menus.set(menu.root,menus.size);
    const termOptions = menu ? [...menu.root.querySelectorAll('li,dd,[role=option],div,span,p')]
      .filter(c=>visible(c)&&term.test(text(c))&&menuFor(c)?.option===c&&menuFor(c)?.root===menu.root) : [];
    // Expose select labels as page evidence for the model, never option values.
    const options = e.matches('select') ? [...e.options].map(o=>clean(o.textContent)).slice(0,100) : null;
    const selected = e.matches('select') ? clean(e.selectedOptions[0]?.textContent) :
      (e.matches('input[readonly]') && term.test(e.value) ? clean(e.value) : '');
    const item = {i,tag:e.tagName.toLowerCase(),type:e.getAttribute('type')||'',
      role:e.getAttribute('role')||nativeRole,
      name:label||selected,context:group ? String(group.innerText || '').trim().slice(0,280) : '',
      ...(options ? {options,selected_option:selected} : selected ? {selected_option:selected} : {}),
      ...(options ? {disabled_options:[...e.options].filter(o=>o.disabled).map(o=>clean(o.textContent))} : {}),
      ...(menu ? {semester_kind:'option',semester_menu:menus.get(menu.root),semester_index:termOptions.indexOf(e)+1,
        semester_option_count:termOptions.length} : term.test(selected||label) ? {semester_kind:'trigger'} : {}),
      opens_new_page:e.getAttribute('target')==='_blank'||/\bwindow\.open\s*\(/.test(e.getAttribute('onclick')||''),
      submits_form:!!e.form && ['submit','image','reset'].includes(e.type),
      search_submit:!!e.form && e.form.method.toLowerCase()==='get' && /^(搜索|查找|search)$/i.test(label),
      context_labels:[...new Set(labels)],context_id:group ? groups.get(group) : null,
      record_id:record ? records.get(record) : null,
      href:e.matches('a') ? e.getAttribute('href')||'' : '',
      disabled:!!e.disabled || e.getAttribute('aria-disabled')==='true'||e.classList.contains('disabled')||e.classList.contains('is-disabled'),
      in_viewport:box.bottom>0 && box.right>0 && box.top<innerHeight && box.left<innerWidth};
    return item.name || item.context ? [item] : [];
  });
  const headings = all(headingsSelector).filter(visible).map(e => ({
    level:Number(e.getAttribute('aria-level')||e.tagName.slice(1))||0,text:text(e).slice(0,160),
    in_list:!!e.closest('li,tr,[role=listitem],[role=row]')}));
  const landmarks = all('header,nav,main,aside,[role=navigation],[role=main],[role=dialog]')
    .filter(visible).map(e=>({role:e.getAttribute('role')||e.tagName.toLowerCase(),
      name:clean(e.getAttribute('aria-label')||e.querySelector(headingsSelector)?.innerText).slice(0,120)}));
  return {controls,headings:headings.slice(0,60),heading_count:headings.length,
    landmarks:landmarks.slice(0,30),landmark_count:landmarks.length,
    viewport:{width:innerWidth,height:innerHeight},
    scroll:{x:Math.round(scrollX),y:Math.round(scrollY),height:document.documentElement.scrollHeight}};
}'''


async def inventory(context, target, active_page=None):
    pages, frames, candidates = [], [], []
    names = {target['course_name'], target['assignment_name']}
    for pi, page in enumerate(context.pages):
        if page.is_closed():
            continue
        pages.append({'page_id': str(pi), 'url': public_url(page.url),
                      'title': (await page.title())[:120], 'frame_count': len(page.frames), 'active':page is active_page})
        if active_page is not None and page is not active_page:
            continue  # Keep tab summaries; do not mix old portal controls into the current course page.
        for fi, frame in enumerate(page.frames):
            # Detached/loading frames should invalidate this observation, rather
            # than silently expose a partial inventory as complete.
            controls = frame.locator(CONTROLS)
            data = await controls.evaluate_all(FRAME_SNAPSHOT)
            data['headings'] = sorted(data['headings'], key=lambda h:h['text'] not in names)[:20]
            data['landmarks'] = data['landmarks'][:10]
            frame_id = f'p{pi}f{fi}'
            frames.append({'frame_id': frame_id, 'page_id': str(pi), 'name': frame.name[:100],
                           'url': public_url(frame.url), **{k:v for k,v in data.items() if k != 'controls'},
                           'element_count': len(data['controls'])})
            for item in data['controls']:
                idx = item.pop('i')
                if item['href']:
                    item['href'] = public_url(urljoin(frame.url, item['href']))
                group = item['context_id']
                item['context_id'] = f'{frame_id}c{group}' if group is not None else ''
                record = item['record_id']
                item['record_id'] = f'{frame_id}r{record}' if record is not None else ''
                if 'semester_menu' in item:
                    item['semester_menu'] = f'{frame_id}m{item["semester_menu"]}'
                ref = f'{frame_id}e{idx}'
                candidates.append(({'ref':ref,'page_id':str(pi),'frame_id':frame_id,**item}, page, controls.nth(idx)))
    def priority(candidate):
        item = candidate[0]
        if item.get('semester_kind')=='option' and item['in_viewport']: return 0
        if item.get('semester_kind')=='trigger' and item['in_viewport']: return 1
        if names.intersection([item['name'], *item['context_labels']]): return 2
        if item['in_viewport'] and re_navigation(item['name']): return 3
        return 4 if item['in_viewport'] else 5
    chosen = []
    size = len(json.dumps({'pages':pages,'frames':frames}, ensure_ascii=False))
    for candidate in sorted(candidates, key=priority):
        extra = len(json.dumps(candidate[0], ensure_ascii=False)) + 2
        if size + extra > MAX_SNAPSHOT_CHARS:
            continue
        chosen.append(candidate)
        size += extra
        if len(chosen) == MAX_ELEMENTS:
            break
    targets = {}
    for kind, title in (('course',target['course_name']),('assignment',target['assignment_name'])):
        groups = {}
        for item,_,_ in candidates:
            if title == item['name'] or title in item['context_labels']:
                group = item['record_id'] or item['context_id'] or item['ref']
                groups[group] = bool(target.get('term') and target['term'] in item['context'])
        targets[kind] = {'group_count':len(groups), 'count_basis':'record',
                         'record_ids':list(groups)[:12],
                         'term_matches':[group for group,matched in groups.items() if matched][:2]}
    refs = {item['ref']:(page,loc,item) for item,page,loc in chosen}
    return {'pages':pages,'frames':frames,'elements':[item for item,_,_ in chosen],
            'active_page_id':str(context.pages.index(active_page)) if active_page in context.pages else None,
            'target_candidates':targets,
            'coverage':{'total_elements':len(candidates),'included_elements':len(chosen),
                        'truncated':len(chosen)<len(candidates),
                        'scope':'active_page' if active_page is not None else 'all_pages',
                        'snapshot_char_budget':MAX_SNAPSHOT_CHARS,
                        'landmarks_truncated':any(f['landmark_count']>len(f['landmarks']) for f in frames),
                        'headings_truncated':any(f['heading_count']>len(f['headings']) for f in frames)}}, refs


def re_navigation(name):
    return any(word in name for word in ('个人空间','我的课程','课程','作业','批阅','更多','导出','搜索'))
