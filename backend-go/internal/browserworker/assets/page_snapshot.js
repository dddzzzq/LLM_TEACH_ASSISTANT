els => {
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
    // Expose stable business IDs, never the URL query or authentication fields.
    const source = record || e;
    const link = e.matches('a[href]') ? e : source.querySelector('a[href]');
    const navigation=e.getAttribute('onclick')||source.getAttribute('onclick')||'';
    const literal=navigation.match(/["']((?:https:\/\/|\/)[^"']+)["']/)?.[1]||'';
    let query = new URLSearchParams();
    try { query = new URL(link?.getAttribute('href')||literal||location.href,location.href).searchParams; } catch {}
    const attr = names => names.map(n=>e.getAttribute(n)||source.getAttribute(n)).find(Boolean)||'';
    const param = names => [...query].find(([k])=>names.includes(k.toLowerCase()))?.[1]||'';
    const platform_ids = {
      course_id:attr(['data-courseid','data-course-id','courseid'])||param(['courseid']),
      class_id:attr(['data-clazzid','data-classid','clazzid','classid'])||param(['clazzid','classid']),
      homework_id:attr(['data-workid','data-work-id','workid'])||param(['workid','homeworkid','taskid'])};
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
      platform_ids,
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
}