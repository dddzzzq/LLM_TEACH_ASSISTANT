els => els.map(e => {
  const row = e.closest('[cname],.myde_course_item') || e.closest('li,tr,article,[data-courseid],[data-workid],[courseid],[workid],.courseItem,.course-item,.workList') || e;
  const a = e.matches('a') ? e : row.querySelector('a[href],a[onclick]');
  const title = row.querySelector('[data-title],h1,h2,h3,h4,.course-name,.myde_course_name,.workName,.work-name');
  const attr = names => [e,a,row].filter(Boolean).flatMap(n => names.map(k=>n.getAttribute(k))).find(Boolean)||'';
  const navigation=e.getAttribute('onclick')||a?.getAttribute('onclick')||row.getAttribute('onclick')||'';
  const classCall = navigation.match(/(?:changeClazz|changeClass|selectClass)\s*\(([^)]*)\)/i);
  const classArgs = classCall ? [...classCall[1].matchAll(/(?:^|,)\s*["']?(\d+)["']?(?=\s*(?:,|$))/g)].map(m=>m[1]) : [];
  // The legacy teacher menu keeps its class ID in a bare numeric data attribute.
  const classData = classCall && e.matches('li.classli') && /^\d+$/.test(e.getAttribute('data')||'') ? e.getAttribute('data') : '';
  const courseParam = new URL(document.baseURI).searchParams.get('courseid');
  const classCandidates = [...new Set(classArgs.length>1?classArgs.filter(value=>value!==courseParam):classArgs)];
  const literal=navigation.match(/["']((?:https:\/\/|\/)[^"']+)["']/)?.[1]||'';
  const candidates=[a?.getAttribute('href'),e.getAttribute('data-url'),row.getAttribute('data-url'),literal].filter(Boolean);
  let href='';
  for(const value of candidates) {
    if(/^(?:javascript:|#)/i.test(value)) continue;
    try { href=new URL(value,document.baseURI).href; break; } catch {}
  }
  const path=[];
  for(let n=row; n && n!==document.documentElement; n=n.parentElement)
    path.push(n.tagName+':'+[...n.parentElement.children].indexOf(n));
  const terms=[...document.querySelectorAll('select option:checked')].map(n=>(n.textContent||'').trim()).filter(t=>/^\d{4}\s*[-—–/]\s*\d{4}\s*第[一二三123]学期$/.test(t));
  return {
    source_route:document.baseURI,source_path:path.reverse().join('/'),
    source_page:(document.querySelector('[aria-current=page],.pagination .active,.pagination .current')?.textContent||'').trim().slice(0,80),
    course_id:attr(['data-courseid','data-course-id','courseid']),
    work_id:attr(['data-workid','data-work-id','workid']),
    class_id:attr(['data-clazzid','data-clazz-id','data-classid','data-class-id','clazzid','classid'])||(e.tagName==='OPTION'?e.value:'')||classData||(classCandidates.length===1?classCandidates[0]:''),
    class_name:e.getAttribute('data-clazzname')||e.getAttribute('data-class-name')||row.getAttribute('data-class-name')||'',
    href,
    title:((e.tagName==='OPTION'?e.textContent:'')||row.getAttribute('cname')||e.getAttribute('data-title')||title?.getAttribute('data-title')||title?.innerText||e.getAttribute('title')||a?.getAttribute('title')||e.innerText||'').trim().replace(/\s+/g,' ').slice(0,255),
    term:(row.getAttribute('data-term')||row.querySelector('[data-term],.term,.semester')?.textContent||(new Set(terms).size===1?terms[0]: '')||'').trim().slice(0,128),
    state:(row.getAttribute('data-state')||row.querySelector('.status,.state')?.textContent||'').trim().slice(0,64),
    deadline:(row.getAttribute('data-deadline')||row.querySelector('.deadline')?.textContent||'').trim().slice(0,128)
  };
})