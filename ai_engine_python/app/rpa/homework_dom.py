"""Read published homework identities from DOM evidence, without running handlers."""

HOMEWORK_ROWS = "li:has(.list_li_tit),li:has(a.piyueBtn)"
HOMEWORK_SELECTOR = (
    "a[href*='workid=' i],a[href*='homeworkid=' i],a[href*='taskid=' i],"
    "[data-workid],[data-work-id],[workid],[data-homework-id],[homeworkid],"
    "[data-task-id],[taskid],[onclick*='workid' i],[onclick*='homeworkid' i],"
    "input[name='workid' i],input[name='homeworkid' i]," + HOMEWORK_ROWS
)


# generic_extract is the shared course/class record reader. Homework evidence
# needs its own row/identity handling: old portals use JS buttons and hidden IDs.
def homework_extract(generic_extract):
    return r"""els => {
  const visible = e => !!e.getClientRects().length && getComputedStyle(e).visibility!=='hidden' && !e.closest('[hidden],[aria-hidden=true]');
  const clean = v => String(v||'').trim().replace(/\s+/g,' ');
  const valid = v => /^[A-Za-z0-9_-]{1,128}$/.test(v||'');
  const names = ['data-workid','data-work-id','workid','data-homework-id','homeworkid','data-task-id','taskid'];
  const idKeys = ['workid','homeworkid','taskid'];
  const trusted = u => ['learning.xidian.edu.cn','chaoxing.com'].includes(u.hostname)||u.hostname.endsWith('.chaoxing.com')||u.hostname.endsWith('.chaoxing.com.cn');
  const namedArguments = script => {
    const ids=[];
    for(const call of script.matchAll(/\b([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(([^()]*)\)/g)) {
      let fn=window;
      for(const key of call[1].replace(/^window\./,'').split('.')) {
        const descriptor=fn&&Object.getOwnPropertyDescriptor(fn,key);
        fn=descriptor?.value; // Never invoke a getter or the handler.
      }
      if(typeof fn!=='function') continue;
      const source=Function.prototype.toString.call(fn);
      const signature=source.match(/^(?:async\s+)?(?:function(?:\s+[\w$]+)?|[\w$]+)?\s*\(([^)]*)\)/);
      const parameters=signature?.[1].split(',').map(p=>p.trim());
      if(!parameters||parameters.some(p=>!/^[$A-Za-z_][\w$]*$/.test(p))) continue;
      const args=[];
      let rest=call[2], ok=true;
      while(rest.trim()) {
        const token=rest.match(/^\s*(?:'([^'\\]*)'|"([^"\\]*)"|(\d+)|(?:this|[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*))\s*(,|$)/);
        if(!token) {ok=false;break;}
        args.push(token[1]??token[2]??token[3]??null);
        rest=rest.slice(token[0].length);
      }
      if(ok) parameters.forEach((p,i)=>{if(idKeys.includes(p.toLowerCase())&&valid(args[i]))ids.push(args[i])});
    }
    return ids;
  };
  const output = [];
  for (const e of els) {
    const row=e.closest('li,tr,article,[data-workid],[data-work-id],[workid],[data-homework-id],[homeworkid],[data-task-id],[taskid]')||e;
    if (!visible(row)) continue;
    const raw=(GENERIC)([e])[0];
    const heading=row.querySelector('[data-work-title],.list_li_tit,.workName,.work-name,.work-title,.workTitle,[data-title],h1,h2,h3,h4');
    const titleLink=heading?.querySelector('a[title]');
    raw.title=clean(titleLink?.getAttribute('title')||heading?.getAttribute('data-work-title')||heading?.getAttribute('title')||heading?.getAttribute('data-title')||heading?.innerText||raw.title).slice(0,255);
    if (!raw.title || /^(查看|批阅|作业|更多|下一页|上一页|首页|末页|加载更多)$/.test(raw.title)) continue;
    const nodes=[...new Set([row,e,...row.querySelectorAll('a,button,[onclick],[data-url],input[type=hidden]')])];
    const ids=new Set();
    const add=v=>{if(valid(v))ids.add(v)};
    let href='';
    for(const node of nodes) {
      for(const name of names) add(node.getAttribute(name));
      if(node.matches('input[type=hidden]') && idKeys.includes((node.getAttribute('name')||'').toLowerCase())) add(node.getAttribute('value'));
      const script=node.getAttribute('onclick')||'';
      // Only literal URLs / explicitly named literal IDs are read. Never eval
      // a handler, infer an argument position, or use a DOM row number as an ID.
      const urls=[node.getAttribute('href'),node.getAttribute('data-url'),
        ...[...script.matchAll(/["']((?:https?:\/\/|\/|\.\.?\/)[^"']*)["']/g)].map(m=>m[1])];
      for(const value of urls.filter(Boolean)) {
        if(/^(javascript:|#)/i.test(value)) continue;
        try {
          const url=new URL(value,document.baseURI);
          if(!['https:','http:'].includes(url.protocol)||!trusted(url)) continue;
          let work=false;
          for(const [k,v] of url.searchParams) if(idKeys.includes(k.toLowerCase())&&valid(v)) {add(v);work=true;}
          if(work&&!href) href=url.href;
        } catch {}
      }
      for(const match of script.matchAll(/\b(?:workId|homeworkId|taskId)\s*[:=]\s*(?:['"]([A-Za-z0-9_-]{1,128})['"]|(\d+)(?=[\s,;)}&]|$))/gi)) add(match[1]||match[2]);
      for(const value of namedArguments(script)) add(value);
    }
    if(!ids.size) {
      const prefix=(row.id||'').match(/^(?:work|homework|workid|workli)[_-]([A-Za-z0-9_-]{1,128})$/i);
      if(prefix) add(prefix[1]);
      // data-id is meaningful only on a positively identified legacy work row.
      if(row.matches('li')&&row.querySelector('.list_li_tit')&&row.querySelector('a.piyueBtn')) add(row.getAttribute('data-id'));
    }
    raw.work_id=ids.size===1?[...ids][0]:'';
    raw.identity_conflict=ids.size>1;
    raw.href=href||raw.href;
    raw.state=clean(row.getAttribute('data-state')||row.querySelector('.status,.state')?.textContent||raw.state).slice(0,64);
    raw.deadline=clean(row.getAttribute('data-deadline')||row.querySelector('.deadline,.time')?.textContent||raw.deadline).slice(0,128);
    output.push(raw);
  }
  return output;
}""".replace("GENERIC", generic_extract)


HOMEWORK_DIAGNOSTICS = r"""els => ({
  candidate_count:els.length,
  samples:els.slice(0,5).map(e=>({
    tag:e.tagName.toLowerCase(),
    classes:[...e.classList].filter(c=>/^[\w-]{1,64}$/.test(c)).slice(0,6),
    attributes:e.getAttributeNames().filter(n=>/^(data-|id$|class$|onclick$|href$|workid$|homeworkid$)/i.test(n)),
    handler_functions:[...(e.getAttribute('onclick')||'').matchAll(/([\w$]+)\s*\(/g)].map(m=>m[1]).slice(0,4),
    has_title:!!e.querySelector('.list_li_tit,.workName,.work-name,h2'),
    has_review:!!e.querySelector('a.piyueBtn')
  }))
})"""
