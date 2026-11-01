(elements,{delta=0}) => {
  const visible=e=>!!e.getClientRects().length && e.getBoundingClientRect().width>0 && e.getBoundingClientRect().height>0 && getComputedStyle(e).visibility!=='hidden' && !e.closest('[hidden],[aria-hidden=true]');
  const rows=[...new Set(elements.filter(visible).filter(e=>! /^(作业|查看|批阅|更多|下一页|上一页|首页|末页|加载更多)$/.test((e.innerText||'').trim()))
    .map(e=>e.closest('[cname],.myde_course_item,li,tr,article,[data-workid],[data-work-id],[workid],[data-homework-id]')||e))];
  const root=document.scrollingElement;
  const containers=new Map();
  for(const row of rows) {
    for(let e=row.parentElement;e;e=e.parentElement) {
      if(e!==root && (!/(auto|scroll)/.test(getComputedStyle(e).overflowY)||e.clientHeight<=0||e.scrollHeight<=e.clientHeight+4)) continue;
      if(!containers.has(e)) containers.set(e,new Set());
      containers.get(e).add(row);
    }
  }
  const ranked=[...containers].sort((a,b)=>b[1].size-a[1].size||a[0].clientHeight-b[0].clientHeight);
  const target=ranked[0]?.[0]||root;
  const inner=target&&target!==root;
  const next=[...document.querySelectorAll('a[rel=next],.pagination-next,button,a,[role=button],[onclick],.load-more,[data-load-more]')].some(e=>{
    if(!visible(e)||!(e.matches('a[rel=next],.pagination-next')||/^(下一页|加载更多)$/.test((e.innerText||'').trim()))) return false;
    return !e.disabled&&e.getAttribute('aria-disabled')!=='true'&&!e.closest('.disabled,.is-disabled');
  });
  const loading=[...document.querySelectorAll('[aria-busy=true],.loading,.load-more-loading,[data-loading=true]')].some(visible);
  if(delta&&target) (inner?target:window).scrollBy({top:delta,behavior:'instant'});
  // In legacy quirks-mode pages scrollingElement is body and clientHeight
  // can equal its full content height. Window dimensions describe the viewport.
  const top=inner?target.scrollTop:scrollY;
  const height=inner?target.scrollHeight:Math.max(root?.scrollHeight||0,document.documentElement.scrollHeight);
  const viewport=inner?target.clientHeight:innerHeight;
  const remaining=Math.max(0,height-viewport-top);
  return {record_count:rows.length,container:inner?'inner':'document',top:Math.round(top),height,viewport,
    remaining:Math.ceil(remaining),needs_scroll:remaining>4,has_next:next,loading,
    at_end:remaining<=4&&!next&&!loading};
}