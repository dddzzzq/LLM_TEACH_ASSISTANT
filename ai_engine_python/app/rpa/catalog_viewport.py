"""Locate and scroll the live catalog region, including iframe/inner scroll areas."""

import asyncio
import time

from .errors import RpaIssue


# Recomputed for each observation/action; no stored element indexes or CSS writes.
# Pagination and completion use the same business frame and scroll container.
CATALOG_VIEWPORT = r"""(elements,{delta=0}) => {
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
}"""


class CatalogViewport:
    def __init__(self, reader, selector, kind):
        self.reader, self.selector, self.kind = reader, selector, kind

    async def locate(self):
        choices = []
        async for frame in self.reader.surfaces():
            value = await frame.locator(self.selector).evaluate_all(CATALOG_VIEWPORT, {"delta": 0})
            choices.append((value["record_count"], frame, value))
        if not choices:
            raise RpaIssue("CATALOG_VIEWPORT_UNAVAILABLE", "当前没有可读取的课程或作业页面")
        # Ignore unrelated outer navigation/sidebar frames when records live in
        # a child frame. Stable tie order follows the current page's frame order.
        _, frame, value = max(choices, key=lambda choice: choice[0])
        return frame, {"kind": self.kind, **value}

    async def observation(self):
        _, value = await self.locate()
        return value

    async def scroll(self, delta):
        frame, before = await self.locate()
        await frame.locator(self.selector).evaluate_all(CATALOG_VIEWPORT, {"delta": delta})
        # Wait for native scroll/IntersectionObserver and a bounded lazy-load
        # quiet period. This never advances a second page or scrolls again.
        deadline, quiet_since = time.monotonic() + 3, time.monotonic()
        previous = None
        while True:
            await asyncio.sleep(0.1)
            after = await frame.locator(self.selector).evaluate_all(CATALOG_VIEWPORT, {"delta": 0})
            marker = (after["top"], after["height"], after["record_count"], after["loading"])
            if marker != previous or after["loading"]:
                quiet_since = time.monotonic()
            previous = marker
            if time.monotonic() >= deadline or time.monotonic() - quiet_since >= 0.6:
                break
        if (
            delta > 0
            and before["needs_scroll"]
            and after["top"] <= before["top"]
            and after["record_count"] <= before["record_count"]
        ):
            raise RpaIssue(
                "CATALOG_SCROLL_STALLED",
                "目录列表未能向下滚动，请检查当前列表是否加载完成",
                {"viewport": after},
            )
        return {"kind": self.kind, **after}
