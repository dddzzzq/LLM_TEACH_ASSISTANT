"""Read the selected published question, retaining bounded original resources.

Navigation still belongs to the existing page Agent. No HTML, cookies or signed
resource URLs leave this reader; parsing is delegated to the existing backend.
"""

import asyncio
import base64
import hashlib
import re
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from .errors import RpaIssue
from .portal_catalog import query_id

MAX_RESOURCE = 20 * 1024 * 1024
MAX_TOTAL = 60 * 1024 * 1024
MAX_SOURCES = 40
QUESTION_ROOTS = (
    "[data-question-content],.mark_item,.question-item,.questionLi,.TiMu,"
    ".questionContent,.work-question,.subject-item"
)

EXTRACT = r"""roots => {
  const output=[], excluded='script,style,noscript,button,input,select,textarea,nav,header,footer,'+
    '[hidden],[aria-hidden=true],.mark_answer,.mark_key,.mark_score,.answer,.analysis,'+
    '[data-answer],[data-student-answer],.student-answer,.mark_btn';
  const visible=e=>!!e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden'&&
    !e.closest(excluded);
  const clean=s=>String(s||'').replace(/\u00a0/g,' ').trim();
  let buffer='';
  const flush=()=>{const text=clean(buffer).replace(/\n[ \t]+/g,'\n').replace(/\n{3,}/g,'\n\n');
    if(text)output.push({kind:'text',text});buffer='';};
  const walk=node=>{
    if(node.nodeType===Node.TEXT_NODE){buffer+=node.textContent;return;}
    if(node.nodeType!==Node.ELEMENT_NODE||!visible(node))return;
    if(node.matches('img,svg,canvas')){
      flush(); const index=output.length;
      node.setAttribute('data-rpa-question-resource',String(index));
      output.push({kind:'image',name:clean(node.getAttribute('alt'))||'题目图片',
        url:node.currentSrc||node.getAttribute('src')||'',resource_index:index});return;
    }
    if(node.matches('a[href],object[data],iframe[src],embed[src]')){
      const url=node.href||node.data||node.src||'';
      const name=clean(node.innerText||node.getAttribute('title')||node.getAttribute('download'));
      if(node.matches('object,iframe,embed')||/\.(pdf|docx?|txt|md)(?:[?#]|$)/i.test(url)||/\.(pdf|docx?|txt|md)$/i.test(name)||
        /附件|下载/.test(name)||node.hasAttribute('download')){
        flush();const index=output.length;
        node.setAttribute('data-rpa-question-resource',String(index));
        output.push({kind:'attachment',name:name||'题目附件',url,resource_index:index});return;
      }
    }
    const block=/^(P|DIV|LI|TR|H[1-6]|SECTION|ARTICLE|PRE)$/.test(node.tagName);
    if(block&&buffer&&!buffer.endsWith('\n'))buffer+='\n';
    if(node.tagName==='BR')buffer+='\n';
    for(const child of node.childNodes)walk(child);
    if(node.tagName==='TD'||node.tagName==='TH')buffer+='\t';
    if(block)buffer+='\n';
  };
  const outer=roots.filter(e=>visible(e)&&!roots.some(p=>p!==e&&p.contains(e)));
  for(const root of outer){walk(root);flush();}
  return {sources:output.slice(0,41),truncated:output.length>40||output.some(s=>(s.text||'').length>100000)};
}"""


def resource_url(value):
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    return (
        parsed.scheme == "https"
        and not parsed.username
        and not parsed.password
        and parsed.port in (None, 443)
        and (
            host == "learning.xidian.edu.cn"
            or host == "chaoxing.com"
            or host.endswith((".chaoxing.com", ".chaoxing.com.cn"))
        )
    )


class HomeworkDetailReader:
    def __init__(self, task):
        self.task = task

    async def frame(self):
        task = self.task
        selected = task.state["target"].get("portal") or {}
        if not selected.get("homework_id") or not task.adapter.course_confirmed:
            return None
        page = task.page
        if not page or page.is_closed():
            return None
        for frame in page.frames:
            if not resource_url(frame.url):
                continue
            course = query_id(frame.url, ("courseid",))
            if course and course != selected["course_id"].split(":", 1)[0]:
                continue
            work = query_id(frame.url, ("workid", "homeworkid", "taskid"))
            hidden = await frame.locator(
                'input[type=hidden][name="workid" i],input[type=hidden][id="workid" i],'
                "[data-homework-detail][data-workid]"
            ).evaluate_all(
                "els=>els.map(e=>e.value||e.getAttribute('data-workid')).filter(Boolean)"
            )
            ids = set(hidden + ([work] if work else []))
            if ids and ids != {selected["homework_id"]}:
                continue
            titles = await frame.locator(
                "h1,h2,h3,.mark_title,.work-title,.workTitle"
            ).evaluate_all(
                "els=>els.filter(e=>e.getClientRects().length&&!e.closest('li,tr,[role=list]'))"
                ".map(e=>(e.innerText||'').trim())"
            )
            title = task.state["target"]["assignment_name"]
            if title not in titles:
                continue
            # A linked title navigation may omit the work ID from its destination.
            linked = getattr(task, "detail_page", None) is page
            if ids == {selected["homework_id"]} or linked:
                return frame
        return None

    async def observation(self):
        return {"ready": bool(await self.frame()), "read_only": True}

    async def fetch(self, url):
        # Validate each redirect before the shared authenticated request context
        # follows it. Never send this context to an arbitrary attachment host.
        for _ in range(4):
            if not resource_url(url):
                raise ValueError("附件地址不属于已支持的教学平台域名")
            response = await self.task.context.request.get(url, timeout=15000, max_redirects=0)
            try:
                if response.status in (301, 302, 303, 307, 308):
                    url = urljoin(url, response.headers.get("location", ""))
                    continue
                if response.status != 200:
                    raise ValueError("题目资源暂时无法下载")
                if response.headers.get("content-type", "").split(";", 1)[0].lower() in {
                    "text/html",
                    "application/xhtml+xml",
                }:
                    raise ValueError("资源返回了网页，尚未取得题目原文件")
                if int(response.headers.get("content-length", "0")) > MAX_RESOURCE:
                    raise ValueError("单个题目资源超过20MiB")
                data = await response.body()
                if len(data) > MAX_RESOURCE:
                    raise ValueError("单个题目资源超过20MiB")
                return data, response.headers.get("content-type", "").split(";", 1)[0]
            finally:
                await response.dispose()
        raise ValueError("题目资源跳转次数过多")

    async def read(self):
        task = self.task
        frame = await self.frame()
        if not frame:
            raise RpaIssue("DETAIL_TARGET_UNVERIFIED", "请打开所选作业名称对应的作业详情页")
        roots = frame.locator(QUESTION_ROOTS)
        if not await roots.count():
            # A semantic main region is an explicit fallback, never the whole
            # page body (which can include student answers and unrelated menus).
            roots = frame.locator('[data-homework-detail],main,[role="main"]')
        if not await roots.count():
            raise RpaIssue("DETAIL_CONTENT_MISSING", "已进入作业详情，但尚未定位到题目正文区域")
        raw = await roots.evaluate_all(EXTRACT)
        labels = {
            task.state["target"]["assignment_name"],
            "作业详情",
            "显示答案",
            "答案",
            "加载中",
            "加载中...",
            "正在加载...",
        }
        cleaned = []
        for source in raw["sources"]:
            if source["kind"] == "text":
                source["text"] = "\n".join(
                    line for line in source["text"].splitlines() if line.strip() not in labels
                ).strip()
                if not source["text"]:
                    continue
            cleaned.append(source)
        raw["sources"] = cleaned
        warnings = ["题目内容超过读取上限，未完整提取"] if raw["truncated"] else []
        sources, total = [], 0
        folder = Path(task.manager.download_root) / task.state["job_id"] / "requirements"
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        for index, source in enumerate(raw["sources"][:MAX_SOURCES]):
            if not task.automatic():
                raise RpaIssue("DETAIL_STOPPED", "作业详情读取已暂停或取消")
            if source["kind"] == "text":
                sources.append({"kind": "text", "text": source["text"][:100000]})
                continue
            entry = {"kind": source["kind"], "name": source["name"][:255]}
            try:
                url = urljoin(frame.url, source["url"])
                if source["kind"] == "attachment" and (
                    not source["url"] or source["url"].startswith(("javascript:", "#", "blob:"))
                ):
                    # Some question attachments are real site-owned download
                    # controls rather than direct file URLs. Bind only this
                    # control inside the verified question region; never eval
                    # its handler or pass model-supplied JS/URLs.
                    control = frame.locator(
                        f'[data-rpa-question-resource="{source["resource_index"]}"]'
                    )
                    async with asyncio.timeout(20):
                        async with task.page.expect_download(timeout=10000) as pending:
                            await control.click(timeout=5000)
                        download = await pending.value
                        path = await download.path()
                        if not path or Path(path).stat().st_size > MAX_RESOURCE:
                            raise ValueError("题目附件无法下载或体积过大")
                        data, mime = Path(path).read_bytes(), "application/octet-stream"
                        entry["name"] = Path(download.suggested_filename).name[:255]
                elif source["url"].startswith("data:image/"):
                    header, encoded = source["url"].split(",", 1)
                    if len(encoded) > MAX_RESOURCE * 2 or ";base64" not in header:
                        raise ValueError("图片编码不受支持或体积过大")
                    data, mime = base64.b64decode(encoded, validate=True), header[5:].split(";")[0]
                elif source["url"] and resource_url(url):
                    try:
                        data, mime = await self.fetch(url)
                    except Exception:
                        if source["kind"] != "image":
                            raise
                        image = frame.locator(
                            f'[data-rpa-question-resource="{source["resource_index"]}"]'
                        )
                        data, mime = await image.screenshot(timeout=8000), "image/png"
                elif source["kind"] == "image":
                    image = frame.locator(
                        f'[data-rpa-question-resource="{source["resource_index"]}"]'
                    )
                    data, mime = await image.screenshot(timeout=8000), "image/png"
                else:
                    raise ValueError("附件地址不属于已支持的教学平台域名")
                total += len(data)
                if not data or len(data) > MAX_RESOURCE or total > MAX_TOTAL:
                    raise ValueError("题目资源为空或超过读取体积上限")
                extensions = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                    "image/gif": ".gif",
                    "application/pdf": ".pdf",
                    "application/msword": ".doc",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                    "text/plain": ".txt",
                    "text/markdown": ".md",
                }
                ext = extensions.get(mime) or Path(unquote(urlsplit(url).path)).suffix.lower()
                if ext not in {
                    ".pdf",
                    ".doc",
                    ".docx",
                    ".txt",
                    ".md",
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                    ".gif",
                }:
                    named = re.search(r"\.(pdf|docx?|txt|md)(?:\b|$)", entry["name"], re.I)
                    if named:
                        ext = "." + named[1].lower()
                if ext not in {
                    ".pdf",
                    ".doc",
                    ".docx",
                    ".txt",
                    ".md",
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                    ".gif",
                }:
                    raise ValueError("题目附件格式暂不支持")
                digest = hashlib.sha256(data).hexdigest()
                path = folder / f"{index:02d}-{digest[:16]}{ext}"
                path.write_bytes(data)
                path.chmod(0o600)
                entry.update(path=str(path), sha256=digest, size=len(data))
                if entry["name"] in {"题目附件", "题目图片"}:
                    entry["name"] += ext
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Errors must not include signed URLs or request headers.
                entry["error"] = "题目资源读取失败，请核对原始图片或附件"
                warnings.append(entry["name"] + "：" + entry["error"])
                task.event("部分作业要求资源未能读取（" + type(exc).__name__ + "）")
            sources.append(entry)
        if not sources:
            raise RpaIssue("DETAIL_EMPTY", "作业详情正文尚未加载或没有可读取内容")
        if not task.automatic():
            raise RpaIssue("DETAIL_STOPPED", "作业详情读取已暂停或取消")
        selected = task.state["target"]["portal"]
        task.update(
            homework_detail={
                "course_id": selected["course_id"],
                "homework_id": selected["homework_id"],
                "title": task.state["target"]["assignment_name"],
                "sources": sources,
                "warnings": warnings,
                "complete": not warnings,
            },
            status="SUCCEEDED",
            stage="DONE",
            message="作业详情已读取，正在解析文字、图片和附件",
        )
