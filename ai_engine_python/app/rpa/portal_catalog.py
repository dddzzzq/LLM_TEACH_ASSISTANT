"""Read-only DOM catalog adapter. Stable platform identifiers are mandatory."""

import os
import re
import hashlib
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit

from .errors import RpaIssue
from .homework_dom import HOMEWORK_SELECTOR, HOMEWORK_ROWS, HOMEWORK_DIAGNOSTICS, homework_extract
from .page_snapshot import public_url
from .catalog_viewport import CatalogViewport


ROUTING_KEYS = {
    "courseid",
    "clazzid",
    "classid",
    "workid",
    "homeworkid",
    "taskid",
    "cpi",
    "ut",
    "mooc2",
    "enc",
    "t",
    "type",
    "page",
    "pageindex",
    "pagesize",
    "roleid",
}
COURSE_SELECTOR = (
    "a[href*='courseid=' i],[data-courseid],[data-course-id],[courseid],[cname],.myde_course_item"
)
WORK_SELECTOR = HOMEWORK_SELECTOR
CLASS_SELECTOR = (
    "[data-clazzid],[data-clazz-id],[data-classid],[data-class-id],[clazzid],[classid],"
    "select[name*='clazz' i] option,select[name*='class' i] option,"
    "select[id*='clazz' i] option,select[id*='class' i] option,"
    "li[onclick*='changeClass' i],li[onclick*='changeClazz' i],li[onclick*='selectClass' i]"
)
EXTRACT = r"""els => els.map(e => {
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
})"""


def portal_url(value):
    try:
        p = urlsplit(value)
        host = p.hostname or ""
        if p.scheme != "https" or p.username or p.password or p.port or p.fragment:
            return False
        if (
            host != "learning.xidian.edu.cn"
            and host != "chaoxing.com"
            and not host.endswith((".chaoxing.com", ".chaoxing.com.cn"))
        ):
            return False
        return not re.search(r"login|logout|delete|publish", p.path, re.I)
    except ValueError:
        return False


def clean_url(value):
    if not portal_url(value):
        return ""
    p = urlsplit(value)
    query = parse_qs(p.query, keep_blank_values=True)
    if any(key.lower() not in ROUTING_KEYS for key in query):
        return ""
    return urlunsplit((p.scheme, p.netloc, p.path, urlencode(query, doseq=True), ""))


def query_id(value, names):
    query = {
        key.lower(): values[0] for key, values in parse_qs(urlsplit(value).query).items() if values
    }
    return next((query[name] for name in names if query.get(name)), "")


def normalize_record(raw, kind, base, *, allow_unresolved=False):
    destination = urljoin(base, raw.get("href", "")) if raw.get("href") else ""
    link = clean_url(destination)
    names = ("courseid",) if kind == "course" else ("workid", "homeworkid", "taskid")
    platform_id = raw.get("course_id" if kind == "course" else "work_id") or (
        query_id(destination, names) if portal_surface(destination) else ""
    )
    if not raw.get("title") or (not allow_unresolved and (not link or not platform_id)):
        return None
    class_id = raw.get("class_id") or (
        query_id(destination, ("clazzid", "classid")) if portal_surface(destination) else ""
    )
    if kind == "course" and class_id and platform_id:
        platform_id += ":" + class_id
    return {
        "platform_id": platform_id,
        "title": raw["title"],
        "url": link,
        "term": raw.get("term", ""),
        "state": raw.get("state", ""),
        "deadline": raw.get("deadline", ""),
        "class_id": class_id,
        "class_name": raw.get("class_name", ""),
    }


def source_key(raw):
    # A local collection reference, never a published platform ID. Hash the
    # source URL so signed parameters cannot enter the journal or model context.
    identity = [
        raw.get(k, "") for k in ("source_route", "source_path", "source_page", "title", "term")
    ]
    return "pending:" + hashlib.sha256("\x00".join(identity).encode()).hexdigest()[:24]


def portal_surface(value):
    return portal_url(value.replace("http://", "https://", 1))


class PortalCatalog:
    def __init__(self, context, update):
        self.context = context
        self.update = update

    async def surfaces(self):
        for page in list(self.context.pages):
            if page.is_closed() or not portal_surface(page.url):
                continue
            for frame in page.frames:
                if portal_surface(frame.url) or frame.url in {"about:blank", "about:srcdoc"}:
                    yield frame

    async def records(self, kind, *, allow_unresolved=False):
        selector = os.environ.get(
            "PORTAL_COURSE_SELECTOR" if kind == "course" else "PORTAL_HOMEWORK_SELECTOR",
            COURSE_SELECTOR if kind == "course" else WORK_SELECTOR,
        )
        found = {}
        async for frame in self.surfaces():
            extract = homework_extract(EXTRACT) if kind == "homework" else EXTRACT
            for raw in await frame.locator(selector).evaluate_all(extract):
                if raw.get("identity_conflict"):
                    raise RpaIssue(
                        "CATALOG_AMBIGUOUS",
                        "同一作业行出现多个官网作业标识，目录未发布",
                        await self.diagnostics(kind),
                    )
                row = normalize_record(raw, kind, frame.url, allow_unresolved=allow_unresolved)
                if row:
                    row["source_key"] = source_key(raw)
                    key = row["platform_id"] or row["source_key"]
                    prior = found.get(key)
                    if prior and prior["title"] != row["title"]:
                        raise RpaIssue("CATALOG_AMBIGUOUS", "同一官网标识对应多个名称，目录未发布")
                    found[key] = row
        return list(found.values())

    async def diagnostics(self, kind):
        frames = []
        selector = os.environ.get(
            "PORTAL_COURSE_SELECTOR" if kind == "course" else "PORTAL_HOMEWORK_SELECTOR",
            COURSE_SELECTOR if kind == "course" else WORK_SELECTOR,
        )
        for page in self.context.pages:
            if page.is_closed():
                continue
            for frame in page.frames[:20]:
                supported = portal_surface(page.url) and (
                    portal_surface(frame.url) or frame.url in {"about:blank", "about:srcdoc"}
                )
                details = {"location": public_url(frame.url), "supported": supported}
                if supported:
                    details.update(await frame.locator(selector).evaluate_all(HOMEWORK_DIAGNOSTICS))
                    details["legacy_work_rows"] = await frame.locator(HOMEWORK_ROWS).count()
                frames.append(details)
        return {"kind": kind, "frames": frames[:20]}

    async def classes(self, page, course):
        classes = {}
        async for frame in PortalCatalog(_SinglePageContext(page), self.update).surfaces():
            for value in await frame.locator(
                os.environ.get("PORTAL_CLASS_SELECTOR", CLASS_SELECTOR)
            ).evaluate_all(EXTRACT):
                if (
                    value["class_id"] not in ("", "0", "-1", "all")
                    and value["title"]
                    and not re.fullmatch(r"(?:全部|所有|请选择)(?:班级)?", value["title"])
                ):
                    classes[value["class_id"]] = {
                        "platform_id": value["class_id"],
                        "name": value["class_name"] or value["title"],
                    }
        if not classes and course.get("class_id") and course.get("class_name"):
            classes[course["class_id"]] = {
                "platform_id": course["class_id"],
                "name": course["class_name"],
            }
        return list(classes.values())

    async def empty(self, kind):
        pattern = re.compile(
            "暂无课程|没有课程|暂无相关课程"
            if kind == "course"
            else "暂无作业|没有作业|暂无相关作业|还没有作业|未布置作业|暂未(?:布置|发布)作业"
        )
        async for frame in self.surfaces():
            match = frame.get_by_text(pattern)
            if await match.count() and await match.first.is_visible():
                return True
        return False


class _SinglePageContext:
    def __init__(self, page):
        self.pages = [page]


class CatalogCollection:
    """Validate and accumulate the page Agent's reads; never click or navigate."""

    def __init__(self, task):
        self.task = task
        self.progress = task.state.setdefault(
            "catalog_progress", {"courses": {}, "courses_complete": False}
        )
        # Browser handles and the original URL stay in memory, outside the
        # journal/model context. Only a successful list read can set a bookmark.
        self.list_page = None
        self.list_url = ""
        self.list_sources = set()

    def observation(self):
        courses = self.progress["courses"]
        current = self.task.catalog_course_pages.get(self.task.page)
        completed = sum(bool(c.get("complete")) for c in courses.values())
        return {
            "courses_complete": self.progress["courses_complete"],
            "current_course_id": current,
            "return_required": bool(current and courses[current].get("complete")),
            "next_course_id": next((k for k, c in courses.items() if not c.get("complete")), None),
            "total_courses": len(courses),
            "completed_courses": completed,
            "courses": [
                {
                    "course_id": key,
                    "title": value["title"],
                    "term": value["term"],
                    "homework_count": len(value["homeworks"]),
                    "complete": value.get("complete", False),
                    "empty": value.get("empty", False),
                    "identity_pending": not bool(value["platform_id"]),
                }
                for key, value in self.progress["courses"].items()
            ],
        }

    def viewport(self, page=None, kind=None):
        page = page or self.task.page
        kind = kind or ("homework" if self.task.catalog_course_pages.get(page) else "course")
        selector = os.environ.get(
            "PORTAL_COURSE_SELECTOR" if kind == "course" else "PORTAL_HOMEWORK_SELECTOR",
            COURSE_SELECTOR if kind == "course" else WORK_SELECTOR,
        )
        reader = PortalCatalog(_SinglePageContext(page), self.task.update)
        return CatalogViewport(reader, selector, kind)

    async def at_end(self, page, kind):
        return (await self.viewport(page, kind).observation())["at_end"]

    async def read_courses(self):
        page = self.task.page
        reader = PortalCatalog(_SinglePageContext(page), self.task.update)
        rows = await reader.records("course", allow_unresolved=True)
        empty = not rows and await reader.empty("course")
        if not rows and not empty:
            raise RpaIssue(
                "CATALOG_CARDS_UNAVAILABLE",
                "未能读取当前页的课程卡片，请核对页面加载和课程卡片适配规则",
            )
        for row in rows:
            key = row["platform_id"] or row["source_key"]
            # Keep the same local reference when a previously opened card has
            # acquired its real platform ID. Pending references are never published.
            prior = next(
                (
                    c
                    for c in self.progress["courses"].values()
                    if c.get("source_key") == row["source_key"]
                ),
                None,
            )
            if prior and not prior["platform_id"] and row["platform_id"]:
                prior.update({k: row[k] for k in ("platform_id", "url", "class_id", "class_name")})
                continue
            if prior and (not row["platform_id"] or prior["platform_id"] == row["platform_id"]):
                continue
            prior = self.progress["courses"].get(key)
            if prior and (prior["title"], prior["term"]) != (row["title"], row["term"]):
                raise RpaIssue("CATALOG_AMBIGUOUS", "同一课程标识对应不同记录，请核查当前页面")
            self.progress["courses"][key] = prior or {
                **row,
                "classes": [],
                "homeworks": {},
                "complete": False,
            }
        if len(self.progress["courses"]) > 500:
            raise RpaIssue("CATALOG_LIMIT", "课程数超过本次读取上限")
        # Once the last page was read, returning to page one must not erase
        # coverage of the other pages already collected in this task.
        self.progress["courses_complete"] = (
            self.progress["courses_complete"] or empty or await self.at_end(page, "course")
        )
        self.list_page, self.list_url = page, page.url
        self.list_sources = {r["source_key"] for r in rows}
        self.task.catalog_course_pages.pop(page, None)
        self.task.update(catalog_progress=self.progress)

    async def is_course_list(self, page):
        if page.is_closed() or page.url != self.list_url:
            return False
        reader = PortalCatalog(_SinglePageContext(page), self.task.update)
        rows = await reader.records("course", allow_unresolved=True)
        if not self.list_sources:
            return not rows and await reader.empty("course")
        return bool(self.list_sources & {r["source_key"] for r in rows})

    def resolve_identity(self, reference, page):
        course = self.progress["courses"][reference]
        if course["platform_id"]:
            return True
        if self.task.catalog_course_pages.get(page) != reference:
            return False
        identities = {
            (query_id(frame.url, ("courseid",)), query_id(frame.url, ("clazzid", "classid")))
            for frame in page.frames
            if portal_surface(frame.url) and query_id(frame.url, ("courseid",))
        }
        course_ids = {value[0] for value in identities}
        class_ids = {value[1] for value in identities if value[1]}
        if len(course_ids) != 1 or len(class_ids) > 1:
            return False
        actual = next(iter(course_ids))
        clazz = next(iter(class_ids), "")
        course["platform_id"] = actual + (":" + clazz if clazz else "")
        course["class_id"] = clazz
        course["url"] = next(
            (
                clean_url(frame.url)
                for frame in page.frames
                if query_id(frame.url, ("courseid",)) == actual and clean_url(frame.url)
            ),
            "",
        )
        return True

    async def read_homeworks(self, course_id):
        course = self.progress["courses"].get(course_id)
        if not course:
            raise ValueError("必须先读取官网课程记录")
        page = self.task.page
        if not course["platform_id"]:
            if self.task.catalog_course_pages.get(page) != course_id:
                raise RpaIssue(
                    "CATALOG_COURSE_UNVERIFIED", "请先通过该课程卡片进入课程页面，再核验官网标识"
                )
            if not self.resolve_identity(course_id, page):
                raise RpaIssue(
                    "CATALOG_COURSE_ID_PENDING",
                    "已进入课程，但当前页面尚未暴露唯一官网课程标识，请继续打开该课程的作业页",
                )
        expected = course["platform_id"].split(":", 1)
        current_ids = {
            query_id(frame.url, ("courseid",))
            for frame in page.frames
            if portal_surface(frame.url) and query_id(frame.url, ("courseid",))
        }
        current_classes = {
            query_id(frame.url, ("clazzid", "classid"))
            for frame in page.frames
            if portal_surface(frame.url) and query_id(frame.url, ("clazzid", "classid"))
        }
        if (current_ids and current_ids != {expected[0]}) or (
            len(expected) > 1 and current_classes and current_classes != {expected[1]}
        ):
            raise RpaIssue(
                "CATALOG_COURSE_UNVERIFIED", "当前页面的官网课程或班级标识与所选课程不一致"
            )
        matched = any(
            query_id(frame.url, ("courseid",)) == expected[0]
            and (len(expected) == 1 or query_id(frame.url, ("clazzid", "classid")) == expected[1])
            for frame in page.frames
            if portal_url(frame.url)
        )
        if not matched and self.task.catalog_course_pages.get(page) != course_id:
            raise RpaIssue("CATALOG_COURSE_UNVERIFIED", "当前页面与要读取的官网课程不一致")
        reader = PortalCatalog(_SinglePageContext(page), self.task.update)
        rows = await reader.records("homework", allow_unresolved=True)
        if any(not row["platform_id"] for row in rows):
            raise RpaIssue(
                "HOMEWORK_IDS_UNAVAILABLE",
                "已识别作业，但缺少官网作业标识，请核对该作业的页面入口",
                await reader.diagnostics("homework"),
            )
        empty = not rows and await reader.empty("homework")
        if not rows and not empty:
            raise RpaIssue(
                "HOMEWORK_IDS_UNAVAILABLE",
                "当前页面未提取到可核验的作业记录，请核对作业列表结构或加载状态",
                await reader.diagnostics("homework"),
            )
        classes = await reader.classes(page, course)
        course["classes"] = classes or course["classes"]
        for row in rows:
            prior = course["homeworks"].get(row["platform_id"])
            if prior and prior["title"] != row["title"]:
                raise RpaIssue("CATALOG_AMBIGUOUS", "同一作业标识对应不同名称，请核查当前页面")
            course["homeworks"][row["platform_id"]] = {
                key: row[key] for key in ("platform_id", "title", "url", "state", "deadline")
            }
        course["empty"] = empty and not course["homeworks"]
        course["complete"] = bool(empty) or await self.at_end(page, "homework")
        self.task.state["last_catalog_read"] = {
            "course_id": course_id,
            "rows": len(rows),
            "total_homeworks": len(course["homeworks"]),
            "complete": course["complete"],
            "empty": course["empty"],
        }
        if len(course["homeworks"]) > 2000 or len(course["classes"]) > 500:
            raise RpaIssue("CATALOG_LIMIT", "课程目录超过本次读取上限")
        self.task.update(catalog_progress=self.progress)
        self.task.event(
            f"已读取课程“{course['title']}”：当前页 {len(rows)} 个作业，累计 {len(course['homeworks'])} 个"
            + (
                "；官网明确无作业，可返回课程列表"
                if course["empty"]
                else "；已读完，可返回课程列表"
                if course["complete"]
                else "；继续读取下一页或滚动加载"
            )
        )

    async def record_navigation(self, locator, page):
        raw = await locator.evaluate("e => (" + EXTRACT + ")([e])[0]")
        key = source_key(raw)
        row = normalize_record(raw, "course", page.url, allow_unresolved=True)
        if row and row["platform_id"]:
            for reference, course in self.progress["courses"].items():
                if course["platform_id"] == row["platform_id"]:
                    return reference
        for reference, course in self.progress["courses"].items():
            if course.get("source_key") == key:
                return reference
        current = self.task.catalog_course_pages.get(page)
        if current:
            course = self.progress["courses"][current]
            self.resolve_identity(current, page)
            classes = await PortalCatalog(_SinglePageContext(page), self.task.update).classes(
                page, course
            )
            if classes:
                course["classes"] = classes
                self.task.update(catalog_progress=self.progress)
        return current

    def finish(self):
        courses = self.progress["courses"]
        if not self.progress["courses_complete"] or any(
            not c["complete"] or not c["platform_id"] for c in courses.values()
        ):
            raise ValueError("课程或作业目录尚未完整读取，请继续读取未完成的页面")
        if sum(len(c["homeworks"]) for c in courses.values()) > 10000:
            raise RpaIssue("CATALOG_LIMIT", "作业数超过本次读取上限")
        if len({c["platform_id"] for c in courses.values()}) != len(courses):
            raise RpaIssue("CATALOG_AMBIGUOUS", "多个课程卡片指向同一官网标识，请核对课程列表")
        return {
            "complete": True,
            "courses": [
                {
                    **{k: c[k] for k in ("platform_id", "title", "term", "url", "classes")},
                    "homeworks": list(c["homeworks"].values()),
                }
                for c in courses.values()
            ],
        }
