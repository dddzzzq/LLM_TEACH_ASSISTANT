"""Code-owned browser action contracts, shared by prompting and execution.

Skills supply navigation knowledge. They cannot register actions or change these
contracts; Task and PortalAdapter additionally enforce live page and task scope.
"""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class BrowserTool:
    description: str
    fields: dict


BROWSER_TOOLS = {
    "fill_portal_credentials": BrowserTool(
        "在已授权登录页填写已保存账号密码并提交；不接收或返回凭据", {}
    ),
    "request_login_code": BrowserTool(
        "在发送验证码页面获取后交还教师填写；企业微信动态码页直接等待输入，不发送短信", {}
    ),
    "submit_login_code": BrowserTool("提交教师已在当前浏览器输入的验证码；不接收或返回验证码", {}),
    "apply_device_trust": BrowserTool("按教师保存的信任设备设置处理当前官网可信浏览器弹窗", {}),
    "read_course_list": BrowserTool("只读取当前课程列表的真实标识和名称；不跳转、不下载", {}),
    "read_homework_list": BrowserTool("只读取当前已核验课程的班级和作业列表", {"course_id": str}),
    "select_homework_detail": BrowserTool(
        "选择当前已读取作业列表中的官网作业，随后点击该作业名称读取详情；不跳转",
        {"homework_id": str},
    ),
    "read_homework_detail": BrowserTool(
        "只读取当前所选作业详情的题目文字、图片和附件；不读取学生答案、不批改、不查重", {}
    ),
    "finish_catalog": BrowserTool("所有课程与作业页面读取完整后，提交本次目录结果", {}),
    "return_to_course_list": BrowserTool(
        "返回 read_course_list 记录的原课程列表页；无作业也须返回并继续下一门课程", {}
    ),
    "solve_slider_captcha": BrowserTool(
        "仅在 SOLVE_CAPTCHA 阶段处理当前登录页已检测到的滑块；本轮最多三次，失败交还用户", {}
    ),
    "click": BrowserTool("点击当前观察中的控件；引用由执行器核验", {"element_ref": str}),
    "fill": BrowserTool(
        "填写搜索字段；只能操作执行器认可的搜索控件", {"element_ref": str, "value": str}
    ),
    "scroll": BrowserTool("垂直滚动当前页面；执行器将幅度限制在 -1000 到 1000", {"delta_y": int}),
    "switch_page": BrowserTool("切换到当前观察中的页面 ID", {"page_id": str}),
    "open_export_settings": BrowserTool(
        "仅在 can_open_export 为 true 时打开已核验作业的导出设置", {}
    ),
    "request_human": BrowserTool("暂停自动操作并说明需要用户处理的事项", {"reason": str}),
    "request_target_confirmation": BrowserTool(
        "仅在课程/作业列表由大模型判断近似名称后，请教师弹窗确认；不点击候选",
        {"kind": str, "element_ref": str, "actual_name": str, "term": str, "reason": str},
    ),
}


def tool_instructions():
    """Describe exactly the same action contracts the executor validates."""
    schemas = []
    for name, tool in BROWSER_TOOLS.items():
        properties = {"tool": {"type": "string", "enum": [name]}}
        properties.update(
            {
                field: {"type": "string" if kind is str else "integer"}
                for field, kind in tool.fields.items()
            }
        )
        schemas.append(
            {
                "description": tool.description,
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            }
        )
    return json.dumps({"oneOf": schemas}, ensure_ascii=False)


def validate_action(action):
    if not isinstance(action, dict) or not isinstance(action.get("tool"), str):
        raise ValueError("浏览器动作必须包含工具名称")
    tool = BROWSER_TOOLS.get(action["tool"])
    if tool is None:
        raise ValueError("模型返回了不支持的浏览器动作")
    if set(action) != {"tool", *tool.fields}:
        raise ValueError("浏览器动作参数与工具协议不一致")
    for field, kind in tool.fields.items():
        if type(action[field]) is not kind:
            raise ValueError("浏览器动作参数类型无效")
        optional_term = action["tool"] == "request_target_confirmation" and field == "term"
        if kind is str and field != "value" and not optional_term and not action[field].strip():
            raise ValueError("浏览器动作引用或原因不能为空")
    if action["tool"] == "switch_page" and not action["page_id"].isdecimal():
        raise ValueError("页面 ID 无效")
    return action
