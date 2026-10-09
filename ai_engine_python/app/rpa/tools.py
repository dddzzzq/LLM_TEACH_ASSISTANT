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
    'click': BrowserTool('点击当前观察中的控件；引用由执行器核验', {'element_ref': str}),
    'fill': BrowserTool('填写搜索字段；只能操作执行器认可的搜索控件', {'element_ref': str, 'value': str}),
    'scroll': BrowserTool('垂直滚动当前页面；执行器将幅度限制在 -1000 到 1000', {'delta_y': int}),
    'switch_page': BrowserTool('切换到当前观察中的页面 ID', {'page_id': str}),
    'open_export_settings': BrowserTool('仅在 can_open_export 为 true 时打开已核验作业的导出设置', {}),
    'request_human': BrowserTool('暂停自动操作并说明需要用户处理的事项', {'reason': str}),
}


def tool_instructions():
    """Describe exactly the same action contracts the executor validates."""
    schemas = []
    for name, tool in BROWSER_TOOLS.items():
        properties = {'tool': {'type': 'string', 'enum': [name]}}
        properties.update({field: {'type': 'string' if kind is str else 'integer'}
                           for field, kind in tool.fields.items()})
        schemas.append({'description': tool.description, 'type': 'object',
                        'properties': properties, 'required': list(properties),
                        'additionalProperties': False})
    return json.dumps({'oneOf': schemas}, ensure_ascii=False)


def validate_action(action):
    if not isinstance(action, dict) or not isinstance(action.get('tool'), str):
        raise ValueError('浏览器动作必须包含工具名称')
    tool = BROWSER_TOOLS.get(action['tool'])
    if tool is None:
        raise ValueError('模型返回了不支持的浏览器动作')
    if set(action) != {'tool', *tool.fields}:
        raise ValueError('浏览器动作参数与工具协议不一致')
    for field, kind in tool.fields.items():
        if type(action[field]) is not kind:
            raise ValueError('浏览器动作参数类型无效')
        if kind is str and field != 'value' and not action[field].strip():
            raise ValueError('浏览器动作引用或原因不能为空')
    if action['tool'] == 'switch_page' and not action['page_id'].isdecimal():
        raise ValueError('页面 ID 无效')
    return action
