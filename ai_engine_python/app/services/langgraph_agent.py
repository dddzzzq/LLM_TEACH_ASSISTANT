import os
import json
import time
from typing import Annotated, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from playwright.sync_api import sync_playwright

def load_api_key_from_env():
    """从.env文件直接读取DEEPSEEK_API_KEY"""
    env_path = "/root/autodl-tmp/dzq/LLM_TEACH_ASSISTANT/ai_engine_python/app/.env"
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY="):
                    # 去除引号
                    key = line.split('=', 1)[1].strip()
                    if key.startswith('"') and key.endswith('"'):
                        key = key[1:-1]
                    elif key.startswith("'") and key.endswith("'"):
                        key = key[1:-1]
                    return key
    except Exception as e:
        print(f"[Warning] 无法从 {env_path} 读取API key: {e}")
    return None

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    download_path: str
    assignment_id: str
    action_type: str

@tool
def download_homework_from_web(url: str, assignment_id: str) -> str:
    """从给定的URL自动下载学生的作业压缩包，并准备提交批改。"""
    print(f"[Agent Tool] 启动浏览器前往: {url}")
    download_dir = "/tmp/grading_downloads"
    os.makedirs(download_dir, exist_ok=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, timeout=30000)
            # RPA 占位符：模拟找到并点击下载 zip 的操作
            with page.expect_download(timeout=10000) as download_info:
                # 尝试点击带有 zip 或 download 字样的链接，实际需根据具体网站修改
                page.locator("a:has-text('zip'), a:has-text('Download')").first.click()
            
            download = download_info.value
            save_path = os.path.join(download_dir, f"auto_hw_{assignment_id}_{download.suggested_filename}")
            download.save_as(save_path)
            return f"文件已成功下载至 {save_path}"
        except Exception as e:
            # 如果没找到按钮，创建一个模拟的测试 zip 包以保证流程不中断（用于演示）
            fake_path = os.path.join(download_dir, f"mock_hw_{assignment_id}.zip")
            with open(fake_path, 'wb') as f:
                f.write(b'PK\x05\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
            return f"由于页面结构未知抓取失败，已生成模拟测试文件至 {fake_path}"
        finally:
            browser.close()

@tool
def process_local_file(file_path: str, assignment_id: str) -> str:
    """当用户明确提供了一个本地文件路径(如 /root/.../xxx.zip)要求批改时，调用此工具验证文件。"""
    import os
    print(f"[Agent Tool] 正在验证本地文件: {file_path}")
    if os.path.exists(file_path):
        return f"文件已成功定位至 {file_path}"
    else:
        return f"错误：在本地找不到文件 {file_path}，请检查路径是否正确。"

tools = [download_homework_from_web, process_local_file]

# 使用 DeepSeek API
api_key_from_env = load_api_key_from_env()
if api_key_from_env:
    print(f"[Agent] 从.env文件读取API Key: {api_key_from_env[:10]}...")
    api_key = api_key_from_env
else:
    print("[Agent] 无法从.env读取API Key，使用默认值")

llm = ChatOpenAI(
    model="deepseek-chat", 
    api_key=api_key, 
    base_url="https://api.deepseek.com/v1",
    temperature=0.1
).bind_tools(tools)

def call_model(state: AgentState):
    messages = state.get("messages", [])
    sys_msg = HumanMessage(content="你是一个作业批改系统的智能调度Agent。当用户要求你从网上(提供URL)下载作业时，请提取URL和作业ID(默认为1)，并务必调用 download_homework_from_web 工具。如果用户提供了本地文件路径(如 /root/.../xxx.zip)，请调用 process_local_file 工具验证文件。完成下载或验证后，告诉用户你会立即触发后台批改流水线。")
    
    max_retries = 3
    retry_delay = 2  # 秒
    
    for attempt in range(max_retries):
        try:
            response = llm.invoke([sys_msg] + messages)
            
            action_type = "none"
            assign_id = state.get("assignment_id", "")
            
            if response.tool_calls:
                for tc in response.tool_calls:
                    if tc["name"] == "download_homework_from_web":
                        assign_id = str(tc["args"].get("assignment_id", "1"))
                    elif tc["name"] == "process_local_file":
                        assign_id = str(tc["args"].get("assignment_id", "1"))
            else:
                if "触发" in response.content or "批改" in response.content or "下载" in response.content or "定位" in response.content:
                    action_type = "trigger_pipeline"

            return {"messages": [response], "assignment_id": assign_id, "action_type": action_type}
            
        except Exception as e:
            print(f"[Agent] LLM调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # 递增延迟
            else:
                # 所有重试都失败，返回后备响应
                print("[Agent] LLM调用完全失败，使用后备响应")
                fallback_response = HumanMessage(content="抱歉，AI服务暂时无法访问。我会尝试帮你处理请求。如果你提供了文件路径，我会尝试处理本地文件；如果是网络下载请求，请检查网络连接。")
                return {"messages": [fallback_response], "assignment_id": "1", "action_type": "none"}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"

def execute_tools_and_update(state: AgentState):
    tool_node = ToolNode(tools)
    result = tool_node.invoke(state)
    path = ""
    last_msg = result["messages"][-1]
    
    if "文件已成功下载至 " in last_msg.content:
        path = last_msg.content.split("文件已成功下载至 ")[1].strip()
    elif "已生成模拟测试文件至 " in last_msg.content:
        path = last_msg.content.split("已生成模拟测试文件至 ")[1].strip()
    elif "文件已成功定位至 " in last_msg.content:
        path = last_msg.content.split("文件已成功定位至 ")[1].strip()
        
    return {"messages": result["messages"], "download_path": path}

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools_and_update)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app_agent = workflow.compile()

def run_agent_chat(user_message: str):
    final_state = app_agent.invoke({"messages": [("user", user_message)]})
    return {
        "reply_text": final_state["messages"][-1].content,
        "action_type": final_state.get("action_type", "none"),
        "downloaded_file_path": final_state.get("download_path", ""),
        "target_assignment_id": final_state.get("assignment_id", "")
    }