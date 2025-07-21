import re
import json
import requests
from ..core.config import settings
from typing import Optional
from ..schemas.models import LLMAnalysis, PlagiarismReport, AIGCReport # 引入 AIGCReport

class DeepSeekService:
    """封装所有与DeepSeek API的交互逻辑。"""
    def __init__(self):
        self.api_key = settings.DEEPSEEK_API_KEY
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _call_api(self, user_prompt: str, system_prompt: str, temperature: float = 0.0) -> str:
        """一个通用的、私有的API调用方法。"""
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
        }
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=180)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"调用DeepSeek API时出错: {e}")
            raise

    def analyze_plagiarism(self, text1: str, student1_id: str, text2: str, student2_id: str) -> Optional[LLMAnalysis]:
        """调用LLM来深度分析两个文本的相似性，即查重检测"""
        text1 = json.dumps(text1[:20000])
        text2 = json.dumps(text2[:20000])
        system_prompt = "你是一位学术诚信审查官AI。你的输出必须是一个单一、有效的JSON对象，不能包含任何其他内容，使用中文"
        user_prompt = f"""
        你是一位经验丰富的学术诚信审查官。你的任务是判断两份学生作业之间是否存在抄袭。

        [作业信息]:
        - 作业A来自学生: {student1_id}
        - 作业B来自学生: {student2_id}
        这两份作业在初步的关键词频率检测中显示出高度相似性。你需要进行深度语义分析，尤其关注作业中的变量命名以及写法等相似性。

        [作业A内容]:
        ---
        {text1[:20000]}
        ---

        [作业B内容]:
        ---
        {text2[:20000]}
        ---

        [你的任务]:
        请仔细比对两份作业，并严格按照以下JSON格式返回你的分析结果。不要包含任何额外的解释。
        {{
          "is_plagiarized": <如果是抄袭则为 true，否则为 false>,
          "reasoning": "<详细解释你判断的理由，例如：'两份代码的核心算法逻辑完全相同，仅变量名不同' 或 '尽管主题相同，但论述结构和具体案例完全不同，不像抄袭'。>",
          "suspicious_parts": [
            "<引用你认为可疑的具体文本片段1>",
            "<引用另一个可疑的文本片段>"
          ]
        }}
        """
        try:
            response_str = self._call_api(user_prompt, system_prompt, 0.1)
            json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return LLMAnalysis(**data)
        except Exception as e:
            print(f"LLM查重分析时出错: {e}")
        return None

    def grade_homework(self, question: str, rubric: dict, student_answer: str, plagiarism_report: Optional[PlagiarismReport] = None, aigc_report: Optional[AIGCReport] = None) -> dict:
        """调用DeepSeek API来批改作业，现在可以接收查重报告和AIGC检测报告作为参考。"""
        MAX_CHARS = 40000 
        if len(student_answer) > MAX_CHARS:
            print(f"警告: 学生提交内容过长({len(student_answer)}字符)，将被截断为{MAX_CHARS}字符。")
            student_answer = student_answer[:MAX_CHARS] + "\n\n[...内容过长，已被截断...]"

        student_answer = json.dumps(student_answer)

        rubric_str = ""
        for key, value in rubric.items():
            rubric_str += f"- 标准: '{key}', 描述: '{value['description']}', 满分: {value['score']}\n"

        system_prompt = """
        你是一位一丝不苟、公平公正的大学教授AI。你的任务是为一个学生的项目评分，使用中文
        你必须精确地遵循所有指令。你的最终输出必须是一个单一、有效的JSON对象，不能包含任何其他内容。
        不要在JSON对象之前或之后包含任何文本、解释或Markdown格式。
        """
        
        plagiarism_context = ""
        if plagiarism_report and plagiarism_report.llm_analysis and plagiarism_report.llm_analysis.is_plagiarized:
            plagiarism_context = f"""
            [学术诚信警报]:
            AI深度分析表明，本次提交存在高度抄袭的可能性。请在评分时仔细参考此报告。
            分析理由: {plagiarism_report.llm_analysis.reasoning}
            ---
            """
        
        aigc_context = ""
        if aigc_report and aigc_report.ai_probability > 0.8: # 如果AI生成概率很高
            aigc_context = f"""
            [AIGC内容警报]:
            我们的检测模型发现，这份作业有 {aigc_report.ai_probability*100:.1f}% 的可能性是由AI生成的。
            请在评估学生的原创性和真实理解程度时，将此信息作为重要参考。
            ---
            """

        user_prompt = f"""
        请为学生的项目按照以下评分细则评分，然后以指定的JSON格式提供最终输出。

        构建一个包含最终结果的单一JSON对象。该JSON对象必须包含 "total_score"、"overall_feedback" 和 "score_details" 这几个键。"score_details" 必须是一个对象数组，每个对象包含 "criterion"、"score"、"max_score" 和 "feedback"。

        ---
        [任务信息]
        题目: {question}
        评分细则:
        {rubric_str}
        ---
        {plagiarism_context}
        {aigc_context}
        [学生提交内容]
        {student_answer}
        ---

        现在，请仅以所要求的JSON格式提供你的最终评估。
        """
        try:
            response_str = self._call_api(user_prompt, system_prompt, 0.2)
            json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group(0))
                if "total_score" in parsed_json and "overall_feedback" in parsed_json and "score_details" in parsed_json:
                    return parsed_json
                else:
                    error_feedback = f"AI返回了有效的JSON，但键名不匹配。内容: {json.dumps(parsed_json, ensure_ascii=False)}"
                    print(f"[格式错误] {error_feedback}")
                    return {"total_score": -1, "overall_feedback": error_feedback, "score_details": []}
            else:
                return {"total_score": -1, "overall_feedback": f"AI返回格式错误，无法解析JSON。原始返回: {response_str}", "score_details": []}
        except Exception as e:
            print(f"调用DeepSeek API进行评分时出错: {e}")
            return {"total_score": -1, "overall_feedback": f"调用AI服务时发生错误: {e}", "score_details": []}

deepseek_service = DeepSeekService()
