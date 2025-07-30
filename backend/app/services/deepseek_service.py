import re
import json
import requests
from ..core.config import settings
from typing import List, Optional, Dict
from ..schemas.models import PlagiarismReport, AIGCReport

class DeepSeekService:
    def __init__(self):
        self.api_key = settings.DEEPSEEK_API_KEY
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _call_api(self, user_prompt: str, system_prompt: str) -> str:
        """一个通用的、私有的API调用方法。"""
        payload = {"model": "deepseek-chat", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]}
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=180)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"调用DeepSeek API时出错: {e}")
            raise

#     def analyze_plagiarism(self, text1: str, student1_id: str, text2: str, student2_id: str) -> Optional[LLMAnalysis]:
#         """调用LLM来深度分析两个文本的相似性，即查重检测"""
#         text1 = json.dumps(text1[:20000])
#         text2 = json.dumps(text2[:20000])
#         system_prompt = "你是一位学术诚信审查官AI。你的输出必须是一个单一、有效的JSON对象，不能包含任何其他内容，使用中文"
#         user_prompt = f"""
#         你是一位经验丰富的学术诚信审查官。你的任务是判断两份学生作业之间是否存在抄袭。
                                         
#         [作业信息]:
#         - 作业A来自学生: {student1_id}
#         - 作业B来自学生: {student2_id}
#         这两份作业在初步的关键词频率检测中显示出高度相似性。你需要进行深度语义分析，尤其关注作业中的变量命名以及写法等相似性。

#         [作业A内容]:
#         ---
#         {text1[:20000]}
#         ---

#         [作业B内容]:
#         ---
#         {text2[:20000]}
#         ---

#         [你的任务]:
#         请仔细比对两份作业，并严格按照以下JSON格式返回你的分析结果。不要包含任何额外的解释。
#         {{
#           "is_plagiarized": <如果是抄袭则为 true，否则为 false>,
#           "reasoning": "<详细解释你判断的理由，例如：'两份代码的核心算法逻辑完全相同，仅变量名不同' 或 '尽管主题相同，但论述结构和具体案例完全不同，不像抄袭'。>",
#           "suspicious_parts": [
#             "<引用你认为可疑的具体文本片段1>",
#             "<引用另一个可疑的文本片段>"
#           ]
#         }}
#         """
#         try:
#             response_str = self._call_api(user_prompt, system_prompt, 0.1)
#             json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
#             if json_match:
#                 data = json.loads(json_match.group(0))
#                 return LLMAnalysis(**data)
#         except Exception as e:
#             print(f"LLM查重分析时出错: {e}")
#         return None

    def _get_text_plagiarism_prompt(self, text1: str, text2: str) -> str:
        return f"""
        你是一位经验丰富的学术评审专家。请对比以下两份**实验报告**，扮演一个客观的第三方顾问角色。
        你的任务是提供一份详细的辅助决策报告，包含：
        1.  一个0到100的**语义相似度分数**。
        2.  详细的**分析理由**，关注论点、结构和措辞。
        3.  列出1-3个最能支撑你结论的**核心文本片段**作为证据。

        [报告 A]:
        ---
        {text1[:20000]}
        ---
        [报告 B]:
        ---
        {text2[:20000]}
        ---
        请严格按照以下JSON格式返回你的分析报告:
        {{
          "similarity_score": <number>,
          "reasoning": "<string>",
          "suspicious_parts": [
            {{ "student_A_content": "<string>", "student_B_content": "<string>" }}
          ]
        }}
        """

    def _get_code_plagiarism_prompt(self, code1: str, code2: str) -> str:
        return f"""
        你是一位资深的软件工程技术主管。请对比以下两份**源代码**，扮演一个客观的第三方代码审查顾问角色。
        你的任务是提供一份详细的辅助决策报告，包含：
        1.  一个0到100的**逻辑与结构相似度分数**。
        2.  详细的**分析理由**，关注算法、结构、命名和注释。
        3.  列出1-3个最能支撑你结论的**核心代码片段**作为证据。

        [代码 A]:
        ---
        {code1[:20000]}
        ---
        [代码 B]:
        ---
        {code2[:20000]}
        ---
        请严格按照以下JSON格式返回你的分析报告:
        {{
          "similarity_score": <number>,
          "reasoning": "<string>",
          "suspicious_parts": [
            {{ "student_A_content": "<string>", "student_B_content": "<string>" }}
          ]
        }}
        """

    def analyze_plagiarism(self, content1: str, content2: str, content_type: str) -> Optional[Dict]:
        system_prompt = "你是一个客观、精准的分析助手。你的输出必须是一个单一、有效的JSON对象，不能包含任何其他内容。"
        
        if content_type == 'text':
            user_prompt = self._get_text_plagiarism_prompt(content1, content2)
        elif content_type == 'code':
            user_prompt = self._get_code_plagiarism_prompt(content1, content2)
        else:
            return None

        try:
            response_str = self._call_api(user_prompt, system_prompt)
            json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            print(f"LLM抄袭分析时出错: {e}")
        return None
    
#   def grade_homework(self, question: str, rubric: dict, student_answer: str, plagiarism_report: Optional[PlagiarismReport] = None, aigc_report: Optional[AIGCReport] = None) -> dict:
#         """调用DeepSeek API来批改作业，现在可以接收查重报告和AIGC检测报告作为参考。"""
#         MAX_CHARS = 40000 
#         if len(student_answer) > MAX_CHARS:
#             print(f"警告: 学生提交内容过长({len(student_answer)}字符)，将被截断为{MAX_CHARS}字符。")
#             student_answer = student_answer[:MAX_CHARS] + "\n\n[...内容过长，已被截断...]"

#         student_answer = json.dumps(student_answer)

#         rubric_str = ""
#         for key, value in rubric.items():
#             rubric_str += f"- 标准: '{key}', 描述: '{value['description']}', 满分: {value['score']}\n"

#         system_prompt = """
#         你是一位一丝不苟、公平公正的大学教授AI。你的任务是为一个学生的项目评分，使用中文
#         你必须精确地遵循所有指令。你的最终输出必须是一个单一、有效的JSON对象，不能包含任何其他内容。
#         不要在JSON对象之前或之后包含任何文本、解释或Markdown格式。
#         """
        
#         plagiarism_context = ""
#         if plagiarism_report and plagiarism_report.llm_analysis and plagiarism_report.llm_analysis.is_plagiarized:
#             plagiarism_context = f"""
#             [学术诚信警报]:
#             AI深度分析表明，本次提交存在高度抄袭的可能性。请在评分时仔细参考此报告。
#             分析理由: {plagiarism_report.llm_analysis.reasoning}
#             ---
#             """
        
#         aigc_context = ""
#         if aigc_report and aigc_report.ai_probability > 0.8: # 如果AI生成概率很高
#             aigc_context = f"""
#             [AIGC内容警报]:
#             我们的检测模型发现，这份作业有 {aigc_report.ai_probability*100:.1f}% 的可能性是由AI生成的。
#             请在评估学生的原创性和真实理解程度时，将此信息作为重要参考。
#             ---
#             """

#         user_prompt = f"""
#         请为学生的项目按照以下评分细则评分，然后以指定的JSON格式提供最终输出。

#         构建一个包含最终结果的单一JSON对象。该JSON对象必须包含 "total_score"、"overall_feedback" 和 "score_details" 这几个键。"score_details" 必须是一个对象数组，每个对象包含 "criterion"、"score"、"max_score" 和 "feedback"。

#         ---
#         [任务信息]
#         题目: {question}
#         评分细则:
#         {rubric_str}
#         ---
#         {plagiarism_context}
#         {aigc_context}
#         [学生提交内容]
#         {student_answer}
#         ---

#         现在，请仅以所要求的JSON格式提供你的最终评估。
#         """
#         try:
#             response_str = self._call_api(user_prompt, system_prompt, 0.2)
#             json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
#             if json_match:
#                 parsed_json = json.loads(json_match.group(0))
#                 if "total_score" in parsed_json and "overall_feedback" in parsed_json and "score_details" in parsed_json:
#                     return parsed_json
#                 else:
#                     error_feedback = f"AI返回了有效的JSON，但键名不匹配。内容: {json.dumps(parsed_json, ensure_ascii=False)}"
#                     print(f"[格式错误] {error_feedback}")
#                     return {"total_score": -1, "overall_feedback": error_feedback, "score_details": []}
#             else:
#                 return {"total_score": -1, "overall_feedback": f"AI返回格式错误，无法解析JSON。原始返回: {response_str}", "score_details": []}
#         except Exception as e:
#             print(f"调用DeepSeek API进行评分时出错: {e}")
#             return {"total_score": -1, "overall_feedback": f"调用AI服务时发生错误: {e}", "score_details": []}

# deepseek_service = DeepSeekService()

    def grade_homework(self, question: str, rubric: dict, student_answer: str, plagiarism_reports: List[PlagiarismReport] = [], aigc_report: Optional[AIGCReport] = None) -> dict:
        plagiarism_context = ""
        if plagiarism_reports:
            highest_plagiarism_score = 0
            worst_report = None
            for report in plagiarism_reports:
                if report.llm_analysis and report.llm_analysis.similarity_score > highest_plagiarism_score:
                    highest_plagiarism_score = report.llm_analysis.similarity_score
                    worst_report = report
            
            if highest_plagiarism_score > 70 and worst_report:
                 plagiarism_context = f"""
                [学术诚信警报]:
                AI深度分析表明，本次提交与学生'{worst_report.similar_to}'的'{worst_report.content_type}'部分存在高度相似（{highest_plagiarism_score}/100分）。
                分析理由: {worst_report.llm_analysis.reasoning}
                ---
                """
        
        aigc_context = ""
        if aigc_report and aigc_report.ai_probability > 0.8:
            aigc_context = f"""
            [AIGC内容警报]:
            检测模型发现，这份作业的'{aigc_report.detection_source}'部分有 {aigc_report.ai_probability*100:.1f}% 的可能性由AI生成。
            ---
            """
        
        system_prompt = "你是一位一丝不苟的大学教授AI。你的输出必须是一个单一、有效的JSON对象。"
        user_prompt = f"""
        请为学生的项目评分。
        [任务信息]
        题目: {question}
        评分细则: {json.dumps(rubric, ensure_ascii=False)}
        ---
        {plagiarism_context}
        {aigc_context}
        [学生提交内容]
        {json.dumps(student_answer[:20000])}
        ---
        请严格按照以下JSON格式提供你的最终评估:
        {{
          "total_score": <number>,
          "overall_feedback": "<string>",
          "score_details": [
            {{ "criterion": "<string>", "score": <number>, "max_score": <number>, "feedback": "<string>" }}
          ]
        }}
        """
        try:
            response_str = self._call_api(user_prompt, system_prompt)
            json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception:
            pass
        return {"total_score": -1, "overall_feedback": "AI评分服务出错", "score_details": []}

deepseek_service = DeepSeekService()
