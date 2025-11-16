import re
import json
import time
import os
import requests
from ..core.config import settings
from typing import List, Optional, Dict, Tuple
from ..schemas.models import PlagiarismReport, AIGCReport, CodeDocMatchReport

class DeepSeekService:
    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _call_api_with_usage(self, user_prompt: str, system_prompt: str) -> Optional[Dict]:
        """
        一个通用的、私有的API调用方法，返回完整的API响应体（包含usage）。
        增加自动重试和对不完整/无效JSON响应、多种编码格式的安全处理。
        """
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        max_retries = 3
        backoff_factor = 2
        response = None

        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=180)
                response.raise_for_status() 
                
                # 强制UTF-8编码
                if response.encoding is None or response.encoding.lower() == 'iso-8859-1':
                    response.encoding = 'utf-8'
                
                response_text = response.text
                
                if not response_text.strip():
                    raise ValueError("API返回了空响应内容")

                # 直接返回解析后的完整JSON数据
                return json.loads(response_text)

            except (requests.exceptions.RequestException, ValueError, json.JSONDecodeError) as e:
                if response is not None:
                    print("========================= DEBUG INFO =========================")
                    print(f"原始响应状态码 (Status Code): {response.status_code}")
                    print(f"原始响应内容 (Raw Response Text): '{response.text}'")
                    print("============================================================")

                print(f"调用DeepSeek API时出错 (第 {attempt + 1} 次尝试): {e}")
                
                if attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    print(f"将在 {wait_time} 秒后进行第 {attempt + 2} 次尝试...")
                    time.sleep(wait_time)
                else:
                    print("调用DeepSeek API失败，已达到最大重试次数。")
                    # 返回None而不是抛出异常，让调用方处理
                    return None
        return None

    def _call_api(self, user_prompt: str, system_prompt: str) -> str:
        """一个通用的、私有的API调用方法，增加自动重试和对不完整/无效JSON响应、多种编码格式的安全处理。"""
        full_response = self._call_api_with_usage(user_prompt, system_prompt)
        
        if full_response:
            try:
                return full_response['choices'][0]['message']['content']
            except (KeyError, IndexError):
                print(f"警告: LLM API返回了不完整的JSON数据: {full_response}")
                return None
        return None
    
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
        escaped_text1 = json.dumps(text1[:25000], ensure_ascii=False)
        escaped_text2 = json.dumps(text2[:25000], ensure_ascii=False)
        # return f"""
        # 你是一位经验丰富的学术评审专家。请对比以下两份**实验报告**，扮演一个客观的第三方顾问角色。
        # 你的任务是提供一份详细的辅助决策中文报告，包含：
        # 1.  一个0到100的**语义相似度分数**。
        # 2.  详细的**分析理由**，关注论点、结构和措辞。
        # 3.  列出1-3个最能支撑你结论的**核心文本片段**作为证据。

        # [报告 A]:
        # ---
        # {escaped_text1[:20000]}
        # ---
        # [报告 B]:
        # ---
        # {escaped_text2[:20000]}
        # ---
        # 请严格按照以下JSON格式返回你的分析报告:
        # {{
        #   "similarity_score": <number>,
        #   "reasoning": "<string>",
        #   "suspicious_parts": [
        #     {{ "student_A_content": "<string>", "student_B_content": "<string>" }}
        #   ]
        # }}
        # """
        return f"""
        你是一位经验丰富的大学教授。请对比以下两份**实验报告**，扮演一个客观的第三方顾问角色。
        你的任务是检测两份报告的抄袭情况，但是并不严格不容忍抄袭问题，在教学场景下，允许适当的文本复用，但是要求有自己的思考，最后提供一份详细的辅助决策中文报告，包含：
        1.  一个0到100的**语义相似度分数**。
        2.  详细的**分析理由**，关注论点、结构和措辞。
        3.  列出1-3个最能支撑你结论的**核心文本片段**作为证据。

        [报告 A]:
        ---
        {escaped_text1[:25000]}
        ---
        [报告 B]:
        ---
        {escaped_text2[:25000]}
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
        escaped_code1 = json.dumps(code1[:25000], ensure_ascii=False)
        escaped_code2 = json.dumps(code2[:25000], ensure_ascii=False)
        # return f"""
        # 你是一位资深的软件工程技术主管。请对比以下两份**源代码**，扮演一个客观的第三方代码审查顾问角色。
        # 你的任务是提供一份详细的辅助决策中文报告，包含：
        # 1.  一个0到100的**逻辑与结构相似度分数**。
        # 2.  详细的**分析理由**，关注算法、结构、命名和注释。
        # 3.  列出1-3个最能支撑你结论的**核心代码片段**作为证据。

        # [代码 A]:
        # ---
        # {escaped_code1[:20000]}
        # ---
        # [代码 B]:
        # ---
        # {escaped_code2[:20000]}
        # ---
        # 请严格按照以下JSON格式返回你的分析报告:
        # {{
        #   "similarity_score": <number>,
        #   "reasoning": "<string>",
        #   "suspicious_parts": [
        #     {{ "student_A_content": "<string>", "student_B_content": "<string>" }}
        #   ]
        # }}
        # """
        # return f"""
        # 你是一位资深的软件工程技术主管。请对比以下两份**源代码**，扮演一个客观的第三方代码审查顾问角色。
        # 你的任务是提供一份详细的辅助决策中文报告，包含：
        # 1.  一个0到100的**逻辑与结构相似度分数**。
        # 2.  详细的**分析理由**，关注算法逻辑以及实现思想。
        # 3.  列出1-3个最能支撑你结论的**核心代码片段**作为证据。

        # [代码 A]:
        # ---
        # {escaped_code1[:20000]}
        # ---
        # [代码 B]:
        # ---
        # {escaped_code2[:20000]}
        # ---
        # 请严格按照以下JSON格式返回你的分析报告:
        # {{
        #   "similarity_score": <number>,
        #   "reasoning": "<string>",
        #   "suspicious_parts": [
        #     {{ "student_A_content": "<string>", "student_B_content": "<string>" }}
        #   ]
        # }}
        # """
        return f"""
        你是一位经验丰富的大学教授。请对比以下两份**源代码**，扮演一个客观的第三方代码审查顾问角色。
        你的任务是检测两份代码的抄袭情况，但是并不严格不容忍抄袭问题，在教学场景下，允许相当一部分的代码复用，但是要求有自己的思考，最后提供一份详细的辅助决策中文报告，包含：
        1.  一个0到100的**语义相似度分数**。
        2.  详细的**分析理由**，关注论点、结构和措辞。
        3.  列出1-3个最能支撑你结论的**核心文本片段**作为证据。

        [报告 A]:
        ---
        {escaped_code1[:25000]}
        ---
        [报告 B]:
        ---
        {escaped_code2[:25000]}
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
            # 确保在任何情况下都返回正确的元组格式
            return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # 默认的usage信息，以防API调用失败
        usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        analysis_result = None

        # try:
        #     response_str = self._call_api(user_prompt, system_prompt)
        #     json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
        #     if json_match:
        #         return json.loads(json_match.group(0))
        # except Exception as e:
        #     print(f"LLM抄袭分析时出错: {e}")
        # return None
        try:
            # 调用能返回完整响应（包括usage）的新方法
            full_response = self._call_api_with_usage(user_prompt, system_prompt)
            
            if full_response:
                # 提取 content 和 usage
                response_str = full_response.get('choices', [{}])[0].get('message', {}).get('content')
                usage_data = full_response.get('usage', {})
                
                # 更新usage信息
                usage_info = {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0)
                }

                if response_str:
                    json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
                    if json_match:
                        analysis_result = json.loads(json_match.group(0))

        except Exception as e:
            print(f"LLM抄袭分析时出错: {e}")
        
        return analysis_result, usage_info
    
    def analyze_code_doc_match(self, code_content: str, doc_content: str) -> Tuple[Optional[Dict], Dict]:
        """调用LLM来评估代码和文档的匹配度。"""
        system_prompt = "你是一位经验丰富的计算机科学课程助教。你的输出必须是一个单一、有效的JSON对象，不能包含任何其他内容。"
        
        user_prompt = f"""
        你是一位经验丰富的计算机科学课程助教，你的任务是评估学生提交的作业中，代码和项目文档之间的一致性和匹配程度。但是可能存在上下文限制，请你适当给出分数。

        ---
        【项目代码】
        ```
        {json.dumps(code_content[:30000], ensure_ascii=False)}
        ```

        ---
        【项目文档】
        ```text
        {json.dumps(doc_content[:25000], ensure_ascii=False)}
        ```

        请基于上面提供的【项目代码】和【项目文档】，进行综合评估，并遵循以下要求：

        1.  **评估维度**：
            * **完整性**：文档是否覆盖了代码中的主要功能、模块和核心逻辑？
            * **准确性**：文档的描述是否准确地反映了代码的实际功能和实现方式？
            * **清晰度**：文档的语言是否清晰易懂，有助于理解代码？

        2.  **评分标准**：
            * 请给出一个0到100的匹配度总分。
            * 90-100分：完美匹配，文档详尽、准确、清晰。
            * 70-89分：良好匹配，大部分功能有描述，但有少量遗漏或不准确之处。
            * 50-69分：基本匹配，文档只描述了部分核心功能，或存在明显与代码不符之处。
            * 0-49分：严重不匹配，文档内容空洞、错误，或与代码完全脱节。

        3.  **输出格式**：
        请严格按照以下JSON格式返回你的分析报告:
        {{
          "score": <number>,
          "reasoning": "<string>",
        }}
            * `score` 字段为0-100的整数。
            * `reasoning` 字段为一段不超过100字的简短评语，总结你的评估依据。
        """

        usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        analysis_result = None
        try:
            full_response = self._call_api_with_usage(user_prompt, system_prompt)
            if full_response:
                response_str = full_response.get('choices', [{}])[0].get('message', {}).get('content')
                usage_data = full_response.get('usage', {})
                usage_info = {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0)
                }
                if response_str:
                    json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
                    if json_match:
                        analysis_result = json.loads(json_match.group(0))
        except Exception as e:
            print(f"LLM代码-文档匹配度分析时出错: {e}")

        return analysis_result, usage_info
    
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

    def grade_homework(self, question: str, rubric: dict, student_answer: str, plagiarism_reports: List[PlagiarismReport] = [], aigc_report: Optional[AIGCReport] = None, code_doc_match_report: Optional[CodeDocMatchReport] = None) -> dict:
        plagiarism_context = ""
        if plagiarism_reports:
            highest_plagiarism_score = 0
            worst_report = None
            for report in plagiarism_reports:
                if report.llm_analysis and report.llm_analysis.similarity_score > highest_plagiarism_score:
                    highest_plagiarism_score = report.llm_analysis.similarity_score
                    worst_report = report
            
            if highest_plagiarism_score > 95 and worst_report:
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
            检测模型发现，这份作业的'{aigc_report.detection_source}'部分有 {aigc_report.ai_probability * 100:.1f}% 的可能性由AI生成。
            ---
            """
        
        # 新增匹配报告
        match_context = ""
        if code_doc_match_report and code_doc_match_report.score < 70:
            match_context = f"""
            [代码-文档不匹配警报]:
            AI分析发现，代码与文档的匹配度较低（{code_doc_match_report.score}/100分）。
            理由: {code_doc_match_report.reasoning}
            这可能表明学生未认真撰写文档，请在评分时予以考虑。
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
        {match_context}
        [学生提交内容]
        {json.dumps(student_answer[:50000])}
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
            if response_str:
                json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
        except Exception as e:
            print(f"评分时发生错误: {e}")

        return {"total_score": -1, "overall_feedback": "AI评分服务出错", "score_details": []}

deepseek_service = DeepSeekService()