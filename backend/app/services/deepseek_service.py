import re
import json
import time
import os
import requests
import asyncio
from ..core.config import settings
from typing import List, Optional, Dict, Tuple, Any
from ..schemas.models import PlagiarismReport, AIGCReport, CodeDocMatchReport

class DeepSeekService:
    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", settings.DEEPSEEK_API_KEY)
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _call_api_with_usage(self, user_prompt: str, system_prompt: str, temperature: float = 0.2, response_format: Optional[Dict[str, str]] = None) -> Optional[Dict]:
        """
        一个通用的、私有的API调用方法，返回完整的API响应体（包含usage）。
        增加自动重试和对不完整/无效JSON响应、多种编码格式的安全处理。
        """
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
        }

        if response_format:
            payload["response_format"] = response_format
        
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

    def _call_api(self, user_prompt: str, system_prompt: str, temperature: float = 0.2, response_format: Optional[Dict[str, str]] = None) -> Optional[str]:
        """一个通用的、私有的API调用方法，增加自动重试和对不完整/无效JSON响应、多种编码格式的安全处理。"""
        full_response = self._call_api_with_usage(user_prompt, system_prompt, temperature, response_format)
        
        if full_response:
            try:
                return full_response['choices'][0]['message']['content']
            except (KeyError, IndexError):
                print(f"警告: LLM API返回了不完整的JSON数据: {full_response}")
                return None
        return None
    
    def _call_api_json(self, user_prompt: str, system_prompt: str, temperature: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        调用API并期望返回一个JSON对象。
        """
        response_str = self._call_api(
            user_prompt, 
            system_prompt, 
            temperature,
            response_format={"type": "json_object"}
        )
        
        if not response_str:
            print("API调用未返回任何内容")
            return None
            
        try:
            # response_format="json_object" 模式会确保返回的是一个合法的JSON字符串
            return json.loads(response_str)
        except json.JSONDecodeError:
            print(f"API返回的不是有效的JSON: {response_str}")
            # 尝试从Markdown代码块中提取
            json_match = re.search(r'```json\n(\{.*?\})\n```', response_str, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    print("从Markdown中提取JSON也失败了")
                    
        return None

    
    def analyze_plagiarism(self, content1: str, content2: str, content_type: str) -> Optional[Dict]:
        system_prompt = "你是一个客观、精准的分析助手。你的输出必须是一个单一、有效的JSON对象，不能包含任何其他内容。"
        
        if content_type == 'text':
            user_prompt = self._get_text_plagiarism_prompt(content1, content2)
        elif content_type == 'code':
            user_prompt = self._get_code_plagiarism_prompt(content1, content2)
        else:
            return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        analysis_result = None

        try:
            full_response = self._call_api_with_usage(user_prompt, system_prompt, 0.1, response_format={"type": "json_object"})
            
            if full_response:
                response_str = full_response.get('choices', [{}])[0].get('message', {}).get('content')
                usage_data = full_response.get('usage', {})
                
                usage_info = {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0)
                }

                if response_str:
                    analysis_result = json.loads(response_str)

        except Exception as e:
            print(f"LLM抄袭分析时出错: {e}")
        
        return analysis_result, usage_info
    
    
    def analyze_code_doc_match(self, code_content: str, doc_content: str) -> Tuple[Optional[Dict], Dict]:
        """调用LLM来评估代码和文档的匹配度。"""
        system_prompt = "你是一位经验丰富的计算机科学课程助教。你的输出必须是一个单一、有效的JSON对象，不能包含任何其他内容。"
        
        user_prompt = f"""
        你是一位经验丰富的计算机科学课程助教，你的任务是评估学生提交的作业中，代码和项目文档之间的一致性和匹配程度。但是可能存在上下文限制，请你适当给出分数。

        
        【项目代码】
        ```
        {json.dumps(code_content[:30000], ensure_ascii=False)}
        ```

        
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
            full_response = self._call_api_with_usage(user_prompt, system_prompt, 0.1, response_format={"type": "json_object"})
            if full_response:
                response_str = full_response.get('choices', [{}])[0].get('message', {}).get('content')
                usage_data = full_response.get('usage', {})
                usage_info = {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0)
                }
                if response_str:
                    analysis_result = json.loads(response_str)
        except Exception as e:
            print(f"LLM代码-文档匹配度分析时出错: {e}")

        return analysis_result, usage_info
    
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
                
                """
        
        aigc_context = ""
        if aigc_report and aigc_report.ai_probability > 0.8:
            aigc_context = f"""
            [AIGC内容警报]:
            检测模型发现，这份作业的'{aigc_report.detection_source}'部分有 {aigc_report.ai_probability * 100:.1f}% 的可能性由AI生成。
            
            """
        
        # 新增匹配报告
        match_context = ""
        if code_doc_match_report and code_doc_match_report.score < 70:
            match_context = f"""
            [代码-文档不匹配警报]:
            AI分析发现，代码与文档的匹配度较低（{code_doc_match_report.score}/100分）。
            理由: {code_doc_match_report.reasoning}
            这可能表明学生未认真撰写文档，请在评分时予以考虑。
            
            """

        system_prompt = "你是一位一丝不苟的大学教授AI。你的输出必须是一个单一、有效的JSON对象。"
        user_prompt = f"""
        请为学生的项目评分。
        [任务信息]
        题目: {question}
        评分细则: {json.dumps(rubric, ensure_ascii=False)}
        
        {plagiarism_context}
        {aigc_context}
        {match_context}
        [学生提交内容]
        {json.dumps(student_answer[:50000])}
        
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
            response_json = self._call_api_json(user_prompt, system_prompt, 0.1)
            if response_json:
                return response_json
        except Exception as e:
            print(f"评分时发生错误: {e}")

        return {"total_score": -1, "overall_feedback": "AI评分服务出错", "score_details": []}

    #  辅助函数 (原版，保持不变) 
    def _get_text_plagiarism_prompt(self, text1: str, text2: str) -> str:
        escaped_text1 = json.dumps(text1[:25000], ensure_ascii=False)
        escaped_text2 = json.dumps(text2[:25000], ensure_ascii=False)
        return f"""
        你是一位经验丰富的大学教授。请对比以下两份**实验报告**，扮演一个客观的第三方顾问角色。
        你的任务是检测两份报告的抄袭情况，但是并不严格不容忍抄袭问题，在教学场景下，允许适当的文本复用，但是要求有自己的思考，最后提供一份详细的辅助决策中文报告，包含：
        1.  一个0到100的**语义相似度分数**。
        2.  详细的**分析理由**，关注论点、结构和措辞。
        3.  列出1-3个最能支撑你结论的**核心文本片段**作为证据。

        [报告 A]:
        
        {escaped_text1}
        
        [报告 B]:
        
        {escaped_text2}
        
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
        return f"""
        你是一位经验丰富的大学教授。请对比以下两份**源代码**，扮演一个客观的第三方代码审查顾问角色。
        你的任务是检测两份代码的抄袭情况，但是并不严格不容忍抄袭问题，在教学场景下，允许相当一部分的代码复用，但是要求有自己的思考，最后提供一份详细的辅助决策中文报告，包含：
        1.  一个0到100的**语义相似度分数**。
        2.  详细的**分析理由**，关注论点、结构和措辞。
        3.  列出1-3个最能支撑你结论的**核心文本片段**作为证据。

        [报告 A]:
        
        {escaped_code1}
        
        [报告 B]:
        
        {escaped_code2}
        
        请严格按照以下JSON格式返回你的分析报告:
        {{
          "similarity_score": <number>,
          "reasoning": "<string>",
          "suspicious_parts": [
            {{ "student_A_content": "<string>", "student_B_content": "<string>" }}
          ]
        }}
        """
    
    async def identify_question_number(self, ocr_text: str, question_list: str) -> int:
        """
        判断OCR文本最可能属于试卷中的哪一道题。
        返回题号 (1, 2, 3...)，如果无法识别则返回 0。
        """
        # 修复：在 System Prompt 中明确包含 "JSON" 关键字
        system_prompt = (
            "你是一个智能试卷分析助手。"
            "你的任务是根据提供的OCR文本片段和试卷题目列表，判断这段文本最可能是对哪道题的回答。"
            "请务必以 JSON 格式输出结果 (Must output JSON)."
        )
        
        user_prompt = f"""
        [试卷题目列表]:
        {question_list}
        
        [图片OCR文本]:
        
        {ocr_text[:2000]} 
        
        
        [任务]:
        请分析[图片OCR文本]的内容，它看起来是针对[试卷题目列表]中哪一道题的回答？
        请注意：
        1. 学生可能会写 "1. xxx", "第一题", "(1)" 等标识。
        2. 如果没有明确标识，请根据语义内容匹配题目。
        3. 如果文本包含多道题的回答，请返回第一道出现的题号。
        4. 如果完全无关或无法识别，返回0。
        
        [输出格式]:
        请严格返回如下 JSON 格式：
        {{
            "question_number": <int>
        }}
        """
        
        try:
            response_json = await asyncio.to_thread(
                self._call_api_json,
                user_prompt,
                system_prompt,
                0.1
            )
            if response_json:
                return response_json.get("question_number", 0)
        except Exception as e:
            print(f"识别题号失败: {e}")
        return 0


    #  试卷评分新方法 

    async def grade_exam_question(self, question: str, standard_answer: str, rubric: str, max_score: float, full_student_text: str) -> Optional[Dict[str, Any]]:
        """
        调用DeepSeek API批改单个试卷题目（可能包含多个子问题）
        优化：采用 Chain of Thought (CoT) 策略，强制要求先生成评语，最后生成分数。
        """
        system_prompt = (
            "你是一位经验丰富、富有洞察力且评分人性化的大学教授。"
            "你非常擅长理解由OCR（光学字符识别）生成的文本，能够自动纠正识别错误并还原学生原意。"
            "在评分时，请注重语义的正确性，对OCR错误保持高度包容，并乐于给予学生鼓励分。"
            "**核心指令**：为了保证评分的准确性和逻辑一致性，请务必**先进行详细的分析和点评（Thinking）**，仔细推导每一步的得分和扣分，**最后**再根据你的分析总结出最终得分。"
            "你必须严格按照指定的JSON格式返回结果。"
        )
        
        user_prompt = f"""
        请评分以下题目：

        [题目内容]:
        {question}

        [标准答案]:
        {standard_answer}

        [题目总分]:
        {max_score} 分

        [评分标准]:
        {rubric}

        [学生试卷完整OCR文本]:
        (学生在试卷上所有题目的作答内容都在这里，请仔细查找对应的作答区域)
        
        {full_student_text[:50000]}
        

        [评分原则 / Grading Philosophy]:
        1.  **思维链 (Chain of Thought)**: 请不要直接给出分数。**必须先在 'feedback' 字段中详细写出你的思考过程**。逐条对比评分标准，说明学生答对了哪些点，答错了哪些点，哪些地方因为OCR错误被纠正了。
        2.  **分数一致性**: 'score' 字段的值必须是你 'feedback' 中所有得分点的总和。**先写评语，根据评语算分。**
        3.  **OCR容错**: 自动修正OCR错误。只要推断出原意，不扣分。
        4.  **概念理解满分策略**: 意思对就给满分，不要死扣字眼。

        [你的任务步骤]:
        1.  **提取**: 从OCR文本中定位学生答案。
        2.  **分析 (Feedback Generation)**: 编写评语。逐项分析得分点。如果扣分，明确写出理由。**在评语的最后，请显式地写出计算过程，例如：“得分点1得3分，得分点2得4分，扣分点X扣1分，总计XX分”。**
        3.  **总结 (Final Score)**: 将第2步计算出的总分填入 'score' 字段。

        [输出格式 - 务必遵守字段顺序]:
        请严格按照以下JSON格式返回，**注意字段顺序**：
        {{
          "student_answer_extracted": "<string, 提取出的学生作答内容>",
          "feedback": "<string, 详细的评分理由和计算过程>",
          "score": <number, 最终得分，必须与feedback中的计算一致>
        }}
        """
        
        try:
            # 这是一个异步方法，所以我们使用 await
            response_json = await asyncio.to_thread(
                self._call_api_json,
                user_prompt, 
                system_prompt,
                0.1
            )
            
            # 确保分数不会超过最大分
            if response_json and 'score' in response_json:
                # 再次做一层防御性编程：确保分数不超过max_score
                response_json['score'] = min(max_score, float(response_json['score']))
                
            return response_json
        except Exception as e:
            print(f"调用LLM批改题目时出错: {e}")
            return None

    async def summarize_exam_performance(self, all_feedback: List[str]) -> Optional[str]:
        """
        调用DeepSeek API为学生的整张试卷生成总结报告
        """
        system_prompt = (
            "你是一位经验丰富、富有同理心的辅导教师。"
            "你的任务是根据学生试卷上每道题的得分和评语，生成一份总体总结和学习建议。"
            "报告应简明扼要，重点突出知识点的掌握情况和改进方向。"
            "直接返回报告文本，不要使用JSON。"
        )
        
        feedback_str = "\n".join(all_feedback)
        
        user_prompt = f"""
        以下是学生在一张试卷上所有题目的得分和评语列表：

        
        {feedback_str}
        

        [任务]:
        请根据以上信息，为该学生撰写一份100-200字的试卷总结报告，包括：
        1.  总体表现（哪些方面做得好，哪些方面有欠缺）。
        2.  具体的学习建议（针对薄弱环节）。

        [输出]:
        请直接输出报告正文（纯文本）。
        """
        
        try:
            # 这是一个异步方法，所以我们使用 await
            response_text = await asyncio.to_thread(
                self._call_api,
                user_prompt, 
                system_prompt, 
                0.5 # 总结时温度可以稍高，增加多样性
            )
            return response_text
        except Exception as e:
            print(f"调用LLM生成总结报告时出错: {e}")
            return None


deepseek_service = DeepSeekService()