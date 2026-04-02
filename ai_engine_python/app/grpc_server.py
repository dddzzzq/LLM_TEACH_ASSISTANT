import sys
import os
import grpc
import time
import json
import logging
import multiprocessing
import tempfile
import asyncio
from concurrent import futures

# 动态引入根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
sys.path.append(os.path.join(os.path.dirname(__file__), 'pb2'))

import compute_service_pb2 as pb2
import compute_service_pb2_grpc as pb2_grpc

from app.schemas.models import PlagiarismReport, AIGCReport, CodeDocMatchReport

# 拦截子进程无意义的导入，只有主进程才加载大模型
if multiprocessing.current_process().name == 'MainProcess':
    from app.services.plagiarism_service import plagiarism_service
    from app.services.aigc_service import aigc_detector_service
    from app.services.deepseek_service import deepseek_service
    from app.services.ocr_service import ocr_service_instance
else:
    plagiarism_service = None
    aigc_detector_service = None
    deepseek_service = None
    ocr_service_instance = None

root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

class ComputeServicer(pb2_grpc.ComputeServiceServicer):
    # 作业批改
    def ExtractText(self, request, context):
        start_time = time.time()
        try:
            suffix = os.path.splitext(request.filename)[1]
            if not suffix: suffix = '.png'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(request.file_content)
                tmp_path = tmp.name
                
            ocr_text = ocr_service_instance.get_concatenated_text([tmp_path])
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
            return pb2.ExtractResponse(text_content=ocr_text)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.ExtractResponse(text_content=f"【系统降级】图片识别失败: {str(e)}")

    def CheckPlagiarism(self, request, context):
        try:
            student_texts = dict(request.student_texts)
            results = plagiarism_service.check_plagiarism_in_batch(student_texts)
            suspicious_text_pairs = results.get("suspicious_text_pairs", [])
            suspicious_code_pairs = results.get("suspicious_code_pairs", [])
            separated_contents = results.get("separated_contents", {})

            llm_analysis_cache = {}
            all_suspicious_pairs = [(*pair, 'text') for pair in suspicious_text_pairs] + [(*pair, 'code') for pair in suspicious_code_pairs]
            
            for i, (s1, s2, initial_score, content_type) in enumerate(all_suspicious_pairs, 1):
                content1 = separated_contents.get(s1, {}).get(content_type)
                content2 = separated_contents.get(s2, {}).get(content_type)
                if not content1 or not content2: continue
                
                llm_analysis, _ = deepseek_service.analyze_plagiarism(content1, content2, content_type)
                if llm_analysis:
                    llm_analysis_cache[(s1, s2, content_type)] = {'initial_score': initial_score, 'llm_analysis': llm_analysis}
            
            final_plag_data = {sid: [] for sid in student_texts.keys()}
            for (s1, s2, content_type), analysis_result in llm_analysis_cache.items():
                llm_data = analysis_result.get('llm_analysis')
                if llm_data:
                    final_plag_data[s1].append({"similar_to": s2, "content_type": content_type, "initial_score": float(analysis_result['initial_score']), "llm_analysis": llm_data})
                    final_plag_data[s2].append({"similar_to": s1, "content_type": content_type, "initial_score": float(analysis_result['initial_score']), "llm_analysis": llm_data})

            return pb2.PlagiarismResponse(plagiarism_results_json=json.dumps(final_plag_data, ensure_ascii=False))
        except Exception as e:
            return pb2.PlagiarismResponse(plagiarism_results_json="{}")

    def DetectAIGC(self, request, context):
        try:
            report = aigc_detector_service.detect(request.text_content)
            return pb2.AIGCResponse(aigc_report_json=json.dumps(report, ensure_ascii=False))
        except Exception as e:
            return pb2.AIGCResponse(aigc_report_json="{}")

    def GradeHomework(self, request, context):
        sid = request.student_id
        try:
            rubric = json.loads(request.rubric_json) if request.rubric_json else {}
            plag_reports_raw = json.loads(request.plagiarism_reports_json) if request.plagiarism_reports_json else []
            plag_reports = []
            for p in plag_reports_raw:
                try: plag_reports.append(PlagiarismReport(**p))
                except: pass
            aigc_report_raw = json.loads(request.aigc_report_json) if request.aigc_report_json else None
            aigc_report = None
            if aigc_report_raw and "ai_probability" in aigc_report_raw:
                try: aigc_report = AIGCReport(**aigc_report_raw)
                except: pass
            
            prose_content, code_content = aigc_detector_service._separate_content(request.student_text)
            match_analysis, _ = deepseek_service.analyze_code_doc_match(code_content=code_content, doc_content=prose_content, assignment_requirement=request.question)
            code_doc_match_report = None
            if match_analysis and "score" in match_analysis:
                try: code_doc_match_report = CodeDocMatchReport(**match_analysis)
                except: pass

            ai_result = deepseek_service.grade_homework(question=request.question, rubric=rubric, student_answer=request.student_text, plagiarism_reports=plag_reports, aigc_report=aigc_report, code_doc_match_report=code_doc_match_report)
            
            return pb2.GradeResponse(total_score=ai_result.get("total_score", -1.0), feedback=ai_result.get("overall_feedback", "评分失败"), merged_content=request.student_text, code_doc_match_report_json=json.dumps(match_analysis, ensure_ascii=False) if match_analysis else "{}")
        except Exception as e:
            return pb2.GradeResponse(total_score=-1.0, feedback=f"评分失败: {str(e)}")

    # 试卷部分
    def IdentifyQuestionNumber(self, request, context):
        start_time = time.time()
        try:
            logging.info("试卷批改节点: 开始识别OCR文本归属题号...")
            predicted_q_num = asyncio.run(
                deepseek_service.identify_question_number(
                    ocr_text=request.ocr_text, 
                    question_list=request.question_list_str
                )
            )
            logging.info(f"[题号识别] 完成，识别结果: 第 {predicted_q_num} 题, 耗时: {time.time()-start_time:.2f}s")
            return pb2.IdentifyQuestionResponse(question_number=predicted_q_num)
        
        except Exception as e:
            logging.error(f"[IdentifyQuestionNumber] 失败: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.IdentifyQuestionResponse(question_number=0)

    def GradeExamQuestion(self, request, context):
        start_time = time.time()
        try:
            logging.info(f"试卷批改节点: 开始分析单道试卷题目逻辑...")
            grading_result = asyncio.run(
                deepseek_service.grade_exam_question(
                    question=request.question_text,
                    standard_answer=request.standard_answer,
                    rubric=request.rubric,
                    max_score=request.max_score,
                    full_student_text=request.full_student_text
                )
            )
            
            if not grading_result:
                raise ValueError("大模型未能返回有效的批改JSON数据")
                
            logging.info(f"[单题批改] 完成，得分: {grading_result.get('score', 0)}, 耗时: {time.time()-start_time:.2f}s")
            return pb2.GradeExamQuestionResponse(
                score=float(grading_result.get("score", 0.0)),
                feedback=grading_result.get("feedback", "无评语"),
                student_answer_extracted=grading_result.get("student_answer_extracted", "未能提取到答案")
            )
            
        except Exception as e:
            logging.error(f"[GradeExamQuestion] 失败: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.GradeExamQuestionResponse(score=0.0, feedback=f"评分失败: {str(e)}", student_answer_extracted="")

    def SummarizeExam(self, request, context):
        start_time = time.time()
        try:
            logging.info(f"试卷批改节点: 正在生成整卷总评与学习建议...")
            summary_report = asyncio.run(
                deepseek_service.summarize_exam_performance(list(request.all_feedback))
            )
            logging.info(f"[试卷总评] 完成，耗时: {time.time()-start_time:.2f}s")
            return pb2.SummarizeExamResponse(summary_report=summary_report or "生成总结报告失败。")
            
        except Exception as e:
            logging.error(f"[SummarizeExam] 失败: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.SummarizeExamResponse(summary_report=f"总结失败: {str(e)}")

    def PoolScores(self, request, context):
        start_time = time.time()
        try:
            logging.info(f"成绩池化节点: 开始对作业 {request.assignment_id} 进行成绩池化处理...")
            
            # 调用DeepSeek服务进行成绩池化
            pooled_results = asyncio.run(
                deepseek_service.pool_scores(
                    assignment_id=request.assignment_id,
                    scores_data_json=request.scores_data_json
                )
            )
            
            if not pooled_results:
                logging.error(f"[PoolScores] 大模型未能返回有效的池化结果")
                return pb2.PoolScoresResponse(
                    success=False,
                    pooled_results_json="{}",
                    error_message="AI成绩池化处理失败"
                )
            
            logging.info(f"[成绩池化] 完成，处理了 {len(pooled_results) if isinstance(pooled_results, list) else '未知'} 条记录，耗时: {time.time()-start_time:.2f}s")
            return pb2.PoolScoresResponse(
                success=True,
                pooled_results_json=json.dumps(pooled_results, ensure_ascii=False),
                error_message=""
            )
            
        except Exception as e:
            logging.error(f"[PoolScores] 失败: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.PoolScoresResponse(
                success=False,
                pooled_results_json="{}",
                error_message=f"成绩池化处理失败: {str(e)}"
            )


def serve():
    logging.info("正在初始化 AI 模型节点...")
    # 强制预热模型
    _ = aigc_detector_service
    _ = plagiarism_service

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_ComputeServiceServicer_to_server(ComputeServicer(), server)
    
    server.add_insecure_port('0.0.0.0:50051')
    server.start()
    logging.info("Python AI 节点启动成功，纯粹作为推理机监听 50051 端口...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()