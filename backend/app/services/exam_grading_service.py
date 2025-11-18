import asyncio
import traceback
import json
import logging
import os
import uuid
import tempfile
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession

from .ocr_service import ocr_service_instance
from .deepseek_service import deepseek_service
from ..db import database, crud_exams
from ..schemas import models as schemas

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_exam_submission(exam_id: int, student_id: str, image_bytes_list: List[bytes]):
    """
    后台任务：处理学生提交的试卷图片
    1. 将图片字节保存为临时文件
    2. OCR 识别所有图片
    3. 获取试卷的所有题目
    4. 迭代每道题，调用LLM进行评分
    5. 保存每道题的得分
    6. 调用LLM生成总结报告
    7. 保存总结报告
    8. 清理临时文件
    """
    logger.info(f"开始处理试卷 (ExamID: {exam_id}, StudentID: {student_id})")
    
    if not ocr_service_instance:
        logger.error("OCR服务未初始化，无法处理试卷")
        return

    # 1. 将图片字节保存为临时文件
    temp_dir = tempfile.mkdtemp()
    image_path_list = []
    try:
        for i, img_bytes in enumerate(image_bytes_list):
            file_name = f"{uuid.uuid4()}.jpg" # 假设图片是jpeg或png，paddleocr不关心后缀
            file_path = os.path.join(temp_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(img_bytes)
            image_path_list.append(file_path)
        
        logger.info(f"已将 {len(image_path_list)} 张图片保存到临时目录: {temp_dir}")

        async with database.AsyncSessionLocal() as db_session:
            try:
                # 1. 创建或获取 StudentExam 实例
                student_exam = await crud_exams.create_student_exam(
                    db=db_session, 
                    exam_id=exam_id, 
                    student_id=student_id
                )
                logger.info(f"StudentExam ID: {student_exam.id}")

                # 2. OCR 识别所有图片并合并文本 (使用文件路径)
                logger.info("开始OCR识别...")
                full_ocr_text = ocr_service_instance.get_concatenated_text(image_path_list)
                logger.info(f"OCR 识别完成。总长度: {len(full_ocr_text)} 字符。")
                
                # 3. 获取试卷的所有题目
                questions = await crud_exams.get_exam_questions(db=db_session, exam_id=exam_id)
                if not questions:
                    logger.warning(f"试卷 (ExamID: {exam_id}) 没有任何题目，无法评分。")
                    return
                
                logger.info(f"共获取到 {len(questions)} 道题目，开始逐一评分...")

                all_answers = []
                total_score = 0.0

                # 4. 迭代每道题，调用LLM进行评分
                for question in questions:
                    try:
                        logger.info(f"正在评分: 题目 {question.question_number}...")
                        
                        # 调用LLM评分
                        grading_result = await deepseek_service.grade_exam_question(
                            question=question.question_text,
                            standard_answer=question.standard_answer,
                            rubric=question.rubric,
                            max_score=question.max_score,
                            full_student_text=full_ocr_text
                        )
                        
                        if not grading_result:
                            logger.warning(f"题目 {question.question_number} LLM评分失败，跳过。")
                            grading_result = {"score": 0.0, "feedback": "LLM评分失败", "student_answer_extracted": "未能提取答案"}

                        # 5. 保存每道题的得分
                        answer_data = schemas.StudentQuestionAnswerCreate(
                            student_exam_id=student_exam.id,
                            exam_question_id=question.id,
                            student_answer_text=grading_result.get("student_answer_extracted", "N/A"),
                            score=grading_result.get("score", 0.0),
                            feedback=grading_result.get("feedback", "无评语")
                        )
                        
                        db_answer = await crud_exams.create_student_question_answer(db=db_session, answer_data=answer_data)
                        all_answers.append(db_answer)
                        total_score += db_answer.score
                        logger.info(f"题目 {question.question_number} 评分完成。得分: {db_answer.score}")

                    except Exception as e_q:
                        logger.error(f"处理题目 {question.question_number} 时发生错误: {e_q}")
                        traceback.print_exc()

                logger.info(f"所有题目评分完成。总得分: {total_score}")

                # 6. 调用LLM生成总结报告
                all_feedback = [
                    f"题号 {ans.question.question_number} (满分 {ans.question.max_score}): 得分 {ans.score}, 评语: {ans.feedback}"
                    for ans in all_answers
                ]
                summary_report_text = await deepseek_service.summarize_exam_performance(all_feedback)
                
                if not summary_report_text:
                    summary_report_text = "生成总结报告失败。"
                
                logger.info("总结报告生成完毕。")

                # 7. 保存总结报告
                report_data = schemas.StudentExamReportCreate(
                    student_exam_id=student_exam.id,
                    total_score=total_score,
                    summary_report=summary_report_text
                )
                await crud_exams.create_student_exam_report(db=db_session, report_data=report_data)
                
                logger.info(f"成功保存学生 {student_id} 的试卷报告 (ExamID: {exam_id})。")

            except Exception as e:
                logger.error(f"处理试卷 (ExamID: {exam_id}, StudentID: {student_id}) 时发生严重错误: {e}")
                traceback.print_exc()
                await db_session.rollback()
    
    finally:
        # 8. 清理临时文件
        try:
            for file_path in image_path_list:
                if os.path.exists(file_path):
                    os.remove(file_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
            logger.info(f"已清理临时文件: {temp_dir}")
        except Exception as e:
            logger.error(f"清理临时文件时出错: {e}")