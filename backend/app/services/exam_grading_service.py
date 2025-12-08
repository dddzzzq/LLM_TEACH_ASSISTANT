import asyncio
import traceback
import json
import logging
import os
import uuid
import shutil
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession

from .ocr_service import ocr_service_instance
from .deepseek_service import deepseek_service
from ..db import database, crud_exams
from ..schemas import models as schemas

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义统一的上传目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # backend/
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

async def process_exam_submission(exam_id: int, student_id: str, image_bytes_list: List[bytes]):
    """
    后台任务：处理学生提交的试卷图片
    1. 保存图片
    2. 逐张OCR
    3. 识别每张图片对应的题号
    4. 存入数据库（关联题号）
    5. 汇总所有文本进行评分
    """
    logger.info(f"开始处理试卷 (ExamID: {exam_id}, StudentID: {student_id})")
    
    if not ocr_service_instance:
        logger.error("OCR服务未初始化，无法处理试卷")
        return

    exam_upload_dir = os.path.join(UPLOAD_DIR, "exams", str(exam_id), student_id)
    os.makedirs(exam_upload_dir, exist_ok=True)

    try:
        async with database.AsyncSessionLocal() as db_session:
            # 1. 获取试卷信息（需要题目列表来辅助LLM判断图片归属）
            questions = await crud_exams.get_exam_questions(db=db_session, exam_id=exam_id)
            if not questions:
                logger.warning(f"试卷 (ExamID: {exam_id}) 没有任何题目，无法评分。")
                return
            
            # 构建题目列表字符串，供LLM参考
            question_list_str = "\n".join([f"题号 {q.question_number}: {q.question_text[:50]}..." for q in questions])

            # 2. 创建 StudentExam 实例
            student_exam = await crud_exams.create_student_exam(
                db=db_session, 
                exam_id=exam_id, 
                student_id=student_id
            )

            full_ocr_text = "" # 汇总所有文本用于最后的详细评分

            # 3. 处理每一张图片
            for i, img_bytes in enumerate(image_bytes_list):
                #  保存文件 
                file_name = f"{uuid.uuid4()}.png"
                file_path = os.path.join(exam_upload_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(img_bytes)
                
                relative_url = f"/uploads/exams/{exam_id}/{student_id}/{file_name}"
                
                #  单张图片 OCR 
                # 以前是最后一次性OCR，现在因为要判断每张图属于哪道题，必须逐张做
                page_ocr_text = ocr_service_instance.get_concatenated_text([file_path])
                logger.info(f"图片 {i+1} OCR 完成，长度: {len(page_ocr_text)}")
                
                #  累加到总文本 
                full_ocr_text += f"\n[图片{i+1}内容]:\n{page_ocr_text}\n"

                #  LLM 判别题号 
                # 只有当识别出文本时才去问LLM，省点Token
                target_question_id = None
                if len(page_ocr_text.strip()) > 5:
                    predicted_q_num = await deepseek_service.identify_question_number(page_ocr_text, question_list_str)
                    logger.info(f"图片 {i+1} 识别为第 {predicted_q_num} 题")
                    
                    # 找到对应的数据库 question_id
                    if predicted_q_num > 0:
                        for q in questions:
                            if q.question_number == predicted_q_num:
                                target_question_id = q.id
                                break
                
                #  保存图片记录到数据库 
                await crud_exams.create_student_exam_image(
                    db=db_session,
                    student_exam_id=student_exam.id,
                    image_path=relative_url,
                    exam_question_id=target_question_id # 关联识别出的题目ID
                )

            # 4. 评分流程 (保持原逻辑，使用 full_ocr_text)
            logger.info(f"开始评分流程，全卷OCR长度: {len(full_ocr_text)}")
            all_answers = []
            total_score = 0.0

            for question in questions:
                try:
                    grading_result = await deepseek_service.grade_exam_question(
                        question=question.question_text,
                        standard_answer=question.standard_answer,
                        rubric=question.rubric,
                        max_score=question.max_score,
                        full_student_text=full_ocr_text
                    )
                    
                    if not grading_result:
                        grading_result = {"score": 0.0, "feedback": "LLM评分失败", "student_answer_extracted": "未能提取答案"}

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

                except Exception as e_q:
                    logger.error(f"处理题目 {question.question_number} 时发生错误: {e_q}")
                    traceback.print_exc()

            # 5. 总结报告
            all_feedback = [
                f"题号 {ans.question.question_number} (满分 {ans.question.max_score}): 得分 {ans.score}, 评语: {ans.feedback}"
                for ans in all_answers
            ]
            summary_report_text = await deepseek_service.summarize_exam_performance(all_feedback)
            
            if not summary_report_text:
                summary_report_text = "生成总结报告失败。"

            # 6. 保存总结
            report_data = schemas.StudentExamReportCreate(
                student_exam_id=student_exam.id,
                total_score=total_score,
                summary_report=summary_report_text
            )
            await crud_exams.create_student_exam_report(db=db_session, report_data=report_data)
            logger.info("试卷处理完成。")

    except Exception as e:
        logger.error(f"处理试卷异常: {e}")
        traceback.print_exc()