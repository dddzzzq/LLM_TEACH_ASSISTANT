from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload, joinedload
from typing import List, Optional, Dict, Any

from . import models
from ..schemas import models as schemas

#  Exam (试卷) 

async def create_exam(db: AsyncSession, exam: schemas.ExamCreate) -> models.Exam:
    db_exam = models.Exam(name=exam.name, question_count=0, total_score=0.0) # 初始化
    db.add(db_exam)
    await db.commit()
    await db.refresh(db_exam)
    return db_exam

async def get_exam(db: AsyncSession, exam_id: int) -> Optional[models.Exam]:
    result = await db.execute(
        select(models.Exam)
        .options(selectinload(models.Exam.questions))
        .filter(models.Exam.id == exam_id)
    )
    return result.scalars().first()

async def get_exams(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[models.Exam]:
    result = await db.execute(
        select(models.Exam)
        .order_by(models.Exam.id.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()

async def delete_exam(db: AsyncSession, exam_id: int) -> Optional[models.Exam]:
    exam = await db.get(models.Exam, exam_id)
    if exam:
        await db.delete(exam)
        await db.commit()
        return exam
    return None

#  ExamQuestion (试卷题目) 

async def create_exam_question(db: AsyncSession, exam_id: int, question: schemas.ExamQuestionCreate) -> models.ExamQuestion:
    db_question = models.ExamQuestion(
        exam_id=exam_id,
        question_number=question.question_number,
        question_text=question.question_text,
        standard_answer=question.standard_answer,
        rubric=question.rubric,
        max_score=question.max_score
    )
    db.add(db_question)
    
    # 更新试卷题目总数和总分
    exam = await db.get(models.Exam, exam_id)
    if exam:
        exam.question_count = (exam.question_count or 0) + 1
        exam.total_score = (exam.total_score or 0.0) + question.max_score # < 累加总分
        
    await db.commit()
    await db.refresh(db_question)
    return db_question

async def get_exam_questions(db: AsyncSession, exam_id: int) -> List[models.ExamQuestion]:
    result = await db.execute(
        select(models.ExamQuestion)
        .filter(models.ExamQuestion.exam_id == exam_id)
        .order_by(models.ExamQuestion.question_number)
    )
    return result.scalars().all()

#  StudentExam (学生试卷) 

async def create_student_exam(db: AsyncSession, exam_id: int, student_id: str) -> models.StudentExam:
    """
    创建或获取一个学生的试卷实例
    """
    result = await db.execute(
        select(models.StudentExam)
        .options(
            selectinload(models.StudentExam.report),
            selectinload(models.StudentExam.answers),
            selectinload(models.StudentExam.images) 
        )
        .filter_by(exam_id=exam_id, student_id=student_id)
    )
    db_student_exam = result.scalars().first()
    
    if db_student_exam:
        if db_student_exam.report:
            await db.delete(db_student_exam.report)
        for answer in db_student_exam.answers:
            await db.delete(answer)
        for img in db_student_exam.images:
            await db.delete(img)
            
        await db.commit()
    else:
        db_student_exam = models.StudentExam(exam_id=exam_id, student_id=student_id)
        db.add(db_student_exam)
        await db.commit()
        await db.refresh(db_student_exam)
        
    return db_student_exam

#  保存学生试卷图片记录，支持关联题目ID 
async def create_student_exam_image(db: AsyncSession, student_exam_id: int, image_path: str, exam_question_id: Optional[int] = None) -> models.StudentExamImage:
    db_image = models.StudentExamImage(
        student_exam_id=student_exam_id,
        image_path=image_path,
        exam_question_id=exam_question_id 
    )
    db.add(db_image)
    await db.commit()
    await db.refresh(db_image)
    return db_image

#  StudentQuestionAnswer (学生答案) 

async def create_student_question_answer(db: AsyncSession, answer_data: schemas.StudentQuestionAnswerCreate) -> models.StudentQuestionAnswer:
    db_answer = models.StudentQuestionAnswer(
        student_exam_id=answer_data.student_exam_id,
        exam_question_id=answer_data.exam_question_id,
        student_answer_text=answer_data.student_answer_text,
        score=answer_data.score,
        feedback=answer_data.feedback
    )
    db.add(db_answer)
    await db.commit()
    await db.refresh(db_answer)
    return db_answer

#  StudentExamReport (学生报告) 

async def create_student_exam_report(db: AsyncSession, report_data: schemas.StudentExamReportCreate) -> models.StudentExamReport:
    db_report = models.StudentExamReport(
        student_exam_id=report_data.student_exam_id,
        total_score=report_data.total_score,
        summary_report=report_data.summary_report
    )
    db.add(db_report)
    await db.commit()
    await db.refresh(db_report)
    return db_report

async def get_student_exam_reports(db: AsyncSession, exam_id: int) -> List[Dict[str, Any]]:
    """
    获取某次试卷所有学生的简要报告（学号和总分）
    """
    result = await db.execute(
        select(models.StudentExam.student_id, models.StudentExamReport.total_score, models.StudentExamReport.student_exam_id)
        .join(models.StudentExamReport, models.StudentExam.id == models.StudentExamReport.student_exam_id)
        .filter(models.StudentExam.exam_id == exam_id)
    )
    reports = result.all()
    return [
        {"student_id": r.student_id, "total_score": r.total_score, "student_exam_id": r.student_exam_id}
        for r in reports
    ]

async def get_student_detailed_report(db: AsyncSession, student_exam_id: int) -> Optional[Dict[str, Any]]:
    """
    获取单个学生的详细报告，包括总结和所有题目的得分详情
    修改说明：增加了对 student_exam.images 的预加载，并将其加入返回结果
    """
    # 1. 获取报告，同时加载关联的 student_exam 和 student_exam 里的 images
    report_result = await db.execute(
        select(models.StudentExamReport)
        .options(
            # 加载 student_exam
            joinedload(models.StudentExamReport.student_exam)
            # 进一步加载 student_exam 下的 images (关键修改)
            .selectinload(models.StudentExam.images)
        )
        .filter(models.StudentExamReport.student_exam_id == student_exam_id)
    )
    report = report_result.scalars().first()
    
    if not report:
        return None

    # 2. 获取每道题的回答详情
    answers_result = await db.execute(
        select(models.StudentQuestionAnswer)
        .options(joinedload(models.StudentQuestionAnswer.question))
        .filter(models.StudentQuestionAnswer.student_exam_id == student_exam_id)
        .order_by(models.StudentQuestionAnswer.exam_question_id)
    )
    answers = answers_result.scalars().all()
    
    # 3. 构建符合 Pydantic Schema (StudentExamDetailedReport) 的字典
    return {
        "report": report,
        "answers": answers,
        "student_id": report.student_exam.student_id,
        "exam_id": report.student_exam.exam_id,
        # 显式将加载出来的图片列表传给 schema 中的 images 字段
        "images": report.student_exam.images 
    }

async def delete_student_exam(db: AsyncSession, student_exam_id: int) -> bool:
    student_exam = await db.get(models.StudentExam, student_exam_id)
    if student_exam:
        await db.delete(student_exam)
        await db.commit()
        return True
    return False