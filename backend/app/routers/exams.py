import asyncio
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    status
)
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from ..db import database, crud_exams
from ..schemas import models as schemas
from ..services import exam_grading_service

router = APIRouter(prefix="/exams", tags=["试卷与评分"])

#  试卷 (Exam) 

@router.post("/", response_model=schemas.ExamInDB, status_code=status.HTTP_201_CREATED)
async def create_new_exam(
    exam: schemas.ExamCreate,
    db_session: AsyncSession = Depends(database.get_db),
):
    """
    创建一个新的试卷（例如 "2022年期末试卷"）
    """
    return await crud_exams.create_exam(db=db_session, exam=exam)

@router.get("/", response_model=List[schemas.ExamInDB])
async def read_all_exams(
    skip: int = 0,
    limit: int = 100,
    db_session: AsyncSession = Depends(database.get_db),
):
    """
    获取所有试卷的列表
    """
    return await crud_exams.get_exams(db=db_session, skip=skip, limit=limit)

@router.get("/{exam_id}", response_model=schemas.ExamWithQuestions)
async def read_single_exam_with_questions(
    exam_id: int,
    db_session: AsyncSession = Depends(database.get_db),
):
    """
    获取单个试卷的详细信息，包含所有题目
    """
    db_exam = await crud_exams.get_exam(db=db_session, exam_id=exam_id)
    if not db_exam:
        raise HTTPException(status_code=404, detail="未找到该试卷")
    
    # crud_exams.get_exam 已经 eager load 了 questions
    return db_exam

@router.delete("/{exam_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_single_exam(
    exam_id: int,
    db_session: AsyncSession = Depends(database.get_db),
):
    """
    删除一个试卷及其所有相关数据
    """
    deleted_exam = await crud_exams.delete_exam(db=db_session, exam_id=exam_id)
    if not deleted_exam:
        raise HTTPException(status_code=404, detail="未找到该试卷")
    return None # 204 No Content

#  试卷题目 (ExamQuestion) 

@router.post("/{exam_id}/questions", response_model=schemas.ExamQuestionInDB, status_code=status.HTTP_201_CREATED)
async def add_question_to_exam(
    exam_id: int,
    question: schemas.ExamQuestionCreate,
    db_session: AsyncSession = Depends(database.get_db),
):
    """
    为指定ID的试卷添加一道题目
    """
    db_exam = await crud_exams.get_exam(db=db_session, exam_id=exam_id)
    if not db_exam:
        raise HTTPException(status_code=404, detail="未找到该试卷")
    
    return await crud_exams.create_exam_question(db=db_session, exam_id=exam_id, question=question)

#  试卷评分 (Grading) 

@router.post("/{exam_id}/grade_submission", status_code=status.HTTP_202_ACCEPTED)
async def grade_student_exam_submission(
    exam_id: int,
    background_tasks: BackgroundTasks,
    student_id: str = Form(...),
    images: List[UploadFile] = File(...),
    db_session: AsyncSession = Depends(database.get_db),
):
    """
    上传一个学生的多张试卷图片，并开始后台评分
    """
    db_exam = await crud_exams.get_exam(db=db_session, exam_id=exam_id)
    if not db_exam:
        raise HTTPException(status_code=404, detail="未找到该试卷")

    if not images:
        raise HTTPException(status_code=400, detail="没有上传任何图片")

    # 立即读取所有图片内容，因为 UploadFile 对象在后台任务中可能失效
    image_bytes_list = []
    for image in images:
        if not image.content_type or not image.content_type.startswith("image/"):
             raise HTTPException(status_code=400, detail=f"文件 {image.filename} 不是图片格式")
        image_bytes_list.append(await image.read())

    # 添加后台任务
    background_tasks.add_task(
        exam_grading_service.process_exam_submission, 
        exam_id=exam_id, 
        student_id=student_id, 
        image_bytes_list=image_bytes_list
    )

    return {"message": f"已收到学生 {student_id} 的试卷，正在后台处理中。"}


@router.get("/{exam_id}/results", response_model=List[schemas.StudentExamResultSummary])
async def get_exam_results_summary(
    exam_id: int,
    db_session: AsyncSession = Depends(database.get_db),
):
    """
    获取某次试卷所有学生的简要成绩列表（学号, 总分）
    """
    db_exam = await crud_exams.get_exam(db=db_session, exam_id=exam_id)
    if not db_exam:
        raise HTTPException(status_code=404, detail="未找到该试卷")
        
    results = await crud_exams.get_student_exam_reports(db=db_session, exam_id=exam_id)
    return results

@router.get("/{exam_id}/results/{student_exam_id}", response_model=schemas.StudentExamDetailedReport)
async def get_student_detailed_report(
    exam_id: int, # exam_id 仅用于路由匹配，实际查询用 student_exam_id
    student_exam_id: int,
    db_session: AsyncSession = Depends(database.get_db),
):
    """
    获取单个学生的详细评分报告（总结、每题得分、评语等）
    """
    report_data = await crud_exams.get_student_detailed_report(db=db_session, student_exam_id=student_exam_id)
    
    if not report_data:
        raise HTTPException(status_code=404, detail="未找到该学生的评分报告")
        
    if report_data["exam_id"] != exam_id:
        raise HTTPException(status_code=400, detail="报告ID与试卷ID不匹配")

    # Pydantic 模型会自动处理 report 和 answers
    return report_data

#  新增删除路由 
@router.delete("/{exam_id}/results/{student_exam_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_student_exam_result(
    exam_id: int,
    student_exam_id: int,
    db_session: AsyncSession = Depends(database.get_db),
):
    """
    删除特定学生的试卷评分结果
    """
    # 也可以先检查 exam_id 是否匹配，这里简化处理直接根据 student_exam_id 删除
    success = await crud_exams.delete_student_exam(db=db_session, student_exam_id=student_exam_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="未找到该评分记录")
    
    return None