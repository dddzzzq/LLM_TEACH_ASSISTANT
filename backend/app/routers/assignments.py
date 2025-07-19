from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession
import zipfile
import io
import os
from typing import List, Optional

from ..db import crud, database
from ..schemas import models as schemas
from ..services.grading_service import grading_service
from ..services.deepseek_service import deepseek_service
from ..services.plagiarism_service import plagiarism_service

# 定义后端相关路由

# 为作业批改分配路由
router = APIRouter(prefix="/assignments", tags=["作业与评分"])

# 后端成绩处理逻辑
async def grade_and_save_submission(
    db_session: AsyncSession,
    assignment: schemas.AssignmentInDB,
    student_id: str,
    merged_content: str,
    plagiarism_report_for_student: Optional[schemas.PlagiarismReport],
):
    """Grades a single student's submission and saves it to the database."""
    ai_result_data = deepseek_service.grade_homework(
        question=assignment.question,
        rubric=assignment.rubric,
        student_answer=merged_content,
        plagiarism_report=plagiarism_report_for_student,
    )
    submission_data = schemas.SubmissionCreate(
        student_id=student_id,
        score=ai_result_data.get("total_score", -1),
        feedback=ai_result_data.get("overall_feedback", "评分失败"),
        merged_content=merged_content,
        assignment_id=assignment.id,
        plagiarism_report=plagiarism_report_for_student,
    )
    await crud.create_submission(db=db_session, submission=submission_data)
    print(f"已完成对 {student_id} 的评分并存入数据库。")

# 处理批次文件
async def process_batch_file(assignment_id: int, batch_bytes: bytes):
    """Processes a batch ZIP file in a background task."""
    async with database.AsyncSessionLocal() as db_session:
        assignment = await crud.get_assignment(db=db_session, assignment_id=assignment_id)
        if not assignment:
            return print(f"错误：在后台任务中找不到作业ID {assignment_id}")

        try:
            student_texts = {}
            with zipfile.ZipFile(io.BytesIO(batch_bytes), "r") as zip_ref:
                for item_info in zip_ref.infolist():
                    if item_info.is_dir():
                        continue

                    try:
                        student_filename = item_info.filename.encode("cp437").decode("gbk")
                    except:
                        student_filename = item_info.filename

                    if student_filename.startswith("__MACOSX/") or os.path.basename(
                        student_filename
                    ) == ".DS_Store":
                        continue

                    student_id = os.path.splitext(os.path.basename(student_filename))[0]
                    student_file_bytes = zip_ref.read(item_info)
                    merged_content = grading_service.process_archive(
                        student_file_bytes, student_filename
                    )
                    if merged_content.strip():
                        student_texts[student_id] = merged_content

            print("正在对所有提交内容进行两阶段查重...")
            plagiarism_reports = plagiarism_service.check_plagiarism_in_batch(
                student_texts
            )
            print("查重完成。")

            for student_id, merged_content in student_texts.items():
                await grade_and_save_submission(
                    db_session,
                    assignment,
                    student_id,
                    merged_content,
                    plagiarism_reports.get(student_id),
                )
        except Exception as e:
            print(f"处理作业ID {assignment_id} 的批量文件时发生严重错误: {e}")


@router.post("/", response_model=schemas.AssignmentInDB)
async def create_new_assignment(
    assignment: schemas.AssignmentCreate,
    db_session: AsyncSession = Depends(database.get_db),
):
    """Creates a new assignment."""
    return await crud.create_assignment(db=db_session, assignment=assignment)


@router.get("/", response_model=List[schemas.AssignmentInDB])
async def read_all_assignments(
    skip: int = 0, limit: int = 100, db_session: AsyncSession = Depends(database.get_db)
):
    """Reads all assignments with pagination."""
    return await crud.get_assignments(db=db_session, skip=skip, limit=limit)


@router.get("/{assignment_id}", response_model=schemas.AssignmentWithSubmissions)
async def read_single_assignment(
    assignment_id: int, db_session: AsyncSession = Depends(database.get_db)
):
    """Reads a single assignment with its submissions."""
    db_assignment = await crud.get_assignment(db=db_session, assignment_id=assignment_id)
    if not db_assignment:
        raise HTTPException(status_code=404, detail="未找到该作业")
    return db_assignment


@router.get("/{assignment_id}/results", response_model=List[schemas.SubmissionInDB])
async def read_assignment_results(
    assignment_id: int, db_session: AsyncSession = Depends(database.get_db)
):
    """Reads all submission results for a specific assignment."""
    return await crud.get_submissions_for_assignment(
        db=db_session, assignment_id=assignment_id
    )


@router.post("/{assignment_id}/submit", status_code=202)
async def submit_and_grade_batch(
    assignment_id: int,
    background_tasks: BackgroundTasks,
    batch_file: UploadFile = File(...),
    db_session: AsyncSession = Depends(database.get_db),
):
    """Accepts a ZIP file of submissions and queues them for background processing."""
    assignment = await crud.get_assignment(db=db_session, assignment_id=assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="未找到该作业")

    if not batch_file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400, detail="请上传一个包含所有学生提交内容的ZIP压缩包。"
        )

    batch_bytes = await batch_file.read()
    background_tasks.add_task(process_batch_file, assignment_id, batch_bytes)
    return {"message": "已收到批量提交文件，正在后台处理中。您可稍后刷新查看结果。"}


@router.delete("/{assignment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_single_assignment(
    assignment_id: int, db_session: AsyncSession = Depends(database.get_db)
):
    """Deletes an entire assignment and all its related submissions."""
    deleted_assignment = await crud.delete_assignment(
        db=db_session, assignment_id=assignment_id
    )
    if not deleted_assignment:
        raise HTTPException(status_code=404, detail="未找到该作业")
    return


@router.delete("/{assignment_id}/results", status_code=status.HTTP_200_OK)
async def delete_all_submissions(
    assignment_id: int, db_session: AsyncSession = Depends(database.get_db)
):
    """Deletes all graded results under a specific assignment."""
    assignment = await crud.get_assignment(db=db_session, assignment_id=assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="未找到该作业")

    deleted_count = await crud.delete_all_submissions_for_assignment(
        db=db_session, assignment_id=assignment_id
    )
    return {"message": f"成功删除 {deleted_count} 条评分记录。"}


@router.delete("/submissions/{submission_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_single_submission(
    submission_id: int, db_session: AsyncSession = Depends(database.get_db)
):
    """Deletes a single student's submission record."""
    deleted_submission = await crud.delete_submission(
        db=db_session, submission_id=submission_id
    )
    if not deleted_submission:
        raise HTTPException(status_code=404, detail="未找到该提交记录")
    return