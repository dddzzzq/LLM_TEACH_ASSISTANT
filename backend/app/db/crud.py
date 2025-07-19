from typing import List, Optional

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from ..schemas import models as schemas
from . import models

async def create_assignment(db: AsyncSession, assignment: schemas.AssignmentCreate) -> models.Assignment:
    """
    在数据库中创建一个新的作业任务。

    Args:
        db (AsyncSession): 数据库会话对象。
        assignment (schemas.AssignmentCreate): 包含新作业信息的Pydantic模型。

    Returns:
        models.Assignment: 新创建的、包含数据库生成ID的作业对象。
    """
    db_assignment = models.Assignment(
        task_name=assignment.task_name,
        question=assignment.question,
        rubric=assignment.rubric
    )
    db.add(db_assignment)
    await db.commit()
    await db.refresh(db_assignment)
    return db_assignment


async def get_assignment(db: AsyncSession, assignment_id: int) -> Optional[models.Assignment]:
    """
    根据ID从数据库中获取单个作业任务及其所有提交记录。

    Args:
        db (AsyncSession): 数据库会话对象。
        assignment_id (int): 作业任务的ID。

    Returns:
        Optional[models.Assignment]: 找到的作业对象，如果不存在则返回None。
    """
    result = await db.execute(
        select(models.Assignment)
        .options(selectinload(models.Assignment.submissions))
        .filter(models.Assignment.id == assignment_id)
    )
    return result.scalars().first()


async def get_assignments(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[models.Assignment]:
    """
    从数据库中获取作业任务列表（不包含提交记录）。

    Args:
        db (AsyncSession): 数据库会话对象。
        skip (int): 跳过的记录数。
        limit (int): 返回的最大记录数。

    Returns:
        List[models.Assignment]: 作业对象列表。
    """
    result = await db.execute(
        select(models.Assignment)
        .order_by(models.Assignment.id.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


async def delete_assignment(db: AsyncSession, assignment_id: int) -> Optional[models.Assignment]:
    """
    根据ID从数据库中删除一个作业任务及其所有关联的提交记录。

    Args:
        db (AsyncSession): 数据库会话对象。
        assignment_id (int): 要删除的作业任务的ID。

    Returns:
        Optional[models.Assignment]: 被删除的作业对象，如果未找到则返回None。
    """
    assignment = await get_assignment(db, assignment_id)
    if assignment:
        await db.delete(assignment)
        await db.commit()
        return assignment
    return None


async def create_submission(db: AsyncSession, submission: schemas.SubmissionCreate) -> models.Submission:
    """
    在数据库中为某个作业任务创建一个新的提交记录。

    Args:
        db (AsyncSession): 数据库会话对象。
        submission (schemas.SubmissionCreate): 包含新提交记录信息的Pydantic模型。

    Returns:
        models.Submission: 新创建的提交记录对象。
    """
    db_submission = models.Submission(
        student_id=submission.student_id,
        score=submission.score,
        feedback=submission.feedback,
        merged_content=submission.merged_content,
        assignment_id=submission.assignment_id,
        plagiarism_report=submission.plagiarism_report
    )
    db.add(db_submission)
    await db.commit()
    await db.refresh(db_submission)
    return db_submission


async def get_submissions_for_assignment(db: AsyncSession, assignment_id: int) -> List[models.Submission]:
    """
    获取指定作业任务下的所有提交记录。

    Args:
        db (AsyncSession): 数据库会话对象。
        assignment_id (int): 作业任务的ID。

    Returns:
        List[models.Submission]: 该作业下的提交记录对象列表。
    """
    result = await db.execute(
        select(models.Submission)
        .filter(models.Submission.assignment_id == assignment_id)
        .order_by(models.Submission.id)
    )
    return result.scalars().all()


async def delete_submission(db: AsyncSession, submission_id: int) -> Optional[models.Submission]:
    """
    根据ID从数据库中删除单个提交记录。

    Args:
        db (AsyncSession): 数据库会话对象。
        submission_id (int): 要删除的提交记录的ID。

    Returns:
        Optional[models.Submission]: 被删除的提交记录对象，如果未找到则返回None。
    """
    result = await db.execute(select(models.Submission).filter(models.Submission.id == submission_id))
    submission = result.scalars().first()
    if submission:
        await db.delete(submission)
        await db.commit()
        return submission
    return None


async def delete_all_submissions_for_assignment(db: AsyncSession, assignment_id: int) -> int:
    """
    一键清空（删除）指定作业下的所有提交记录。

    Args:
        db (AsyncSession): 数据库会话对象。
        assignment_id (int): 作业任务的ID。

    Returns:
        int: 被删除的记录数量。
    """
    stmt = delete(models.Submission).where(models.Submission.assignment_id == assignment_id)
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount
