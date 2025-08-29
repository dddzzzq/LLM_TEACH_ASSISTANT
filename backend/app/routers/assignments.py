from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import zipfile
import io
import os
import traceback
from typing import List, Optional
from tqdm import tqdm
import pandas as pd
from urllib.parse import quote

from ..db import crud, database
from ..schemas import models as schemas
from ..services.grading_service import grading_service
from ..services.deepseek_service import deepseek_service
from ..services.plagiarism_service import plagiarism_service
from ..services.aigc_service import aigc_detector_service

# 用于处理 /assignments/* 的路由
router = APIRouter(prefix="/assignments", tags=["作业与评分"])
# 用于处理 /submissions/* 的路由
submission_router = APIRouter(prefix="/submissions", tags=["学生提交"])

async def process_batch_file(assignment_id: int, batch_bytes: bytes):
    """处理上交的batch压缩文件"""
    async with database.AsyncSessionLocal() as db_session:
        assignment = await crud.get_assignment(db=db_session, assignment_id=assignment_id)
        if not assignment:
            print(f"错误：在后台任务中找不到作业ID {assignment_id}")
            return

        try:
            # 第一步处理压缩文件进行内容提取
            student_texts = {}
            with zipfile.ZipFile(io.BytesIO(batch_bytes), "r") as zip_ref:
                for item_info in tqdm(zip_ref.infolist(), desc="步骤 1/5: 解压并提取文件"):
                    if item_info.is_dir(): continue
                    try: student_filename = item_info.filename.encode("cp437").decode("gbk")
                    except: student_filename = item_info.filename
                    if student_filename.startswith("__MACOSX/") or os.path.basename(student_filename) == ".DS_Store": continue
                    student_id = os.path.splitext(os.path.basename(student_filename))[0]
                    if student_id not in student_texts: student_texts[student_id] = ""
                    file_content = grading_service.process_archive(zip_ref.read(item_info), student_filename)
                    student_texts[student_id] += f"--- 文件开始: {student_filename} ---\n\n{file_content}\n\n--- 文件结束: {student_filename} ---\n\n"

            print("步骤 2/5: 初步查重与内容分离...")
            plagiarism_results = plagiarism_service.check_plagiarism_in_batch(student_texts)
            suspicious_text_pairs = plagiarism_results["suspicious_text_pairs"]
            suspicious_code_pairs = plagiarism_results["suspicious_code_pairs"]
            separated_contents = plagiarism_results["separated_contents"]
            print(f"初步查重完成。发现 {len(suspicious_text_pairs)} 对可疑文本, {len(suspicious_code_pairs)} 对可疑代码。")

            print("步骤 3/5: 对可疑配对进行LLM深度分析...")
            llm_analysis_cache = {}
            all_suspicious_pairs = [(*pair, 'text') for pair in suspicious_text_pairs] + [(*pair, 'code') for pair in suspicious_code_pairs]
            
            for s1, s2, initial_score, content_type in tqdm(all_suspicious_pairs, desc="步骤 3/5: LLM深度查重分析"):
                content1 = separated_contents.get(s1, {}).get(content_type)
                content2 = separated_contents.get(s2, {}).get(content_type)
                if not content1 or not content2: continue
                
                llm_analysis = deepseek_service.analyze_plagiarism(content1, content2, content_type)
                if llm_analysis:
                    llm_analysis_cache[(s1, s2, content_type)] = {'initial_score': initial_score, 'llm_analysis': llm_analysis}
            print("LLM深度分析完成。")

            print("步骤 4/5: 对所有提交内容进行AIGC检测...")
            aigc_reports = {
                sid: aigc_detector_service.detect(content) 
                for sid, content in tqdm(student_texts.items(), desc="步骤 4/5: AIGC内容检测")
            }
            print("AIGC检测完成。")

            print("整理报告并准备评分...")
            final_student_data = {
                sid: {
                    "plagiarism_reports": [],
                    "aigc_report": schemas.AIGCReport(**aigc_reports[sid]) if "error" not in aigc_reports[sid] else None
                } for sid in student_texts.keys()
            }

            for (s1, s2, content_type), analysis_result in llm_analysis_cache.items():
                report_for_s1 = schemas.PlagiarismReport(
                    similar_to=s2,
                    content_type=content_type,
                    initial_score=analysis_result['initial_score'],
                    llm_analysis=analysis_result['llm_analysis']
                )
                final_student_data[s1]["plagiarism_reports"].append(report_for_s1)
                
                report_for_s2 = schemas.PlagiarismReport(
                    similar_to=s1,
                    content_type=content_type,
                    initial_score=analysis_result['initial_score'],
                    llm_analysis=analysis_result['llm_analysis']
                )
                final_student_data[s2]["plagiarism_reports"].append(report_for_s2)

            print("步骤 5/5: 开始逐一评分并保存...")
            for student_id, merged_content in tqdm(student_texts.items(), desc="步骤 5/5: LLM评分并保存"):
                student_reports = final_student_data[student_id]
                
                ai_result = deepseek_service.grade_homework(
                    question=assignment.question, rubric=assignment.rubric, student_answer=merged_content,
                    plagiarism_reports=student_reports["plagiarism_reports"], 
                    aigc_report=student_reports["aigc_report"]
                )
                
                submission_data = schemas.SubmissionCreate(
                    student_id=student_id, score=ai_result.get("total_score", -1),
                    feedback=ai_result.get("overall_feedback", "评分失败"), merged_content=merged_content,
                    assignment_id=assignment_id, 
                    plagiarism_reports=student_reports["plagiarism_reports"],
                    aigc_report=student_reports["aigc_report"],
                    is_human_reviewed=False,
                    human_feedback=None,
                    human_score=0
                )
                await crud.create_submission(db=db_session, submission=submission_data)
            
            print("所有学生的作业处理完成并已存入数据库。")

        except Exception as e:
            print(f"处理作业ID {assignment_id} 的批量文件时发生严重错误: {e}")
            traceback.print_exc()


# 创建任务路由
@router.post("/", response_model=schemas.AssignmentInDB)
async def create_new_assignment(
    assignment: schemas.AssignmentCreate,
    db_session: AsyncSession = Depends(database.get_db),
):
    return await crud.create_assignment(db=db_session, assignment=assignment)

# 读取全部任务路由，用于显示作业列表
@router.get("/", response_model=List[schemas.AssignmentInDB])
async def read_all_assignments(
    skip: int = 0, limit: int = 200, db_session: AsyncSession = Depends(database.get_db)
):
    return await crud.get_assignments(db=db_session, skip=skip, limit=limit)

# 读取某一作业路由
@router.get("/{assignment_id}", response_model=schemas.AssignmentWithSubmissions)
async def read_single_assignment(
    assignment_id: int, db_session: AsyncSession = Depends(database.get_db)
):
    db_assignment = await crud.get_assignment(db=db_session, assignment_id=assignment_id)
    if not db_assignment:
        raise HTTPException(status_code=404, detail="未找到该作业")
    return db_assignment

# 导出excel格式
@router.get("/{assignment_id}/export", response_class=StreamingResponse)
async def export_assignment_results(
    assignment_id: int, db_session: AsyncSession = Depends(database.get_db)
):
    assignment = await crud.get_assignment(db=db_session, assignment_id=assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="作业未找到")
    
    results = await crud.get_submissions_for_assignment(
        db = db_session, assignment_id=assignment_id
    )

    if not results:
        raise HTTPException(status_code=404, detail="该作业没有任何评分结果可以导出")
    
    # 将作业转换为字典列表，方便处理
    data_to_export = []
    for res in results:
        final_score = res.human_score if res.is_human_reviewed and res.human_score is not None else res.score

        # 简化查重报告
        max_plagiarism_score = 0
        if res.plagiarism_reports:
            scores = [r.get('llm_analysis', {}).get('similarity_score', 0) for r in res.plagiarism_reports]
            if scores:
                max_plagiarism_score = max(scores)

        # 简化AIGC报告
        aigc_risk = "未检测"
        if res.aigc_report:
            prob = res.aigc_report.get('ai_probability', 0)
            source = res.aigc_report.get('detection_source', '')
            aigc_risk = f"{prob:.1f}% AI生成 ({source})" if source else f"{prob:.1f}% AI生成"


        data_to_export.append({
            "学生ID": res.student_id,
            "AI评分": res.score,
            "AI评语": res.feedback,
            "是否人工复核": "是" if res.is_human_reviewed else "否",
            "人工评分": res.human_score if res.is_human_reviewed else "",
            "人工评语": res.human_feedback if res.is_human_reviewed else "",
            "最终得分": final_score,
            "最高抄袭风险(LLM)": f"{max_plagiarism_score}分" if max_plagiarism_score > 0 else "无风险",
            "AIGC风险": aigc_risk,
            "提交内容摘要": res.merged_content[:200] + "..." if res.merged_content and len(res.merged_content) > 200 else res.merged_content
        })

    # 2. 使用 Pandas 创建 DataFrame
    df = pd.DataFrame(data_to_export)

    # 3. 创建一个内存中的二进制流来保存Excel文件
    output_stream = io.BytesIO()
    df.to_excel(output_stream, index=False, sheet_name='评分结果')
    output_stream.seek(0) # 将指针移到流的开头

    # 4. 设置HTTP头，告诉浏览器这是一个需要下载的文件
    filename = f"{assignment.task_name}_评分结果.xlsx"
    headers = {
        'Content-Disposition': f"attachment; filename*=UTF-8''{quote(filename)}"
    }

    return StreamingResponse(
        output_stream,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )

# 读取单一作业详情
@router.get("/{assignment_id}/results", response_model=List[schemas.SubmissionInDB])
async def read_assignment_results(
    assignment_id: int, db_session: AsyncSession = Depends(database.get_db)
):
    return await crud.get_submissions_for_assignment(
        db=db_session, assignment_id=assignment_id
    )

# 提交单个提交路由
@router.post("/{assignment_id}/submit", status_code=202)
async def submit_and_grade_batch(
    assignment_id: int,
    background_tasks: BackgroundTasks,
    batch_file: UploadFile = File(...),
    db_session: AsyncSession = Depends(database.get_db),
):
    assignment = await crud.get_assignment(db=db_session, assignment_id=assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="未找到该作业")
    if not batch_file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="请上传一个包含所有学生提交内容的ZIP压缩包。")
    batch_bytes = await batch_file.read()
    background_tasks.add_task(process_batch_file, assignment_id, batch_bytes)
    return {"message": "已收到批量提交文件，正在后台处理中。您可稍后刷新查看结果。"}

# 删除单个作业路由
@router.delete("/{assignment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_single_assignment(
    assignment_id: int, db_session: AsyncSession = Depends(database.get_db)
):
    deleted_assignment = await crud.delete_assignment(db=db_session, assignment_id=assignment_id)
    if not deleted_assignment:
        raise HTTPException(status_code=404, detail="未找到该作业")
    return

# 删除单个作业所有结果
@router.delete("/{assignment_id}/results", status_code=status.HTTP_200_OK)
async def delete_all_submissions(
    assignment_id: int, db_session: AsyncSession = Depends(database.get_db)
):
    assignment = await crud.get_assignment(db=db_session, assignment_id=assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="未找到该作业")
    deleted_count = await crud.delete_all_submissions_for_assignment(db=db_session, assignment_id=assignment_id)
    return {"message": f"成功删除 {deleted_count} 条评分记录。"}

# 使用和提交有关的路由删除和更新单条记录
@submission_router.put("/{submission_id}", response_model=schemas.SubmissionInDB)
async def review_and_update_submission(
    submission_id: int,
    submission_update: schemas.SubmissionUpdate,
    db_session: AsyncSession = Depends(database.get_db),
):
    updated_submission = await crud.update_submission(
        db=db_session, 
        submission_id=submission_id, 
        submission_update=submission_update
    )
    if not updated_submission:
        raise HTTPException(status_code=404, detail="未找到该提交记录")
    return updated_submission

@submission_router.delete("/{submission_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_single_submission(
    submission_id: int, db_session: AsyncSession = Depends(database.get_db)
):
    deleted_submission = await crud.delete_submission(db=db_session, submission_id=submission_id)
    if not deleted_submission:
        raise HTTPException(status_code=404, detail="未找到该提交记录")
