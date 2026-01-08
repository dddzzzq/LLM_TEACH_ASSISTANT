import re
import json
import time
import os
import traceback
import asyncio
from typing import List, Optional
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
import pandas as pd
from urllib.parse import quote
import pprint
from tqdm import tqdm

from ..db import crud, database
from ..schemas import models as schemas
from ..services.grading_service import grading_service
from ..services.deepseek_service import deepseek_service
from ..services.plagiarism_service import plagiarism_service
from ..services.aigc_service import aigc_detector_service

router = APIRouter(prefix="/assignments", tags=["作业与评分"])
submission_router = APIRouter(prefix="/submissions", tags=["学生提交"])

async def process_batch_file(assignment_id: int, batch_bytes: bytes):
    # 记录批处理开始时间和初始化日志数据
    batch_start_time = time.time()
    batch_log_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "assignment_id": assignment_id,
        "status": "processing",
        "student_count": 0,
        "total_processing_time_seconds": 0,
        "step1_unzip_time_seconds": 0,
        "step2_plagiarism_check_time_seconds": 0,
        "step3_code_doc_match_time_seconds": 0,
        "step4_llm_analysis_time_seconds": 0,
        "step5_aigc_detect_time_seconds": 0,
        "step6_grading_time_seconds": 0,
        "error_message": None
    }
    
    async with database.AsyncSessionLocal() as db_session:
        assignment = await crud.get_assignment(db=db_session, assignment_id=assignment_id)
        if not assignment:
            error_msg = f"错误：在后台任务中找不到作业ID {assignment_id}"
            print(error_msg)
            # 记录失败日志
            batch_log_data["status"] = "failed"
            batch_log_data["error_message"] = error_msg
            _save_batch_log(batch_log_data)
            return

        try:
            # ===== 步骤 1/5: 解压并提取文件 =====
            step1_start = time.time()
            student_texts = {}
            with zipfile.ZipFile(io.BytesIO(batch_bytes), "r") as zip_ref:
                for item_info in tqdm(zip_ref.infolist(), desc="步骤 1/5: 解压并提取文件"):
                    if item_info.is_dir(): continue
                    try: student_filename = item_info.filename.encode("cp437").decode("gbk")
                    except: student_filename = item_info.filename
                    if student_filename.startswith("__MACOSX/") or os.path.basename(student_filename) == ".DS_Store": continue
                    student_id = os.path.splitext(os.path.basename(student_filename))[0]
                    if student_id not in student_texts: student_texts[student_id] = ""
                    file_content = grading_service.process_archive(zip_ref.read(item_info), student_filename, 0)
                    student_texts[student_id] += f"--- 文件开始: {student_filename} ---\n\n{file_content}\n\n--- 文件结束: {student_filename} ---\n\n"
            batch_log_data["step1_unzip_time_seconds"] = round(time.time() - step1_start, 2)
            
            # 更新学生数量
            batch_log_data["student_count"] = len(student_texts)
            
            # ===== 步骤 2/5: 初步查重与内容分离 =====
            step2_start = time.time()
            print("步骤 2/5: 初步查重与内容分离...")
            plagiarism_results = plagiarism_service.check_plagiarism_in_batch(student_texts)
            suspicious_text_pairs = plagiarism_results["suspicious_text_pairs"]
            suspicious_code_pairs = plagiarism_results["suspicious_code_pairs"]
            separated_contents = plagiarism_results["separated_contents"]
            print(f"初步查重完成。发现 {len(suspicious_text_pairs)} 对可疑文本, {len(suspicious_code_pairs)} 对可疑代码。")
            batch_log_data["step2_plagiarism_check_time_seconds"] = round(time.time() - step2_start, 2)
            
            # ===== 步骤 3/5: 代码-文档匹配度分析 =====
            step3_start = time.time()
            print("步骤 3/5: 对每个学生提交的代码和文档进行匹配度分析...")
            code_doc_match_reports = {}
            for sid, content in tqdm(separated_contents.items(), desc="步骤 3/5: 代码-文档匹配度分析"):
                if content.get("code") and content.get("text"):
                    print(f"\nDEBUG: Analyzing code/doc match for {sid}...")
                    match_analysis, _ = deepseek_service.analyze_code_doc_match(
                        code_content=content["code"], 
                        doc_content=content["text"],
                        assignment_requirement=assignment.question # 传入作业要求
                    )
                    print(f"DEBUG: LLM raw result for {sid}:")
                    pprint.pprint(match_analysis)
                    if match_analysis:
                        try:
                            report_instance = schemas.CodeDocMatchReport(**match_analysis)
                            code_doc_match_reports[sid] = report_instance
                            print(f"DEBUG: Successfully created CodeDocMatchReport for {sid}.")
                        except Exception as e:
                            print(f"DEBUG: Pydantic validation failed for {sid}! Error: {e}")
                            print(f"DEBUG: Data that failed validation: {match_analysis}")
                    else:
                        print(f"DEBUG: 'match_analysis' is None or empty for {sid}.")
                else:
                    print(f"\nDEBUG: Skipping code/doc match for {sid} due to missing content. Has code: {bool(content.get('code'))}, Has text: {bool(content.get('text'))}")

            print("代码-文档匹配度分析完成。")
            print("DEBUG: Final code_doc_match_reports dictionary:")
            pprint.pprint(code_doc_match_reports)
            batch_log_data["step3_code_doc_match_time_seconds"] = round(time.time() - step3_start, 2)

            # ===== 步骤 4/5: 对可疑配对进行LLM深度分析 =====
            step4_start = time.time()
            print("步骤 4/5: 对可疑配对进行LLM深度分析...")
            llm_analysis_cache = {}
            all_suspicious_pairs = [(*pair, 'text') for pair in suspicious_text_pairs] + [(*pair, 'code') for pair in suspicious_code_pairs]
            
            for s1, s2, initial_score, content_type in tqdm(all_suspicious_pairs, desc="步骤 4/5: LLM深度查重分析"):
                content1 = separated_contents.get(s1, {}).get(content_type)
                content2 = separated_contents.get(s2, {}).get(content_type)
                if not content1 or not content2: continue
                
                llm_analysis, _ = deepseek_service.analyze_plagiarism(content1, content2, content_type)
                if llm_analysis:
                    llm_analysis_cache[(s1, s2, content_type)] = {'initial_score': initial_score, 'llm_analysis': llm_analysis}
            print("LLM深度分析完成。")
            batch_log_data["step4_llm_analysis_time_seconds"] = round(time.time() - step4_start, 2)

            # ===== 步骤 5/5: 对所有提交内容进行AIGC检测 =====
            step5_start = time.time()
            print("步骤 5/5: 对所有提交内容进行AIGC检测...")
            aigc_reports = {
                sid: aigc_detector_service.detect(content) 
                for sid, content in tqdm(student_texts.items(), desc="步骤 5/5: AIGC内容检测")
            }
            print("AIGC检测完成。")
            batch_log_data["step5_aigc_detect_time_seconds"] = round(time.time() - step5_start, 2)

            print("整理报告并准备评分...")
            final_student_data = {
                sid: {
                    "plagiarism_reports": [],
                    "aigc_report": schemas.AIGCReport(**aigc_reports[sid]) if "error" not in aigc_reports.get(sid, {}) else None,
                    "code_doc_match_report": code_doc_match_reports.get(sid)
                } for sid in student_texts.keys()
            }

            for (s1, s2, content_type), analysis_result in llm_analysis_cache.items():
                llm_analysis_data = analysis_result.get('llm_analysis')
                if llm_analysis_data:
                    report_for_s1 = schemas.PlagiarismReport(
                        similar_to=s2,
                        content_type=content_type,
                        initial_score=analysis_result['initial_score'],
                        llm_analysis=schemas.LLMPlagiarismAnalysis(**llm_analysis_data)
                    )
                    final_student_data[s1]["plagiarism_reports"].append(report_for_s1)
                    
                    report_for_s2 = schemas.PlagiarismReport(
                        similar_to=s1,
                        content_type=content_type,
                        initial_score=analysis_result['initial_score'],
                        llm_analysis=schemas.LLMPlagiarismAnalysis(**llm_analysis_data)
                    )
                    final_student_data[s2]["plagiarism_reports"].append(report_for_s2)

            # ===== 步骤 6/6: 开始逐一评分并保存 =====
            step6_start = time.time()
            print("步骤 6/6: 开始逐一评分并保存...")
            for student_id, merged_content in tqdm(student_texts.items(), desc="步骤 6/6: LLM评分并保存"):
                student_data = final_student_data[student_id]
                
                ai_result = deepseek_service.grade_homework(
                    question=assignment.question, rubric=assignment.rubric, student_answer=merged_content,
                    plagiarism_reports=student_data["plagiarism_reports"], 
                    aigc_report=student_data["aigc_report"],
                    code_doc_match_report=student_data["code_doc_match_report"]
                )
                
                submission_data = schemas.SubmissionCreate(
                    student_id=student_id, score=ai_result.get("total_score", -1),
                    feedback=ai_result.get("overall_feedback", "评分失败"), merged_content=merged_content,
                    assignment_id=assignment_id, 
                    plagiarism_reports=student_data["plagiarism_reports"],
                    aigc_report=student_data["aigc_report"],
                    code_doc_match_report=student_data["code_doc_match_report"],
                    is_human_reviewed=False,
                    human_feedback=None,
                    human_score=None
                )
                await crud.create_submission(db=db_session, submission=submission_data)
            
            print("所有学生的作业处理完成并已存入数据库。")
            batch_log_data["step6_grading_time_seconds"] = round(time.time() - step6_start, 2)
            
            # 记录成功完成
            batch_log_data["status"] = "completed"
            
        except Exception as e:
            error_msg = f"处理作业ID {assignment_id} 的批量文件时发生严重错误: {e}"
            print(error_msg)
            traceback.print_exc()
            # 记录失败状态
            batch_log_data["status"] = "failed"
            batch_log_data["error_message"] = str(e)
        
        finally:
            # 计算总处理时间并保存日志
            batch_end_time = time.time()
            batch_log_data["total_processing_time_seconds"] = round(batch_end_time - batch_start_time, 2)
            _save_batch_log(batch_log_data)

def _save_batch_log(log_data: dict):
    """保存批处理日志到文件"""
    try:
        log_dir = "logs"
        log_file = os.path.join(log_dir, "batch_processing.log")
        os.makedirs(log_dir, exist_ok=True)
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"警告：保存批处理日志时出错: {e}")

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
    """
    将指定作业的所有评分结果及详细抄袭报告导出为Excel文件。
    """
    assignment = await crud.get_assignment(db=db_session, assignment_id=assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="未找到该作业")

    results = await crud.get_submissions_for_assignment(
        db=db_session, assignment_id=assignment_id
    )

    if not results:
        raise HTTPException(status_code=404, detail="该作业没有任何评分结果可以导出")

    summary_data = []
    for res in results:
        final_score = res.human_score if res.is_human_reviewed and res.human_score is not None else res.score
        
        max_plagiarism_score = 0
        if res.plagiarism_reports:
            # 确保plagiarism_reports中的每个元素都是Pydantic模型或可以正确访问属性的字典
            valid_reports = [schemas.PlagiarismReport.model_validate(r) for r in res.plagiarism_reports]
            scores = [r.llm_analysis.similarity_score for r in valid_reports if r.llm_analysis]
            if scores:
                max_plagiarism_score = max(scores)
        
        aigc_risk = "未检测"
        if res.aigc_report:
            aigc_report_obj = schemas.AIGCReport.model_validate(res.aigc_report)
            prob = aigc_report_obj.ai_probability * 100
            source = aigc_report_obj.detection_source
            aigc_risk = f"{prob:.1f}% AI生成 ({source})" if source else f"{prob:.1f}% AI生成"
        
        # 因为学生的代码可能包含库函数实现，因此修改prompt，将任务内容传入，方便比对
        code_doc_match = "未检测"
        if res.code_doc_match_report:
            match_report = schemas.CodeDocMatchReport.model_validate(res.code_doc_match_report)
            code_doc_match = f"{match_report.score}分"
            
        summary_data.append({
            "学生ID": res.student_id,
            "最终得分": final_score,
            "AI评分": res.score,
            "是否人工复核": "是" if res.is_human_reviewed else "否",
            "人工评分": res.human_score if res.is_human_reviewed else "",
            "最高抄袭风险(LLM)": f"{max_plagiarism_score}分" if max_plagiarism_score > 0 else "无风险",
            "AIGC风险": aigc_risk,
            "代码-文档匹配度": code_doc_match, 
            "AI评语": res.feedback,
            "人工评语": res.human_feedback if res.is_human_reviewed else "",
        })
    df_summary = pd.DataFrame(summary_data)

    plagiarism_details_data = []
    for res in results:
        if not res.plagiarism_reports:
            continue
        valid_reports = [schemas.PlagiarismReport.model_validate(r) for r in res.plagiarism_reports]
        for report in valid_reports:
            if not report.llm_analysis:
                continue
            
            suspicious_parts_text = ""
            if report.llm_analysis.suspicious_parts:
                parts_list = []
                for i, part in enumerate(report.llm_analysis.suspicious_parts, 1):
                    parts_list.append(
                        f"--- 证据 {i} ---\n"
                        f"学生({res.student_id}):\n{part.student_A_content}\n\n"
                        f"学生({report.similar_to}):\n{part.student_B_content}\n"
                    )
                suspicious_parts_text = "\n".join(parts_list)

            plagiarism_details_data.append({
                "学生ID": res.student_id,
                "相似对象ID": report.similar_to,
                "内容类型": "代码" if report.content_type == 'code' else "文本",
                "LLM评估分数": report.llm_analysis.similarity_score,
                "LLM分析理由": report.llm_analysis.reasoning,
                "具体相似片段证据": suspicious_parts_text 
            })
    df_plagiarism = pd.DataFrame(plagiarism_details_data)

    output_stream = io.BytesIO()
    with pd.ExcelWriter(output_stream, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='评分结果汇总', index=False)
        if not df_plagiarism.empty:
            df_plagiarism.to_excel(writer, sheet_name='抄袭检测详情', index=False)

    output_stream.seek(0)
    
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