from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import json
from ..schemas.models import GradingTaskRequest, HolisticGradingResponse
from ..services.grading_service import grading_service
from ..services.deepseek_service import deepseek_service

# 定义作业批改路由，后台调用
router = APIRouter(
    prefix="/homework",
    tags=["Homework Grading"]
)

# 该路由的POST方法，用于接收一个包含学生所有项目文件的ZIP或RAR压缩包，
@router.post("/grade", response_model=HolisticGradingResponse)
async def grade_homework_submission(
    task_name: str = Form(..., description="作业任务的名称"),
    question: str = Form(..., description="题目的具体描述"),
    rubric: str = Form(..., description="结构化的评分标准 (必须是JSON格式的字符串)"),
    file: UploadFile = File(..., description="包含所有学生作业文件的ZIP或RAR压缩包")
):
    """
    接收一个包含学生所有项目文件的ZIP或RAR压缩包，
    合并其内容后，进行一次性的综合AI批改。
    """
    allowed_extensions = ['.zip', '.rar']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="文件格式错误，请上传ZIP或RAR压缩包。")

    try:
        rubric_dict = json.loads(rubric)
        task_data = GradingTaskRequest(
            task_name=task_name,
            question=question,
            rubric=rubric_dict
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="`rubric`字段不是一个有效的JSON字符串。")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"请求数据验证失败: {e}")

    try:
        file_bytes = await file.read()
        merged_content = grading_service.process_archive(file_bytes, file.filename)

        if not merged_content.strip():
            raise ValueError("压缩包中没有找到任何可读取内容的文件。")

        ai_result = deepseek_service.grade_homework(
            question=task_data.question,
            rubric=task_data.rubric,
            student_answer=merged_content
        )

        return HolisticGradingResponse(
            task_name=task_data.task_name,
            question=task_data.question,
            score=ai_result.get("score", -1),
            feedback=ai_result.get("feedback", "批改失败"),
            merged_content=merged_content
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"批改任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")