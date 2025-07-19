from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- 查重报告模型 ---
class SimilarityMatch(BaseModel):
    similar_to: str = Field(description="最高相似度对应的学生ID")
    score: float = Field(description="TF-IDF计算出的最高相似度分数")

class LLMAnalysis(BaseModel):
    is_plagiarized: bool = Field(description="LLM判断是否构成抄袭")
    reasoning: str = Field(description="LLM给出的判断理由")
    suspicious_parts: List[str] = Field(description="LLM找出的可疑文本片段")

class PlagiarismReport(BaseModel):
    highest_similarity: Optional[SimilarityMatch] = None
    llm_analysis: Optional[LLMAnalysis] = None

# --- 提交记录模型 ---
class SubmissionBase(BaseModel):
    student_id: str
    score: float
    feedback: str
    plagiarism_report: Optional[PlagiarismReport] = None

class SubmissionCreate(SubmissionBase):
    merged_content: str
    assignment_id: int

class SubmissionInDB(SubmissionBase):
    id: int
    assignment_id: int
    class Config:
        from_attributes = True

# --- 作业任务模型 ---
class AssignmentBase(BaseModel):
    task_name: str
    question: str
    rubric: Dict[str, Any]

class AssignmentCreate(AssignmentBase):
    pass

class AssignmentWithSubmissions(AssignmentBase):
    id: int
    submissions: List[SubmissionInDB] = []
    class Config:
        from_attributes = True

class AssignmentInDB(AssignmentBase):
    id: int
    class Config:
        from_attributes = True