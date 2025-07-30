from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- AIGC检测报告模型 ---
class AIGCReport(BaseModel):
    predicted_label: str = Field(description="预测标签，如 'AI生成' 或 '人类写作'")
    confidence: float = Field(description="模型对预测标签的置信度")
    ai_probability: float = Field(description="文本由AI生成的概率")

# --- 查重报告模型 ---
# class SimilarityMatch(BaseModel):
#     similar_to: str = Field(description="最高相似度对应的学生ID")
#     score: float = Field(description="TF-IDF计算出的最高相似度分数")

# class LLMAnalysis(BaseModel):
#     is_plagiarized: bool = Field(description="LLM判断是否构成抄袭")
#     reasoning: str = Field(description="LLM给出的判断理由")
#     suspicious_parts: List[str] = Field(description="LLM找出的可疑文本片段")

# class PlagiarismReport(BaseModel):
#     highest_similarity: Optional[SimilarityMatch] = None
#     llm_analysis: Optional[LLMAnalysis] = None

# 修改抄袭检测数据模型
class SuspiciousPart (BaseModel):
    student_A_content: str
    student_B_content: str

class LLMPlagiarismAnalysis(BaseModel):
    similarity_score: int = Field(description="LLM给出的0-100的相似度分数")
    reasoning: str = Field(description="LLM给出的详细分析理由")
    suspicious_parts: List[SuspiciousPart] = Field(description="具体的相似片段证据")

class PlagiarismReport(BaseModel):
    similar_to: str = Field(description="与哪个学生最相似")
    initial_score: float = Field(description="第一阶段计算出的语义相似度分数")
    content_type: str = Field(description="内容类型: 'text' 或 'code'")
    llm_analysis: Optional[LLMPlagiarismAnalysis] = None


# --- 提交记录模型 ---
class SubmissionBase(BaseModel):
    student_id: str
    score: float
    feedback: str
    plagiarism_reports: List[PlagiarismReport] = []
    aigc_report: Optional[AIGCReport] = None  # 新增aigc检测报告

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