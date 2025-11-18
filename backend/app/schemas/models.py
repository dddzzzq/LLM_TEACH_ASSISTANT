from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- 现有模型 ---

# --- 新增教师人工审查更新模型 ---
class SubmissionUpdate(BaseModel):
    human_score: float    # 得分
    human_feedback: str     # 评语

# --- AIGC检测报告模型 ---
class AIGCReport(BaseModel):
    predicted_label: str = Field(description="预测标签，如 'AI生成' 或 '人类写作'")
    confidence: float = Field(description="模型对预测标签的置信度")
    ai_probability: float = Field(description="文本由AI生成的概率")
    detection_source: Optional[str] = Field(None, description="风险最高的文本来源，如 '文字报告' 或 '源代码'")

# --- 查重报告模型 ---
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

# 新增代码文档匹配得分
# +++ 新增代码与文档匹配度报告模型 +++
class CodeDocMatchReport(BaseModel):
    score: int = Field(description="LLM给出的0-100的匹配度分数")
    reasoning: str = Field(description="LLM给出的详细分析理由")


# --- 提交记录模型 ---
class SubmissionBase(BaseModel):
    student_id: str
    score: float
    feedback: str
    plagiarism_reports: List[PlagiarismReport] = []
    aigc_report: Optional[AIGCReport] = None  # 新增aigc检测报告
    is_human_reviewed: bool     # 新增字段：是否人工复查
    human_feedback: Optional[str] = None    # 新增字段：教师评语
    human_score: Optional[float] = None # <-- 修改这里，允许为None      # 新增字段：教师评分
    code_doc_match_report: Optional[CodeDocMatchReport] = None  # 新增代码文档匹配报告


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

class AssignmentInDB(BaseModel):
    id: int
    class Config:
        from_attributes = True

# --- 新增试卷模型 ---

# 试卷 (Exam)
class ExamBase(BaseModel):
    name: str

class ExamCreate(ExamBase):
    pass

class ExamInDB(ExamBase):
    id: int
    question_count: int

    class Config:
        from_attributes = True

# 试卷题目 (ExamQuestion)
class ExamQuestionBase(BaseModel):
    question_number: int
    question_text: str
    standard_answer: str
    rubric: str # <--- 修改：从 Dict[str, Any] 改为 str
    max_score: float # <--- 新增

class ExamQuestionCreate(ExamQuestionBase):
    pass

class ExamQuestionInDB(ExamQuestionBase):
    id: int
    exam_id: int

    class Config:
        from_attributes = True

# 包含题目的试卷详情
class ExamWithQuestions(ExamInDB):
    questions: List[ExamQuestionInDB] = []

# 学生单题答案 (StudentQuestionAnswer)
class StudentQuestionAnswerBase(BaseModel):
    student_answer_text: str
    score: float
    feedback: str

class StudentQuestionAnswerCreate(StudentQuestionAnswerBase):
    student_exam_id: int
    exam_question_id: int

class StudentQuestionAnswerInDB(StudentQuestionAnswerBase):
    id: int
    student_exam_id: int
    exam_question_id: int
    question: ExamQuestionInDB # 嵌套题目信息

    class Config:
        from_attributes = True

# 学生总结报告 (StudentExamReport)
class StudentExamReportBase(BaseModel):
    total_score: float
    summary_report: str

class StudentExamReportCreate(StudentExamReportBase):
    student_exam_id: int

class StudentExamReportInDB(StudentExamReportBase):
    id: int
    student_exam_id: int

    class Config:
        from_attributes = True

# 试卷的学生成绩列表（简要）
class StudentExamResultSummary(BaseModel):
    student_id: str
    total_score: float
    student_exam_id: int # 用于跳转到详细页

# 学生的详细报告（总结 + 题目列表）
class StudentExamDetailedReport(BaseModel):
    student_id: str
    exam_id: int
    report: StudentExamReportInDB
    answers: List[StudentQuestionAnswerInDB]

    class Config:
        from_attributes = True