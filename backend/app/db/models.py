from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
import json
from .database import Base
from ..schemas.models import PlagiarismReport, AIGCReport, CodeDocMatchReport # 引入aigc检测模型模块，代码文档匹配度得分
from typing import List, Optional, Dict

# --- 现有模型 (作业) ---

class Assignment(Base):
    __tablename__ = "assignments"
    id = Column(Integer, primary_key=True, index=True)
    task_name = Column(String, index=True)
    question = Column(Text)
    _rubric_json = Column("rubric", Text)
    submissions = relationship("Submission", back_populates="assignment", cascade="all, delete-orphan")

    @property
    def rubric(self):
        return json.loads(self._rubric_json)

    @rubric.setter
    def rubric(self, value):
        self._rubric_json = json.dumps(value, ensure_ascii=False)

class Submission(Base):
    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, index=True)
    score = Column(Float)
    feedback = Column(Text)
    merged_content = Column(Text)
    _plagiarism_reports_json = Column("plagiarism_reports", Text, nullable=True)
    _aigc_report_json = Column("aigc_report", Text, nullable=True)
    assignment_id = Column(Integer, ForeignKey("assignments.id"))
    assignment = relationship("Assignment", back_populates="submissions")
    # 新增代码文档匹配得分
    _code_doc_match_report_json = Column("code_doc_match_report", Text, nullable=True)
    # 新增教师复查功能
    is_human_reviewed = Column(Boolean, default=False, nullable=False)
    human_feedback = Column(Text, nullable=True) # 存储教师的最终评语
    human_score = Column(Float)     # 存储教师评分

    @property
    def plagiarism_reports(self):
        if self._plagiarism_reports_json is None:
            return []
        reports = json.loads(self._plagiarism_reports_json)
        # 兼容旧数据格式
        return [PlagiarismReport.model_validate(r) for r in reports]

    @plagiarism_reports.setter
    def plagiarism_reports(self, value: Optional[List[Dict]]):   # 修改为报告列表
        if value is None:
            self._plagiarism_reports_json = None
        else:
            reports_as_dicts = [report.model_dump(by_alias=True) for report in value]
            self._plagiarism_reports_json = json.dumps(reports_as_dicts, ensure_ascii=False)

    # 新增代码和文档匹配得分属性
    @property
    def code_doc_match_report(self) -> Optional[CodeDocMatchReport]:
        if self._code_doc_match_report_json is None:
            return None
        try:
            return CodeDocMatchReport.model_validate(json.loads(self._code_doc_match_report_json))
        except (json.JSONDecodeError, TypeError):
            return None

    @code_doc_match_report.setter
    def code_doc_match_report(self, value: Optional[CodeDocMatchReport]):
        if value is None:
            self._code_doc_match_report_json = None
        else:
            self._code_doc_match_report_json = value.model_dump_json() if hasattr(value, 'model_dump_json') else value.json()

    @property
    def aigc_report(self):
        if self._aigc_report_json is None:
            return None
        return json.loads(self._aigc_report_json)

    @aigc_report.setter
    def aigc_report(self, value: Optional[AIGCReport]):
        if value is None:
            self._aigc_report_json = None
        else:
            self._aigc_report_json = value.model_dump_json()


# --- 新增模型 (试卷) ---

class Exam(Base):
    """
    试卷模型
    """
    __tablename__ = "exams"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    question_count = Column(Integer, default=0)
    
    questions = relationship("ExamQuestion", back_populates="exam", cascade="all, delete-orphan")
    student_exams = relationship("StudentExam", back_populates="exam", cascade="all, delete-orphan")

class ExamQuestion(Base):
    """
    试卷中的题目
    """
    __tablename__ = "exam_questions"
    id = Column(Integer, primary_key=True, index=True)
    exam_id = Column(Integer, ForeignKey("exams.id"))
    question_number = Column(Integer)
    question_text = Column(Text)
    standard_answer = Column(Text)
    rubric = Column(Text) # <--- 修改：不再是JSON，而是纯文本
    max_score = Column(Float, default=10.0) # <--- 新增：题目总分
    
    exam = relationship("Exam", back_populates="questions")
    student_answers = relationship("StudentQuestionAnswer", back_populates="question")

    # <--- 删除：移除 @property 和 @setter for rubric ---

class StudentExam(Base):
    """
    学生的一次试卷提交（关联所有图片和答案）
    """
    __tablename__ = "student_exams"
    id = Column(Integer, primary_key=True, index=True)
    exam_id = Column(Integer, ForeignKey("exams.id"))
    student_id = Column(String, index=True)
    # student_name = Column(String, nullable=True) # 暂时先用 student_id
    
    exam = relationship("Exam", back_populates="student_exams")
    answers = relationship("StudentQuestionAnswer", back_populates="student_exam", cascade="all, delete-orphan")
    report = relationship("StudentExamReport", back_populates="student_exam", uselist=False, cascade="all, delete-orphan")

class StudentQuestionAnswer(Base):
    """
    学生对单个题目的回答、得分和评语
    """
    __tablename__ = "student_question_answers"
    id = Column(Integer, primary_key=True, index=True)
    student_exam_id = Column(Integer, ForeignKey("student_exams.id"))
    exam_question_id = Column(Integer, ForeignKey("exam_questions.id"))
    student_answer_text = Column(Text) # LLM从OCR结果中提取的学生答案
    score = Column(Float)
    feedback = Column(Text) # LLM给出的评判依据
    
    student_exam = relationship("StudentExam", back_populates="answers")
    question = relationship("ExamQuestion", back_populates="student_answers")

class StudentExamReport(Base):
    """
    学生的总成绩和总结报告
    """
    __tablename__ = "student_exam_reports"
    id = Column(Integer, primary_key=True, index=True)
    student_exam_id = Column(Integer, ForeignKey("student_exams.id"), unique=True)
    total_score = Column(Float)
    summary_report = Column(Text) # LLM生成的总结
    
    student_exam = relationship("StudentExam", back_populates="report")