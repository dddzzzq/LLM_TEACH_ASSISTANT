from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
import json
from .database import Base
from ..schemas.models import PlagiarismReport, AIGCReport, CodeDocMatchReport
from typing import List, Optional, Dict


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
    
    _code_doc_match_report_json = Column("code_doc_match_report", Text, nullable=True)
    is_human_reviewed = Column(Boolean, default=False, nullable=False)
    human_feedback = Column(Text, nullable=True)
    human_score = Column(Float)

    @property
    def plagiarism_reports(self):
        if self._plagiarism_reports_json is None:
            return []
        reports = json.loads(self._plagiarism_reports_json)
        return [PlagiarismReport.model_validate(r) for r in reports]

    @plagiarism_reports.setter
    def plagiarism_reports(self, value: Optional[List[Dict]]):
        if value is None:
            self._plagiarism_reports_json = None
        else:
            reports_as_dicts = [report.model_dump(by_alias=True) for report in value]
            self._plagiarism_reports_json = json.dumps(reports_as_dicts, ensure_ascii=False)

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


#  试卷相关模型 

class Exam(Base):
    __tablename__ = "exams"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    question_count = Column(Integer, default=0)
    total_score = Column(Float, default=0.0)
    
    questions = relationship("ExamQuestion", back_populates="exam", cascade="all, delete-orphan")
    student_exams = relationship("StudentExam", back_populates="exam", cascade="all, delete-orphan")

class ExamQuestion(Base):
    __tablename__ = "exam_questions"
    id = Column(Integer, primary_key=True, index=True)
    exam_id = Column(Integer, ForeignKey("exams.id"))
    question_number = Column(Integer)
    question_text = Column(Text)
    standard_answer = Column(Text)
    rubric = Column(Text) 
    max_score = Column(Float, default=10.0) 
    
    exam = relationship("Exam", back_populates="questions")
    student_answers = relationship("StudentQuestionAnswer", back_populates="question")
    # 反向关联图片（可选，如果需要直接通过题目找图片）
    # images = relationship("StudentExamImage", back_populates="question")

class StudentExam(Base):
    __tablename__ = "student_exams"
    id = Column(Integer, primary_key=True, index=True)
    exam_id = Column(Integer, ForeignKey("exams.id"))
    student_id = Column(String, index=True)
    
    exam = relationship("Exam", back_populates="student_exams")
    answers = relationship("StudentQuestionAnswer", back_populates="student_exam", cascade="all, delete-orphan")
    report = relationship("StudentExamReport", back_populates="student_exam", uselist=False, cascade="all, delete-orphan")
    
    # 关联图片
    images = relationship("StudentExamImage", back_populates="student_exam", cascade="all, delete-orphan")

class StudentExamImage(Base):
    """
    存储学生上传的试卷图片路径
    """
    __tablename__ = "student_exam_images"
    id = Column(Integer, primary_key=True, index=True)
    student_exam_id = Column(Integer, ForeignKey("student_exams.id"))
    image_path = Column(String) # 存储相对路径
    
    # 关联具体的题目ID。如果一张图包含多道题，这里可以存第一道题的ID，或者改为多对多（为了简单，我们假设主要归属于某一道题，或者为NULL表示未识别/公共页）
    exam_question_id = Column(Integer, ForeignKey("exam_questions.id"), nullable=True)
    
    student_exam = relationship("StudentExam", back_populates="images")
    question = relationship("ExamQuestion") # 关联到题目

class StudentQuestionAnswer(Base):
    __tablename__ = "student_question_answers"
    id = Column(Integer, primary_key=True, index=True)
    student_exam_id = Column(Integer, ForeignKey("student_exams.id"))
    exam_question_id = Column(Integer, ForeignKey("exam_questions.id"))
    student_answer_text = Column(Text) 
    score = Column(Float)
    feedback = Column(Text) 
    
    student_exam = relationship("StudentExam", back_populates="answers")
    question = relationship("ExamQuestion", back_populates="student_answers")

class StudentExamReport(Base):
    __tablename__ = "student_exam_reports"
    id = Column(Integer, primary_key=True, index=True)
    student_exam_id = Column(Integer, ForeignKey("student_exams.id"), unique=True)
    total_score = Column(Float)
    summary_report = Column(Text) 
    
    student_exam = relationship("StudentExam", back_populates="report")