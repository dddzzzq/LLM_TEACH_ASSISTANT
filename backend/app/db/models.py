from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey
from sqlalchemy.orm import relationship
import json
from .database import Base
from ..schemas.models import PlagiarismReport
from typing import Optional

# 数据库中主要有两个数据模型，一是任务/作业模型，二是提交记录模型

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
    _plagiarism_report_json = Column("plagiarism_report", Text, nullable=True)
    assignment_id = Column(Integer, ForeignKey("assignments.id"))
    assignment = relationship("Assignment", back_populates="submissions")

    @property
    def plagiarism_report(self):
        if self._plagiarism_report_json is None:
            return None
        return json.loads(self._plagiarism_report_json)

    @plagiarism_report.setter
    def plagiarism_report(self, value: Optional[PlagiarismReport]):
        if value is None:
            self._plagiarism_report_json = None
        else:
            self._plagiarism_report_json = value.model_dump_json()