from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
import json
from .database import Base
from ..schemas.models import PlagiarismReport, AIGCReport # 引入aigc检测模型模块
from typing import List, Optional, Dict

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
    _plagiarism_reports_json = Column("plagiarism_reports", Text, nullable=True)
    _aigc_report_json = Column("aigc_report", Text, nullable=True)
    assignment_id = Column(Integer, ForeignKey("assignments.id"))
    assignment = relationship("Assignment", back_populates="submissions")
    # 新增教师复查功能
    is_human_reviewed = Column(Boolean, default=False, nullable=False)
    human_feedback = Column(Text, nullable=True) # 存储教师的最终评语
    human_score = Column(Float)     # 存储教师评分

    @property
    def plagiarism_reports(self):
        if self._plagiarism_reports_json is None:
            return []
        return json.loads(self._plagiarism_reports_json)

    @plagiarism_reports.setter
    def plagiarism_reports(self, value: Optional[List[Dict]]):   # 修改为报告列表
        if value is None:
            self._plagiarism_reports_json = None
        else:
            reports_as_dicts = [report.model_dump(by_alias=True) for report in value]
            self._plagiarism_reports_json = json.dumps(reports_as_dicts, ensure_ascii=False)

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
