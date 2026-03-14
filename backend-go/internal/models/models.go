package models

import "time"

// =========================
//    作业模块
// =========================

// Assignment 作业表
type Assignment struct {
	ID          uint         `gorm:"primaryKey"`                                           // 主键
	TaskName    string       `gorm:"size:255;index"`                                       // 任务名称
	Question    string       `gorm:"type:longtext"`                                        // 任务描述 (升级为 longtext)
	Rubric      string       `gorm:"column:rubric;type:longtext"`                          // 任务评分标准 (升级为 longtext)
	Submissions []Submission `gorm:"foreignKey:AssignmentID;constraint:OnDelete:CASCADE;"` // 学生提交，外键
}

// Submission 学生提交记录表
type Submission struct {
	ID                 uint    `gorm:"primaryKey"`    // 主键
	StudentID          string  `gorm:"size:64;index"` // 学生姓名
	Score              float64 // 得分
	Feedback           string  `gorm:"type:longtext"`                              // ai反馈评价 (升级为 longtext)
	MergeContent       string  `gorm:"type:longtext"`                              // 学生提交合并内容 (升级为 longtext)
	PlagiarismReport   string  `gorm:"column:plagiarism_reports;type:longtext"`    // 抄袭报告，存为json格式 (升级为 longtext)
	AIGCReport         string  `gorm:"column:aigc_reports;type:longtext"`          // aigc检测报告，存为json格式 (升级为 longtext)
	CodeDocMatchReport string  `gorm:"column:code_doc_match_report;type:longtext"` // 代码文本相似度检测，存为json格式 (升级为 longtext)
	AssignmentID       uint    `gorm:"index"`                                      // 根据其分组
	IsHumanReviewed    bool    `gorm:"type:boolean;default:false;not null"`        // 教师是否复查
	HumanFeedback      string  `gorm:"type:longtext"`                              // 教师复查评语 (升级为 longtext)
	HumanScore         float64 // 教师评分
}

// ========================
//
//	试卷模块
//
// ========================
// Exam 试卷模型
type Exam struct {
	ID            uint           `json:"id" gorm:"primaryKey"`                                                // 主键，唯一
	Name          string         `json:"name" gorm:"type:varchar(255);not null"`                              // 试卷名
	QuestionCount int            `json:"question_count" gorm:"default:0"`                                     // 题数
	TotalScore    float64        `json:"total_score" gorm:"default:0"`                                        // 总分
	Questions     []ExamQuestion `json:"questions" gorm:"foreignKey:ExamID;constraint:OnDelete:CASCADE;"`     // 问题
	StudentExams  []StudentExam  `json:"student_exams" gorm:"foreignKey:ExamID;constraint:OnDelete:CASCADE;"` // 学生试卷
	CreatedAt     time.Time      `json:"created_at"`
	UpdatedAt     time.Time      `json:"updated_at"`
}

// ExamQuestion 试卷题目模型
type ExamQuestion struct {
	ID             uint    `json:"id" gorm:"primaryKey"`                      // 主键
	ExamID         uint    `json:"exam_id" gorm:"not null;index"`             // 试卷id
	QuestionNumber int     `json:"question_number" gorm:"not null"`           // 题号
	QuestionText   string  `json:"question_text" gorm:"type:text;not null"`   // 问题
	StandardAnswer string  `json:"standard_answer" gorm:"type:text;not null"` // 标准答案
	Rubric         string  `json:"rubric" gorm:"type:text;not null"`          // 纯文本评分标准
	MaxScore       float64 `json:"max_score" gorm:"not null"`                 // 满分
}

// StudentExam 某个学生单次试卷批改记录
type StudentExam struct {
	ID        uint                `json:"id" gorm:"primaryKey"`
	ExamID    uint                `json:"exam_id" gorm:"not null;index"`
	StudentID string              `json:"student_id" gorm:"type:varchar(100);not null;index"`
	Images    []StudentExamImage  `json:"images" gorm:"foreignKey:StudentExamID;constraint:OnDelete:CASCADE;"`
	Answers   []StudentExamAnswer `json:"answers" gorm:"foreignKey:StudentExamID;constraint:OnDelete:CASCADE;"`
	Report    ExamReport          `json:"report" gorm:"foreignKey:StudentExamID;constraint:OnDelete:CASCADE;"` // 1对1关联
	CreatedAt time.Time           `json:"created_at"`
	UpdatedAt time.Time           `json:"updated_at"`
}

// StudentExamImage 学生上传的试卷图片
type StudentExamImage struct {
	ID             uint   `json:"id" gorm:"primaryKey"`
	StudentExamID  uint   `json:"student_exam_id" gorm:"not null;index"`
	ImagePath      string `json:"image_path" gorm:"type:varchar(255);not null"`
	ImageIndex     int    `json:"image_index" gorm:"not null"` // 图片顺序/页码
	ExamQuestionID *uint  `json:"exam_question_id" gorm:"index"`
}

// StudentExamAnswer 单道题的批改结果
type StudentExamAnswer struct {
	ID             uint         `json:"id" gorm:"primaryKey"`
	StudentExamID  uint         `json:"student_exam_id" gorm:"not null;index"`
	ExamQuestionID uint         `json:"exam_question_id" gorm:"not null;index"`
	Question       ExamQuestion `json:"question" gorm:"foreignKey:ExamQuestionID;constraint:OnDelete:CASCADE;"` // 关联具体的题目信息
	OCRText        string       `json:"ocr_text" gorm:"type:text"`                                              // 大模型识别出的该题手写文字
	Score          float64      `json:"score"`                                                                  // 该题得分
	Feedback       string       `json:"feedback" gorm:"type:text"`                                              // 该题批改反馈
}

// ExamReport 整份试卷的综合批改报告
type ExamReport struct {
	ID            uint    `json:"id" gorm:"primaryKey"`
	StudentExamID uint    `json:"student_exam_id" gorm:"uniqueIndex;not null"` // 1对1关联，需加 uniqueIndex
	TotalScore    float64 `json:"total_score"`                                 // 整卷总分
	Summary       string  `json:"summary" gorm:"type:text"`                    // 整体评价/总结
}
