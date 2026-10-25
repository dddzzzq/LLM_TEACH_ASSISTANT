package models

// Assignment 作业表
type Assignment struct {
	OwnerID     uint         `json:"owner_id" gorm:"not null;default:0;index"`
	ID          uint         `gorm:"primaryKey"`                                           // 主键
	CourseName  string       `gorm:"size:255;index;not null"`                              // 课程名称
	ClassName   string       `gorm:"size:255;index;not null"`                              // 班级名称
	TaskName    string       `gorm:"size:255;index"`                                       // 作业名称
	Question    string       `gorm:"type:longtext"`                                        // 任务描述 (升级为 longtext)
	Rubric      string       `gorm:"column:rubric;type:longtext"`                          // 任务评分标准 (升级为 longtext)
	Submissions []Submission `gorm:"foreignKey:AssignmentID;constraint:OnDelete:CASCADE;"` // 学生提交，外键
}

// Submission 学生提交记录表
type Submission struct {
	MaterialBatchID    string  `gorm:"size:36;index"`
	GradingJobID       string  `gorm:"size:36;index"`
	ID                 uint    `gorm:"primaryKey"`                                                                                                  // 主键
	StudentID          string  `gorm:"size:64;index;index:idx_submissions_assignment_student,priority:2"`                                           // 学生学号
	StudentName        string  `gorm:"size:64"`                                                                                                     // 学生姓名
	Score              float64 `gorm:"index:idx_submissions_assignment_score,priority:2"`                                                           // 得分
	Feedback           string  `gorm:"type:longtext"`                                                                                               // ai反馈评价 (升级为 longtext)
	MergeContent       string  `gorm:"type:longtext"`                                                                                               // 学生提交合并内容 (升级为 longtext)
	PlagiarismReport   string  `gorm:"column:plagiarism_reports;type:longtext"`                                                                     // 抄袭报告，存为json格式 (升级为 longtext)
	AIGCReport         string  `gorm:"column:aigc_reports;type:longtext"`                                                                           // aigc检测报告，存为json格式 (升级为 longtext)
	CodeDocMatchReport string  `gorm:"column:code_doc_match_report;type:longtext"`                                                                  // 代码文本相似度检测，存为json格式 (升级为 longtext)
	AssignmentID       uint    `gorm:"index;index:idx_submissions_assignment_score,priority:1;index:idx_submissions_assignment_student,priority:1"` // 根据其分组
	IsHumanReviewed    bool    `gorm:"type:boolean;default:false;not null"`                                                                         // 教师是否复查
	HumanFeedback      string  `gorm:"type:longtext"`                                                                                               // 教师复查评语 (升级为 longtext)
	HumanScore         float64 // 教师评分
}
