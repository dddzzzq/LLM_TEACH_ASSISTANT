package schemas

// ExamCreate 创建试卷请求
type ExamCreate struct {
	Name string `json:"name" binding:"required"`
}

// ExamQuestionCreate 添加试卷题目请求
type ExamQuestionCreate struct {
	QuestionNumber int     `json:"question_number" binding:"required"`
	QuestionText   string  `json:"question_text" binding:"required"`
	StandardAnswer string  `json:"standard_answer" binding:"required"`
	Rubric         string  `json:"rubric" binding:"required"` // 纯文本形式的评分标准
	MaxScore       float64 `json:"max_score" binding:"required"`
}

type ExamResultSummary struct {
	StudentID     string  `json:"student_id"`
	TotalScore    float64 `json:"total_score"`
	StudentExamID uint    `json:"student_exam_id"`
}
