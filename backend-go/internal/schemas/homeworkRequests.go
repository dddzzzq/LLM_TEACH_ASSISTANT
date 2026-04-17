package schemas

// AssignmentCreate 对应前端新建作业时发送的 JSON
type AssignmentCreate struct {
	CourseName string                 `json:"course_name" binding:"required"`
	ClassName  string                 `json:"class_name" binding:"required"`
	TaskName   string                 `json:"task_name" binding:"required"`
	Question   string                 `json:"question" binding:"required"`
	Rubric     map[string]interface{} `json:"rubric" binding:"required"` // 前端传来的是 JSON 对象
}

// AssignmentResponse 对应返回给前端的作业数据
type AssignmentResponse struct {
	ID         uint                   `json:"id"`
	CourseName string                 `json:"course_name"`
	ClassName  string                 `json:"class_name"`
	TaskName   string                 `json:"task_name"`
	Question   string                 `json:"question"`
	Rubric     map[string]interface{} `json:"rubric"`
}

// 更新Submission 对应前端发送的 JSON
type SubmissionUpdate struct {
	Score           *float64 `json:"score"`
	Feedback        *string  `json:"feedback"`
	HumanScore      *float64 `json:"human_score"`
	HumanFeedback   *string  `json:"human_feedback"`
	IsHumanReviewed *bool    `json:"is_human_reviewed"`
}
