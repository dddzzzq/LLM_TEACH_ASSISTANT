package mq

// ============================
//    RPA 抓取任务消息
// ============================

// RPAFetchMessage RPA抓取任务消息结构
// 用于Agent投递抓取任务到 topic_rpa_fetch 队列
type RPAFetchMessage struct {
	JobID          string `json:"job_id"`
	Username       string `json:"username"`
	Password       string `json:"password"`
	CourseName     string `json:"course_name"`
	AssignmentName string `json:"assignment_name"`
}

// ============================
//    作业批改任务消息
// ============================

// HomeworkTaskMessage 作业批改任务消息结构
type HomeworkTaskMessage struct {
	JobID        string `json:"job_id"`
	AssignmentID uint   `json:"assignment_id"` // 改为uint类型
	ZipPath      string `json:"zip_path"`
}

// ============================
//    试卷批改任务消息
// ============================

// ExamTaskMessage 试卷批改任务消息结构
type ExamTaskMessage struct {
	JobID      string   `json:"job_id"`
	ExamID     string   `json:"exam_id"`
	StudentID  string   `json:"student_id"`
	ImagePaths []string `json:"image_paths"`
}
