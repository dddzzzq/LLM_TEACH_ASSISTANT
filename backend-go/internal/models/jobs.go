package models

import "time"

// AsyncJobStatus 异步任务状态枚举
type AsyncJobStatus string

const (
	JobStatusPending    AsyncJobStatus = "PENDING"
	JobStatusProcessing AsyncJobStatus = "PROCESSING"
	JobStatusSuccess    AsyncJobStatus = "SUCCESS"
	JobStatusFailed     AsyncJobStatus = "FAILED"
	JobStatusCancelled  AsyncJobStatus = "CANCELLED"
)

// AsyncJobType 异步任务类型枚举
type AsyncJobType string

const (
	JobTypeHomework AsyncJobType = "HOMEWORK"
	JobTypeExam     AsyncJobType = "EXAM"
)

// AsyncJob 异步任务表
type AsyncJob struct {
	MaterialBatchID  string         `json:"material_batch_id,omitempty" gorm:"size:36;index"`
	PayloadJSON      string         `json:"-" gorm:"type:longtext"`
	DispatchState    string         `json:"-" gorm:"size:16;index"`
	QuestionSnapshot string         `json:"-" gorm:"type:longtext"`
	RubricSnapshot   string         `json:"-" gorm:"type:longtext"`
	OwnerID          uint           `json:"owner_id,omitempty" gorm:"index;default:0"`
	SkillName        string         `json:"skill_name,omitempty" gorm:"size:64"`
	SkillVersion     string         `json:"skill_version,omitempty" gorm:"size:64"`
	ID               string         `json:"id" gorm:"type:char(36);primaryKey"`                   // UUID 主键
	JobType          AsyncJobType   `json:"job_type" gorm:"type:varchar(20);not null;index"`      // 任务类型: HOMEWORK 或 EXAM
	ReferenceID      string         `json:"reference_id" gorm:"type:varchar(255);not null;index"` // 关联作业或考试ID
	StudentID        string         `json:"student_id" gorm:"type:varchar(100);index"`            // 针对试卷的学生ID
	Status           AsyncJobStatus `json:"status" gorm:"type:varchar(20);not null;index"`        // 状态: PENDING, PROCESSING, SUCCESS, FAILED
	Message          string         `json:"message" gorm:"type:text"`                             // 错误信息等
	CreatedAt        time.Time      `json:"created_at" gorm:"autoCreateTime"`                     // 创建时间
	UpdatedAt        time.Time      `json:"updated_at" gorm:"autoUpdateTime"`                     // 更新时间
}
