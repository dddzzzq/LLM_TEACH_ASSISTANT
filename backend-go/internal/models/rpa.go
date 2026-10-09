package models

import "time"

// RPAJob contains no credentials, cookies, page HTML or browser input.
type RPAJob struct {
	PendingAction      string     `json:"-" gorm:"type:longtext"`
	NavigationJSON     string     `json:"-" gorm:"type:longtext"`
	DecisionOwner      string     `json:"-" gorm:"size:36"`
	DecisionLeaseUntil *time.Time `json:"-"`

	ID             string    `json:"job_id" gorm:"type:char(36);primaryKey"`
	UserID         uint      `json:"-" gorm:"not null;index"`
	CourseName     string    `json:"course_name" gorm:"size:255;not null"`
	AssignmentName string    `json:"assignment_name" gorm:"size:255;not null"`
	Term           string    `json:"term" gorm:"size:255"`
	Status         string    `json:"status" gorm:"size:32;index"`
	Stage          string    `json:"stage" gorm:"size:32"`
	Message        string    `json:"message" gorm:"type:text"`
	StateJSON      string    `json:"-" gorm:"type:longtext"`
	StateVersion   int64     `json:"-"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}

type RPAFile struct {
	ID            string    `json:"id" gorm:"type:char(40);primaryKey"`
	JobID         string    `json:"job_id" gorm:"type:char(36);index;not null"`
	ExportRef     string    `json:"export_ref" gorm:"size:128"`
	ClassName     string    `json:"class_name" gorm:"size:255"`
	Path          string    `json:"path" gorm:"type:text"`
	SHA256        string    `json:"sha256" gorm:"size:64"`
	Size          int64     `json:"size"`
	AssignmentID  uint      `json:"assignment_id"`
	GradingJobID  string    `json:"grading_job_id" gorm:"size:36"`
	GradingStatus string    `json:"grading_status" gorm:"size:32;index"`
	Message       string    `json:"message" gorm:"type:text"`
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at"`
}
