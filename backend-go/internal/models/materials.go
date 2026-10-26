package models

import "time"

// MaterialBatch is a durable preparation task. Preparing files never implies grading.
type MaterialBatch struct {
	PortalClassID  string     `json:"portal_class_id,omitempty" gorm:"size:36;index"`
	ID             string     `json:"id" gorm:"size:36;primaryKey"`
	OwnerID        uint       `json:"-" gorm:"index;not null"`
	AssignmentID   uint       `json:"assignment_id" gorm:"index"`
	CourseName     string     `json:"course_name" gorm:"size:255"`
	AssignmentName string     `json:"assignment_name" gorm:"size:255"`
	ClassName      string     `json:"class_name" gorm:"size:255"`
	SourcePath     string     `json:"source_path" gorm:"type:text"`
	SourceSHA256   string     `json:"-" gorm:"size:64"`
	SourceJobID    string     `json:"source_job_id,omitempty" gorm:"size:36;index"`
	SourceFileID   string     `json:"source_file_id,omitempty" gorm:"size:40;index"`
	OutputDir      string     `json:"output_dir" gorm:"type:text"`
	Status         string     `json:"status" gorm:"size:32;index"`
	Message        string     `json:"message" gorm:"type:text"`
	ManifestJSON   string     `json:"-" gorm:"type:longtext"`
	ManifestSHA256 string     `json:"-" gorm:"size:64"`
	ManifestPath   string     `json:"-" gorm:"type:text"`
	StudentCount   int        `json:"student_count"`
	IssueCount     int        `json:"issue_count"`
	AutoGrade      bool       `json:"auto_grade"`
	GradingState   string     `json:"grading_state" gorm:"size:32;index"`
	GradingJobID   string     `json:"grading_job_id,omitempty" gorm:"size:36;index"`
	LeaseOwner     string     `json:"-" gorm:"size:36"`
	LeaseUntil     *time.Time `json:"-"`
	RequestHash    string     `json:"-" gorm:"size:64"`
	CreatedAt      time.Time  `json:"created_at"`
	UpdatedAt      time.Time  `json:"updated_at"`
}
