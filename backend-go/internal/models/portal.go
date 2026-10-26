package models

import "time"

// PortalOperation is a durable outbox and checkpoint for authentication or catalog reads.
// Neither PayloadJSON nor events may contain passwords, OTPs, cookies or tickets.
type PortalOperation struct {
	ID                string    `json:"id" gorm:"type:char(36);primaryKey"`
	ConnectionID      string    `json:"connection_id" gorm:"size:36;not null;index"`
	OwnerID           uint      `json:"-" gorm:"not null;index"`
	Type              string    `json:"type" gorm:"size:32;not null"`
	Status            string    `json:"status" gorm:"size:32;not null;index"`
	Stage             string    `json:"stage" gorm:"size:64"`
	Message           string    `json:"message" gorm:"size:512"`
	ErrorCode         string    `json:"error_code" gorm:"size:64"`
	CredentialVersion uint64    `json:"-"`
	ProfileGeneration uint64    `json:"-"`
	StateVersion      uint64    `json:"-"`
	Sequence          uint64    `json:"-"`
	PayloadJSON       string    `json:"-" gorm:"type:longtext"`
	CreatedAt         time.Time `json:"created_at"`
	UpdatedAt         time.Time `json:"updated_at"`
}

type PortalOperationEvent struct {
	ID          uint64    `json:"id" gorm:"primaryKey"`
	OperationID string    `json:"operation_id" gorm:"size:36;not null;uniqueIndex:idx_portal_event_seq,priority:1"`
	Seq         uint64    `json:"seq" gorm:"not null;uniqueIndex:idx_portal_event_seq,priority:2"`
	Version     uint      `json:"version" gorm:"not null;default:1"`
	Type        string    `json:"type" gorm:"size:64;not null"`
	PayloadJSON string    `json:"payload" gorm:"type:text"`
	CreatedAt   time.Time `json:"created_at"`
}

type PortalCatalogSnapshot struct {
	ContentJSON   string    `json:"-" gorm:"type:longtext"`
	ID            string    `json:"id" gorm:"type:char(36);primaryKey"`
	ConnectionID  string    `json:"connection_id" gorm:"size:36;not null;index"`
	OwnerID       uint      `json:"-" gorm:"not null;index"`
	CourseCount   int       `json:"course_count"`
	HomeworkCount int       `json:"homework_count"`
	CreatedAt     time.Time `json:"created_at"`
}

type PortalCourse struct {
	ID           string `json:"id" gorm:"type:char(36);primaryKey"`
	ConnectionID string `json:"-" gorm:"size:36;not null;index"`
	SnapshotID   string `json:"snapshot_id" gorm:"size:36;not null;index"`
	PlatformID   string `json:"platform_id" gorm:"size:255;not null"`
	Title        string `json:"title" gorm:"size:255;not null"`
	Term         string `json:"term" gorm:"size:128"`
	URL          string `json:"-" gorm:"type:text"`
}

type PortalHomework struct {
	ID           string `json:"id" gorm:"type:char(36);primaryKey"`
	ConnectionID string `json:"-" gorm:"size:36;not null;index"`
	CourseID     string `json:"course_id" gorm:"size:36;not null;index"`
	SnapshotID   string `json:"snapshot_id" gorm:"size:36;not null;index"`
	PlatformID   string `json:"platform_id" gorm:"size:255;not null"`
	Title        string `json:"title" gorm:"size:255;not null"`
	State        string `json:"state" gorm:"size:64"`
	Deadline     string `json:"deadline" gorm:"size:128"`
	URL          string `json:"-" gorm:"type:text"`
}

// Detail cache is separate from catalog rows, which are replaced on every sync.
// Original resource paths are private; only owner-checked handlers serve them.
type PortalHomeworkDetail struct {
	HomeworkID        string    `json:"homework_id" gorm:"type:char(36);primaryKey"`
	OwnerID           uint      `json:"-" gorm:"not null;index"`
	ConnectionID      string    `json:"-" gorm:"size:36;not null;index"`
	ProfileGeneration uint64    `json:"-"`
	SnapshotID        string    `json:"snapshot_id" gorm:"size:36;not null"`
	OperationID       string    `json:"operation_id" gorm:"size:36;not null"`
	Question          string    `json:"question" gorm:"type:longtext"`
	Complete          bool      `json:"complete"`
	ContentJSON       string    `json:"-" gorm:"type:longtext"`
	UpdatedAt         time.Time `json:"updated_at"`
}

type PortalCourseClass struct {
	ID         string `json:"id" gorm:"type:char(36);primaryKey"`
	CourseID   string `json:"course_id" gorm:"size:36;not null;index"`
	SnapshotID string `json:"snapshot_id" gorm:"size:36;not null;index"`
	PlatformID string `json:"platform_id" gorm:"size:255;not null"`
	Name       string `json:"name" gorm:"size:255;not null"`
}

type PortalAssignmentBinding struct {
	ID           string `json:"id" gorm:"type:char(36);primaryKey"`
	OwnerID      uint   `json:"-" gorm:"not null;index"`
	ConnectionID string `json:"-" gorm:"size:36;not null;index"`
	HomeworkID   string `json:"homework_id" gorm:"size:36;not null"`
	ClassID      string `json:"class_id" gorm:"size:36;not null"`
	AssignmentID uint   `json:"assignment_id" gorm:"not null"`
}

type PortalWorkflow struct {
	ProfileGeneration uint64    `json:"-" gorm:"not null;default:1"`
	ID                string    `json:"id" gorm:"type:char(36);primaryKey"`
	OwnerID           uint      `json:"-" gorm:"not null;index"`
	ConnectionID      string    `json:"connection_id" gorm:"size:36;not null;index"`
	Status            string    `json:"status" gorm:"size:32;not null;index"`
	RequestHash       string    `json:"-" gorm:"size:64;not null"`
	CreatedAt         time.Time `json:"created_at"`
	UpdatedAt         time.Time `json:"updated_at"`
}

type PortalWorkflowItem struct {
	ID           string    `json:"id" gorm:"type:char(36);primaryKey"`
	WorkflowID   string    `json:"workflow_id" gorm:"size:36;not null;index"`
	OwnerID      uint      `json:"-" gorm:"not null;index"`
	SnapshotID   string    `json:"snapshot_id" gorm:"size:36;not null"`
	HomeworkID   string    `json:"homework_id" gorm:"size:36;not null"`
	ClassID      string    `json:"class_id" gorm:"size:36;not null"`
	AssignmentID uint      `json:"assignment_id"`
	CourseName   string    `json:"course_name" gorm:"size:255"`
	HomeworkName string    `json:"homework_name" gorm:"size:255"`
	ClassName    string    `json:"class_name" gorm:"size:255"`
	Question     string    `json:"-" gorm:"type:longtext"`
	Rubric       string    `json:"-" gorm:"type:longtext"`
	TargetJSON   string    `json:"-" gorm:"type:longtext"`
	Status       string    `json:"status" gorm:"size:32;not null;index"`
	Message      string    `json:"message" gorm:"size:512"`
	RPAJobID     string    `json:"rpa_job_id" gorm:"size:36"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
}
