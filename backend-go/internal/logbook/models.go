package logbook

import "time"

type Entry struct {
	ID         uint64    `json:"id" gorm:"primaryKey"`
	EventKey   string    `json:"-" gorm:"size:64;uniqueIndex"`
	Source     string    `json:"source" gorm:"size:16;index:idx_log_source_time,priority:1;index:idx_log_context,priority:1"`
	Generation string    `json:"-" gorm:"size:64;index:idx_log_context,priority:2"`
	Line       int64     `json:"line" gorm:"index:idx_log_context,priority:3"`
	At         time.Time `json:"time" gorm:"index;index:idx_log_source_time,priority:2;index:idx_log_level_time,priority:2"`
	Estimated  bool      `json:"estimated_time"`
	Level      string    `json:"level" gorm:"size:8;index:idx_log_level_time,priority:1"`
	Kind       string    `json:"kind" gorm:"size:12"`
	Message    string    `json:"message" gorm:"type:text"`
	TaskIDs    string    `json:"-" gorm:"type:text"`
	Tasks      []string  `json:"task_ids" gorm:"-"`
}

func (Entry) TableName() string { return "log_entries" }

type TaskLink struct {
	EntryID uint64 `gorm:"primaryKey"`
	TaskID  string `gorm:"size:64;primaryKey;index"`
}

func (TaskLink) TableName() string { return "log_task_links" }

type FileCursor struct {
	Source       string `gorm:"size:16;primaryKey"`
	FileID       string `gorm:"size:128"`
	Generation   string `gorm:"size:64"`
	Offset       int64
	Line         int64
	LastAt       time.Time
	LastLevel    string `gorm:"size:8"`
	Discarding   bool
	PrivateBlock bool
	UpdatedAt    time.Time
}

func (FileCursor) TableName() string { return "log_file_cursors" }

type SourceInfo struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Available bool   `json:"available"`
	Size      int64  `json:"size"`
	Indexed   int64  `json:"indexed_bytes"`
}

var sourceNames = map[string]string{"backend": "后端与批改", "ai": "AI 与模型", "rpa": "教务抓取", "kafka": "消息队列", "frontend": "前端服务", "task": "任务事件"}
var fileSources = []string{"backend", "rpa", "ai", "kafka", "frontend"}
