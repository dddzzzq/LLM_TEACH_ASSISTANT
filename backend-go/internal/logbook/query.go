package logbook

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"gorm.io/gorm"
	"grading-gateway/internal/models"
)

var ErrScope = errors.New("任务不存在或无权查看日志")
var ErrQuery = errors.New("日志筛选参数无效")

type Actor struct {
	UserID uint
	Admin  bool
}
type Filter struct {
	Source   string    `json:"source"`
	Level    string    `json:"level"`
	TaskID   string    `json:"task_id"`
	Keyword  string    `json:"q"`
	From     time.Time `json:"from"`
	To       time.Time `json:"to"`
	HideHTTP bool      `json:"hide_http"`
}
type Page struct {
	Items      []Entry `json:"items"`
	NextCursor string  `json:"next_cursor"`
	HasMore    bool    `json:"has_more"`
}
type pageCursor struct {
	At     time.Time `json:"at"`
	ID     uint64    `json:"id"`
	Filter string    `json:"filter"`
}

func (s *Service) scope(ctx context.Context, actor Actor, taskID string) ([]string, error) {
	if actor.UserID == 0 {
		return nil, ErrScope
	}
	db := s.DB.WithContext(ctx)
	if taskID != "" {
		if len(taskID) > 64 {
			return nil, ErrQuery
		}
		var job models.RPAJob
		q := db.Where("id = ?", taskID)
		if !actor.Admin {
			q = q.Where("user_id = ?", actor.UserID)
		}
		err := q.First(&job).Error
		if err == nil {
			ids := []string{job.ID}
			var related []string
			if err := db.Model(&models.RPAFile{}).Where("job_id = ? AND grading_job_id <> ''", job.ID).Pluck("grading_job_id", &related).Error; err != nil {
				return nil, err
			}
			return append(ids, related...), nil
		}
		if !errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, err
		}
		var grading models.AsyncJob
		q = db.Where("id = ?", taskID)
		if !actor.Admin {
			q = q.Where("owner_id = ? OR reference_id IN (SELECT id FROM rpa_jobs WHERE user_id = ?)", actor.UserID, actor.UserID)
		}
		if err := q.First(&grading).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return nil, ErrScope
			}
			return nil, err
		}
		ids := []string{grading.ID}
		parent := db.Where("id = ?", grading.ReferenceID)
		if !actor.Admin {
			parent = parent.Where("user_id = ?", actor.UserID)
		}
		if parent.First(&job).Error == nil {
			ids = append(ids, job.ID)
		}
		return ids, nil
	}
	if actor.Admin {
		return nil, nil
	}
	var ids, gradingIDs []string
	if err := db.Model(&models.RPAJob{}).Where("user_id = ?", actor.UserID).Pluck("id", &ids).Error; err != nil {
		return nil, err
	}
	if err := db.Model(&models.AsyncJob{}).Where("owner_id = ? OR reference_id IN (SELECT id FROM rpa_jobs WHERE user_id = ?)", actor.UserID, actor.UserID).Pluck("id", &gradingIDs).Error; err != nil {
		return nil, err
	}
	return append(append([]string{}, ids...), gradingIDs...), nil
}

func restrict(q *gorm.DB, ids []string) *gorm.DB {
	if ids == nil {
		return q
	}
	if len(ids) == 0 {
		return q.Where("1 = 0")
	}
	return q.Where("id IN (SELECT entry_id FROM log_task_links WHERE task_id IN ?)", ids)
}

func (s *Service) query(ctx context.Context, actor Actor, f Filter) (*gorm.DB, error) {
	if utf8.RuneCountInString(f.Keyword) > 300 || len(strings.Fields(f.Keyword)) > 8 || f.From.IsZero() || f.To.IsZero() || f.From.After(f.To) {
		return nil, ErrQuery
	}
	if f.Source != "" && f.Source != "all" {
		if _, ok := sourceNames[f.Source]; !ok {
			return nil, ErrQuery
		}
	}
	if !actor.Admin && (f.Source == "kafka" || f.Source == "frontend") {
		return nil, ErrScope
	}
	ids, err := s.scope(ctx, actor, f.TaskID)
	if err != nil {
		return nil, err
	}
	q := restrict(s.DB.WithContext(ctx).Model(&Entry{}), ids).Where("at >= ? AND at <= ?", f.From, f.To)
	if f.Source != "" && f.Source != "all" {
		q = q.Where("source = ?", f.Source)
	}
	if !actor.Admin {
		q = q.Where("source NOT IN ?", []string{"kafka", "frontend"})
	}
	if f.Level != "" && f.Level != "all" {
		if f.Level != "ERROR" && f.Level != "WARN" && f.Level != "INFO" && f.Level != "DEBUG" {
			return nil, ErrQuery
		}
		q = q.Where("level = ?", f.Level)
	}
	if f.HideHTTP {
		q = q.Where("kind <> ?", "access")
	}
	for _, term := range strings.Fields(f.Keyword) {
		term = strings.NewReplacer("!", "!!", "%", "!%", "_", "!_").Replace(term)
		q = q.Where("message LIKE ? ESCAPE '!'", "%"+term+"%")
	}
	return q, nil
}

func hydrate(entries []Entry) []Entry {
	if entries == nil {
		entries = []Entry{}
	}
	for i := range entries {
		entries[i].Message = Redact(entries[i].Message)
		entries[i].Tasks = []string{}
		_ = json.Unmarshal([]byte(entries[i].TaskIDs), &entries[i].Tasks)
	}
	return entries
}

func (s *Service) Search(ctx context.Context, actor Actor, f Filter, cursor string, limit int) (Page, error) {
	q, err := s.query(ctx, actor, f)
	if err != nil {
		return Page{}, err
	}
	if limit < 1 || limit > 200 || len(cursor) > 1000 {
		return Page{}, ErrQuery
	}
	encoded, _ := json.Marshal(f)
	fingerprint := digest(fmt.Sprintf("%d:%t:%s", actor.UserID, actor.Admin, encoded))
	if cursor != "" {
		raw, err := base64.RawURLEncoding.DecodeString(cursor)
		if err != nil {
			return Page{}, ErrQuery
		}
		var c pageCursor
		if json.Unmarshal(raw, &c) != nil || c.Filter != fingerprint || c.ID == 0 || c.At.IsZero() {
			return Page{}, ErrQuery
		}
		q = q.Where("at < ? OR (at = ? AND id < ?)", c.At, c.At, c.ID)
	}
	var entries []Entry
	if err := q.Order("at DESC,id DESC").Limit(limit + 1).Find(&entries).Error; err != nil {
		return Page{}, err
	}
	page := Page{HasMore: len(entries) > limit}
	if page.HasMore {
		entries = entries[:limit]
		last := entries[len(entries)-1]
		raw, _ := json.Marshal(pageCursor{At: last.At, ID: last.ID, Filter: fingerprint})
		page.NextCursor = base64.RawURLEncoding.EncodeToString(raw)
	}
	page.Items = hydrate(entries)
	return page, nil
}

func (s *Service) Context(ctx context.Context, actor Actor, id uint64) ([]Entry, error) {
	ids, err := s.scope(ctx, actor, "")
	if err != nil {
		return nil, err
	}
	base := func() *gorm.DB {
		q := restrict(s.DB.WithContext(ctx).Model(&Entry{}), ids)
		if !actor.Admin {
			q = q.Where("source NOT IN ?", []string{"kafka", "frontend"})
		}
		return q
	}
	var entry Entry
	if err := base().Where("id = ?", id).First(&entry).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, ErrScope
		}
		return nil, err
	}
	var entries []Entry
	if entry.Source == "task" {
		var before, after []Entry
		if err := base().Where("source = ? AND generation = ? AND id < ?", entry.Source, entry.Generation, entry.ID).Order("id DESC").Limit(8).Find(&before).Error; err != nil {
			return nil, err
		}
		if err := base().Where("source = ? AND generation = ? AND id >= ?", entry.Source, entry.Generation, entry.ID).Order("id ASC").Limit(9).Find(&after).Error; err != nil {
			return nil, err
		}
		for i := len(before) - 1; i >= 0; i-- {
			entries = append(entries, before[i])
		}
		entries = append(entries, after...)
	} else {
		if err := base().Where("source = ? AND generation = ? AND line BETWEEN ? AND ?", entry.Source, entry.Generation, max(1, entry.Line-8), entry.Line+8).Order("line ASC").Limit(17).Find(&entries).Error; err != nil {
			return nil, err
		}
	}
	return hydrate(entries), nil
}

func (s *Service) Export(ctx context.Context, actor Actor, f Filter) ([]Entry, error) {
	q, err := s.query(ctx, actor, f)
	if err != nil {
		return nil, err
	}
	var entries []Entry
	if err := q.Order("at DESC,id DESC").Limit(2000).Find(&entries).Error; err != nil {
		return nil, err
	}
	return hydrate(entries), nil
}

func (s *Service) Sources(ctx context.Context, actor Actor) ([]SourceInfo, error) {
	var cursors []FileCursor
	if err := s.DB.WithContext(ctx).Find(&cursors).Error; err != nil {
		return nil, err
	}
	indexed := map[string]int64{}
	for _, c := range cursors {
		indexed[c.Source] = c.Offset
	}
	sources := []SourceInfo{{ID: "task", Name: sourceNames["task"], Available: true}}
	for _, source := range fileSources {
		if !actor.Admin && (source == "kafka" || source == "frontend") {
			continue
		}
		item := SourceInfo{ID: source, Name: sourceNames[source], Indexed: indexed[source]}
		if info, err := os.Lstat(filepath.Join(s.Root, source+".log")); err == nil && info.Mode().IsRegular() {
			item.Available = true
			item.Size = info.Size()
		}
		sources = append(sources, item)
	}
	return sources, nil
}

type TaskOption struct {
	ID       string    `json:"id"`
	ParentID string    `json:"parent_id,omitempty"`
	Kind     string    `json:"kind"`
	Course   string    `json:"course"`
	Name     string    `json:"name"`
	Class    string    `json:"class_name"`
	Status   string    `json:"status"`
	At       time.Time `json:"updated_at"`
}

func (s *Service) Tasks(ctx context.Context, actor Actor) ([]TaskOption, error) {
	db := s.DB.WithContext(ctx)
	var jobs []models.RPAJob
	q := db
	if !actor.Admin {
		q = q.Where("user_id = ?", actor.UserID)
	}
	if err := q.Order("updated_at DESC").Limit(60).Find(&jobs).Error; err != nil {
		return nil, err
	}
	byID := map[string]models.RPAJob{}
	out := []TaskOption{}
	for _, job := range jobs {
		byID[job.ID] = job
		out = append(out, TaskOption{ID: job.ID, Kind: "抓取", Course: job.CourseName, Name: job.AssignmentName, Status: job.Status, At: job.UpdatedAt})
	}
	var grading []models.AsyncJob
	q = db.Where("job_type IN ?", []string{"HOMEWORK", "EXAM", "homework_materials"})
	if !actor.Admin {
		q = q.Where("owner_id = ? OR reference_id IN (SELECT id FROM rpa_jobs WHERE user_id = ?)", actor.UserID, actor.UserID)
	}
	if err := q.Order("updated_at DESC").Limit(60).Find(&grading).Error; err != nil {
		return nil, err
	}
	gradingIDs := []string{}
	for _, job := range grading {
		gradingIDs = append(gradingIDs, job.ID)
	}
	classes := map[string]string{}
	if len(gradingIDs) > 0 {
		var files []models.RPAFile
		if err := db.Select("grading_job_id", "class_name").Where("grading_job_id IN ?", gradingIDs).Find(&files).Error; err != nil {
			return nil, err
		}
		for _, file := range files {
			classes[file.GradingJobID] = file.ClassName
		}
	}
	for _, job := range grading {
		item := TaskOption{ID: job.ID, Kind: "批改", Name: job.ReferenceID, Status: string(job.Status), At: job.UpdatedAt}
		if job.JobType == "homework_materials" {
			item.Kind = "材料整理"
		}
		if parent, ok := byID[job.ReferenceID]; ok {
			item.Course, item.Name, item.ParentID = parent.CourseName, parent.AssignmentName, parent.ID
		}
		item.Class = classes[job.ID]
		out = append(out, item)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].At.After(out[j].At) })
	if len(out) > 100 {
		out = out[:100]
	}
	return out, nil
}
