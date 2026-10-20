package logbook

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	gormlogger "gorm.io/gorm/logger"
	"grading-gateway/internal/models"
)

type Service struct {
	DB           *gorm.DB
	Root         string
	Ready        bool
	mu           sync.Mutex
	rpaAt, jobAt time.Time
	rpaID, jobID string
	lastClean    time.Time
}

func New(db *gorm.DB, root string) *Service {
	return &Service{DB: db.Session(&gorm.Session{Logger: db.Logger.LogMode(gormlogger.Silent)}), Root: root, rpaAt: time.Now().AddDate(0, 0, -30), jobAt: time.Now().AddDate(0, 0, -30)}
}
func (s *Service) Init() error {
	err := s.DB.AutoMigrate(&Entry{}, &TaskLink{}, &FileCursor{})
	s.Ready = err == nil
	return err
}
func (s *Service) Run(ctx context.Context) {
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()
	for {
		if err := s.Sync(ctx); err != nil {
			log.Printf("[LogIndex] 日志索引暂未更新（%T）", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func (s *Service) Sync(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	ctx, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()
	if !s.Ready {
		return errors.New("index unavailable")
	}
	for _, source := range fileSources {
		if err := s.ingestFile(ctx, source); err != nil {
			return err
		}
	}
	if err := s.ingestTasks(ctx); err != nil {
		return err
	}
	if time.Since(s.lastClean) > time.Hour {
		if err := s.clean(ctx); err != nil {
			return err
		}
	}
	return nil
}

func (s *Service) store(tx *gorm.DB, entries []Entry) error {
	for start := 0; start < len(entries); start += 200 {
		end := min(start+200, len(entries))
		batch := entries[start:end]
		candidateIDs := []string{}
		for _, e := range batch {
			var ids []string
			_ = json.Unmarshal([]byte(e.TaskIDs), &ids)
			candidateIDs = append(candidateIDs, ids...)
		}
		known := map[string]bool{}
		if len(candidateIDs) > 0 {
			var rpaIDs, gradingIDs []string
			if err := tx.Model(&models.RPAJob{}).Where("id IN ?", candidateIDs).Pluck("id", &rpaIDs).Error; err != nil {
				return err
			}
			if err := tx.Model(&models.AsyncJob{}).Where("id IN ?", candidateIDs).Pluck("id", &gradingIDs).Error; err != nil {
				return err
			}
			for _, id := range append(rpaIDs, gradingIDs...) {
				known[id] = true
			}
		}
		for i := range batch {
			var ids []string
			_ = json.Unmarshal([]byte(batch[i].TaskIDs), &ids)
			valid := []string{}
			for _, id := range ids {
				if known[id] {
					valid = append(valid, id)
				}
			}
			raw, _ := json.Marshal(valid)
			batch[i].TaskIDs = string(raw)
		}
		if err := tx.Clauses(clause.OnConflict{DoNothing: true}).Create(&batch).Error; err != nil {
			return err
		}
		keys := []string{}
		for _, e := range batch {
			keys = append(keys, e.EventKey)
		}
		var saved []Entry
		if err := tx.Select("id", "task_ids").Where("event_key IN ?", keys).Find(&saved).Error; err != nil {
			return err
		}
		links := []TaskLink{}
		for _, e := range saved {
			var ids []string
			_ = json.Unmarshal([]byte(e.TaskIDs), &ids)
			for _, id := range ids {
				links = append(links, TaskLink{EntryID: e.ID, TaskID: id})
			}
		}
		if len(links) > 0 {
			if err := tx.Clauses(clause.OnConflict{DoNothing: true}).CreateInBatches(links, 200).Error; err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *Service) ingestFile(ctx context.Context, source string) error {
	path := filepath.Join(s.Root, source+".log")
	info, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return nil
	} // Never follow a log symlink to configuration files.
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	opened, err := file.Stat()
	if err != nil {
		return err
	}
	if !opened.Mode().IsRegular() || !os.SameFile(info, opened) {
		return nil
	}
	var cursor FileCursor
	err = s.DB.WithContext(ctx).First(&cursor, "source = ?", source).Error
	if err != nil && !errors.Is(err, gorm.ErrRecordNotFound) {
		return err
	}
	identity := fileIdentity(info)
	if cursor.FileID != identity || info.Size() < cursor.Offset || cursor.Generation == "" {
		cursor = FileCursor{Source: source, FileID: identity, Generation: digest(fmt.Sprint(identity, time.Now().UnixNano())), LastAt: time.Now()}
	}
	if cursor.Offset == info.Size() {
		return nil
	}
	if _, err = file.Seek(cursor.Offset, io.SeekStart); err != nil {
		return err
	}
	reader := bufio.NewReader(io.LimitReader(file, 2<<20))
	entries := []Entry{}
	for len(entries) < 2000 {
		start := cursor.Offset
		raw, readErr := reader.ReadString('\n')
		complete := strings.HasSuffix(raw, "\n")
		if !complete && start+int64(len(raw)) >= info.Size() {
			break
		} // An unfinished line is retried on the next tick.
		if raw == "" {
			break
		}
		cursor.Offset += int64(len(raw))
		if cursor.Discarding {
			cursor.Discarding = !complete
			if readErr != nil {
				break
			}
			continue
		}
		cursor.Line++
		if !complete {
			cursor.Discarding = true
		}
		if len(raw) > 16384 {
			raw = raw[:16384] + " [超长日志已截断]"
		}
		raw = strings.TrimRight(raw, "\r\n")
		if strings.Contains(raw, "-----BEGIN ") && strings.Contains(raw, "PRIVATE KEY-----") {
			cursor.PrivateBlock = true
		}
		text := Redact(raw)
		if cursor.PrivateBlock {
			text = "[私钥内容已隐藏]"
			if strings.Contains(raw, "-----END ") {
				cursor.PrivateBlock = false
			}
		}
		at, known := parseTime(raw)
		if known {
			cursor.LastAt = at
		}
		if !known {
			at = cursor.LastAt
			if at.IsZero() {
				at = time.Now()
			}
		}
		level := classify(text, cursor.LastLevel)
		cursor.LastLevel = level
		kind := "message"
		if strings.Contains(text, "[GIN]") || strings.Contains(text, "[GIN-debug]") {
			kind = "access"
		}
		ids := refs(text)
		encoded, _ := json.Marshal(ids)
		if strings.TrimSpace(text) != "" {
			entries = append(entries, Entry{EventKey: digest(fmt.Sprintf("%s:%s:%d", source, cursor.Generation, start)), Source: source, Generation: cursor.Generation, Line: cursor.Line, At: at, Estimated: !known, Level: level, Kind: kind, Message: text, TaskIDs: string(encoded)})
		}
		if readErr != nil {
			break
		}
	}
	return s.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := s.store(tx, entries); err != nil {
			return err
		}
		return tx.Save(&cursor).Error
	})
}

func taskEntry(key, id, ref, message string, at time.Time, level string) Entry {
	ids := []string{id}
	if ref != "" && ref != id {
		ids = append(ids, ref)
	}
	encoded, _ := json.Marshal(ids)
	return Entry{EventKey: digest(key), Source: "task", Generation: id, At: at, Level: level, Kind: "event", Message: Redact(message), TaskIDs: string(encoded)}
}

func (s *Service) ingestTasks(ctx context.Context) error {
	var jobs []models.RPAJob
	if err := s.DB.WithContext(ctx).Where("updated_at > ? OR (updated_at = ? AND id > ?)", s.rpaAt, s.rpaAt, s.rpaID).Order("updated_at ASC,id ASC").Limit(100).Find(&jobs).Error; err != nil {
		return err
	}
	entries := []Entry{}
	for _, job := range jobs {
		var state struct {
			Events []struct {
				Time    float64 `json:"time"`
				Message string  `json:"message"`
			} `json:"events"`
		}
		_ = json.Unmarshal([]byte(job.StateJSON), &state)
		for _, e := range state.Events {
			if e.Time <= 0 {
				continue
			}
			at := time.UnixMilli(int64(e.Time * 1000))
			entries = append(entries, taskEntry(fmt.Sprintf("rpa:%s:%.6f:%s", job.ID, e.Time, e.Message), job.ID, "", e.Message, at, classify(e.Message, "")))
		}
		entries = append(entries, taskEntry(fmt.Sprintf("rpa-state:%s:%d:%s", job.ID, job.StateVersion, job.Status), job.ID, "", fmt.Sprintf("抓取状态 %s · %s：%s", job.Status, job.Stage, job.Message), job.UpdatedAt, classify(job.Message, "")))
	}
	var grading []models.AsyncJob
	if err := s.DB.WithContext(ctx).Where("job_type IN ? AND (updated_at > ? OR (updated_at = ? AND id > ?))", []string{"HOMEWORK", "EXAM", "homework_materials"}, s.jobAt, s.jobAt, s.jobID).Order("updated_at ASC,id ASC").Limit(100).Find(&grading).Error; err != nil {
		return err
	}
	for _, job := range grading {
		ref := ""
		if taskIDs.MatchString(job.ReferenceID) {
			ref = job.ReferenceID
		}
		label := "批改"
		if job.JobType == "homework_materials" {
			label = "材料整理"
		}
		if job.MaterialBatchID != "" {
			ref = job.MaterialBatchID
		}
		entries = append(entries, taskEntry(fmt.Sprintf("grading:%s:%s:%s", job.ID, job.UpdatedAt.Format(time.RFC3339Nano), job.Status), job.ID, ref, fmt.Sprintf("%s状态 %s：%s", label, job.Status, job.Message), job.UpdatedAt, classify(job.Message, "")))
	}
	if err := s.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error { return s.store(tx, entries) }); err != nil {
		return err
	}
	if len(jobs) > 0 {
		last := jobs[len(jobs)-1]
		s.rpaAt, s.rpaID = last.UpdatedAt, last.ID
	}
	if len(grading) > 0 {
		last := grading[len(grading)-1]
		s.jobAt, s.jobID = last.UpdatedAt, last.ID
	}
	return nil
}

func (s *Service) clean(ctx context.Context) error {
	var ids []uint64
	if err := s.DB.WithContext(ctx).Model(&Entry{}).Where("at < ?", time.Now().AddDate(0, 0, -30)).Limit(2000).Pluck("id", &ids).Error; err != nil {
		return err
	}
	if len(ids) == 0 {
		s.lastClean = time.Now()
		return nil
	}
	err := s.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := tx.Where("entry_id IN ?", ids).Delete(&TaskLink{}).Error; err != nil {
			return err
		}
		return tx.Where("id IN ?", ids).Delete(&Entry{}).Error
	})
	// Keep draining full batches on the next tick instead of waiting an hour.
	if err == nil && len(ids) < 2000 {
		s.lastClean = time.Now()
	}
	return err
}
