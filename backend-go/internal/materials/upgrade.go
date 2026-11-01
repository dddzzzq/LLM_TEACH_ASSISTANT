package materials

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"grading-gateway/internal/models"
)

// EnsureCurrentReader refreshes old manifests from retained originals without
// changing results or their teacher reviews. Previous prepared roots stay intact.
func (s *Service) EnsureCurrentReader(ctx context.Context, actor Actor, id string) (models.MaterialBatch, error) {
	batch, err := s.Owned(ctx, actor, id)
	if err != nil {
		return batch, err
	}
	current, err := LoadManifest(batch.ManifestPath, batch.ManifestSHA256)
	if err != nil {
		return batch, err
	}
	if current.ReaderVersion >= CurrentReaderVersion {
		return batch, nil
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	err = s.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&batch, "id = ? AND owner_id = ?", id, actor.UserID).Error; err != nil {
			return err
		}
		if batch.Status != "READY" && batch.Status != "NEEDS_REVIEW" {
			return fmt.Errorf("材料未就绪，不能更新读取清单")
		}
		if batch.GradingJobID != "" {
			var job models.AsyncJob
			if err := tx.First(&job, "id = ?", batch.GradingJobID).Error; err != nil {
				return err
			}
			if job.Status != models.JobStatusSuccess && job.Status != models.JobStatusFailed && job.Status != models.JobStatusCancelled {
				return fmt.Errorf("批次正在批改，稍后再更新材料清单")
			}
		}
		old, err := LoadManifest(batch.ManifestPath, batch.ManifestSHA256)
		if err != nil {
			return err
		}
		if old.ReaderVersion >= CurrentReaderVersion {
			return nil
		}
		output, err := s.allowed(batch.OutputDir)
		if err != nil {
			return err
		}
		root := filepath.Join(output, uuid.NewString())
		published := false
		defer func() {
			if !published {
				_ = os.RemoveAll(root)
			}
		}()
		sourceBatch := batch
		retained := filepath.Join(filepath.Dir(batch.ManifestPath), "original", filepath.Base(batch.SourcePath))
		if info, err := os.Stat(retained); err == nil && info.Mode().IsRegular() {
			sourceBatch.SourcePath = retained
		}
		if _, err := s.allowed(sourceBatch.SourcePath); err != nil {
			return err
		}
		source, err := snapshotSource(ctx, sourceBatch, root)
		if err != nil {
			return err
		}
		updated, err := prepare(ctx, batch.ID, batch.AssignmentID, source, root)
		if err != nil {
			return err
		}
		// Updating supported formats may add files, but cannot silently replace
		// or discard material that was used by previous scoring/reviews.
		files := map[string]string{}
		for _, student := range updated.Students {
			for _, file := range student.Files {
				files[file.Path] = file.SHA256
			}
		}
		for _, student := range old.Students {
			for _, file := range student.Files {
				if files[file.Path] != file.SHA256 {
					return fmt.Errorf("原材料已变化，不能自动更新；请新建材料批次")
				}
			}
		}
		raw, err := json.MarshalIndent(updated, "", "  ")
		if err != nil {
			return err
		}
		path := filepath.Join(root, "manifest.json")
		if err := os.WriteFile(path, raw, 0600); err != nil {
			return err
		}
		hash, err := HashFile(path)
		if err != nil {
			return err
		}
		status := "READY"
		if len(updated.Issues) > 0 {
			status = "NEEDS_REVIEW"
		}
		if err := tx.Model(&batch).Updates(map[string]any{"status": status, "manifest_path": path, "manifest_sha256": hash, "manifest_json": string(raw), "student_count": len(updated.Students), "issue_count": len(updated.Issues), "message": "材料清单已更新，保留原始附件来源及新增支持的脚本文件"}).Error; err != nil {
			return err
		}
		if err := tx.First(&batch, "id = ?", id).Error; err != nil {
			return err
		}
		published = true
		return nil
	})
	return batch, err
}
