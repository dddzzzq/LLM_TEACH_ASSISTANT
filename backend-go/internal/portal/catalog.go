package portal

import (
	"encoding/json"
	"errors"
	"net/url"
	"strings"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"grading-gateway/internal/models"
)

type CatalogClass struct {
	PlatformID string `json:"platform_id"`
	Name       string `json:"name"`
}
type CatalogHomework struct {
	PlatformID string `json:"platform_id"`
	Title      string `json:"title"`
	State      string `json:"state"`
	Deadline   string `json:"deadline"`
	URL        string `json:"url"`
}
type CatalogCourse struct {
	PlatformID string            `json:"platform_id"`
	Title      string            `json:"title"`
	Term       string            `json:"term"`
	URL        string            `json:"url"`
	Classes    []CatalogClass    `json:"classes"`
	Homeworks  []CatalogHomework `json:"homeworks"`
}
type Catalog struct {
	Complete bool            `json:"complete"`
	Courses  []CatalogCourse `json:"courses"`
}

func targetURL(raw string) bool {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme != "https" || u.User != nil || u.Fragment != "" || u.Port() != "" {
		return false
	}
	h := u.Hostname()
	if h != "learning.xidian.edu.cn" && h != "chaoxing.com" && !strings.HasSuffix(h, ".chaoxing.com") && !strings.HasSuffix(h, ".chaoxing.com.cn") {
		return false
	}
	if strings.Contains(strings.ToLower(u.Path), "login") || strings.Contains(strings.ToLower(u.Path), "logout") {
		return false
	}
	for key := range u.Query() {
		switch strings.ToLower(key) {
		case "courseid", "clazzid", "classid", "workid", "homeworkid", "taskid", "cpi", "ut", "mooc2", "enc", "t", "type", "page", "pageindex", "pagesize", "roleid":
		default:
			return false
		}
	}
	// `enc` is a portal routing signature, never a credential release destination.
	return len(raw) <= 4096
}
func validText(value string, max int) bool {
	return strings.TrimSpace(value) != "" && len([]rune(value)) <= max
}
func validateCatalog(c Catalog) error {
	if !c.Complete || c.Courses == nil || len(c.Courses) > 500 {
		return errors.New("目录未完整读取，保留上次成功目录")
	}
	seen := map[string]bool{}
	for _, course := range c.Courses {
		if !validText(course.PlatformID, 255) || strings.HasPrefix(course.PlatformID, "pending:") || !validText(course.Title, 255) || len([]rune(course.Term)) > 128 || (course.URL != "" && !targetURL(course.URL)) || seen[course.PlatformID] || len(course.Homeworks) > 2000 || len(course.Classes) > 500 {
			return errors.New("课程目录结构无效")
		}
		seen[course.PlatformID] = true
		works := map[string]bool{}
		classes := map[string]bool{}
		for _, work := range course.Homeworks {
			if !validText(work.PlatformID, 255) || strings.HasPrefix(work.PlatformID, "pending:") || !validText(work.Title, 255) || (work.URL != "" && !targetURL(work.URL)) || works[work.PlatformID] || len([]rune(work.State)) > 64 || len([]rune(work.Deadline)) > 128 {
				return errors.New("作业目录结构无效")
			}
			works[work.PlatformID] = true
		}
		for _, class := range course.Classes {
			if !validText(class.PlatformID, 255) || !validText(class.Name, 255) || classes[class.PlatformID] {
				return errors.New("班级目录结构无效")
			}
			classes[class.PlatformID] = true
		}
	}
	return nil
}
func publishCatalog(tx *gorm.DB, c *models.PortalConnection, catalog Catalog) error {
	if err := validateCatalog(catalog); err != nil {
		return err
	}
	snapshot := models.PortalCatalogSnapshot{ID: uuid.NewString(), ConnectionID: c.ID, OwnerID: c.OwnerID, CourseCount: len(catalog.Courses)}
	for _, course := range catalog.Courses {
		snapshot.HomeworkCount += len(course.Homeworks)
	}
	raw, _ := json.Marshal(catalog)
	snapshot.ContentJSON = string(raw)
	if err := tx.Create(&snapshot).Error; err != nil {
		return err
	}
	for _, entry := range catalog.Courses {
		course := models.PortalCourse{ID: stableID(c.ID, "course", entry.PlatformID), ConnectionID: c.ID, SnapshotID: snapshot.ID, PlatformID: entry.PlatformID, Title: entry.Title, Term: entry.Term, URL: entry.URL}
		if err := tx.Clauses(clause.OnConflict{UpdateAll: true}).Create(&course).Error; err != nil {
			return err
		}
		for _, value := range entry.Classes {
			row := models.PortalCourseClass{ID: stableID(c.ID, "class", entry.PlatformID+":"+value.PlatformID), CourseID: course.ID, SnapshotID: snapshot.ID, PlatformID: value.PlatformID, Name: value.Name}
			if err := tx.Clauses(clause.OnConflict{UpdateAll: true}).Create(&row).Error; err != nil {
				return err
			}
		}
		for _, value := range entry.Homeworks {
			row := models.PortalHomework{ID: stableID(c.ID, "homework", entry.PlatformID+":"+value.PlatformID), ConnectionID: c.ID, CourseID: course.ID, SnapshotID: snapshot.ID, PlatformID: value.PlatformID, Title: value.Title, State: value.State, Deadline: value.Deadline, URL: value.URL}
			a, err := ensureImportedAssignment(tx, c, course, &row)
			if err != nil {
				return err
			}
			if err = tx.Model(&a).Update("requirements_status", "PENDING").Error; err != nil {
				return err
			}
			if err := tx.Clauses(clause.OnConflict{UpdateAll: true}).Create(&row).Error; err != nil {
				return err
			}
		}
	}
	now := time.Now()
	c.ActiveSnapshotID = snapshot.ID
	c.LastSyncedAt = &now
	return tx.Model(c).Updates(map[string]any{"active_snapshot_id": snapshot.ID, "last_synced_at": now, "imports_paused": false}).Error
}
