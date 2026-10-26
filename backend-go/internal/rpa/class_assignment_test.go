package rpa

import (
	"archive/zip"
	"encoding/json"
	"github.com/google/uuid"
	"grading-gateway/internal/database"
	"grading-gateway/internal/materials"
	"grading-gateway/internal/models"
	"os"
	"path/filepath"
	"testing"
)

func TestIntegrationSeparateClassArchivesUseSameImportedAssignment(t *testing.T) {
	integrationDB(t)
	if err := database.DB.AutoMigrate(&models.PortalWorkflowItem{}); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	t.Setenv("HOMEWORK_ALLOWED_ROOTS", root)
	t.Setenv("HOMEWORK_MATERIALS_ROOT", filepath.Join(root, "prepared"))
	a := models.Assignment{OwnerID: 8401, CourseName: "按班级批改测试", TaskName: uuid.NewString(), ClassName: "全部班级", PortalConnectionID: uuid.NewString(), PortalHomeworkID: uuid.NewString(), Question: "官网题目", Rubric: `{"正确性":100}`, RequirementsStatus: "READY"}
	if err := database.DB.Create(&a).Error; err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"A班", "B班"} {
		item := models.PortalWorkflowItem{ID: uuid.NewString(), WorkflowID: uuid.NewString(), OwnerID: a.OwnerID, AssignmentID: a.ID, ClassID: uuid.NewString(), ClassName: name, Question: a.Question, Rubric: a.Rubric, Status: "RUNNING"}
		if err := database.DB.Create(&item).Error; err != nil {
			t.Fatal(err)
		}
		target, _ := json.Marshal(PortalTarget{CourseID: "11", HomeworkID: "33", ClassID: item.ClassID, ClassName: name})
		job := models.RPAJob{ID: uuid.NewString(), UserID: a.OwnerID, CourseName: a.CourseName, AssignmentName: a.TaskName, AssignmentID: a.ID, PortalConnectionID: a.PortalConnectionID, PortalWorkflowItemID: item.ID, PortalTargetJSON: string(target), AfterDownload: "extract", Status: "SUCCEEDED"}
		dir := filepath.Join(root, job.ID)
		_ = os.MkdirAll(dir, 0700)
		path := filepath.Join(dir, "student.zip")
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		archive := zip.NewWriter(f)
		entry, _ := archive.Create("20260001_张三/report.txt")
		_, _ = entry.Write([]byte("学生材料"))
		_ = archive.Close()
		_ = f.Close()
		hash, _ := materials.HashFile(path)
		info, _ := os.Stat(path)
		state, _ := json.Marshal(map[string]any{"class_scope": map[string]any{"mode": "selected", "classes": []map[string]string{{"platform_id": item.ClassID}}}, "files": []map[string]any{{"status": "VERIFIED", "path": path, "sha256": hash, "size": info.Size(), "class_name": "导出时使用的班级编号", "export_ref": name}}})
		job.StateJSON = string(state)
		wrong := job
		wrong.StateJSON = `{"class_scope":{"mode":"all"},"files":[]}`
		if err = importFiles(wrong); err == nil {
			t.Fatal("selected class silently widened to all classes")
		}
		if err = database.DB.Create(&job).Error; err != nil {
			t.Fatal(err)
		}
		if err = importFiles(job); err != nil {
			t.Fatal(err)
		}
		var file models.RPAFile
		database.DB.First(&file, "job_id = ?", job.ID)
		if err = dispatchFile(file); err != nil {
			t.Fatal(err)
		}
		database.DB.First(&file, "id = ?", file.ID)
		var batch models.MaterialBatch
		database.DB.First(&batch, "id = ?", file.MaterialBatchID)
		if file.AssignmentID != a.ID || batch.AssignmentID != a.ID || batch.ClassName != name || batch.PortalClassID != item.ClassID {
			t.Fatal("class identity lost or another assignment created")
		}
		// A different class must not slip into this selected export.
		file.ClassName = "不属于本次任务的班级"
		database.DB.Save(&file)
		if err = dispatchFile(file); err == nil {
			t.Fatal("wrong class accepted into selected export")
		}
	}
	var count int64
	database.DB.Model(&models.Assignment{}).Where("owner_id = ? AND task_name = ?", a.OwnerID, a.TaskName).Count(&count)
	if count != 1 {
		t.Fatal("classes created duplicate homework")
	}
}
