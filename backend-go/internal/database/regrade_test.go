package database_test

import (
	"testing"

	"github.com/google/uuid"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
	"grading-gateway/internal/testutil"
)

func TestIntegrationRegradeUpdatesOnlyAIFieldsInTheSameBatch(t *testing.T) {
	db, owner := testutil.ExecutionDB(t)
	old := database.DB
	database.DB = db
	t.Cleanup(func() { database.DB = old })
	if err := db.AutoMigrate(&models.Assignment{}, &models.Submission{}); err != nil {
		t.Fatal(err)
	}
	a := models.Assignment{OwnerID: owner, CourseName: "测试课", TaskName: "重新批改"}
	if err := db.Create(&a).Error; err != nil {
		t.Fatal(err)
	}
	batch := uuid.NewString()
	original := models.Submission{AssignmentID: a.ID, MaterialBatchID: batch, GradingJobID: "original-job", StudentID: "same-student", Score: 80, Feedback: "原AI评语", IsHumanReviewed: true, HumanScore: 88, HumanFeedback: "教师保留意见"}
	other := models.Submission{AssignmentID: a.ID, MaterialBatchID: uuid.NewString(), GradingJobID: "other-class-job", StudentID: original.StudentID, Score: 55}
	for _, row := range []*models.Submission{&original, &other} {
		if err := db.Create(row).Error; err != nil {
			t.Fatal(err)
		}
	}
	updated := models.Submission{AssignmentID: a.ID, MaterialBatchID: batch, GradingJobID: "new-job", StudentID: original.StudentID, StudentName: "学生", Score: 0, Feedback: "新AI评语", MergeContent: "新的完整材料", PlagiarismReport: "[]", AIGCReport: "{}", CodeDocMatchReport: "{}", ReadingStatus: "COMPLETE", ReadingReport: `{"status":"COMPLETE"}`, ScoreDetails: `[{"score":0}]`}
	if err := database.SaveRegradedSubmission(updated); err != nil {
		t.Fatal(err)
	}
	var saved models.Submission
	if err := db.First(&saved, original.ID).Error; err != nil {
		t.Fatal(err)
	}
	if saved.Score != 0 || saved.Feedback != updated.Feedback || saved.GradingJobID != updated.GradingJobID || saved.ReadingStatus != "COMPLETE" || saved.ScoreDetails != updated.ScoreDetails || saved.PlagiarismReport != "[]" || saved.AIGCReport != "{}" {
		t.Fatal("AI fields were not refreshed", saved.ID)
	}
	if !saved.IsHumanReviewed || saved.HumanScore != 88 || saved.HumanFeedback != original.HumanFeedback {
		t.Fatal("teacher review changed")
	}
	var untouched models.Submission
	db.First(&untouched, other.ID)
	if untouched.Score != 55 || untouched.GradingJobID != other.GradingJobID {
		t.Fatal("another batch was changed")
	}
	var count int64
	db.Model(&models.Submission{}).Where("assignment_id = ?", a.ID).Count(&count)
	if count != 2 {
		t.Fatal("regrade duplicated student result", count)
	}
	// A teacher may have deleted all results before clicking regrade.
	db.Delete(&saved)
	if err := database.SaveRegradedSubmission(updated); err != nil {
		t.Fatal(err)
	}
	var recreated models.Submission
	if err := db.Where("assignment_id = ? AND material_batch_id = ?", a.ID, batch).First(&recreated).Error; err != nil {
		t.Fatal(err)
	}
	if recreated.IsHumanReviewed || recreated.Score != 0 || recreated.GradingJobID != "new-job" {
		t.Fatal("deleted result could not be regraded")
	}
}
