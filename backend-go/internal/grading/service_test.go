package grading

import (
	"archive/zip"
	"context"
	"errors"
	"gorm.io/gorm"
	"grading-gateway/internal/models"
	"os"
	"path/filepath"
	"testing"
)

type memoryRepo struct {
	assignment models.Assignment
	exam       models.Exam
	jobs       map[string]models.AsyncJob
	createErr  error
}

func (r *memoryRepo) Assignment(context.Context, uint) (models.Assignment, error) {
	return r.assignment, nil
}
func (r *memoryRepo) Exam(context.Context, uint) (models.Exam, error) { return r.exam, nil }
func (r *memoryRepo) Job(_ context.Context, id string) (models.AsyncJob, error) {
	j, ok := r.jobs[id]
	if !ok {
		return j, gorm.ErrRecordNotFound
	}
	return j, nil
}
func (r *memoryRepo) Create(_ context.Context, j *models.AsyncJob) error {
	if r.createErr != nil {
		return r.createErr
	}
	r.jobs[j.ID] = *j
	return nil
}
func (r *memoryRepo) Fail(_ context.Context, id, msg string) error {
	j := r.jobs[id]
	j.Status = models.JobStatusFailed
	j.Message = msg
	r.jobs[id] = j
	return nil
}
func fixtureRepo() *memoryRepo {
	return &memoryRepo{assignment: models.Assignment{ID: 1, Question: "编写程序", Rubric: `{"正确性":100}`}, exam: models.Exam{ID: 2, Questions: []models.ExamQuestion{{QuestionNumber: 1, QuestionText: "问题", StandardAnswer: "答案", Rubric: "标准", MaxScore: 10}}}, jobs: map[string]models.AsyncJob{}}
}
func fixtureZip(t *testing.T, name string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "homework.zip")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	w := zip.NewWriter(f)
	entry, err := w.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = entry.Write([]byte("student answer"))
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	f.Close()
	return path
}
func TestSubmissionPersistsBeforePublishAndDeduplicatesWithinRequest(t *testing.T) {
	repo := fixtureRepo()
	count := 0
	service := Service{Repo: repo, PublishHomework: func(id string, assignment uint, path string) error {
		count++
		if _, ok := repo.jobs[id]; !ok {
			t.Fatal("published before persistence")
		}
		return nil
	}}
	actor := Actor{UserID: 7, Role: "teacher"}
	request := Request{Kind: "homework", ResourceID: "1", FilePath: fixtureZip(t, "20260001/report.txt"), SkillName: "grade-homework", SkillVersion: "hash"}
	first, err := service.Submit(context.Background(), actor, request, "turn-1")
	if err != nil || first.Outcome != "accepted" {
		t.Fatalf("%+v %v", first, err)
	}
	second, err := service.Submit(context.Background(), actor, request, "turn-1")
	if err != nil || first.JobID != second.JobID || count != 1 {
		t.Fatal("duplicate submission")
	}
	job := repo.jobs[first.JobID]
	if job.OwnerID != 7 || job.SkillName != "grade-homework" || job.SkillVersion != "hash" {
		t.Fatal("provenance missing")
	}
	if _, err := service.Status(context.Background(), Actor{UserID: 8, Role: "teacher"}, first.JobID); err == nil {
		t.Fatal("cross-owner access")
	}
	if _, err := service.Status(context.Background(), Actor{UserID: 7, Role: "student"}, first.JobID); err == nil {
		t.Fatal("student could query teacher task")
	}
}
func TestSubmissionWaitsForInputsAndNeverPublishesOnValidationOrPersistenceFailure(t *testing.T) {
	for _, test := range []string{"student", "rubric", "empty_rubric", "question", "exam_answer", "duplicate_question", "missing_file", "bad_zip", "traversal", "db_failure"} {
		t.Run(test, func(t *testing.T) {
			repo := fixtureRepo()
			called := false
			service := Service{Repo: repo, PublishHomework: func(string, uint, string) error { called = true; return nil }, PublishExam: func(string, string, string, []string) error { called = true; return nil }}
			actor := Actor{UserID: 7, Role: "teacher"}
			request := Request{Kind: "homework", ResourceID: "1", FilePath: fixtureZip(t, "s/report.txt")}
			switch test {
			case "student":
				actor.Role = "student"
			case "rubric":
				repo.assignment.Rubric = "broken"
			case "empty_rubric":
				repo.assignment.Rubric = "{}"
			case "question":
				repo.assignment.Question = ""
			case "exam_answer":
				request.Kind = "exam"
				repo.exam.Questions[0].StandardAnswer = ""
			case "duplicate_question":
				request.Kind = "exam"
				repo.exam.Questions = append(repo.exam.Questions, repo.exam.Questions[0])
			case "missing_file":
				request.FilePath = ""
			case "bad_zip":
				request.FilePath = filepath.Join(t.TempDir(), "missing.zip")
			case "traversal":
				request.FilePath = fixtureZip(t, "../escape")
			case "db_failure":
				repo.createErr = errors.New("db unavailable")
			}
			result, err := service.Submit(context.Background(), actor, request, "turn")
			if err == nil && result.Outcome != "needs_input" {
				t.Fatalf("unexpected success: %+v", result)
			}
			if called || len(repo.jobs) > 0 {
				t.Fatal("invalid input produced task")
			}
		})
	}
}
func TestExamSubmissionKeepsPageOrderAndQueueFailureIsVisible(t *testing.T) {
	repo := fixtureRepo()
	paths := []string{filepath.Join(t.TempDir(), "2.png"), filepath.Join(t.TempDir(), "1.png")}
	for _, p := range paths {
		os.WriteFile(p, []byte("image"), 0600)
	}
	service := Service{Repo: repo, PublishExam: func(id, exam, student string, images []string) error {
		if images[0] != paths[0] || images[1] != paths[1] {
			t.Fatal("page order changed")
		}
		return errors.New("queue unavailable")
	}}
	result, err := service.Submit(context.Background(), Actor{UserID: 7, Role: "teacher"}, Request{Kind: "exam", ResourceID: "2", StudentID: "20260001", ImagePaths: paths}, "turn")
	if err != nil || result.Outcome != "failed" || result.Status != models.JobStatusFailed || repo.jobs[result.JobID].Status != models.JobStatusFailed {
		t.Fatalf("queue error hidden: %+v %v", result, err)
	}
}
