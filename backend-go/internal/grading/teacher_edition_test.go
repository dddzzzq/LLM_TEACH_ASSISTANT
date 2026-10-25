package grading

import (
	"context"
	"grading-gateway/internal/models"
	"testing"
)

func TestRemovedExamCannotCreateOrPublishJobs(t *testing.T) {
	repo := fixtureRepo()
	called := false
	service := Service{Repo: repo, PublishHomework: func(string, uint, string) error { called = true; return nil }}
	for _, request := range []Request{{Kind: "exam", ResourceID: "2"}, {Kind: "exam", ResourceID: "2", BatchID: "existing-materials"}} {
		if _, err := service.Submit(context.Background(), Actor{UserID: 7, Role: "teacher"}, request, "unsupported-exam"); err == nil {
			t.Fatal("exam request accepted")
		}
	}
	if called || len(repo.jobs) > 0 {
		t.Fatal("removed exam feature produced work")
	}
	repo.jobs["legacy"] = models.AsyncJob{ID: "legacy", OwnerID: 7, JobType: models.JobTypeExam}
	if _, err := service.Status(context.Background(), Actor{UserID: 7, Role: "teacher"}, "legacy"); err == nil {
		t.Fatal("teacher edition exposed an exam result route")
	}
}
