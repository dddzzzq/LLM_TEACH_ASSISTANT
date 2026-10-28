package portal

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"grading-gateway/internal/models"
)

func TestReviewRequiresExplicitAcknowledgementAndRevision(t *testing.T) {
	valid := ReviewDetailRequest{SnapshotID: "snapshot", OperationID: "operation", Question: "核对后的要求", Confirmed: true}
	if err := validateDetailReview(valid); err != nil {
		t.Fatal(err)
	}
	for _, edit := range []func(*ReviewDetailRequest){
		func(r *ReviewDetailRequest) { r.Confirmed = false },
		func(r *ReviewDetailRequest) { r.OperationID = "" },
		func(r *ReviewDetailRequest) { r.SnapshotID = "" },
		func(r *ReviewDetailRequest) { r.Question = " \n " },
		func(r *ReviewDetailRequest) { r.Question = strings.Repeat("a", 100001) },
	} {
		copy := valid
		edit(&copy)
		if validateDetailReview(copy) == nil {
			t.Fatal("invalid review accepted")
		}
	}
}

func TestIntegrationTeacherReviewPublishesOnlyOwnedCurrentRevision(t *testing.T) {
	r, owner, c, works := detailFixture(t)
	ctx := context.Background()
	op, err := r.Service.QueueHomeworkDetail(ctx, owner, works[0].ID, c.ActiveSnapshotID)
	if err != nil {
		t.Fatal(err)
	}
	var target detailTarget
	_ = json.Unmarshal([]byte(op.PayloadJSON), &target)
	candidate := &HomeworkDetail{CourseID: target.Target.CourseID, HomeworkID: target.Target.HomeworkID, Title: target.HomeworkName, Question: "提交课件示例5.5与5.6的截图，不要压缩包", RawQuestion: "原始页面文字", Complete: false, Sources: []DetailSource{{Kind: "text", Text: "原始页面文字"}}, Warnings: []string{"课件示例正文未提供"}}
	if err = r.apply(ctx, op, operationState{Status: "SUCCEEDED", Stage: "DONE", StateVersion: 1, HomeworkDetail: candidate}); err != nil {
		t.Fatal(err)
	}
	var a models.Assignment
	r.Service.DB.First(&a, works[0].AssignmentID)
	if a.Question != "" || a.RequirementsStatus != "REVIEW_REQUIRED" {
		t.Fatal("unconfirmed content published")
	}
	req := ReviewDetailRequest{SnapshotID: c.ActiveSnapshotID, OperationID: op.ID, Question: "补充示例说明后，提交两个运行截图，不要压缩包", Confirmed: true}
	if err = r.Service.ReviewHomeworkDetail(ctx, owner+9999, works[0].ID, req); err == nil {
		t.Fatal("cross-owner acknowledgement")
	}
	stale := req
	stale.OperationID = "stale-operation"
	if err = r.Service.ReviewHomeworkDetail(ctx, owner, works[0].ID, stale); err != ErrConflict {
		t.Fatalf("stale revision: %v", err)
	}
	if err = r.Service.ReviewHomeworkDetail(ctx, owner, works[0].ID, req); err != nil {
		t.Fatal(err)
	}
	r.Service.DB.First(&a, works[0].AssignmentID)
	if a.RequirementsStatus != "READY" || a.Question != req.Question {
		t.Fatal("review not published atomically")
	}
	view, err := r.Service.HomeworkDetail(ctx, owner, works[0].ID)
	if err != nil {
		t.Fatal(err)
	}
	detail := view["detail"].(HomeworkDetail)
	if !detail.Complete || detail.Review == nil || detail.RawQuestion != candidate.RawQuestion || len(detail.Warnings) != 1 {
		t.Fatal("lost review evidence")
	}
	if err = r.Service.ReviewHomeworkDetail(ctx, owner, works[0].ID, req); err != nil {
		t.Fatal("idempotent retry", err)
	}
	req.Question = "另一个页面中的旧内容"
	if err = r.Service.ReviewHomeworkDetail(ctx, owner, works[0].ID, req); err != ErrConflict {
		t.Fatal("concurrent editor overwrote reviewed content")
	}
	newer, err := r.Service.QueueHomeworkDetail(ctx, owner, works[0].ID, c.ActiveSnapshotID)
	if err != nil {
		t.Fatal(err)
	}
	req.OperationID = op.ID
	if err = r.Service.ReviewHomeworkDetail(ctx, owner, works[0].ID, req); err != ErrConflict {
		t.Fatal("approved while new read active")
	}
	_ = newer
}
