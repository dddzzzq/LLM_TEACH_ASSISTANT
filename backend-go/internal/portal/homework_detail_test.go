package portal

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"grading-gateway/internal/agent/modelclient"
	"grading-gateway/internal/models"
	"grading-gateway/internal/rpa"
)

func echoRequirementModel(_ context.Context, messages []map[string]interface{}) (*modelclient.LLMResponse, error) {
	blocks := messages[1]["content"].([]map[string]any)
	var input struct{ Sources []struct{ ID, Text string } }
	_ = json.Unmarshal([]byte(blocks[0]["text"].(string)), &input)
	var ids, parts []string
	for _, source := range input.Sources {
		ids = append(ids, source.ID)
		parts = append(parts, source.Text)
	}
	result, _ := json.Marshal(map[string]any{"requirements": strings.Join(parts, "\n\n"), "source_ids": ids, "images": []any{}, "uncertainties": []string{}})
	return &modelclient.LLMResponse{Content: string(result), FinishReason: "stop", Model: "fixture-model"}, nil
}

func detailFixture(t *testing.T) (*Runner, uint, models.PortalConnection, []models.PortalHomework) {
	t.Helper()
	s, owner := portalDB(t)
	if err := s.DB.AutoMigrate(&models.PortalHomeworkDetail{}); err != nil {
		t.Fatal(err)
	}
	r := NewRunner(s, NewBroker(s))
	op, err := s.Save(context.Background(), owner, SaveRequest{"detail-teacher", "fixture-password", true})
	if err != nil {
		t.Fatal(err)
	}
	catalog := fixtureCatalog()
	if err = r.apply(context.Background(), op, operationState{Status: "SUCCEEDED", Stage: "DONE", VerifiedSubject: "detail-teacher", Catalog: &catalog, StateVersion: 1}); err != nil {
		t.Fatal(err)
	}
	c, _ := s.connection(context.Background(), owner, false)
	var works []models.PortalHomework
	if err = s.DB.Where("connection_id = ?", c.ID).Order("platform_id").Find(&works).Error; err != nil {
		t.Fatal(err)
	}
	return r, owner, c, works
}

func TestIntegrationHomeworkDetailsReuseAgentAndKeepQuestionsSeparate(t *testing.T) {
	r, owner, c, works := detailFixture(t)
	ctx := context.Background()
	for i, work := range works {
		op, err := r.Service.QueueHomeworkDetail(ctx, owner, work.ID, c.ActiveSnapshotID)
		if err != nil {
			t.Fatal(err)
		}
		again, err := r.Service.QueueHomeworkDetail(ctx, owner, work.ID, c.ActiveSnapshotID)
		if err != nil || again.ID != op.ID {
			t.Fatal("duplicate detail job")
		}
		if _, err = r.Service.QueueHomeworkDetail(ctx, owner, works[(i+1)%len(works)].ID, c.ActiveSnapshotID); err != ErrConflict {
			t.Fatal("parallel profile access was allowed")
		}
		if err = r.ensureTask(ctx, op); err != nil {
			t.Fatal(err)
		}
		var job models.RPAJob
		r.Service.DB.First(&job, "id = ?", op.ID)
		var nav rpa.NavigationSnapshot
		_ = json.Unmarshal([]byte(job.NavigationJSON), &nav)
		if job.TaskType != "HOMEWORK_DETAIL" || nav.Skill.Name != "read-homework-detail" || job.PortalTargetJSON == "" {
			t.Fatal("did not reuse the page Agent")
		}
		var target detailTarget
		_ = json.Unmarshal([]byte(op.PayloadJSON), &target)
		question := []string{"实现并发下载器", "实现二叉树遍历"}[i]
		resolved, err := resolveDetailWithModel(ctx, op, &HomeworkDetail{CourseID: target.Target.CourseID, HomeworkID: target.Target.HomeworkID, Title: target.HomeworkName, Complete: true, Sources: []DetailSource{{Kind: "text", Text: question}}}, echoRequirementModel)
		if err != nil {
			t.Fatal(err)
		}
		if err = r.apply(ctx, op, operationState{Status: "SUCCEEDED", Stage: "DONE", StateVersion: 1, HomeworkDetail: resolved}); err != nil {
			t.Fatal(err)
		}
		// Simulate ordinary RPA scheduler persisting the browser completion.
		r.Service.DB.Model(&job).Update("status", "SUCCEEDED")
		if err = r.CancelDetailRead(ctx, owner, op.ID); err != nil {
			t.Fatal(err)
		}
		r.Service.DB.First(&job, "id = ?", op.ID)
		if job.Status != "SUCCEEDED" {
			t.Fatal("a late cancellation changed a completed browser job")
		}
		view, err := r.Service.HomeworkDetail(ctx, owner, work.ID)
		if err != nil || view["detail"].(HomeworkDetail).Question != question {
			t.Fatal("question missing")
		}
		if _, err = r.Service.HomeworkDetail(ctx, owner+100000, work.ID); err == nil {
			t.Fatal("cross-owner detail access")
		}
	}
	flow, err := r.Service.CreateWorkflow(ctx, owner, WorkflowRequest{Source: "portal", SnapshotID: c.ActiveSnapshotID, Items: []Selection{{HomeworkID: works[0].ID}, {HomeworkID: works[1].ID}}, UsePortalRequirements: true, Rubric: json.RawMessage(`{"正确性":100}`)}, "detail-separate")
	if err != nil {
		t.Fatal(err)
	}
	var items []models.PortalWorkflowItem
	r.Service.DB.Where("workflow_id = ?", flow.ID).Find(&items)
	if len(items) != 2 || items[0].Question == items[1].Question {
		t.Fatal("different assignments shared a question")
	}
	for _, item := range items {
		var a models.Assignment
		r.Service.DB.First(&a, item.AssignmentID)
		if a.Question != item.Question {
			t.Fatal("grading did not receive the selected question")
		}
	}
}

func TestIntegrationIncompleteOrCancelledDetailCannotStartAutomaticGrading(t *testing.T) {
	r, owner, c, works := detailFixture(t)
	ctx := context.Background()
	op, err := r.Service.QueueHomeworkDetail(ctx, owner, works[0].ID, c.ActiveSnapshotID)
	if err != nil {
		t.Fatal(err)
	}
	var selected detailTarget
	_ = json.Unmarshal([]byte(op.PayloadJSON), &selected)
	raw := &HomeworkDetail{CourseID: selected.Target.CourseID, HomeworkID: selected.Target.HomeworkID, Title: selected.HomeworkName, Complete: false, Question: "部分要求", Sources: []DetailSource{{Kind: "text", Text: "部分要求"}}, Warnings: []string{"附件失败"}}
	if err = r.apply(ctx, op, operationState{Status: "SUCCEEDED", Stage: "DONE", StateVersion: 1, HomeworkDetail: raw}); err != nil {
		t.Fatal(err)
	}
	_, err = r.Service.CreateWorkflow(ctx, owner, WorkflowRequest{Source: "portal", SnapshotID: c.ActiveSnapshotID, Items: []Selection{{HomeworkID: works[0].ID}}, UsePortalRequirements: true, Rubric: json.RawMessage(`{"正确性":100}`)}, "incomplete-detail")
	if err == nil {
		t.Fatal("incomplete requirements were used for automatic grading")
	}
	op, err = r.Service.QueueHomeworkDetail(ctx, owner, works[1].ID, c.ActiveSnapshotID)
	if err != nil {
		t.Fatal(err)
	}
	if err = r.CancelDetailRead(ctx, owner, op.ID); err != nil {
		t.Fatal(err)
	}
	_ = json.Unmarshal([]byte(op.PayloadJSON), &selected)
	raw.CourseID, raw.HomeworkID = selected.Target.CourseID, selected.Target.HomeworkID
	if err = r.apply(ctx, op, operationState{Status: "SUCCEEDED", Stage: "DONE", StateVersion: 2, HomeworkDetail: raw}); err != nil {
		t.Fatal(err)
	}
	var count int64
	r.Service.DB.Model(&models.PortalHomeworkDetail{}).Where("homework_id = ?", works[1].ID).Count(&count)
	if count != 0 {
		t.Fatal("late cancelled result published")
	}
}

func TestRequirementResourcesRejectTraversalAndVerifyDigest(t *testing.T) {
	root := t.TempDir()
	t.Setenv("RPA_DOWNLOAD_DIR", root)
	base := filepath.Join(root, "job", "requirements")
	if err := os.MkdirAll(base, 0700); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(base, "question.txt")
	data := []byte("真实附件要求")
	if err := os.WriteFile(file, data, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := detailResourcePath("job", filepath.Join(root, "secret")); err == nil {
		t.Fatal("path escape")
	}
	target := detailTarget{HomeworkName: "作业", Target: rpa.PortalTarget{CourseID: "11", HomeworkID: "33"}}
	payload, _ := json.Marshal(target)
	op := models.PortalOperation{ID: "job", PayloadJSON: string(payload)}
	digest := sha256.Sum256(data)
	raw := &HomeworkDetail{CourseID: "11", HomeworkID: "33", Title: "作业", Complete: true, Sources: []DetailSource{{Kind: "attachment", Name: "题目.txt", Path: file, SHA256: hex.EncodeToString(digest[:]), Size: int64(len(data))}}}
	result, err := resolveDetailWithModel(context.Background(), op, raw, echoRequirementModel)
	if err != nil || !result.Complete || !strings.Contains(result.Question, string(data)) {
		t.Fatalf("resource parse: %v", err)
	}
	raw.Sources[0].SHA256 = strings.Repeat("0", 64)
	if _, err = resolveDetailWithModel(context.Background(), op, raw, echoRequirementModel); err == nil {
		t.Fatal("tampered resource accepted")
	}
}
