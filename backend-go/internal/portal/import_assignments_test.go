package portal

import (
	"context"
	"encoding/json"
	"grading-gateway/internal/models"
	"grading-gateway/internal/rpa"
	"testing"
)

func TestIntegrationSyncCreatesAssignmentsAndPreservesRubricOnRefresh(t *testing.T) {
	r, owner, c, works := detailFixture(t)
	ctx := context.Background()
	var a models.Assignment
	if err := r.Service.DB.First(&a, "id = ? AND owner_id = ?", works[0].AssignmentID, owner).Error; err != nil {
		t.Fatal(err)
	}
	if a.Question != "" || a.RequirementsStatus != "PENDING" || a.Rubric != DefaultHomeworkRubric || a.PortalHomeworkID != works[0].ID {
		t.Fatal("catalog did not immediately create an ungraded assignment with template")
	}
	const custom = `{"实验功能":80,"实验报告":20}`
	if err := r.Service.SaveAssignmentRubric(ctx, owner, a.ID, json.RawMessage(custom)); err != nil {
		t.Fatal(err)
	}
	op, err := r.Service.QueueHomeworkDetail(ctx, owner, works[0].ID, c.ActiveSnapshotID)
	if err != nil {
		t.Fatal(err)
	}
	var target detailTarget
	_ = json.Unmarshal([]byte(op.PayloadJSON), &target)
	detail := &HomeworkDetail{CourseID: target.Target.CourseID, HomeworkID: target.Target.HomeworkID, Title: target.HomeworkName, Question: "官网要求：每隔500ms输出状态", Complete: true, Sources: []DetailSource{{Kind: "text", Text: "官网要求"}}}
	if err = r.apply(ctx, op, operationState{Status: "SUCCEEDED", Stage: "DONE", StateVersion: 1, HomeworkDetail: detail}); err != nil {
		t.Fatal(err)
	}
	r.Service.DB.First(&a, a.ID)
	if a.Question != detail.Question || a.Rubric != custom || a.RequirementsStatus != "READY" {
		t.Fatal("detail not written to the already-created assignment")
	}
	catalog := fixtureCatalog()
	catalog.Courses[0].Homeworks[0].Title = "修改后的作业名称"
	sync, err := r.Service.Queue(ctx, owner, "CATALOG_SYNC")
	if err != nil {
		t.Fatal(err)
	}
	if err = r.apply(ctx, sync, operationState{Status: "SUCCEEDED", Stage: "DONE", StateVersion: 1, Catalog: &catalog}); err != nil {
		t.Fatal(err)
	}
	r.Service.DB.First(&a, a.ID)
	var count int64
	r.Service.DB.Model(&models.Assignment{}).Where("owner_id = ?", owner).Count(&count)
	if count != 2 || a.Rubric != custom || a.TaskName != "修改后的作业名称" || a.RequirementsStatus != "PENDING" {
		t.Fatal("refresh duplicated assignments or overwrote teacher rubric")
	}
	if works[0].AssignmentID == works[1].AssignmentID {
		t.Fatal("same-name courses in different terms were merged")
	}
}

func TestIntegrationExistingCatalogReusesCachedRequirementsWithoutBrowserWork(t *testing.T) {
	r, owner, c, works := detailFixture(t)
	work := works[0]
	_ = r.Service.DB.Model(&work).Update("assignment_id", 0).Error
	_ = r.Service.DB.Model(&models.Assignment{}).Where("id = ?", works[0].AssignmentID).Updates(map[string]any{"requirements_status": "", "question": "", "rubric": `{"教师原标准":100}`}).Error
	cache := models.PortalHomeworkDetail{HomeworkID: work.ID, OwnerID: owner, ConnectionID: c.ID, ProfileGeneration: c.ProfileGeneration, SnapshotID: c.ActiveSnapshotID, OperationID: "previous-read", Question: "已核验的官网要求", Complete: true, ContentJSON: "{}"}
	if err := r.Service.DB.Create(&cache).Error; err != nil {
		t.Fatal(err)
	}
	r.reconcileImportedAssignments(context.Background())
	var a models.Assignment
	r.Service.DB.First(&a, works[0].AssignmentID)
	if a.Question != cache.Question || a.RequirementsStatus != "READY" || a.Rubric != `{"教师原标准":100}` {
		t.Fatal("upgrade lost valid requirements or teacher settings")
	}
	updated, _ := r.Service.connection(context.Background(), owner, false)
	if !updated.ImportsPaused || updated.ActiveOperationID != "" {
		t.Fatal("upgrade launched browser work for an inactive saved account")
	}
}

func TestIntegrationImportsQueueOneExistingDetailSkillAndCanPause(t *testing.T) {
	r, owner, c, works := detailFixture(t)
	ctx := context.Background()
	if err := r.Service.PauseImports(ctx, owner, true); err != nil {
		t.Fatal(err)
	}
	r.tickImportedRequirements(ctx)
	paused, _ := r.Service.connection(ctx, owner, false)
	if paused.ActiveOperationID != "" {
		t.Fatal("paused import queued work")
	}
	_ = r.Service.PauseImports(ctx, owner, false)
	r.tickImportedRequirements(ctx)
	r.tickImportedRequirements(ctx)
	active, _ := r.Service.connection(ctx, owner, false)
	if active.ActiveOperationID == "" {
		t.Fatal("did not automatically read imported homework requirements")
	}
	var op models.PortalOperation
	r.Service.DB.First(&op, "id = ?", active.ActiveOperationID)
	if op.Type != "HOMEWORK_DETAIL" {
		t.Fatal("import unexpectedly started grading")
	}
	if err := r.ensureTask(ctx, op); err != nil {
		t.Fatal(err)
	}
	var job models.RPAJob
	r.Service.DB.First(&job, "id = ?", op.ID)
	var nav rpa.NavigationSnapshot
	_ = json.Unmarshal([]byte(job.NavigationJSON), &nav)
	if nav.Skill.Name != "read-homework-detail" {
		t.Fatal("did not reuse the existing page skill")
	}
	if err := r.CancelDetailRead(ctx, owner, op.ID); err != nil {
		t.Fatal(err)
	}
	var queued int64
	r.Service.DB.Model(&models.PortalOperation{}).Where("connection_id = ? AND type = ?", c.ID, "HOMEWORK_DETAIL").Count(&queued)
	if queued != 1 {
		t.Fatal("repeated scheduler tick duplicated read operations")
	}
	var assignment models.Assignment
	r.Service.DB.First(&assignment, "portal_homework_id IN ? AND requirements_status = ?", []string{works[0].ID, works[1].ID}, "CANCELLED")
	if assignment.ID == 0 {
		t.Fatal("cancelled import would be queued forever")
	}
}

func TestIntegrationClassesUseOneSavedAssignmentAndStableClassIDs(t *testing.T) {
	r, owner, c, works := detailFixture(t)
	ctx := context.Background()
	first := works[0]
	another := models.PortalCourseClass{ID: stableID(c.ID, "extra-class", "33"), CourseID: first.CourseID, SnapshotID: c.ActiveSnapshotID, PlatformID: "333", Name: "A班"}
	if err := r.Service.DB.Create(&another).Error; err != nil {
		t.Fatal(err)
	}
	var classes []models.PortalCourseClass
	r.Service.DB.Where("course_id = ?", first.CourseID).Find(&classes)
	if err := r.Service.DB.Model(&models.Assignment{}).Where("id = ?", first.AssignmentID).Updates(map[string]any{"question": "来自官网的完整题目", "requirements_status": "READY"}).Error; err != nil {
		t.Fatal(err)
	}
	if err := r.Service.SaveAssignmentRubric(ctx, owner, first.AssignmentID, json.RawMessage(`{"教师标准":100}`)); err != nil {
		t.Fatal(err)
	}
	ids := []string{classes[0].ID, classes[1].ID}
	flow, err := r.Service.GradeAssignmentClasses(ctx, owner, first.AssignmentID, ids, "same-master")
	if err != nil {
		t.Fatal(err)
	}
	again, err := r.Service.GradeAssignmentClasses(ctx, owner, first.AssignmentID, ids, "same-master")
	if err != nil || again.ID != flow.ID {
		t.Fatal("grade request was not idempotent")
	}
	var items []models.PortalWorkflowItem
	r.Service.DB.Where("workflow_id = ?", flow.ID).Find(&items)
	if len(items) != 2 || items[0].AssignmentID != first.AssignmentID || items[1].AssignmentID != first.AssignmentID || items[0].ClassID == items[1].ClassID {
		t.Fatal("classes created separate assignments or lost class identity")
	}
	for _, item := range items {
		if item.Question != "来自官网的完整题目" || item.Rubric != `{"教师标准":100}` {
			t.Fatal("grading ignored saved requirements or edited rubric")
		}
	}
	var count int64
	r.Service.DB.Model(&models.Assignment{}).Where("owner_id = ?", owner).Count(&count)
	if count != 2 {
		t.Fatal("grading created another assignment")
	}
	if _, err = r.Service.GradeAssignmentClasses(ctx, owner+10000, first.AssignmentID, ids, "foreign"); err == nil {
		t.Fatal("cross-owner grading allowed")
	}
	var wrong models.PortalCourseClass
	r.Service.DB.Where("course_id = ?", works[1].CourseID).First(&wrong)
	if _, err = r.Service.GradeAssignmentClasses(ctx, owner, first.AssignmentID, []string{wrong.ID}, "wrong-class"); err == nil {
		t.Fatal("class from another course accepted")
	}
	_ = r.Service.DB.Model(&models.Assignment{}).Where("id = ?", first.AssignmentID).Update("requirements_status", "REVIEW_REQUIRED").Error
	if _, err = r.Service.GradeAssignmentClasses(ctx, owner, first.AssignmentID, ids, "not-ready"); err == nil {
		t.Fatal("incomplete requirements used for grading")
	}
}
