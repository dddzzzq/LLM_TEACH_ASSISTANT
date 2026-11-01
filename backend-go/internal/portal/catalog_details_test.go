package portal

import (
	"context"
	"encoding/json"
	"testing"

	"grading-gateway/internal/models"
)

func catalogWithDetails() Catalog {
	catalog := fixtureCatalog()
	catalog.DetailsIncluded = true
	for i := range catalog.Courses {
		course := &catalog.Courses[i]
		for j := range course.Homeworks {
			work := &course.Homeworks[j]
			work.Detail = &HomeworkDetail{CourseID: course.PlatformID, HomeworkID: work.PlatformID, Title: work.Title, Complete: true, Sources: []DetailSource{{Kind: "text", Text: "要求 " + work.PlatformID}}}
		}
	}
	return catalog
}

func TestCatalogDetailsResolveEachQuestionAndRejectMissingOrMixedTargets(t *testing.T) {
	catalog := catalogWithDetails()
	op := models.PortalOperation{ID: "sync", ConnectionID: "connection"}
	result, err := resolveCatalogDetails(context.Background(), op, &catalog, echoRequirementModel)
	if err != nil {
		t.Fatal(err)
	}
	for i, course := range result.Courses {
		work := course.Homeworks[0]
		if !work.Detail.Complete || work.Detail.Question != "要求 "+work.PlatformID || work.Detail.Processing == nil {
			t.Fatal("lost or mixed requirements", work.Detail)
		}
		if catalog.Courses[i].Homeworks[0].Detail.Processing != nil {
			t.Fatal("mutated the raw worker checkpoint")
		}
	}
	for _, mutation := range []func(*Catalog){
		func(c *Catalog) { c.Courses[0].Homeworks[0].Detail = nil },
		func(c *Catalog) { c.Courses[0].Homeworks[0].Detail.CourseID = "wrong" },
		func(c *Catalog) { c.Courses[0].Homeworks[0].Detail.HomeworkID = "wrong" },
		func(c *Catalog) { c.Courses[0].Homeworks[0].Detail.Title = "wrong" },
	} {
		bad := catalogWithDetails()
		mutation(&bad)
		if _, err := resolveCatalogDetails(context.Background(), op, &bad, echoRequirementModel); err == nil {
			t.Fatal("accepted missing or mismatched details")
		}
	}
}

func TestIntegrationCombinedSyncPublishesRequirementsWithoutDetailJobsAndAllowsReread(t *testing.T) {
	r, owner, _, works := detailFixture(t)
	ctx := context.Background()
	op, err := r.Service.Queue(ctx, owner, "CATALOG_SYNC")
	if err != nil {
		t.Fatal(err)
	}
	if err = r.ensureTask(ctx, op); err != nil {
		t.Fatal(err)
	}
	catalog := catalogWithDetails()
	resolved, err := resolveCatalogDetails(ctx, op, &catalog, echoRequirementModel)
	if err != nil {
		t.Fatal(err)
	}
	state := operationState{Status: "SUCCEEDED", Stage: "DONE", StateVersion: 1, Catalog: resolved}
	for i := 0; i < 2; i++ {
		if err = r.apply(ctx, op, state); err != nil {
			t.Fatal(err)
		}
	}
	r.tickImportedRequirements(ctx)
	var count int64
	r.Service.DB.Model(&models.PortalOperation{}).Where("connection_id = ? AND type = ?", op.ConnectionID, "HOMEWORK_DETAIL").Count(&count)
	if count != 0 {
		t.Fatal("sync queued another browser operation")
	}
	for _, work := range works {
		var a models.Assignment
		r.Service.DB.First(&a, work.AssignmentID)
		if a.RequirementsStatus != "READY" || a.Question != "要求 "+work.PlatformID {
			t.Fatal("requirements not saved", a.ID, a.RequirementsStatus)
		}
		view, err := r.Service.HomeworkDetail(ctx, owner, work.ID)
		if err != nil || view["operation_id"] != op.ID {
			t.Fatal("combined requirements unavailable", err)
		}
	}
	c, _ := r.Service.connection(ctx, owner, false)
	// In production the RPA runner commits this terminal state before applying
	// the portal result. This test drives apply directly, so finish its job too.
	if err := r.Service.DB.Model(&models.RPAJob{}).Where("id = ?", op.ID).Update("status", "SUCCEEDED").Error; err != nil {
		t.Fatal(err)
	}
	reread, err := r.Service.QueueHomeworkDetail(ctx, owner, works[0].ID, c.ActiveSnapshotID)
	if err != nil || reread.Type != "HOMEWORK_DETAIL" {
		t.Fatal("standalone reread unavailable", err)
	}
	var target detailTarget
	_ = json.Unmarshal([]byte(reread.PayloadJSON), &target)
	if target.HomeworkID != works[0].ID {
		t.Fatal("wrong reread target:", target.HomeworkID)
	}
}
