package portal

import (
	"context"
	"testing"

	"grading-gateway/internal/models"
)

func TestIntegrationAuthenticationPublishesCatalogWithoutAnotherOperation(t *testing.T) {
	s, owner := portalDB(t)
	ctx := context.Background()
	runner := NewRunner(s, NewBroker(s))
	op, err := s.Save(ctx, owner, SaveRequest{"one-browser-teacher", "fixture-password", true})
	if err != nil {
		t.Fatal(err)
	}
	if err = runner.ensureTask(ctx, op); err != nil {
		t.Fatal(err)
	}
	// A wrong identity cannot become READY even during collection.
	if err = runner.apply(ctx, op, operationState{Status: "RUNNING", Stage: "READING_CATALOG", VerifiedSubject: "wrong-teacher", StateVersion: 1}); err == nil {
		t.Fatal("accepted another teacher identity")
	}
	progress := operationState{Status: "RUNNING", Stage: "READING_CATALOG", VerifiedSubject: "one-browser-teacher", StateVersion: 1}
	if err = runner.apply(ctx, op, progress); err != nil {
		t.Fatal(err)
	}
	view, err := s.View(ctx, owner)
	if err != nil || view.Status != "READY" || view.Operation == nil || view.Operation.ID != op.ID {
		t.Fatalf("authentication did not retain the original active operation: %+v %v", view, err)
	}
	complete := operationState{Status: "SUCCEEDED", Stage: "DONE", VerifiedSubject: "one-browser-teacher", Catalog: ptrCatalog(fixtureCatalog()), StateVersion: 2}
	if err = runner.apply(ctx, *view.Operation, complete); err != nil {
		t.Fatal(err)
	}
	// Duplicate receipt is idempotent and cannot enqueue a second browser job.
	if err = runner.apply(ctx, op, complete); err != nil {
		t.Fatal(err)
	}
	c, err := s.connection(ctx, owner, false)
	if err != nil || c.ActiveOperationID != "" || c.ActiveSnapshotID == "" || c.Status != models.PortalConnectionReady {
		t.Fatalf("combined authentication/catalog did not finish: %+v", err)
	}
	var operations, jobs, snapshots int64
	s.DB.Model(&models.PortalOperation{}).Where("connection_id = ?", c.ID).Count(&operations)
	s.DB.Model(&models.RPAJob{}).Where("portal_connection_id = ?", c.ID).Count(&jobs)
	s.DB.Model(&models.PortalCatalogSnapshot{}).Where("connection_id = ?", c.ID).Count(&snapshots)
	if operations != 1 || jobs != 1 || snapshots != 1 {
		t.Fatalf("duplicate auth/catalog execution: operations=%d jobs=%d snapshots=%d", operations, jobs, snapshots)
	}
}

func TestIntegrationCatalogFailureKeepsVerifiedIdentityAndPreviousSnapshot(t *testing.T) {
	s, owner := portalDB(t)
	ctx := context.Background()
	runner := NewRunner(s, NewBroker(s))
	op, err := s.Save(ctx, owner, SaveRequest{"catalog-failure-teacher", "fixture-password", true})
	if err != nil {
		t.Fatal(err)
	}
	if err = runner.apply(ctx, op, operationState{Status: "SUCCEEDED", Stage: "DONE", VerifiedSubject: "catalog-failure-teacher", Catalog: ptrCatalog(fixtureCatalog()), StateVersion: 1}); err != nil {
		t.Fatal(err)
	}
	c, _ := s.connection(ctx, owner, false)
	snapshot := c.ActiveSnapshotID
	op, err = s.Queue(ctx, owner, "AUTH_BOOTSTRAP")
	if err != nil {
		t.Fatal(err)
	}
	if err = runner.apply(ctx, op, operationState{Status: "RUNNING", Stage: "READING_CATALOG", VerifiedSubject: "catalog-failure-teacher", StateVersion: 1}); err != nil {
		t.Fatal(err)
	}
	if err = runner.apply(ctx, op, operationState{Status: "FAILED", Stage: "READING_CATALOG", VerifiedSubject: "catalog-failure-teacher", StateVersion: 2}); err != nil {
		t.Fatal(err)
	}
	c, _ = s.connection(ctx, owner, false)
	if c.ActiveSnapshotID != snapshot || c.Status != models.PortalConnectionReady || c.ActiveOperationID != "" {
		t.Fatal("collection failure discarded a valid authentication or previous catalog")
	}
}
