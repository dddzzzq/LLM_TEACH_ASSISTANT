package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/xuri/excelize/v2"
	"grading-gateway/internal/database"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"
	"grading-gateway/internal/testutil"
	"net/http/httptest"
	"testing"
)

func TestIntegrationAssignmentCatalogListsEmptyClassesAndKeepsUnassignedResults(t *testing.T) {
	db, owner := testutil.ExecutionDB(t)
	old := database.DB
	database.DB = db
	t.Cleanup(func() { database.DB = old })
	if err := db.AutoMigrate(&models.Assignment{}, &models.Submission{}, &models.MaterialBatch{}, &models.PortalConnection{}, &models.PortalHomework{}, &models.PortalCourseClass{}); err != nil {
		t.Fatal(err)
	}
	connection := models.PortalConnection{ID: uuid.NewString(), OwnerID: owner, Provider: "test", ActiveSnapshotID: uuid.NewString(), AccountMask: "test", CredentialCiphertext: []byte{}, CredentialNonce: []byte{}}
	if err := db.Create(&connection).Error; err != nil {
		t.Fatal(err)
	}
	a := models.Assignment{OwnerID: owner, CourseName: "课程", TaskName: "多班作业", ClassName: "全部班级", PortalConnectionID: connection.ID, PortalHomeworkID: uuid.NewString()}
	if err := db.Create(&a).Error; err != nil {
		t.Fatal(err)
	}
	homework := models.PortalHomework{ID: a.PortalHomeworkID, ConnectionID: connection.ID, CourseID: uuid.NewString(), SnapshotID: connection.ActiveSnapshotID, AssignmentID: a.ID}
	if err := db.Create(&homework).Error; err != nil {
		t.Fatal(err)
	}
	currentClasses := []models.PortalCourseClass{{ID: uuid.NewString(), CourseID: homework.CourseID, SnapshotID: connection.ActiveSnapshotID, Name: "一班"}, {ID: uuid.NewString(), CourseID: homework.CourseID, SnapshotID: connection.ActiveSnapshotID, Name: "空班级"}}
	if err := db.Create(&currentClasses).Error; err != nil {
		t.Fatal(err)
	}
	batch := models.MaterialBatch{ID: uuid.NewString(), AssignmentID: a.ID, OwnerID: owner, PortalClassID: currentClasses[0].ID, ClassName: "一班", Status: "NEEDS_REVIEW", IssueCount: 2, GradingState: "NOT_REQUESTED"}
	if err := db.Create(&batch).Error; err != nil {
		t.Fatal(err)
	}
	for _, sub := range []models.Submission{{AssignmentID: a.ID, MaterialBatchID: batch.ID, StudentID: "same-id", Score: 90, IsHumanReviewed: true, HumanScore: 88}, {AssignmentID: a.ID, StudentID: "same-id", Score: 50}} {
		if err := db.Create(&sub).Error; err != nil {
			t.Fatal(err)
		}
	}
	classes, err := assignmentClassCatalog(context.Background(), db, a)
	if err != nil || len(classes) != 3 {
		t.Fatalf("empty portal classes or legacy scope missing: %+v %v", classes, err)
	}
	byID := map[string]AssignmentClass{}
	for _, class := range classes {
		byID[class.ID] = class
	}
	first := byID[currentClasses[0].ID]
	if first.MaterialStatus != "ATTENTION" || first.GradingStatus != "GRADED" || first.IssueCount != 2 || first.GradedCount != 1 || first.ReviewedCount != 1 {
		t.Fatalf("class summary does not reflect material and review state: %+v", first)
	}
	if legacy := byID[unassignedClassID]; legacy.GradedCount != 1 || legacy.ReviewedCount != 0 {
		t.Fatalf("legacy class summary mixed another class: %+v", legacy)
	}
	var filtered []models.Submission
	q := submissionClassScope(db.Where("assignment_id = ?", a.ID), db, fmt.Sprint(a.ID), unassignedClassID)
	if err := q.Find(&filtered).Error; err != nil || len(filtered) != 1 || filtered[0].Score != 50 {
		t.Fatalf("unassigned scope mixed or lost legacy result: %+v %v", filtered, err)
	}
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(func(c *gin.Context) {
		c.Set(string(middleware.UserIDKey), owner+1)
		c.Set(string(middleware.RoleKey), "teacher")
	})
	router.GET("/assignments/:id/classes", middleware.AssignmentAccess(db, false), GetAssignmentClasses)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, httptest.NewRequest("GET", fmt.Sprintf("/assignments/%d/classes", a.ID), nil))
	if w.Code != 404 {
		t.Fatalf("another teacher read the class catalog: %d %s", w.Code, w.Body.String())
	}
}

func TestIntegrationAssignmentResultsAndExportRespectStableClassIDs(t *testing.T) {
	db, owner := testutil.ExecutionDB(t)
	old := database.DB
	database.DB = db
	t.Cleanup(func() { database.DB = old })
	if err := db.AutoMigrate(&models.Assignment{}, &models.Submission{}, &models.MaterialBatch{}); err != nil {
		t.Fatal(err)
	}
	a := models.Assignment{OwnerID: owner, CourseName: "同步课程", TaskName: "同步作业", ClassName: "全部班级", Question: "官网题目", Rubric: `{"正确性":100}`}
	if err := db.Create(&a).Error; err != nil {
		t.Fatal(err)
	}
	batchIDs := []string{uuid.NewString(), uuid.NewString()}
	for i, classID := range []string{"class-a", "class-b"} {
		batch := models.MaterialBatch{ID: batchIDs[i], OwnerID: owner, AssignmentID: a.ID, PortalClassID: classID, ClassName: "同名班级"}
		if err := db.Create(&batch).Error; err != nil {
			t.Fatal(err)
		}
		sub := models.Submission{AssignmentID: a.ID, StudentID: fmt.Sprintf("student-%d", i), StudentName: "学生", MaterialBatchID: batch.ID, Score: 90}
		if err := db.Create(&sub).Error; err != nil {
			t.Fatal(err)
		}
	}
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.GET("/assignments/:id/results", GetAssignmentSubmissions)
	router.GET("/assignments/:id/export", ExportAssignmentExcel)
	request := func(path string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		router.ServeHTTP(w, httptest.NewRequest("GET", path, nil))
		return w
	}
	base := fmt.Sprintf("/assignments/%d", a.ID)
	response := request(base + "/results?class_id=class-a")
	var results []struct {
		StudentID string `json:"student_id"`
	}
	if json.Unmarshal(response.Body.Bytes(), &results) != nil || len(results) != 1 || results[0].StudentID != "student-0" {
		t.Fatal("results mixed classes with identical display names")
	}
	response = request(base + "/results?class_id=class-a&batch_id=" + batchIDs[1])
	_ = json.Unmarshal(response.Body.Bytes(), &results)
	if len(results) != 0 {
		t.Fatal("batch filter widened class scope")
	}
	response = request(base + "/export?class_id=class-b")
	if response.Code != 200 {
		t.Fatal("class export failed")
	}
	workbook, err := excelize.OpenReader(bytes.NewReader(response.Body.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	defer workbook.Close()
	rows, err := workbook.GetRows("成绩单")
	if err != nil || len(rows) != 2 || rows[1][0] != "student-1-学生" {
		t.Fatal("export included another class or missed selected class")
	}
}

func TestIntegrationDeleteResultsKeepsOtherClassesAndMaterials(t *testing.T) {
	db, owner := testutil.ExecutionDB(t)
	old := database.DB
	database.DB = db
	t.Cleanup(func() { database.DB = old })
	if err := db.AutoMigrate(&models.Assignment{}, &models.Submission{}, &models.MaterialBatch{}, &models.AsyncJob{}); err != nil {
		t.Fatal(err)
	}
	a := models.Assignment{OwnerID: owner, CourseName: "测试课程", TaskName: "按班删除", ClassName: "全部班级", PortalHomeworkID: "portal"}
	if err := db.Create(&a).Error; err != nil {
		t.Fatal(err)
	}
	for _, class := range []string{"class-a", "class-a", "class-b"} {
		b := models.MaterialBatch{ID: uuid.NewString(), AssignmentID: a.ID, OwnerID: owner, PortalClassID: class, ClassName: "同名班级", Status: "READY"}
		if err := db.Create(&b).Error; err != nil {
			t.Fatal(err)
		}
		if err := db.Create(&models.Submission{AssignmentID: a.ID, MaterialBatchID: b.ID, StudentID: "same-id", Score: 90}).Error; err != nil {
			t.Fatal(err)
		}
	}
	db.Create(&models.Submission{AssignmentID: a.ID, StudentID: "legacy", Score: 60})
	var activeBatch models.MaterialBatch
	db.Where("assignment_id = ? AND portal_class_id = ?", a.ID, "class-a").First(&activeBatch)
	job := models.AsyncJob{ID: uuid.NewString(), OwnerID: owner, JobType: models.JobTypeHomework, ReferenceID: fmt.Sprint(a.ID), MaterialBatchID: activeBatch.ID, Status: models.JobStatusProcessing}
	if err := db.Create(&job).Error; err != nil {
		t.Fatal(err)
	}
	requestOwner := owner
	role := "teacher"
	router := gin.New()
	router.Use(func(c *gin.Context) {
		c.Set(string(middleware.UserIDKey), requestOwner)
		c.Set(string(middleware.RoleKey), role)
	})
	router.DELETE("/assignments/:id/results", middleware.AssignmentAccess(db, false), ClearAllSubmissions)
	base := fmt.Sprintf("/assignments/%d/results", a.ID)
	request := func(suffix string) int {
		w := httptest.NewRecorder()
		router.ServeHTTP(w, httptest.NewRequest("DELETE", base+suffix, nil))
		return w.Code
	}
	count := func() int64 {
		var n int64
		db.Model(&models.Submission{}).Where("assignment_id = ?", a.ID).Count(&n)
		return n
	}
	requestOwner = owner + 1
	if status := request("?class_id=class-b"); status != 404 {
		t.Fatal("other teacher deleted results", status)
	}
	requestOwner = owner
	role = "student"
	if status := request("?class_id=class-b"); status != 403 {
		t.Fatal("student deleted results", status)
	}
	role = "teacher"
	if status := request("?class_id="); status != 400 || count() != 4 {
		t.Fatal("empty class widened deletion", status)
	}
	if status := request("?class_id=local:invalid"); status != 204 || count() != 4 {
		t.Fatal("invalid class widened deletion", status)
	}
	if status := request("?class_id=class-b"); status != 204 || count() != 3 {
		t.Fatal("active other class blocked deletion or scope widened", status)
	}
	if status := request("?class_id=class-a"); status != 409 || count() != 3 {
		t.Fatal("deleted results during grading", status)
	}
	db.Model(&job).Update("status", models.JobStatusSuccess)
	if status := request("?class_id=class-a"); status != 204 || count() != 1 {
		t.Fatal("did not delete all batches of this class only", status)
	}
	if status := request("?class_id=__unassigned__"); status != 204 || count() != 0 {
		t.Fatal("legacy scope could not be cleared", status)
	}
	var batches int64
	db.Model(&models.MaterialBatch{}).Where("assignment_id = ?", a.ID).Count(&batches)
	if batches != 3 {
		t.Fatal("deletion removed prepared materials")
	}
}
