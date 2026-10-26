package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/xuri/excelize/v2"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
	"grading-gateway/internal/testutil"
	"net/http/httptest"
	"testing"
)

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
