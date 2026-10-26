package handlers

import (
	"encoding/json"
	"net/http"
	"path/filepath"
	"strconv"

	"github.com/gin-gonic/gin"
	"grading-gateway/internal/database"
	"grading-gateway/internal/grading"
	"grading-gateway/internal/materials"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"
)

func materialActor(c *gin.Context) materials.Actor {
	id, _ := middleware.GetUserIDFromContext(c)
	role, _ := middleware.GetRoleFromContext(c)
	return materials.Actor{UserID: id, Role: role}
}
func CreateMaterials(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 16384)
	var request materials.Request
	if c.ShouldBindJSON(&request) != nil {
		c.JSON(400, gin.H{"error": "材料整理参数无效"})
		return
	}
	batch, err := materials.Default().Create(c.Request.Context(), materialActor(c), request, c.GetHeader("Idempotency-Key"))
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	c.JSON(202, batch)
}

func RetryMaterials(c *gin.Context) {
	batch, err := materials.Default().Retry(c.Request.Context(), materialActor(c), c.Param("id"))
	if err != nil {
		c.JSON(409, gin.H{"error": err.Error()})
		return
	}
	c.JSON(202, batch)
}
func ListMaterials(c *gin.Context) {
	actor := materialActor(c)
	q := database.DB.WithContext(c.Request.Context()).Where("owner_id = ?", actor.UserID)
	if class := c.Query("class_id"); class != "" {
		q = q.Where("portal_class_id = ?", class)
	}
	if source := c.Query("source_job_id"); source != "" {
		q = q.Where("source_job_id = ?", source)
	}
	if value := c.Query("assignment_id"); value != "" {
		id, err := strconv.ParseUint(value, 10, 32)
		if err != nil {
			c.JSON(400, gin.H{"error": "作业ID无效"})
			return
		}
		q = q.Where("assignment_id = ?", id)
	}
	var batches []models.MaterialBatch
	if err := q.Order("created_at DESC").Limit(100).Find(&batches).Error; err != nil {
		c.JSON(503, gin.H{"error": "无法读取材料批次"})
		return
	}
	ids := []string{}
	for _, b := range batches {
		if b.GradingJobID != "" {
			ids = append(ids, b.GradingJobID)
		}
	}
	jobs := []models.AsyncJob{}
	if len(ids) > 0 {
		database.DB.Where("id IN ? AND owner_id = ?", ids, actor.UserID).Find(&jobs)
	}
	c.Header("Cache-Control", "no-store")
	c.JSON(200, gin.H{"items": batches, "grading_jobs": jobs})
}
func GetMaterials(c *gin.Context) {
	batch, err := materials.Default().Owned(c.Request.Context(), materialActor(c), c.Param("id"))
	if err != nil {
		c.JSON(404, gin.H{"error": err.Error()})
		return
	}
	var manifest materials.Manifest
	_ = json.Unmarshal([]byte(batch.ManifestJSON), &manifest)
	var job models.AsyncJob
	if batch.GradingJobID != "" {
		database.DB.Where("id = ? AND owner_id = ?", batch.GradingJobID, batch.OwnerID).First(&job)
	}
	prepared := ""
	if batch.ManifestPath != "" {
		prepared = filepath.Dir(batch.ManifestPath)
	}
	c.Header("Cache-Control", "no-store")
	c.JSON(200, gin.H{"batch": batch, "manifest": manifest, "prepared_dir": prepared, "grading_job": job})
}
func GradeMaterials(c *gin.Context) {
	var request struct {
		AllowPartial bool `json:"allow_partial"`
	}
	if c.ShouldBindJSON(&request) != nil {
		c.JSON(400, gin.H{"error": "批改参数无效"})
		return
	}
	actor := materialActor(c)
	result, err := grading.SubmitBatch(c.Request.Context(), grading.Actor{UserID: actor.UserID, Role: actor.Role}, grading.Request{Kind: "homework", BatchID: c.Param("id"), AllowPartial: request.AllowPartial})
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	code := 202
	if result.Outcome == "needs_input" {
		code = 409
	}
	c.JSON(code, result)
}
func CancelMaterials(c *gin.Context) {
	if err := materials.Default().Cancel(c.Request.Context(), materialActor(c), c.Param("id")); err != nil {
		c.JSON(409, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, gin.H{"ok": true})
}
