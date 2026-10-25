package handlers

import (
	"github.com/gin-gonic/gin"
	"grading-gateway/internal/grading"
	"grading-gateway/internal/middleware"
	"net/http"
)

func gradingActor(c *gin.Context) (grading.Actor, bool) {
	userID, _ := middleware.GetUserIDFromContext(c)
	role, _ := middleware.GetRoleFromContext(c)
	actor := grading.Actor{UserID: userID, Role: role}
	if err := actor.Check(); err != nil {
		c.JSON(http.StatusForbidden, gin.H{"error": err.Error(), "detail": err.Error()})
		return actor, false
	}
	return actor, true
}
func checkGradingTarget(c *gin.Context, actor grading.Actor, kind, id string) bool {
	result, err := grading.Default().Check(c.Request.Context(), actor, kind, id)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error(), "detail": err.Error()})
		return false
	}
	if result.Outcome == "needs_input" {
		c.JSON(http.StatusUnprocessableEntity, gin.H{"outcome": result.Outcome, "error": result.Message, "detail": result.Message, "missing": result.Missing, "result_url": result.ResultURL})
		return false
	}
	return true
}
func submitGradingUpload(c *gin.Context, actor grading.Actor, request grading.Request) {
	result, err := grading.Default().Submit(c.Request.Context(), actor, request, "")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error(), "detail": err.Error()})
		return
	}
	status := http.StatusAccepted
	if result.Outcome == "needs_input" {
		status = http.StatusUnprocessableEntity
	}
	if result.Outcome == "failed" {
		status = http.StatusServiceUnavailable
	}
	c.JSON(status, gin.H{"outcome": result.Outcome, "job_id": result.JobID, "job_type": result.JobType, "status": result.Status, "message": result.Message, "missing": result.Missing, "result_url": result.ResultURL, "file_path": request.FilePath, "image_paths": request.ImagePaths})
}
