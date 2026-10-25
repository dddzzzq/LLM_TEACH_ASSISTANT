package handlers

import (
	"errors"
	"strconv"

	"grading-gateway/internal/execution"
	"grading-gateway/internal/middleware"

	"github.com/gin-gonic/gin"
)

func executionError(c *gin.Context, err error) {
	if errors.Is(err, execution.ErrNotFound) {
		c.JSON(404, gin.H{"error": "会话或执行记录不存在，或无权查看"})
		return
	}
	c.JSON(503, gin.H{"error": "执行记录暂不可用"})
}

func (h *SessionHandler) GetExecutions(c *gin.Context) {
	user, err := middleware.GetUserIDFromContext(c)
	if err != nil {
		c.AbortWithStatus(401)
		return
	}
	runs, err := (execution.Store{DB: h.db}).Runs(c.Request.Context(), user, c.Param("session_id"), 20)
	if err != nil {
		executionError(c, err)
		return
	}
	c.JSON(200, gin.H{"runs": runs})
}

func (h *SessionHandler) GetExecutionEvents(c *gin.Context) {
	user, err := middleware.GetUserIDFromContext(c)
	if err != nil {
		c.AbortWithStatus(401)
		return
	}
	after, err := strconv.ParseUint(c.DefaultQuery("after_seq", "0"), 10, 64)
	if err != nil {
		c.JSON(400, gin.H{"error": "after_seq 无效"})
		return
	}
	limit, err := strconv.Atoi(c.DefaultQuery("limit", "50"))
	if err != nil || limit < 1 || limit > 200 {
		c.JSON(400, gin.H{"error": "limit 必须在 1 到 200 之间"})
		return
	}
	run, events, err := (execution.Store{DB: h.db}).Events(c.Request.Context(), user, c.Param("session_id"), c.Param("run_id"), after, limit)
	if err != nil {
		executionError(c, err)
		return
	}
	next := after
	if len(events) > 0 {
		next = events[len(events)-1].Sequence
	}
	c.JSON(200, gin.H{"run": run, "events": events, "next_seq": next, "has_more": next < run.Sequence})
}

func (h *SessionHandler) GetExecutionContext(c *gin.Context) {
	user, err := middleware.GetUserIDFromContext(c)
	if err != nil {
		c.AbortWithStatus(401)
		return
	}
	run, view, err := (execution.Store{DB: h.db}).Reconstruct(c.Request.Context(), user, c.Param("session_id"), c.Param("run_id"))
	if err != nil {
		executionError(c, err)
		return
	}
	c.JSON(200, gin.H{"run": run, "projection": view})
}
