package handlers

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"grading-gateway/internal/logbook"
	"grading-gateway/internal/middleware"
)

type LogHandler struct{ Service *logbook.Service }

func logActor(c *gin.Context) logbook.Actor {
	id, _ := middleware.GetUserIDFromContext(c)
	role, _ := middleware.GetRoleFromContext(c)
	return logbook.Actor{UserID: id, Admin: role == "admin"}
}
func logError(c *gin.Context, err error) {
	if errors.Is(err, logbook.ErrScope) {
		c.JSON(404, gin.H{"error": "任务或日志不存在，或无权查看"})
		return
	}
	if errors.Is(err, logbook.ErrQuery) {
		c.JSON(400, gin.H{"error": "筛选参数或分页游标无效，请重新搜索"})
		return
	}
	c.JSON(503, gin.H{"error": "日志索引暂不可用，请稍后重试"})
}
func logFilter(c *gin.Context) (logbook.Filter, error) {
	f := logbook.Filter{Source: c.DefaultQuery("source", "all"), Level: c.DefaultQuery("level", "all"), TaskID: strings.TrimSpace(c.Query("task_id")), Keyword: strings.TrimSpace(c.Query("q")), HideHTTP: c.DefaultQuery("hide_http", "true") != "false"}
	var err error
	if c.Query("from") == "" || c.Query("to") == "" {
		return f, logbook.ErrQuery
	}
	f.From, err = time.Parse(time.RFC3339Nano, c.Query("from"))
	if err != nil {
		return f, logbook.ErrQuery
	}
	f.To, err = time.Parse(time.RFC3339Nano, c.Query("to"))
	if err != nil {
		return f, logbook.ErrQuery
	}
	return f, nil
}
func (h LogHandler) Register(group *gin.RouterGroup) {
	group.Use(middleware.RBACMiddleware("admin"))
	group.Use(func(c *gin.Context) {
		c.Header("Cache-Control", "no-store")
		c.Header("X-Content-Type-Options", "nosniff")
		if !h.Service.Ready {
			c.AbortWithStatusJSON(503, gin.H{"error": "日志索引尚未就绪"})
			return
		}
		c.Next()
	})
	group.GET("/sources", h.Sources)
	group.GET("/tasks", h.Tasks)
	group.GET("/search", h.Search)
	group.GET("/export", h.Export)
	group.GET("/entries/:id/context", h.Context)
}
func (h LogHandler) Sources(c *gin.Context) {
	sources, err := h.Service.Sources(c.Request.Context(), logActor(c))
	if err != nil {
		logError(c, err)
		return
	}
	c.JSON(200, gin.H{"sources": sources, "retention_days": 30, "task_scoped": !logActor(c).Admin})
}
func (h LogHandler) Tasks(c *gin.Context) {
	items, err := h.Service.Tasks(c.Request.Context(), logActor(c))
	if err != nil {
		logError(c, err)
		return
	}
	c.JSON(200, items)
}
func (h LogHandler) Search(c *gin.Context) {
	f, err := logFilter(c)
	if err != nil {
		logError(c, err)
		return
	}
	limit, err := strconv.Atoi(c.DefaultQuery("limit", "100"))
	if err != nil {
		logError(c, logbook.ErrQuery)
		return
	}
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()
	page, err := h.Service.Search(ctx, logActor(c), f, c.Query("cursor"), limit)
	if err != nil {
		logError(c, err)
		return
	}
	c.JSON(200, page)
}
func (h LogHandler) Context(c *gin.Context) {
	id, err := strconv.ParseUint(c.Param("id"), 10, 64)
	if err != nil || id == 0 {
		logError(c, logbook.ErrQuery)
		return
	}
	items, err := h.Service.Context(c.Request.Context(), logActor(c), id)
	if err != nil {
		logError(c, err)
		return
	}
	c.JSON(200, gin.H{"items": items, "permission_filtered": !logActor(c).Admin})
}
func (h LogHandler) Export(c *gin.Context) {
	f, err := logFilter(c)
	if err != nil {
		logError(c, err)
		return
	}
	ctx, cancel := context.WithTimeout(c.Request.Context(), 8*time.Second)
	defer cancel()
	items, err := h.Service.Export(ctx, logActor(c), f)
	if err != nil {
		logError(c, err)
		return
	}
	var b strings.Builder
	b.WriteString("教学助手日志 · 已脱敏 · 最多2000条 · 时间含时区\n")
	for _, e := range items {
		estimated := ""
		if e.Estimated {
			estimated = " [时间估算]"
		}
		fmt.Fprintf(&b, "%s [%s] [%s]%s %s\n", e.At.Format(time.RFC3339Nano), e.Source, e.Level, estimated, e.Message)
	}
	c.Header("Content-Disposition", `attachment; filename="logs-`+time.Now().Format("20060102-150405")+`.txt"`)
	c.Data(http.StatusOK, "text/plain; charset=utf-8", []byte(b.String()))
}
