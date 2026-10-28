package handlers

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"
	"grading-gateway/internal/portal"
)

type PortalHandler struct{ Runner *portal.Runner }

func NewPortalHandler(db *gorm.DB) *PortalHandler { return &PortalHandler{portal.Runtime(db)} }
func (h *PortalHandler) Register(g *gin.RouterGroup) {
	g.Use(func(c *gin.Context) { c.Header("Cache-Control", "no-store"); c.Next() })
	g.GET("/connection", h.Connection)
	g.PUT("/connection", h.Save)
	g.DELETE("/connection", h.Disconnect)
	g.POST("/connection/reauth", h.Reauth)
	g.POST("/catalog-syncs", h.Sync)
	g.PUT("/imports", h.PauseImports)
	g.GET("/courses", h.Courses)
	g.GET("/courses/:id/homeworks", h.Homeworks)
	g.GET("/assignments/:id", h.AssignmentPortal)
	g.PUT("/assignments/:id/rubric", h.AssignmentRubric)
	g.POST("/assignments/:id/grade", h.GradeAssignmentClasses)
	g.GET("/homeworks/:id/detail", h.HomeworkDetail)
	g.POST("/homeworks/:id/detail-review", h.ReviewHomeworkDetail)
	g.POST("/homeworks/:id/detail-reads", h.ReadHomeworkDetail)
	g.GET("/homeworks/:id/detail/sources/:source", h.HomeworkDetailFile)
	g.GET("/operations/:id", h.Operation)
	g.POST("/operations/:id/cancel", h.CancelDetailRead)
	g.GET("/operations/:id/events", h.Events)
	g.GET("/operations/:id/browser", h.Browser)
	g.POST("/auth-runs/:id/:action", h.AuthAction)
	g.GET("/legacy-assignments", h.LegacyAssignments)
	g.PUT("/legacy-assignments/:id", h.ClaimAssignment)
}
func portalOwner(c *gin.Context) uint { owner, _ := middleware.GetUserIDFromContext(c); return owner }
func portalAssignmentID(c *gin.Context) (uint, bool) {
	id, err := strconv.ParseUint(c.Param("id"), 10, 32)
	if err != nil || id == 0 {
		c.JSON(400, gin.H{"detail": "作业ID无效"})
		return 0, false
	}
	return uint(id), true
}
func (h *PortalHandler) AssignmentPortal(c *gin.Context) {
	id, ok := portalAssignmentID(c)
	if !ok {
		return
	}
	data, err := h.Runner.Service.AssignmentPortal(c.Request.Context(), portalOwner(c), id)
	if err != nil {
		portalError(c, err)
		return
	}
	c.JSON(200, data)
}
func (h *PortalHandler) AssignmentRubric(c *gin.Context) {
	id, ok := portalAssignmentID(c)
	if !ok {
		return
	}
	var req struct {
		Rubric json.RawMessage `json:"rubric"`
	}
	if !portalBody(c, &req) {
		return
	}
	if err := h.Runner.Service.SaveAssignmentRubric(c.Request.Context(), portalOwner(c), id, req.Rubric); err != nil {
		portalError(c, err)
		return
	}
	c.JSON(200, gin.H{"message": "评分标准已保存，后续班级批改使用此标准"})
}
func (h *PortalHandler) GradeAssignmentClasses(c *gin.Context) {
	id, ok := portalAssignmentID(c)
	if !ok {
		return
	}
	var req struct {
		ClassIDs []string `json:"class_ids"`
	}
	if !portalBody(c, &req) {
		return
	}
	flow, err := h.Runner.Service.GradeAssignmentClasses(c.Request.Context(), portalOwner(c), id, req.ClassIDs, c.GetHeader("Idempotency-Key"))
	if err != nil {
		portalError(c, err)
		return
	}
	c.JSON(202, flow)
}
func portalError(c *gin.Context, err error) {
	status := 400
	message := err.Error()
	switch {
	case errors.Is(err, portal.ErrNotFound), errors.Is(err, gorm.ErrRecordNotFound):
		status = 404
		message = portal.ErrNotFound.Error()
	case errors.Is(err, portal.ErrConflict):
		status = 409
	case errors.Is(err, portal.ErrUnavailable):
		status = 503
	}
	// SQL errors can contain encrypted values or raw parameters; never serialize them.
	if !errors.Is(err, portal.ErrNotFound) && !errors.Is(err, portal.ErrConflict) && !errors.Is(err, portal.ErrUnavailable) {
		if strings.Contains(message, "Error ") || strings.Contains(message, "sql:") || strings.Contains(message, "dial tcp") {
			status = 503
			message = "教务数据暂不可用"
		}
	}
	c.JSON(status, gin.H{"detail": message})
}
func portalBody(c *gin.Context, target any) bool {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 256*1024)
	decoder := json.NewDecoder(c.Request.Body)
	decoder.DisallowUnknownFields()
	if decoder.Decode(target) != nil {
		c.JSON(400, gin.H{"detail": "请求格式无效"})
		return false
	}
	if decoder.Decode(&struct{}{}) != io.EOF {
		c.JSON(400, gin.H{"detail": "请求只能包含一个 JSON 对象"})
		return false
	}
	return true
}
func (h *PortalHandler) Connection(c *gin.Context) {
	v, err := h.Runner.Service.View(c.Request.Context(), portalOwner(c))
	if err != nil {
		portalError(c, err)
		return
	}
	c.JSON(200, v)
}
func (h *PortalHandler) Save(c *gin.Context) {
	var req portal.SaveRequest
	if !portalBody(c, &req) {
		return
	}
	op, err := h.Runner.Service.Save(c.Request.Context(), portalOwner(c), req)
	req.Password = ""
	if err != nil {
		portalError(c, err)
		return
	}
	c.JSON(202, gin.H{"operation_id": op.ID, "operation": op})
}
func (h *PortalHandler) Disconnect(c *gin.Context) {
	if err := h.Runner.Service.Disconnect(c.Request.Context(), portalOwner(c)); err != nil {
		portalError(c, err)
		return
	}
	c.JSON(202, gin.H{"message": "连接已停用，正在删除专用浏览器状态"})
}
func (h *PortalHandler) queue(c *gin.Context, kind string) {
	op, err := h.Runner.Service.Queue(c.Request.Context(), portalOwner(c), kind)
	if err != nil {
		portalError(c, err)
		return
	}
	c.JSON(202, gin.H{"operation_id": op.ID, "operation": op})
}
func (h *PortalHandler) Reauth(c *gin.Context) { h.queue(c, "AUTH_BOOTSTRAP") }
func (h *PortalHandler) Sync(c *gin.Context)   { h.queue(c, "CATALOG_SYNC") }
func (h *PortalHandler) PauseImports(c *gin.Context) {
	var req struct {
		Paused bool `json:"paused"`
	}
	if !portalBody(c, &req) {
		return
	}
	if err := h.Runner.Service.PauseImports(c.Request.Context(), portalOwner(c), req.Paused); err != nil {
		portalError(c, err)
		return
	}
	c.JSON(200, gin.H{"paused": req.Paused})
}
func (h *PortalHandler) Courses(c *gin.Context) {
	rows, err := h.Runner.Service.Courses(c.Request.Context(), portalOwner(c))
	if err != nil {
		portalError(c, err)
		return
	}
	c.JSON(200, rows)
}
func (h *PortalHandler) Homeworks(c *gin.Context) {
	data, err := h.Runner.Service.Homeworks(c.Request.Context(), portalOwner(c), c.Param("id"))
	if err != nil {
		portalError(c, err)
		return
	}
	c.JSON(200, data)
}
func (h *PortalHandler) HomeworkDetail(c *gin.Context) {
	data, err := h.Runner.Service.HomeworkDetail(c.Request.Context(), portalOwner(c), c.Param("id"))
	if err != nil {
		portalError(c, err)
		return
	}
	c.JSON(200, data)
}
func (h *PortalHandler) ReadHomeworkDetail(c *gin.Context) {
	var req struct {
		SnapshotID string `json:"snapshot_id"`
	}
	if !portalBody(c, &req) {
		return
	}
	op, err := h.Runner.Service.QueueHomeworkDetail(c.Request.Context(), portalOwner(c), c.Param("id"), req.SnapshotID)
	if err != nil {
		portalError(c, err)
		return
	}
	c.JSON(202, gin.H{"operation_id": op.ID, "operation": op})
}
func (h *PortalHandler) ReviewHomeworkDetail(c *gin.Context) {
	var req portal.ReviewDetailRequest
	if !portalBody(c, &req) {
		return
	}
	if err := h.Runner.Service.ReviewHomeworkDetail(c.Request.Context(), portalOwner(c), c.Param("id"), req); err != nil {
		portalError(c, err)
		return
	}
	c.JSON(200, gin.H{"message": "作业要求已核对，后续批改使用已确认的正文", "requirements_status": "READY"})
}
func (h *PortalHandler) HomeworkDetailFile(c *gin.Context) {
	path, name, err := h.Runner.Service.HomeworkDetailFile(c.Request.Context(), portalOwner(c), c.Param("id"), c.Param("source"))
	if err != nil {
		portalError(c, err)
		return
	}
	c.Header("X-Content-Type-Options", "nosniff")
	c.FileAttachment(path, name)
}
func (h *PortalHandler) CancelDetailRead(c *gin.Context) {
	if err := h.Runner.CancelDetailRead(c.Request.Context(), portalOwner(c), c.Param("id")); err != nil {
		portalError(c, err)
		return
	}
	c.JSON(200, gin.H{"message": "作业要求读取已取消"})
}
func (h *PortalHandler) Operation(c *gin.Context) {
	op, err := h.Runner.Service.OwnedOperation(c.Request.Context(), portalOwner(c), c.Param("id"))
	if err != nil {
		portalError(c, err)
		return
	}
	c.JSON(200, op)
}
func (h *PortalHandler) Events(c *gin.Context) {
	after, _ := strconv.ParseUint(c.Query("after"), 10, 64)
	rows, err := h.Runner.Service.Events(c.Request.Context(), portalOwner(c), c.Param("id"), after)
	if err != nil {
		portalError(c, err)
		return
	}
	c.JSON(200, rows)
}
func (h *PortalHandler) Browser(c *gin.Context) {
	raw, err := h.Runner.Control(c.Request.Context(), portalOwner(c), c.Param("id"), "browser", nil)
	if err != nil {
		c.JSON(409, gin.H{"detail": "认证浏览器当前不可用，请刷新状态"})
		return
	}
	c.Data(200, "application/json", raw)
}
func (h *PortalHandler) AuthAction(c *gin.Context) {
	if _, err := h.Runner.Service.OwnedOperation(c.Request.Context(), portalOwner(c), c.Param("id")); err != nil {
		portalError(c, err)
		return
	}
	c.JSON(410, gin.H{"detail": "请在原 RPA 任务浏览器完成认证，并使用原暂停与交还自动化操作"})
}

func (h *PortalHandler) CreateWorkflow(c *gin.Context) {
	var req portal.WorkflowRequest
	if !portalBody(c, &req) {
		return
	}
	flow, err := h.Runner.Service.CreateWorkflow(c.Request.Context(), portalOwner(c), req, c.GetHeader("Idempotency-Key"))
	if err != nil {
		portalError(c, err)
		return
	}
	c.JSON(202, flow)
}
func (h *PortalHandler) Workflow(c *gin.Context) {
	data, err := h.Runner.Service.Workflow(c.Request.Context(), portalOwner(c), c.Param("id"))
	if err != nil {
		portalError(c, err)
		return
	}
	c.JSON(200, data)
}
func (h *PortalHandler) CancelWorkflow(c *gin.Context) {
	if err := h.Runner.Service.CancelWorkflow(c.Request.Context(), portalOwner(c), c.Param("id")); err != nil {
		portalError(c, err)
		return
	}
	c.JSON(200, gin.H{"ok": true})
}
func (h *PortalHandler) RetryWorkflow(c *gin.Context) {
	if err := h.Runner.Service.RetryWorkflow(c.Request.Context(), portalOwner(c), c.Param("id")); err != nil {
		portalError(c, err)
		return
	}
	c.JSON(202, gin.H{"ok": true})
}
func (h *PortalHandler) LegacyAssignments(c *gin.Context) {
	role, _ := middleware.GetRoleFromContext(c)
	if role != "admin" {
		c.AbortWithStatus(403)
		return
	}
	if h.Runner.Service.DB == nil {
		portalError(c, portal.ErrUnavailable)
		return
	}
	rows := []models.Assignment{}
	if err := h.Runner.Service.DB.Where("owner_id = 0").Order("id DESC").Limit(200).Find(&rows).Error; err != nil {
		portalError(c, err)
		return
	}
	c.JSON(200, rows)
}
func (h *PortalHandler) ClaimAssignment(c *gin.Context) {
	role, _ := middleware.GetRoleFromContext(c)
	if role != "admin" {
		c.AbortWithStatus(403)
		return
	}
	var body struct {
		OwnerID uint `json:"owner_id"`
	}
	if !portalBody(c, &body) {
		return
	}
	db := h.Runner.Service.DB
	if db == nil {
		portalError(c, portal.ErrUnavailable)
		return
	}
	var user models.User
	if db.Where("id = ? AND role IN ?", body.OwnerID, []string{"teacher", "admin"}).First(&user).Error != nil {
		c.JSON(400, gin.H{"detail": "请选择有效教师"})
		return
	}
	result := db.Model(&models.Assignment{}).Where("id = ? AND owner_id = 0", c.Param("id")).Update("owner_id", body.OwnerID)
	if result.Error != nil {
		portalError(c, result.Error)
		return
	}
	if result.RowsAffected != 1 {
		c.JSON(409, gin.H{"detail": "作业已分配或不存在"})
		return
	}
	c.JSON(200, gin.H{"ok": true})
}
