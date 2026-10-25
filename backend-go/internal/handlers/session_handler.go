package handlers

import (
	"errors"
	"log"
	"net/http"
	"time"

	"grading-gateway/internal/agent"
	"grading-gateway/internal/execution"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// SessionHandler 会话处理器
type SessionHandler struct {
	db *gorm.DB
}

// NewSessionHandler 创建会话处理器
func NewSessionHandler(db *gorm.DB) *SessionHandler {
	return &SessionHandler{db: db}
}

// GetUserSessions 获取用户的所有会话
func (h *SessionHandler) GetUserSessions(c *gin.Context) {
	// 获取当前用户ID
	userID, err := middleware.GetUserIDFromContext(c)
	if err != nil {
		log.Printf("GetUserSessions: 未授权访问: %v", err)
		c.JSON(http.StatusUnauthorized, gin.H{
			"error": "未授权访问",
		})
		return
	}

	// 读取持久化会话服务（保留原接口格式）。
	memoryManager := agent.NewRedisMemoryManager(20, 24*time.Hour, h.db)

	// 获取用户的所有会话
	sessions, err := memoryManager.GetUserSessions(userID)
	if err != nil {
		log.Printf("GetUserSessions: 获取用户会话失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "获取会话列表失败",
		})
		return
	}

	c.JSON(http.StatusOK, sessions)
}

// GetSessionHistory 获取特定会话的历史消息
func (h *SessionHandler) GetSessionHistory(c *gin.Context) {
	// 获取当前用户ID
	userID, err := middleware.GetUserIDFromContext(c)
	if err != nil {
		log.Printf("GetSessionHistory: 未授权访问: %v", err)
		c.JSON(http.StatusUnauthorized, gin.H{
			"error": "未授权访问",
		})
		return
	}

	// 获取会话ID
	sessionID := c.Param("session_id")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "会话ID不能为空",
		})
		return
	}

	// 读取持久化会话服务（保留原接口格式）。
	memoryManager := agent.NewRedisMemoryManager(20, 24*time.Hour, h.db)

	// 获取会话的历史消息
	messages, err := memoryManager.GetRecentMessages(userID, sessionID)
	if err != nil {
		log.Printf("GetSessionHistory: 获取会话消息失败: %v", err)
		status := http.StatusInternalServerError
		if errors.Is(err, gorm.ErrRecordNotFound) || errors.Is(err, execution.ErrNotFound) {
			status = http.StatusNotFound
		}
		c.JSON(status, gin.H{"error": "会话不存在、无权访问或历史读取失败"})
		return
	}

	// 转换消息格式
	var responseMessages []gin.H
	for _, msg := range messages {
		responseMessages = append(responseMessages, gin.H{
			"role":      msg.Role,
			"content":   msg.Content,
			"timestamp": msg.Timestamp,
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"session_id": sessionID,
		"messages":   responseMessages,
		"count":      len(responseMessages),
	})
}

// CreateSession 创建新会话
func (h *SessionHandler) CreateSession(c *gin.Context) {
	// 获取当前用户ID
	userID, err := middleware.GetUserIDFromContext(c)
	if err != nil {
		log.Printf("CreateSession: 未授权访问: %v", err)
		c.JSON(http.StatusUnauthorized, gin.H{
			"error": "未授权访问",
		})
		return
	}

	// 解析请求
	var request struct {
		Title string `json:"title" binding:"required"`
	}
	if err := c.ShouldBindJSON(&request); err != nil {
		log.Printf("CreateSession: 无效的请求格式: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "无效的请求格式，请提供 title",
		})
		return
	}

	// 读取持久化会话服务（保留原接口格式）。
	memoryManager := agent.NewRedisMemoryManager(20, 24*time.Hour, h.db)

	// 创建新会话
	sessionID, err := memoryManager.CreateNewSession(userID, request.Title)
	if err != nil {
		log.Printf("CreateSession: 创建会话失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "创建会话失败",
		})
		return
	}

	// 返回新创建的会话
	var chatSession models.ChatSession
	if err := h.db.First(&chatSession, "id = ?", sessionID).Error; err != nil {
		log.Printf("CreateSession: 获取会话信息失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "获取会话信息失败",
		})
		return
	}

	c.JSON(http.StatusCreated, chatSession)
}

// DeleteSession 删除会话
func (h *SessionHandler) DeleteSession(c *gin.Context) {
	// 获取当前用户ID
	userID, err := middleware.GetUserIDFromContext(c)
	if err != nil {
		log.Printf("DeleteSession: 未授权访问: %v", err)
		c.JSON(http.StatusUnauthorized, gin.H{
			"error": "未授权访问",
		})
		return
	}

	// 获取会话ID
	sessionID := c.Param("session_id")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "会话ID不能为空",
		})
		return
	}

	// 验证用户是否有权删除该会话
	var chatSession models.ChatSession
	result := h.db.Where("id = ? AND user_id = ?", sessionID, userID).First(&chatSession)
	if result.Error != nil {
		log.Printf("DeleteSession: 会话不存在或无权访问: session_id=%s, user_id=%d", sessionID, userID)
		c.JSON(http.StatusNotFound, gin.H{
			"error": "会话不存在或您无权访问",
		})
		return
	}

	// 读取持久化会话服务（保留原接口格式）。
	memoryManager := agent.NewRedisMemoryManager(20, 24*time.Hour, h.db)
	if memoryManager != nil {
		// 清除Redis中的会话数据
		if err := memoryManager.ClearSession(userID, sessionID); err != nil {
			log.Printf("DeleteSession: 清除Redis会话数据失败: %v", err)
			// 继续执行，不中断删除流程
		}
	}

	// Match the session -> run locking order used by execution writes.
	err = h.db.WithContext(c.Request.Context()).Transaction(func(tx *gorm.DB) error {
		var session models.ChatSession
		if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).First(&session, "id = ? AND user_id = ?", sessionID, userID).Error; err != nil {
			return err
		}
		var active int64
		if err := tx.Model(&models.AgentRun{}).Where("session_id = ? AND status = ? AND lease_until > ?", sessionID, execution.Running, time.Now()).Count(&active).Error; err != nil {
			return err
		}
		if active > 0 {
			return execution.ErrBusy
		}
		for _, model := range []any{&models.AgentEvent{}, &models.AgentRun{}, &models.ChatMessage{}} {
			if err := tx.Where("session_id = ?", sessionID).Delete(model).Error; err != nil {
				return err
			}
		}
		return tx.Delete(&session).Error
	})
	if err != nil {
		if errors.Is(err, execution.ErrBusy) {
			c.JSON(409, gin.H{"error": "会话仍在执行，请完成后再删除"})
			return
		}
		c.JSON(500, gin.H{"error": "删除会话失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message": "会话删除成功",
	})
}

// UpdateSession 更新会话信息（如标题）
func (h *SessionHandler) UpdateSession(c *gin.Context) {
	// 获取当前用户ID
	userID, err := middleware.GetUserIDFromContext(c)
	if err != nil {
		log.Printf("UpdateSession: 未授权访问: %v", err)
		c.JSON(http.StatusUnauthorized, gin.H{
			"error": "未授权访问",
		})
		return
	}

	// 获取会话ID
	sessionID := c.Param("session_id")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "会话ID不能为空",
		})
		return
	}

	// 解析请求
	var request struct {
		Title string `json:"title" binding:"required"`
	}
	if err := c.ShouldBindJSON(&request); err != nil {
		log.Printf("UpdateSession: 无效的请求格式: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "无效的请求格式，请提供 title",
		})
		return
	}

	// 验证用户是否有权更新该会话
	var chatSession models.ChatSession
	result := h.db.Where("id = ? AND user_id = ?", sessionID, userID).First(&chatSession)
	if result.Error != nil {
		log.Printf("UpdateSession: 会话不存在或无权访问: session_id=%s, user_id=%d", sessionID, userID)
		c.JSON(http.StatusNotFound, gin.H{
			"error": "会话不存在或您无权访问",
		})
		return
	}

	// 更新会话标题
	chatSession.Title = request.Title
	if err := h.db.Save(&chatSession).Error; err != nil {
		log.Printf("UpdateSession: 更新会话失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "更新会话失败",
		})
		return
	}

	c.JSON(http.StatusOK, chatSession)
}

// GetSessionInfo 获取会话基本信息
func (h *SessionHandler) GetSessionInfo(c *gin.Context) {
	// 获取当前用户ID
	userID, err := middleware.GetUserIDFromContext(c)
	if err != nil {
		log.Printf("GetSessionInfo: 未授权访问: %v", err)
		c.JSON(http.StatusUnauthorized, gin.H{
			"error": "未授权访问",
		})
		return
	}

	// 获取会话ID
	sessionID := c.Param("session_id")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "会话ID不能为空",
		})
		return
	}

	// 验证用户是否有权访问该会话
	var chatSession models.ChatSession
	result := h.db.Where("id = ? AND user_id = ?", sessionID, userID).First(&chatSession)
	if result.Error != nil {
		log.Printf("GetSessionInfo: 会话不存在或无权访问: session_id=%s, user_id=%d", sessionID, userID)
		c.JSON(http.StatusNotFound, gin.H{
			"error": "会话不存在或您无权访问",
		})
		return
	}

	// 读取持久化会话服务（保留原接口格式）。
	memoryManager := agent.NewRedisMemoryManager(20, 24*time.Hour, h.db)
	var messageCount int64 = 0
	if memoryManager != nil {
		count, err := memoryManager.GetMessageCount(userID, sessionID)
		if err == nil {
			messageCount = count
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"session": chatSession,
		"stats": gin.H{
			"message_count": messageCount,
		},
	})
}
