package handlers

import (
	"log"
	"net/http"

	"grading-gateway/internal/agent"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
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

	// 获取 Redis 记忆管理器
	memoryManager := agent.GetGlobalRedisMemoryManager()
	if memoryManager == nil {
		log.Printf("GetUserSessions: Redis 记忆管理器未初始化")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "系统内部错误",
		})
		return
	}

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

	// 获取 Redis 记忆管理器
	memoryManager := agent.GetGlobalRedisMemoryManager()
	if memoryManager == nil {
		log.Printf("GetSessionHistory: Redis 记忆管理器未初始化")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "系统内部错误",
		})
		return
	}

	// 检查会话是否存在（验证用户权限）
	exists, err := memoryManager.SessionExists(userID, sessionID)
	if err != nil {
		log.Printf("GetSessionHistory: 检查会话存在失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "系统内部错误",
		})
		return
	}

	if !exists {
		// 检查MySQL中是否存在该会话（可能Redis过期了）
		var chatSession models.ChatSession
		result := h.db.Where("id = ? AND user_id = ?", sessionID, userID).First(&chatSession)
		if result.Error != nil {
			log.Printf("GetSessionHistory: 会话不存在或无权访问: session_id=%s, user_id=%d", sessionID, userID)
			c.JSON(http.StatusNotFound, gin.H{
				"error": "会话不存在或您无权访问",
			})
			return
		}

		// 尝试从MySQL恢复会话到Redis
		if err := memoryManager.RestoreSessionToRedis(userID, sessionID); err != nil {
			log.Printf("GetSessionHistory: 恢复会话到Redis失败: %v", err)
		}
	}

	// 获取会话的历史消息
	messages, err := memoryManager.GetRecentMessages(userID, sessionID)
	if err != nil {
		log.Printf("GetSessionHistory: 获取会话消息失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "获取会话历史失败",
		})
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

	// 获取 Redis 记忆管理器
	memoryManager := agent.GetGlobalRedisMemoryManager()
	if memoryManager == nil {
		log.Printf("CreateSession: Redis 记忆管理器未初始化")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "系统内部错误",
		})
		return
	}

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

	// 获取 Redis 记忆管理器
	memoryManager := agent.GetGlobalRedisMemoryManager()
	if memoryManager != nil {
		// 清除Redis中的会话数据
		if err := memoryManager.ClearSession(userID, sessionID); err != nil {
			log.Printf("DeleteSession: 清除Redis会话数据失败: %v", err)
			// 继续执行，不中断删除流程
		}
	}

	// 从MySQL中删除会话和关联的消息
	tx := h.db.Begin()

	// 删除会话相关的消息
	if err := tx.Where("session_id = ?", sessionID).Delete(&models.ChatMessage{}).Error; err != nil {
		tx.Rollback()
		log.Printf("DeleteSession: 删除会话消息失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "删除会话失败",
		})
		return
	}

	// 删除会话
	if err := tx.Delete(&chatSession).Error; err != nil {
		tx.Rollback()
		log.Printf("DeleteSession: 删除会话失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "删除会话失败",
		})
		return
	}

	if err := tx.Commit().Error; err != nil {
		log.Printf("DeleteSession: 事务提交失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "删除会话失败",
		})
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

	// 获取 Redis 记忆管理器
	memoryManager := agent.GetGlobalRedisMemoryManager()
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
