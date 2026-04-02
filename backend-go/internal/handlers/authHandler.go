package handlers

import (
	"log"
	"net/http"

	"grading-gateway/internal/auth"
	"grading-gateway/internal/models"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
)

// LoginRequest 登录请求结构
type LoginRequest struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
}

// LoginResponse 登录响应结构
type LoginResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	UserID       uint   `json:"user_id"`
	Role         string `json:"role"`
	Name         string `json:"name"`
}

// RefreshRequest 刷新令牌请求
type RefreshRequest struct {
	RefreshToken string `json:"refresh_token" binding:"required"`
}

// RegisterRequest 注册请求结构
type RegisterRequest struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
	Name     string `json:"name" binding:"required"`
	Role     string `json:"role" binding:"required,oneof=student teacher admin"`
}

// AuthHandler 认证处理器
type AuthHandler struct {
	db *gorm.DB
}

// NewAuthHandler 创建认证处理器
func NewAuthHandler(db *gorm.DB) *AuthHandler {
	return &AuthHandler{db: db}
}

// Login 处理用户登录
func (h *AuthHandler) Login(c *gin.Context) {
	var req LoginRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		log.Printf("Login: 无效的请求格式: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "无效的请求格式，请提供 username 和 password",
		})
		return
	}

	// 查找用户
	var user models.User
	if err := h.db.Where("username = ?", req.Username).First(&user).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			log.Printf("Login: 用户不存在: %s", req.Username)
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "用户名或密码错误",
			})
			return
		}
		log.Printf("Login: 数据库查询错误: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "服务器内部错误",
		})
		return
	}

	// 验证密码
	if !auth.VerifyPassword(req.Password, user.PasswordHash) {
		log.Printf("Login: 密码错误 for user: %s", req.Username)
		c.JSON(http.StatusUnauthorized, gin.H{
			"error": "用户名或密码错误",
		})
		return
	}

	// 生成双令牌
	accessToken, refreshToken, err := auth.GenerateTokenPair(user.ID, user.Username, string(user.Role), nil)
	if err != nil {
		log.Printf("Login: 生成令牌失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "服务器内部错误",
		})
		return
	}

	// 返回响应
	c.JSON(http.StatusOK, LoginResponse{
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
		UserID:       user.ID,
		Role:         string(user.Role),
		Name:         user.Name,
	})
}

// Refresh 刷新访问令牌
func (h *AuthHandler) Refresh(c *gin.Context) {
	var req RefreshRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		log.Printf("Refresh: 无效的请求格式: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "无效的请求格式，请提供 refresh_token",
		})
		return
	}

	// 解析刷新令牌获取用户ID
	userID, err := auth.ParseRefreshToken(req.RefreshToken, nil)
	if err != nil {
		log.Printf("Refresh: 无效的刷新令牌: %v", err)
		c.JSON(http.StatusUnauthorized, gin.H{
			"error": "无效的刷新令牌",
		})
		return
	}

	// 查找用户
	var user models.User
	if err := h.db.First(&user, userID).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			log.Printf("Refresh: 用户不存在: %d", userID)
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "用户不存在",
			})
			return
		}
		log.Printf("Refresh: 数据库查询错误: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "服务器内部错误",
		})
		return
	}

	// 生成新的双令牌
	newAccessToken, newRefreshToken, err := auth.RefreshTokenPair(req.RefreshToken, user.ID, user.Username, string(user.Role), nil)
	if err != nil {
		log.Printf("Refresh: 刷新令牌失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "服务器内部错误",
		})
		return
	}

	// 返回响应
	c.JSON(http.StatusOK, LoginResponse{
		AccessToken:  newAccessToken,
		RefreshToken: newRefreshToken,
		UserID:       user.ID,
		Role:         string(user.Role),
		Name:         user.Name,
	})
}

// Register 处理用户注册（仅管理员可用，实际应用中应加权限控制）
func (h *AuthHandler) Register(c *gin.Context) {
	var req RegisterRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		log.Printf("Register: 无效的请求格式: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "无效的请求格式，请提供 username、password、name 和 role",
		})
		return
	}

	// 检查用户名是否已存在
	var existingUser models.User
	if err := h.db.Where("username = ?", req.Username).First(&existingUser).Error; err == nil {
		log.Printf("Register: 用户名已存在: %s", req.Username)
		c.JSON(http.StatusConflict, gin.H{
			"error": "用户名已存在",
		})
		return
	} else if err != gorm.ErrRecordNotFound {
		log.Printf("Register: 数据库查询错误: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "服务器内部错误",
		})
		return
	}

	// 哈希密码
	passwordHash, err := auth.HashPassword(req.Password)
	if err != nil {
		log.Printf("Register: 密码哈希失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "服务器内部错误",
		})
		return
	}

	// 创建用户
	user := models.User{
		Username:     req.Username,
		PasswordHash: passwordHash,
		Name:         req.Name,
		Role:         models.UserRole(req.Role),
	}

	if err := h.db.Create(&user).Error; err != nil {
		log.Printf("Register: 创建用户失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "服务器内部错误",
		})
		return
	}

	// 生成双令牌
	accessToken, refreshToken, err := auth.GenerateTokenPair(user.ID, user.Username, string(user.Role), nil)
	if err != nil {
		log.Printf("Register: 生成令牌失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "服务器内部错误",
		})
		return
	}

	// 返回响应
	c.JSON(http.StatusCreated, LoginResponse{
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
		UserID:       user.ID,
		Role:         string(user.Role),
		Name:         user.Name,
	})
}

// Profile 获取用户个人信息
func (h *AuthHandler) Profile(c *gin.Context) {
	// 从上下文中获取用户ID（需要 AuthMiddleware）
	userIDValue, exists := c.Get("userID")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{
			"error": "未授权访问",
		})
		return
	}

	userID, ok := userIDValue.(uint)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "内部服务器错误",
		})
		return
	}

	// 查找用户
	var user models.User
	if err := h.db.First(&user, userID).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{
				"error": "用户不存在",
			})
			return
		}
		log.Printf("Profile: 数据库查询错误: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "服务器内部错误",
		})
		return
	}

	// 返回用户信息（排除敏感字段）
	c.JSON(http.StatusOK, gin.H{
		"id":         user.ID,
		"username":   user.Username,
		"role":       user.Role,
		"name":       user.Name,
		"created_at": user.CreatedAt,
	})
}
