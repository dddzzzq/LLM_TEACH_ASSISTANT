package middleware

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"grading-gateway/internal/auth"

	"github.com/gin-gonic/gin"
)

// 上下文键类型
type contextKey string

const (
	// 上下文键
	UserIDKey   contextKey = "userID"
	UsernameKey contextKey = "username"
	RoleKey     contextKey = "role"
)

// AuthMiddleware 认证中间件
func AuthMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 从 Authorization 头获取令牌
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error": "Authorization header is required",
			})
			return
		}

		// 检查 Bearer 前缀
		parts := strings.Split(authHeader, " ")
		if len(parts) != 2 || parts[0] != "Bearer" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error": "Authorization header must be in format: Bearer <token>",
			})
			return
		}

		tokenString := parts[1]

		// 解析访问令牌
		claims, err := auth.ParseAccessToken(tokenString, nil)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error": fmt.Sprintf("Invalid or expired token: %v", err),
			})
			return
		}

		// 将用户信息存储到上下文中
		c.Set(string(UserIDKey), claims.UserID)
		c.Set(string(UsernameKey), claims.Username)
		c.Set(string(RoleKey), claims.Role)

		// 继续处理请求
		c.Next()
	}
}

// RBACMiddleware 基于角色的访问控制中间件
func RBACMiddleware(allowedRoles ...string) gin.HandlerFunc {
	return func(c *gin.Context) {
		// 从上下文中获取角色
		roleValue, exists := c.Get(string(RoleKey))
		if !exists {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{
				"error": "Role not found in context",
			})
			return
		}

		role, ok := roleValue.(string)
		if !ok {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{
				"error": "Invalid role type in context",
			})
			return
		}

		// 检查角色是否在允许的列表中
		roleAllowed := false
		for _, allowedRole := range allowedRoles {
			if role == allowedRole {
				roleAllowed = true
				break
			}
		}

		if !roleAllowed {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"error": fmt.Sprintf("Access denied. Required role: %v, your role: %s", allowedRoles, role),
			})
			return
		}

		// 继续处理请求
		c.Next()
	}
}

// GetUserIDFromContext 从 Gin 上下文中获取用户ID
func GetUserIDFromContext(c *gin.Context) (uint, error) {
	userIDValue, exists := c.Get(string(UserIDKey))
	if !exists {
		return 0, fmt.Errorf("userID not found in context")
	}

	userID, ok := userIDValue.(uint)
	if !ok {
		return 0, fmt.Errorf("invalid userID type in context")
	}

	return userID, nil
}

// GetUsernameFromContext 从 Gin 上下文中获取用户名
func GetUsernameFromContext(c *gin.Context) (string, error) {
	usernameValue, exists := c.Get(string(UsernameKey))
	if !exists {
		return "", fmt.Errorf("username not found in context")
	}

	username, ok := usernameValue.(string)
	if !ok {
		return "", fmt.Errorf("invalid username type in context")
	}

	return username, nil
}

// GetRoleFromContext 从 Gin 上下文中获取角色
func GetRoleFromContext(c *gin.Context) (string, error) {
	roleValue, exists := c.Get(string(RoleKey))
	if !exists {
		return "", fmt.Errorf("role not found in context")
	}

	role, ok := roleValue.(string)
	if !ok {
		return "", fmt.Errorf("invalid role type in context")
	}

	return role, nil
}

// GetUserIDFromContextMust 从 Gin 上下文中获取用户ID，如果不存在则 panic
func GetUserIDFromContextMust(c *gin.Context) uint {
	userID, err := GetUserIDFromContext(c)
	if err != nil {
		panic(fmt.Sprintf("GetUserIDFromContextMust failed: %v", err))
	}
	return userID
}

// GetUsernameFromContextMust 从 Gin 上下文中获取用户名，如果不存在则 panic
func GetUsernameFromContextMust(c *gin.Context) string {
	username, err := GetUsernameFromContext(c)
	if err != nil {
		panic(fmt.Sprintf("GetUsernameFromContextMust failed: %v", err))
	}
	return username
}

// GetRoleFromContextMust 从 Gin 上下文中获取角色，如果不存在则 panic
func GetRoleFromContextMust(c *gin.Context) string {
	role, err := GetRoleFromContext(c)
	if err != nil {
		panic(fmt.Sprintf("GetRoleFromContextMust failed: %v", err))
	}
	return role
}

// ContextWithUserInfo 将用户信息添加到 context.Context 中
func ContextWithUserInfo(ctx context.Context, userID uint, role string) context.Context {
	ctx = context.WithValue(ctx, UserIDKey, userID)
	ctx = context.WithValue(ctx, RoleKey, role)
	return ctx
}

// GetUserIDFromContextContext 从 context.Context 中获取用户ID
func GetUserIDFromContextContext(ctx context.Context) (uint, bool) {
	userIDValue := ctx.Value(UserIDKey)
	if userIDValue == nil {
		return 0, false
	}
	userID, ok := userIDValue.(uint)
	return userID, ok
}

// GetRoleFromContextContext 从 context.Context 中获取角色
func GetRoleFromContextContext(ctx context.Context) (string, bool) {
	roleValue := ctx.Value(RoleKey)
	if roleValue == nil {
		return "", false
	}
	role, ok := roleValue.(string)
	return role, ok
}

// StudentAccessGuard 学生访问拦截中间件
// 学生只能访问 AI 教学助手，其他写操作 API 均返回 403 Forbidden
func StudentAccessGuard() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 从上下文中获取角色
		roleValue, exists := c.Get(string(RoleKey))
		if !exists {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{
				"error": "Role not found in context",
			})
			return
		}

		role, ok := roleValue.(string)
		if !ok {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{
				"error": "Invalid role type in context",
			})
			return
		}

		// 如果是学生角色，直接返回 403 Forbidden
		if role == "student" {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"error": "学生角色无权访问此接口，请联系教师或管理员",
			})
			return
		}

		// 教师和管理员可以继续访问
		c.Next()
	}
}
