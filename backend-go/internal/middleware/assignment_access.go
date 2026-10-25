package middleware

import (
	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
	"grading-gateway/internal/models"
	"net/http"
)

// AssignmentAccess authorizes before any handler can read shared caches or export results.
// Unclaimed legacy assignments are visible only to admins and cannot be modified.
func AssignmentAccess(db *gorm.DB, submission bool) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Param("id") == "" {
			c.Next()
			return
		}
		if role, _ := GetRoleFromContext(c); role == "student" && c.Request.Method != http.MethodGet {
			c.AbortWithStatusJSON(403, gin.H{"detail": "学生无权修改作业或成绩"})
			return
		}
		if db == nil {
			c.AbortWithStatusJSON(503, gin.H{"detail": "数据库未就绪"})
			return
		}
		owner, _ := GetUserIDFromContext(c)
		role, _ := GetRoleFromContext(c)
		username, _ := GetUsernameFromContext(c)
		assignmentID := c.Param("id")
		var sub models.Submission
		if submission {
			if db.WithContext(c.Request.Context()).First(&sub, "id = ?", assignmentID).Error != nil {
				c.AbortWithStatusJSON(404, gin.H{"detail": "记录不存在或无权访问"})
				return
			}
			assignmentID = ""
		}
		var a models.Assignment
		q := db.WithContext(c.Request.Context())
		var err error
		if submission {
			err = q.First(&a, sub.AssignmentID).Error
		} else {
			err = q.First(&a, "id = ?", assignmentID).Error
		}
		if err != nil {
			c.AbortWithStatusJSON(404, gin.H{"detail": "记录不存在或无权访问"})
			return
		}
		allowed := owner != 0 && a.OwnerID == owner
		if role == "admin" && c.Request.Method == http.MethodGet {
			allowed = true
		}
		if role == "student" && c.Request.Method == http.MethodGet {
			if submission {
				allowed = sub.StudentID == username
			} else {
				var count int64
				db.Model(&models.Submission{}).Where("assignment_id = ? AND student_id = ?", a.ID, username).Count(&count)
				allowed = count > 0
			}
			if c.FullPath() == "/assignments/:id/export" {
				allowed = false
			}
		}
		if !allowed {
			c.AbortWithStatusJSON(404, gin.H{"detail": "记录不存在或无权访问；历史作业需管理员明确分配归属"})
			return
		}
		c.Next()
	}
}
