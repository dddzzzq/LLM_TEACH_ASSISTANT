package models

import "time"

// UserRole 用户角色枚举
type UserRole string

const (
	RoleStudent UserRole = "student"
	RoleTeacher UserRole = "teacher"
	RoleAdmin   UserRole = "admin"
)

// User 用户表
type User struct {
	ID           uint      `gorm:"primaryKey"`                      // 主键
	Username     string    `gorm:"size:64;uniqueIndex;not null"`    // 学号/工号，唯一索引
	PasswordHash string    `gorm:"type:varchar(255);not null"`      // bcrypt 加密后的密码
	Role         UserRole  `gorm:"type:varchar(20);not null;index"` // 角色: student/teacher/admin
	Name         string    `gorm:"size:128;not null"`               // 姓名
	CreatedAt    time.Time `gorm:"autoCreateTime"`                  // 创建时间
	UpdatedAt    time.Time `gorm:"autoUpdateTime"`                  // 更新时间
}
