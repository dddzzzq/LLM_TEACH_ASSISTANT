package models

import "time"

// ChatSession 聊天会话表
type ChatSession struct {
	ID        string    `json:"id" gorm:"type:char(36);primaryKey"`        // UUID 主键
	UserID    uint      `json:"user_id" gorm:"not null;index"`             // 关联用户ID
	Title     string    `json:"title" gorm:"type:varchar(255);default:''"` // 会话标题
	CreatedAt time.Time `json:"created_at" gorm:"autoCreateTime"`          // 创建时间
	UpdatedAt time.Time `json:"updated_at" gorm:"autoUpdateTime"`          // 更新时间

	// 关联关系
	User     User          `json:"user,omitempty" gorm:"foreignKey:UserID;constraint:OnDelete:CASCADE;"`        // 关联用户
	Messages []ChatMessage `json:"messages,omitempty" gorm:"foreignKey:SessionID;constraint:OnDelete:CASCADE;"` // 关联消息
}

// ChatMessage 聊天消息表
type ChatMessage struct {
	ID        uint      `json:"id" gorm:"primaryKey"`                                                                              // 主键
	SessionID string    `json:"session_id" gorm:"type:char(36);not null;index;index:idx_chat_messages_session_created,priority:1"` // 关联会话ID
	Role      string    `json:"role" gorm:"type:varchar(20);not null"`                                                             // 角色: user/assistant
	Content   string    `json:"content" gorm:"type:longtext;not null"`                                                             // 消息内容
	CreatedAt time.Time `json:"created_at" gorm:"autoCreateTime;index:idx_chat_messages_session_created,priority:2"`               // 创建时间

	// 关联关系
	Session ChatSession `json:"session,omitempty" gorm:"foreignKey:SessionID;constraint:OnDelete:CASCADE;"` // 关联会话
}
