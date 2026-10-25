package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strconv"
	"strings"
	"sync"
	"time"

	"grading-gateway/internal/database"
	"grading-gateway/internal/execution"
	"grading-gateway/internal/models"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

// Message 表示对话中的一条消息
type Message struct {
	Role      string    `json:"role"`      // "user" 或 "assistant"
	Content   string    `json:"content"`   // 消息内容
	Timestamp time.Time `json:"timestamp"` // 时间戳
}

// RedisMemoryManager 基于 Redis 的对话记忆管理器
type RedisMemoryManager struct {
	maxMessages int           // 每个会话最大消息数
	messageTTL  time.Duration // Redis 键过期时间
	db          *gorm.DB      // MySQL 数据库连接（持久化真源）
}

// NewRedisMemoryManager 创建新的 Redis 对话记忆管理器
func NewRedisMemoryManager(maxMessages int, messageTTL time.Duration, db *gorm.DB) *RedisMemoryManager {
	if maxMessages <= 0 {
		maxMessages = 20 // 默认保留20条消息
	}
	if messageTTL <= 0 {
		messageTTL = 24 * time.Hour // 默认24小时过期
	}

	return &RedisMemoryManager{
		maxMessages: maxMessages,
		messageTTL:  messageTTL,
		db:          db,
	}
}

// GetMemoryKey 生成 Redis 键
func (rmm *RedisMemoryManager) GetMemoryKey(userID uint, sessionID string) string {
	return database.RedisKey(strconv.FormatUint(uint64(userID), 10), sessionID)
}

// AddMessage is a legacy display-message API. Durable writes precede best-effort cache refresh.
// New agent turns use execution.Store so the event and display projection commit together.
func (rmm *RedisMemoryManager) AddMessage(userID uint, sessionID, role, content string) error {
	if err := rmm.ensureChatSessionExists(userID, sessionID); err != nil {
		return err
	}
	if err := rmm.db.Create(&models.ChatMessage{SessionID: sessionID, Role: role, Content: content}).Error; err != nil {
		return err
	}
	if err := rmm.RestoreSessionToRedis(userID, sessionID); err != nil {
		log.Printf("chat cache refresh failed: %v", err)
	}
	return nil
}

func (rmm *RedisMemoryManager) ensureChatSessionExists(userID uint, sessionID string) error {
	var session models.ChatSession
	err := rmm.db.First(&session, "id = ?", sessionID).Error
	if err == gorm.ErrRecordNotFound {
		return rmm.db.Create(&models.ChatSession{ID: sessionID, UserID: userID, Title: "新会话"}).Error
	}
	if err != nil {
		return err
	}
	if session.UserID != userID {
		return gorm.ErrRecordNotFound
	}
	return nil
}

// Read the durable display projection, so Redis loss or a stale cache cannot lose a turn.
func (rmm *RedisMemoryManager) GetRecentMessages(userID uint, sessionID string) ([]Message, error) {
	return rmm.GetRecentMessagesWithLimit(userID, sessionID, rmm.maxMessages)
}
func (rmm *RedisMemoryManager) GetRecentMessagesWithLimit(userID uint, sessionID string, limit int) ([]Message, error) {
	if limit <= 0 {
		limit = rmm.maxMessages
	}
	rows, err := (execution.Store{DB: rmm.db}).Display(context.Background(), userID, sessionID, limit)
	if err != nil {
		return nil, err
	}
	messages := make([]Message, 0, len(rows))
	for _, row := range rows {
		messages = append(messages, Message{Role: row.Role, Content: row.Content, Timestamp: row.Timestamp})
	}
	return messages, nil
}

// GetFormattedHistory 获取格式化的对话历史，用于 LLM 输入
func (rmm *RedisMemoryManager) GetFormattedHistory(userID uint, sessionID string) (string, error) {
	messages, err := rmm.GetRecentMessages(userID, sessionID)
	if err != nil {
		return "", err
	}

	if len(messages) == 0 {
		return "", nil
	}

	var history strings.Builder
	for _, msg := range messages {
		history.WriteString(msg.Role)
		history.WriteString(": ")
		history.WriteString(msg.Content)
		history.WriteString("\n")
	}

	return history.String(), nil
}

// ClearSession 清空指定用户会话的记忆
func (rmm *RedisMemoryManager) ClearSession(userID uint, sessionID string) error {
	client := database.GetRedisClient()
	if client == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	return client.Del(ctx, rmm.GetMemoryKey(userID, sessionID)).Err()
}

func (rmm *RedisMemoryManager) GetMessageCount(userID uint, sessionID string) (int64, error) {
	messages, err := rmm.GetRecentMessages(userID, sessionID)
	return int64(len(messages)), err
}

func (rmm *RedisMemoryManager) SessionExists(userID uint, sessionID string) (bool, error) {
	var count int64
	err := rmm.db.Model(&models.ChatSession{}).Where("id = ? AND user_id = ?", sessionID, userID).Count(&count).Error
	return count > 0, err
}

// CreateNewSession 创建新会话并返回会话ID
func (rmm *RedisMemoryManager) CreateNewSession(userID uint, title string) (string, error) {
	sessionID := uuid.New().String()

	// 在 MySQL 中创建会话记录
	chatSession := models.ChatSession{
		ID:     sessionID,
		UserID: userID,
		Title:  title,
	}

	if err := rmm.db.Create(&chatSession).Error; err != nil {
		return "", fmt.Errorf("failed to create chat session in MySQL: %w", err)
	}

	return sessionID, nil
}

// GetUserSessions 获取用户的所有会话
func (rmm *RedisMemoryManager) GetUserSessions(userID uint) ([]models.ChatSession, error) {
	var sessions []models.ChatSession
	if err := rmm.db.Where("user_id = ?", userID).Order("updated_at DESC").Find(&sessions).Error; err != nil {
		return nil, fmt.Errorf("failed to get user sessions: %w", err)
	}
	return sessions, nil
}

// GetSessionMessagesFromDB 从 MySQL 获取会话的历史消息（用于持久化恢复）
func (rmm *RedisMemoryManager) GetSessionMessagesFromDB(sessionID string) ([]Message, error) {
	var chatMessages []models.ChatMessage
	if err := rmm.db.Where("session_id = ?", sessionID).Order("created_at ASC, id ASC").Find(&chatMessages).Error; err != nil {
		return nil, fmt.Errorf("failed to get session messages from DB: %w", err)
	}

	var messages []Message
	for _, msg := range chatMessages {
		messages = append(messages, Message{
			Role:      msg.Role,
			Content:   msg.Content,
			Timestamp: msg.CreatedAt,
		})
	}

	return messages, nil
}

// RestoreSessionToRedis 将 MySQL 中的会话消息恢复到 Redis
func (rmm *RedisMemoryManager) RestoreSessionToRedis(userID uint, sessionID string) error {
	messages, err := rmm.GetRecentMessages(userID, sessionID)
	if err != nil {
		return err
	}
	client := database.GetRedisClient()
	if client == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	key := rmm.GetMemoryKey(userID, sessionID)
	pipe := client.TxPipeline()
	pipe.Del(ctx, key)
	for _, msg := range messages {
		data, err := json.Marshal(msg)
		if err != nil {
			return err
		}
		pipe.LPush(ctx, key, data)
	}
	pipe.Expire(ctx, key, rmm.messageTTL)
	_, err = pipe.Exec(ctx)
	return err
}

// reverseMessages 反转消息顺序（因为 Redis LPUSH 将最新消息放在列表头部）
func (rmm *RedisMemoryManager) reverseMessages(messages []Message) []Message {
	reversed := make([]Message, len(messages))
	for i, j := 0, len(messages)-1; i < len(messages); i, j = i+1, j-1 {
		reversed[i] = messages[j]
	}
	return reversed
}

// 全局 Redis 记忆管理器实例
var (
	globalRedisMemoryManager *RedisMemoryManager
	onceRedis                sync.Once
)

// GetGlobalRedisMemoryManager 获取全局 Redis 记忆管理器实例（单例模式）
func GetGlobalRedisMemoryManager() *RedisMemoryManager {
	return globalRedisMemoryManager
}

// InitRedisMemoryManager 初始化全局 Redis 记忆管理器
func InitRedisMemoryManager(maxMessages int, messageTTL time.Duration, db *gorm.DB) {
	onceRedis.Do(func() {
		globalRedisMemoryManager = NewRedisMemoryManager(maxMessages, messageTTL, db)
	})
}

// 为了向后兼容，保留旧的内存管理器接口，但标记为已废弃
// MemoryManager 已废弃，请使用 RedisMemoryManager
type MemoryManager struct {
	*RedisMemoryManager
}

// GetGlobalMemoryManager 获取全局记忆管理器实例（单例模式），已废弃，请使用 GetGlobalRedisMemoryManager
func GetGlobalMemoryManager() *MemoryManager {
	return &MemoryManager{GetGlobalRedisMemoryManager()}
}
