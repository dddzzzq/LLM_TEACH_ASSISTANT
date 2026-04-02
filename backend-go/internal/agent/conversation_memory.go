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
	db          *gorm.DB      // MySQL 数据库连接（用于异步持久化）
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

// AddMessage 添加一条消息到指定用户的会话记忆（Redis + 异步持久化到 MySQL）
func (rmm *RedisMemoryManager) AddMessage(userID uint, sessionID, role, content string) error {
	ctx := context.Background()
	redisKey := rmm.GetMemoryKey(userID, sessionID)

	// 创建消息对象
	message := Message{
		Role:      role,
		Content:   content,
		Timestamp: time.Now(),
	}

	// 序列化消息为 JSON
	messageJSON, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w", err)
	}

	// 使用事务确保 Redis 操作原子性
	err = database.RetryRedisOperation(func() error {
		pipe := database.GetRedisClient().TxPipeline()

		// 将消息添加到列表头部
		pipe.LPush(ctx, redisKey, messageJSON)

		// 修剪列表，只保留最近的 maxMessages 条消息
		pipe.LTrim(ctx, redisKey, 0, int64(rmm.maxMessages-1))

		// 设置键的过期时间
		pipe.Expire(ctx, redisKey, rmm.messageTTL)

		_, err := pipe.Exec(ctx)
		return err
	}, 3, 100*time.Millisecond)

	if err != nil {
		return fmt.Errorf("failed to add message to Redis: %w", err)
	}

	// 异步持久化到 MySQL
	go rmm.persistMessageToMySQL(userID, sessionID, role, content)

	return nil
}

// persistMessageToMySQL 异步将消息持久化到 MySQL 数据库
func (rmm *RedisMemoryManager) persistMessageToMySQL(userID uint, sessionID, role, content string) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("panic in persistMessageToMySQL: %v", r)
		}
	}()

	// 确保会话存在
	err := rmm.ensureChatSessionExists(userID, sessionID)
	if err != nil {
		log.Printf("failed to ensure chat session exists: %v", err)
		return
	}

	// 创建聊天消息记录
	chatMessage := models.ChatMessage{
		SessionID: sessionID,
		Role:      role,
		Content:   content,
	}

	// 异步写入数据库
	if err := rmm.db.Create(&chatMessage).Error; err != nil {
		log.Printf("failed to persist message to MySQL: %v", err)
	}
}

// ensureChatSessionExists 确保聊天会话在 MySQL 中存在
func (rmm *RedisMemoryManager) ensureChatSessionExists(userID uint, sessionID string) error {
	var chatSession models.ChatSession
	result := rmm.db.Where("id = ?", sessionID).First(&chatSession)

	if result.Error == gorm.ErrRecordNotFound {
		// 创建新会话
		chatSession = models.ChatSession{
			ID:     sessionID,
			UserID: userID,
			Title:  fmt.Sprintf("会话 %s", time.Now().Format("2006-01-02 15:04")),
		}
		return rmm.db.Create(&chatSession).Error
	}

	return result.Error
}

// GetRecentMessages 获取指定用户会话的最近消息（从 Redis）
func (rmm *RedisMemoryManager) GetRecentMessages(userID uint, sessionID string) ([]Message, error) {
	ctx := context.Background()
	redisKey := rmm.GetMemoryKey(userID, sessionID)

	// 获取列表中的所有消息
	messagesJSON, err := database.ListRange(ctx, redisKey, 0, -1)
	if err != nil {
		// 如果键不存在，返回空列表
		if err.Error() == "redis: nil" {
			return []Message{}, nil
		}
		return nil, fmt.Errorf("failed to get messages from Redis: %w", err)
	}

	// 反序列化消息
	var messages []Message
	for _, msgJSON := range messagesJSON {
		var msg Message
		if err := json.Unmarshal([]byte(msgJSON), &msg); err != nil {
			log.Printf("failed to unmarshal message: %v", err)
			continue
		}
		messages = append(messages, msg)
	}

	// 返回消息（Redis 列表是先进后出，需要反转顺序）
	return rmm.reverseMessages(messages), nil
}

// GetRecentMessagesWithLimit 获取指定数量的最近消息
func (rmm *RedisMemoryManager) GetRecentMessagesWithLimit(userID uint, sessionID string, limit int) ([]Message, error) {
	if limit <= 0 {
		return rmm.GetRecentMessages(userID, sessionID)
	}

	ctx := context.Background()
	redisKey := rmm.GetMemoryKey(userID, sessionID)

	// 获取最新的 limit 条消息
	startIdx := int64(0)
	stopIdx := int64(limit - 1)
	messagesJSON, err := database.ListRange(ctx, redisKey, startIdx, stopIdx)
	if err != nil {
		// 如果键不存在，返回空列表
		if err.Error() == "redis: nil" {
			return []Message{}, nil
		}
		return nil, fmt.Errorf("failed to get messages from Redis: %w", err)
	}

	// 反序列化消息
	var messages []Message
	for _, msgJSON := range messagesJSON {
		var msg Message
		if err := json.Unmarshal([]byte(msgJSON), &msg); err != nil {
			log.Printf("failed to unmarshal message: %v", err)
			continue
		}
		messages = append(messages, msg)
	}

	// 返回消息（Redis 列表是先进后出，需要反转顺序）
	return rmm.reverseMessages(messages), nil
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
	ctx := context.Background()
	redisKey := rmm.GetMemoryKey(userID, sessionID)

	return database.DeleteKey(ctx, redisKey)
}

// GetMessageCount 获取指定会话的消息数量
func (rmm *RedisMemoryManager) GetMessageCount(userID uint, sessionID string) (int64, error) {
	ctx := context.Background()
	redisKey := rmm.GetMemoryKey(userID, sessionID)

	result, err := database.GetRedisClient().LLen(ctx, redisKey).Result()
	if err != nil {
		if err.Error() == "redis: nil" {
			return 0, nil
		}
		return 0, fmt.Errorf("failed to get message count: %w", err)
	}

	return result, nil
}

// SessionExists 检查会话是否存在
func (rmm *RedisMemoryManager) SessionExists(userID uint, sessionID string) (bool, error) {
	ctx := context.Background()
	redisKey := rmm.GetMemoryKey(userID, sessionID)

	return database.KeyExists(ctx, redisKey)
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
	if err := rmm.db.Where("session_id = ?", sessionID).Order("created_at ASC").Find(&chatMessages).Error; err != nil {
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
	messages, err := rmm.GetSessionMessagesFromDB(sessionID)
	if err != nil {
		return err
	}

	// 清空现有 Redis 数据
	if err := rmm.ClearSession(userID, sessionID); err != nil {
		return err
	}

	// 将消息批量添加到 Redis
	for _, msg := range messages {
		if err := rmm.AddMessage(userID, sessionID, msg.Role, msg.Content); err != nil {
			return err
		}
	}

	return nil
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
	onceRedis.Do(func() {
		// 需要数据库连接，这里延迟初始化
		// 实际使用时应通过依赖注入传递 db 连接
		log.Println("Warning: RedisMemoryManager not initialized. Call InitRedisMemoryManager first.")
	})
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
	onceRedis.Do(func() {
		log.Println("Warning: MemoryManager is deprecated. Please use RedisMemoryManager instead.")
	})
	return &MemoryManager{GetGlobalRedisMemoryManager()}
}
