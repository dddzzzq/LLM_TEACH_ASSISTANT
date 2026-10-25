package execution

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"grading-gateway/internal/models"
)

type DisplayMessage struct {
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
}

// Display rebuilds the chat surface from facts. chat_messages remains a compatibility
// projection; loss of that projection or Redis does not lose an event-backed conversation.
func (s Store) Display(ctx context.Context, user uint, session string, limit int) ([]DisplayMessage, error) {
	if limit <= 0 || limit > 100 {
		limit = 20
	}
	runs, err := s.Runs(ctx, user, session, limit)
	if err != nil {
		return nil, err
	}
	messages := []DisplayMessage{}
	if len(runs) == 0 {
		var legacy []models.ChatMessage
		if err := s.DB.WithContext(ctx).Where("session_id = ?", session).Order("created_at DESC, id DESC").Limit(limit).Find(&legacy).Error; err != nil {
			return nil, err
		}
		for i := len(legacy) - 1; i >= 0; i-- {
			messages = append(messages, DisplayMessage{Role: legacy[i].Role, Content: legacy[i].Content, Timestamp: legacy[i].CreatedAt})
		}
		return messages, nil
	}
	for i := len(runs) - 1; i >= 0; i-- {
		var events []models.AgentEvent
		if err := s.DB.WithContext(ctx).Where("run_id = ? AND sequence <= ? AND type IN ?", runs[i].ID, runs[i].Sequence, []string{"history.imported", "user.message", "run.finished", "run.interrupted"}).Order("sequence ASC").Find(&events).Error; err != nil {
			return nil, err
		}
		for _, event := range events {
			if event.Version != Version {
				return nil, fmt.Errorf("unsupported execution event version %d", event.Version)
			}
			switch event.Type {
			case "history.imported":
				var rows []Message
				if err := json.Unmarshal(event.Payload, &rows); err != nil {
					return nil, err
				}
				for _, row := range rows {
					at := event.CreatedAt
					if row.Timestamp != nil {
						at = *row.Timestamp
					}
					messages = append(messages, DisplayMessage{Role: row.Role, Content: row.Content, Timestamp: at})
				}
			case "user.message":
				var row Message
				if err := json.Unmarshal(event.Payload, &row); err != nil {
					return nil, err
				}
				messages = append(messages, DisplayMessage{Role: "user", Content: row.Content, Timestamp: event.CreatedAt})
			default:
				var final Final
				if err := json.Unmarshal(event.Payload, &final); err != nil {
					return nil, err
				}
				if final.Reply != "" {
					messages = append(messages, DisplayMessage{Role: "assistant", Content: final.Reply, Timestamp: event.CreatedAt})
				}
			}
		}
	}
	if len(messages) > limit {
		messages = messages[len(messages)-limit:]
	}
	return messages, nil
}
