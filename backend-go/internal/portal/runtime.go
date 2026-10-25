package portal

import (
	"context"
	"gorm.io/gorm"
	"grading-gateway/internal/models"
	"grading-gateway/internal/rpa"
	"log"
	"sync"
)

var runtimeMu sync.Mutex
var runtimeRunner *Runner

func Runtime(db *gorm.DB) *Runner {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()
	if runtimeRunner != nil && runtimeRunner.Service.DB == db {
		return runtimeRunner
	}
	s := New(db)
	return NewRunner(s, NewBroker(s))
}
func Start(ctx context.Context, db *gorm.DB) {
	runtimeMu.Lock()
	s := New(db)
	r := NewRunner(s, NewBroker(s))
	runtimeRunner = r
	runtimeMu.Unlock()
	if err := r.Broker.Listen(ctx); err != nil {
		log.Printf("教务凭据通道未启动（%T）；其他批改入口仍可使用", err)
	}
	rpa.PortalSession = func(ctx context.Context, owner uint, connection, job string) (map[string]any, error) {
		var task models.RPAJob
		if err := db.WithContext(ctx).Where("id = ? AND user_id = ? AND portal_connection_id = ?", job, owner, connection).First(&task).Error; err != nil {
			return nil, ErrNotFound
		}
		if task.TaskType == "AUTH_BOOTSTRAP" || task.TaskType == "CATALOG_SYNC" {
			return r.Session(ctx, owner, connection, job, "")
		}
		return r.Session(ctx, owner, connection, "", job)
	}
	go r.Run(ctx)
}
