package models

import "time"

type PortalConnectionStatus string

const (
	PortalConnectionInitializing PortalConnectionStatus = "INITIALIZING"
	PortalConnectionReady        PortalConnectionStatus = "READY"
	PortalConnectionReauth       PortalConnectionStatus = "REAUTH_REQUIRED"
	PortalConnectionDisabled     PortalConnectionStatus = "DISABLED"
)

// PortalConnection 保存教师与教务平台之间的连接配置。
type PortalConnection struct {
	ID string `json:"id" gorm:"type:char(36);primaryKey"`

	OwnerID uint `json:"-" gorm:"not null;uniqueIndex:idx_portal_owner_provider,priority:1"`

	Provider string `json:"provider" gorm:"size:64;not null;uniqueIndex:idx_portal_owner_provider,priority:2"`

	AccountMask string `json:"account_mask" gorm:"size:128;not null"`

	CredentialCiphertext []byte `json:"-" gorm:"type:blob;not null"`
	CredentialNonce      []byte `json:"-" gorm:"type:varbinary(12);not null"`
	CredentialKeyID      string `json:"-" gorm:"size:64;not null"`
	CredentialVersion    uint64 `json:"-" gorm:"not null;default:1"`

	ProfileGeneration uint64     `json:"-" gorm:"not null;default:1"`
	ActiveSnapshotID  string     `json:"active_snapshot_id" gorm:"size:36"`
	ImportsPaused     bool       `json:"imports_paused" gorm:"not null;default:false"`
	ActiveOperationID string     `json:"-" gorm:"size:36"`
	LastSyncedAt      *time.Time `json:"last_synced_at"`

	TrustDevice bool `json:"trust_device" gorm:"not null;default:false"`

	Status PortalConnectionStatus `json:"status" gorm:"size:32;not null;default:REAUTH_REQUIRED"`

	VerifiedSubject string `json:"-" gorm:"size:255"`

	LastAuthenticatedAt *time.Time `json:"last_authenticated_at"`

	CreatedAt time.Time `json:"created_at" gorm:"autoCreateTime"`
	UpdatedAt time.Time `json:"updated_at" gorm:"autoUpdateTime"`
}
