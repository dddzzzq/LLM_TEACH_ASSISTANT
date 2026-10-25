package portal

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"grading-gateway/internal/models"
)

type credentialTicket struct {
	ConnectionID, OperationID, JobID string
	OwnerID                          uint
	Version, Generation              uint64
	Until                            time.Time
}
type Broker struct {
	Service *Service
	mu      sync.Mutex
	tickets map[[32]byte]credentialTicket
	ready   bool
}

func NewBroker(s *Service) *Broker {
	return &Broker{Service: s, tickets: make(map[[32]byte]credentialTicket)}
}
func BrokerSocket() string {
	if path := os.Getenv("PORTAL_CREDENTIAL_SOCKET"); path != "" {
		return path
	}
	return filepath.Join(RuntimeRoot(), "portal-credentials.sock")
}
func (b *Broker) Issue(c models.PortalConnection, opID, jobID string) (string, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.ready {
		return "", ErrUnavailable
	}
	for key, value := range b.tickets {
		if time.Now().After(value.Until) {
			delete(b.tickets, key)
		}
	}
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", err
	}
	token := hex.EncodeToString(raw)
	b.tickets[sha256.Sum256([]byte(token))] = credentialTicket{c.ID, opID, jobID, c.OwnerID, c.CredentialVersion, c.ProfileGeneration, time.Now().Add(2 * time.Minute)}
	return token, nil
}
func (b *Broker) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if r.Method != "POST" || r.URL.Path != "/credentials" {
		http.NotFound(w, r)
		return
	}
	key := sha256.Sum256([]byte(r.Header.Get("X-Portal-Ticket")))
	b.mu.Lock()
	ticket, ok := b.tickets[key]
	delete(b.tickets, key)
	b.mu.Unlock()
	if !ok || time.Now().After(ticket.Until) {
		http.Error(w, "expired ticket", 401)
		return
	}
	c, err := b.Service.connection(r.Context(), ticket.OwnerID, false)
	if err != nil || c.ID != ticket.ConnectionID || c.CredentialVersion != ticket.Version || c.ProfileGeneration != ticket.Generation || c.Status == models.PortalConnectionDisabled {
		http.Error(w, "revoked ticket", 403)
		return
	}
	if ticket.OperationID != "" {
		op, err := b.Service.OwnedOperation(r.Context(), ticket.OwnerID, ticket.OperationID)
		if err != nil || terminal(op.Status) || c.ActiveOperationID != op.ID {
			http.Error(w, "revoked operation", 403)
			return
		}
	} else {
		if c.Status != models.PortalConnectionReady || c.ActiveOperationID != "" {
			http.Error(w, "authentication required", 403)
			return
		}
		if !b.Service.downloadAllowed(r.Context(), c, ticket.JobID) {
			http.Error(w, "revoked job", 403)
			return
		}
	}
	value, err := b.Service.credentials(r.Context(), c)
	if err != nil {
		http.Error(w, "credentials unavailable", 503)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

// The socket is local, mode 0600, and accepts only short-lived single-use tickets.
func (b *Broker) Listen(ctx context.Context) error {
	path := BrokerSocket()
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	if info, err := os.Lstat(path); err == nil {
		if info.Mode()&os.ModeSocket == 0 {
			return errors.New("credential socket path is not a socket")
		}
		if conn, err := net.DialTimeout("unix", path, time.Second); err == nil {
			conn.Close()
			return errors.New("credential broker already running")
		}
		if err = os.Remove(path); err != nil {
			return err
		}
	}
	listener, err := net.Listen("unix", path)
	if err != nil {
		return err
	}
	if err = os.Chmod(path, 0600); err != nil {
		listener.Close()
		return err
	}
	b.mu.Lock()
	b.ready = true
	b.mu.Unlock()
	server := &http.Server{Handler: b, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 5 * time.Second, WriteTimeout: 5 * time.Second}
	go func() { <-ctx.Done(); _ = server.Close() }()
	go func() { _ = server.Serve(listener); b.mu.Lock(); b.ready = false; b.mu.Unlock() }()
	return nil
}
