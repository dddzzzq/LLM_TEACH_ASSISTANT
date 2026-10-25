package handlers

import (
	"context"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"grading-gateway/internal/auth"
	"grading-gateway/internal/rpa"
)

// Browser WebSockets cannot set an Authorization header. Authenticate their
// first message, with a short deadline; never put credentials in the URL.
func RPABrowserStream(c *gin.Context) {
	upgrader := websocket.Upgrader{HandshakeTimeout: 5 * time.Second, CheckOrigin: browserStreamOrigin}
	client, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		return
	}
	defer client.Close()
	client.SetReadLimit(16 * 1024)
	client.SetReadDeadline(time.Now().Add(5 * time.Second))
	var login struct {
		Type  string `json:"type"`
		Token string `json:"token"`
	}
	if client.ReadJSON(&login) != nil || login.Type != "auth" {
		streamReject(client, "请先登录", 4401)
		return
	}
	claims, err := auth.ParseAccessToken(login.Token, nil)
	login.Token = ""
	if err != nil || claims.ExpiresAt == nil {
		streamReject(client, "登录已过期，请重新连接", 4401)
		return
	}
	if claims.Role != "teacher" && claims.Role != "admin" {
		streamReject(client, "当前角色无权访问浏览器", 4403)
		return
	}
	job, err := rpa.Owned(claims.UserID, c.Param("id"))
	if err != nil {
		streamReject(client, "任务不存在或无权访问", 4404)
		return
	}
	ctx, cancel := context.WithDeadline(c.Request.Context(), claims.ExpiresAt.Time)
	defer cancel()
	worker, err := rpa.OpenBrowserStream(ctx, job.ID)
	if err != nil {
		streamReject(client, err.Error(), 4503)
		return
	}
	defer worker.Close()
	// An idle connection ends within 45 seconds if heartbeats stop. Expiry is
	// enforced independently, so heartbeats cannot extend an access token.
	stop := context.AfterFunc(ctx, func() { client.Close(); worker.Close() })
	defer stop()
	worker.SetReadLimit(8 * 1024 * 1024)
	done := make(chan struct{}, 2)
	relay := func(source, target *websocket.Conn) {
		defer func() { done <- struct{}{} }()
		for {
			source.SetReadDeadline(time.Now().Add(45 * time.Second))
			kind, data, err := source.ReadMessage()
			if err != nil {
				return
			}
			if kind != websocket.TextMessage {
				return
			}
			target.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if target.WriteMessage(kind, data) != nil {
				return
			}
		}
	}
	go relay(client, worker)
	go relay(worker, client)
	<-done
	client.Close()
	worker.Close()
	<-done
}

func streamReject(conn *websocket.Conn, message string, code int) {
	conn.SetWriteDeadline(time.Now().Add(time.Second))
	_ = conn.WriteJSON(map[string]any{"type": "error", "message": message})
	_ = conn.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(code, ""), time.Now().Add(time.Second))
}

func browserStreamOrigin(request *http.Request) bool {
	origin := request.Header.Get("Origin")
	if origin == "" {
		return true
	} // Native clients must still authenticate.
	parsed, err := url.Parse(origin)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return false
	}
	if strings.EqualFold(parsed.Host, request.Host) {
		return true
	}
	for _, allowed := range strings.Split(os.Getenv("RPA_BROWSER_ALLOWED_ORIGINS"), ",") {
		if origin == strings.TrimSpace(allowed) {
			return true
		}
	}
	return false
}
