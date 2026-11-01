package browserworker

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	pw "github.com/mxschmitt/playwright-go"
)

type stalledAttachment struct {
	pw.Download
	canceled, ended chan struct{}
	once            sync.Once
}

type attachmentContext struct{ pw.BrowserContext }

func (d *stalledAttachment) Path() (string, error) {
	<-d.canceled
	close(d.ended)
	return "", errors.New("download canceled")
}
func (d *stalledAttachment) Cancel() error {
	d.once.Do(func() { close(d.canceled) })
	return nil
}

func TestQuestionAttachmentDownloadCanBeStopped(t *testing.T) {
	for _, mode := range []string{"timeout", "cancel", "takeover"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			state := object{"status": "RUNNING", "control_mode": "AUTO"}
			if mode == "cancel" {
				cancel()
			}
			if mode == "takeover" {
				state["control_mode"] = "HUMAN"
			}
			task := &Task{ctx: ctx, state: state, context: &attachmentContext{}}
			download := &stalledAttachment{canceled: make(chan struct{}), ended: make(chan struct{})}
			timeout := 10 * time.Second
			if mode == "timeout" {
				timeout = 10 * time.Millisecond
			}
			finished := make(chan error, 1)
			go func() { _, err := task.detailDownloadPath(download, timeout); finished <- err }()
			select {
			case err := <-finished:
				if err == nil {
					t.Fatal("stalled attachment was accepted")
				}
			case <-time.After(time.Second):
				t.Fatal("stalled attachment blocked cancellation or timeout")
			}
			select {
			case <-download.ended:
			case <-time.After(time.Second):
				t.Fatal("download was not canceled")
			}
		})
	}
}
