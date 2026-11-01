package playwright

import (
	"sync"
	"testing"
)

func TestRequestPagePublication(t *testing.T) {
	frame := &frameImpl{}
	frame.channel = &channel{object: frame}
	request := &requestImpl{channelOwner: channelOwner{initializer: map[string]any{"frame": frame.channel}}}
	if request.safePage() != nil || request.Frame() != nil {
		t.Fatal("a request must not expose an uninitialized page")
	}
	start := make(chan struct{})
	var readers sync.WaitGroup
	for i := 0; i < 8; i++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			<-start
			for j := 0; j < 10000; j++ {
				page := request.safePage()
				if page != nil && page.closedOrCrashed == nil {
					t.Error("request observed a partially initialized page")
					return
				}
				if request.Frame() != nil && frame.Page() != page && page != nil {
					t.Error("request frame refers to a different page")
					return
				}
			}
		}()
	}
	close(start)
	page := &pageImpl{}
	page.closedOrCrashed = make(chan error, 1)
	frame.page.Store(page)
	readers.Wait()
	if request.safePage() != page || request.Frame() != frame {
		t.Fatal("request did not expose the published page")
	}
}
