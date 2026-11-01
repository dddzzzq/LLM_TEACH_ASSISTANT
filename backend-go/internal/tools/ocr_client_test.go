package tools

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"grading-gateway/pb"
)

type ocrClientFixture struct {
	pb.ComputeServiceClient
	active  atomic.Int32
	maximum atomic.Int32
	calls   atomic.Int32
	delay   time.Duration
}

func (f *ocrClientFixture) ExtractText(ctx context.Context, req *pb.ExtractRequest, options ...grpc.CallOption) (*pb.ExtractResponse, error) {
	f.calls.Add(1)
	n := f.active.Add(1)
	defer f.active.Add(-1)
	for old := f.maximum.Load(); n > old; old = f.maximum.Load() {
		if f.maximum.CompareAndSwap(old, n) {
			break
		}
	}
	select {
	case <-time.After(f.delay):
		return &pb.ExtractResponse{TextContent: "image text"}, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func TestOCRLimitsRPCConcurrencyBeforeStartingDeadline(t *testing.T) {
	fixture := &ocrClientFixture{delay: 70 * time.Millisecond}
	slots := make(chan struct{}, 2)
	var wg sync.WaitGroup
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := requestOCR(context.Background(), fixture, slots, 100*time.Millisecond, "scan.png", nil); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if fixture.maximum.Load() != 2 || fixture.calls.Load() != 6 {
		t.Fatal("OCR parallelism lost or limit exceeded", fixture.maximum.Load(), fixture.calls.Load())
	}
}

func TestOCRCancelledWaitNeverReachesPredictor(t *testing.T) {
	slots := make(chan struct{}, 1)
	slots <- struct{}{}
	fixture := &ocrClientFixture{}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if _, err := requestOCR(ctx, fixture, slots, time.Second, "scan.png", nil); err == nil {
		t.Fatal("cancelled queue wait was accepted")
	}
	if fixture.calls.Load() != 0 {
		t.Fatal("cancelled image reached OCR")
	}
}
