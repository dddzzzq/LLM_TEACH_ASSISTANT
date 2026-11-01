package tools

import (
	"context"
	"errors"
	"os"
	"strconv"
	"time"

	"grading-gateway/internal/grpcclient"
	"grading-gateway/pb"
)

var ocrSlots = make(chan struct{}, ocrSetting("OCR_WORKERS", 2, 8))
var ocrTimeout = time.Duration(ocrSetting("OCR_TASK_TIMEOUT_SECONDS", 120, 600)) * time.Second

func ocrSetting(key string, fallback, maximum int) int {
	value, err := strconv.Atoi(os.Getenv(key))
	if err != nil || value < 1 || value > maximum {
		return fallback
	}
	return value
}

func extractOCR(ctx context.Context, filename string, data []byte) (*pb.ExtractResponse, error) {
	return requestOCR(ctx, grpcclient.Client, ocrSlots, ocrTimeout, filename, data)
}

func requestOCR(ctx context.Context, client pb.ComputeServiceClient, slots chan struct{}, timeout time.Duration, filename string, data []byte) (*pb.ExtractResponse, error) {
	if client == nil {
		return nil, errors.New("OCR unavailable")
	}
	// Backpressure happens before the RPC deadline starts. Twenty document
	// readers must not enqueue twenty 60-second RPCs into two native predictors.
	select {
	case slots <- struct{}{}:
		defer func() { <-slots }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	rpcCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return client.ExtractText(rpcCtx, &pb.ExtractRequest{Filename: filename, FileContent: data})
}
