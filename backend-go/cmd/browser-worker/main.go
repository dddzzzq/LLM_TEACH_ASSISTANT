package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	pw "github.com/mxschmitt/playwright-go"
	"grading-gateway/internal/browserworker"
)

func main() {
	install := flag.Bool("install", false, "install the pinned Playwright driver and Chromium")
	flag.Parse()
	if *install {
		if e := pw.Install(&pw.RunOptions{Browsers: []string{"chromium"}}); e != nil {
			log.Fatal("Go Playwright 安装失败: ", e)
		}
		return
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	manager, e := browserworker.New(ctx, browserworker.DefaultConfig())
	if e != nil {
		log.Fatal(e)
	}
	defer manager.Close()
	port := os.Getenv("RPA_CONTROL_PORT")
	if port == "" {
		port = "8765"
	}
	server := &http.Server{Addr: "127.0.0.1:" + port, Handler: manager.Handler(), ReadHeaderTimeout: 5 * time.Second, IdleTimeout: 45 * time.Second, MaxHeaderBytes: 16 * 1024}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdown)
	}()
	log.Printf("Go Playwright Worker 监听 %s", server.Addr)
	if e = server.ListenAndServe(); e != nil && !errors.Is(e, http.ErrServerClosed) {
		log.Print("浏览器 Worker 监听失败")
		stop()
	}
}
