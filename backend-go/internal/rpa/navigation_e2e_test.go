package rpa

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/google/uuid"
	"grading-gateway/internal/agent/catalog"
	"grading-gateway/internal/agent/modelclient"
	"grading-gateway/internal/browseragent"
)

func TestEinoThroughPythonWorkerDownloadsVerifiedArchive(t *testing.T) {
	python := os.Getenv("RPA_BROWSER_PYTHON")
	if python == "" {
		t.Skip("set RPA_BROWSER_PYTHON to run real Go/Eino/Python/Chromium integration")
	}
	root, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()
	dir := t.TempDir()
	token := filepath.Join(dir, "token")
	if err = os.WriteFile(token, []byte("e2e-fixture-token"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("RPA_CONTROL_TOKEN_FILE", token)
	t.Setenv("RPA_CONTROL_URL", fmt.Sprintf("http://127.0.0.1:%d", port))
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, python, "tests/fixtures/browser_worker_server.py")
	cmd.Dir = filepath.Join(root, "ai_engine_python")
	cmd.Env = append(os.Environ(), "PYTHONPATH="+cmd.Dir, "RPA_FIXTURE_PORT="+strconv.Itoa(port), "RUNTIME_DIR="+dir, "RPA_DOWNLOAD_DIR="+filepath.Join(dir, "downloads"), "RPA_PROXY_URL=", "DEEPSEEK_API_KEY=", "PAGE_AGENT_API_KEY=")
	log, err := os.Create(filepath.Join(dir, "worker.log"))
	if err != nil {
		t.Fatal(err)
	}
	defer log.Close()
	cmd.Stdout = log
	cmd.Stderr = log
	if err = cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		cmd.Process.Signal(os.Interrupt)
		cmd.Wait()
		if t.Failed() {
			b, _ := os.ReadFile(log.Name())
			t.Log(string(b))
		}
	}()
	for {
		_, err = WorkerRequest(ctx, "GET", "/health", nil)
		if err == nil {
			break
		}
		if ctx.Err() != nil {
			t.Fatal("Worker startup timeout")
		}
		time.Sleep(100 * time.Millisecond)
	}
	var caps struct {
		Schema json.RawMessage `json:"action_schema"`
	}
	b, err := WorkerRequest(ctx, "GET", "/capabilities", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &caps); err != nil {
		t.Fatal(err)
	}
	skills, err := catalog.Default("portal-browser")
	if err != nil {
		t.Fatal(err)
	}
	snapshot := NavigationSnapshot{Skill: skills["fetch-homework"], ActionSchema: caps.Schema, Protocol: "browser.v1"}
	id := uuid.NewString()
	_, err = WorkerRequest(ctx, "POST", "/tasks/"+id, map[string]any{"target": Target{CourseName: "操作系统", AssignmentName: "第一次作业"}})
	if err != nil {
		t.Fatal(err)
	}
	decisions := 0
	decide := func(ctx context.Context, input json.RawMessage, doc catalog.Document, contract json.RawMessage) (json.RawMessage, error) {
		decisions++
		round := 0
		return browseragent.Decide(ctx, input, doc, contract, func(context.Context, []map[string]interface{}, []map[string]interface{}) (*modelclient.LLMResponse, error) {
			round++
			name, args := "load_skill", `{"skill":"fetch-homework"}`
			if round > 1 {
				var view struct {
					Stage       string `json:"stage"`
					Observation struct {
						CanExport bool `json:"can_open_export"`
						Elements  []struct {
							Name string `json:"name"`
							Ref  string `json:"ref"`
						} `json:"elements"`
					} `json:"observation"`
				}
				if err := json.Unmarshal(input, &view); err != nil {
					return nil, err
				}
				if view.Observation.CanExport {
					name, args = "open_export_settings", "{}"
				} else {
					wanted := "操作系统"
					if view.Stage != "LOCATE_COURSE" {
						wanted = "批阅"
					}
					for _, e := range view.Observation.Elements {
						if e.Name == wanted {
							name = "click"
							v, _ := json.Marshal(map[string]string{"element_ref": e.Ref})
							args = string(v)
							break
						}
					}
					if name == "load_skill" {
						return nil, fmt.Errorf("fixture target missing")
					}
				}
			}
			return &modelclient.LLMResponse{ToolCalls: []modelclient.ToolCall{{ID: uuid.NewString(), Type: "function", Function: modelclient.FunctionCall{Name: name, Arguments: args}}}}, nil
		})
	}
	journal := &memoryJournal{}
	for {
		if ctx.Err() != nil {
			t.Fatal("download timed out")
		}
		raw, err := WorkerRequest(ctx, "GET", "/tasks/"+id, nil)
		if err != nil {
			t.Fatal(err)
		}
		var state struct {
			Status string                                  `json:"status"`
			Reason string                                  `json:"waiting_reason"`
			Files  []struct{ Status, Path, SHA256 string } `json:"files"`
		}
		if err = json.Unmarshal(raw, &state); err != nil {
			t.Fatal(err)
		}
		if state.Status == "SUCCEEDED" {
			if decisions < 3 || len(state.Files) != 1 || state.Files[0].Status != "VERIFIED" || len(state.Files[0].SHA256) != 64 {
				t.Fatalf("invalid completion %s", raw)
			}
			if _, err = os.Stat(state.Files[0].Path); err != nil {
				t.Fatal(err)
			}
			count, _ := WorkerRequest(ctx, "GET", "/fixture/submissions", nil)
			if string(count) != `{"count":1}` {
				t.Fatalf("duplicate export: %s", count)
			}
			return
		}
		if state.Status == "FAILED" || state.Status == "PARTIAL_SUCCESS" {
			t.Fatalf("download failed: %s", raw)
		}
		if state.Status == "WAITING_USER" {
			switch state.Reason {
			case "LOGIN":
				_, err = WorkerRequest(ctx, "POST", "/fixture/"+id+"/login", nil)
			case "CLASS_SCOPE":
				_, err = WorkerRequest(ctx, "POST", "/tasks/"+id+"/class-scope", map[string]any{"mode": "selected", "class_ids": []string{"class_0"}})
			default:
				t.Fatalf("unexpected human handoff: %s", raw)
			}
			if err != nil {
				t.Fatal(err)
			}
			if _, err = WorkerRequest(ctx, "POST", "/tasks/"+id+"/resume", nil); err != nil {
				t.Fatal(err)
			}
		} else {
			if _, err = drivePage(ctx, id, snapshot, WorkerRequest, decide, journal); err != nil {
				t.Fatal(err)
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
}
