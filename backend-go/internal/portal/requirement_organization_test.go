package portal

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"grading-gateway/internal/agent/modelclient"
	"grading-gateway/internal/models"
	"grading-gateway/internal/rpa"
)

func requirementFixture() (models.PortalOperation, *HomeworkDetail) {
	payload, _ := json.Marshal(detailTarget{HomeworkName: "编程作业", Target: rpa.PortalTarget{CourseID: "course", HomeworkID: "work"}})
	return models.PortalOperation{ID: "test-detail", PayloadJSON: string(payload)}, &HomeworkDetail{
		CourseID: "course", HomeworkID: "work", Title: "编程作业", Complete: true,
		Sources: []DetailSource{{Kind: "text", Text: "重新编辑\n每隔500ms输出下载状态。命令 multidownload <configfile> <outputdir>。提交学号+姓名.zip。\n答案：\n难度：0.8"}},
	}
}

func TestRequirementsSendOriginalImageToConfiguredModel(t *testing.T) {
	var original bytes.Buffer
	_ = png.Encode(&original, image.NewRGBA(image.Rect(0, 0, 3, 3)))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var body struct {
			Model          string
			ResponseFormat map[string]string `json:"response_format"`
			Messages       []struct {
				Role    string
				Content json.RawMessage
			}
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body.Model != "existing-model" || body.ResponseFormat["type"] != "json_object" || req.Header.Get("Authorization") != "Bearer test-key" {
			t.Error("did not reuse model configuration")
		}
		var blocks []struct {
			Type     string
			Text     string
			ImageURL struct{ URL, Detail string } `json:"image_url"`
		}
		_ = json.Unmarshal(body.Messages[1].Content, &blocks)
		if body.Messages[1].Role != "user" || len(blocks) != 3 || blocks[2].Type != "image_url" {
			t.Error("original image was not supplied to model")
		}
		imageURL := blocks[2].ImageURL
		if imageURL.URL != "data:image/png;base64,"+base64.StdEncoding.EncodeToString(original.Bytes()) || imageURL.Detail != "original" {
			t.Error("image bytes or original detail changed")
		}
		if !strings.Contains(blocks[0].Text, "500ms") || !strings.Contains(blocks[0].Text, "[原图 2.1]") {
			t.Error("text or figure association lost")
		}
		result := map[string]any{"requirements": "## 功能要求\n每隔500ms输出下载状态。\n\n```sh\nmultidownload <configfile> <outputdir>\n```\n\n## 提交要求\n提交学号+姓名.zip。", "source_ids": []string{"1", "2"}, "images": []map[string]string{{"id": "2.1", "text": "使用 TypeScript 和 Node.js，并发下载多个文件。"}}, "uncertainties": []string{}}
		content, _ := json.Marshal(result)
		_ = json.NewEncoder(w).Encode(map[string]any{"model": "returned-vision-model", "choices": []any{map[string]any{"finish_reason": "stop", "message": map[string]string{"content": string(content)}}}})
	}))
	defer server.Close()
	t.Setenv("AGENT_MODEL", "existing-model")
	t.Setenv("AGENT_MODEL_URL", server.URL)
	t.Setenv("DEEPSEEK_API_KEY", "test-key")
	root := t.TempDir()
	t.Setenv("RPA_DOWNLOAD_DIR", root)
	op, raw := requirementFixture()
	dir := filepath.Join(root, op.ID, "requirements")
	_ = os.MkdirAll(dir, 0700)
	path := filepath.Join(dir, "question.png")
	_ = os.WriteFile(path, original.Bytes(), 0600)
	digest := sha256.Sum256(original.Bytes())
	raw.Sources = append(raw.Sources, DetailSource{Kind: "image", Name: "题目.png", Path: path, Size: int64(original.Len()), SHA256: hex.EncodeToString(digest[:])})
	result, err := resolveDetail(context.Background(), op, raw)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Complete || !result.Processing.UsedVision || result.Processing.Model != "returned-vision-model" {
		t.Fatalf("vision result not ready: %+v", result)
	}
	for _, literal := range []string{"500ms", "multidownload <configfile> <outputdir>", "学号+姓名.zip"} {
		if !strings.Contains(result.Question, literal) {
			t.Fatalf("lost requirement %s", literal)
		}
	}
	if strings.Contains(result.Question, "重新编辑") || !strings.Contains(result.RawQuestion, "重新编辑") || !strings.Contains(result.RawQuestion, "TypeScript") || result.Sources[1].Path != path {
		t.Fatal("organized and original material were not kept separately")
	}
	if raw.Sources[1].Text != "" || raw.Sources[0].ID != "" {
		t.Fatal("mutated acquired browser record")
	}
}

func TestUnusableOrganizationNeverBecomesAutomaticRequirements(t *testing.T) {
	for _, tc := range []struct {
		name, content, finish string
		failure               bool
	}{
		{"invalid-json", "not json", "stop", false},
		{"truncated", `{"requirements":"部分","source_ids":["1"],"images":[],"uncertainties":[]}`, "length", false},
		{"missing-source", `{"requirements":"部分","source_ids":[],"images":[],"uncertainties":[]}`, "stop", false},
		{"unexpected-image", `{"requirements":"部分","source_ids":["1"],"images":[{"id":"99","text":"其他图"}],"uncertainties":[]}`, "stop", false},
		{"uncertain", `{"requirements":"每隔500ms输出。","source_ids":["1"],"images":[],"uncertainties":["提交文件名无法确认"]}`, "stop", false},
		{"model-error", "", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			op, raw := requirementFixture()
			result, err := resolveDetailWithModel(context.Background(), op, raw, func(context.Context, []map[string]interface{}) (*modelclient.LLMResponse, error) {
				if tc.failure {
					return nil, errors.New("HTTP 400")
				}
				return &modelclient.LLMResponse{Content: tc.content, FinishReason: tc.finish}, nil
			})
			if err != nil || result.Complete || len(result.Warnings) == 0 || !strings.Contains(result.RawQuestion, "500ms") {
				t.Fatalf("incomplete result accepted or material lost: %v %+v", err, result)
			}
		})
	}
}

func TestRequirementCancellationReachesModel(t *testing.T) {
	op, raw := requirementFixture()
	ctx, cancel := context.WithCancel(context.Background())
	_, err := resolveDetailWithModel(ctx, op, raw, func(ctx context.Context, _ []map[string]interface{}) (*modelclient.LLMResponse, error) {
		cancel()
		return nil, ctx.Err()
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatal("cancelled model processing continued")
	}
}
