package tools

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"grading-gateway/internal/materials"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"
)

const studentEvidenceLimit = 8 << 20

var credentialValue = regexp.MustCompile(`\bsk-[A-Za-z0-9_-]{16,}`)
var downloadRunDirectory = regexp.MustCompile(`^downloads-\d{10,}$`)

type materialEvidence struct {
	ID         string   `json:"id"`
	Path       string   `json:"path"`
	Role       string   `json:"role"`
	SHA256     string   `json:"sha256"`
	Size       int64    `json:"size"`
	Status     string   `json:"status"`
	Text       string   `json:"text,omitempty"`
	Warnings   []string `json:"warnings,omitempty"`
	Redactions int      `json:"redactions,omitempty"`
}
type readingReport struct {
	Status       string                    `json:"status"`
	Warnings     []string                  `json:"warnings"`
	Files        []materialEvidence        `json:"files"`
	ChunksRead   []string                  `json:"chunks_read,omitempty"`
	ReviewStage  string                    `json:"review_stage,omitempty"`
	Archives     []materials.ArchiveSource `json:"submission_archives,omitempty"`
	SkippedFiles []materials.SkippedFile   `json:"skipped_files,omitempty"`
	Summary      json.RawMessage           `json:"summary,omitempty"`
}
type studentMaterials struct {
	Schema       string                    `json:"schema"`
	Files        []materialEvidence        `json:"files"`
	Reading      readingReport             `json:"reading"`
	Archives     []materials.ArchiveSource `json:"submission_archives"`
	SkippedFiles []materials.SkippedFile   `json:"skipped_files,omitempty"`
}

func materialRole(name string) string {
	n := strings.ToLower(filepath.ToSlash(name))
	parts := strings.Split(n, "/")
	extension := filepath.Ext(n)
	// Reports remain evidence even when students place them in an output
	// directory. Directory names alone must not hide an answer or a paper.
	if extension == ".docx" || extension == ".doc" || extension == ".pdf" || extension == ".md" || extension == ".txt" && (strings.Contains(filepath.Base(n), "报告") || strings.Contains(filepath.Base(n), "作答") || strings.Contains(filepath.Base(n), "answer")) {
		return "report"
	}
	for _, p := range parts[:len(parts)-1] {
		if p == "downloads" || p == "download" || p == "output" || p == "outputs" || downloadRunDirectory.MatchString(p) {
			switch extension {
			case ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py", ".go", ".java", ".c", ".cpp", ".h", ".cc", ".hpp", ".cs", ".rs", ".kt", ".swift", ".php", ".rb", ".vue", ".svelte":
				// A directory label cannot prove submitted source is a download artifact.
				continue
			}
			return "artifact"
		}
		if p == "node_modules" || p == "vendor" || p == "third_party" || p == "site-packages" {
			return "dependency"
		}
	}
	base, ext := filepath.Base(n), filepath.Ext(n)
	if ext == ".zip" || ext == ".rar" || ext == ".7z" {
		return "unsupported"
	}
	if strings.HasSuffix(base, "lock.json") || strings.HasSuffix(base, "lock.yaml") || base == "yarn.lock" {
		return "dependency"
	}
	if ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp" || ext == ".gif" {
		return "evidence"
	}
	if ext == ".json" || ext == ".yaml" || ext == ".yml" || ext == ".toml" || ext == ".xml" {
		return "config"
	}
	if ext == ".txt" || ext == ".log" || ext == ".csv" {
		if strings.Contains(base, "log") || strings.Contains(base, "运行") || strings.Contains(base, "测试") || strings.Contains(base, "结果") {
			return "evidence"
		}
		return "report"
	}
	if materials.FileKind(n) == "code" {
		return "code"
	}
	return "unsupported"
}

var imageCacheLocks sync.Map

// Cache only successful OCR, keyed by content and reader version, never filename.
func cachedStudentOCR(ctx context.Context, name string, data []byte) (string, error) {
	digest := sha256.Sum256(data)
	cacheRoot := os.Getenv("RUNTIME_DIR")
	if cacheRoot == "" {
		return requirementOCR(ctx, name, data)
	}
	cacheRoot = filepath.Join(cacheRoot, "material-ocr-cache", "v1")
	path := filepath.Join(cacheRoot, hex.EncodeToString(digest[:])+".json")
	lock, _ := imageCacheLocks.LoadOrStore(path, &sync.Mutex{})
	lock.(*sync.Mutex).Lock()
	defer lock.(*sync.Mutex).Unlock()
	if raw, err := os.ReadFile(path); err == nil && len(raw) < studentEvidenceLimit {
		var text string
		if json.Unmarshal(raw, &text) == nil && text != "" {
			return text, nil
		}
	}
	text, err := requirementOCR(ctx, name, data)
	if err != nil {
		return "", err
	}
	if os.MkdirAll(cacheRoot, 0700) == nil {
		if f, err := os.CreateTemp(cacheRoot, ".ocr-*"); err == nil {
			raw, _ := json.Marshal(text)
			_, writeErr := f.Write(raw)
			closeErr := f.Close()
			if writeErr == nil && closeErr == nil {
				_ = os.Rename(f.Name(), path)
			}
			_ = os.Remove(f.Name())
		}
	}
	return text, nil
}

func prepareStudentMaterials(ctx context.Context, task GradingTask, readImage requirementImageReader) studentMaterials {
	result := studentMaterials{Schema: "student-materials.v1", Files: []materialEvidence{}, Reading: readingReport{Status: "COMPLETE", Warnings: []string{}, Files: []materialEvidence{}}}
	result.Archives = task.Sources
	result.Reading.Archives = task.Sources
	for _, skipped := range task.SkippedFiles {
		// Download payloads and build bookkeeping are inventory, not evidence.
		if role := materialRole(skipped.Path); role == "artifact" || role == "dependency" || strings.HasSuffix(strings.ToLower(skipped.Path), ".map") || strings.HasSuffix(strings.ToLower(skipped.Path), ".tsbuildinfo") {
			continue
		}
		result.SkippedFiles = append(result.SkippedFiles, skipped)
		result.Reading.SkippedFiles = append(result.Reading.SkippedFiles, skipped)
		result.Reading.Status = "NEEDS_REVIEW"
		result.Reading.Warnings = append(result.Reading.Warnings, skipped.Path+"："+skipped.Reason)
	}
	paths := append([]string(nil), task.FilePaths...)
	if len(paths) > 2000 {
		result.Reading.Status = "NEEDS_REVIEW"
		result.Reading.Warnings = append(result.Reading.Warnings, "文件数超过2000项读取预算，完整原始清单仍保留，需复核")
		paths = paths[:2000]
	}
	sort.SliceStable(paths, func(i, j int) bool {
		priority := func(s string) int {
			switch materialRole(s) {
			case "artifact", "dependency":
				return 2
			default:
				return 0
			}
		}
		if priority(paths[i]) != priority(paths[j]) {
			return priority(paths[i]) < priority(paths[j])
		}
		return paths[i] < paths[j]
	})
	total := 0
	for i, filePath := range paths {
		rel := filepath.Base(filePath)
		if task.RootPath != "" {
			if p, err := filepath.Rel(task.RootPath, filePath); err == nil && p != ".." && !strings.HasPrefix(p, ".."+string(filepath.Separator)) {
				rel = filepath.ToSlash(p)
			}
		}
		f := materialEvidence{ID: fmt.Sprintf("F%04d", i+1), Path: rel, Role: materialRole(rel), Status: "READ", Warnings: []string{}}
		info, err := os.Stat(filePath)
		if err != nil || !info.Mode().IsRegular() {
			f.Status = "FAILED"
			f.Warnings = append(f.Warnings, "材料文件无法读取")
		} else {
			f.Size = info.Size()
			f.SHA256, err = materials.HashFile(filePath)
			if err != nil {
				f.Status = "FAILED"
				f.Warnings = append(f.Warnings, "材料校验无法完成")
			} else if f.Role == "artifact" || f.Role == "dependency" {
				f.Status = "INVENTORY_ONLY"
			} else if f.Role == "unsupported" || info.Size() > 32<<20 {
				f.Status = "FAILED"
				f.Warnings = append(f.Warnings, "材料格式不支持或超过32MiB读取上限")
			} else if data, err := os.ReadFile(filePath); err != nil {
				f.Status = "FAILED"
				f.Warnings = append(f.Warnings, "材料文件无法读取")
			} else {
				ext := strings.ToLower(filepath.Ext(filePath))
				if ext == ".docx" || ext == ".doc" || ext == ".pdf" || f.Role == "evidence" && (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp" || ext == ".gif") {
					f.Text, f.Warnings = extractDocument(ctx, rel, data, readImage, studentDocumentLimits)
				} else {
					f.Text = ExtractContent(rel, data, 0)
				}
				f.Redactions = len(credentialValue.FindAllStringIndex(f.Text, -1))
				f.Text = credentialValue.ReplaceAllString(f.Text, "[凭据值已脱敏]")
				if !utf8.ValidString(f.Text) || strings.ContainsAny(f.Text, "\x00\ufffd") {
					f.Warnings = append(f.Warnings, "提取文字存在乱码或无效字符，需复核原文件")
				}
				if len(data) > 0 && strings.TrimSpace(f.Text) == "" {
					f.Warnings = append(f.Warnings, "未提取到有效内容，不能视为未提交")
				}
				if len(f.Text) > studentEvidenceLimit || total+len(f.Text) > studentEvidenceLimit {
					f.Text = ""
					f.Warnings = append(f.Warnings, "评分材料超过8MiB读取预算，未静默截断，请复核")
				}
				if len(f.Warnings) > 0 {
					f.Status = "PARTIAL"
				}
				total += len(f.Text)
			}
		}
		if f.Status == "FAILED" || f.Status == "PARTIAL" {
			result.Reading.Status = "NEEDS_REVIEW"
			for _, w := range f.Warnings {
				result.Reading.Warnings = append(result.Reading.Warnings, rel+"："+w)
			}
		}
		result.Files = append(result.Files, f)
		meta := f
		meta.Text = ""
		result.Reading.Files = append(result.Reading.Files, meta)
	}
	if len(paths) == 0 {
		result.Reading.Status = "NEEDS_REVIEW"
		result.Reading.Warnings = append(result.Reading.Warnings, "没有可读取的材料文件")
	}
	if result.Reading.Status == "NEEDS_REVIEW" {
		result.Reading.ReviewStage = "READING"
	}
	return result
}

func (s studentMaterials) jsonText() string { data, _ := json.Marshal(s); return string(data) }
func (s studentMaterials) analysisText() string {
	var out strings.Builder
	for _, f := range s.Files {
		if f.Role == "artifact" || f.Role == "dependency" || f.Role == "config" || f.Text == "" {
			continue
		}
		fmt.Fprintf(&out, "--- 文件开始: %s ---\n%s\n--- 文件结束: %s ---\n\n", f.Path, f.Text, f.Path)
	}
	return out.String()
}
