package tools

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"grading-gateway/internal/grpcclient"
	"grading-gateway/internal/materials"
)

// Explicit opt-in: only re-read local files/OCR. Never invokes grading or writes
// submissions, scores, workflows or teacher reviews to the database.
func TestRealStudentReadingAudit(t *testing.T) {
	path := os.Getenv("MATERIAL_AUDIT_INVENTORY")
	if path == "" {
		t.Skip("requires explicit local audit inventory")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		ID       int    `json:"id"`
		Manifest string `json:"manifest"`
		Student  string `json:"student"`
	}
	if err = json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	prior := grpcclient.Client
	grpcclient.InitGrpcClient()
	defer func() { grpcclient.CloseGrpcClient(); grpcclient.Client = prior }()
	root := filepath.Join(filepath.Dir(path), "reading-v2")
	if err = os.MkdirAll(root, 0700); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	slots := make(chan struct{}, 6)
	for _, entry := range cases {
		entry := entry
		wg.Add(1)
		go func() {
			defer wg.Done()
			slots <- struct{}{}
			defer func() { <-slots }()
			m, err := materials.LoadManifest(entry.Manifest, "")
			if err != nil {
				t.Error(err)
				return
			}
			for _, student := range m.Students {
				if student.Name != entry.Student {
					continue
				}
				task := GradingTask{RootPath: filepath.Dir(entry.Manifest), StudentID: student.ID, StudentName: student.Name, Sources: m.StudentSources(student), SkippedFiles: student.SkippedFiles}
				for _, f := range student.Files {
					task.FilePaths = append(task.FilePaths, filepath.Join(task.RootPath, filepath.FromSlash(f.Path)))
				}
				result := prepareStudentMaterials(t.Context(), task, cachedStudentOCR)
				data := result.jsonText()
				name := filepath.Join(root, student.ID+".json")
				if err = os.WriteFile(name, []byte(data), 0600); err != nil {
					t.Error(err)
				}
				if entry.ID == 146 || entry.ID == 128 || entry.ID == 127 || entry.ID == 129 || entry.ID == 148 {
					maximum := 150000
					if entry.ID == 148 {
						// Keep ambiguous submitted scripts even inside downloads;
						// only clear payload/dependency bodies are omitted.
						maximum = 1 << 20
					}
					if !strings.Contains(data, "setInterval") || len(data) > maximum {
						t.Errorf("core evidence not recovered or artifact still fills request: id=%d bytes=%d", entry.ID, len(data))
					}
				}
				if entry.ID == 144 && (!strings.Contains(data, "页面图像识别") || !strings.Contains(data, "已下载")) {
					t.Error("PDF run screenshots were not read")
				}
				if entry.ID == 138 && !strings.Contains(data, "Word 内嵌图片") {
					t.Error("Word run screenshot was not read")
				}
				t.Logf("record=%d student=%s reading=%s bytes=%d warnings=%v", entry.ID, student.ID, result.Reading.Status, len(data), result.Reading.Warnings)
			}
		}()
	}
	wg.Wait()
}
