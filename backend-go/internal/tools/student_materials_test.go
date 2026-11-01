package tools

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/grpc"
	"grading-gateway/internal/materials"
	"grading-gateway/pb"
)

func TestStudentMaterialsKeepCoreAfterHugeDownloads(t *testing.T) {
	root := t.TempDir()
	task := GradingTask{RootPath: root}
	for name, text := range map[string]string{"downloads/big.txt": strings.Repeat("a", 7<<20), "src/index.ts": "await Promise.all(tasks); setInterval(report,500)", "report.md": "完整设计报告", "files.json": "[]"} {
		path := filepath.Join(root, name)
		os.MkdirAll(filepath.Dir(path), 0700)
		os.WriteFile(path, []byte(text), 0600)
		task.FilePaths = append(task.FilePaths, path)
	}
	parsed := prepareStudentMaterials(t.Context(), task, func(context.Context, string, []byte) (string, error) {
		t.Fatal("download artifact invoked OCR")
		return "", nil
	})
	if parsed.Reading.Status != "COMPLETE" || !strings.Contains(parsed.jsonText(), "Promise.all") || len(parsed.jsonText()) > 10000 {
		t.Fatal("lost core material or included huge artifact", parsed.Reading)
	}
	for _, f := range parsed.Files {
		if f.Path == "downloads/big.txt" && (f.Status != "INVENTORY_ONLY" || f.Text != "" || f.SHA256 == "") {
			t.Fatal("artifact has no inventory", f)
		}
	}
}

func TestDirectoryLabelsCannotHideSubmittedSourceOrReports(t *testing.T) {
	for _, name := range []string{"downloads/src/main.ts", "output/main.py", "outputs/实验报告.docx"} {
		if role := materialRole(name); role == "artifact" || role == "dependency" {
			t.Fatalf("submitted material hidden by directory label: %s", name)
		}
	}
	if materialRole("downloads/node_modules/lib/index.js") != "dependency" {
		t.Fatal("source exception must not include third-party dependencies")
	}
	if materialRole("downloads-1793250582471/file3.jpg") != "artifact" {
		t.Fatal("timestamped download output treated as submitted report evidence")
	}
}

func TestWordImagePositionAndTableBoundaries(t *testing.T) {
	var buf bytes.Buffer
	writer := zip.NewWriter(&buf)
	parts := map[string]string{
		"word/document.xml":            `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:body><w:p><w:r><w:t>运行结果</w:t></w:r></w:p><w:p><w:r><w:drawing><a:blip r:embed="rId1"/></w:drawing></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>测试</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>通过</w:t></w:r></w:p></w:tc></w:tr></w:tbl><w:p><w:r><w:t>结论</w:t></w:r></w:p></w:body></w:document>`,
		"word/_rels/document.xml.rels": `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="media/run.png"/></Relationships>`,
		"word/media/run.png":           "synthetic-image",
	}
	for name, data := range parts {
		w, _ := writer.Create(name)
		w.Write([]byte(data))
	}
	writer.Close()
	calls := 0
	text, warnings := extractDocument(t.Context(), "报告.docx", buf.Bytes(), func(context.Context, string, []byte) (string, error) { calls++; return "成功3个，失败0个", nil }, studentDocumentLimits)
	if calls != 1 || len(warnings) > 0 || strings.Index(text, "成功3个") < strings.Index(text, "运行结果") || strings.Index(text, "成功3个") > strings.Index(text, "结论") || !strings.Contains(text, "[表格开始]") || !strings.Contains(text, "\t|\t") {
		t.Fatal("Word evidence lost order or table boundaries", text, warnings, calls)
	}
}

func TestStudentReadFailureRequiresReview(t *testing.T) {
	root := t.TempDir()
	file := filepath.Join(root, "截图.png")
	os.WriteFile(file, []byte("synthetic-image"), 0600)
	parsed := prepareStudentMaterials(t.Context(), GradingTask{RootPath: root, FilePaths: []string{file}}, func(context.Context, string, []byte) (string, error) { return "", context.DeadlineExceeded })
	if parsed.Reading.Status != "NEEDS_REVIEW" || len(parsed.Reading.Warnings) == 0 || parsed.Files[0].Status != "PARTIAL" {
		t.Fatal("failed OCR became missing submission", parsed)
	}
}

func TestWordFormulaCannotSilentlyDisappear(t *testing.T) {
	var buf bytes.Buffer
	writer := zip.NewWriter(&buf)
	part, _ := writer.Create("word/document.xml")
	part.Write([]byte(`<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><w:body><w:p><m:oMath><m:r><m:t>x+y</m:t></m:r></m:oMath></w:p></w:body></w:document>`))
	writer.Close()
	text, warnings := extractDocument(t.Context(), "answer.docx", buf.Bytes(), nil, studentDocumentLimits)
	if !strings.Contains(text, "x+y") || len(warnings) == 0 {
		t.Fatal("formula content silently lost or incorrectly considered complete", text, warnings)
	}
}

func TestCredentialValuesAreRedactedWithoutChangingOriginalEvidence(t *testing.T) {
	root := t.TempDir()
	file := filepath.Join(root, "client.py")
	secret := "sk-" + strings.Repeat("x", 40)
	source := "client(api_key='" + secret + "')"
	os.WriteFile(file, []byte(source), 0600)
	parsed := prepareStudentMaterials(t.Context(), GradingTask{RootPath: root, FilePaths: []string{file}}, nil)
	if parsed.Files[0].Redactions != 1 || strings.Contains(parsed.jsonText(), secret) || !strings.Contains(parsed.Files[0].Text, "凭据值已脱敏") {
		t.Fatal("credential value reached model evidence")
	}
	actual, _ := os.ReadFile(file)
	if string(actual) != source {
		t.Fatal("original evidence changed")
	}
}

type batchPlagFixture struct {
	pb.ComputeServiceClient
	pairs map[string]bool
	calls int
}

func (f *batchPlagFixture) CheckPlagiarism(_ context.Context, r *pb.PlagiarismRequest, _ ...grpc.CallOption) (*pb.PlagiarismResponse, error) {
	f.calls++
	for a := range r.StudentTexts {
		for b := range r.StudentTexts {
			if a < b {
				f.pairs[a+"/"+b] = true
			}
		}
	}
	return &pb.PlagiarismResponse{PlagiarismResultsJson: "{}"}, nil
}
func TestPlagiarismBatchesStillCompareAcrossGroups(t *testing.T) {
	fixture := &batchPlagFixture{pairs: map[string]bool{}}
	_, err := checkPlagiarismBatched(t.Context(), fixture, map[string]string{"a": strings.Repeat("a", 80), "b": strings.Repeat("b", 80), "c": strings.Repeat("c", 80), "d": strings.Repeat("d", 80)}, 130)
	if err != nil || len(fixture.pairs) != 6 || fixture.calls != 6 {
		t.Fatal("cross-group comparisons lost", fixture, err)
	}
}

func TestGradeReadingMetadataSurvivesJSON(t *testing.T) {
	parsed := studentMaterials{Schema: "student-materials.v1", Reading: readingReport{Status: "COMPLETE"}, Files: []materialEvidence{{ID: "F0001", Path: "src/main.ts", Role: "code", Status: "READ", Text: "code"}}}
	var decoded studentMaterials
	if err := json.Unmarshal([]byte(parsed.jsonText()), &decoded); err != nil || decoded.Files[0].Path != "src/main.ts" {
		t.Fatal(decoded, err)
	}
}

func TestStudentReaderSharesModernScriptFormatsAndProvenance(t *testing.T) {
	root := t.TempDir()
	task := GradingTask{RootPath: root, Sources: []materials.ArchiveSource{{ID: "source-1", Name: "20260001_张三.zip", Format: "zip", SHA256: "original-archive", StudentID: "20260001"}}}
	for _, name := range []string{"test/main.test.mjs", "demo/server.cjs", "src/view.tsx", "src/page.jsx"} {
		path := filepath.Join(root, name)
		os.MkdirAll(filepath.Dir(path), 0700)
		os.WriteFile(path, []byte("actual source"), 0600)
		task.FilePaths = append(task.FilePaths, path)
	}
	parsed := prepareStudentMaterials(t.Context(), task, nil)
	if parsed.Reading.Status != "COMPLETE" || len(parsed.Files) != 4 || len(parsed.Reading.Archives) != 1 {
		t.Fatal("source evidence missing", parsed.Reading)
	}
	for _, file := range parsed.Files {
		if file.Role != "code" || file.Status != "READ" || file.Text != "actual source" {
			t.Fatal("script not read", file)
		}
	}
	if !strings.Contains(parsed.jsonText(), "20260001_张三.zip") {
		t.Fatal("original attachment missing from model input")
	}
	task.SkippedFiles = []materials.SkippedFile{{Path: "answer.pptx", Reason: "暂不支持"}}
	if parsed = prepareStudentMaterials(t.Context(), task, nil); parsed.Reading.Status != "NEEDS_REVIEW" {
		t.Fatal("unread answer claimed complete")
	}
}
