package materials

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"grading-gateway/internal/models"
)

func archiveBytes(t *testing.T, files map[string]string) []byte {
	t.Helper()
	var out bytes.Buffer
	z := zip.NewWriter(&out)
	for name, data := range files {
		w, err := z.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		w.Write([]byte(data))
	}
	if err := z.Close(); err != nil {
		t.Fatal(err)
	}
	return out.Bytes()
}
func save(t *testing.T, path string, content []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, content, 0600); err != nil {
		t.Fatal(err)
	}
}
func TestPrepareNestedArchivesPreservesFilesAndValidatesSnapshot(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "input")
	save(t, filepath.Join(source, "20260001-张三.zip"), archiveBytes(t, map[string]string{"src/main.py": "print(1)", "test/main.py": "print(2)", "设计报告.txt": "分析与设计", "node_modules/dependency.js": "ignore"}))
	save(t, filepath.Join(source, "20260002_李四.zip"), archiveBytes(t, map[string]string{"报告.md": "report", "project.zip": string(archiveBytes(t, map[string]string{"src/a.java": "class A {}"}))}))
	out := filepath.Join(root, "output")
	m, err := prepare(context.Background(), uuid.NewString(), 1, source, out)
	if err != nil || len(m.Students) != 2 || len(m.Issues) != 0 {
		t.Fatalf("unexpected grouping %+v %v", m, err)
	}
	if len(m.Students[0].Files) != 3 || len(m.Students[1].Files) != 2 {
		t.Fatalf("nested files missing or dependencies retained: %+v", m)
	}
	manifest := filepath.Join(out, "manifest.json")
	raw, _ := json.Marshal(m)
	save(t, manifest, raw)
	hash, _ := HashFile(manifest)
	if _, err := LoadManifest(manifest, hash); err != nil {
		t.Fatal(err)
	}
	save(t, filepath.Join(out, filepath.FromSlash(m.Students[0].Files[0].Path)), []byte("changed"))
	if _, err := LoadManifest(manifest, hash); err == nil {
		t.Fatal("mutated material accepted")
	}
}

func TestPrepareKeepsModernScriptsAndArchiveProvenance(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "class.zip")
	save(t, source, archiveBytes(t, map[string]string{"20260001_张三.zip": string(archiveBytes(t, map[string]string{
		"project.zip": string(archiveBytes(t, map[string]string{"src/main.ts": "code", "test/main.test.mjs": "node test", "demo/server.cjs": "server", "src/view.tsx": "view", "src/page.jsx": "page", "报告.md": "report", "notes.unknown": "not supported", "run.bat": "node src/main.js", "run.cmd": "node src/main.js", "bin/tool": "#!/usr/bin/env node\nconsole.log('run')"})),
	}))}))
	m, err := prepare(t.Context(), "batch", 1, source, filepath.Join(root, "out"))
	if err != nil || len(m.Students) != 1 {
		t.Fatal(err, m)
	}
	if m.ReaderVersion != CurrentReaderVersion || len(m.Sources) != 3 {
		t.Fatal("missing archive chain", m.Sources)
	}
	student := m.Students[0]
	if len(student.Files) != 9 || len(student.SkippedFiles) != 1 || len(student.SourceIDs) != 3 {
		t.Fatal("scripts or omitted-file inventory lost", student)
	}
	for _, file := range student.Files {
		if strings.HasSuffix(file.Path, ".mjs") || strings.HasSuffix(file.Path, ".cjs") {
			if file.Kind != "code" {
				t.Fatal("script was not source", file)
			}
		}
	}
	sources := m.StudentSources(student)
	if sources[0].Name != "class.zip" || sources[0].StudentID != "" || sources[1].Name != "20260001_张三.zip" || sources[1].StudentID != student.ID || sources[2].ParentID != sources[1].ID || sources[1].ParentID != sources[0].ID {
		t.Fatal("archive attribution lost", sources)
	}
	for _, source := range sources {
		if source.SHA256 == "" || source.Format != "zip" || source.Size <= 0 {
			t.Fatal("archive identity missing", source)
		}
	}
}
func TestPrepareReportsConflictsAndTraversal(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "input")
	save(t, filepath.Join(source, "20260001_张三.zip"), archiveBytes(t, map[string]string{"a.py": "v1"}))
	save(t, filepath.Join(source, "20260001-张三.zip"), archiveBytes(t, map[string]string{"a.py": "v2"}))
	save(t, filepath.Join(source, "20260002_李四.zip"), archiveBytes(t, map[string]string{"a.py": "ok"}))
	save(t, filepath.Join(source, "20260003_王五.zip"), archiveBytes(t, map[string]string{"../escape.txt": "bad"}))
	m, err := prepare(context.Background(), uuid.NewString(), 1, source, filepath.Join(root, "out"))
	if err != nil || len(m.Students) != 1 || m.Students[0].ID != "20260002" || len(m.Issues) != 2 {
		t.Fatalf("unsafe selection: %+v %v", m, err)
	}
	if _, err := os.Stat(filepath.Join(root, "escape.txt")); !os.IsNotExist(err) {
		t.Fatal("archive escaped")
	}
}
func TestPrepareStudentDirectoryAndPathBoundary(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "20260001张三")
	save(t, filepath.Join(source, "src", "main.py"), []byte("ok"))
	m, err := prepare(context.Background(), uuid.NewString(), 1, source, filepath.Join(root, "out"))
	if err != nil || len(m.Students) != 1 || m.Students[0].Name != "张三" {
		t.Fatalf("student dir: %+v %v", m, err)
	}
	s := &Service{AllowedRoots: []string{source}}
	if _, err := s.allowed(root); err == nil {
		t.Fatal("parent directory accepted")
	}
	link := filepath.Join(source, "linked")
	if err := os.Symlink(root, link); err == nil {
		if _, err := s.allowed(link); err == nil {
			t.Fatal("symlink escaped allowed roots")
		}
	}
	if path, err := NormalizePath(`D:\dzq\作业`); err != nil || (!strings.HasPrefix(path, "/mnt/d/") && !drivePath.MatchString(path)) {
		t.Fatalf("windows path: %s %v", path, err)
	}
}

func TestUnpackFiltersDependenciesBeforeReadingAndCountingAndSkipsLinks(t *testing.T) {
	root := t.TempDir()
	var data bytes.Buffer
	z := zip.NewWriter(&data)
	for i := 0; i < maxFiles+5; i++ {
		w, err := z.Create(fmt.Sprintf("project/.VENV/lib/dependency-%d.py", i))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write([]byte("ignored")); err != nil {
			t.Fatal(err)
		}
	}
	for name, content := range map[string]string{"src/main.py": "print(1)", "报告.md": "report", `project\node_modules\module.js`: "ignored"} {
		w, err := z.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write([]byte(content)); err != nil {
			t.Fatal(err)
		}
	}
	header := &zip.FileHeader{Name: "src/linked.py", Method: zip.Store}
	header.SetMode(os.ModeSymlink | 0777)
	w, err := z.CreateHeader(header)
	if err != nil {
		t.Fatal(err)
	}
	w.Write([]byte("../../outside.py"))
	if err := z.Close(); err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(root, "20260001_张三.zip")
	save(t, source, data.Bytes())
	out := filepath.Join(root, "extracted")
	p := &preparer{ctx: context.Background()}
	if err := p.unpack(source, out); err != nil {
		t.Fatal(err)
	}
	if p.files != 2 || p.expanded != int64(len("print(1)report")) || p.manifest.FilteredEntries != maxFiles+6 || p.manifest.SkippedSymlinks != 1 || len(p.manifest.Issues) != 1 {
		t.Fatalf("dependencies consumed budget or link stopped archive: files=%d bytes=%d manifest=%+v", p.files, p.expanded, p.manifest)
	}
	if _, err := os.Lstat(filepath.Join(out, "src", "linked.py")); !os.IsNotExist(err) {
		t.Fatal("symbolic link was materialized")
	}
	if _, err := os.Stat(filepath.Join(out, "project")); !os.IsNotExist(err) {
		t.Fatal("dependency tree was written")
	}
}

func TestPreparationKeepsCompleteStudentsAndExcludesIncompleteSubmission(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "input")
	save(t, filepath.Join(source, "20260001_张三", "main.py"), []byte("good"))
	save(t, filepath.Join(source, "20260002_李四", "main.py"), []byte("partial"))
	save(t, filepath.Join(source, "20260002_李四", "z-broken.zip"), []byte("invalid ZIP"))
	m, err := prepare(context.Background(), uuid.NewString(), 1, source, filepath.Join(root, "out"))
	if err != nil || len(m.Students) != 1 || m.Students[0].ID != "20260001" || len(m.Issues) != 1 {
		t.Fatalf("incomplete student accepted or complete student lost: %+v %v", m, err)
	}
}

func TestPreparationBudgetPreservesOnlyCompleteStudents(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "input")
	save(t, filepath.Join(source, "20260001_张三", "main.py"), []byte("good"))
	save(t, filepath.Join(source, "20260002_李四", "a.py"), []byte("partial"))
	save(t, filepath.Join(source, "20260002_李四", "b.py"), []byte("over budget"))
	p := &preparer{ctx: context.Background(), root: filepath.Join(root, "out"), files: maxFiles - 2, manifest: Manifest{Version: 1, BatchID: uuid.NewString()}, students: map[string]*Student{}, origins: map[string]string{}, originDepth: map[string]int{}, conflicts: map[string]bool{}, incomplete: map[string]bool{}}
	err := p.visit(source, "", "", "", 0)
	if !errors.Is(err, errPreparationLimit) {
		t.Fatalf("missing precise budget failure: %v", err)
	}
	m, err := p.finish("input", err)
	if err != nil || len(m.Students) != 1 || m.Students[0].ID != "20260001" || len(m.Issues) == 0 {
		t.Fatalf("partial recovery: %+v %v", m, err)
	}
	path := filepath.Join(p.root, "manifest.json")
	raw, _ := json.Marshal(m)
	save(t, path, raw)
	hash, _ := HashFile(path)
	if _, err := LoadManifest(path, hash); err != nil {
		t.Fatal("retained manifest is unusable", err)
	}
}

func TestPrepareDownloadedArchiveWithoutGrading(t *testing.T) {
	source := os.Getenv("MATERIALS_VERIFY_ARCHIVE")
	if source == "" {
		t.Skip("optional read-only verification of a downloaded archive")
	}
	out := t.TempDir()
	m, err := prepare(context.Background(), uuid.NewString(), 3, source, out)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Students) != 144 || len(m.Issues) != 0 {
		t.Fatalf("download recovery: students=%d issues=%d", len(m.Students), len(m.Issues))
	}
	manifest := filepath.Join(out, "manifest.json")
	raw, _ := json.Marshal(m)
	save(t, manifest, raw)
	hash, _ := HashFile(manifest)
	if _, err := LoadManifest(manifest, hash); err != nil {
		t.Fatal(err)
	}
	files := 0
	for _, student := range m.Students {
		files += len(student.Files)
	}
	t.Logf("verified downloaded archive: students=%d files=%d filtered_entries=%d skipped_symlinks=%d issues=%d", len(m.Students), files, m.FilteredEntries, m.SkippedSymlinks, len(m.Issues))
}

func TestIntegrationPreparationIsDurableOwnedAndNeverImplicitlyGrades(t *testing.T) {
	dsn := os.Getenv("RPA_TEST_DSN")
	if dsn == "" {
		t.Skip("requires isolated RPA_TEST_DSN")
	}
	if !strings.Contains(dsn, "/rpa_test_") {
		t.Fatal("refusing production DB")
	}
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { sql, _ := db.DB(); sql.Close() })
	if err := db.AutoMigrate(&models.MaterialBatch{}, &models.AsyncJob{}, &models.Assignment{}, &models.RPAJob{}, &models.RPAFile{}); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	source := filepath.Join(root, "input")
	save(t, filepath.Join(source, "20260001_张三.zip"), archiveBytes(t, map[string]string{"main.py": "print(1)", "报告.md": "report"}))
	s := &Service{DB: db, Root: filepath.Join(root, "out"), AllowedRoots: []string{root}}
	ctx := context.Background()
	actor := Actor{UserID: 8101, Role: "teacher"}
	r := Request{SourcePath: source, CourseName: "分布式计算", AssignmentName: "第一次作业", ClassName: "A"}
	first, err := s.Create(ctx, actor, r, "prepare-once")
	if err != nil {
		t.Fatal(err)
	}
	again, err := s.Create(ctx, actor, r, "prepare-once")
	if err != nil || first.ID != again.ID {
		t.Fatal("duplicate preparation")
	}
	if _, err := s.Owned(ctx, Actor{UserID: 8102, Role: "teacher"}, first.ID); err == nil {
		t.Fatal("cross-owner readable")
	}
	r.AutoGrade = true
	if _, err := s.Create(ctx, actor, r, "prepare-once"); err == nil {
		t.Fatal("same key changed processing intent")
	}
	if err := s.ProcessOne(ctx, first.ID); err != nil {
		t.Fatal(err)
	}
	first, err = s.Owned(ctx, actor, first.ID)
	if err != nil || first.Status != "READY" || first.StudentCount != 1 || first.GradingJobID != "" || first.AutoGrade {
		t.Fatalf("preparation implicitly graded: %+v %v", first, err)
	}
	oldPath := first.ManifestPath
	if err := s.ProcessOne(ctx, first.ID); err != nil {
		t.Fatal(err)
	}
	first, _ = s.Owned(ctx, actor, first.ID)
	if first.ManifestPath != oldPath {
		t.Fatal("ready batch re-extracted")
	}
	r.AutoGrade = true
	cancelled, err := s.Create(ctx, actor, r, "cancel-before-run")
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Cancel(ctx, actor, cancelled.ID); err != nil {
		t.Fatal(err)
	}
	if err := s.ProcessOne(ctx, cancelled.ID); err != nil {
		t.Fatal(err)
	}
	cancelled, _ = s.Owned(ctx, actor, cancelled.ID)
	if cancelled.Status != "CANCELLED" || cancelled.AutoGrade {
		t.Fatal("cancelled batch resurrected")
	}
	recovery, err := s.Create(ctx, actor, r, "recovery")
	if err != nil {
		t.Fatal(err)
	}
	db.Model(&recovery).Updates(map[string]any{"status": "PROCESSING", "lease_owner": "dead-process", "lease_until": time.Now().Add(-time.Minute)})
	if err := s.ProcessOne(ctx, recovery.ID); err != nil {
		t.Fatal(err)
	}
	recovery, _ = s.Owned(ctx, actor, recovery.ID)
	if recovery.Status != "READY" || !recovery.AutoGrade || recovery.GradingJobID != "" {
		t.Fatal("durable preparation intent lost")
	}
	previousManifest := recovery.ManifestPath
	db.Model(&recovery).Updates(map[string]any{"status": "FAILED", "message": "old preparation failed"})
	if _, err := s.Retry(ctx, Actor{UserID: 8102, Role: "teacher"}, recovery.ID); err == nil {
		t.Fatal("cross-owner retry accepted")
	}
	if _, err := s.Retry(ctx, Actor{UserID: 8101, Role: "student"}, recovery.ID); err == nil {
		t.Fatal("student retry accepted")
	}
	if _, err := s.Retry(ctx, actor, cancelled.ID); err == nil {
		t.Fatal("cancelled batch resurrected")
	}
	retried, err := s.Retry(ctx, actor, recovery.ID)
	if err != nil || retried.Status != "PENDING" || retried.AutoGrade || retried.GradingState != "NOT_REQUESTED" || retried.ManifestPath != "" {
		t.Fatalf("unsafe retry: %+v %v", retried, err)
	}
	if _, err := s.Retry(ctx, actor, recovery.ID); err == nil {
		t.Fatal("running retry reset")
	}
	if err := s.ProcessOne(ctx, recovery.ID); err != nil {
		t.Fatal(err)
	}
	retried, _ = s.Owned(ctx, actor, recovery.ID)
	if retried.Status != "READY" || retried.AutoGrade || retried.GradingJobID != "" || retried.ManifestPath == previousManifest {
		t.Fatalf("retry reused execution or implicitly graded: %+v", retried)
	}
	if _, err := os.Stat(previousManifest); err != nil {
		t.Fatal("previous execution was removed")
	}
	partialSource := filepath.Join(root, "partial-input")
	save(t, filepath.Join(partialSource, "20260011_甲", "main.py"), []byte("complete"))
	save(t, filepath.Join(partialSource, "20260012_乙", "main.py"), []byte("incomplete"))
	save(t, filepath.Join(partialSource, "20260012_乙", "z.zip"), []byte("corrupt"))
	r.SourcePath = partialSource
	r.AutoGrade = false
	partial, err := s.Create(ctx, actor, r, "partial-recovery")
	if err != nil {
		t.Fatal(err)
	}
	if err := s.ProcessOne(ctx, partial.ID); err != nil {
		t.Fatal(err)
	}
	partial, _ = s.Owned(ctx, actor, partial.ID)
	if partial.Status != "NEEDS_REVIEW" || partial.StudentCount != 1 || partial.IssueCount == 0 || partial.GradingJobID != "" {
		t.Fatalf("partial batch lost usable manifest: %+v", partial)
	}
	if _, err := LoadManifest(partial.ManifestPath, partial.ManifestSHA256); err != nil {
		t.Fatal(err)
	}
	previousPartialManifest := partial.ManifestPath
	// After a parser fix or a corrected source file, a reviewed partial batch
	// can be prepared again, preserving its old execution and never grading.
	save(t, filepath.Join(partialSource, "20260012_乙", "z.zip"), archiveBytes(t, map[string]string{"report.md": "fixed report"}))
	if _, err := s.Retry(ctx, actor, partial.ID); err != nil {
		t.Fatal("cannot retry partial preparation", err)
	}
	if err := s.ProcessOne(ctx, partial.ID); err != nil {
		t.Fatal(err)
	}
	partial, _ = s.Owned(ctx, actor, partial.ID)
	if partial.Status != "READY" || partial.StudentCount != 2 || partial.IssueCount != 0 || partial.AutoGrade || partial.GradingJobID != "" || partial.ManifestPath == previousPartialManifest {
		t.Fatalf("partial retry lost material or implicitly graded: %+v", partial)
	}
	if _, err := os.Stat(previousPartialManifest); err != nil {
		t.Fatal("previous partial manifest was removed", err)
	}
	// A directly uploaded 7z uses the same durable preparation and snapshot path.
	sevenSource := filepath.Join(root, "20260021_七号.7z")
	save(t, sevenSource, sevenZipFixture(t, "submission"))
	r.SourcePath = sevenSource
	sevenBatch, err := s.Create(ctx, actor, r, "direct-7z")
	if err != nil {
		t.Fatal(err)
	}
	if err := s.ProcessOne(ctx, sevenBatch.ID); err != nil {
		t.Fatal(err)
	}
	sevenBatch, err = s.Owned(ctx, actor, sevenBatch.ID)
	if err != nil || sevenBatch.Status != "READY" || sevenBatch.StudentCount != 1 || sevenBatch.IssueCount != 0 || sevenBatch.AutoGrade || sevenBatch.GradingJobID != "" {
		t.Fatalf("7z failed durable preparation or implicitly graded: %+v %v", sevenBatch, err)
	}
	if _, err := LoadManifest(sevenBatch.ManifestPath, sevenBatch.ManifestSHA256); err != nil {
		t.Fatal(err)
	}
}
