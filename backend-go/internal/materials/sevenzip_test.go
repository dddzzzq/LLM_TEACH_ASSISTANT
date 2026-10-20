package materials

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/uuid"
)

func sevenZipFixture(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name+".7z.b64"))
	if err != nil {
		t.Fatal(err)
	}
	data, err := base64.StdEncoding.DecodeString(strings.TrimSpace(string(raw)))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestPrepare7zPreservesChinesePathsAndProducesUsableManifest(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "20260001_张三.7Z")
	save(t, source, sevenZipFixture(t, "submission"))
	out := filepath.Join(root, "out")
	m, err := prepare(context.Background(), uuid.NewString(), 1, source, out)
	if err != nil || len(m.Students) != 1 || len(m.Issues) != 0 {
		t.Fatalf("7z preparation: %+v %v", m, err)
	}
	student := m.Students[0]
	if student.ID != "20260001" || student.Name != "张三" || len(student.Files) != 4 || m.FilteredEntries != 2 {
		t.Fatalf("7z grouping/filtering: %+v", m)
	}
	contents := map[string]string{}
	for _, file := range student.Files {
		data, err := os.ReadFile(filepath.Join(out, filepath.FromSlash(file.Path)))
		if err != nil {
			t.Fatal(err)
		}
		rel := strings.TrimPrefix(file.Path, "students/20260001_张三/20260001_张三/")
		contents[file.Kind+":"+rel] = string(data)
	}
	for key, want := range map[string]string{
		"code:src/main.py": "print('程序')\n", "code:test/main.py": "print('test')\n",
		"report:设计报告.md": "分析与设计\n", "report:empty.txt": "",
	} {
		if got, ok := contents[key]; !ok || got != want {
			t.Fatalf("missing/changed %s: %+v", key, contents)
		}
	}
	manifest := filepath.Join(out, "manifest.json")
	raw, _ := json.Marshal(m)
	save(t, manifest, raw)
	hash, _ := HashFile(manifest)
	if _, err := LoadManifest(manifest, hash); err != nil {
		t.Fatal("prepared 7z files cannot be consumed by grading", err)
	}
}

func TestPrepareMixedZIPAnd7zNesting(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "class.zip")
	save(t, source, archiveBytes(t, map[string]string{
		"20260001_张三.7z": string(sevenZipFixture(t, "submission")),
		"20260002_李四.7z": string(sevenZipFixture(t, "nested")),
		"20260003_王五.zip": string(archiveBytes(t, map[string]string{
			"project.7z": string(sevenZipFixture(t, "nested")),
		})),
	}))
	m, err := prepare(context.Background(), uuid.NewString(), 1, source, filepath.Join(root, "out"))
	if err != nil || len(m.Students) != 3 || len(m.Issues) != 0 {
		t.Fatalf("mixed nesting: %+v %v", m, err)
	}
	for i, want := range []int{4, 2, 2} {
		if len(m.Students[i].Files) != want {
			t.Fatalf("mixed nesting lost material: %+v", m.Students[i])
		}
	}
}

func TestUnpack7zRejectsUnsafeEncryptedAndDamagedArchives(t *testing.T) {
	for _, tc := range []struct{ fixture, message string }{
		{"encrypted_headers", "7z"}, {"encrypted_data", "7z"},
		{"duplicate", "重复路径"}, {"traversal", "越界路径"},
		{"corrupt_data", "校验失败"}, {"oversize", "256MiB"},
	} {
		t.Run(tc.fixture, func(t *testing.T) {
			root := t.TempDir()
			source := filepath.Join(root, "submission.7z")
			save(t, source, sevenZipFixture(t, tc.fixture))
			p := &preparer{ctx: context.Background()}
			err := p.unpack(source, filepath.Join(root, "out"))
			if err == nil || !strings.Contains(err.Error(), tc.message) {
				t.Fatalf("unsafe 7z accepted or unclear error: %v", err)
			}
			if _, err := os.Stat(filepath.Join(root, "escape.txt")); !os.IsNotExist(err) {
				t.Fatal("7z path escaped extraction directory")
			}
			if tc.fixture == "oversize" && (p.files != 0 || p.expanded != 0) {
				t.Fatal("oversized entry was decoded/written before checking size")
			}
		})
	}
}

func TestUnpack7zSkipsLinksAndKeepsNormalFiles(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "submission.7z")
	save(t, source, sevenZipFixture(t, "symlink"))
	out := filepath.Join(root, "out")
	p := &preparer{ctx: context.Background()}
	if err := p.unpack(source, out); err != nil {
		t.Fatal(err)
	}
	if p.files != 1 || p.manifest.SkippedSymlinks != 1 || len(p.manifest.Issues) != 1 {
		t.Fatalf("7z link not filtered: %+v", p.manifest)
	}
	if _, err := os.Lstat(filepath.Join(out, "linked.py")); !os.IsNotExist(err) {
		t.Fatal("7z link was materialized")
	}
	data, err := os.ReadFile(filepath.Join(out, "main.py"))
	if err != nil || string(data) != "print('link fixture')\n" {
		t.Fatal("normal 7z file lost", err)
	}
}

func TestUnpack7zHonorsCancellationAndGlobalBudget(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "submission.7z")
	save(t, source, sevenZipFixture(t, "submission"))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	p := &preparer{ctx: ctx}
	if err := p.unpack(source, filepath.Join(root, "cancelled")); !errors.Is(err, context.Canceled) || p.files != 0 {
		t.Fatal("7z ignored cancellation", err)
	}
	p = &preparer{ctx: context.Background(), files: maxFiles}
	if err := p.unpack(source, filepath.Join(root, "budget")); !errors.Is(err, errPreparationLimit) {
		t.Fatal("7z ignored global file budget", err)
	}
}

func TestPrepareBroken7zDoesNotAdmitPartialStudent(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "input")
	save(t, filepath.Join(source, "20260001_张三.7z"), sevenZipFixture(t, "submission"))
	save(t, filepath.Join(source, "20260002_李四", "main.py"), []byte("partial"))
	save(t, filepath.Join(source, "20260002_李四", "project.7z"), sevenZipFixture(t, "corrupt_data"))
	save(t, filepath.Join(source, "20260003_王五.7z"), []byte("not a 7z file"))
	m, err := prepare(context.Background(), uuid.NewString(), 1, source, filepath.Join(root, "out"))
	if err != nil || len(m.Students) != 1 || m.Students[0].ID != "20260001" || len(m.Issues) != 2 {
		t.Fatalf("partial/broken 7z admitted: %+v %v", m, err)
	}
}

func TestPrepareDownloaded7zMaterialsWithoutGrading(t *testing.T) {
	archives := os.Getenv("MATERIALS_VERIFY_7Z_ARCHIVES")
	if archives == "" {
		t.Skip("optional read-only verification of downloaded archives containing 7z")
	}
	for _, source := range strings.Split(archives, "|") {
		t.Run(filepath.Base(source), func(t *testing.T) {
			out := t.TempDir()
			m, err := prepare(context.Background(), uuid.NewString(), 1, source, out)
			if err != nil || len(m.Students) == 0 || len(m.Issues) != 0 {
				t.Fatalf("downloaded 7z material: students=%d issues=%+v err=%v", len(m.Students), m.Issues, err)
			}
			manifest := filepath.Join(out, "manifest.json")
			raw, _ := json.Marshal(m)
			save(t, manifest, raw)
			hash, _ := HashFile(manifest)
			if _, err := LoadManifest(manifest, hash); err != nil {
				t.Fatal(err)
			}
			count := 0
			for _, student := range m.Students {
				count += len(student.Files)
			}
			t.Logf("verified actual material: students=%d files=%d filtered=%d issues=%d", len(m.Students), count, m.FilteredEntries, len(m.Issues))
		})
	}
}
