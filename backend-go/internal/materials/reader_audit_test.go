package materials

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Explicit opt-in, reads retained archives into new diagnostic roots only.
// No database, queue, grading model or existing material revision is modified.
func TestRealMaterialReaderRebuildAudit(t *testing.T) {
	input := os.Getenv("MATERIAL_REBUILD_INVENTORY")
	if input == "" {
		t.Skip("requires explicit local material inventory")
	}
	raw, err := os.ReadFile(input)
	if err != nil {
		t.Fatal(err)
	}
	var entries []map[string]any
	if err = json.Unmarshal(raw, &entries); err != nil {
		t.Fatal(err)
	}
	manifests := map[string]string{}
	added := 0
	for _, entry := range entries {
		path := entry["manifest"].(string)
		if _, ok := manifests[path]; ok {
			continue
		}
		old, err := LoadManifest(path, "")
		if err != nil {
			t.Fatal(err)
		}
		archives, err := filepath.Glob(filepath.Join(filepath.Dir(path), "original", "*.zip"))
		if err != nil || len(archives) != 1 {
			t.Fatal("expected one retained ZIP", err)
		}
		out, err := os.MkdirTemp("", "material-reader-v2-")
		if err != nil {
			t.Fatal(err)
		}
		current, err := prepare(t.Context(), old.BatchID, old.AssignmentID, archives[0], out)
		if err != nil || len(current.Students) != len(old.Students) {
			t.Fatal("rebuild lost students", err)
		}
		files := map[string]string{}
		for _, student := range current.Students {
			for _, file := range student.Files {
				files[file.Path] = file.SHA256
				if strings.HasSuffix(file.Path, ".mjs") || strings.HasSuffix(file.Path, ".cjs") {
					added++
				}
			}
			if student.ID == "24039100636" || student.ID == "24039100401" {
				if len(current.StudentSources(student)) < 2 {
					t.Fatal("missing original student ZIP", student.ID)
				}
				found := false
				for _, file := range student.Files {
					found = found || strings.Contains(file.Path, "test/") && strings.HasSuffix(file.Path, ".mjs")
				}
				if !found {
					t.Fatal("known submitted test missing", student.ID)
				}
			}
		}
		for _, student := range old.Students {
			for _, file := range student.Files {
				if files[file.Path] != file.SHA256 {
					t.Fatal("rebuild changed old evidence", file.Path)
				}
			}
		}
		newPath := filepath.Join(out, "manifest.json")
		data, _ := json.MarshalIndent(current, "", "  ")
		if err = os.WriteFile(newPath, data, 0600); err != nil {
			t.Fatal(err)
		}
		if _, err = LoadManifest(newPath, ""); err != nil {
			t.Fatal(err)
		}
		manifests[path] = newPath
		t.Logf("batch=%s students=%d archives=%d new_manifest=%s", old.BatchID, len(current.Students), len(current.Sources), newPath)
	}
	if added < 6 {
		t.Fatal("expected actual JS module tests recovered", added)
	}
	for _, entry := range entries {
		entry["manifest"] = manifests[entry["manifest"].(string)]
	}
	data, _ := json.MarshalIndent(entries, "", "  ")
	output := os.Getenv("MATERIAL_REBUILD_OUTPUT")
	if output == "" {
		t.Fatal("explicit output inventory required")
	}
	if err = os.WriteFile(output, data, 0600); err != nil {
		t.Fatal(err)
	}
	t.Logf("recovered_mjs_cjs=%d students=%d", added, len(entries))
}
