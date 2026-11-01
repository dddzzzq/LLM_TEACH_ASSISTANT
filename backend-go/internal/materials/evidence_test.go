package materials

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/uuid"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"grading-gateway/internal/models"
)

func TestIntegrationEvidenceRequiresOwnerManifestAndUnchangedFile(t *testing.T) {
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
	sql, _ := db.DB()
	defer sql.Close()
	if err = db.AutoMigrate(&models.MaterialBatch{}); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	source := filepath.Join(root, "input", "20260001_张三")
	save(t, filepath.Join(source, "main.ts"), []byte("source evidence"))
	id := uuid.NewString()
	output := filepath.Join(root, "prepared")
	manifest, err := prepare(t.Context(), id, 1, source, output)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(output, "manifest.json")
	raw, _ := json.Marshal(manifest)
	save(t, path, raw)
	digest, _ := HashFile(path)
	batch := models.MaterialBatch{ID: id, OwnerID: 8701, ManifestPath: path, ManifestSHA256: digest}
	if err = db.Create(&batch).Error; err != nil {
		t.Fatal(err)
	}
	s := &Service{DB: db}
	relative := manifest.Students[0].Files[0].Path
	file, entry, err := s.OpenEvidence(t.Context(), Actor{UserID: 8701, Role: "teacher"}, id, relative)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := io.ReadAll(file)
	file.Close()
	if string(data) != "source evidence" || entry.Path != relative {
		t.Fatal("wrong evidence")
	}
	for _, test := range []struct {
		actor Actor
		path  string
	}{{Actor{UserID: 8702, Role: "teacher"}, relative}, {Actor{UserID: 8701, Role: "student"}, relative}, {Actor{UserID: 8701, Role: "teacher"}, "../outside"}} {
		if file, _, err := s.OpenEvidence(t.Context(), test.actor, id, test.path); err == nil {
			file.Close()
			t.Fatal("unsafe evidence access", test)
		}
	}
	os.WriteFile(filepath.Join(output, relative), []byte("tampered"), 0600)
	if file, _, err := s.OpenEvidence(t.Context(), Actor{UserID: 8701, Role: "teacher"}, id, relative); err == nil {
		file.Close()
		t.Fatal("changed evidence served")
	}
}
