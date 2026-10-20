package materials

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

type File struct {
	Path   string `json:"path"`
	Kind   string `json:"kind"`
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}
type Student struct {
	ID       string   `json:"student_id"`
	Name     string   `json:"student_name"`
	Files    []File   `json:"files"`
	Warnings []string `json:"warnings"`
}
type Issue struct {
	Path    string `json:"path"`
	Message string `json:"message"`
}
type Manifest struct {
	Version         int       `json:"version"`
	BatchID         string    `json:"batch_id"`
	AssignmentID    uint      `json:"assignment_id"`
	Students        []Student `json:"students"`
	Issues          []Issue   `json:"issues"`
	FilteredEntries int       `json:"filtered_entries,omitempty"`
	SkippedSymlinks int       `json:"skipped_symlinks,omitempty"`
}

func HashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// LoadManifest validates both immutable file contents and containment before grading.
func LoadManifest(path, expectedHash string) (Manifest, error) {
	var m Manifest
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Size() > 16<<20 {
		return m, fmt.Errorf("材料清单不存在或过大")
	}
	if expectedHash != "" {
		actual, err := HashFile(path)
		if err != nil || actual != expectedHash {
			return m, fmt.Errorf("材料清单已改变，请重新导入")
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil || json.Unmarshal(raw, &m) != nil || m.Version != 1 || len(m.Students) == 0 {
		return m, fmt.Errorf("材料清单无效或没有可批改学生")
	}
	root, err := canonical(filepath.Dir(path))
	if err != nil {
		return m, err
	}
	seen := map[string]bool{}
	for _, student := range m.Students {
		if student.ID == "" || seen[student.ID] || len(student.Files) == 0 {
			return m, fmt.Errorf("学生分组无效或重复")
		}
		seen[student.ID] = true
		for _, file := range student.Files {
			rel, err := safeRelative(file.Path)
			if err != nil || !strings.HasPrefix(filepath.ToSlash(rel), "students/") {
				return m, fmt.Errorf("材料路径无效")
			}
			full, err := canonical(filepath.Join(root, rel))
			if err != nil || !beneath(root, full) {
				return m, fmt.Errorf("材料越界")
			}
			actual, err := HashFile(full)
			if err != nil || actual != file.SHA256 {
				return m, fmt.Errorf("学生 %s 的文件已改变，请重新导入", student.ID)
			}
		}
	}
	return m, nil
}
