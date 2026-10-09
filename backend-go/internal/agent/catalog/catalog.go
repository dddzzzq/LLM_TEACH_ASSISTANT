// Package catalog owns immutable Skill documents independently of the Agent framework.
package catalog

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/goccy/go-yaml"
)

type Document struct {
	Name         string `yaml:"name" json:"name"`
	Description  string `yaml:"description" json:"description"`
	Version      string `json:"version"`
	Release      string `json:"release,omitempty"`
	Instructions string `json:"instructions"`
}
type Catalog map[string]Document

func (c Catalog) Names() []string {
	out := make([]string, 0, len(c))
	for n := range c {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}
func (c Catalog) Summary() string {
	var b strings.Builder
	for _, n := range c.Names() {
		d := c[n]
		fmt.Fprintf(&b, "- %s: %s\n", n, d.Description)
	}
	return b.String()
}

// Load scans only immediate package directories and never executes package contents.
func Load(root, runtimeName string) (Catalog, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	out := Catalog{}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		dir := filepath.Join(root, entry.Name())
		p := filepath.Join(dir, "SKILL.md")
		data, err := os.ReadFile(p)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return nil, err
		}
		text := strings.ReplaceAll(string(data), "\r\n", "\n")
		parts := strings.SplitN(text, "\n---\n", 2)
		if !strings.HasPrefix(text, "---\n") || len(parts) != 2 {
			return nil, fmt.Errorf("%s 缺少 Skill 元数据", entry.Name())
		}
		var d Document
		if err := yaml.Unmarshal([]byte(strings.TrimPrefix(parts[0], "---\n")), &d); err != nil {
			return nil, err
		}
		d.Instructions = strings.TrimSpace(parts[1])
		if d.Name != entry.Name() || d.Description == "" || d.Instructions == "" {
			return nil, fmt.Errorf("%s Skill 名称或正文无效", entry.Name())
		}
		var meta struct {
			Version  string   `yaml:"version"`
			Runtimes []string `yaml:"runtimes"`
		}
		raw, err := os.ReadFile(filepath.Join(dir, "skill.yaml"))
		if err == nil {
			if err = yaml.Unmarshal(raw, &meta); err != nil {
				return nil, err
			}
		} else if !os.IsNotExist(err) {
			return nil, err
		}
		if len(meta.Runtimes) == 0 {
			meta.Runtimes = []string{"teaching-assistant"}
			if d.Name == "fetch-homework" {
				meta.Runtimes = []string{"portal-browser"}
			}
		}
		match := runtimeName == ""
		for _, r := range meta.Runtimes {
			if r == runtimeName {
				match = true
			}
		}
		if !match {
			continue
		}
		hash := sha256.New()
		err = filepath.WalkDir(dir, func(path string, e os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if e.Type()&os.ModeSymlink != 0 {
				return fmt.Errorf("Skill 不允许符号链接: %s", path)
			}
			if e.IsDir() {
				return nil
			}
			content, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			rel, _ := filepath.Rel(dir, path)
			fmt.Fprintf(hash, "%s\x00%d\x00", rel, len(content))
			hash.Write(content)
			return nil
		})
		if err != nil {
			return nil, err
		}
		d.Version = hex.EncodeToString(hash.Sum(nil))
		d.Release = meta.Version
		out[d.Name] = d
	}
	return out, nil
}
func Default(runtimeName string) (Catalog, error) {
	if root := os.Getenv("TEACH_SKILLS_DIR"); root != "" {
		return Load(root, runtimeName)
	}
	_, source, _, _ := runtime.Caller(0)
	candidates := []string{"skills", "../skills", filepath.Join(filepath.Dir(source), "../../../../skills")}
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Join(filepath.Dir(exe), "../skills"))
	}
	for _, root := range candidates {
		if info, err := os.Stat(root); err == nil && info.IsDir() {
			return Load(root, runtimeName)
		}
	}
	return nil, fmt.Errorf("未找到 Skill 目录，请配置 TEACH_SKILLS_DIR")
}
