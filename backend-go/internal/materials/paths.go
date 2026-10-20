// Package materials prepares immutable homework batches, independently of grading.
package materials

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

var drivePath = regexp.MustCompile(`^[A-Za-z]:[\\/]`)
var badComponent = regexp.MustCompile(`[<>:"/\\|?*\x00-\x1f]`)

func NormalizePath(value string) (string, error) {
	value = strings.TrimSpace(value)
	if value == "" || strings.ContainsRune(value, 0) {
		return "", fmt.Errorf("请提供有效的文件或目录位置")
	}
	if drivePath.MatchString(value) && runtime.GOOS != "windows" {
		value = "/mnt/" + strings.ToLower(value[:1]) + "/" + strings.ReplaceAll(value[3:], `\`, "/")
	}
	if !filepath.IsAbs(value) {
		return "", fmt.Errorf("请使用绝对路径，例如 D:\\dzq\\homework")
	}
	return filepath.Clean(value), nil
}

func beneath(root, path string) bool {
	rel, err := filepath.Rel(root, path)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel)
}

// canonical resolves existing ancestors too, preventing symlink escapes on new outputs.
func canonical(path string) (string, error) {
	resolved, err := filepath.EvalSymlinks(path)
	if err == nil {
		return resolved, nil
	}
	if !os.IsNotExist(err) {
		return "", err
	}
	parent := filepath.Dir(path)
	if parent == path {
		return "", err
	}
	base, err := canonical(parent)
	if err != nil {
		return "", err
	}
	return filepath.Join(base, filepath.Base(path)), nil
}

func (s *Service) allowed(value string) (string, error) {
	path, err := NormalizePath(value)
	if err != nil {
		return "", err
	}
	path, err = canonical(path)
	if err != nil {
		return "", fmt.Errorf("无法读取该路径或其父目录")
	}
	for _, root := range s.AllowedRoots {
		r, err := canonical(root)
		if err == nil && beneath(r, path) {
			return path, nil
		}
	}
	return "", fmt.Errorf("路径不在允许的作业目录内；请使用 D 盘作业目录或由管理员配置 HOMEWORK_ALLOWED_ROOTS")
}

func (s *Service) ValidateOutputRoot(value string) (string, error) {
	if value == "" {
		return s.Root, nil
	}
	return s.allowed(value)
}

func component(value string) string {
	value = strings.Trim(badComponent.ReplaceAllString(strings.TrimSpace(value), "_"), ". ")
	if value == "" {
		value = "未指定"
	}
	runes := []rune(value)
	if len(runes) > 64 {
		value = string(runes[:64])
	}
	switch strings.ToUpper(value) {
	case "CON", "PRN", "AUX", "NUL", "COM1", "LPT1":
		value = "_" + value
	}
	return value
}

func safeRelative(value string) (string, error) {
	value = strings.ReplaceAll(value, `\`, "/")
	if strings.HasPrefix(value, "/") || strings.Contains(value, ":") || strings.ContainsRune(value, 0) {
		return "", fmt.Errorf("压缩包包含无效路径")
	}
	for _, part := range strings.Split(value, "/") {
		if part == ".." {
			return "", fmt.Errorf("压缩包包含越界路径")
		}
	}
	clean := filepath.Clean(filepath.FromSlash(value))
	if clean == "." {
		return "", fmt.Errorf("空文件路径")
	}
	return clean, nil
}
