package tools

import (
	"path/filepath"
	"strings"
)

func IsIgnoredFile(filePath string) bool {
	filePath = filepath.ToSlash(filePath)
	parts := strings.Split(filePath, "/")
	filename := filepath.Base(filePath)

	for _, part := range parts {
		if ignoredDirs[part] {
			return true
		}
	}
	if strings.HasPrefix(filename, ".") {
		return true
	}

	ext := strings.ToLower(filepath.Ext(filename))
	if !allowedExtensions[ext] {
		return true
	}

	// 过滤系统生成的无关文件
	if ext == ".doc" || ext == ".docx" {
		if strings.Count(filename, "-") >= 3 && strings.Contains(filename, "学院") {
			return true
		}
	}
	return false
}

func IsLikelyText(content []byte) bool {
	if len(content) == 0 {
		return true
	}
	nonPrintable := 0
	checkLen := len(content)
	if checkLen > 1000 {
		checkLen = 1000
	}

	for _, b := range content[:checkLen] {
		if b < 32 && b != '\n' && b != '\r' && b != '\t' {
			nonPrintable++
		}
	}
	return float64(nonPrintable)/float64(checkLen) < 0.2
}
