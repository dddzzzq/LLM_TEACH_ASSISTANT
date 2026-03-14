package tools

import (
	"archive/zip"
	"bytes"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"

	"github.com/ledongthuc/pdf"
	"github.com/nwaples/rardecode"
)

// ---------------- 解析功能函数 ----------------

func parseZip(data []byte, depth int) string {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return fmt.Sprintf("【ZIP读取失败: %v】", err)
	}

	var merged strings.Builder
	for _, f := range zr.File {
		if f.FileInfo().IsDir() || IsIgnoredFile(f.Name) {
			continue
		}

		rc, err := f.Open()
		if err != nil {
			continue
		}
		content, _ := io.ReadAll(rc)
		rc.Close()

		text := ExtractContent(f.Name, content, depth)
		if strings.TrimSpace(text) != "" {
			merged.WriteString(fmt.Sprintf("--- 文件开始: %s ---\n%s\n--- 文件结束: %s ---\n\n", f.Name, text, f.Name))
		}
	}
	return merged.String()
}

func parseRar(data []byte, depth int) string {
	rr, err := rardecode.NewReader(bytes.NewReader(data), "")
	if err != nil {
		return fmt.Sprintf("【RAR读取失败: %v】", err)
	}

	var merged strings.Builder
	for {
		header, err := rr.Next()
		if err == io.EOF {
			break
		}
		if header.IsDir || IsIgnoredFile(header.Name) {
			continue
		}

		buf := new(bytes.Buffer)
		_, err = io.Copy(buf, rr)
		if err != nil {
			continue
		}

		text := ExtractContent(header.Name, buf.Bytes(), depth)
		if strings.TrimSpace(text) != "" {
			merged.WriteString(fmt.Sprintf("--- 文件开始: %s ---\n%s\n--- 文件结束: %s ---\n\n", header.Name, text, header.Name))
		}
	}
	return merged.String()
}

func parseDocx(data []byte, filename string) string {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return fmt.Sprintf("【Docx解析失败: %s】", filename)
	}

	for _, f := range zr.File {
		if f.Name == "word/document.xml" {
			rc, _ := f.Open()
			content, _ := io.ReadAll(rc)
			rc.Close()

			re := regexp.MustCompile(`<w:t[^>]*>(.*?)</w:t>`)
			matches := re.FindAllStringSubmatch(string(content), -1)

			var sb strings.Builder
			for _, m := range matches {
				if len(m) > 1 {
					sb.WriteString(m[1] + "\n")
				}
			}
			return sb.String()
		}
	}
	return ""
}

func parsePDF(data []byte, filename string) string {
	// 针对 PDF 的临时文件解析策略（库要求）
	tmpFile, err := os.CreateTemp("", "go-pdf-*.pdf")
	if err != nil {
		return ""
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	tmpFile.Write(data)
	tmpFile.Close()

	f, r, err := pdf.Open(tmpPath)
	if err != nil {
		return fmt.Sprintf("【PDF解析失败: %s】", filename)
	}
	defer f.Close()

	var sb strings.Builder
	b, err := r.GetPlainText()
	if err == nil {
		buf := new(bytes.Buffer)
		buf.ReadFrom(b)
		sb.WriteString(buf.String())
	}
	return sb.String()
}
