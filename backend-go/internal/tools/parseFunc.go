package tools

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

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
		return documentParseFailure(filename, err)
	}

	for _, f := range zr.File {
		if f.Name == "word/document.xml" {
			rc, err := f.Open()
			if err != nil {
				return documentParseFailure(filename, err)
			}
			defer rc.Close()
			text, err := extractWordText(rc)
			if err != nil {
				return documentParseFailure(filename, err)
			}
			return text
		}
	}
	return documentParseFailure(filename, fmt.Errorf("word/document.xml is missing"))
}

func extractWordText(reader io.Reader) (string, error) {
	decoder := xml.NewDecoder(reader)
	var text strings.Builder
	for {
		token, err := decoder.Token()
		if err == io.EOF {
			return strings.TrimSpace(text.String()), nil
		}
		if err != nil {
			return "", err
		}
		switch element := token.(type) {
		case xml.StartElement:
			if !isWordNamespace(element.Name.Space) {
				continue
			}
			switch element.Name.Local {
			case "pPr", "rPr":
				if err := decoder.Skip(); err != nil {
					return "", err
				}
			case "t":
				var value string
				if err := decoder.DecodeElement(&value, &element); err != nil {
					return "", err
				}
				text.WriteString(value)
			case "tab":
				text.WriteByte('\t')
			case "br", "cr":
				text.WriteByte('\n')
			}
		case xml.EndElement:
			if isWordNamespace(element.Name.Space) && element.Name.Local == "p" {
				text.WriteByte('\n')
			}
		}
	}
}

func isWordNamespace(namespace string) bool {
	return namespace == "http://schemas.openxmlformats.org/wordprocessingml/2006/main" ||
		namespace == "http://purl.oclc.org/ooxml/wordprocessingml/main" ||
		namespace == "http://schemas.microsoft.com/office/word/2003/wordml"
}

func parseDoc(data []byte, filename string) string {
	// Some .doc submissions are Word 2003 XML rather than binary Word files.
	content := bytes.TrimSpace(bytes.TrimPrefix(data, []byte{0xef, 0xbb, 0xbf}))
	if bytes.HasPrefix(content, []byte("<")) {
		text, err := extractWordText(bytes.NewReader(content))
		if err != nil {
			return documentParseFailure(filename, err)
		}
		return text
	}
	if !bytes.HasPrefix(data, []byte{0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1}) {
		return documentParseFailure(filename, fmt.Errorf("unsupported .doc format"))
	}

	file, err := os.CreateTemp("", "teaching-doc-*.doc")
	if err != nil {
		return documentParseFailure(filename, err)
	}
	defer os.Remove(file.Name())
	_, writeErr := file.Write(data)
	closeErr := file.Close()
	if writeErr != nil {
		return documentParseFailure(filename, writeErr)
	}
	if closeErr != nil {
		return documentParseFailure(filename, closeErr)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	output, err := exec.CommandContext(ctx, "antiword", "-m", "UTF-8.txt", file.Name()).Output()
	if err != nil {
		return documentParseFailure(filename, fmt.Errorf("antiword conversion failed: %w", err))
	}
	return strings.TrimSpace(string(output))
}

func documentParseFailure(filename string, err error) string {
	log.Printf("[文档解析失败] 文件: %s, 原因: %v", filename, err)
	return fmt.Sprintf("【文档解析失败: %s】", filename)
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
