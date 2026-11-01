package tools

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// ExtractRequirement reuses document readers and the existing OCR RPC.
// Missing pages/images are explicit incomplete results for both reader paths.
func ExtractRequirement(ctx context.Context, filename string, data []byte) (string, []string) {
	return extractRequirement(ctx, filename, data, requirementOCR)
}

// RequirementImage keeps the original bytes (or a rendered PDF page) for vision.
// It is transient and must not be serialized into the public detail response.
type RequirementImage struct {
	Name string
	Data []byte
}

// ExtractRequirementWithImages reuses the document readers, collecting images
// for the configured model instead of flattening them with OCR. Placeholders
// preserve the association between document text and its figures/pages.
func ExtractRequirementWithImages(ctx context.Context, filename string, data []byte) (string, []RequirementImage, []string) {
	var images []RequirementImage
	var total int
	text, warnings := extractRequirement(ctx, filename, data, func(_ context.Context, name string, image []byte) (string, error) {
		total += len(image)
		if total > 30<<20 || len(images) >= 40 {
			return "", errors.New("requirement images exceed reading limit")
		}
		images = append(images, RequirementImage{Name: name, Data: image})
		return fmt.Sprintf("[原图 %d]", len(images)), nil
	})
	return text, images, warnings
}

type requirementImageReader func(context.Context, string, []byte) (string, error)

type documentLimits struct {
	textBytes, pages, images int
	timeout                  time.Duration
}

var questionDocumentLimits = documentLimits{100000, 30, 20, 90 * time.Second}
var studentDocumentLimits = documentLimits{8 << 20, 200, 200, 5 * time.Minute}

func extractRequirement(ctx context.Context, filename string, data []byte, readImage requirementImageReader) (string, []string) {
	return extractDocument(ctx, filename, data, readImage, questionDocumentLimits)
}

func extractDocument(ctx context.Context, filename string, data []byte, readImage requirementImageReader, limits documentLimits) (string, []string) {
	ctx, cancel := context.WithTimeout(ctx, limits.timeout)
	defer cancel()
	var text string
	var warnings []string
	switch strings.ToLower(filepath.Ext(filename)) {
	case ".png", ".jpg", ".jpeg", ".webp", ".gif":
		value, err := readImage(ctx, filename, data)
		if err != nil {
			return "", []string{"图片未能识别出有效文字，请核对原图"}
		}
		text = value
	case ".docx":
		zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
		if err != nil {
			return "", []string{"Word 文件无法读取"}
		}
		var expanded uint64
		for _, entry := range zr.File {
			expanded += entry.UncompressedSize64
			if expanded > 60<<20 || entry.UncompressedSize64 > 20<<20 {
				return "", []string{"Word 解压内容超过读取上限"}
			}
		}
		text, warnings = readWordDocument(ctx, zr, readImage, limits.images)
	case ".doc":
		text = parseDoc(data, filename)
		// antiword extracts text only; do not silently claim embedded figures were read.
		warnings = append(warnings, "旧版 Word 已提取正文，内嵌图片需要核对原文件；建议使用 DOCX 或 PDF")
	case ".pdf":
		return documentPDF(ctx, data, readImage, limits)
	case ".txt", ".md":
		text = ExtractContent(filename, data, 0)
	default:
		return "", []string{"暂不支持该文档附件格式"}
	}
	text = strings.TrimSpace(text)
	if text == "" || strings.Contains(text, "【文档解析失败:") {
		return "", append(warnings, "文档未提取到有效正文")
	}
	if len(text) > limits.textBytes {
		return "", append(warnings, "文档正文超过读取上限，需复核原文件")
	}
	return text, warnings
}

func requirementOCR(ctx context.Context, filename string, data []byte) (string, error) {
	result, err := extractOCR(ctx, filename, data)
	if err != nil {
		return "", err
	}
	text := strings.TrimSpace(result.TextContent)
	if text == "" || strings.Contains(text, "[未识别到文本]") || strings.Contains(text, "【系统降级】") {
		return "", errors.New("OCR returned no readable content")
	}
	return text, nil
}

func requirementPDF(ctx context.Context, data []byte, readImage requirementImageReader) (string, []string) {
	return documentPDF(ctx, data, readImage, questionDocumentLimits)
}

func documentPDF(ctx context.Context, data []byte, readImage requirementImageReader, limits documentLimits) (string, []string) {
	dir, err := os.MkdirTemp("", "teaching-question-*")
	if err != nil {
		return "", []string{"PDF 临时目录不可用"}
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "question.pdf")
	if os.WriteFile(path, data, 0600) != nil {
		return "", []string{"PDF 文件无法读取"}
	}
	info, err := exec.CommandContext(ctx, "pdfinfo", path).Output()
	if err != nil {
		// Retain the existing text reader as a fallback, reporting lack of OCR coverage.
		return parsePDF(data, "question.pdf"), []string{"PDF 逐页解析不可用，扫描页和内嵌图片尚未核验"}
	}
	pages := 0
	for _, line := range strings.Split(string(info), "\n") {
		if strings.HasPrefix(line, "Pages:") {
			pages, _ = strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, "Pages:")))
		}
	}
	if pages < 1 || pages > limits.pages {
		return "", []string{fmt.Sprintf("PDF 页数无效或超过%d页读取上限", limits.pages)}
	}
	// Image-bearing pages also need OCR when they contain a text layer. This
	// prevents a short caption from hiding a scanned question below it.
	imagePages := map[int]bool{}
	listing, imageErr := exec.CommandContext(ctx, "pdfimages", "-list", path).Output()
	for _, line := range strings.Split(string(listing), "\n") {
		fields := strings.Fields(line)
		if len(fields) > 2 {
			page, _ := strconv.Atoi(fields[0])
			if page > 0 {
				imagePages[page] = true
			}
		}
	}
	var output []string
	var warnings []string
	for page := 1; page <= pages; page++ {
		if ctx.Err() != nil {
			warnings = append(warnings, "PDF 解析已超时或取消，后续页面未读取")
			break
		}
		n := strconv.Itoa(page)
		value, textErr := exec.CommandContext(ctx, "pdftotext", "-enc", "UTF-8", "-f", n, "-l", n, "-layout", path, "-").Output()
		text := strings.TrimSpace(string(value))
		invalidText := strings.ContainsRune(text, '\x00') || strings.ContainsRune(text, '\ufffd')
		if textErr != nil || text == "" || invalidText || imagePages[page] || imageErr != nil {
			prefix := filepath.Join(dir, "page-"+n)
			_, renderErr := exec.CommandContext(ctx, "pdftoppm", "-f", n, "-l", n, "-scale-to", "2400", "-singlefile", "-png", path, prefix).Output()
			image, readErr := os.ReadFile(prefix + ".png")
			if renderErr != nil || readErr != nil || len(image) > 20<<20 {
				warnings = append(warnings, fmt.Sprintf("PDF 第%d页图片未能读取", page))
			} else if ocr, err := readImage(ctx, "page-"+n+".png", image); err != nil {
				warnings = append(warnings, fmt.Sprintf("PDF 第%d页图片未能读取或超过上限，请核对原文件", page))
			} else if text == "" || invalidText {
				text = ocr
			} else {
				text += "\n[页面图像识别，可能与正文重复]\n" + ocr
			}
		}
		if text == "" {
			warnings = append(warnings, fmt.Sprintf("PDF 第%d页未提取到正文", page))
		}
		output = append(output, fmt.Sprintf("[第%d页]\n%s", page, text))
	}
	text := strings.Join(output, "\n\n")
	if len(text) > limits.textBytes {
		return "", append(warnings, "PDF 正文超过读取上限，需复核原文件")
	}
	return text, warnings
}
