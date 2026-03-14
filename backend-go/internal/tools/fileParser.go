package tools

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"

	"grading-gateway/internal/grpcclient"
	"grading-gateway/pb"

	"golang.org/x/text/encoding/simplifiedchinese"
)

// 定义允许和过滤的扩展名
var ignoredDirs = map[string]bool{
	"__MACOSX": true, "node_modules": true, ".git": true, "venv": true,
	"build": true, "dist": true, "target": true, "bin": true, "obj": true,
	"Source": true,
}
var allowedExtensions = map[string]bool{
	".py": true, ".java": true, ".c": true, ".cpp": true, ".h": true,
	".js": true, ".ts": true, ".vue": true, ".html": true,
	".go": true, ".rs": true, ".sql": true, ".sh": true,
	".zip": true, ".rar": true, ".7z": true,
	".txt": true, ".md": true, ".docx": true, ".pdf": true,
	".png": true, ".jpg": true, ".jpeg": true,
}

// 递归处理文件字节流，实现深度解压、格式转译与 OCR 路由
func ExtractContent(filename string, content []byte, depth int) string {
	if len(content) == 0 {
		return ""
	}
	lowerFilename := strings.ToLower(filename)
	ext := filepath.Ext(lowerFilename)

	// 1. 拦截嵌套压缩包
	if ext == ".zip" {
		log.Printf("Go 解析嵌套 ZIP: %s", filename)
		return parseZip(content, depth+1)
	}
	if ext == ".rar" {
		log.Printf("Go 解析嵌套 RAR: %s", filename)
		return parseRar(content, depth+1)
	}

	// 2. 解析常见文档格式
	if ext == ".docx" || bytes.HasPrefix(content, []byte("PK\x03\x04")) {
		return parseDocx(content, filename)
	}
	if ext == ".pdf" || bytes.HasPrefix(content, []byte("%PDF")) {
		return parsePDF(content, filename)
	}

	// 3. 路由图片给 Python 进行 OCR
	if ext == ".png" || ext == ".jpg" || ext == ".jpeg" {
		log.Printf("拦截到图片 %s，发起 RPC调用Python进行OCR...", filename)
		// 设定 OCR 提取超时时间 60 秒
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		res, err := grpcclient.Client.ExtractText(ctx, &pb.ExtractRequest{
			Filename:    filename,
			FileContent: content,
		})

		if err != nil {
			log.Printf("图片 %s OCR 失败: %v", filename, err)
			return fmt.Sprintf("【图片OCR失败: %s】", filename)
		}
		return fmt.Sprintf("--- [图片文件内容 (OCR): %s] ---\n%s\n", filename, res.TextContent)
	}

	// 4. 纯文本及代码的强制解码
	if !IsLikelyText(content) {
		log.Printf("丢弃乱码/二进制文件: %s", filename)
		return ""
	}

	if utf8.Valid(content) {
		return string(content)
	}

	// 尝试 GBK 解码
	decoder := simplifiedchinese.GB18030.NewDecoder()
	decoded, err := decoder.Bytes(content)
	if err == nil {
		return string(decoded)
	}

	return string(content)
}
