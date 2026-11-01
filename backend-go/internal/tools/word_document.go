package tools

import (
	"archive/zip"
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"path"
	"strings"
)

// Preserve paragraph/table order and insert OCR at each drawing's location.
func readWordDocument(ctx context.Context, zr *zip.Reader, readImage requirementImageReader, maxImages int) (string, []string) {
	entries := map[string]*zip.File{}
	for _, f := range zr.File {
		entries[f.Name] = f
	}
	read := func(name string) ([]byte, error) {
		f := entries[name]
		if f == nil || f.UncompressedSize64 > 20<<20 {
			return nil, fmt.Errorf("missing or oversized Word part")
		}
		r, err := f.Open()
		if err != nil {
			return nil, err
		}
		defer r.Close()
		return io.ReadAll(io.LimitReader(r, (20<<20)+1))
	}
	warnings := []string{}
	seen := map[string]string{}
	imageCount := 0
	imageText := func(name string) string {
		if value, ok := seen[name]; ok {
			return value
		}
		imageCount++
		label := fmt.Sprintf("[Word 内嵌图片 %d：%s]", imageCount, path.Base(name))
		value := label + "\n[图片未能读取，需复核]"
		if imageCount > maxImages {
			warnings = append(warnings, "Word 内嵌图片超过读取上限")
		} else if data, err := read(name); err != nil {
			warnings = append(warnings, label+"无法读取")
		} else if text, err := readImage(ctx, path.Base(name), data); err != nil {
			warnings = append(warnings, label+"识别失败，需核对原图")
		} else {
			value = label + "\n" + text
		}
		seen[name] = value
		return value
	}
	parsePart := func(name string) (string, error) {
		data, err := read(name)
		if err != nil {
			return "", err
		}
		rels := map[string]string{}
		if relData, err := read(path.Join(path.Dir(name), "_rels", path.Base(name)+".rels")); err == nil {
			var document struct {
				Relationships []struct {
					ID     string `xml:"Id,attr"`
					Target string `xml:"Target,attr"`
					Mode   string `xml:"TargetMode,attr"`
				} `xml:"Relationship"`
			}
			if err := xml.Unmarshal(relData, &document); err != nil {
				warnings = append(warnings, "Word 图片关联无法解析")
			}
			for _, rel := range document.Relationships {
				target := path.Clean(path.Join(path.Dir(name), strings.TrimPrefix(rel.Target, "/")))
				if strings.HasPrefix(rel.Target, "/word/") {
					target = path.Clean(strings.TrimPrefix(rel.Target, "/"))
				}
				if rel.Mode != "External" && strings.HasPrefix(target, "word/media/") {
					rels[rel.ID] = target
				}
			}
		}
		decoder := xml.NewDecoder(strings.NewReader(string(data)))
		var text strings.Builder
		for {
			token, err := decoder.Token()
			if err == io.EOF {
				break
			}
			if err != nil {
				return "", err
			}
			switch el := token.(type) {
			case xml.StartElement:
				if el.Name.Local == "oMath" || el.Name.Local == "chart" || el.Name.Local == "OLEObject" || el.Name.Local == "relIds" {
					warnings = append(warnings, "Word 包含公式、图表或嵌入对象，文字及图片转写无法完整表达，需核对原文件")
				}
				if el.Name.Local == "t" && strings.Contains(el.Name.Space, "/math") {
					var value string
					if err := decoder.DecodeElement(&value, &el); err != nil {
						return "", err
					}
					text.WriteString(value)
				}
				if isWordNamespace(el.Name.Space) {
					switch el.Name.Local {
					case "pPr", "rPr":
						if err := decoder.Skip(); err != nil {
							return "", err
						}
					case "t":
						var value string
						if err := decoder.DecodeElement(&value, &el); err != nil {
							return "", err
						}
						text.WriteString(value)
					case "tab":
						text.WriteByte('\t')
					case "br", "cr":
						text.WriteByte('\n')
					case "tbl":
						text.WriteString("\n[表格开始]\n")
					}
				}
				if el.Name.Local == "blip" || el.Name.Local == "imagedata" {
					for _, attr := range el.Attr {
						if attr.Name.Local == "embed" || attr.Name.Local == "id" {
							if target, ok := rels[attr.Value]; ok {
								text.WriteString("\n" + imageText(target) + "\n")
							} else {
								warnings = append(warnings, "Word 图片引用无法定位，需核对原图")
							}
						}
					}
				}
			case xml.EndElement:
				if isWordNamespace(el.Name.Space) {
					switch el.Name.Local {
					case "p", "tr":
						text.WriteByte('\n')
					case "tc":
						text.WriteString("\t|\t")
					case "tbl":
						text.WriteString("[表格结束]\n")
					}
				}
			}
		}
		return strings.TrimSpace(text.String()), nil
	}
	text, err := parsePart("word/document.xml")
	if err != nil {
		return "", []string{"Word 正文无法解析"}
	}
	for _, f := range zr.File {
		if strings.HasPrefix(f.Name, "word/") && (strings.HasPrefix(path.Base(f.Name), "header") || strings.HasPrefix(path.Base(f.Name), "footer") || f.Name == "word/footnotes.xml" || f.Name == "word/endnotes.xml") && strings.HasSuffix(f.Name, ".xml") {
			if extra, err := parsePart(f.Name); err == nil {
				text += "\n[Word 附加内容：" + path.Base(f.Name) + "]\n" + extra
			} else {
				warnings = append(warnings, "Word 附加内容未读取："+path.Base(f.Name))
			}
		}
	}
	// Older exported DOCX documents can omit explicit drawing relationships.
	for _, f := range zr.File {
		if strings.HasPrefix(f.Name, "word/media/") && !f.FileInfo().IsDir() {
			if _, ok := seen[f.Name]; !ok {
				text += "\n" + imageText(f.Name)
			}
		}
	}
	return text, warnings
}
