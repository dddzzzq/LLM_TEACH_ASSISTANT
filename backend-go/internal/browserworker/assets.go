// Package browserworker implements browser.v1 using Go Playwright. It does not
// call a model or load navigation Skills; those belong to internal/browseragent.
package browserworker

import (
	"bytes"
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// The DOM scripts retain the existing execution provider's semantic inventory
// and catalog extraction. They execute in the page, never in a Python process.
//
//go:embed assets/*
var assets embed.FS

type object = map[string]any

var constants = func() map[string]string {
	b, _ := assets.ReadFile("assets/constants.json")
	var v map[string]string
	if err := json.Unmarshal(b, &v); err != nil {
		panic(err)
	}
	return v
}()

func script(name string) string {
	b, err := assets.ReadFile("assets/" + name + ".js")
	if err != nil {
		panic(err)
	}
	return string(b)
}
func str(v any) string { s, _ := v.(string); return s }
func num(v any) float64 {
	switch v := v.(type) {
	case float64:
		return v
	case int:
		return float64(v)
	case int64:
		return float64(v)
	case uint64:
		return float64(v)
	case json.Number:
		n, _ := v.Float64()
		return n
	}
	return 0
}
func flag(v any) bool { b, _ := v.(bool); return b }
func obj(v any) object {
	if o, ok := v.(map[string]any); ok && o != nil {
		return o
	}
	return object{}
}
func array(v any) []any {
	if a, ok := v.([]any); ok {
		return a
	}
	return []any{}
}
func objects(v any) []object {
	a := []object{}
	for _, e := range array(v) {
		if o, ok := e.(map[string]any); ok {
			a = append(a, o)
		}
	}
	return a
}
func clone(v object) object {
	b, _ := json.Marshal(v)
	var out object
	_ = json.Unmarshal(b, &out)
	return out
}
func hash(v any) string {
	b, _ := json.Marshal(v)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}
func textHash(s string) string   { h := sha256.Sum256([]byte(s)); return hex.EncodeToString(h[:]) }
func textDigest(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }

// The Python worker persisted json.dumps(sort_keys=True, ensure_ascii=False)
// digests, whose separators include spaces. Preserve those durable receipts
// when a scheduler retries the same action after switching execution providers.
func pythonActionDigest(payload object) string {
	var encoded bytes.Buffer
	encoder := json.NewEncoder(&encoded)
	encoder.SetEscapeHTML(false)
	if encoder.Encode(payload) != nil {
		return ""
	}
	var output bytes.Buffer
	quoted, escaped := false, false
	for _, c := range bytes.TrimSuffix(encoded.Bytes(), []byte{'\n'}) {
		output.WriteByte(c)
		if quoted {
			if escaped {
				escaped = false
			} else if c == '\\' {
				escaped = true
			} else if c == '"' {
				quoted = false
			}
		} else if c == '"' {
			quoted = true
		} else if c == ',' || c == ':' {
			output.WriteByte(' ')
		}
	}
	return textDigest(output.Bytes())
}
func parseSize(s string) int64          { n, _ := strconv.ParseInt(s, 10, 64); return n }
func jsonMarshal(v any) ([]byte, error) { return json.Marshal(v) }
func bounded(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	return string([]rune(s)[:n])
}
func contains(s string, words ...string) bool {
	for _, w := range words {
		if strings.Contains(s, w) {
			return true
		}
	}
	return false
}
func now() float64 { return float64(time.Now().UnixMilli()) / 1000 }
func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
func publicURL(value string) string {
	u, e := url.Parse(value)
	if e != nil || u.Hostname() == "" || (u.Scheme != "http" && u.Scheme != "https") {
		return ""
	}
	u.User = nil
	u.RawQuery = ""
	u.Fragment = ""
	return u.String()
}
func allowedURL(value string) bool {
	if value == "about:blank" || value == "about:srcdoc" {
		return true
	}
	u, e := url.Parse(value)
	if e != nil || u.User != nil || (u.Scheme != "http" && u.Scheme != "https") {
		return false
	}
	h := strings.ToLower(u.Hostname())
	for _, d := range []string{"xidian.edu.cn", "chaoxing.com", "chaoxing.com.cn"} {
		if h == d || strings.HasSuffix(h, "."+d) {
			return true
		}
	}
	return false
}
func portalURL(value string) bool {
	u, e := url.Parse(value)
	return e == nil && u.Scheme == "https" && u.User == nil && u.Port() == "" && u.Fragment == "" && resourceURL(value) && !regexp.MustCompile(`(?i)login|logout|delete|publish`).MatchString(u.Path)
}
func queryID(value string, names ...string) string {
	u, e := url.Parse(value)
	if e != nil {
		return ""
	}
	for _, n := range names {
		for k, v := range u.Query() {
			if strings.EqualFold(k, n) && len(v) > 0 && v[0] != "" {
				return v[0]
			}
		}
	}
	return ""
}
func safeName(s string) string {
	s = regexp.MustCompile(`[\x00-\x1f<>:"/\\|?*]`).ReplaceAllString(s, "_")
	s = strings.Trim(s, " .")
	if s == "" {
		s = "download.zip"
	}
	return bounded(s, 180)
}
func atomicJSON(path string, value any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	b, e := json.Marshal(value)
	if e != nil {
		return e
	}
	f, e := os.CreateTemp(filepath.Dir(path), ".journal-*")
	if e != nil {
		return e
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if e = f.Chmod(0600); e == nil {
		_, e = f.Write(b)
	}
	if e == nil {
		e = f.Sync()
	}
	closeErr := f.Close()
	if e == nil {
		e = closeErr
	}
	if e != nil {
		return e
	}
	return os.Rename(tmp, path)
}

type issue struct {
	Code, Message string
	Evidence      object
}

func (e *issue) Error() string           { return e.Message }
func problem(code, message string) error { return &issue{Code: code, Message: message} }
func publicError(err error) object {
	var e *issue
	if errors.As(err, &e) {
		return object{"code": e.Code, "message": e.Message, "evidence": e.Evidence}
	}
	return object{"code": "PAGE_OPERATION_FAILED", "message": "页面操作未完成，请检查当前浏览器后重试"}
}
func validateAction(a object) error {
	var schema struct {
		OneOf []struct{ Properties map[string]json.RawMessage }
	}
	b, _ := assets.ReadFile("assets/actions.json")
	_ = json.Unmarshal(b, &schema)
	for _, v := range schema.OneOf {
		var name struct{ Enum []string }
		_ = json.Unmarshal(v.Properties["tool"], &name)
		if len(name.Enum) != 1 || name.Enum[0] != str(a["tool"]) {
			continue
		}
		if len(a) != len(v.Properties) {
			return problem("INVALID_ACTION", "动作参数与工具协议不一致")
		}
		for k, raw := range v.Properties {
			var field struct{ Type string }
			_ = json.Unmarshal(raw, &field)
			x, ok := a[k]
			if !ok {
				return problem("INVALID_ACTION", "动作缺少参数")
			}
			if field.Type == "string" {
				s, ok := x.(string)
				if !ok || (k != "value" && k != "term" && strings.TrimSpace(s) == "") {
					return problem("INVALID_ACTION", "动作参数类型或内容无效")
				}
			}
			if field.Type == "integer" {
				n, ok := x.(float64)
				if !ok || n != float64(int64(n)) {
					return problem("INVALID_ACTION", "动作参数必须是整数")
				}
			}
		}
		return nil
	}
	return fmt.Errorf("不支持的浏览器动作")
}
