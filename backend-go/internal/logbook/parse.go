package logbook

import (
	"crypto/sha256"
	"encoding/hex"
	"net/url"
	"regexp"
	"strings"
	"time"
)

var ansi = regexp.MustCompile(`\x1b\[[0-?]*[ -/]*[@-~]`)
var urls = regexp.MustCompile(`(?:https?|mysql|redis|postgres(?:ql)?)://[^\s"'<>]+`)
var bearer = regexp.MustCompile(`(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+`)
var authHeader = regexp.MustCompile(`(?i)(["']?(?:authorization|set-cookie|cookie)["']?\s*[:=]\s*).*`)
var credentials = regexp.MustCompile(`(?i)(["']?(?:[a-z_]*api[_-]?key|password|passwd|pwd|[a-z_]*secret|access[_-]?token|refresh[_-]?token|token|csrf[_-]?token)["']?\s*[:=]\s*)(?:\[REDACTED\]|"[^"]*"|'[^']*'|[^\s,;}\]]+)`)
var dsn = regexp.MustCompile(`\b([A-Za-z0-9_.-]+):[^\s@]*@tcp\(`)
var apiKey = regexp.MustCompile(`\bsk-[A-Za-z0-9_-]{12,}`)
var jwt = regexp.MustCompile(`\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+`)
var taskIDs = regexp.MustCompile(`(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b`)
var timestamp = regexp.MustCompile(`\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:\d{2})?`)
var levelTag = regexp.MustCompile(`(?i)\b(ERROR|FATAL|CRITICAL|PANIC|WARNING|WARN|INFO|DEBUG)\b`)

func digest(text string) string { b := sha256.Sum256([]byte(text)); return hex.EncodeToString(b[:]) }

// Only sanitized text enters the index, search responses, context and export.
func Redact(text string) string {
	text = ansi.ReplaceAllString(strings.ToValidUTF8(text, "�"), "")
	text = strings.Map(func(r rune) rune {
		if r < 32 && r != '\t' && r != '\n' {
			return -1
		}
		return r
	}, text)
	text = urls.ReplaceAllStringFunc(text, func(raw string) string {
		u, err := url.Parse(raw)
		if err != nil {
			return "[链接已隐藏]"
		}
		return u.Scheme + "://" + u.Host + "/[链接已隐藏]"
	})
	text = bearer.ReplaceAllString(text, "Bearer [REDACTED]")
	text = authHeader.ReplaceAllString(text, "${1}[REDACTED]")
	text = credentials.ReplaceAllString(text, "${1}[REDACTED]")
	text = dsn.ReplaceAllString(text, "${1}:[REDACTED]@tcp(")
	text = apiKey.ReplaceAllString(text, "[REDACTED]")
	return jwt.ReplaceAllString(text, "[REDACTED]")
}

func parseTime(text string) (time.Time, bool) {
	raw := timestamp.FindString(text)
	if raw == "" {
		return time.Time{}, false
	}
	raw = strings.ReplaceAll(strings.ReplaceAll(raw, "/", "-"), ",", ".")
	for _, layout := range []string{time.RFC3339Nano, "2006-01-02 15:04:05.999999999", "2006-01-02T15:04:05.999999999"} {
		if at, err := time.ParseInLocation(layout, raw, time.Local); err == nil {
			return at, true
		}
	}
	return time.Time{}, false
}

func classify(text, previous string) string {
	if level := strings.ToUpper(levelTag.FindString(text)); level != "" {
		switch level {
		case "FATAL", "CRITICAL", "PANIC":
			return "ERROR"
		case "WARNING":
			return "WARN"
		default:
			return level
		}
	}
	if strings.Contains(text, "❌") || strings.Contains(text, "失败") || strings.Contains(text, "Traceback (") {
		return "ERROR"
	}
	if strings.Contains(text, "⚠") || strings.Contains(text, "超时") || strings.Contains(text, "人工") || strings.Contains(text, "UserWarning") {
		return "WARN"
	}
	if previous != "" && (strings.HasPrefix(text, " ") || strings.HasPrefix(text, "\t")) {
		return previous
	}
	return "INFO"
}

func refs(text string) []string {
	result := []string{}
	seen := map[string]bool{}
	for _, id := range taskIDs.FindAllString(strings.ToLower(text), -1) {
		if !seen[id] {
			result = append(result, id)
			seen[id] = true
		}
	}
	return result
}
