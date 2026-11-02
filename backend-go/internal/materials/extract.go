package materials

import (
	"archive/zip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/bodgit/sevenzip"
	"github.com/nwaples/rardecode"
	"golang.org/x/text/encoding/simplifiedchinese"
)

const maxFiles = 30000
const maxExpanded = int64(8 << 30)
const maxFileSize = int64(256 << 20)
const maxDepth = 6

var errPreparationLimit = errors.New("材料数量或体积超过限制")

var studentName = regexp.MustCompile(`^([0-9]{5,20})(?:[-_\s]*)([^0-9].*)?$`)
var ignored = map[string]bool{".git": true, ".runtime": true, "node_modules": true, "venv": true, ".venv": true, "__macosx": true, "__pycache__": true, "target": true, "dist": true, "build": true}
var codeExt = map[string]bool{".py": true, ".java": true, ".c": true, ".cpp": true, ".cc": true, ".h": true, ".hpp": true, ".js": true, ".mjs": true, ".cjs": true, ".jsx": true, ".ts": true, ".tsx": true, ".vue": true, ".svelte": true, ".html": true, ".css": true, ".go": true, ".rs": true, ".cs": true, ".kt": true, ".swift": true, ".php": true, ".rb": true, ".sql": true, ".sh": true, ".bat": true, ".cmd": true, ".ps1": true, ".ipynb": true, ".m": true, ".r": true}
var reportExt = map[string]bool{".pdf": true, ".doc": true, ".docx": true, ".md": true, ".txt": true}
var otherExt = map[string]bool{".png": true, ".jpg": true, ".jpeg": true, ".webp": true, ".gif": true, ".json": true, ".xml": true, ".yaml": true, ".yml": true, ".toml": true, ".csv": true, ".log": true}

func identity(value string) (string, string) {
	value = strings.TrimSuffix(filepath.Base(value), filepath.Ext(value))
	m := studentName.FindStringSubmatch(value)
	if len(m) == 0 {
		return "", ""
	}
	name := strings.Trim(m[2], "-_ .")
	if name == "" {
		name = "未知"
	}
	return m[1], name
}
func archive(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return ext == ".zip" || ext == ".rar" || ext == ".7z"
}
func kind(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	if ext == "" && strings.ToLower(filepath.Base(filepath.Dir(path))) == "bin" {
		return "code"
	}
	if codeExt[ext] {
		return "code"
	}
	if reportExt[ext] {
		return "report"
	}
	if otherExt[ext] {
		return "other"
	}
	return ""
}

// FileKind is shared with the grading reader so accepted source formats cannot
// disappear between material preparation and model input.
func FileKind(path string) string { return kind(path) }

type preparer struct {
	ctx          context.Context
	root         string
	manifest     Manifest
	students     map[string]*Student
	origins      map[string]string
	originDepth  map[string]int
	conflicts    map[string]bool
	incomplete   map[string]bool
	expanded     int64
	files        int
	archiveChain []string
}

func (p *preparer) student(id, name string) *Student {
	student := p.students[id]
	if student == nil {
		student = &Student{ID: id, Name: name, Files: []File{}, Warnings: []string{}}
		p.students[id] = student
	}
	for _, source := range p.archiveChain {
		found := false
		for _, old := range student.SourceIDs {
			found = found || old == source
		}
		if !found {
			student.SourceIDs = append(student.SourceIDs, source)
		}
	}
	return student
}

func (p *preparer) budgetError() error {
	if p.files > maxFiles {
		return fmt.Errorf("%w：文件处理数量 %d，限额 %d；后续材料尚未处理", errPreparationLimit, p.files, maxFiles)
	}
	if p.expanded > maxExpanded {
		return fmt.Errorf("%w：累计写入超过8GiB；后续材料尚未处理", errPreparationLimit)
	}
	return nil
}

func ignoredPath(path string) bool {
	if strings.HasPrefix(strings.ToLower(filepath.Base(path)), ".env") || protectedPath(path) {
		return true
	}
	for _, part := range strings.Split(filepath.ToSlash(path), "/") {
		if ignored[strings.ToLower(part)] {
			return true
		}
	}
	return false
}

func (p *preparer) issue(path string, err error) {
	if len(p.manifest.Issues) < 1000 {
		p.manifest.Issues = append(p.manifest.Issues, Issue{Path: path, Message: err.Error()})
	}
}
func (p *preparer) copyFile(source, dest string) error {
	if err := p.ctx.Err(); err != nil {
		return err
	}
	info, err := os.Lstat(source)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() > maxFileSize {
		return fmt.Errorf("不支持的文件或单文件超过256MiB")
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0700); err != nil {
		return err
	}
	in, err := os.Open(source)
	if err != nil {
		return err
	}
	defer in.Close()
	actual, err := in.Stat()
	if err != nil || !os.SameFile(info, actual) {
		return fmt.Errorf("扫描期间文件发生变化")
	}
	out, err := os.OpenFile(dest, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	defer out.Close()
	n, err := io.Copy(out, io.LimitReader(in, maxFileSize+1))
	if err != nil {
		return err
	}
	if n > maxFileSize {
		return fmt.Errorf("单文件超过256MiB")
	}
	p.expanded += n
	p.files++
	return p.budgetError()
}
func (p *preparer) visit(path, id, name, prefix string, depth int) (visitErr error) {
	if ignoredPath(path) {
		p.manifest.FilteredEntries++
		return nil
	}
	defer func() {
		if visitErr != nil && id != "" {
			p.incomplete[id] = true
		}
	}()
	if depth > maxDepth {
		return fmt.Errorf("压缩包嵌套超过6层")
	}
	if err := p.ctx.Err(); err != nil {
		return err
	}
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() && !info.IsDir() {
		return fmt.Errorf("不支持符号链接或特殊文件")
	}
	if info.IsDir() {
		return filepath.WalkDir(path, func(full string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if err := p.ctx.Err(); err != nil {
				return err
			}
			if full == path {
				return nil
			}
			if d.IsDir() {
				if ignoredPath(full) || d.Name() == "original" || d.Name() == ".prepare" {
					p.manifest.FilteredEntries++
					return filepath.SkipDir
				}
				return nil
			}
			if d.Type()&os.ModeSymlink != 0 {
				p.manifest.SkippedSymlinks++
				p.issue(d.Name(), fmt.Errorf("已跳过符号链接"))
				return nil
			}
			rel, _ := filepath.Rel(path, full)
			sid, sname := id, name
			if sid == "" {
				for _, part := range strings.Split(filepath.ToSlash(rel), "/") {
					if sid, sname = identity(part); sid != "" {
						break
					}
				}
			}
			if err := p.visit(full, sid, sname, filepath.Join(prefix, rel), depth); err != nil {
				p.issue(rel, err)
				if errors.Is(err, errPreparationLimit) || p.ctx.Err() != nil {
					return err
				}
			}
			return p.budgetError()
		})
	}
	if id == "" {
		id, name = identity(filepath.Base(path))
	}
	if archive(path) {
		archiveID, _ := identity(filepath.Base(path))
		role := "nested_archive"
		if id == "" {
			role = "batch_export"
		} else if archiveID == id && p.origins[id] == "" {
			role = "student_submission"
		}
		if id != "" && archiveID == id {
			// A repeated student archive needs a teacher's choice; never silently select one.
			if previous := p.origins[id]; previous != "" && previous != prefix && depth <= p.originDepth[id] {
				p.conflicts[id] = true
				p.issue(prefix, fmt.Errorf("学号 %s 存在多个提交，请保留需要批改的版本后重新导入", id))
				return nil
			}
			if p.origins[id] == "" {
				p.origins[id] = prefix
				p.originDepth[id] = depth
			}
		}
		tmp, err := os.MkdirTemp(filepath.Join(p.root, ".prepare"), "archive-")
		if err != nil {
			return err
		}
		if err := p.unpack(path, tmp); err != nil {
			return err
		}
		hash, err := HashFile(path)
		if err != nil {
			return err
		}
		parent := ""
		if len(p.archiveChain) > 0 {
			parent = p.archiveChain[len(p.archiveChain)-1]
		}
		digest := sha256.Sum256([]byte(parent + "\x00" + filepath.ToSlash(prefix) + "\x00" + hash))
		sourceID := hex.EncodeToString(digest[:])
		p.manifest.Sources = append(p.manifest.Sources, ArchiveSource{ID: sourceID, Role: role, Name: filepath.Base(path), Format: strings.TrimPrefix(strings.ToLower(filepath.Ext(path)), "."), SHA256: hash, Size: info.Size(), ParentID: parent, EntryPath: filepath.ToSlash(prefix), StudentID: id})
		p.archiveChain = append(p.archiveChain, sourceID)
		defer func() { p.archiveChain = p.archiveChain[:len(p.archiveChain)-1] }()
		base := strings.TrimSuffix(prefix, filepath.Ext(prefix))
		if id == "" {
			base = ""
		}
		return p.visit(tmp, id, name, base, depth+1)
	}
	if kind(path) == "" || strings.HasPrefix(filepath.Base(path), ".") {
		if id != "" && !strings.HasPrefix(filepath.Base(path), ".") {
			p.student(id, name).SkippedFiles = append(p.student(id, name).SkippedFiles, SkippedFile{Path: filepath.ToSlash(prefix), Reason: "格式暂不支持，原始附件保留"})
		}
		return nil
	}
	if id == "" {
		p.issue(prefix, fmt.Errorf("无法从文件名或上级目录识别学号姓名"))
		return nil
	}
	student := p.student(id, name)
	if student.Name != name && name != "未知" {
		p.conflicts[id] = true
		p.issue(prefix, fmt.Errorf("同一学号对应不同姓名"))
		return nil
	}
	rel, err := safeRelative(prefix)
	if err != nil {
		return err
	}
	rel = filepath.Join("students", component(id+"_"+name), rel)
	dest := filepath.Join(p.root, rel)
	if err := p.copyFile(path, dest); err != nil {
		if os.IsExist(err) {
			p.conflicts[id] = true
		}
		return err
	}
	hash, err := HashFile(dest)
	if err != nil {
		return err
	}
	saved, err := os.Stat(dest)
	if err != nil {
		return err
	}
	student.Files = append(student.Files, File{Path: filepath.ToSlash(rel), Kind: kind(path), SHA256: hash, Size: saved.Size()})
	return nil
}

func (p *preparer) unpack(source, dest string) error {
	// Filter before opening ZIP entries or writing any bytes. Dependency trees
	// must not consume the extraction budget before the directory walker sees them.
	entryPath := func(name string, mode os.FileMode) (string, bool, error) {
		if err := p.ctx.Err(); err != nil {
			return "", false, err
		}
		rel, err := safeRelative(name)
		if err != nil {
			return "", false, err
		}
		if ignoredPath(rel) {
			p.manifest.FilteredEntries++
			if mode&os.ModeSymlink != 0 {
				p.manifest.SkippedSymlinks++
			}
			return "", true, nil
		}
		if mode&os.ModeSymlink != 0 || !mode.IsRegular() && !mode.IsDir() {
			p.manifest.SkippedSymlinks++
			p.issue(filepath.Join(filepath.Base(source), rel), fmt.Errorf("已跳过符号链接或特殊文件，未读取链接目标"))
			return "", true, nil
		}
		return rel, false, nil
	}
	write := func(rel string, mode os.FileMode, size int64, reader io.Reader) error {
		if err := p.ctx.Err(); err != nil {
			return err
		}
		if size > maxFileSize {
			return fmt.Errorf("压缩包单文件超过256MiB")
		}
		out := filepath.Join(dest, rel)
		if !beneath(dest, out) {
			return fmt.Errorf("压缩包路径越界")
		}
		if mode.IsDir() {
			return os.MkdirAll(out, 0700)
		}
		if err := os.MkdirAll(filepath.Dir(out), 0700); err != nil {
			return err
		}
		f, err := os.OpenFile(out, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			return fmt.Errorf("压缩包有重复路径或无法写入")
		}
		n, err := io.Copy(f, io.LimitReader(reader, maxFileSize+1))
		f.Close()
		if err != nil {
			return err
		}
		p.expanded += n
		p.files++
		if n > maxFileSize {
			return fmt.Errorf("压缩包单文件超过256MiB")
		}
		return p.budgetError()
	}
	switch strings.ToLower(filepath.Ext(source)) {
	case ".zip":
		z, err := zip.OpenReader(source)
		if err != nil {
			return fmt.Errorf("ZIP损坏或无法打开")
		}
		defer z.Close()
		for _, f := range z.File {
			name := f.Name
			if !utf8.ValidString(name) {
				if decoded, err := simplifiedchinese.GB18030.NewDecoder().String(name); err == nil {
					name = decoded
				}
			}
			rel, skip, err := entryPath(name, f.Mode())
			if err != nil {
				return err
			}
			if skip || f.FileInfo().IsDir() {
				continue
			}
			r, err := f.Open()
			if err != nil {
				return fmt.Errorf("ZIP加密或损坏")
			}
			err = write(rel, f.Mode(), int64(f.UncompressedSize64), r)
			r.Close()
			if err != nil {
				return err
			}
		}
		return nil
	case ".rar":
		r, err := rardecode.OpenReader(source, "")
		if err != nil {
			return fmt.Errorf("RAR加密、损坏或格式不受支持")
		}
		defer r.Close()
		for {
			h, err := r.Next()
			if err == io.EOF {
				return nil
			}
			if err != nil {
				return fmt.Errorf("RAR读取失败")
			}
			mode := h.Mode()
			rel, skip, err := entryPath(h.Name, mode)
			if err != nil {
				return err
			}
			if skip {
				continue
			}
			if err := write(rel, mode, h.UnPackedSize, r); err != nil {
				return err
			}
		}
	case ".7z":
		z, err := sevenzip.OpenReader(source)
		if err != nil {
			return fmt.Errorf("7z加密、损坏或无法打开")
		}
		defer z.Close()
		// Keep archive order and close each entry before opening the next one:
		// solid 7z archives can then reuse their decompression stream.
		for _, f := range z.File {
			rel, skip, err := entryPath(f.Name, f.Mode())
			if err != nil {
				return err
			}
			if skip || f.FileInfo().IsDir() {
				continue
			}
			// Check the unsigned size before converting it or opening a decoder.
			if f.UncompressedSize > uint64(maxFileSize) {
				return fmt.Errorf("压缩包单文件超过256MiB")
			}
			r, err := f.Open()
			if err != nil {
				return fmt.Errorf("7z加密、损坏或压缩算法不受支持")
			}
			size := int64(f.UncompressedSize)
			limited := &io.LimitedReader{R: contextReader{ctx: p.ctx, reader: r}, N: size + 1}
			checksum := crc32.NewIEEE()
			err = write(rel, f.Mode(), size, io.TeeReader(limited, checksum))
			closeErr := r.Close()
			if err != nil {
				var readErr *sevenzip.ReadError
				if errors.As(err, &readErr) {
					return fmt.Errorf("7z读取失败：文件可能已加密、损坏或压缩算法不受支持")
				}
				return err
			}
			if closeErr != nil || size+1-limited.N != size {
				return fmt.Errorf("7z文件不完整或读取失败")
			}
			// The library validates headers but does not validate each file's
			// contents. A zero CRC may mean the archive omitted that checksum.
			if f.CRC32 != 0 && checksum.Sum32() != f.CRC32 {
				return fmt.Errorf("7z文件校验失败：压缩包损坏或需要密码")
			}
		}
		return nil
	}
	return fmt.Errorf("不支持的压缩格式")
}

func prepare(ctx context.Context, batchID string, assignmentID uint, source, root string) (Manifest, error) {
	p := &preparer{ctx: ctx, root: root, manifest: Manifest{Version: 1, ReaderVersion: CurrentReaderVersion, BatchID: batchID, AssignmentID: assignmentID, Students: []Student{}, Issues: []Issue{}}, students: map[string]*Student{}, origins: map[string]string{}, originDepth: map[string]int{}, conflicts: map[string]bool{}, incomplete: map[string]bool{}}
	if err := os.MkdirAll(filepath.Join(root, ".prepare"), 0700); err != nil {
		return p.manifest, err
	}
	// Scratch files live only within this unique batch; originals are never moved.
	defer os.RemoveAll(filepath.Join(root, ".prepare"))
	info, err := os.Stat(source)
	if err != nil {
		return p.manifest, err
	}
	prefix := ""
	if !info.IsDir() {
		prefix = filepath.Base(source)
	}
	id, name := "", ""
	if info.IsDir() {
		id, name = identity(filepath.Base(source))
	}
	visitErr := p.visit(source, id, name, prefix, 0)
	return p.finish(prefix, visitErr)
}

// A failed student's partial files never enter the manifest; complete students
// remain usable even when a later archive exhausts the batch budget.
func (p *preparer) finish(prefix string, visitErr error) (Manifest, error) {
	if visitErr != nil {
		p.issue(prefix, visitErr)
	}
	for id, student := range p.students {
		if len(student.Files) == 0 || p.conflicts[id] || p.incomplete[id] {
			continue
		}
		code, report := false, false
		for _, file := range student.Files {
			code = code || file.Kind == "code"
			report = report || file.Kind == "report"
		}
		if !code {
			student.Warnings = append(student.Warnings, "未发现源代码")
		}
		if !report {
			student.Warnings = append(student.Warnings, "未发现设计报告")
		}
		sort.Slice(student.Files, func(i, j int) bool { return student.Files[i].Path < student.Files[j].Path })
		p.manifest.Students = append(p.manifest.Students, *student)
	}
	sort.Slice(p.manifest.Students, func(i, j int) bool { return p.manifest.Students[i].ID < p.manifest.Students[j].ID })
	if visitErr != nil && (p.ctx.Err() != nil || len(p.manifest.Students) == 0) {
		return p.manifest, visitErr
	}
	if len(p.manifest.Students) == 0 {
		return p.manifest, fmt.Errorf("未找到可用学生材料，请检查学号姓名命名、压缩包和重复提交")
	}
	return p.manifest, nil
}
