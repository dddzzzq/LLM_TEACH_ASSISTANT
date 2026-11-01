package materials

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

func (s *Service) OpenEvidence(ctx context.Context, actor Actor, batchID, relative string) (*os.File, File, error) {
	batch, err := s.Owned(ctx, actor, batchID)
	if err != nil {
		return nil, File{}, err
	}
	manifest, err := LoadManifest(batch.ManifestPath, batch.ManifestSHA256)
	if err != nil {
		return nil, File{}, err
	}
	for _, student := range manifest.Students {
		for _, entry := range student.Files {
			if entry.Path != relative {
				continue
			}
			root, err := canonical(filepath.Dir(batch.ManifestPath))
			if err != nil {
				return nil, File{}, err
			}
			full, err := canonical(filepath.Join(root, filepath.FromSlash(entry.Path)))
			if err != nil || !beneath(root, full) {
				return nil, File{}, fmt.Errorf("证据路径无效")
			}
			file, err := os.Open(full)
			if err != nil {
				return nil, File{}, fmt.Errorf("证据无法读取")
			}
			hash := sha256.New()
			_, err = io.Copy(hash, file)
			if err != nil || hex.EncodeToString(hash.Sum(nil)) != entry.SHA256 {
				file.Close()
				return nil, File{}, fmt.Errorf("证据文件已经变化")
			}
			if _, err = file.Seek(0, 0); err != nil {
				file.Close()
				return nil, File{}, err
			}
			return file, entry, nil
		}
	}
	return nil, File{}, fmt.Errorf("材料清单中没有该证据")
}
