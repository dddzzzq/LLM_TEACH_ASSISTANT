// Package portal manages teacher-owned portal connections and catalog workflows.
package portal

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type CredentialCipher struct {
	keyID string
	aead  cipher.AEAD
}
type Credentials struct {
	Username string `json:"username"`
	Password string `json:"password"`
}
type CredentialBinding struct {
	OwnerID                uint
	ConnectionID, Provider string
	Version                uint64
}

func NewCredentialCipher(keyID, encodedKey string) (*CredentialCipher, error) {
	keyID = strings.TrimSpace(keyID)
	if keyID == "" || len(keyID) > 64 {
		return nil, errors.New("教务密钥编号无效")
	}
	key, err := base64.StdEncoding.Strict().DecodeString(encodedKey)
	if err != nil {
		return nil, errors.New("教务密钥必须为有效 Base64")
	}
	defer clear(key)
	if len(key) != 32 {
		return nil, errors.New("教务密钥必须为 32 字节")
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	return &CredentialCipher{keyID, aead}, nil
}

func (b CredentialBinding) aad(keyID string) ([]byte, error) {
	if b.OwnerID == 0 || b.ConnectionID == "" || b.Provider == "" || b.Version == 0 {
		return nil, errors.New("凭据归属信息不完整")
	}
	return json.Marshal([]any{"portal.credentials.v1", keyID, b.OwnerID, b.ConnectionID, b.Provider, b.Version})
}

func (c *CredentialCipher) Encrypt(b CredentialBinding, value Credentials) ([]byte, []byte, error) {
	if value.Username == "" || value.Password == "" || len(value.Username) > 255 || len(value.Password) > 4096 {
		return nil, nil, errors.New("教务账号或密码无效")
	}
	aad, err := b.aad(c.keyID)
	if err != nil {
		return nil, nil, err
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, nil, err
	}
	defer clear(raw)
	nonce := make([]byte, c.aead.NonceSize())
	if _, err = rand.Read(nonce); err != nil {
		return nil, nil, err
	}
	return c.aead.Seal(nil, nonce, raw, aad), nonce, nil
}

func (c *CredentialCipher) Decrypt(b CredentialBinding, keyID string, ciphertext, nonce []byte) (Credentials, error) {
	var result Credentials
	if keyID != c.keyID || len(nonce) != c.aead.NonceSize() {
		return result, errors.New("无法解密教务凭据")
	}
	aad, err := b.aad(keyID)
	if err != nil {
		return result, err
	}
	raw, err := c.aead.Open(nil, nonce, ciphertext, aad)
	if err != nil {
		return result, errors.New("无法解密教务凭据")
	}
	defer clear(raw)
	if json.Unmarshal(raw, &result) != nil || result.Username == "" || result.Password == "" {
		return Credentials{}, errors.New("教务凭据格式无效")
	}
	return result, nil
}

func RuntimeRoot() string {
	if root := os.Getenv("RUNTIME_DIR"); root != "" {
		return root
	}
	return filepath.Join("..", "..", "teaching-runtime")
}

// Keys are provisioned by the operator outside the database; never silently replace a missing key.
func LoadCipher() (*CredentialCipher, error) {
	key := os.Getenv("PORTAL_CREDENTIAL_KEY_BASE64")
	if key == "" {
		path := os.Getenv("PORTAL_CREDENTIAL_KEY_FILE")
		if path == "" {
			path = filepath.Join(RuntimeRoot(), ".portal-credential-key")
		}
		info, err := os.Lstat(path)
		if err != nil {
			return nil, errors.New("尚未配置教务凭据密钥，请按部署说明生成密钥文件")
		}
		if !info.Mode().IsRegular() || info.Mode().Perm()&0077 != 0 {
			return nil, errors.New("教务密钥文件必须为仅运行用户可读写的普通文件（0600）")
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, errors.New("教务凭据密钥不可用")
		}
		defer clear(raw)
		key = strings.TrimSpace(string(raw))
	}
	id := os.Getenv("PORTAL_CREDENTIAL_KEY_ID")
	if id == "" {
		id = "portal-v1"
	}
	c, err := NewCredentialCipher(id, key)
	if err != nil {
		return nil, fmt.Errorf("教务密钥配置错误: %w", err)
	}
	return c, nil
}
