package portal

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"grading-gateway/internal/models"
	"strings"
	"testing"
)

func TestCatalogAcceptsVerifiedIDsWithoutDirectLinksButNeverPendingReferences(t *testing.T) {
	value := fixtureCatalog()
	value.Courses[0].URL = ""
	value.Courses[0].Homeworks[0].URL = ""
	if err := validateCatalog(value); err != nil {
		t.Fatal(err)
	}
	value.Courses[0].PlatformID = "pending:local-observation-only"
	if err := validateCatalog(value); err == nil {
		t.Fatal("published an unresolved course reference as a platform ID")
	}
	value = fixtureCatalog()
	value.Courses[0].URL = "https://untrusted.example/course?courseid=11"
	if err := validateCatalog(value); err == nil {
		t.Fatal("accepted an untrusted nonempty link")
	}
}

func testCipher(t *testing.T) *CredentialCipher {
	t.Helper()
	c, err := NewCredentialCipher("test-v1", base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{0x45}, 32)))
	if err != nil {
		t.Fatal(err)
	}
	return c
}
func TestCredentialCipherBindsOwnerConnectionProviderVersionAndKey(t *testing.T) {
	c := testCipher(t)
	binding := CredentialBinding{7, "connection-1", "xidian", 1}
	input := Credentials{"teacher-123", "test-secret-password"}
	ciphertext, nonce, err := c.Encrypt(binding, input)
	if err != nil {
		t.Fatal(err)
	}
	value, err := c.Decrypt(binding, c.keyID, ciphertext, nonce)
	if err != nil || value != input {
		t.Fatalf("roundtrip: %v", err)
	}
	next, nextNonce, err := c.Encrypt(binding, input)
	if err != nil || bytes.Equal(nextNonce, nonce) || bytes.Equal(next, ciphertext) {
		t.Fatal("nonce reused")
	}
	for _, change := range []CredentialBinding{{8, binding.ConnectionID, binding.Provider, 1}, {7, "other", binding.Provider, 1}, {7, binding.ConnectionID, "other", 1}, {7, binding.ConnectionID, binding.Provider, 2}} {
		if _, err := c.Decrypt(change, c.keyID, ciphertext, nonce); err == nil {
			t.Fatal("accepted reassigned ciphertext")
		}
	}
	for _, bad := range [][]byte{nil, nonce[:1], append([]byte{nonce[0] ^ 1}, nonce[1:]...)} {
		if _, err := c.Decrypt(binding, c.keyID, ciphertext, bad); err == nil {
			t.Fatal("accepted invalid nonce")
		}
	}
	mutated := append([]byte(nil), ciphertext...)
	mutated[0] ^= 1
	if _, err := c.Decrypt(binding, c.keyID, mutated, nonce); err == nil {
		t.Fatal("accepted tampering")
	}
	if _, err := c.Decrypt(binding, "other-key", ciphertext, nonce); err == nil {
		t.Fatal("accepted wrong key ID")
	}
	other, _ := NewCredentialCipher(c.keyID, base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{0x12}, 32)))
	if _, err := other.Decrypt(binding, c.keyID, ciphertext, nonce); err == nil {
		t.Fatal("accepted wrong key")
	}
	raw, _ := json.Marshal(models.PortalConnection{CredentialCiphertext: ciphertext, CredentialNonce: nonce, CredentialKeyID: c.keyID, VerifiedSubject: input.Username})
	for _, secret := range []string{"credential", input.Username, input.Password, base64.StdEncoding.EncodeToString(ciphertext)} {
		if strings.Contains(strings.ToLower(string(raw)), secret) {
			t.Fatal("serialized authentication secret")
		}
	}
}
func TestCredentialKeyValidation(t *testing.T) {
	for _, value := range []string{"", "not-base64", base64.StdEncoding.EncodeToString(make([]byte, 16)), base64.StdEncoding.EncodeToString(make([]byte, 24)), base64.StdEncoding.EncodeToString(make([]byte, 33))} {
		if _, err := NewCredentialCipher("v1", value); err == nil {
			t.Fatal("accepted invalid key")
		}
	}
	for _, id := range []string{"", " ", strings.Repeat("a", 65)} {
		if _, err := NewCredentialCipher(id, base64.StdEncoding.EncodeToString(make([]byte, 32))); err == nil {
			t.Fatal("accepted invalid key ID")
		}
	}
}
func TestCatalogRejectsIncompleteAmbiguousAndUnsafeTargets(t *testing.T) {
	good := Catalog{Complete: true, Courses: []CatalogCourse{{PlatformID: "11:22", Title: "同名课程", URL: "https://mooc1.chaoxing.com/course?courseid=11&clazzid=22", Classes: []CatalogClass{{PlatformID: "22", Name: "A班"}}, Homeworks: []CatalogHomework{{PlatformID: "33", Title: "作业", URL: "https://mooc1.chaoxing.com/work?workid=33&courseid=11"}}}}}
	if err := validateCatalog(good); err != nil {
		t.Fatal(err)
	}
	for _, raw := range []string{"http://mooc1.chaoxing.com/work?workid=1", "https://evilchaoxing.com/work?workid=1", "https://mooc1.chaoxing.com/work?access_token=secret", "https://user:pass@mooc1.chaoxing.com/work", "https://mooc1.chaoxing.com/logout"} {
		if targetURL(raw) {
			t.Fatal("unsafe catalog URL")
		}
	}
	bad := good
	bad.Complete = false
	if validateCatalog(bad) == nil {
		t.Fatal("accepted partial catalog")
	}
	bad = good
	bad.Courses = append(append([]CatalogCourse(nil), good.Courses...), good.Courses[0])
	if validateCatalog(bad) == nil {
		t.Fatal("accepted duplicate identity")
	}
}
