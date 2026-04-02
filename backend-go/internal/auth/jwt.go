package auth

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/crypto/bcrypt"
)

// JWT 配置
type JWTConfig struct {
	AccessTokenSecret  string
	RefreshTokenSecret string
	AccessTokenExpiry  time.Duration // 例如 15 分钟
	RefreshTokenExpiry time.Duration // 例如 7 天
}

// 默认配置从环境变量读取
func DefaultConfig() *JWTConfig {
	// 从环境变量读取密钥，如果不存在则使用默认值（仅用于开发）
	accessSecret := os.Getenv("JWT_ACCESS_SECRET")
	if accessSecret == "" {
		accessSecret = "default_access_secret_change_in_production"
	}

	refreshSecret := os.Getenv("JWT_REFRESH_SECRET")
	if refreshSecret == "" {
		refreshSecret = "default_refresh_secret_change_in_production"
	}

	// 从环境变量读取过期时间，单位分钟/天
	accessExpiryMinutes := 15
	if env := os.Getenv("JWT_ACCESS_EXPIRY_MINUTES"); env != "" {
		if val, err := strconv.Atoi(env); err == nil && val > 0 {
			accessExpiryMinutes = val
		}
	}

	refreshExpiryDays := 7
	if env := os.Getenv("JWT_REFRESH_EXPIRY_DAYS"); env != "" {
		if val, err := strconv.Atoi(env); err == nil && val > 0 {
			refreshExpiryDays = val
		}
	}

	return &JWTConfig{
		AccessTokenSecret:  accessSecret,
		RefreshTokenSecret: refreshSecret,
		AccessTokenExpiry:  time.Duration(accessExpiryMinutes) * time.Minute,
		RefreshTokenExpiry: time.Duration(refreshExpiryDays) * 24 * time.Hour,
	}
}

// 自定义 JWT Claims，包含用户ID、用户名和角色
type CustomClaims struct {
	UserID   uint   `json:"user_id"`
	Username string `json:"username"`
	Role     string `json:"role"`
	jwt.RegisteredClaims
}

// 生成访问令牌（Access Token）
func GenerateAccessToken(userID uint, username, role string, config *JWTConfig) (string, error) {
	if config == nil {
		config = DefaultConfig()
	}

	claims := CustomClaims{
		UserID:   userID,
		Username: username,
		Role:     role,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(config.AccessTokenExpiry)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
			Issuer:    "grading-gateway",
			Subject:   strconv.FormatUint(uint64(userID), 10),
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(config.AccessTokenSecret))
}

// 生成刷新令牌（Refresh Token）
func GenerateRefreshToken(userID uint, config *JWTConfig) (string, error) {
	if config == nil {
		config = DefaultConfig()
	}

	claims := jwt.RegisteredClaims{
		ExpiresAt: jwt.NewNumericDate(time.Now().Add(config.RefreshTokenExpiry)),
		IssuedAt:  jwt.NewNumericDate(time.Now()),
		NotBefore: jwt.NewNumericDate(time.Now()),
		Issuer:    "grading-gateway",
		Subject:   strconv.FormatUint(uint64(userID), 10),
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(config.RefreshTokenSecret))
}

// 解析访问令牌
func ParseAccessToken(tokenString string, config *JWTConfig) (*CustomClaims, error) {
	if config == nil {
		config = DefaultConfig()
	}

	token, err := jwt.ParseWithClaims(tokenString, &CustomClaims{}, func(token *jwt.Token) (interface{}, error) {
		// 验证签名方法
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return []byte(config.AccessTokenSecret), nil
	})

	if err != nil {
		return nil, err
	}

	if claims, ok := token.Claims.(*CustomClaims); ok && token.Valid {
		return claims, nil
	}

	return nil, errors.New("invalid access token")
}

// 解析刷新令牌
func ParseRefreshToken(tokenString string, config *JWTConfig) (uint, error) {
	if config == nil {
		config = DefaultConfig()
	}

	token, err := jwt.ParseWithClaims(tokenString, &jwt.RegisteredClaims{}, func(token *jwt.Token) (interface{}, error) {
		// 验证签名方法
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return []byte(config.RefreshTokenSecret), nil
	})

	if err != nil {
		return 0, err
	}

	if claims, ok := token.Claims.(*jwt.RegisteredClaims); ok && token.Valid {
		// 从 subject 提取 userID
		if claims.Subject == "" {
			return 0, errors.New("refresh token missing subject")
		}
		userID, err := strconv.ParseUint(claims.Subject, 10, 32)
		if err != nil {
			return 0, fmt.Errorf("invalid user ID in refresh token: %w", err)
		}
		return uint(userID), nil
	}

	return 0, errors.New("invalid refresh token")
}

// 刷新令牌对：使用刷新令牌生成新的访问令牌和刷新令牌
func RefreshTokenPair(refreshToken string, userID uint, username, role string, config *JWTConfig) (string, string, error) {
	// 验证刷新令牌
	parsedUserID, err := ParseRefreshToken(refreshToken, config)
	if err != nil {
		return "", "", fmt.Errorf("invalid refresh token: %w", err)
	}

	// 确保刷新令牌属于同一用户
	if parsedUserID != userID {
		return "", "", errors.New("refresh token user mismatch")
	}

	// 生成新的双令牌
	newAccessToken, err := GenerateAccessToken(userID, username, role, config)
	if err != nil {
		return "", "", fmt.Errorf("failed to generate new access token: %w", err)
	}

	newRefreshToken, err := GenerateRefreshToken(userID, config)
	if err != nil {
		return "", "", fmt.Errorf("failed to generate new refresh token: %w", err)
	}

	return newAccessToken, newRefreshToken, nil
}

// 密码哈希与验证
func HashPassword(password string) (string, error) {
	bytes, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return "", fmt.Errorf("failed to hash password: %w", err)
	}
	return string(bytes), nil
}

func VerifyPassword(password, hash string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(password))
	return err == nil
}

// 生成双令牌对
func GenerateTokenPair(userID uint, username, role string, config *JWTConfig) (accessToken, refreshToken string, err error) {
	if config == nil {
		config = DefaultConfig()
	}

	accessToken, err = GenerateAccessToken(userID, username, role, config)
	if err != nil {
		return "", "", fmt.Errorf("failed to generate access token: %w", err)
	}

	refreshToken, err = GenerateRefreshToken(userID, config)
	if err != nil {
		return "", "", fmt.Errorf("failed to generate refresh token: %w", err)
	}

	return accessToken, refreshToken, nil
}
