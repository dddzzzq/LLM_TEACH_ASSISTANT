package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// LLMResponse 表示 LLM 的响应结果
type LLMResponse struct {
	// Content LLM 返回的文本内容
	Content string
	// ToolCalls 工具调用列表，如果没有工具调用则为空
	ToolCalls []ToolCall
}

// ToolCall 表示一个工具调用请求
type ToolCall struct {
	// ID 工具调用的唯一标识符
	ID string
	// Type 工具调用类型，固定为 "function"
	Type string
	// Function 函数调用详情
	Function FunctionCall
}

// FunctionCall 表示函数调用的详细信息
type FunctionCall struct {
	// Name 被调用的函数名
	Name string
	// Arguments 函数参数的 JSON 字符串
	Arguments string
}

// DeepSeekClient 封装了与 DeepSeek API 的交互
type DeepSeekClient struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// NewDeepSeekClient 创建一个新的 DeepSeek 客户端
// apiKey 如果为空，会尝试从环境变量 DEEPSEEK_API_KEY 读取
func NewDeepSeekClient(apiKey string) *DeepSeekClient {
	if apiKey == "" {
		apiKey = os.Getenv("DEEPSEEK_API_KEY")
	}

	return &DeepSeekClient{
		apiKey:  apiKey,
		baseURL: "https://api.deepseek.com/v1/chat/completions",
		httpClient: &http.Client{
			Timeout: 120 * time.Second, // 设置较长的超时时间以适应复杂的思考
		},
	}
}

// CallDeepSeekWithTools 调用 DeepSeek API，支持系统提示、用户消息和工具定义
// 这是包级的便捷函数，内部使用默认客户端
func CallDeepSeekWithTools(systemPrompt string, userMessage string, tools []map[string]interface{}) (*LLMResponse, error) {
	client := NewDeepSeekClient("")
	return client.CallWithTools(systemPrompt, userMessage, tools)
}

// CallWithTools 实例方法版本的调用
func (c *DeepSeekClient) CallWithTools(systemPrompt string, userMessage string, tools []map[string]interface{}) (*LLMResponse, error) {
	if c.apiKey == "" {
		return nil, fmt.Errorf("DeepSeek API key not provided. Set DEEPSEEK_API_KEY environment variable")
	}

	// 构建请求消息
	messages := []map[string]interface{}{
		{
			"role":    "system",
			"content": systemPrompt,
		},
		{
			"role":    "user",
			"content": userMessage,
		},
	}

	// 构建请求体
	requestBody := map[string]interface{}{
		"model":       "deepseek-chat", // DeepSeek 的模型名称
		"messages":    messages,
		"temperature": 0.3,
		"max_tokens":  4000,
	}

	// 如果有工具，添加到请求中
	if len(tools) > 0 {
		requestBody["tools"] = tools
		requestBody["tool_choice"] = "auto" // 让模型自行决定是否调用工具
	}

	// 编码请求体
	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	// 创建 HTTP 请求
	req, err := http.NewRequest("POST", c.baseURL, bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	// 设置请求头
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	// 发送请求
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to DeepSeek API: %w", err)
	}
	defer resp.Body.Close()

	// 读取响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// 检查 HTTP 状态码
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("DeepSeek API returned error: %s, body: %s", resp.Status, string(body))
	}

	// 解析响应
	var apiResponse map[string]interface{}
	if err := json.Unmarshal(body, &apiResponse); err != nil {
		return nil, fmt.Errorf("failed to unmarshal API response: %w", err)
	}

	// 提取消息内容
	choices, ok := apiResponse["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return nil, fmt.Errorf("invalid API response format: missing choices")
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid API response format: choice is not an object")
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid API response format: missing message")
	}

	// 构建响应结果
	result := &LLMResponse{}

	// 提取文本内容
	if content, ok := message["content"].(string); ok && content != "" {
		result.Content = content
	}

	// 提取工具调用
	if toolCalls, ok := message["tool_calls"].([]interface{}); ok && len(toolCalls) > 0 {
		for _, tc := range toolCalls {
			if toolCall, ok := tc.(map[string]interface{}); ok {
				parsedToolCall := ToolCall{
					ID:   getString(toolCall, "id"),
					Type: getString(toolCall, "type"),
				}

				if funcData, ok := toolCall["function"].(map[string]interface{}); ok {
					parsedToolCall.Function = FunctionCall{
						Name:      getString(funcData, "name"),
						Arguments: getString(funcData, "arguments"),
					}
				}

				result.ToolCalls = append(result.ToolCalls, parsedToolCall)
			}
		}
	}

	return result, nil
}

// getString 安全地从 map 中获取字符串值
func getString(m map[string]interface{}, key string) string {
	if val, ok := m[key]; ok {
		if str, ok := val.(string); ok {
			return str
		}
	}
	return ""
}

// HasToolCalls 检查响应是否包含工具调用
func (r *LLMResponse) HasToolCalls() bool {
	return len(r.ToolCalls) > 0
}

// GetToolCall 根据函数名获取工具调用
func (r *LLMResponse) GetToolCall(functionName string) (*ToolCall, bool) {
	for _, tc := range r.ToolCalls {
		if tc.Function.Name == functionName {
			return &tc, true
		}
	}
	return nil, false
}
