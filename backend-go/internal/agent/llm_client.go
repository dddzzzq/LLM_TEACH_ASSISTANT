package agent

import "grading-gateway/internal/agent/modelclient"

type LLMResponse = modelclient.LLMResponse
type ToolCall = modelclient.ToolCall
type FunctionCall = modelclient.FunctionCall
type DeepSeekClient = modelclient.DeepSeekClient

var NewDeepSeekClient = modelclient.NewDeepSeekClient
var CallDeepSeekWithTools = modelclient.CallDeepSeekWithTools
