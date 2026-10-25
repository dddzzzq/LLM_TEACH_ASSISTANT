package agent

import (
	"context"
	"fmt"
	"grading-gateway/internal/agent/einoruntime"
	"grading-gateway/internal/agent/modelclient"
	"grading-gateway/internal/execution"
)

type ModelCall = modelclient.Call
type RunResult struct {
	Reply        string
	JobID        string
	JobType      string
	ResultURL    string
	LoadedSkills map[string]string
}

type DialogueOptions struct {
	History       []execution.Message
	Writer        *execution.Writer
	ModelMetadata map[string]any
}

// RunDialogue 是业务边界; Eino 控制整个 Agent/Tool Loop.
func RunDialogue(ctx context.Context, system, user, role, username string, registry *ToolRegistry, definitions []map[string]interface{}, catalog SkillCatalog, call ModelCall, options ...DialogueOptions) (RunResult, error) {
	var opts DialogueOptions
	if len(options) > 0 {
		opts = options[0]
	}
	var record execution.Record
	if opts.Writer != nil {
		record = opts.Writer.Record
	}
	g := newGateway(role, username, registry)
	g.record = record
	if key, ok := ctx.Value(requestKeyContext{}).(string); ok && key != "" && len(key) <= 128 {
		g.runID = key
	}
	tools := []einoruntime.Tool{}
	for _, definition := range definitions {
		f, ok := definition["function"].(map[string]interface{})
		if !ok {
			continue
		}
		name, _ := f["name"].(string)
		t, ok := registry.GetTool(name)
		if !ok {
			continue
		}
		description, _ := f["description"].(string)
		tools = append(tools, einoruntime.Tool{Name: name, Description: description, Schema: []byte(t.Schema()), Execute: func(ctx context.Context, args string) (string, error) { return g.execute(ctx, name, args) }})
	}
	if role != "teacher" && role != "admin" {
		catalog = nil
	}
	system += "\n执行对应批改前先 load_skill。Skill 工具接受 skill 名称。缺少目标或材料时询问用户。任务受理后说明任务 ID 并结束本轮，不轮询等待。"
	reply, err := einoruntime.Run(ctx, einoruntime.Config{Name: "teaching-assistant", Instruction: system, Call: call, Tools: tools, Skills: catalog, SkillLoaded: g.skillLoaded, BeforeModel: g.beforeModel, DisableTools: func() bool { return g.submitted }, History: opts.History, Record: record, ModelMetadata: opts.ModelMetadata}, user)
	g.result.Reply = reply
	if opts.Writer != nil && opts.Writer.Err() != nil {
		return g.result, opts.Writer.Err()
	}
	if err != nil && g.lastResult != "" {
		if g.submitted && g.result.JobID != "" {
			if g.result.JobType == "portal_catalog_sync" {
				g.result.Reply = fmt.Sprintf("已提交教务目录同步任务，任务 ID：%s。请在教务接入设置页查看进度；本次仅同步目录，尚未提交下载或批改任务。", g.result.JobID)
			} else {
				g.result.Reply = fmt.Sprintf("任务已受理，任务 ID：%s。请在任务面板查看进度，是否完成以实际任务状态为准。", g.result.JobID)
			}
		} else {
			g.result.Reply = "工具返回结果：\n" + g.lastResult
		}
		return g.result, nil
	}
	return g.result, err
}
