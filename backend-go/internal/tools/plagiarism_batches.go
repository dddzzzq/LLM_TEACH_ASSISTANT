package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	"google.golang.org/grpc/status"
	"grading-gateway/pb"
)

func grpcFailureCode(err error) string { return status.Code(err).String() }

// Every group is checked against every later group, preserving cross-batch
// comparisons. Splitting the class must never isolate students from each other.
func checkPlagiarismBatched(ctx context.Context, client pb.ComputeServiceClient, texts map[string]string, budget int) (map[string]interface{}, error) {
	result := map[string]interface{}{}
	if len(texts) < 2 {
		return result, nil
	}
	ids := make([]string, 0, len(texts))
	for id := range texts {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	groups := []map[string]string{}
	current := map[string]string{}
	size := 0
	for _, id := range ids {
		n := len(id) + len(texts[id]) + 32
		if n > budget {
			return nil, fmt.Errorf("单份查重材料超过传输预算")
		}
		if size+n > budget && len(current) > 0 {
			groups = append(groups, current)
			current = map[string]string{}
			size = 0
		}
		current[id] = texts[id]
		size += n
	}
	if len(current) > 0 {
		groups = append(groups, current)
	}
	seen := map[string]bool{}
	for i, g := range groups {
		for j := i; j < len(groups); j++ {
			batch := map[string]string{}
			for id, text := range g {
				batch[id] = text
			}
			if j != i {
				for id, text := range groups[j] {
					batch[id] = text
				}
			}
			if len(batch) < 2 {
				continue
			}
			response, err := client.CheckPlagiarism(ctx, &pb.PlagiarismRequest{StudentTexts: batch})
			if err != nil {
				return result, err
			}
			var reports map[string][]map[string]interface{}
			if response == nil || json.Unmarshal([]byte(response.PlagiarismResultsJson), &reports) != nil {
				return result, fmt.Errorf("查重响应格式无效")
			}
			for id, items := range reports {
				for _, item := range items {
					key := id + "/" + fmt.Sprint(item["similar_to"]) + "/" + fmt.Sprint(item["content_type"])
					if seen[key] {
						continue
					}
					seen[key] = true
					old, _ := result[id].([]interface{})
					result[id] = append(old, item)
				}
			}
		}
	}
	return result, nil
}
