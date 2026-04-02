package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"grading-gateway/internal/database"
	"grading-gateway/internal/grpcclient"
	"grading-gateway/internal/models"
	"grading-gateway/pb"
)

// ScoreReportForPooling 用于池化的学生成绩报告
type ScoreReportForPooling struct {
	StudentID     string  `json:"student_id"`
	OriginalScore float64 `json:"original_score"`
	Feedback      string  `json:"feedback"`
	AIGCScore     float64 `json:"aigc_score,omitempty"`
	HasPlagiarism bool    `json:"has_plagiarism,omitempty"`
	CodeDocMatch  float64 `json:"code_doc_match,omitempty"`
}

// PooledScoreResult 从gRPC服务返回的池化结果
type PooledScoreResult struct {
	StudentID     string  `json:"student_id"`
	OriginalScore float64 `json:"original_score"`
	PooledScore   float64 `json:"pooled_score"`
	PoolReason    string  `json:"pool_reason"`
}

// AddPoolingToPipeline 在批改流水线完成后添加池化处理（使用gRPC AI服务）
func AddPoolingToPipeline(assignmentID string) {
	log.Printf("[成绩池化] 开始对作业 %s 进行AI成绩池化处理...", assignmentID)

	// 等待一段时间确保所有批改完成
	time.Sleep(5 * time.Second)

	// 1. 获取作业的所有提交
	var submissions []models.Submission
	if err := database.DB.Where("assignment_id = ?", assignmentID).Find(&submissions).Error; err != nil {
		log.Printf("[成绩池化] 获取作业提交失败: %v", err)
		return
	}

	if len(submissions) == 0 {
		log.Printf("[成绩池化] 该作业没有提交记录")
		return
	}

	// 2. 构建发送给AI的JSON数据
	scoresData, err := buildScoresData(submissions)
	if err != nil {
		log.Printf("[成绩池化] 构建成绩数据失败: %v", err)
		return
	}

	// 3. 调用gRPC AI服务进行成绩池化
	pooledResults, err := callAIPoolingService(assignmentID, scoresData)
	if err != nil {
		log.Printf("[成绩池化] AI池化服务调用失败: %v", err)
		return
	}

	// 4. 更新数据库
	if err := updateDatabaseWithPooledScores(assignmentID, pooledResults); err != nil {
		log.Printf("[成绩池化] 更新数据库失败: %v", err)
		return
	}

	log.Printf("[成绩池化] 完成! 共处理 %d 名学生的成绩", len(pooledResults))
}

// buildScoresData 构建发送给AI的成绩数据
func buildScoresData(submissions []models.Submission) (string, error) {
	var scoreReports []ScoreReportForPooling

	for _, sub := range submissions {
		// 解析AIGC报告
		aigcScore := 0.0
		if sub.AIGCReport != "" && sub.AIGCReport != "{}" {
			var aigc map[string]interface{}
			if err := json.Unmarshal([]byte(sub.AIGCReport), &aigc); err == nil {
				if prob, ok := aigc["ai_probability"].(float64); ok {
					aigcScore = prob * 100 // 转换为百分比
				}
			}
		}

		// 解析抄袭报告
		hasPlagiarism := false
		if sub.PlagiarismReport != "" && sub.PlagiarismReport != "[]" {
			var plagReports []map[string]interface{}
			if err := json.Unmarshal([]byte(sub.PlagiarismReport), &plagReports); err == nil {
				for _, report := range plagReports {
					if llmData, ok := report["llm_analysis"].(map[string]interface{}); ok {
						if similarity, ok := llmData["similarity_score"].(float64); ok && similarity > 90 {
							hasPlagiarism = true
							break
						}
					}
				}
			}
		}

		// 解析代码文档匹配度
		codeDocMatch := 0.0
		if sub.CodeDocMatchReport != "" && sub.CodeDocMatchReport != "{}" {
			var match map[string]interface{}
			if err := json.Unmarshal([]byte(sub.CodeDocMatchReport), &match); err == nil {
				if score, ok := match["score"].(float64); ok {
					codeDocMatch = score
				}
			}
		}

		scoreReports = append(scoreReports, ScoreReportForPooling{
			StudentID:     sub.StudentID,
			OriginalScore: sub.Score,
			Feedback:      sub.Feedback,
			AIGCScore:     aigcScore,
			HasPlagiarism: hasPlagiarism,
			CodeDocMatch:  codeDocMatch,
		})
	}

	scoresData, err := json.Marshal(scoreReports)
	if err != nil {
		return "", fmt.Errorf("JSON序列化失败: %v", err)
	}

	return string(scoresData), nil
}

// callAIPoolingService 调用gRPC AI服务进行成绩池化
func callAIPoolingService(assignmentID string, scoresData string) ([]PooledScoreResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()

	request := &pb.PoolScoresRequest{
		AssignmentId:   assignmentID,
		ScoresDataJson: scoresData,
	}

	response, err := grpcclient.Client.PoolScores(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("gRPC调用失败: %v", err)
	}

	if !response.Success {
		return nil, fmt.Errorf("AI池化处理失败: %s", response.ErrorMessage)
	}

	// 解析返回的JSON数据
	var pooledResults []PooledScoreResult
	if err := json.Unmarshal([]byte(response.PooledResultsJson), &pooledResults); err != nil {
		return nil, fmt.Errorf("解析池化结果失败: %v", err)
	}

	log.Printf("[成绩池化] AI服务返回 %d 条池化结果", len(pooledResults))
	return pooledResults, nil
}

// updateDatabaseWithPooledScores 用池化后的成绩更新数据库
func updateDatabaseWithPooledScores(assignmentID string, pooledResults []PooledScoreResult) error {
	// 开始数据库事务
	tx := database.DB.Begin()
	if tx.Error != nil {
		return tx.Error
	}
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	updateCount := 0
	for _, result := range pooledResults {
		// 查找对应的提交记录
		var submission models.Submission
		if err := tx.Where("assignment_id = ? AND student_id = ?", assignmentID, result.StudentID).First(&submission).Error; err != nil {
			log.Printf("[成绩池化] 找不到学生 %s 的提交记录: %v", result.StudentID, err)
			continue
		}

		// 更新分数和反馈
		originalScore := submission.Score
		submission.Score = result.PooledScore

		// 在原有反馈基础上添加池化说明
		newFeedback := fmt.Sprintf("%s\n\n【成绩池化处理】\n- 原始分数: %.2f\n- 池化后分数: %.2f\n- 调整理由: %s",
			submission.Feedback, originalScore, result.PooledScore, result.PoolReason)

		// 截断反馈文本，防止超出数据库字段长度限制
		if len(newFeedback) > 10000 {
			newFeedback = newFeedback[:10000]
		}
		submission.Feedback = newFeedback

		// 保存更新
		if err := tx.Save(&submission).Error; err != nil {
			log.Printf("[成绩池化] 更新学生 %s 成绩失败: %v", result.StudentID, err)
			continue
		}

		updateCount++
		log.Printf("[成绩池化] 已更新学生 %s: %.2f -> %.2f (%s)",
			result.StudentID, originalScore, result.PooledScore, result.PoolReason)
	}

	// 提交事务
	if err := tx.Commit().Error; err != nil {
		tx.Rollback()
		return fmt.Errorf("提交事务失败: %v", err)
	}

	log.Printf("[成绩池化] 成功更新 %d/%d 名学生的成绩", updateCount, len(pooledResults))
	return nil
}
