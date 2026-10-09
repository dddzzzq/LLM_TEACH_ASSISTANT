package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"

	"grading-gateway/internal/cache"
	"grading-gateway/internal/database"
	"grading-gateway/internal/grading"
	"grading-gateway/internal/middleware"
	"grading-gateway/internal/models"
	"grading-gateway/internal/schemas"
	"grading-gateway/internal/tools"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/xuri/excelize/v2"
	"gorm.io/gorm"
)

// 创建作业handler
func CreateAssignment(c *gin.Context) {
	var req schemas.AssignmentCreate
	if err := c.ShouldBindJSON(&req); err != nil {
		// 处理失败
		log.Printf("创建作业请求格式存在问题： %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"detail": "创建作业数据请求格式有误", "error": err.Error()})
		return
	}

	// 请求成功，创建作业并入库
	// json化rubric
	rubricBytes, _ := json.Marshal(req.Rubric)
	var assignment = models.Assignment{
		CourseName: req.CourseName,
		ClassName:  req.ClassName,
		TaskName:   req.TaskName,
		Question:   req.Question,
		Rubric:     string(rubricBytes),
	}

	// 入库失败
	if err := database.DB.Create(&assignment).Error; err != nil {
		log.Printf("创建作业入库失败：%v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"message": "创建作业入库失败", "error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, schemas.AssignmentResponse{
		ID:       assignment.ID,
		TaskName: assignment.TaskName,
		Question: assignment.Question,
		Rubric:   req.Rubric,
	})
}

// 获取所有作业handler
func GetAssignments(c *gin.Context) {
	// 从数据库中获取所有assignmnets
	var assignments []models.Assignment
	if err := database.DB.Order("id desc").Find(&assignments).Error; err != nil {
		log.Printf("获取所有作业失败：%v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"message": "获取所有作业失败！", "error": err.Error()})
		return
	}

	// 获取成功后返回json
	var res []schemas.AssignmentResponse
	for _, ass := range assignments {
		var rubricMap map[string]interface{}
		if ass.Rubric != "" {
			if err := json.Unmarshal([]byte(ass.Rubric), &rubricMap); err != nil {
				rubricMap = make(map[string]interface{})
			}
		} else {
			rubricMap = make(map[string]interface{})
		}

		res = append(res, schemas.AssignmentResponse{
			ID:         ass.ID,
			CourseName: ass.CourseName,
			ClassName:  ass.ClassName,
			TaskName:   ass.TaskName,
			Question:   ass.Question,
			Rubric:     rubricMap,
		})
	}

	if res == nil {
		res = []schemas.AssignmentResponse{}
	}
	c.JSON(http.StatusOK, res)
}

// 获取单个作业详情handler
func GetAssignment(c *gin.Context) {
	idStr := c.Param("id")
	assignmentID, err := strconv.ParseUint(idStr, 10, 32)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": "无效的作业ID"})
		return
	}

	// 获取用户角色和用户名
	role, roleErr := middleware.GetRoleFromContext(c)
	username, usernameErr := middleware.GetUsernameFromContext(c)

	// 1. 【动静分离 - 静态部分】：从 Redis 缓存中获取作业基础信息（带防穿透、防击穿保护）
	assignmentBase, err := cache.GetAssignmentWithCache(c.Request.Context(), uint(assignmentID))
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			log.Printf("获取单个作业未找到：%v", err)
			c.JSON(http.StatusNotFound, gin.H{"message": "获取单个作业详情出错", "detail": "作业未找到"})
			return
		}
		log.Printf("获取单个作业出错：%v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"message": "系统繁忙，获取作业详情失败"})
		return
	}

	// 2. 【动静分离 - 动态部分】：根据角色权限，单独去 MySQL 查询属于该用户的 Submissions
	var submissions []models.Submission
	subQuery := database.DB.Where("assignment_id = ?", assignmentID)

	// 如果是学生，只查询自己的提交
	if roleErr == nil && usernameErr == nil && role == "student" {
		subQuery = subQuery.Where("student_id = ?", username).Order("score DESC")
	} else {
		// 教师/管理员获取所有提交
		subQuery = subQuery.Order("score DESC")
	}

	if err := subQuery.Find(&submissions).Error; err != nil {
		log.Printf("获取作业提交记录出错：%v", err)
	}

	// 3. 组装数据并返回 JSON
	var rubricMap map[string]interface{}
	if assignmentBase.Rubric != "" {
		json.Unmarshal([]byte(assignmentBase.Rubric), &rubricMap)
	} else {
		rubricMap = make(map[string]interface{})
	}

	var formattedSubmissions []map[string]interface{}
	for _, sub := range submissions {
		// 这里刻意不返回 merged_content，以减小单次作业详情响应体积
		var plag []map[string]interface{}
		if sub.PlagiarismReport != "" && sub.PlagiarismReport != "[]" {
			_ = json.Unmarshal([]byte(sub.PlagiarismReport), &plag)
		} else {
			plag = make([]map[string]interface{}, 0)
		}

		var aigc map[string]interface{}
		if sub.AIGCReport != "" && sub.AIGCReport != "{}" {
			_ = json.Unmarshal([]byte(sub.AIGCReport), &aigc)
		}

		var match map[string]interface{}
		if sub.CodeDocMatchReport != "" && sub.CodeDocMatchReport != "{}" {
			_ = json.Unmarshal([]byte(sub.CodeDocMatchReport), &match)
		}

		formattedSubmissions = append(formattedSubmissions, map[string]interface{}{
			"id":                    sub.ID,
			"student_id":            sub.StudentID,
			"student_name":          sub.StudentName,
			"score":                 sub.Score,
			"feedback":              sub.Feedback,
			"plagiarism_reports":    plag,
			"aigc_report":           aigc,
			"code_doc_match_report": match,
			"assignment_id":         sub.AssignmentID,
			"is_human_reviewed":     sub.IsHumanReviewed,
			"human_feedback":        sub.HumanFeedback,
			"human_score":           sub.HumanScore,
		})
	}

	if formattedSubmissions == nil {
		formattedSubmissions = make([]map[string]interface{}, 0)
	}

	c.JSON(http.StatusOK, gin.H{
		"id":          assignmentBase.ID,
		"course_name": assignmentBase.CourseName,
		"class_name":  assignmentBase.ClassName,
		"task_name":   assignmentBase.TaskName,
		"question":    assignmentBase.Question,
		"rubric":      rubricMap,
		"submissions": formattedSubmissions,
	})
}

func GetAssignmentSubmissions(c *gin.Context) {
	assignmentID := c.Param("id")

	// 获取用户角色和用户名
	role, roleErr := middleware.GetRoleFromContext(c)
	username, usernameErr := middleware.GetUsernameFromContext(c)

	var submissions = make([]models.Submission, 0)
	query := database.DB.Where("assignment_id = ?", assignmentID)

	// 如果是学生，只能查看自己的提交
	if roleErr == nil && usernameErr == nil && role == "student" {
		query = query.Where("student_id = ?", username)
	}

	if err := query.Order("score DESC").Find(&submissions).Error; err != nil {
		log.Printf("获取所有提交出错: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"message": "获取所有提交出错", "error": err.Error()})
		return
	}

	// 格式化输出
	var formattedSubmissions []map[string]interface{}
	for _, sub := range submissions {
		formattedSubmissions = append(formattedSubmissions, tools.FormatSubmission(sub))
	}

	if formattedSubmissions == nil {
		formattedSubmissions = make([]map[string]interface{}, 0)
	}

	c.JSON(http.StatusOK, formattedSubmissions)
}

// 删除作业handler
func DeleteAssignment(c *gin.Context) {
	id := c.Param("id")

	// 获取一致性服务
	consistencySvc := database.GetConsistencyService()
	ctx := context.Background()

	//
	assignmentKey := fmt.Sprintf("assignment:%s", id)

	// 使用一致性服务删除缓存和数据
	err := consistencySvc.DeleteThenInvalidate(ctx, "assignment", assignmentKey, func() error {
		// 这是实际的MySQL删除操作
		return database.DB.Delete(&models.Assignment{}, id).Error
	})

	if err != nil {
		log.Printf("删除作业失败： %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "删除作业失败"})
		return
	}

	// 清理作业相关的提交缓存
	consistencySvc.InvalidateByPattern(ctx, fmt.Sprintf("*assignment:%s*", id))
	consistencySvc.InvalidateByPattern(ctx, fmt.Sprintf("*submission:*:%s*", id))
	// 【新增】：主动清理我们刚刚引入的 DB 旁路缓存
	assignmentID, _ := strconv.ParseUint(id, 10, 32)
	cache.InvalidateAssignmentCache(ctx, uint(assignmentID))

	c.Status(http.StatusNoContent)
}

// 上传学生提交handler
func UploadAssignment(c *gin.Context) {
	actor, ok := gradingActor(c)
	if !ok {
		return
	}
	assignmentID := c.Param("id")
	if !checkGradingTarget(c, actor, "homework", assignmentID) {
		return
	}
	file, err := c.FormFile("batch_file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": "请上传本次作业 ZIP"})
		return
	}
	uploadDir := "./uploads/assignments"
	if err := os.MkdirAll(uploadDir, 0755); err != nil {
		c.JSON(500, gin.H{"detail": "无法创建上传目录"})
		return
	}
	savePath := filepath.Join(uploadDir, fmt.Sprintf("assignment_%s_%s%s", assignmentID, uuid.NewString(), filepath.Ext(file.Filename)))
	if err := c.SaveUploadedFile(file, savePath); err != nil {
		c.JSON(500, gin.H{"detail": "文件保存失败"})
		return
	}
	submitGradingUpload(c, actor, grading.Request{Kind: "homework", ResourceID: assignmentID, FilePath: savePath})
}

// 一键清空所有学生提交handler
func ClearAllSubmissions(c *gin.Context) {
	assignmentID := c.Param("id")

	if err := database.DB.Where("assignment_id = ?", assignmentID).Delete(&models.Submission{}).Error; err != nil {
		log.Printf("一键清空失败： %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "一键清空失败", "error": err.Error()})
		return
	}

	c.Status(http.StatusNoContent)
}

// 导出excel文件handler
func ExportAssignmentExcel(c *gin.Context) {
	id := c.Param("id")
	var assignment models.Assignment

	// 预加载该作业的所有批改记录
	if err := database.DB.Preload("Submissions").First(&assignment, id).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"detail": "未找到该作业"})
		return
	}

	f := excelize.NewFile()
	defer func() {
		if err := f.Close(); err != nil {
			log.Printf("关闭 Excel 文件失败: %v", err)
		}
	}()

	// Sheet 1: 成绩单
	sheetName := "成绩单"
	index, _ := f.NewSheet(sheetName)
	f.SetActiveSheet(index)

	headers := []string{"学号&姓名", "AI评分", "AI评语", "是否人工复核", "人工最终评分", "人工评语", "AIGC生成概率", "代码与文档匹配度"}
	for i, header := range headers {
		cellLabel, _ := excelize.CoordinatesToCellName(i+1, 1)
		f.SetCellValue(sheetName, cellLabel, header)
	}

	for i, sub := range assignment.Submissions {
		row := i + 2

		aigcProb := "N/A"
		if sub.AIGCReport != "" && sub.AIGCReport != "{}" {
			var aigc map[string]interface{}
			if err := json.Unmarshal([]byte(sub.AIGCReport), &aigc); err == nil {
				if prob, ok := aigc["ai_probability"]; ok {
					aigcProb = fmt.Sprintf("%.2f%%", prob.(float64)*100)
				}
			}
		}

		matchScore := "N/A"
		if sub.CodeDocMatchReport != "" && sub.CodeDocMatchReport != "{}" {
			var match map[string]interface{}
			if err := json.Unmarshal([]byte(sub.CodeDocMatchReport), &match); err == nil {
				if score, ok := match["score"]; ok {
					matchScore = fmt.Sprintf("%v/100", score)
				}
			}
		}

		isReviewed := "否"
		if sub.IsHumanReviewed {
			isReviewed = "是"
		}

		humanScore := ""
		if sub.HumanScore != 0 {
			humanScore = fmt.Sprintf("%.2f", sub.HumanScore)
		}

		f.SetCellValue(sheetName, fmt.Sprintf("A%d", row), sub.StudentID+"-"+sub.StudentName)
		f.SetCellValue(sheetName, fmt.Sprintf("B%d", row), sub.Score)
		f.SetCellValue(sheetName, fmt.Sprintf("C%d", row), sub.Feedback)
		f.SetCellValue(sheetName, fmt.Sprintf("D%d", row), isReviewed)
		f.SetCellValue(sheetName, fmt.Sprintf("E%d", row), humanScore)
		f.SetCellValue(sheetName, fmt.Sprintf("F%d", row), sub.HumanFeedback)
		f.SetCellValue(sheetName, fmt.Sprintf("G%d", row), aigcProb)
		f.SetCellValue(sheetName, fmt.Sprintf("H%d", row), matchScore)
	}

	f.SetColWidth(sheetName, "A", "A", 25)
	f.SetColWidth(sheetName, "B", "B", 10)
	f.SetColWidth(sheetName, "C", "C", 50)
	f.SetColWidth(sheetName, "D", "D", 15)
	f.SetColWidth(sheetName, "E", "E", 15)
	f.SetColWidth(sheetName, "F", "F", 50)
	f.SetColWidth(sheetName, "G", "H", 20)

	// Sheet 2: 抄袭检测报告
	plagSheetName := "抄袭检测报告"
	f.NewSheet(plagSheetName)

	plagHeaders := []string{"学生A", "学生B", "相似内容类型", "初始相似度得分", "AI抄袭判定", "AI详细分析"}
	for i, header := range plagHeaders {
		cellLabel, _ := excelize.CoordinatesToCellName(i+1, 1)
		f.SetCellValue(plagSheetName, cellLabel, header)
	}

	plagRow := 2
	// 使用 map 记录已经输出过的双向关系，避免 A抄B 和 B抄A 重复出现
	seenPairs := make(map[string]bool)

	for _, sub := range assignment.Submissions {
		if sub.PlagiarismReport == "" || sub.PlagiarismReport == "[]" {
			continue
		}

		var plagReports []map[string]interface{}
		if err := json.Unmarshal([]byte(sub.PlagiarismReport), &plagReports); err == nil {
			for _, report := range plagReports {
				studentB := fmt.Sprintf("%v", report["similar_to"])
				contentType := fmt.Sprintf("%v", report["content_type"])

				// 构造唯一键，确保 A-B 和 B-A 只记录一次
				pairKey1 := fmt.Sprintf("%s-%s-%s", sub.StudentID, studentB, contentType)
				pairKey2 := fmt.Sprintf("%s-%s-%s", studentB, sub.StudentID, contentType)

				if seenPairs[pairKey1] || seenPairs[pairKey2] {
					continue
				}
				seenPairs[pairKey1] = true

				initialScore := fmt.Sprintf("%v", report["initial_score"])

				aiAnalysis := ""
				aiConclusion := ""
				if llmData, ok := report["llm_analysis"].(map[string]interface{}); ok {
					if isPlag, ok := llmData["is_plagiarism"].(bool); ok {
						if isPlag {
							aiConclusion = "判定为抄袭"
						} else {
							aiConclusion = "判定为独立完成"
						}
					}
					if reason, ok := llmData["reasoning"].(string); ok {
						aiAnalysis = reason
					}
				}

				f.SetCellValue(plagSheetName, fmt.Sprintf("A%d", plagRow), sub.StudentID+"-"+sub.StudentName)
				f.SetCellValue(plagSheetName, fmt.Sprintf("B%d", plagRow), studentB)
				f.SetCellValue(plagSheetName, fmt.Sprintf("C%d", plagRow), contentType)
				f.SetCellValue(plagSheetName, fmt.Sprintf("D%d", plagRow), initialScore)
				f.SetCellValue(plagSheetName, fmt.Sprintf("E%d", plagRow), aiConclusion)
				f.SetCellValue(plagSheetName, fmt.Sprintf("F%d", plagRow), aiAnalysis)

				plagRow++
			}
		}
	}

	f.SetColWidth(plagSheetName, "A", "B", 20)
	f.SetColWidth(plagSheetName, "C", "D", 15)
	f.SetColWidth(plagSheetName, "E", "E", 15)
	f.SetColWidth(plagSheetName, "F", "F", 80)

	// 删除默认新建的空 Sheet1
	f.DeleteSheet("Sheet1")

	// 导出返回
	var buffer bytes.Buffer
	if err := f.Write(&buffer); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "生成 Excel 失败"})
		return
	}

	fileName := fmt.Sprintf("作业报表_%s.xlsx", assignment.TaskName)
	encodedFileName := url.QueryEscape(fileName)

	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename*=utf-8''%s", encodedFileName))
	c.Header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
	c.Header("Content-Transfer-Encoding", "binary")

	c.Data(http.StatusOK, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", buffer.Bytes())
}
