package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/xuri/excelize/v2"
	"grading-gateway/internal/database"
	"grading-gateway/internal/models"
)

func ExportAssignmentExcel(c *gin.Context) {
	var a models.Assignment
	if err := database.DB.WithContext(c.Request.Context()).First(&a, "id = ?", c.Param("id")).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"detail": "未找到该作业"})
		return
	}
	query := database.DB.WithContext(c.Request.Context()).Where("assignment_id = ?", a.ID)
	if batch := c.Query("batch_id"); batch != "" {
		query = query.Where("material_batch_id = ?", batch)
	}
	query = submissionClassScope(query, database.DB, c.Param("id"), c.Query("class_id"))
	var submissions []models.Submission
	if err := query.Order("id ASC").Find(&submissions).Error; err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"detail": "暂时无法读取成绩"})
		return
	}
	var batches []models.MaterialBatch
	if err := database.DB.WithContext(c.Request.Context()).Select("id,portal_class_id,class_name").Where("assignment_id = ?", a.ID).Find(&batches).Error; err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"detail": "暂时无法读取班级归属"})
		return
	}
	classes, err := assignmentClassCatalog(c.Request.Context(), database.DB, a)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"detail": "暂时无法读取班级列表"})
		return
	}
	f, err := assignmentWorkbook(a, classes, batches, submissions, c.Query("class_id"))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "生成 Excel 失败"})
		return
	}
	defer f.Close()
	var buffer bytes.Buffer
	if err = f.Write(&buffer); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": "生成 Excel 失败"})
		return
	}
	filename := fmt.Sprintf("作业报表_%s.xlsx", a.TaskName)
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename*=utf-8''%s", url.PathEscape(filename)))
	c.Header("Content-Transfer-Encoding", "binary")
	c.Data(http.StatusOK, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", buffer.Bytes())
}

// assignmentWorkbook preserves every submission and its reports. Full exports
// use a separate pair of sheets for each class; identical names/IDs never merge.
func assignmentWorkbook(a models.Assignment, classes []AssignmentClass, batches []models.MaterialBatch, submissions []models.Submission, selected string) (*excelize.File, error) {
	f := excelize.NewFile()
	used := map[string]bool{"sheet1": true}
	write := func(name string, rows []models.Submission, single bool) error {
		grades, reports := "成绩单", "抄袭检测报告"
		if !single {
			grades = uniqueSheetName(name+"_成绩", used)
			reports = uniqueSheetName(name+"_查重", used)
		}
		return writeClassSheets(f, grades, reports, rows)
	}
	if selected != "" {
		if err := write("", submissions, true); err != nil {
			f.Close()
			return nil, err
		}
	} else {
		batchClasses := map[string]AssignmentClass{}
		for _, batch := range batches {
			batchClasses[batch.ID] = batchClass(batch)
		}
		groups := map[string][]models.Submission{}
		for _, sub := range submissions {
			class, ok := batchClasses[sub.MaterialBatchID]
			if !ok {
				class = legacyAssignmentClass(a)
			}
			groups[class.ID] = append(groups[class.ID], sub)
		}
		if len(classes) == 0 {
			classes = []AssignmentClass{legacyAssignmentClass(a)}
		}
		for _, class := range classes {
			if err := write(class.Name, groups[class.ID], false); err != nil {
				f.Close()
				return nil, err
			}
		}
	}
	if err := f.DeleteSheet("Sheet1"); err != nil {
		f.Close()
		return nil, err
	}
	f.SetActiveSheet(0)
	return f, nil
}

func uniqueSheetName(name string, used map[string]bool) string {
	name = strings.Map(func(r rune) rune {
		if strings.ContainsRune("[]:*?/\\", r) || r < 32 {
			return '_'
		}
		return r
	}, name)
	name = strings.Trim(name, " '")
	if name == "" {
		name = "班级"
	}
	for n := 1; ; n++ {
		suffix := ""
		if n > 1 {
			suffix = "_" + strconv.Itoa(n)
		}
		runes := []rune(name)
		if len(runes) > 31-len(suffix) {
			runes = runes[:31-len(suffix)]
		}
		candidate := string(runes) + suffix
		key := strings.ToLower(candidate)
		if !used[key] {
			used[key] = true
			return candidate
		}
	}
}

func writeClassSheets(f *excelize.File, grades, reports string, submissions []models.Submission) error {
	if _, err := f.NewSheet(grades); err != nil {
		return err
	}
	if _, err := f.NewSheet(reports); err != nil {
		return err
	}
	setRow := func(sheet string, row int, values []interface{}) error {
		cell, _ := excelize.CoordinatesToCellName(1, row)
		return f.SetSheetRow(sheet, cell, &values)
	}
	if err := setRow(grades, 1, []interface{}{"学号&姓名", "AI评分", "AI评语", "是否人工复核", "人工最终评分", "人工评语", "AIGC生成概率", "代码与文档匹配度"}); err != nil {
		return err
	}
	if err := setRow(reports, 1, []interface{}{"学生A", "学生B", "相似内容类型", "初始相似度得分", "AI抄袭判定", "AI详细分析"}); err != nil {
		return err
	}
	reportRow := 2
	seenPairs := map[string]bool{}
	for i, sub := range submissions {
		aigcProb, matchScore := "N/A", "N/A"
		var aigc map[string]interface{}
		if json.Unmarshal([]byte(sub.AIGCReport), &aigc) == nil {
			if probability, ok := aigc["ai_probability"].(float64); ok {
				aigcProb = fmt.Sprintf("%.2f%%", probability*100)
			}
		}
		var match map[string]interface{}
		if json.Unmarshal([]byte(sub.CodeDocMatchReport), &match) == nil {
			if score, ok := match["score"]; ok {
				matchScore = fmt.Sprintf("%v/100", score)
			}
		}
		reviewed := "否"
		var humanScore interface{} = ""
		if sub.IsHumanReviewed {
			reviewed, humanScore = "是", sub.HumanScore // A reviewed zero is a real score.
		}
		if err := setRow(grades, i+2, []interface{}{sub.StudentID + "-" + sub.StudentName, sub.Score, sub.Feedback, reviewed, humanScore, sub.HumanFeedback, aigcProb, matchScore}); err != nil {
			return err
		}
		var plagiarism []map[string]interface{}
		if json.Unmarshal([]byte(sub.PlagiarismReport), &plagiarism) != nil {
			continue
		}
		for _, report := range plagiarism {
			other, kind := fmt.Sprint(report["similar_to"]), fmt.Sprint(report["content_type"])
			pair, reverse := sub.StudentID+"\x00"+other+"\x00"+kind, other+"\x00"+sub.StudentID+"\x00"+kind
			if seenPairs[pair] || seenPairs[reverse] {
				continue
			}
			seenPairs[pair] = true
			conclusion, reasoning := "", ""
			if analysis, ok := report["llm_analysis"].(map[string]interface{}); ok {
				if isPlag, ok := analysis["is_plagiarism"].(bool); ok {
					conclusion = "判定为独立完成"
					if isPlag {
						conclusion = "判定为抄袭"
					}
				}
				reasoning, _ = analysis["reasoning"].(string)
			}
			if err := setRow(reports, reportRow, []interface{}{sub.StudentID + "-" + sub.StudentName, other, kind, fmt.Sprint(report["initial_score"]), conclusion, reasoning}); err != nil {
				return err
			}
			reportRow++
		}
	}
	for _, spec := range []struct {
		start, end string
		width      float64
	}{{"A", "A", 25}, {"B", "B", 12}, {"C", "C", 50}, {"D", "E", 16}, {"F", "F", 50}, {"G", "H", 22}} {
		if err := f.SetColWidth(grades, spec.start, spec.end, spec.width); err != nil {
			return err
		}
	}
	_ = f.SetColWidth(reports, "A", "E", 20)
	return f.SetColWidth(reports, "F", "F", 80)
}
