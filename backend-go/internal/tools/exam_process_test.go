package tools

import (
	"context"
	"errors"
	"google.golang.org/grpc"
	"grading-gateway/internal/models"
	"grading-gateway/pb"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

type examClient struct {
	pb.ComputeServiceClient
	mode   string
	graded atomic.Int32
}

func (c *examClient) ExtractText(_ context.Context, r *pb.ExtractRequest, _ ...grpc.CallOption) (*pb.ExtractResponse, error) {
	if c.mode == "ocr_error" {
		return nil, errors.New("ocr failed")
	}
	if c.mode == "empty_ocr" {
		return &pb.ExtractResponse{}, nil
	}
	return &pb.ExtractResponse{TextContent: string(r.FileContent)}, nil
}
func (c *examClient) IdentifyQuestionNumber(context.Context, *pb.IdentifyQuestionRequest, ...grpc.CallOption) (*pb.IdentifyQuestionResponse, error) {
	return &pb.IdentifyQuestionResponse{}, nil
}
func (c *examClient) GradeExamQuestion(_ context.Context, r *pb.GradeExamQuestionRequest, _ ...grpc.CallOption) (*pb.GradeExamQuestionResponse, error) {
	c.graded.Add(1)
	if c.mode == "grade_error" {
		return nil, errors.New("grading failed")
	}
	score := float32(3)
	if c.mode == "out_of_range" {
		score = 99
	}
	if !strings.Contains(r.FullStudentText, "page one") || strings.Index(r.FullStudentText, "page one") > strings.Index(r.FullStudentText, "page two") {
		return nil, errors.New("page order lost")
	}
	return &pb.GradeExamQuestionResponse{Score: score, Feedback: "evidence", StudentAnswerExtracted: "answer"}, nil
}
func (c *examClient) SummarizeExam(context.Context, *pb.SummarizeExamRequest, ...grpc.CallOption) (*pb.SummarizeExamResponse, error) {
	if c.mode == "summary_error" {
		return nil, errors.New("summary failed")
	}
	return &pb.SummarizeExamResponse{SummaryReport: "summary"}, nil
}
func TestExamFailuresNeverBecomeZeroScores(t *testing.T) {
	paths := []string{filepath.Join(t.TempDir(), "1.png"), filepath.Join(t.TempDir(), "2.png")}
	os.WriteFile(paths[0], []byte("page one"), 0600)
	os.WriteFile(paths[1], []byte("page two"), 0600)
	exam := models.Exam{Questions: []models.ExamQuestion{{ID: 1, QuestionNumber: 1, MaxScore: 5}, {ID: 2, QuestionNumber: 2, MaxScore: 5}}}
	for _, mode := range []string{"ocr_error", "empty_ocr", "grade_error", "out_of_range", "summary_error", "success"} {
		t.Run(mode, func(t *testing.T) {
			client := &examClient{mode: mode}
			images, answers, total, _, err := evaluateExam(exam, paths, client)
			if mode == "success" {
				if err != nil || total != 6 || len(images) != 2 || len(answers) != 2 {
					t.Fatalf("unexpected result %v %v", total, err)
				}
			} else if err == nil || answers != nil {
				t.Fatal("failure produced a valid score")
			}
			if (mode == "ocr_error" || mode == "empty_ocr") && client.graded.Load() != 0 {
				t.Fatal("unreadable exam was graded")
			}
		})
	}
}
