package tools

import (
	"archive/zip"
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"
	"testing"

	"google.golang.org/grpc"
	"grading-gateway/internal/grpcclient"
	"grading-gateway/pb"
)

type requirementOCRFixture struct {
	pb.ComputeServiceClient
	calls int
}

func (f *requirementOCRFixture) ExtractText(ctx context.Context, req *pb.ExtractRequest, opts ...grpc.CallOption) (*pb.ExtractResponse, error) {
	f.calls++
	return &pb.ExtractResponse{TextContent: "图片中的题目要求"}, nil
}

func TestRequirementDocxIncludesEmbeddedImages(t *testing.T) {
	prior := grpcclient.Client
	t.Cleanup(func() { grpcclient.Client = prior })
	fixture := &requirementOCRFixture{}
	grpcclient.Client = fixture
	var buffer bytes.Buffer
	zw := zip.NewWriter(&buffer)
	w, _ := zw.Create("word/document.xml")
	_, _ = w.Write([]byte(`<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>完成程序并提交报告</w:t></w:r></w:p></w:body></w:document>`))
	w, _ = zw.Create("word/media/image1.png")
	_, _ = w.Write([]byte("synthetic-image"))
	_ = zw.Close()
	text, warnings := ExtractRequirement(context.Background(), "题目.docx", buffer.Bytes())
	if len(warnings) != 0 || !strings.Contains(text, "完成程序") || !strings.Contains(text, "图片中的题目要求") || fixture.calls != 1 {
		t.Fatalf("incomplete DOCX: %q %v", text, warnings)
	}
	text, images, warnings := ExtractRequirementWithImages(context.Background(), "题目.docx", buffer.Bytes())
	if len(warnings) != 0 || len(images) != 1 || string(images[0].Data) != "synthetic-image" || !strings.Contains(text, "完成程序") || !strings.Contains(text, "[原图 1]") || fixture.calls != 1 {
		t.Fatal("vision path lost original Word image or called OCR instead")
	}
}

func questionPDFFixture(content string) []byte {
	objects := []string{"<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>", "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", fmt.Sprintf("<< /Length %d >>\nstream\n%s\nendstream", len(content), content)}
	var output strings.Builder
	output.WriteString("%PDF-1.4\n")
	offsets := []int{0}
	for i, obj := range objects {
		offsets = append(offsets, output.Len())
		fmt.Fprintf(&output, "%d 0 obj\n%s\nendobj\n", i+1, obj)
	}
	xref := output.Len()
	fmt.Fprintf(&output, "xref\n0 %d\n0000000000 65535 f \n", len(objects)+1)
	for _, offset := range offsets[1:] {
		fmt.Fprintf(&output, "%010d 00000 n \n", offset)
	}
	fmt.Fprintf(&output, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF", len(objects)+1, xref)
	return []byte(output.String())
}

func TestRequirementPDFTextAndScannedPageFallback(t *testing.T) {
	if _, err := exec.LookPath("pdfinfo"); err != nil {
		t.Skip("requires poppler-utils")
	}
	prior := grpcclient.Client
	t.Cleanup(func() { grpcclient.Client = prior })
	fixture := &requirementOCRFixture{}
	grpcclient.Client = fixture
	text, warnings := ExtractRequirement(context.Background(), "题目.pdf", questionPDFFixture("BT /F1 12 Tf 30 740 Td (Implement a concurrent downloader) Tj ET"))
	if len(warnings) != 0 || !strings.Contains(text, "concurrent downloader") || fixture.calls != 0 {
		t.Fatalf("PDF text: %q %v", text, warnings)
	}
	text, warnings = ExtractRequirement(context.Background(), "扫描题目.pdf", questionPDFFixture(""))
	if len(warnings) != 0 || !strings.Contains(text, "图片中的题目要求") || fixture.calls != 1 {
		t.Fatalf("PDF OCR fallback: %q %v", text, warnings)
	}
	text, images, warnings := ExtractRequirementWithImages(context.Background(), "扫描题目.pdf", questionPDFFixture(""))
	if len(warnings) != 0 || len(images) != 1 || !bytes.HasPrefix(images[0].Data, []byte("\x89PNG")) || !strings.Contains(text, "[原图 1]") || fixture.calls != 1 {
		t.Fatal("scanned PDF did not supply its rendered original page to vision")
	}
}

func TestRequirementOCRFailureIsNotAQuestion(t *testing.T) {
	prior := grpcclient.Client
	t.Cleanup(func() { grpcclient.Client = prior })
	grpcclient.Client = nil
	text, warnings := ExtractRequirement(context.Background(), "题目.png", []byte("image"))
	if text != "" || len(warnings) == 0 {
		t.Fatal("OCR failure became usable requirements")
	}
}
