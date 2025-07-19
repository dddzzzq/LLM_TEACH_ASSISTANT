import io
from docx import Document
import PyPDF2

def read_text_from_docx(file_bytes: bytes) -> str:
    """从.docx文件的字节流中读取文本。"""
    try:
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"读取DOCX文件时出错: {e}")
        return ""

def read_text_from_pdf(file_bytes: bytes) -> str:
    """从.pdf文件的字节流中读取文本。"""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"读取PDF文件时出错: {e}")
        return ""