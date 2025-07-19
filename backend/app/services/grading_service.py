import zipfile
import rarfile
import io
import os

from .file_processor import read_text_from_docx, read_text_from_pdf


class GradingService:
    """首先处理所有文本信息，并合并，给出该作业的评分"""

    def _get_content_from_file(self, filename: str, file_bytes: bytes) -> str:
        """提取文本内容，包括多重压缩以及不同形式的文件"""
        lower_filename = filename.lower()
        
        if lower_filename.endswith(".docx"):
            return read_text_from_docx(file_bytes)
        
        if lower_filename.endswith(".pdf"):
            return read_text_from_pdf(file_bytes)
            
        text_extensions = [
            ".txt", ".py", ".js", ".java", ".c", ".cpp", 
            ".h", ".md", ".html", ".css"
        ]
        if any(lower_filename.endswith(ext) for ext in text_extensions):
            return file_bytes.decode("utf-8", errors="ignore")
            
        return ""

    def _process_archive_items(self, archive_ref, item_infos) -> str:
        """Iterates through items in an archive, extracts content, and merges it."""
        merged_contents = []
        
        # 根据文件名排序
        for item_info in sorted(item_infos, key=lambda x: x.filename):
            if item_info.is_dir():
                continue

            filename = item_info.filename
            # 跳过苹果系统文件
            if filename.startswith("__MACOSX/") or os.path.basename(filename) == ".DS_Store":
                continue

            file_content_bytes = archive_ref.read(item_info)
            raw_answer = ""

            # 处理嵌套压缩
            if filename.lower().endswith((".zip", ".rar")):
                try:
                    nested_content = self.process_archive(file_content_bytes, filename)
                    if nested_content.strip():
                        raw_answer = (
                            f"--- 嵌套压缩包 '{filename}' 内容开始 ---\n\n"
                            f"{nested_content}\n"
                            f"--- 嵌套压缩包 '{filename}' 内容结束 ---"
                        )
                except Exception as e:
                    raw_answer = f"--- 无法处理嵌套压缩文件: {filename} (错误: {e}) ---"
            else:
                raw_answer = self._get_content_from_file(filename, file_content_bytes)

            if raw_answer and raw_answer.strip():
                # 增添文件的分割
                if not filename.lower().endswith((".zip", ".rar")):
                    merged_contents.append(
                        f"--- 文件开始: {filename} ---\n\n"
                        f"{raw_answer}\n\n"
                        f"--- 文件结束: {filename} ---\n\n"
                    )
                else:
                    merged_contents.append(raw_answer)
                    
        return "".join(merged_contents)

    def process_archive(self, file_bytes: bytes, original_filename: str) -> str:
        """
        处理嵌套文件逻辑
        """
        file_type = os.path.splitext(original_filename)[1].lower()
        
        try:
            archive_buffer = io.BytesIO(file_bytes)
            if file_type == ".zip":
                with zipfile.ZipFile(archive_buffer, "r") as archive_ref:
                    return self._process_archive_items(archive_ref, archive_ref.infolist())
            
            elif file_type == ".rar":
                with rarfile.RarFile(archive_buffer, "r") as archive_ref:
                    return self._process_archive_items(archive_ref, archive_ref.infolist())
            
            else:
                return self._get_content_from_file(original_filename, file_bytes)

        except Exception as e:
            raise ValueError(f"处理文件 {original_filename} 失败: {e}")


# 创建实例
grading_service = GradingService()