import zipfile
import rarfile
import io
import os
import docx
import chardet 
import py7zr

# 导入文件处理函数
from .file_processor import read_text_from_docx, read_text_from_pdf


class GradingService:
    """首先处理所有文本信息，并合并，给出该作业的评分"""

    # 使用chardet自己判断文件编码方式
    def _get_content_from_file(self, filename: str, file_bytes: bytes) -> str:
        """
        智能提取文本内容，能处理不同格式的文件和多种文本编码。
        """
        lower_filename = filename.lower()
        
        if not file_bytes:
            return ""

        if lower_filename.endswith(".docx"):
            return read_text_from_docx(file_bytes)
        
        if lower_filename.endswith(".pdf"):
            return read_text_from_pdf(file_bytes)
            
        text_extensions = [
            ".txt", ".py", ".js", ".java", ".c", ".cpp", 
            ".h", ".md", ".html", ".css"
        ]

        if any(lower_filename.endswith(ext) for ext in text_extensions):
            try:
                # 使用 chardet 检测文件的编码
                detection = chardet.detect(file_bytes)
                encoding = detection['encoding']
                
                # 如果检测结果可信度较低或未检测到，则默认使用 utf-8
                if encoding is None or detection['confidence'] < 0.9:
                    encoding = 'utf-8'

                # 使用检测到的编码进行解码
                # 使用 errors='ignore' 可以在遇到个别无法解码的字符时跳过，避免整个文件读取失败
                return file_bytes.decode(encoding, errors='ignore')

            except Exception as e:
                # 如果 chardet 或解码过程出现意外错误，提供一个最终的备用方案
                print(f"智能解码文件 {filename} 时失败: {e}，尝试使用备用方案解码...")
                try:
                    return file_bytes.decode('utf-8', errors='ignore')
                except:
                    return file_bytes.decode('gbk', errors='ignore')
                
        return ""
    
    # 新增一个专门处理 .7z 文件的函数
    def _process_7z_items(self, archive_ref) -> str:
        """专门遍历和解析 .7z 压缩包内的项目"""
        merged_contents = []
        
        # py7zr 使用 readall() 一次性读取所有文件到内存
        all_files = archive_ref.readall()
        
        # archive_ref.list() 用于获取文件元信息列表
        for item_info in sorted(archive_ref.list(), key=lambda x: x.filename):
            if item_info.is_directory:
                continue

            filename = item_info.filename
            if filename.startswith("__MACOSX/") or os.path.basename(filename) == ".DS_Store":
                continue

            # 从 all_files 字典中获取文件内容
            file_obj = all_files[filename]
            file_content_bytes = file_obj.read()
            raw_answer = ""

            if filename.lower().endswith((".zip", ".rar", ".7z")):
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
                if not filename.lower().endswith((".zip", ".rar", ".7z")):
                    merged_contents.append(
                        f"--- 文件开始: {filename} ---\n\n"
                        f"{raw_answer}\n\n"
                        f"--- 文件结束: {filename} ---\n\n"
                    )
                else:
                    merged_contents.append(raw_answer)
                    
        return "".join(merged_contents)

    def _process_archive_items(self, archive_ref, item_infos) -> str:
        """Iterates through items in an archive, extracts content, and merges it."""
        merged_contents = []
        
        for item_info in sorted(item_infos, key=lambda x: x.filename):
            if hasattr(item_info, 'is_dir') and item_info.is_dir(): # 兼容rarfile和zipfile
                continue
            if not hasattr(item_info, 'is_dir') and item_info.file_size == 0:
                continue

            # 尝试解码文件名以处理中文乱码
            try:
                filename = item_info.filename.encode('cp437').decode('gbk')
            except:
                filename = item_info.filename

            if filename.startswith("__MACOSX/") or os.path.basename(filename) == ".DS_Store":
                continue

            file_content_bytes = archive_ref.read(item_info)
            raw_answer = ""

            # 增加对于7z文件的处理
            if filename.lower().endswith((".zip", ".rar", '.7z')):
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
                
            elif file_type == ".7z":
                with py7zr.SevenZipFile(archive_buffer, 'r') as archive_ref:
                    return self._process_7z_items(archive_ref)
                
            # .doc/.docx 等非压缩文件，统一由 _get_content_from_file 处理
            else:
                return self._get_content_from_file(original_filename, file_bytes)

        except Exception as e:
            raise ValueError(f"处理文件 {original_filename} 失败: {e}")


# 创建实例
grading_service = GradingService()