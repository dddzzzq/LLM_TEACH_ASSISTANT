import tempfile
import zipfile
import rarfile
import io
import os
import chardet
import py7zr
import logging
import sys
import textract
from .file_processor import read_text_from_docx, read_text_from_pdf

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


IGNORED_DIRS = {
    '__pycache__', 'node_modules', '.git', 
    '.idea', '.vscode', 'venv', 'env', 
    'build', 'dist', 'target', 'bin', 'obj', 'migrations', 'cmake-build-debug', '__MACOSX'
}

ALLOWED_EXTENSIONS = {
    '.py', '.java', '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp',
    '.js', '.jsx', '.ts', '.tsx', '.vue', 'json',
    '.html', '.css', '.scss', '.less',
    '.go', '.rs', '.php', '.rb', '.lua', '.swift',
    '.sql', '.sh', '.bat', '.ps1', 
    '.zip', '.rar', '.7z'
    '.txt', '.md', '.docx', '.doc', '.pdf' 
}

PRIORITY_EXTENSIONS = {
    '.py', '.java', '.c', '.cpp', '.js', '.ts', '.pdf', '.docx', '.md', '.txt'
}

class GradingService:
    def _read_text_from_doc(self, file_bytes: bytes, filename: str) -> str:
        """
        [辅助方法] 从 .doc 文件中提取文本。
        支持：Word 2003 XML (伪装) 和 Word 97-2003 二进制 (textract)
        """

        if textract:
            # textract 需要具体的文件路径，所以我们创建一个临时文件
            suffix = ".doc"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            
            try:
                # 调用 textract (底层调用 antiword)
                # encoding 参数防止输出 bytes
                text_bytes = textract.process(tmp_path)
                return text_bytes.decode("utf-8")
            except Exception as e:
                logger.error(f"Textract 解析失败 {filename}: {e}")
                return f"【系统提示：该 .doc 文件无法解析。可能是文件损坏或加密。错误: {str(e)}】"
            finally:
                # 清理临时文件
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except:
                        pass
        else:
            return "【系统提示：服务器未安装 textract 库或底层依赖(antiword)，无法解析二进制 .doc】"
        
    def is_ignored_file(self, file_path: str) -> bool:
        """
        判断文件是否应该被忽略，并打印原因用于调试
        """
        file_path = file_path.replace('\\', '/')
        parts = file_path.split('/')
        filename = os.path.basename(file_path)
        
        # 1. 检查目录黑名单
        for part in parts:
            if part in IGNORED_DIRS:
                logger.debug(f"🚫 [忽略-目录] {file_path} (目录 '{part}' 在黑名单中)")
                return True
                
        # 2. 忽略隐藏文件
        if filename.startswith('.') and filename != '.gitignore':
            logger.debug(f"🚫 [忽略-隐藏] {file_path}")
            return True

        # 3. 后缀白名单检查
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in ALLOWED_EXTENSIONS:
            # 日志观察是否删掉文件
            logger.debug(f"🚫 [忽略-后缀] {file_path} (后缀 '{ext}' 不在白名单)")
            return True
            
        return False

    def _get_content_from_file(self, filename: str, file_bytes: bytes) -> str:
        lower_filename = filename.lower()
        
        if not file_bytes:
            return ""

        if lower_filename.endswith(".doc"):
            return self._read_text_from_doc(file_bytes, filename)

        if lower_filename.endswith(".docx"):
            return read_text_from_docx(file_bytes)
        
        if lower_filename.endswith(".pdf"):
            return read_text_from_pdf(file_bytes)
            
        # 文本/代码处理
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return file_bytes.decode('gbk')
            except:
                return file_bytes.decode('utf-8', errors='ignore')

    def _process_archive_items(self, archive_ref, item_infos, depth) -> str:
        merged_contents = []
        
        # 预处理：过滤掉目录本身
        file_items = []
        for item in item_infos:
            # 兼容 zipfile 和 rarfile 的目录判断
            is_dir = getattr(item, 'is_dir', False)
            if callable(is_dir): is_dir = is_dir()
            if item.filename.endswith('/'): is_dir = True
            
            if not is_dir:
                # 修复中文文件名乱码 (Zip特有)
                fname = item.filename
                try:
                    fname = fname.encode('cp437').decode('gbk')
                except:
                    pass
                file_items.append({'info': item, 'filename': fname})

        # 排序
        def sort_priority(item):
            _, ext = os.path.splitext(item['filename'])
            return 0 if ext.lower() in PRIORITY_EXTENSIONS else 1
        file_items.sort(key=sort_priority)

        for item in file_items:
            filename = item['filename']
            item_info = item['info']

            # 1. 基础过滤 (白名单/黑名单)
            if self.is_ignored_file(filename):
                continue

            # 2. 过滤掉系统生成的文件
            if filename.lower().endswith('.doc'):
                    # 计算文件名中连字符 '-' 的出现次数
                    hyphen_count = filename.lower().count('-')
                    
                    # 如果连字符数量是 3 或 4 个，则跳过
                    if 3 <= hyphen_count <= 4:
                        logger.info(f"🗑️ [规则过滤] 跳过特定格式Doc文件(含{hyphen_count}个'-'): {filename.lower()}")
                        continue

            try:
                # 读取内容
                if hasattr(archive_ref, 'read'):
                    content_bytes = archive_ref.read(item_info)
                else:
                    content_bytes = b"" 

                raw_answer = ""
                # 递归处理嵌套压缩包
                if filename.lower().endswith((".zip", ".rar", ".7z")):
                    logger.info(f"📂 发现嵌套压缩包: {filename}，正在解压...")
                    raw_answer = self.process_archive(content_bytes, filename, depth=depth + 1)
                else:
                    # 读取文件内容
                    raw_answer = self._get_content_from_file(filename, content_bytes)

                # 只有非空内容才添加
                if raw_answer and raw_answer.strip():
                    logger.info(f"✅ [提取成功] {filename}")
                    merged_contents.append(
                        f"--- 文件开始: {filename} ---\n{raw_answer}\n--- 文件结束: {filename} ---\n"
                    )
                else:
                    logger.warning(f"⚠️ [内容为空] {filename} (提取后为空)")

            except Exception as e:
                logger.error(f"❌ 读取错误 {filename}: {e}")
                merged_contents.append(f"--- 读取失败: {filename} ({e}) ---\n")
                    
        return "\n".join(merged_contents)
    
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

    def process_archive(self, file_bytes: bytes, original_filename: str, depth=0) -> str:
        """入口函数"""
        file_type = os.path.splitext(original_filename)[1].lower()
        logger.info(f"--- 开始处理压缩包 (Depth={depth}): {original_filename} ---")
        
        try:
            buffer = io.BytesIO(file_bytes)
            if file_type == ".zip":
                with zipfile.ZipFile(buffer, "r") as z:
                    return self._process_archive_items(z, z.infolist(), depth)
            elif file_type == ".rar":
                with rarfile.RarFile(buffer, "r") as r:
                    return self._process_archive_items(r, r.infolist(), depth)
            elif file_type == ".7z":
                with py7zr.SevenZipFile(buffer, 'r') as archive_ref:
                    return self._process_7z_items(archive_ref)
            else:
                return self._get_content_from_file(original_filename, file_bytes)

        except Exception as e:
            return f"处理异常: {e}"
        
grading_service = GradingService()

# 本地测试
if __name__ == "__main__":
    TARGET = "/root/autodl-tmp/dzq/LLM_TEACH_ASSISTANT/backend/app/dataset/homework/2.zip" 
    
    TEST_DEPTH = 0

    if os.path.exists(TARGET):
        with open(TARGET, "rb") as f:
            data = f.read()
            svc = GradingService()
            result = svc.process_archive(data, TARGET, depth=TEST_DEPTH)
            
            print("\n" + "="*30)
            print(f"最终提取结果长度: {len(result)} 字符")
            if len(result) < 500:
                print("内容预览:\n", result)
            else:
                print("内容太长，已省略打印。")
            print("="*30)
    else:
        print(f"❌ 找不到文件: {TARGET}")