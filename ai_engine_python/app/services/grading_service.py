import tempfile
import zipfile
import PyPDF2
from docx import Document
import pdfplumber
import rarfile
import io
import os
import chardet
import py7zr
import logging
import sys
import textract
# 为本地测试时使用时兜底
try:
    # 尝试作为包导入 (生产环境/FastAPI环境)
    from .ocr_service import ocr_service_instance 
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from ocr_service import ocr_service_instance
    except ImportError:
        # 如果还是找不到（比如缺少依赖），给个 Mock 防止崩坏
        print("⚠️ 警告: 无法导入 OCR 服务，将使用 Mock 对象。")
        class MockOCR:
            def get_concatenated_text(self, paths): return "[OCR 未加载]"
        ocr_service_instance = MockOCR()
        
try:
    from .file_processor import read_text_from_docx, read_text_from_pdf
except:
    def read_text_from_docx(file_bytes: bytes) -> str:
        """从.docx文件的字节流中读取文本。"""
        try:
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"读取DOCX文件时出错: {e}")
            return ""
        
    def read_text_from_pdf(file_bytes: bytes) -> str:
        """
        [增强版] 从 PDF 读取文本
        策略：pdfplumber -> 结果验证 -> (如果乱码) -> OCR
        """
        text_content = []
        
        # 辅助函数：判断提取出来的这一页是不是乱码
        def is_garbage(text):
            if not text or len(text.strip()) == 0:
                return True
            # 统计控制字符比例
            control_chars = sum(1 for c in text if ord(c) < 32 and c not in ('\n', '\r', '\t'))
            if len(text) > 0 and (control_chars / len(text)) > 0.2:
                return True
            return False

        try:
            # 1. 尝试使用 pdfplumber 解析 (比 PyPDF2 强很多)
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for i, page in enumerate(pdf.pages):
                    # 提取文本
                    page_text = page.extract_text()
                    
                    # 2. 检查提取结果
                    # 如果提取为空，或者包含大量乱码 (比如截图里的 NUL)
                    # if is_garbage(page_text):
                    #     logger.warning(f"⚠️ 第 {i+1} 页文本提取失败或为乱码，尝试 OCR 识别...")
                    #     try:
                    #         # 3. [兜底策略] 将页面渲染为图片进行 OCR
                    #         # resolution=300 保证清晰度
                    #         im = page.to_image(resolution=300).original
                    #         # lang='chi_sim+eng' 同时识别简体中文和英文
                    #         ocr_text = pytesseract.image_to_string(im, lang='chi_sim+eng')
                            
                    #         page_text = f"--- [Page {i+1} (OCR提取)] ---\n{ocr_text}\n"
                    #     except Exception as ocr_e:
                    #         logger.error(f"OCR 失败: {ocr_e}")
                    #         page_text = f"--- [Page {i+1} 解析失败] ---\n"
                    
                    text_content.append(page_text)
                    
            return "\n".join(text_content)

        except Exception as e:
            # logger.error(f"PDF解析严重错误: {e}")
            return ""

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


IGNORED_DIRS = {
    '__pycache__', 'node_modules', '.git', 
    '.idea', '.vscode', 'venv', 'env', 
    'build', 'dist', 'target', 'bin', 'obj', 'migrations', 'cmake-build-debug', '__MACOSX',
    'downloads', 'oh_modules'
}

ALLOWED_EXTENSIONS = {
    '.py', '.java', '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp',
    '.js', '.jsx', '.ts', '.tsx', '.vue', 'json',
    '.html', '.css', '.scss', '.less',
    '.go', '.rs', '.php', '.rb', '.lua', '.swift',
    '.sql', '.sh', '.bat', '.ps1', 
    '.zip', '.rar', '.7z',
    '.txt', '.md', '.docx', '.doc', '.pdf',
    '.ets',
    '.png', '.jpg', '.jpeg'  # 加入图片格式
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
        
        # 4. 删掉特殊的文件
        if filename in ['download.txt', 'output.txt', 'rfc.txt', 'test.txt', 'duopu.txt', 'sample.txt', 'jquery.js',
                        'package.json', 'package-lock.json', 'tsconfig.json', 'large-file.md',
                        'large-text.md', 'test.pdf', 'test-explorer.txt', 'test-final.md', 'filesInfo.txt'
                        ]:
            return True
            
        return False
    
    def _is_likely_text(self, text: str, threshold: float = 0.3) -> bool:
        """
        判断解码后的字符串是否像正常文本。
        如果有太多不可打印字符（乱码），则返回 False。
        """
        if not text:
            return True
            
        # 统计文本中“不可打印字符”的数量 (排除换行符、制表符等正常空白符)
        # category 'Cc' 代表 Control characters (如 NUL, SOH 等)
        # category 'Cn' 代表 Not assigned (通常是乱码)
        import unicodedata
        
        non_printable_count = 0
        check_length = min(len(text), 1000) # 只检查前1000个字符以提高效率
        
        for char in text[:check_length]:
            if char in ('\n', '\r', '\t'):
                continue
            category = unicodedata.category(char)
            if category.startswith('C') or category == 'Co':
                non_printable_count += 1
        
        ratio = non_printable_count / check_length
        return ratio < threshold
    
    def _get_content_from_file(self, filename: str, file_bytes: bytes) -> str:
        """
        由于学生很可能私自更改文件格式，导致读取错误
        在这部分加入兜底策略，一步步试探文件正确格式
        
        :param self: Description
        :param filename: Description
        :type filename: str
        :param file_bytes: Description
        :type file_bytes: bytes
        :return: Description
        :rtype: str
        """
        lower_filename = filename.lower()
        
        if not file_bytes:
            return ""

        # 策略1：优先基于文件头的纠错机制
        
        # 1.1 检测是否为被改名的 Zip/Docx (以 PK 开头)
        if file_bytes.startswith(b'PK\x03\x04'):
            # 很多学生会把 .docx 改名为 .ts 或 .c 交上来
            # 尝试作为 docx 解析
            try:
                logger.info(f"🔍 [智能修正] 检测到 {filename} 可能是伪装的 docx，尝试解析...")
                text = read_text_from_docx(file_bytes)
                if text and len(text.strip()) > 0:
                     return f"【系统提示：该文件名为 {filename}，但检测到其实际为 Office/Zip 格式。已尝试提取文本内容：】\n\n{text}"
            except Exception as e:
                logger.debug(f"Docx 解析尝试失败: {e}")
        
        # 1.2 检测是否为被改名的 PDF (以 %PDF 开头)
        if file_bytes.startswith(b'%PDF'):
            try:
                logger.info(f"🔍 [智能修正] 检测到 {filename} 可能是伪装的 PDF，尝试解析...")
                text = read_text_from_pdf(file_bytes)
                return f"【系统提示：该文件名为 {filename}，但检测到其实际为 PDF 格式。已尝试提取文本内容：】\n\n{text}"
            except:
                pass

        # 1.3 检测是否为二进制 Word doc (以 0xD0CF11E0 开头)
        if file_bytes.startswith(b'\xD0\xCF\x11\xE0'):
             return self._read_text_from_doc(file_bytes, filename)

        # 正常流程：按后缀名处理

        if lower_filename.endswith(".doc"):
            return self._read_text_from_doc(file_bytes, filename)

        if lower_filename.endswith(".docx"):
            return read_text_from_docx(file_bytes)
        
        if lower_filename.endswith(".pdf"):
            return read_text_from_pdf(file_bytes)
        
        image_extensions = ('.png', '.jpg', '.jpeg')
        if lower_filename.endswith(image_extensions):
            if not ocr_service_instance:
                return f"【系统提示：检测到图片文件 {filename}，但OCR服务未启动，无法读取。】"
            
            try:
                # 写入临时文件供 OCR 服务读取
                suffix = os.path.splitext(filename)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(file_bytes)
                    tmp_path = tmp_file.name
                
                logger.info(f"🖼️ [OCR] 正在识别独立图片: {filename}")
                ocr_text = ocr_service_instance.get_concatenated_text([tmp_path])
                
                # 清理临时文件
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
                return f"--- [图片文件内容 (OCR): {filename}] ---\n{ocr_text}\n"
            except Exception as e:
                logger.error(f"图片OCR失败 {filename}: {e}")
                return f"【系统提示：图片 {filename} 识别失败: {e}】"
            
        # 策略2：文本解码与乱码过滤
        
        decoded_text = ""
        encoding_used = "utf-8"
        
        try:
            decoded_text = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                decoded_text = file_bytes.decode('gbk')
                encoding_used = "gbk"
            except:
                # 最后的尝试：忽略错误解码
                decoded_text = file_bytes.decode('utf-8', errors='ignore')
                encoding_used = "utf-8-ignore"

        # 核心兜底：检查解码后的内容是否包含大量控制字符（即截图中的 NUL, ETX 等）
        if not self._is_likely_text(decoded_text):
            logger.warning(f"⚠️ [乱码拦截] {filename} 解码后包含大量二进制控制字符，已忽略。")
            return f"【系统提示：文件 {filename} 无法作为文本读取。它可能是一个二进制文件（如编译后的程序、图片或加密文件）被错误重命名了。】"

        return decoded_text

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
                    
                    # 如果连字符数量是 3 或 4 个，则跳过系统生成的doc文件
                    if 3 <= hyphen_count <= 4:
                        logger.info(f"🗑️ [规则过滤] 跳过特定格式Doc文件(含{hyphen_count}个'-'): {filename.lower()}")
                        continue
            basename = os.path.basename(filename).lower()
                    
                    # 2. 定义黑名单
            BLOCK_LIST = ['axios.zip', 'node_modules.zip', 'vendor.zip']

            # 3. 检查纯文件名是否在黑名单中
            if basename in BLOCK_LIST:
                logger.info(f"🚫 [忽略-黑名单压缩包] {filename}")
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
        """
        专门遍历和解析 .7z 压缩包内的项目
        使用 extractall + TemporaryDirectory 替代已移除的 readall() 方法
        """
        merged_contents = []
        
        # 使用临时目录进行解压，处理完后自动清理
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # 将所有文件解压到临时目录
                archive_ref.extractall(path=tmp_dir)
            except Exception as e:
                logger.error(f"7z解压失败: {e}")
                return f"--- 7z解压失败: {e} ---\n"

            # 遍历临时目录下的所有文件
            for root, dirs, files in os.walk(tmp_dir):
                # 排序确保处理顺序一致
                for filename in sorted(files):
                    full_path = os.path.join(root, filename)
                    
                    # 计算相对路径（去除临时目录前缀），用于展示文件名
                    rel_path = os.path.relpath(full_path, tmp_dir).replace("\\", "/")
                    
                    # 过滤掉不需要的文件
                    if rel_path.startswith("__MACOSX") or ".DS_Store" in rel_path:
                        continue
                    
                    # 基础黑名单/白名单过滤 (复用已有的 is_ignored_file 逻辑)
                    if self.is_ignored_file(rel_path):
                        continue

                    try:
                        # 读取文件内容为 bytes
                        with open(full_path, 'rb') as f:
                            file_content_bytes = f.read()
                        
                        raw_answer = ""

                        # 递归处理嵌套压缩包
                        if rel_path.lower().endswith((".zip", ".rar", ".7z")):
                            try:
                                nested_content = self.process_archive(file_content_bytes, rel_path)
                                if nested_content.strip():
                                    raw_answer = (
                                        f"--- 嵌套压缩包 '{rel_path}' 内容开始 ---\n\n"
                                        f"{nested_content}\n"
                                        f"--- 嵌套压缩包 '{rel_path}' 内容结束 ---"
                                    )
                            except Exception as e:
                                raw_answer = f"--- 无法处理嵌套压缩文件: {rel_path} (错误: {e}) ---"
                        else:
                            # 普通文件处理：调用已有的内容提取逻辑
                            raw_answer = self._get_content_from_file(rel_path, file_content_bytes)

                        # 只有非空内容才添加
                        if raw_answer and raw_answer.strip():
                            if not rel_path.lower().endswith((".zip", ".rar", ".7z")):
                                merged_contents.append(
                                    f"--- 文件开始: {rel_path} ---\n\n"
                                    f"{raw_answer}\n\n"
                                    f"--- 文件结束: {rel_path} ---\n\n"
                                )
                            else:
                                merged_contents.append(raw_answer)
                                
                    except Exception as e:
                        logger.error(f"读取解压后的文件失败 {rel_path}: {e}")
                        merged_contents.append(f"--- 读取失败: {rel_path} ({e}) ---\n")

        return "".join(merged_contents)

    def process_archive(self, file_bytes: bytes, original_filename: str, depth=0) -> str:
        """
        入口函数：处理压缩包，包含文件后缀伪造的智能兜底策略。
        逻辑：优先尝试后缀名对应的格式，失败后轮询其他格式。
        """
        import io
        import zipfile
        import rarfile
        import py7zr

        ext = os.path.splitext(original_filename)[1].lower()
        logger.info(f"--- 开始处理文件 (Depth={depth}): {original_filename} ---")
        
        buffer = io.BytesIO(file_bytes)

        # 同样因为有些学生私自更改文件后缀名导致提取失败，使用兜底策略
        
        def try_zip():
            try:
                buffer.seek(0) # 关键：重置指针
                # 预检查，快速失败
                if not zipfile.is_zipfile(buffer):
                    return None
                buffer.seek(0)
                with zipfile.ZipFile(buffer, "r") as z:
                    return self._process_archive_items(z, z.infolist(), depth)
            except Exception:
                return None

        def try_rar():
            try:
                buffer.seek(0)
                # rarfile 没有像 is_zipfile 那么方便的预检查，直接尝试打开
                with rarfile.RarFile(buffer, "r") as r:
                    return self._process_archive_items(r, r.infolist(), depth)
            except Exception:
                return None

        def try_7z():
            try:
                buffer.seek(0)
                if not py7zr.is_7zfile(buffer):
                    return None
                buffer.seek(0)
                with py7zr.SevenZipFile(buffer, 'r') as z:
                    return self._process_7z_items(z)
            except Exception:
                return None

        # --- 策略编排 ---

        # 1. 如果不是常见的压缩包后缀，直接当做普通文件读取
        if ext not in ['.zip', '.rar', '.7z']:
            return self._get_content_from_file(original_filename, file_bytes)

        # 2. 根据后缀名决定尝试顺序
        strategies = []
        if ext == ".zip":
            strategies = [("zip", try_zip), ("rar", try_rar), ("7z", try_7z)]
        elif ext == ".rar":
            strategies = [("rar", try_rar), ("zip", try_zip), ("7z", try_7z)]
        elif ext == ".7z":
            strategies = [("7z", try_7z), ("zip", try_zip), ("rar", try_rar)]
        
        # 3. 执行策略链
        errors = []
        for fmt_name, handler in strategies:
            result = handler()
            if result is not None:
                # 成功解压
                if f".{fmt_name}" != ext:
                    logger.warning(f"文件 {original_filename} 后缀为 {ext} 但实际是 {fmt_name} 格式")
                return result
            else:
                errors.append(fmt_name)

        # 4. 所有策略都失败了
        error_msg = f"处理异常: 无法解压文件 {original_filename}。已尝试格式: {', '.join(errors)}。文件可能已损坏。"
        logger.error(error_msg)
        return error_msg
        
grading_service = GradingService()

# 本地测试
if __name__ == "__main__":
    # 输入文件路径
    TARGET = "/root/autodl-tmp/dzq/homework/test_10.zip" 
    # 输出文件路径
    OUTPUT_FILE = "/root/autodl-tmp/dzq/homework/extraction_result.txt"
    
    TEST_DEPTH = 0

    if os.path.exists(TARGET):
        with open(TARGET, "rb") as f:
            data = f.read()
            svc = GradingService()
            result = svc.process_archive(data, TARGET, depth=TEST_DEPTH)
            
            try:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
                    out_f.write(result)
                print(f"\n✅ 成功！结果已保存至: {os.path.abspath(OUTPUT_FILE)}")
            except Exception as e:
                print(f"\n❌ 保存文件失败: {e}")
            # --- 修改部分结束 ---

            print("\n" + "="*30)
            print(f"最终提取结果长度: {len(result)} 字符")
            if len(result) < 500:
                print("内容预览:\n", result)
            else:
                print("内容太长，已省略打印。")
            print("="*30)
    else:
        print(f"❌ 找不到文件: {TARGET}")