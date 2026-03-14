import os
import logging
import threading
from typing import List, Dict, Any, Tuple

# 🚨 核心优化：彻底镇压 Paddle C++ 底层的烦人警告
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_WARNINGS'] = '0'

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRService:
    """
    回归最高效的 GPU 独占模式：单例 + 线程锁。
    摒弃多进程引发的 GPU 频繁上下文切换风暴，让单块 GPU 跑出最极致的串行处理速度。
    """
    _instance = None
    _lock = threading.Lock() # 类级别的锁，确保多线程并发时 C++ 底层绝对安全

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(OCRService, cls).__new__(cls)
                    cls._instance._init_ocr()
        return cls._instance

    def _init_ocr(self):
        logger.info("正在初始化单例 OCR 引擎 (独占极速模式)...")
        try:
            from paddleocr import PaddleOCR
            import logging as pd_logging
            pd_logging.getLogger('ppocr').setLevel(pd_logging.ERROR)
            
            # 加载唯一的 OCR 引擎 (已修复 show_log 兼容性问题)
            self.ocr = PaddleOCR(
                use_doc_orientation_classify=True, 
                use_doc_unwarping=True,            
                use_textline_orientation=True,     
                lang='ch'
            )
            logger.info("✅ 单例 OCR 引擎加载就绪！随时准备极速处理。")
        except Exception as e:
            logger.error(f"OCR 初始化失败: {e}", exc_info=True)
            raise RuntimeError("无法初始化 PaddleOCR 服务") from e

    def process_image(self, file_path: str) -> List[Tuple[Any, Tuple[str, float]]]:
        """对单张图片进行 OCR"""
        # 🚨 极度关键：使用线程锁保护 C++ 指针，让到达的 gRPC 请求自动排队
        with self._lock:
            try:
                result_pages = self.ocr.predict(file_path)
                if not result_pages or not result_pages[0]:
                    return []

                data_dict = result_pages[0]
                texts = data_dict.get('rec_texts', [])
                scores = data_dict.get('rec_scores', [])
                polys = data_dict.get('rec_polys', []) 

                if not (len(texts) == len(scores) == len(polys)):
                    min_len = min(len(texts), len(scores), len(polys))
                    texts = texts[:min_len]
                    scores = scores[:min_len]
                    polys = polys[:min_len]

                reconstructed_results = []
                for i in range(len(texts)):
                    reconstructed_results.append([polys[i], (texts[i], scores[i])])
                return reconstructed_results
            except Exception as e:
                logger.error(f"OCR 进程处理报错: {e}")
                return []

    def get_concatenated_text(self, image_path_list: List[str]) -> str:
        """批量处理多张图片并拼接文本"""
        all_text = []
        for i, img_path in enumerate(image_path_list):
            all_text.append(f"--- 图片 {i+1} 开始 ---")
            
            # 这里调用 process_image 时会自动触发排队锁
            results = self.process_image(img_path)
            
            if not results:
                all_text.append("[未识别到文本]")
            else:
                for line in results:
                    text, confidence = line[1]
                    all_text.append(text)
            all_text.append(f"--- 图片 {i+1} 结束 ---")
            
        return "\n".join(all_text)


# =====================================================================
# 预实例化单例：在 gRPC 服务启动时就预热好模型
# =====================================================================
try:
    ocr_service_instance = OCRService()
except Exception as e:
    logger.error(f"创建 OCR 服务失败: {e}")
    ocr_service_instance = None 

if __name__ == '__main__':
    logger.info("OCR 模块已加载")