from paddleocr import PaddleOCR
from typing import List, Dict, Any, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRService:
    """
    封装PaddleOCR，提供一个单例或易于实例化的服务
    """
    _instance = None

    def __init__(self):
        logger.info("正在初始化 OCR 服务...")
        try:
            # 按照用户提供的 2024-03-28 示例代码 和 JSON输出 进行初始化
            self.ocr = PaddleOCR(
                use_doc_orientation_classify=True, # 文本图像预处理
                use_doc_unwarping=True,            # 文本图像预处理
                use_textline_orientation=True,     # 文本行方向分类 (来自用户JSON)
                lang='ch'
            )
            logger.info("OCR 服务已成功初始化。")
        except Exception as e:
            logger.error(f"OCR 初始化失败: {e}", exc_info=True)
            raise RuntimeError("无法初始化 PaddleOCR 服务") from e

    def process_image(self, file_path: str) -> List[Tuple[Any, Tuple[str, float]]]:
        """
        对单个图片文件路径进行OCR识别
        :param file_path: 图片的文件路径
        :return: 格式化为 [bbox, (text, confidence)] 的列表
        """
        try:
            # 按照用户指示，使用 predict 方法 (接受图片路径)
            result_pages = self.ocr.predict(file_path)
            
            if not result_pages or not result_pages[0]:
                return []

            
            # --- 错误修复 ---
            # V7日志显示 predict 返回一个列表，列表的[0]元素是包含 rec_texts 的字典
            # 它没有 .data 属性
            data_dict = result_pages[0]
            
            # 从返回的字典中提取所需列表
            texts = data_dict.get('rec_texts', [])
            scores = data_dict.get('rec_scores', [])
            polys = data_dict.get('rec_polys', []) # 这是检测框 (dt_boxes 的识别对应)
            # --- 修复结束 ---

            if not (len(texts) == len(scores) == len(polys)):
                logger.warning("OCR result lists have mismatched lengths.")
                min_len = min(len(texts), len(scores), len(polys))
                texts = texts[:min_len]
                scores = scores[:min_len]
                polys = polys[:min_len]

            # 重构为 [bbox, (text, confidence)] 格式，以兼容后续服务
            reconstructed_results = []
            for i in range(len(texts)):
                reconstructed_results.append(
                    [polys[i], (texts[i], scores[i])]
                )
            
            return reconstructed_results
        except Exception as e:
            logger.error(f"OCR 识别过程中出错: {e}", exc_info=True)
            return []

    def get_concatenated_text(self, image_path_list: List[str]) -> str:
        """
        处理多张图片，合并所有识别到的文本
        :param image_path_list: 多个图片的路径列表
        :return: 合并后的所有文本，按图片顺序和文本框顺序排列
        """
        all_text = []
        for i, img_path in enumerate(image_path_list):
            all_text.append(f"--- 图片 {i+1} 开始 ---")
            results = self.process_image(img_path) # 该方法已返回 [bbox, (text, score)] 列表
            
            if not results:
                all_text.append("[未识别到文本]")
                all_text.append(f"--- 图片 {i+1} 结束 ---")
                continue

            # 假设PaddleOCR按从上到下的顺序返回结果
            for line in results:
                text, confidence = line[1]
                all_text.append(text)
            
            all_text.append(f"--- 图片 {i+1} 结束 ---")
        
        return "\n".join(all_text)


# 创建一个全局实例供其他服务导入和使用
# 这有助于避免每次请求都重新加载模型
try:
    ocr_service_instance = OCRService()
except Exception as e:
    logger.error(f"创建全局OCR服务实例失败: {e}")
    ocr_service_instance = None # 允许应用启动，但在使用时会失败

if __name__ == '__main__':
    # 用于测试服务是否正常工作
    logger.info("正在测试 OCR 服务...")
    if ocr_service_instance:
        try:
            # 你需要一张测试图片 'test.jpg' 放在同目录下
            test_image_path = "test.jpg" 
            
            # 模仿用户提供的示例代码 (使用路径)
            logger.info(f"正在使用 predict(path='{test_image_path}')...")
            result_pages = ocr_service_instance.ocr.predict(test_image_path)
            
            if result_pages:
                logger.info("PaddleOCR.predict() 原始结果 (res.print()):")
                for res in result_pages:
                    # res 是一个 OCRResult 对象
                    res.print() # 模仿用户的示例
                    
                    # 检查 .data 属性
                    if hasattr(res, 'data') and res.data:
                        texts = res.data.get('rec_texts', [])
                        logger.info(f"--- 从 res.data 提取的文本 ({len(texts)} 行) ---")
                        for text in texts:
                            logger.info(text)
                    else:
                        logger.warning("未在 OCRResult 中找到 .data 属性")
            else:
                logger.info("未识别到文本。")

            # 测试服务的多图字节流合并功能
            full_text = ocr_service_instance.get_concatenated_text([test_image_path, test_image_path])
            logger.info("\n--- 合并测试 (get_concatenated_text) ---")
            logger.info(full_text)
            
        except FileNotFoundError:
            logger.error(f"测试失败：未找到 {test_image_path}。请在 services 目录下放置一张 {test_image_path} 图片。")
        except Exception as e:
            logger.error(f"测试时发生错误: {e}", exc_info=True)
    else:
        logger.error("OCR 服务未初始化，无法运行测试。")