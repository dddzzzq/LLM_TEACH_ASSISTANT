import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, Tuple
import os
import re

class AIGCDetectorService:
    """
    一个使用双预训练模型检测AI生成内容的的服务。
    - 使用中文模型检测文字报告。
    - 使用英文/代码模型检测源代码。
    """
    def __init__(self):
        """
        初始化服务并加载两个专门的模型和分词器。
        """
        self.device = 0 if torch.cuda.is_available() else -1
        
        # --- 定义两个模型 ---
        self.prose_model_name = "Hello-SimpleAI/chatgpt-detector-roberta-chinese" # 用于文字报告
        self.code_model_name = "Hello-SimpleAI/chatgpt-detector-roberta"       # 用于源代码
        
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # --- 加载两个分类器 ---
        self.prose_classifier = self._load_classifier(self.prose_model_name, cache_dir)
        self.code_classifier = self._load_classifier(self.code_model_name, cache_dir)

    def _load_classifier(self, model_name: str, cache_dir: str):
        """辅助函数，用于加载单个分类器模型。"""
        print(f"正在加载模型 '{model_name}' 到设备: {'cuda:0' if self.device == 0 else 'cpu'}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
            
            classifier = pipeline(
                "text-classification", 
                model=model,
                tokenizer=tokenizer,
                device=self.device
            )
            print(f"模型 '{model_name}' 加载成功。")
            return classifier
        except Exception as e:
            print(f"加载模型 '{model_name}' 出错: {e}")
            return None

    def _separate_content(self, merged_text: str) -> Tuple[str, str]:
        """
        从合并的文本中分离出文字报告和源代码。
        """
        prose_parts = []
        code_parts = []
        
        prose_extensions = ['.txt', '.md', '.docx', '.pdf', '.doc']
        file_blocks = re.split(r'--- 文件开始: (.*?) ---', merged_text)
        
        if len(file_blocks) < 2:
            return merged_text, ""

        for i in range(1, len(file_blocks), 2):
            filename = file_blocks[i].strip()
            content = file_blocks[i+1]
            content = re.sub(r'--- 文件结束: (.*?) ---\n\n', '', content)
            
            _, ext = os.path.splitext(filename)
            if ext.lower() in prose_extensions:
                prose_parts.append(content)
            else:
                code_parts.append(content)
                
        return "\n".join(prose_parts), "\n".join(code_parts)

    def _detect_single_part(self, text: str, content_type: str) -> Dict[str, Any]:
        """
        对单个文本块进行检测的核心逻辑，会根据类型选择不同的模型。
        """
        classifier = self.prose_classifier if content_type == 'prose' else self.code_classifier

        if not classifier:
            return {"error": f"{content_type.capitalize()} 模型不可用。"}
        
        if not text or len(text.strip()) < 50:
            return {"predicted_label": "人类写作", "confidence": 1.0, "ai_probability": 0.0}

        try:
            prediction = classifier(text, truncation=True, max_length=512)[0]
            
            # 不同模型的标签可能不同，这里做个兼容
            # 'LABEL_0' 或 'Fake' 通常代表AI
            is_ai_label = prediction['label'] in ['LABEL_0', 'Fake']
            label = "AI生成" if is_ai_label else "人类写作"
            
            ai_score = prediction['score'] if is_ai_label else 1 - prediction['score']
            ai_score *= 100

            return {
                "predicted_label": label,
                "confidence": round(prediction['score'], 4),
                "ai_probability": round(ai_score, 4)
            }
        except Exception as e:
            print(f"AIGC检测过程中出错 ({content_type}): {e}")
            return {"error": f"分析过程中发生错误 ({content_type})"}

    def detect(self, merged_text: str) -> Dict[str, Any]:
        """
        检测合并后的文本。
        该方法会先将文本分离为报告和代码，用不同模型分别检测，然后返回风险最高的结果。
        """
        prose_content, code_content = self._separate_content(merged_text)
        
        prose_result = self._detect_single_part(prose_content, 'prose')
        code_result = self._detect_single_part(code_content, 'code')
        
        if "error" in prose_result: return prose_result
        if "error" in code_result: return code_result
        
        if prose_result['ai_probability'] >= code_result['ai_probability']:
            final_result = prose_result
            final_result['detection_source'] = '文字报告'
        else:
            final_result = code_result
            final_result['detection_source'] = '源代码'
            
        return final_result

# 创建服务的实例
aigc_detector_service = AIGCDetectorService()
