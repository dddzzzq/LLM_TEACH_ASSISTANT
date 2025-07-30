import re
from typing import Dict, List, Optional, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ..schemas.models import PlagiarismReport
from .deepseek_service import deepseek_service
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


class PlagiarismService:
    """
    修改后的抄袭检测服务，采用双模型策略。
    在每个学生提交的内容中分离文本和代码，并进行同类内容的交叉对比。
    """
    def __init__(self):
        # 加载对应的模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.text_model_name = r"D:\DZQ\项目\教改项目-批改Agent\models\bert-base-chinese"
        self.code_model_name = "microsoft/unixcoder-base"
        
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"正在加载文本模型: {self.text_model_name}")
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name, cache_dir=cache_dir)
        self.text_model = AutoModel.from_pretrained(self.text_model_name, cache_dir=cache_dir).to(self.device)
        
        print(f"正在加载代码模型: {self.code_model_name}")
        self.code_tokenizer = AutoTokenizer.from_pretrained(self.code_model_name, cache_dir=cache_dir)
        self.code_model = AutoModel.from_pretrained(self.code_model_name, cache_dir=cache_dir).to(self.device)

    def _get_embedding(self, text: str, model_type: str) -> np.ndarray:
        """通用的调用模型得到嵌入的方法"""
        if not text:
            return np.zeros((1, 768))
        tokenizer = self.text_tokenizer if model_type == 'text' else self.code_tokenizer
        model = self.text_model if model_type == 'text' else self.code_model
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding

    def _separate_content_for_each_student(self, submissions: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """根据每个学生的内容，分离出文本和代码"""
        separated_data = {}
        prose_extensions = ['.txt', '.md', '.docx', '.pdf', '.doc']
        for student_id, merged_content in submissions.items():
            prose_parts, code_parts = [], []
            file_blocks = re.split(r'--- 文件开始: (.*?) ---', merged_content)
            if len(file_blocks) < 2:
                prose_parts.append(merged_content)
            else:
                for i in range(1, len(file_blocks), 2):
                    filename = file_blocks[i].strip()
                    content = re.sub(r'--- 文件结束: (.*?) ---\n\n', '', file_blocks[i+1]).strip()
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in prose_extensions:
                        prose_parts.append(content)
                    else:
                        code_parts.append(content)
            separated_data[student_id] = {"text": "\n".join(prose_parts), "code": "\n".join(code_parts)}
        return separated_data

    def _find_suspicious_pairs(self, separated_data: Dict[str, Dict[str, str]], 
                               content_type: str, threshold: float) -> Set[Tuple[str, str, float]]:
        """根据语义相似度得到可疑抄袭片段"""
        student_ids = list(separated_data.keys())
        if len(student_ids) < 2: return set()
        contents = [separated_data[sid][content_type] for sid in student_ids]
        embeddings = np.vstack([self._get_embedding(c, content_type) for c in contents])
        similarity_matrix = cosine_similarity(embeddings)
        suspicious_pairs = set()
        for i in range(len(student_ids)):
            for j in range(i + 1, len(student_ids)):
                score = similarity_matrix[i, j]
                if score >= threshold:
                    pair = tuple(sorted((student_ids[i], student_ids[j])))
                    suspicious_pairs.add((pair[0], pair[1], score))
        return suspicious_pairs

    def check_plagiarism_in_batch(self, submissions: Dict[str, str]) -> Dict:
        """
        执行初步抄袭检测的主函数。
        返回一个包含所有分析结果的结构化字典。
        """
        separated_data = self._separate_content_for_each_student(submissions)
        suspicious_text_pairs = self._find_suspicious_pairs(separated_data, 'text', 0.90)
        suspicious_code_pairs = self._find_suspicious_pairs(separated_data, 'code', 0.90)
        
        return {
            "suspicious_text_pairs": suspicious_text_pairs,
            "suspicious_code_pairs": suspicious_code_pairs,
            "separated_contents": separated_data
        }


plagiarism_service = PlagiarismService()
