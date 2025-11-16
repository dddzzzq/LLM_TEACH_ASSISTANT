import gc
import re
from typing import Dict, List, Optional, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from ..schemas.models import PlagiarismReport
from .deepseek_service import deepseek_service
import torch
from itertools import combinations, product 
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


class PlagiarismService:
    """
    修改后的抄袭检测服务，采用双模型策略。
    在每个学生提交的内容中分离文本和代码，并进行同类内容的交叉对比。
    """
    def __init__(self, text_model_name: str = None, code_model_name: str = None):
        """
        初始化服务。
        可以从外部传入模型路径，如果未提供，则使用默认路径。
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 如果外部没有指定路径，就使用默认的硬编码路径
        self.text_model_name = text_model_name or r"/root/autodl-tmp/dzq/ai_grading_assistant/models/bert-base-chinese"
        self.code_model_name = code_model_name or r"/root/autodl-tmp/dzq/ai_grading_assistant/models/unixcoder-base"
        
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

    # def _find_suspicious_pairs(self, separated_data: Dict[str, Dict[str, str]], 
    #                            content_type: str, threshold: float) -> Set[Tuple[str, str, float]]:
    #     """根据语义相似度得到可疑抄袭片段"""
    #     student_ids = list(separated_data.keys())
    #     if len(student_ids) < 2: return set()
    #     contents = [separated_data[sid][content_type] for sid in student_ids]
    #     embeddings = np.vstack([self._get_embedding(c, content_type) for c in contents])
    #     similarity_matrix = cosine_similarity(embeddings)
    #     suspicious_pairs = set()
    #     for i in range(len(student_ids)):
    #         for j in range(i + 1, len(student_ids)):
    #             score = similarity_matrix[i, j]
    #             if score >= threshold:
    #                 pair = tuple(sorted((student_ids[i], student_ids[j])))
    #                 suspicious_pairs.add((pair[0], pair[1], score))
    #     return suspicious_pairs

    # 因为电脑内存限制，修改计算方式
    def _find_suspicious_pairs(self, separated_data: Dict[str, Dict[str, str]], 
                                 content_type: str, threshold: float) -> Set[Tuple[str, str, float]]:
        """
        根据语义相似度得到可疑抄袭片段
        对嵌入生成和比较都进行分批处理，优化内存
        """
        student_ids = list(separated_data.keys())
        num_students = len(student_ids)
        if num_students < 2: return set()
        
        contents = [separated_data[sid][content_type] for sid in student_ids]
        suspicious_pairs = set()

        # 关键参数：可以根据内存大小调整这个值
        # 如果内存紧张，调小（如5）；如果内存充裕，可以调大以提升速度
        BATCH_SIZE = 50

        total_comparisons = (num_students * (num_students - 1)) // 2
        
        with tqdm(total=total_comparisons, desc=f"Comparing '{content_type}' pairs") as pbar:
            # 按批次处理
            for i in range(0, num_students, BATCH_SIZE):
                # 定义第一个批次的索引和内容
                batch1_indices = range(i, min(i + BATCH_SIZE, num_students))
                batch1_contents = [contents[k] for k in batch1_indices]
                # 只为当前批次生成嵌入
                batch1_embeddings = [self._get_embedding(c, content_type) for c in batch1_contents]

                # 1. 批次内部比较
                if len(batch1_indices) > 1:
                    for idx1_in_batch, idx2_in_batch in combinations(range(len(batch1_indices)), 2):
                        score = cosine_similarity(batch1_embeddings[idx1_in_batch], batch1_embeddings[idx2_in_batch])[0, 0]
                        if score >= threshold:
                            original_idx1 = batch1_indices[idx1_in_batch]
                            original_idx2 = batch1_indices[idx2_in_batch]
                            pair = tuple(sorted((student_ids[original_idx1], student_ids[original_idx2])))
                            suspicious_pairs.add((pair[0], pair[1], float(score)))
                        pbar.update(1)

                # 2. 与所有后续批次进行比较
                for j in range(i + BATCH_SIZE, num_students, BATCH_SIZE):
                    # 定义第二个批次的索引和内容
                    batch2_indices = range(j, min(j + BATCH_SIZE, num_students))
                    batch2_contents = [contents[k] for k in batch2_indices]
                    # 只为第二个批次生成嵌入
                    batch2_embeddings = [self._get_embedding(c, content_type) for c in batch2_contents]

                    # 比较 batch1 和 batch2 的所有组合
                    for idx1_in_batch, idx2_in_batch in product(range(len(batch1_indices)), range(len(batch2_indices))):
                        score = cosine_similarity(batch1_embeddings[idx1_in_batch], batch2_embeddings[idx2_in_batch])[0, 0]
                        if score >= threshold:
                            original_idx1 = batch1_indices[idx1_in_batch]
                            original_idx2 = batch2_indices[idx2_in_batch]
                            pair = tuple(sorted((student_ids[original_idx1], student_ids[original_idx2])))
                            suspicious_pairs.add((pair[0], pair[1], float(score)))
                        pbar.update(1)

                    # 关键：及时释放第二个批次的内存
                    del batch2_embeddings
                    gc.collect()

                # 关键：及时释放第一个批次的内存
                del batch1_embeddings
                gc.collect()
                
        return suspicious_pairs

    def check_plagiarism_in_batch(self, submissions: Dict[str, str]) -> Dict:
        """
        执行初步抄袭检测的主函数。
        返回一个包含所有分析结果的结构化字典。
        """
        separated_data = self._separate_content_for_each_student(submissions)
        suspicious_text_pairs = self._find_suspicious_pairs(separated_data, 'text', 0.95)
        suspicious_code_pairs = self._find_suspicious_pairs(separated_data, 'code', 0.96)
        # suspicious_text_pairs = self._find_suspicious_pairs(separated_data, 'text', 0.95)
        # suspicious_code_pairs = self._find_suspicious_pairs(separated_data, 'code', 0.98)
        return {
            "suspicious_text_pairs": suspicious_text_pairs,
            "suspicious_code_pairs": suspicious_code_pairs,
            "separated_contents": separated_data
        }


plagiarism_service = PlagiarismService()
