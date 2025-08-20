import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple

# ----------------- 1. 环境设置与模块导入 -----------------
# 将项目根目录添加到Python路径，以便导入app中的模块
# 这使得脚本可以像FastAPI应用一样找到 plagiarism_service 等模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

# 从你的项目中导入核心服务
from app.services.plagiarism_service import PlagiarismService
from app.services.deepseek_service import DeepSeekService
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

# ----------------- 2. 数据集加载 (需要你来实现) -----------------

def load_dataset(dataset_type: str = 'mixed') -> Tuple[Dict[str, str], Set[Tuple[str, str]]]:
    """
    加载并构造数据集。
    你需要根据你的实际文件路径和格式来实现这部分。
    
    返回:
        - submissions (Dict[str, str]): 模拟的学生提交内容, e.g., {'student_1': '...', 'student_2': '...'}
        - ground_truth (Set[Tuple[str, str]]): 真实的抄袭对, e.g., {('student_1', 'student_3'), ...}
          注意：元组内的ID应按字典序排列，以方便后续比对。
    """
    print(f"Loading '{dataset_type}' dataset...")
    # --- 这是一个示例，你需要替换成真实的加载逻辑 ---
    # 假设你已经准备好了混合内容数据集
    if dataset_type == 'mixed':
        # 示例：正样本（抄袭）
        report_A = "这是学生A的原创报告..."
        report_B = "这是学生B抄袭A的报告..." # 模拟洗稿
        code_C = "public class C { ... }"
        code_D = "public class D extends C { ... }" # 模拟代码重用
        
        # 示例：负样本（原创）
        report_E = "学生E的完全原创报告..."
        code_F = "public class F { ... }" # 完全不同的代码
        
        submissions = {
            'student_A': f"--- 文件开始: report.txt ---\n{report_A}\n--- 文件结束: report.txt ---\n\n",
            'student_B': f"--- 文件开始: report.txt ---\n{report_B}\n--- 文件结束: report.txt ---\n\n",
            'student_C': f"--- 文件开始: main.java ---\n{code_C}\n--- 文件结束: main.java ---\n\n",
            'student_D': f"--- 文件开始: main.java ---\n{code_D}\n--- 文件结束: main.java ---\n\n",
            'student_E': f"--- 文件开始: report.txt ---\n{report_E}\n--- 文件结束: report.txt ---\n\n",
            'student_F': f"--- 文件开始: main.java ---\n{code_F}\n--- 文件结束: main.java ---\n\n",
        }
        # 真实标签：B抄了A，D抄了C
        ground_truth = {('student_A', 'student_B'), ('student_C', 'student_D')}
        return submissions, ground_truth
    
    # 你可以在这里添加加载纯代码(SOCO)和纯文本(MSRP)数据集的逻辑
    return {}, set()


# ----------------- 3. 评估指标计算 -----------------

def calculate_metrics(ground_truth: Set[Tuple[str, str]], predictions: Set[Tuple[str, str]], all_pairs: Set[Tuple[str, str]]):
    """计算并返回精确率、召回率和F1分数"""
    
    tp = len(ground_truth.intersection(predictions))
    fp = len(predictions - ground_truth)
    fn = len(ground_truth - predictions)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1_score": f1}


# ----------------- 4. 基线模型与本系统实现 -----------------

def run_baseline_tfidf(submissions: Dict[str, str], threshold: float = 0.9) -> Set[Tuple[str, str]]:
    """基线模型1：传统TF-IDF"""
    print("Running Baseline 1: TF-IDF...")
    student_ids = list(submissions.keys())
    contents = list(submissions.values())
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(contents)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    predictions = set()
    for i in range(len(student_ids)):
        for j in range(i + 1, len(student_ids)):
            if similarity_matrix[i, j] >= threshold:
                pair = tuple(sorted((student_ids[i], student_ids[j])))
                predictions.add(pair)
    return predictions

def run_baseline_single_model(plagiarism_service: PlagiarismService, submissions: Dict[str, str], threshold: float = 0.95) -> Set[Tuple[str, str]]:
    """基线模型2：单一小模型 (仅使用UniXcoder)"""
    print("Running Baseline 2: Single Small Model (UniXcoder)...")
    student_ids = list(submissions.keys())
    # 强制使用代码模型处理所有内容
    embeddings = np.vstack([plagiarism_service._get_embedding(c, 'code') for c in submissions.values()])
    similarity_matrix = cosine_similarity(embeddings)
    
    predictions = set()
    for i in range(len(student_ids)):
        for j in range(i + 1, len(student_ids)):
            if similarity_matrix[i, j] >= threshold:
                pair = tuple(sorted((student_ids[i], student_ids[j])))
                predictions.add(pair)
    return predictions


def run_our_system(plagiarism_service: PlagiarismService, deepseek_service: DeepSeekService, submissions: Dict[str, str], llm_threshold: int = 80) -> Tuple[Set[Tuple[str, str]], int]:
    """本系统：两阶段混合模型"""
    print("Running Our System: Two-Stage Hybrid Model...")
    
    # --- 第一阶段：双小模型筛选 ---
    stage1_results = plagiarism_service.check_plagiarism_in_batch(submissions)
    suspicious_pairs_info = {} # 用于合并文本和代码的相似度结果
    
    for s1, s2, score in stage1_results["suspicious_text_pairs"]:
        pair = tuple(sorted((s1, s2)))
        suspicious_pairs_info[pair] = {'type': 'text', 'initial_score': score}
        
    for s1, s2, score in stage1_results["suspicious_code_pairs"]:
        pair = tuple(sorted((s1, s2)))
        # 如果代码和文本都可疑，优先处理代码
        if pair not in suspicious_pairs_info or score > suspicious_pairs_info[pair]['initial_score']:
            suspicious_pairs_info[pair] = {'type': 'code', 'initial_score': score}
            
    # --- 第二阶段：LLM深度分析 ---
    llm_calls = 0
    predictions = set()
    separated_contents = stage1_results["separated_contents"]
    
    for (s1, s2), info in suspicious_pairs_info.items():
        content_type = info['type']
        content1 = separated_contents.get(s1, {}).get(content_type)
        content2 = separated_contents.get(s2, {}).get(content_type)
        
        if not content1 or not content2: continue
        
        llm_analysis = deepseek_service.analyze_plagiarism(content1, content2, content_type)
        llm_calls += 1
        
        if llm_analysis and llm_analysis.get("similarity_score", 0) >= llm_threshold:
            predictions.add(tuple(sorted((s1, s2))))
            
    return predictions, llm_calls


# ----------------- 5. 消融实验实现 -----------------

def run_ablation_no_separation(plagiarism_service: PlagiarismService, deepseek_service: DeepSeekService, submissions: Dict[str, str], threshold: float = 0.95, llm_threshold: int = 80) -> Tuple[Set[Tuple[str, str]], int]:
    """消融实验A: 无内容分离 (统一用代码模型处理合并内容)"""
    print("Running Ablation A: No Content Separation...")
    
    # --- 阶段一：使用单一模型处理合并后的内容 ---
    student_ids = list(submissions.keys())
    embeddings = np.vstack([plagiarism_service._get_embedding(c, 'code') for c in submissions.values()])
    similarity_matrix = cosine_similarity(embeddings)
    
    suspicious_pairs = set()
    for i in range(len(student_ids)):
        for j in range(i + 1, len(student_ids)):
            if similarity_matrix[i, j] >= threshold:
                suspicious_pairs.add(tuple(sorted((student_ids[i], student_ids[j]))))
    
    # --- 阶段二：LLM分析 (使用通用prompt) ---
    llm_calls = 0
    predictions = set()
    for s1, s2 in suspicious_pairs:
        # 由于没有分离，我们只能使用通用代码prompt进行分析
        llm_analysis = deepseek_service.analyze_plagiarism(submissions[s1], submissions[s2], 'code')
        llm_calls += 1
        if llm_analysis and llm_analysis.get("similarity_score", 0) >= llm_threshold:
            predictions.add((s1, s2))
            
    return predictions, llm_calls
    

# ----------------- 6. 实验主调度函数 -----------------

def main():
    # ---- 初始化服务 ----
    # 实例化服务时，会加载模型，这部分时间不计入处理时间
    print("Initializing services and models...")
    plagiarism_service = PlagiarismService()
    deepseek_service = DeepSeekService()
    print("Initialization complete.")
    
    # ---- 加载数据集 ----
    submissions, ground_truth = load_dataset('mixed')
    student_ids = list(submissions.keys())
    all_possible_pairs = {tuple(sorted((student_ids[i], student_ids[j]))) for i in range(len(student_ids)) for j in range(i + 1, len(student_ids))}
    
    results = {}
    
    # ---- 运行实验 ----
    
    # 基线1: TF-IDF
    start_time = time.time()
    tfidf_preds = run_baseline_tfidf(submissions)
    end_time = time.time()
    results['TF-IDF'] = {
        'metrics': calculate_metrics(ground_truth, tfidf_preds, all_possible_pairs),
        'time (s)': end_time - start_time,
        'llm_calls': 0
    }
    
    # 基线2: 单一小模型
    start_time = time.time()
    single_model_preds = run_baseline_single_model(plagiarism_service, submissions)
    end_time = time.time()
    results['Single Small Model'] = {
        'metrics': calculate_metrics(ground_truth, single_model_preds, all_possible_pairs),
        'time (s)': end_time - start_time,
        'llm_calls': 0
    }
    
    # 我们的系统
    start_time = time.time()
    our_preds, our_llm_calls = run_our_system(plagiarism_service, deepseek_service, submissions)
    end_time = time.time()
    results['Our System'] = {
        'metrics': calculate_metrics(ground_truth, our_preds, all_possible_pairs),
        'time (s)': end_time - start_time,
        'llm_calls': our_llm_calls
    }

    # 消融实验A: 无内容分离
    start_time = time.time()
    ablation_a_preds, ablation_a_llm_calls = run_ablation_no_separation(plagiarism_service, deepseek_service, submissions)
    end_time = time.time()
    results['Ablation (No Separation)'] = {
        'metrics': calculate_metrics(ground_truth, ablation_a_preds, all_possible_pairs),
        'time (s)': end_time - start_time,
        'llm_calls': ablation_a_llm_calls
    }

    # 消融实验B: 无上下文感知提示 (这部分需要你修改DeepSeekService，或在下面创建一个临时版本)
    # 此处仅为示例，你需要创建一个使用通用Prompt的analyse_plagiarism版本
    # temp_deepseek_service = DeepSeekServiceWithGenericPrompt() 
    # start_time = time.time()
    # ...
    # results['Ablation (No Context Prompt)'] = ...

    # ---- 打印结果 ----
    df = pd.DataFrame(results).T
    # 格式化指标列
    df_metrics = df['metrics'].apply(pd.Series)
    df = df.drop('metrics', axis=1).join(df_metrics)
    
    print("\n\n--- Experiment Results ---")
    print(df[['precision', 'recall', 'f1_score', 'time (s)', 'llm_calls']].round(4))


if __name__ == "__main__":
    main()