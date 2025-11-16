import os
import random
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple
from tqdm import tqdm
import re
from docx import Document
import jieba


# 1. 环境设置与模块导入
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from app.services.plagiarism_service import PlagiarismService
from app.services.deepseek_service import DeepSeekService
from app.services.grading_service import GradingService
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

# 2. 模型加载
BASE_MODEL_PATH = r"D:\DZQ\项目\教改项目-批改Agent\models"
TEXT_MODEL_PATH = os.path.join(BASE_MODEL_PATH, "bert-base-chinese")
CODE_MODEL_PATH = os.path.join(BASE_MODEL_PATH, "unixcoder-base")

# 3. 定义辅助函数，处理文本
def preprocess_code(code_text: str) -> str:
    if not isinstance(code_text, str):
        return ""
    processed_text = code_text.replace('\\n', '\n').replace('\\r', '')
    processed_text = re.sub(r'\n\s*\n', '\n', processed_text)
    
    lines = [line.strip() for line in processed_text.split('\n')]
    non_empty_lines = [line for line in lines if line]
    return '\n'.join(non_empty_lines)

def parse_soco_ground_truth_qrel(qrel_path: str, lang_ext: str) -> Set[Tuple[str, str]]:
    """解析SOCO数据集的qrel格式文件。"""
    ground_truth = set()
    try:
        with open(qrel_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    file1 = f"{parts[0]}"
                    file2 = f"{parts[1]}"
                    ground_truth.add(tuple(sorted((file1, file2))))
    except FileNotFoundError:
        print(f"错误：找不到Ground Truth文件 {qrel_path}")
    except Exception as e:
        print(f"解析qrel ground_truth文件时出错: {e}")
    return ground_truth


# 4. 数据集加载
def load_dataset(dataset_type: str, grading_service: GradingService, num_positive: int = 50, num_negative: int = 50) -> Tuple[Dict[str, str], Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """
    加载并构造数据集。
    返回: submissions, ground_truth, all_possible_pairs
    """
    DATASET_ROOT = r"D:\DZQ\项目\教改项目-批改Agent\论文\数据集"
    
    submissions = {}
    ground_truth = set()
    all_possible_pairs = set()
    df = pd.DataFrame()

    if dataset_type in ['mixed_homework_1', 'mixed_homework_2']:
        if dataset_type == 'mixed_homework_1':
            homework_root = r'E:\项目资料\ai_assistant\third\作业'
            ground_truth_path = r'D:\DZQ\项目\教改项目-批改Agent\论文\数据集\mixed_dataset_1.csv'
        elif dataset_type == 'mixed_homework_2':
            homework_root = r'E:\项目资料\ai_assistant\2024研究生课程作业'
            ground_truth_path = r'D:\DZQ\项目\教改项目-批改Agent\论文\数据集\mixed_dataset_2.csv'
        
        print(f"\nLoading Mixed Homework dataset from:{homework_root}")
        print(f"Using ground truth file:{ground_truth_path}")

        try:
            try:
                gt_df = pd.read_csv(ground_truth_path, encoding='utf-8')
            except UnicodeDecodeError:
                print(f"UTF-8 decoding failed for {ground_truth_path}. Trying with GBK encoding.")
                gt_df = pd.read_csv(ground_truth_path, encoding='gbk')
        except FileNotFoundError:
            print(f"错误: 找不到 ground truth 文件 {ground_truth_path}")
            return {}, set(), set()
        except Exception as e:
            print(f"使用UTF-8和GBK读取文件时均出错: {e}")
            return {}, set(), set()
            
        required_students = set(gt_df['student1'].unique()) | set(gt_df['student2'].unique())

        for student_id in tqdm(required_students, desc='Processing student archives:'):
            archive_filename = f"{student_id}.zip"
            archive_path = os.path.join(homework_root, archive_filename)

            if not os.path.exists(archive_path):
                print(f"找不到学生压缩文件{archive_path}，跳过")
                continue

            try:
                with open(archive_path, 'rb') as f:
                    archive_bytes = f.read()
                    merged_content = grading_service.process_archive(archive_bytes, archive_filename)
                    submissions[str(student_id)] = merged_content
            except Exception as e:
                print(f"处理压缩文件{archive_path}时出错{e}")

        for _, row in gt_df.iterrows():
            if row['label'] == 1:
                ground_truth.add(tuple(sorted((str(row['student1']), str(row['student2'])))))
        for _, row in gt_df.iterrows():
            all_possible_pairs.add(tuple(sorted((str(row['student1']), str(row['student2'])))))

        print(f"成功加载 {len(submissions)} 份学生作业，定义了 {len(all_possible_pairs)} 个比对样本对，其中包含 {len(ground_truth)} 对真实抄袭样本。")
        return submissions, ground_truth, all_possible_pairs

    if dataset_type == 'soco_2014':
        lang = 'java'
        soco_root = os.path.join(DATASET_ROOT, "fire14-source-code-test-dataset")
        lang_dir = os.path.join(soco_root, lang)
        ground_truth_path = os.path.join(soco_root,f"soco14-test-{lang}-update.qrel")
        print(f"\nLoading SOCO-2014 ({lang}) dataset from: {lang_dir}")
        full_submissions = {}
        scenarios = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        for scenario in tqdm(scenarios, desc="Scanning all scenarios"):
            scenario_dir = os.path.join(lang_dir, scenario)
            if not os.path.isdir(scenario_dir): continue
            files_in_scenario = [f for f in os.listdir(scenario_dir)]
            for filename in files_in_scenario:
                try:
                    with open(os.path.join(scenario_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                        full_submissions[filename] = f.read()
                except Exception as e:
                    print(f"读取文件 {filename} 时出错: {e}")
            for i in range(len(files_in_scenario)):
                for j in range(i + 1, len(files_in_scenario)):
                    all_possible_pairs.add(tuple(sorted((files_in_scenario[i], files_in_scenario[j]))))
        full_ground_truth = parse_soco_ground_truth_qrel(ground_truth_path, lang)
        loaded_files = set(full_submissions.keys())
        full_ground_truth = {pair for pair in full_ground_truth if pair[0] in loaded_files and pair[1] in loaded_files}
        print(f"全量扫描完成: {len(full_submissions)}个文件, {len(all_possible_pairs)}个总配对, {len(full_ground_truth)}个真实抄袭对。")
        all_negative_pairs = all_possible_pairs - full_ground_truth
        num_positive_to_sample = min(num_positive, len(full_ground_truth))
        num_negative_to_sample = min(num_negative, len(all_negative_pairs))
        sampled_positive_pairs = set(random.sample(list(full_ground_truth), num_positive_to_sample))
        sampled_negative_pairs = set(random.sample(list(all_negative_pairs), num_negative_to_sample))
        ground_truth = sampled_positive_pairs
        final_all_possible_pairs = sampled_positive_pairs.union(sampled_negative_pairs)
        required_files = {file for pair in final_all_possible_pairs for file in pair}
        for filename in required_files:
            code = preprocess_code(full_submissions[filename])
            submissions[filename] = f"--- 文件开始: main.java ---\n{code}\n--- 文件结束: main.java ---\n\n"
        print(f"已构建一个包含 {len(ground_truth)} 个正样本和 {len(sampled_negative_pairs)} 个负样本的测试集。")
        print(f"最终使用 {len(submissions)} 个代码文件，形成 {len(final_all_possible_pairs)} 个需比对的文件对。")
        return submissions, ground_truth, final_all_possible_pairs

    if dataset_type == 'soco_java':
        file_path = os.path.join(DATASET_ROOT, r"SOCO_TRAIN_java\soco_java_pairs_with_code.csv")
    elif dataset_type == 'lcqmc':
        file_path = os.path.join(DATASET_ROOT, r"LCQMC\dev.tsv")
    elif dataset_type == 'pawsx_zh':
        file_path = os.path.join(DATASET_ROOT, r"paws-x-zh\dev.tsv")
    else:
        print(f"未实现的数据集类型: {dataset_type}")
        return {}, set(), set()
        
    print(f"\nLoading dataset from: {file_path}")
    if not os.path.exists(file_path):
        print(f"错误：找不到数据集文件 {file_path}")
        return {}, set(), set()

    try:
        if dataset_type in ['lcqmc', 'pawsx_zh']:
             df = pd.read_csv(file_path, sep='\t', header=None, names=['sentence1', 'sentence2', 'label'], on_bad_lines='skip')
        else: # soco_java
             df = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return {}, set(), set()

    df['label'] = pd.to_numeric(df['label'], errors='coerce').dropna().astype(int)
    df_positive = df[df['label'] == 1]
    df_negative = df[df['label'] == 0]
    num_positive_to_sample = min(num_positive, len(df_positive))
    num_negative_to_sample = min(num_negative, len(df_negative))
    positive_samples = df_positive.sample(n=num_positive_to_sample, random_state=42)
    negative_samples = df_negative.sample(n=num_negative_to_sample, random_state=42)
    final_df = pd.concat([positive_samples, negative_samples]).sample(frac=1, random_state=42).reset_index(drop=True)

    for index, row in tqdm(final_df.iterrows(), total=len(final_df), desc=f"Processing {dataset_type} dataset"):
        id1, id2 = f"pair_{index}_A", f"pair_{index}_B"
        if dataset_type == 'soco_java':
            submissions[id1] = f"--- 文件开始: main.java ---\n{preprocess_code(row['text_1'])}\n--- 文件结束: main.java ---\n\n"
            submissions[id2] = f"--- 文件开始: main.java ---\n{preprocess_code(row['text_2'])}\n--- 文件结束: main.java ---\n\n"
        elif dataset_type in ['lcqmc', 'pawsx_zh']:
            submissions[id1], submissions[id2] = str(row['sentence1']), str(row['sentence2'])
        if row['label'] == 1:
            ground_truth.add(tuple(sorted((id1, id2))))
        all_possible_pairs.add(tuple(sorted((id1, id2))))
            
    print(f"成功处理 {len(final_df)} 对样本，其中 {len(ground_truth)} 对为真实抄袭样本。")
    return submissions, ground_truth, all_possible_pairs

# 5. 评估指标计算
def calculate_metrics(ground_truth: Set[Tuple[str, str]], predictions: Set[Tuple[str, str]], all_pairs: Set[Tuple[str, str]]):
    tp = len(ground_truth.intersection(predictions))
    fp = len(predictions - ground_truth)
    fn = len(ground_truth - predictions)
    tn = len(all_pairs) - (tp + fp + fn)
    total = len(all_pairs)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision_plag = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_plag = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_plag = 2 * (precision_plag * recall_plag) / (precision_plag + recall_plag) if (precision_plag + recall_plag) > 0 else 0.0
    precision_orig = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_orig = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_orig = 2 * (precision_orig * recall_orig) / (precision_orig + recall_orig) if (precision_orig + recall_orig) > 0 else 0.0
    recall_macro = (recall_plag + recall_orig) / 2
    f1_macro = (f1_plag + f1_orig) / 2
    return {"accuracy": accuracy, "precision_plag": precision_plag, "recall_plag": recall_plag, "f1_plag": f1_plag, "recall_macro": recall_macro, "f1_macro": f1_macro, "precision_orig": precision_orig, "recall_orig": recall_orig, "f1_orig": f1_orig}

# 6. 模型运行函数
# def run_baseline_tfidf(submissions: Dict[str, str], all_possible_pairs: Set[Tuple[str, str]], threshold: float = 0.9) -> Set[Tuple[str, str]]:
#     print("Running Baseline 1: TF-IDF...")
#     student_ids, contents = list(submissions.keys()), list(submissions.values())
#     id_to_index = {id: i for i, id in enumerate(student_ids)}
#     vectorizer = TfidfVectorizer()
#     similarity_matrix = cosine_similarity(vectorizer.fit_transform(contents))
#     predictions = set()
#     for file1, file2 in tqdm(all_possible_pairs, desc="Running TF-IDF Baseline"):
#         idx1, idx2 = id_to_index.get(file1), id_to_index.get(file2)
#         if idx1 is not None and idx2 is not None:
#             score = similarity_matrix[idx1, idx2]
#             if score  >= threshold:
#                 predictions.add(tuple(sorted((file1, file2))))
#     return predictions
def run_baseline_tfidf(dataset_name: str, submissions: Dict[str, str], all_possible_pairs: Set[Tuple[str, str]], threshold: float = 0.9) -> Set[Tuple[str, str]]:
    """
    根据数据集类型选择不同的分词器运行TF-IDF基线模型。
    """
    print("Running Baseline 1: TF-IDF...")
    
    # --- 修改部分：根据数据集名称选择分词器 ---
    if dataset_name in ['lcqmc', 'pawsx_zh']:
        print("Using Chinese (jieba) tokenizer for TF-IDF.")
        def chinese_tokenizer(text):
            return jieba.lcut(text)
        vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer)
    else: # 默认为英文处理方式，适用于 'soco_2014' 等
        print("Using default (English) tokenizer for TF-IDF.")
        vectorizer = TfidfVectorizer()
    # --- 修改结束 ---

    student_ids, contents = list(submissions.keys()), list(submissions.values())
    id_to_index = {id: i for i, id in enumerate(student_ids)}
    
    similarity_matrix = cosine_similarity(vectorizer.fit_transform(contents))
    
    predictions = set()
    for file1, file2 in tqdm(all_possible_pairs, desc="Running TF-IDF Baseline"):
        idx1, idx2 = id_to_index.get(file1), id_to_index.get(file2)
        if idx1 is not None and idx2 is not None:
            score = similarity_matrix[idx1, idx2]
            if score >= threshold:
                predictions.add(tuple(sorted((file1, file2))))
                
    return predictions

def run_baseline_single_model(plagiarism_service: PlagiarismService, submissions: Dict[str, str], all_possible_pairs: Set[Tuple[str, str]], model_type: str, threshold: float) -> Set[Tuple[str, str]]:
    model_name = "UniXcoder" if model_type == 'code' else "BERT"
    print(f"Running Baseline: Single Small Model ({model_name})...")
    student_ids = list(submissions.keys())
    id_to_index = {id: i for i, id in enumerate(student_ids)}
    embeddings = np.vstack([plagiarism_service._get_embedding(c, model_type) for c in tqdm(submissions.values(), desc=f"Generating embeddings ({model_name})")])
    similarity_matrix = cosine_similarity(embeddings)
    predictions = set()
    for file1, file2 in tqdm(all_possible_pairs, desc=f"Comparing pairs ({model_name})"):
        idx1, idx2 = id_to_index.get(file1), id_to_index.get(file2)
        if idx1 is not None and idx2 is not None and similarity_matrix[idx1, idx2] >= threshold:
            predictions.add(tuple(sorted((file1, file2))))
    return predictions

def run_dual_model_no_llm(plagiarism_service: PlagiarismService, submissions: Dict[str, str], all_possible_pairs: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    print("Running Experiment: Dual Model (No LLM)...")
    stage1_results = plagiarism_service.check_plagiarism_in_batch(submissions)
    text_pairs = {tuple(sorted((s1, s2))) for s1, s2, _ in stage1_results["suspicious_text_pairs"]}
    code_pairs = {tuple(sorted((s1, s2))) for s1, s2, _ in stage1_results["suspicious_code_pairs"]}
    return (text_pairs | code_pairs).intersection(all_possible_pairs)

# def run_our_system(plagiarism_service: PlagiarismService, deepseek_service: DeepSeekService, submissions: Dict[str, str], all_possible_pairs: Set[Tuple[str, str]], llm_threshold: int = 75) -> Tuple[Set[Tuple[str, str]], int, int, int]:
#     print("Running Our System: Two-Stage Hybrid Model...")
#     stage1_results = plagiarism_service.check_plagiarism_in_batch(submissions)
#     suspicious_pairs_info = {}
#     for s1, s2, score in stage1_results["suspicious_text_pairs"]:
#         suspicious_pairs_info[tuple(sorted((s1, s2)))] = {'type': 'text', 'score': score}
#     for s1, s2, score in stage1_results["suspicious_code_pairs"]:
#         pair = tuple(sorted((s1, s2)))
#         if pair not in suspicious_pairs_info or score > suspicious_pairs_info[pair]['score']:
#             suspicious_pairs_info[pair] = {'type': 'code', 'score': score}
#     valid_suspicious_pairs = {k for k in suspicious_pairs_info if k in all_possible_pairs}
    
#     llm_calls, prompt_tokens, completion_tokens = 0, 0, 0
#     predictions = set()
#     separated = stage1_results["separated_contents"]
#     for s1, s2 in tqdm(valid_suspicious_pairs, desc="Running Our System (LLM Analysis)"):
#         info = suspicious_pairs_info[tuple(sorted((s1, s2)))]
#         c_type = info['type']
#         content1, content2 = separated.get(s1, {}).get(c_type, ""), separated.get(s2, {}).get(c_type, "")
#         if not content1 or not content2: continue
#         llm_analysis, usage = deepseek_service.analyze_plagiarism(content1, content2, c_type)
#         llm_calls += 1
#         prompt_tokens += usage.get("prompt_tokens", 0)
#         completion_tokens += usage.get("completion_tokens", 0)
#         if llm_analysis and llm_analysis.get("similarity_score", 0) >= llm_threshold:
#             predictions.add(tuple(sorted((s1, s2))))
#     return predictions, llm_calls, prompt_tokens, completion_tokens

def run_our_system(plagiarism_service: PlagiarismService, deepseek_service: DeepSeekService, submissions: Dict[str, str], all_possible_pairs: Set[Tuple[str, str]], llm_threshold: int = 75) -> Tuple[Set[Tuple[str, str]], int, int, int]:
    print("Running Our System: Two-Stage Hybrid Model...")

    # ==================== 代码修改部分 ====================
    # 1. 从 all_possible_pairs 提取出所有需要参与比对的独立 submission ID
    print(f"Optimizing Stage 1: Analyzing {len(all_possible_pairs)} submissions")

    # 3. 在第一阶段的查重中，只使用这个精简后的字典
    stage1_results = plagiarism_service.check_plagiarism_in_batch(submissions)
    # =======================================================
    
    suspicious_pairs_info = {}
    for s1, s2, score in stage1_results["suspicious_text_pairs"]:
        suspicious_pairs_info[tuple(sorted((s1, s2)))] = {'type': 'text', 'score': score}
    for s1, s2, score in stage1_results["suspicious_code_pairs"]:
        pair = tuple(sorted((s1, s2)))
        if pair not in suspicious_pairs_info or score > suspicious_pairs_info[pair]['score']:
            suspicious_pairs_info[pair] = {'type': 'code', 'score': score}
            
    # 这里的 valid_suspicious_pairs 逻辑保持不变，它会自然地过滤出属于 all_possible_pairs 的可疑对
    valid_suspicious_pairs = {k for k in suspicious_pairs_info if k in all_possible_pairs}
    
    llm_calls, prompt_tokens, completion_tokens = 0, 0, 0
    predictions = set()
    
    # separated_contents 也是从精简后的 pair_submissions 中获取的，因此这里的 separated 变量名改为 stage1_results["separated_contents"] 
    separated = stage1_results["separated_contents"]
    for s1, s2 in tqdm(valid_suspicious_pairs, desc="Running Our System (LLM Analysis)"):
        info = suspicious_pairs_info[tuple(sorted((s1, s2)))]
        c_type = info['type']
        content1, content2 = separated.get(s1, {}).get(c_type, ""), separated.get(s2, {}).get(c_type, "")
        if not content1 or not content2: continue
        
        llm_analysis, usage = deepseek_service.analyze_plagiarism(content1, content2, c_type)
        llm_calls += 1
        prompt_tokens += usage.get("prompt_tokens", 0)
        completion_tokens += usage.get("completion_tokens", 0)
        
        if llm_analysis and llm_analysis.get("similarity_score", 0) >= llm_threshold:
            predictions.add(tuple(sorted((s1, s2))))
            
    return predictions, llm_calls, prompt_tokens, completion_tokens

def run_ablation_no_separation(plagiarism_service: PlagiarismService, deepseek_service: DeepSeekService, submissions: Dict[str, str], all_possible_pairs: Set[Tuple[str, str]], threshold: float = 0.95, llm_threshold: int = 80) -> Tuple[Set[Tuple[str, str]], int, int, int]:
    print("Running Ablation A: No Content Separation...")
    suspicious_pairs = run_baseline_single_model(plagiarism_service, submissions, all_possible_pairs, 'code', threshold)
    llm_calls, prompt_tokens, completion_tokens = 0, 0, 0
    predictions = set()
    for s1, s2 in tqdm(suspicious_pairs, desc="Running Ablation A (LLM Analysis)"):
        llm_analysis, usage = deepseek_service.analyze_plagiarism(submissions[s1], submissions[s2], 'code')
        llm_calls += 1
        prompt_tokens += usage.get("prompt_tokens", 0)
        completion_tokens += usage.get("completion_tokens", 0)
        if llm_analysis and llm_analysis.get("similarity_score", 0) >= llm_threshold:
            predictions.add(tuple(sorted((s1, s2))))
    return predictions, llm_calls, prompt_tokens, completion_tokens


# 7. 实验主调度函数
def main():
    print("Initializing services and models...")
    grading_service = GradingService()
    plagiarism_service = PlagiarismService(text_model_name=TEXT_MODEL_PATH, code_model_name=CODE_MODEL_PATH)
    deepseek_service = DeepSeekService()
    print("Initialization complete.")

    # dataset_names = ['lcqmc', 'pawsx_zh', 'soco_2014', 'mixed_homework_1', 'mixed_homework_2']
    # dataset_names = ['pawsx_zh', 'mixed_homework_1', 'mixed_homework_2']
    # dataset_names = ['lcqmc', 'soco_2014']
    dataset_names = ['pawsx_zh']
    output_filename = r"D:\DZQ\项目\教改项目-批改Agent\ai_grading_assistant\backend\experiments\experiment_results_test.txt"

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("#################### FINAL EXPERIMENT RESULTS ####################\n\n")

    for dataset_name in dataset_names:
        llm_shold = 72
        print(f"\n==================== Running experiment for dataset: {dataset_name} llm_shold: {llm_shold} ====================")
        
        if dataset_name in ['soco_2014', 'mixed_homework_1', 'mixed_homework_2']:
            submissions, ground_truth, all_possible_pairs = load_dataset(dataset_name, grading_service)
        else:
            submissions, ground_truth, all_possible_pairs = load_dataset(dataset_name, grading_service, num_positive=2, num_negative=2)
        
        if not submissions:
            print(f"数据集 {dataset_name} 为空，跳过此实验。")
            continue

        # if all_possible_pairs is None:
        #     # student_ids = list(submissions.keys())
        #     # all_possible_pairs = {tuple(sorted((student_ids[i], student_ids[j]))) for i in range(len(student_ids)) for j in range(i + 1, len(student_ids))}
        #     all_possible_pairs = ground_truth

        results = {}
        experiments = {
            # "TF-IDF": (run_baseline_tfidf, (dataset_name, submissions, all_possible_pairs, 0.9)),
            # "Single Model (UniXcoder)": (run_baseline_single_model, (plagiarism_service, submissions, all_possible_pairs, 'code', 0.92)),
            # "Single Model (BERT)": (run_baseline_single_model, (plagiarism_service, submissions, all_possible_pairs, 'text', 0.90)),
            # "Dual Model (No LLM)": (run_dual_model_no_llm, (plagiarism_service, submissions, all_possible_pairs)),
            "Our System (w/ LLM)": (run_our_system, (plagiarism_service, deepseek_service, submissions, all_possible_pairs, llm_shold)),
            # "Ablation (No Separation w/ LLM)": (run_ablation_no_separation, (plagiarism_service, deepseek_service, submissions, all_possible_pairs, 0.95, llm_shold))
        }

        for name, (func, args) in experiments.items():
            start_time = time.time()
            result = func(*args)
            end_time = time.time()
            
            prompt_tokens, completion_tokens = 0, 0
            if name in ["Our System (w/ LLM)", "Ablation (No Separation w/ LLM)"]:
                preds, llm_calls, prompt_tokens, completion_tokens = result
            else:
                preds, llm_calls = result, 0
            
            results[name] = {
                'metrics': calculate_metrics(ground_truth, preds, all_possible_pairs),
                'time (s)': end_time - start_time,
                'llm_calls': llm_calls,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens
            }
        
        df = pd.DataFrame(results).T
        df_metrics = df['metrics'].apply(pd.Series)
        df_final = df.drop('metrics', axis=1).join(df_metrics)

        # 跑完一个数据集后立即追加保存结果
        print(f"\n--- Saving results for: {dataset_name} ---")
        with open(output_filename, 'a', encoding='utf-8') as f:
            columns_to_display = [
                'accuracy', 'f1_macro', 'recall_macro',
                'precision_plag', 'recall_plag', 'f1_plag', 
                'precision_orig', 'recall_orig', 'f1_orig',
                'time (s)', 'llm_calls', 'prompt_tokens', 'completion_tokens'
            ]
            df_display = df_final.reindex(columns=columns_to_display).fillna(0)
            
            print(f"\n--- Results for: {dataset_name} ---")
            print(df_display.round(4))
            
            f.write(f"--- Results for: {dataset_name} ---\n")
            f.write(df_display.round(4).to_string())
            f.write("\n\n")
        
        print(f"Results for {dataset_name} have been appended to {output_filename}")

    print(f"\n实验全部完成。最终结果已保存到文件: {output_filename}")

if __name__ == "__main__":
    main()
