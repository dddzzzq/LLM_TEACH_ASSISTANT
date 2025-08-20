import os
import random
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple
from tqdm import tqdm
import re

# 1. 环境设置与模块导入
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from app.services.plagiarism_service import PlagiarismService
from app.services.deepseek_service import DeepSeekService
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
                # 检查格式是否正确
                if len(parts) == 2:
                    # 文件名需要从ID补全扩展名
                    file1 = f"{parts[0]}"
                    file2 = f"{parts[1]}"
                    # 保证文件对的顺序一致性
                    ground_truth.add(tuple(sorted((file1, file2))))
    except FileNotFoundError:
        print(f"错误：找不到Ground Truth文件 {qrel_path}")
    except Exception as e:
        print(f"解析qrel ground_truth文件时出错: {e}")
    return ground_truth


# 4. 数据集加载
def load_dataset(dataset_type: str, num_positive: int = 50, num_negative: int = 50) -> Tuple[Dict[str, str], Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """
    加载并构造数据集。
    返回: submissions, ground_truth, all_possible_pairs
    """
    DATASET_ROOT = r"D:\DZQ\项目\教改项目-批改Agent\论文\数据集"
    
    submissions = {}
    ground_truth = set()
    all_possible_pairs = set()
    df = pd.DataFrame()

    # SOCO-2014 数据集加载逻辑
    if dataset_type == 'soco_2014':
        lang = 'java'
        soco_root = os.path.join(DATASET_ROOT, "fire14-source-code-test-dataset")
        lang_dir = os.path.join(soco_root, lang)
        ground_truth_path = os.path.join(soco_root,f"soco14-test-{lang}-update.qrel")
        
        print(f"\nLoading SOCO-2014 ({lang}) dataset from: {lang_dir}")
        
        # 加载全部数据集
        full_submissions = {}
        scenarios = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        for scenario in tqdm(scenarios, desc="Scanning all scenarios"):
            scenario_dir = os.path.join(lang_dir, scenario)
            if not os.path.isdir(scenario_dir): continue
            
            files_in_scenario = []
            for filename in os.listdir(scenario_dir):
                try:
                    with open(os.path.join(scenario_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                        full_submissions[filename] = f.read()
                    files_in_scenario.append(filename)
                except Exception as e:
                    print(f"读取文件 {filename} 时出错: {e}")

            for i in range(len(files_in_scenario)):
                for j in range(i + 1, len(files_in_scenario)):
                    all_possible_pairs.add(tuple(sorted((files_in_scenario[i], files_in_scenario[j]))))
        
        full_ground_truth = parse_soco_ground_truth_qrel(ground_truth_path, lang)
        loaded_files = set(full_submissions.keys())
        full_ground_truth = {pair for pair in full_ground_truth if pair[0] in loaded_files and pair[1] in loaded_files}
        
        print(f"全量扫描完成: {len(full_submissions)}个文件, {len(all_possible_pairs)}个总配对, {len(full_ground_truth)}个真实抄袭对。")

        # 随机选择样本
        all_negative_pairs = all_possible_pairs - full_ground_truth
        
        num_positive_to_sample = min(num_positive, len(full_ground_truth))
        num_negative_to_sample = min(num_negative, len(all_negative_pairs))

        sampled_positive_pairs = set(random.sample(list(full_ground_truth), num_positive_to_sample))
        sampled_negative_pairs = set(random.sample(list(all_negative_pairs), num_negative_to_sample))

        # 构建最终数据集
        ground_truth = sampled_positive_pairs
        final_all_possible_pairs = sampled_positive_pairs.union(sampled_negative_pairs)

        # 筛选出抽样后实际需要的文件
        required_files = set()
        for pair in final_all_possible_pairs:
            required_files.add(pair[0])
            required_files.add(pair[1])

        # 构建最终的 submissions 字典，并应用预处理
        for filename in required_files:
            code = preprocess_code(full_submissions[filename])
            submissions[filename] = f"--- 文件开始: main.java ---\n{code}\n--- 文件结束: main.java ---\n\n"
            

        print(f"已构建一个包含 {len(ground_truth)} 个正样本和 {len(sampled_negative_pairs)} 个负样本的测试集。")
        print(f"最终使用 {len(submissions)} 个代码文件，形成 {len(final_all_possible_pairs)} 个需比对的文件对。")
        
        return submissions, ground_truth, final_all_possible_pairs

    # 加载soco_java数据集
    if dataset_type == 'soco_java':
        file_path = os.path.join(DATASET_ROOT, r"SOCO_TRAIN_java\soco_java_pairs_with_code.csv")
        print(f"\nLoading SOCO dataset from: {file_path}")
        if not os.path.exists(file_path):
            print(f"错误：找不到数据集文件 {file_path}")
            return {}, set()
        try:
            df = pd.read_csv(file_path)
            required_columns = ['label', 'text_1', 'text_2']
            if not all(col in df.columns for col in required_columns):
                print(f"错误：CSV文件缺少必需的列。需要 {required_columns}，实际列名: {df.columns.tolist()}")
                return {}, set()
        except Exception as e:
            print(f"读取CSV文件时出错: {e}")
            return {}, set()

    # 加载LCQMC数据集
    elif dataset_type == 'lcqmc':
        file_path = os.path.join(DATASET_ROOT, r"LCQMC\dev.tsv")
        print(f"\nLoading LCQMC dataset from: {file_path}")
        if not os.path.exists(file_path):
            print(f"错误：找不到数据集文件 {file_path}")
            return {}, set()
        try:
            # 读取TSV文件，无表头，指定列名
            df = pd.read_csv(file_path, sep='\t', header=None, names=['sentence1', 'sentence2', 'label'], on_bad_lines='skip')
        except Exception as e:
            print(f"读取TSV文件时出错: {e}")
            return {}, set()

    # 加载PAWS-X-ZH数据集
    elif dataset_type == 'pawsx_zh':
        file_path = os.path.join(DATASET_ROOT, r"paws-x-zh\dev.tsv")
        print(f"\nLoading PAWS-X (Chinese) dataset from: {file_path}")
        if not os.path.exists(file_path):
            print(f"错误：找不到数据集文件 {file_path}")
            return {}, set()
        try:
            # 读取tsv文件，无表头
            df = pd.read_csv(file_path, sep='\t', header=None, names=['sentence1', 'sentence2', 'label'], on_bad_lines='skip')
            # # 读取TSV文件，有表头
            # df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip')
            # required_columns = ['sentence1', 'sentence2', 'label']
            # if not all(col in df.columns for col in required_columns):
            #     print(f"错误：TSV文件缺少必需的列。需要 {required_columns}，实际列名: {df.columns.tolist()}")
            #     return {}, set()
        except Exception as e:
            print(f"读取TSV文件时出错: {e}")
            return {}, set()
    
    else:
        print(f"未实现的数据集类型: {dataset_type}")
        return {}, set()

    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    df_positive = df[df['label'] == 1]
    df_negative = df[df['label'] == 0]
    print(f"数据集中总共有 {len(df_positive)} 个正样本和 {len(df_negative)} 个负样本。")

    num_positive_to_sample = min(num_positive, len(df_positive))
    num_negative_to_sample = min(num_negative, len(df_negative))

    if num_positive_to_sample < num_positive:
        print(f"警告: 请求抽取 {num_positive} 个正样本, 但只有 {len(df_positive)} 个可用。")
    if num_negative_to_sample < num_negative:
        print(f"警告: 请求抽取 {num_negative} 个负样本, 但只有 {len(df_negative)} 个可用。")

    positive_samples = df_positive.sample(n=num_positive_to_sample, random_state=42)
    negative_samples = df_negative.sample(n=num_negative_to_sample, random_state=42)

    final_df = pd.concat([positive_samples, negative_samples]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"已构建一个包含 {num_positive_to_sample} 个正样本和 {num_negative_to_sample} 个负样本的测试集。")

    for index, row in tqdm(final_df.iterrows(), total=len(final_df), desc=f"Processing {dataset_type} dataset"):
        student_id_1 = f"pair_{index}_A"
        student_id_2 = f"pair_{index}_B"
        
        # 根据数据集类型填充内容
        if dataset_type == 'soco_java':
            code1 = preprocess_code(row['text_1'])
            code2 = preprocess_code(row['text_2'])
            submissions[student_id_1] = f"--- 文件开始: main.java ---\n{code1}\n--- 文件结束: main.java ---\n\n"
            submissions[student_id_2] = f"--- 文件开始: main.java ---\n{code2}\n--- 文件结束: main.java ---\n\n"
        elif dataset_type in ['lcqmc', 'pawsx_zh']:
            submissions[student_id_1] = str(row['sentence1'])
            submissions[student_id_2] = str(row['sentence2'])

        if row['label'] == 1:
            pair = tuple(sorted((student_id_1, student_id_2)))
            ground_truth.add(pair)
    
    print(f"成功处理 {len(final_df)} 对样本，其中 {len(ground_truth)} 对为真实抄袭样本。")
    return submissions, ground_truth, None


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
    
    # 计算宏平均指标
    recall_macro = (recall_plag + recall_orig) / 2
    f1_macro = (f1_plag + f1_orig) / 2

    return {
        "accuracy": accuracy, 
        "precision_plag": precision_plag, "recall_plag": recall_plag, "f1_plag": f1_plag,
        "recall_macro": recall_macro, "f1_macro": f1_macro, # <--- 新增
        "precision_orig": precision_orig, "recall_orig": recall_orig, "f1_orig": f1_orig
    }

# 运行基线模型
def run_baseline_tfidf(submissions: Dict[str, str], all_possible_pairs: Set[Tuple[str, str]], threshold: float = 0.9) -> Set[Tuple[str, str]]:
    print("Running Baseline 1: TF-IDF...")
    
    # 创建一个文件名到索引的映射，以便快速查找
    student_ids = list(submissions.keys())
    id_to_index = {id: i for i, id in enumerate(student_ids)}
    contents = list(submissions.values())
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(contents)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    predictions = set()
    for file1, file2 in tqdm(all_possible_pairs, desc="Running TF-IDF Baseline"):
        # 查找文件在矩阵中对应的索引
        idx1 = id_to_index.get(file1)
        idx2 = id_to_index.get(file2)
        
        if idx1 is not None and idx2 is not None:
            if similarity_matrix[idx1, idx2] >= threshold:
                predictions.add(tuple(sorted((file1, file2))))
    return predictions

def run_baseline_single_unixcoder_model(plagiarism_service: PlagiarismService, submissions: Dict[str, str], all_possible_pairs: Set[Tuple[str, str]], threshold: float = 0.85) -> Set[Tuple[str, str]]:
    print("Running Baseline 2: Single Small Model (UniXcoder)...")
    student_ids = list(submissions.keys())
    id_to_index = {id: i for i, id in enumerate(student_ids)}
    
    embeddings_list = []
    for c in tqdm(submissions.values(), desc="Generating embeddings (Single UniXcoder)"):
        embeddings_list.append(plagiarism_service._get_embedding(c, 'code'))
    embeddings = np.vstack(embeddings_list)
    similarity_matrix = cosine_similarity(embeddings)
    
    predictions = set()
    for file1, file2 in tqdm(all_possible_pairs, desc="Comparing pairs (Single UniXcoder)"):
        idx1 = id_to_index.get(file1)
        idx2 = id_to_index.get(file2)
        if idx1 is not None and idx2 is not None:
            if similarity_matrix[idx1, idx2] >= threshold:
                predictions.add(tuple(sorted((file1, file2))))
    return predictions

def run_baseline_single_bert_model(plagiarism_service: PlagiarismService, submissions: Dict[str, str], all_possible_pairs: Set[Tuple[str, str]], threshold: float = 0.95) -> Set[Tuple[str, str]]:
    print("Running Baseline 3: Single Small Model (BERT)...")
    student_ids = list(submissions.keys())
    id_to_index = {id: i for i, id in enumerate(student_ids)}
    
    embeddings_list = []
    for c in tqdm(submissions.values(), desc="Generating embeddings (Single BERT)"):
        embeddings_list.append(plagiarism_service._get_embedding(c, 'text'))
    embeddings = np.vstack(embeddings_list)
    similarity_matrix = cosine_similarity(embeddings)
    
    predictions = set()
    for file1, file2 in tqdm(all_possible_pairs, desc="Comparing pairs (Single BERT)"):
        idx1 = id_to_index.get(file1)
        idx2 = id_to_index.get(file2)
        if idx1 is not None and idx2 is not None:
            if similarity_matrix[idx1, idx2] >= threshold:
                predictions.add(tuple(sorted((file1, file2))))
    return predictions

def run_dual_model_no_llm(plagiarism_service: PlagiarismService, submissions: Dict[str, str], all_possible_pairs: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    print("Running Experiment: Dual Model (No LLM)...")
    stage1_results = plagiarism_service.check_plagiarism_in_batch(submissions)
    predictions = set()
    text_pairs = {tuple(sorted((s1, s2))) for s1, s2, score in stage1_results["suspicious_text_pairs"]}
    code_pairs = {tuple(sorted((s1, s2))) for s1, s2, score in stage1_results["suspicious_code_pairs"]}
    predictions.update(text_pairs)
    predictions.update(code_pairs)
    return predictions.intersection(all_possible_pairs)

def run_our_system(plagiarism_service: PlagiarismService, deepseek_service: DeepSeekService, submissions: Dict[str, str], all_possible_pairs: Set[Tuple[str, str]], llm_threshold: int = 75) -> Tuple[Set[Tuple[str, str]], int]:
    print("Running Our System: Two-Stage Hybrid Model...")
    stage1_results = plagiarism_service.check_plagiarism_in_batch(submissions)
    
    suspicious_pairs_info = {}
    for s1, s2, score in stage1_results["suspicious_text_pairs"]:
        suspicious_pairs_info[tuple(sorted((s1, s2)))] = {'type': 'text', 'initial_score': score}
    for s1, s2, score in stage1_results["suspicious_code_pairs"]:
        pair = tuple(sorted((s1, s2)))
        if pair not in suspicious_pairs_info or score > suspicious_pairs_info[pair]['initial_score']:
            suspicious_pairs_info[pair] = {'type': 'code', 'initial_score': score}

    valid_suspicious_pairs = {k: v for k, v in suspicious_pairs_info.items() if k in all_possible_pairs}

    llm_calls = 0
    predictions = set()
    separated_contents = stage1_results["separated_contents"]
    
    for (s1, s2), info in tqdm(valid_suspicious_pairs.items(), desc="Running Our System (LLM Analysis)"):
        content_type = info['type']
        content1 = separated_contents.get(s1, {}).get(content_type, submissions[s1])
        content2 = separated_contents.get(s2, {}).get(content_type, submissions[s2])

        if not content1 or not content2: continue
        
        llm_analysis = deepseek_service.analyze_plagiarism(content1, content2, content_type)
        llm_calls += 1
        
        if llm_analysis and llm_analysis.get("similarity_score", 0) >= llm_threshold:
            predictions.add(tuple(sorted((s1, s2))))
            
    return predictions, llm_calls

def run_ablation_no_separation(plagiarism_service: PlagiarismService, deepseek_service: DeepSeekService, submissions: Dict[str, str], all_possible_pairs: Set[Tuple[str, str]], threshold: float = 0.95, llm_threshold: int = 80) -> Tuple[Set[Tuple[str, str]], int]:
    print("Running Ablation A: No Content Separation...")
    student_ids = list(submissions.keys())
    id_to_index = {id: i for i, id in enumerate(student_ids)}
    
    embeddings_list = []
    for c in tqdm(submissions.values(), desc="Generating embeddings (Ablation A)"):
        embeddings_list.append(plagiarism_service._get_embedding(c, 'code'))
    embeddings = np.vstack(embeddings_list)
    similarity_matrix = cosine_similarity(embeddings)
    
    suspicious_pairs = set()
    for file1, file2 in tqdm(all_possible_pairs, desc="Finding suspicious pairs (Ablation A)"):
        idx1 = id_to_index.get(file1)
        idx2 = id_to_index.get(file2)
        if idx1 is not None and idx2 is not None:
            if similarity_matrix[idx1, idx2] >= threshold:
                suspicious_pairs.add(tuple(sorted((file1, file2))))

    llm_calls = 0
    predictions = set()
    for s1, s2 in tqdm(suspicious_pairs, desc="Running Ablation A (LLM Analysis)"):
        llm_analysis = deepseek_service.analyze_plagiarism(submissions[s1], submissions[s2], 'code')
        llm_calls += 1
        if llm_analysis and llm_analysis.get("similarity_score", 0) >= llm_threshold:
            predictions.add((s1, s2))
            
    return predictions, llm_calls


# 6. 实验主调度函数
def main():
    # 初始化服务及模型
    print("Initializing services and models...")
    plagiarism_service = PlagiarismService(text_model_name=TEXT_MODEL_PATH, code_model_name=CODE_MODEL_PATH)
    deepseek_service = DeepSeekService()
    print("Initialization complete.")

    dataset_names = ['soco_java', 'lcqmc', 'pawsx_zh', 'soco_2014']
    # dataset_names = ['pawsx_zh']
    # dataset_names = ['soco_2014']
    # dataset_names = ['soco_java']
    all_results = {}
    
    output_filename = r"D:\DZQ\项目\教改项目-批改Agent\ai_grading_assistant\backend\experiments\experiment_results_soco_java_2014.txt"

    for dataset_name in dataset_names:
        print(f"\n==================== Running experiment for dataset: {dataset_name} ====================")
        
        if dataset_name == 'soco_2014':
            submissions, ground_truth, all_possible_pairs = load_dataset(dataset_name)
        else:
            submissions, ground_truth, all_possible_pairs_from_loader = load_dataset(dataset_name, num_positive=50, num_negative=50)
            all_possible_pairs = all_possible_pairs_from_loader
        
        if not submissions:
            print(f"数据集 {dataset_name} 为空，跳过此实验。")
            continue

        # 如果没返回all_possible_pairs，在此计算
        if all_possible_pairs is None:
            print("Calculating all possible pairs for stratified sample...")
            student_ids = list(submissions.keys())
            all_possible_pairs = {tuple(sorted((student_ids[i], student_ids[j]))) for i in range(len(student_ids)) for j in range(i + 1, len(student_ids))}
       
        results_for_current_dataset = {}
        
        # 运行实验
        experiments_to_run = {
                "TF-IDF": (run_baseline_tfidf, (submissions, all_possible_pairs)),
                "Single Model (UniXcoder)": (run_baseline_single_unixcoder_model, (plagiarism_service, submissions, all_possible_pairs)),
                "Single Model (BERT)": (run_baseline_single_bert_model, (plagiarism_service, submissions, all_possible_pairs)),
                "Dual Model (No LLM)": (run_dual_model_no_llm, (plagiarism_service, submissions, all_possible_pairs)),
                "Our System (w/ LLM)": (run_our_system, (plagiarism_service, deepseek_service, submissions, all_possible_pairs)),
                "Ablation (No Separation w/ LLM)": (run_ablation_no_separation, (plagiarism_service, deepseek_service, submissions, all_possible_pairs))
            }
        

        for name, (func, args) in experiments_to_run.items():
            start_time = time.time()
            func_result = func(*args)
            if isinstance(func_result, tuple):
                preds, llm_calls = func_result
            else:
                preds, llm_calls = func_result, 0
            end_time = time.time()
            
            results_for_current_dataset[name] = {
                'metrics': calculate_metrics(ground_truth, preds, all_possible_pairs),
                'time (s)': end_time - start_time,
                'llm_calls': llm_calls
            }
        
        df = pd.DataFrame(results_for_current_dataset).T
        df_metrics = df['metrics'].apply(pd.Series)
        df = df.drop('metrics', axis=1).join(df_metrics)
        all_results[dataset_name] = df

    # 保存到文件中
    print(f"\n\n#################### FINAL EXPERIMENT RESULTS ####################")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("#################### FINAL EXPERIMENT RESULTS ####################\n\n")
        
        for dataset_name, df_result in all_results.items():
            print(f"\n--- Results for: {dataset_name} ---")
            columns_to_display = [
                'accuracy', 'f1_macro', 'recall_macro',
                'precision_plag', 'recall_plag', 'f1_plag', 
                'precision_orig', 'recall_orig', 'f1_orig',
                'time (s)', 'llm_calls'
            ]
            print(df_result[columns_to_display].round(4))
            
            f.write(f"--- Results for: {dataset_name} ---\n")
            f.write(df_result[columns_to_display].round(4).to_string())
            f.write("\n\n")

    print(f"\n实验结果已成功保存到文件: {output_filename}")


if __name__ == "__main__":
    main()