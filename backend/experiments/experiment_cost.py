import pandas as pd
import io
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from tabulate import tabulate

# 设置图表的中文字体，以防绘图时出现乱码
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False # 正常显示负号
except Exception as e:
    print(f"设置字体失败: {e}")

def format_df_to_text_table(df_to_format):
    df = df_to_format.round(4).copy()

    def get_str_width(s):
        width = 0
        for char in str(s):
            if '\u4e00' <= char <= '\u9fff': # CJK 统一表意文字范围
                width += 2
            else:
                width += 1
        return width

    # 将所有数据转为字符串以便计算宽度
    for col in df.columns:
        df[col] = df[col].astype(str)

    # 计算每列所需的最大宽度
    column_widths = {}
    for col in df.columns:
        max_width = get_str_width(col)
        for val in df[col]:
            max_width = max(max_width, get_str_width(val))
        column_widths[col] = max_width

    # 构建表头
    header_list = []
    for col in df.columns:
        padding = ' ' * (column_widths[col] - get_str_width(col))
        header_list.append(f"{col}{padding}")
    header_str = ' | '.join(header_list)

    # 构建分隔线
    separator_list = ['-' * width for width in column_widths.values()]
    separator_str = '-+-'.join(separator_list)

    # 构建数据行
    data_rows_list = []
    for _, row in df.iterrows():
        row_list = []
        for col in df.columns:
            val = row[col]
            padding = ' ' * (column_widths[col] - get_str_width(val))
            row_list.append(f"{val}{padding}")
        data_rows_list.append(' | '.join(row_list))

    return '\n'.join([header_str, separator_str] + data_rows_list)

def parse_experiment_results(file_path: str, dataset_name: str) -> pd.DataFrame:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误：找不到实验文件 (Error: Experiment file not found at {file_path})")
        return pd.DataFrame()
    
    # 使用正则表达式找到特定数据集块
    pattern = re.compile(f"--- Results for: {dataset_name} ---\n(.*?)(?=--- Results for:|$)", re.DOTALL)
    match = pattern.search(content)

    if not match:
        print(f"错误：在文件中找不到数据集 '{dataset_name}' 的结果 (Error: Results for dataset '{dataset_name}' not found in the file)")
        return pd.DataFrame()
    
    data_block = match.group(1).strip()
    lines = data_block.split('\n')
    
    header_line = lines[0]
    data_lines = lines[1:]

    columns = re.split(r'\s{1,}', header_line.strip())
    has_model_col_in_header = columns[0].lower() == 'model'
    if not has_model_col_in_header:
        columns.insert(0, 'Model')
    else:
        columns[0] = 'Model'

    # 解析数据行
    parsed_data = []
    for line in data_lines:
        if not line.strip():
            continue
            
        # 使用正则表达式将可能包含空格的模型名称与后面的数值数据分开
        match = re.match(r'^(.*?)\s{2,}([\d\.-].*)$', line)
        if match:
            model_name = match.group(1).strip()
            numeric_data_str = match.group(2).strip()
            
            # 按空格分割数值数据部分
            values = re.split(r'\s+', numeric_data_str)
            
            row_data = [model_name] + values
            parsed_data.append(row_data)

    if not parsed_data:
        print(f"警告：在数据集 '{dataset_name}' 中没有解析到数据行 (Warning: No data lines parsed for dataset '{dataset_name}')")
        return pd.DataFrame()

    # 创建DataFrame，确保列数与数据匹配
    df = pd.DataFrame(parsed_data, columns=columns[:len(parsed_data[0])])

    # 将数值列转换为正确的数字类型
    for col in df.columns:
        if col != 'Model':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 清理列名中可能存在的多余空格
    df.columns = [col.strip() for col in df.columns]

    return df

def calculate_cost_and_efficiency(result_df: pd.DataFrame):
    """
    计算成本与效率指标
    """
    # 1. 定义参数
    LLM_CALL_COST = 0.0263
    TIME_COST = 0.001
    ERROR_REVIEW_COST = 9.19

    # 2. 数据集先验知识
    TOTAL_PAIRS = 100
    ACTUAL_POSITIVES = 50
    ACTUAL_NEGETIVES = 50

    output_data = []

    for _, row in result_df.iterrows():
        # 提取基础指标
        model_name = row['Model']
        time_s = row['time(s)']
        llm_calls = row['llm_calls']
        precision_plag = row['precision_plag']
        recall_plag = row['recall_plag']

        # 3. 计算吞吐量
        throughput = TOTAL_PAIRS / time_s if time_s > 0 else float('inf')

        # 4. 计算TP,FP,FN
        tp = recall_plag * ACTUAL_POSITIVES
        fn = ACTUAL_POSITIVES - tp
        fp = (tp / precision_plag) - tp if precision_plag > 0 else 0

        # 计算成本
        llm_cost = LLM_CALL_COST * llm_calls
        times_cost = TIME_COST * time_s
        review_cost = ERROR_REVIEW_COST * (fp + fn)
        total_cost = llm_cost + times_cost + review_cost

        output_data.append({
            '模型': model_name,
            '吞吐量': throughput,
            '总成本': total_cost,
            'LLM 成本':llm_cost,
            '时间成本': times_cost,
            '审查成本': review_cost,
            '误报数 (FP)': fp,
            '漏报数 (FN)': fn
        })

    return pd.DataFrame(output_data)

def plot_pareto(root_path: str, result_df: pd.DataFrame, cost: str, perf: str, dataset_name: str):
    """
    绘制并保存帕累托前沿图。
    """
    df = result_df.copy()
    cost_label = f"成本: {cost} (越低越好)"
    perf_label = f"性能: {perf} (越高越好)"

    # 一个点是帕累托最优的，当且仅当不存在任何其他点在所有维度上都比它好。
    pareto_front_rows = []
    for index, row in df.iterrows():
        is_dominated = False
        for other_index, other_row in df.iterrows():
            if index == other_index: continue
            if (other_row[cost] <= row[cost]) and (other_row[perf] >= row[perf]): # 在两个维度上都不比它差
                if (other_row[cost] < row[cost]) or (other_row[perf] > row[perf]): # 并且至少在一个维度上严格比它好
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_front_rows.append(row)
            
    pareto_df = pd.DataFrame(pareto_front_rows).sort_values(by=cost)

    # --- 绘图 ---
    plt.figure(figsize=(14, 9))
    plt.scatter(df[cost], df[perf], c='skyblue', label='所有模型', s=60, alpha=0.8, edgecolors='grey')

    if not pareto_df.empty:
        plt.plot(pareto_df[cost], pareto_df[perf], 'r-o', label='帕累托前沿', markersize=10, linewidth=2.5)

    # 为每个点添加标签，突出显示最优解
    for _, row in df.iterrows():
        is_on_front = any(all(row == p_row) for _, p_row in pareto_df.iterrows())
        color = 'red' if is_on_front else 'black'
        weight = 'bold' if is_on_front else 'normal'
        plt.text(row[cost], row[perf], f"  {row['Model']}", fontsize=10, color=color, weight=weight, verticalalignment='bottom')

    plt.title(f'"{dataset_name}" 数据集: 性能 vs. 成本权衡分析', fontsize=18)
    plt.xlabel(cost_label, fontsize=14)
    plt.ylabel(perf_label, fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # --- 保存图像 ---
    output_filename = f"pareto_{dataset_name}_{cost.replace(' ', '_')}_vs_{perf}.png"
    output_path = os.path.join(root_path, "cost")
    os.makedirs(output_path, exist_ok=True) # 确保目录存在
    plt.savefig(os.path.join(output_path, output_filename))
    print(f"已生成帕累托前沿图: {output_filename}")
    plt.close()


if __name__ == '__main__':
    # 定义输入文件路径
    input_file_path = r"experiment_results_with_tokens.txt"
    output_file_path = r"cost_efficiency_test_with_tokens.txt"

    # 从txt文件中读取实验结果
    root_path = r"D:\DZQ\项目\教改项目-批改Agent\ai_grading_assistant\backend\experiments"
    output_path = os.path.join(root_path, output_file_path)
    datasetnames = ['soco_2014', 'lcqmc', 'pawsx_zh', 'mixed_homework_1', 'mixed_homework_2']
    # datasetnames = ['lcqmc']
    for datasetname in datasetnames:
        full_path = os.path.join(root_path, input_file_path)
        df_results = parse_experiment_results(full_path, datasetname)

        if not df_results.empty:
            # 计算成本与效率指标
            df_cost_efficiency = calculate_cost_and_efficiency(df_results)

            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(f"--- “{datasetname}” 数据集成本与效率指标计算结果 ---\n\n")
                f.write(format_df_to_text_table(df_cost_efficiency))
                f.write(f"\n\n{'='*80}\n\n")
                
        df_full_results = pd.merge(df_results, df_cost_efficiency, left_on='Model', right_on='模型')

        # 绘制帕累托前沿图
        plot_pareto(root_path, df_full_results, cost='time(s)', perf='f1_macro', dataset_name=datasetname)
        plot_pareto(root_path, df_full_results, cost='llm_calls', perf='f1_macro', dataset_name=datasetname)