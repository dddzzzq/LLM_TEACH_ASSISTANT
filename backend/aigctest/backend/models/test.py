import torch
from transformers import pipeline

# 1. 加载预训练好的模型
#    我们使用pipeline，这是Hugging Face提供的一个非常简单的高级接口。
#    任务是“文本分类”，模型是“roberta-base-openai-detector”。
#    device=0 表示使用第一个GPU，如果想用CPU，可以设置为 device=-1
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("text-classification", model="roberta-base-openai-detector", device=device)

# 2. 准备需要检测的文本
text_human = "I wrote this text myself, sentence by sentence. It reflects my personal style and includes some of my own quirks and grammatical mistakes, making it feel authentic."
text_ai = "The utilization of advanced neural networks has revolutionized the field of natural language processing, enabling machines to generate coherent and contextually relevant text with unprecedented accuracy and fluency."

# 3. 进行预测
results_human = classifier(text_human)
results_ai = classifier(text_ai)

# 4. 分析结果
print("--- 检测人类编写的文本 ---")
print(f"文本: '{text_human}'")
# 注意：标签的含义需要查看模型文档。对于这个模型：
# LABEL_0 通常代表 AI，LABEL_1 通常代表 Human
# 但为确保准确，我们直接看分数。Real表示人类，Fake表示AI。
# 在新版pipeline中，它可能会直接返回易于理解的标签。
# 我们假设 label 'Real' 代表人类, 'Fake' 代表AI
label_human = results_human[0]['label']
score_human = results_human[0]['score']
print(f"预测结果: {label_human}, 置信度: {score_human:.4f}")


print("\n--- 检测AI生成的文本 ---")
print(f"文本: '{text_ai}'")
label_ai = results_ai[0]['label']
score_ai = results_ai[0]['score']
print(f"预测结果: {label_ai}, 置信度: {score_ai:.4f}")

# 演示如何处理批量文本
texts_to_check = [text_human, text_ai]
batch_results = classifier(texts_to_check)
print("\n--- 批量检测结果 ---")
print(batch_results)