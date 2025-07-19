from typing import Dict, Optional, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ..schemas.models import PlagiarismReport, SimilarityMatch
from .deepseek_service import deepseek_service


class PlagiarismService:
    """
    用来检测抄袭率
    它采用两阶段流程：
    1. 快速的 TF-IDF 余弦相似度检查，以找出可能相似的配对。
    2. 对超过相似度阈值的配对进行 LLM 深度分析。
    """

    def check_plagiarism_in_batch(
        self, submissions: Dict[str, str]
    ) -> Dict[str, PlagiarismReport]:
        """
        分析一批提交的作业是否存在抄袭。

        Args:
            submissions: 一个将学生ID映射到其提交文本的字典。

        Return:
            一个将每个学生ID映射到其 PlagiarismReport 的字典。
        """
        student_ids = list(submissions.keys())
        corpus = list(submissions.values())
        final_reports: Dict[str, PlagiarismReport] = {
            student_id: PlagiarismReport() for student_id in student_ids
        }

        # 抄袭检查至少需要两份提交
        if len(corpus) < 2:
            return final_reports

        # --- 第一阶段: TF-IDF 相似度检查 ---
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except ValueError:
            return final_reports

        highly_suspicious_pairs: Set[frozenset] = set()
        TFIDF_THRESHOLD = 0.7  # 阈值

        for i in range(len(student_ids)):
            highest_score = 0.0
            similar_to_student_index = -1

            # 找出与学生 i 最相似的一份提交
            for j in range(len(student_ids)):
                if i == j:
                    continue
                
                score = similarity_matrix[i][j]
                if score > highest_score:
                    highest_score = score
                    similar_to_student_index = j

            # 在报告中记录最高的相似度匹配
            if similar_to_student_index != -1 and highest_score > 0:
                final_reports[student_ids[i]].highest_similarity = SimilarityMatch(
                    similar_to=student_ids[similar_to_student_index],
                    score=round(highest_score, 4),
                )

            # 如果分数超过阈值，则标记该配对进行深度分析
            if highest_score >= TFIDF_THRESHOLD and similar_to_student_index != -1:
                pair = frozenset([student_ids[i], student_ids[similar_to_student_index]])
                highly_suspicious_pairs.add(pair)

        # --- 第二阶段: 对可疑配对进行 LLM 深度分析 ---
        if highly_suspicious_pairs:
            print(f"发现 {len(highly_suspicious_pairs)} 对高度疑似作业，将进行LLM深度分析...")
            for pair in highly_suspicious_pairs:
                student1_id, student2_id = tuple(pair)
                text1 = submissions[student1_id]
                text2 = submissions[student2_id]

                llm_analysis_result = deepseek_service.analyze_plagiarism(
                    text1, student1_id, text2, student2_id
                )

                if llm_analysis_result:
                    final_reports[student1_id].llm_analysis = llm_analysis_result
                    final_reports[student2_id].llm_analysis = llm_analysis_result

        return final_reports


# 创建实例
plagiarism_service = PlagiarismService()