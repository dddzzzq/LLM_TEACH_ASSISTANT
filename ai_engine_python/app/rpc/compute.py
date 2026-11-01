"""Translate protobuf requests to AI services and preserve the public RPC contract."""

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import grpc

from app.rpc.protocol import pb2, pb2_grpc
from app.schemas.models import PlagiarismReport, AIGCReport, CodeDocMatchReport
from app.tools.rpa_tools import fetch_homework_sync
from app.services.ocr_runtime import OCRCancelled, OCRTimeout, OCRUnavailable
from app.services.student_evidence import decode_materials, analysis_parts, evidence_chunks, render_chunks


@dataclass(frozen=True)
class ComputeServices:
    """Explicit service dependencies; construction does not load model weights."""

    plagiarism_service: Any
    aigc_detector_service: Any
    deepseek_service: Any
    ocr_service_instance: Any


class ComputeServicer(pb2_grpc.ComputeServiceServicer):
    def __init__(self, services: ComputeServices):
        self.services = services

    # 作业批改
    def ExtractText(self, request, context):
        tmp_path = None
        try:
            if not context.is_active():
                raise OCRCancelled("OCR request cancelled")
            if self.services.ocr_service_instance is None:
                raise OCRUnavailable("OCR service unavailable")
            suffix = os.path.splitext(request.filename)[1]
            if not suffix:
                suffix = ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(request.file_content)
                tmp_path = tmp.name

            ocr_text = self.services.ocr_service_instance.get_concatenated_text(
                [tmp_path],
                cancelled=lambda: not context.is_active(),
                timeout=context.time_remaining(),
            )

            return pb2.ExtractResponse(text_content=ocr_text)
        except (OCRCancelled, OCRTimeout, OCRUnavailable) as exc:
            code = (
                grpc.StatusCode.CANCELLED
                if isinstance(exc, OCRCancelled)
                else (
                    grpc.StatusCode.DEADLINE_EXCEEDED
                    if isinstance(exc, OCRTimeout)
                    else grpc.StatusCode.UNAVAILABLE
                )
            )
            context.set_code(code)
            context.set_details(str(exc))
            return pb2.ExtractResponse()
        except Exception:
            logging.exception("OCR RPC failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("图片识别失败")
            return pb2.ExtractResponse(text_content="【系统降级】图片识别失败")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def CheckPlagiarism(self, request, context):
        try:
            student_texts = dict(request.student_texts)
            results = self.services.plagiarism_service.check_plagiarism_in_batch(student_texts)
            suspicious_text_pairs = results.get("suspicious_text_pairs", [])
            suspicious_code_pairs = results.get("suspicious_code_pairs", [])
            separated_contents = results.get("separated_contents", {})

            llm_analysis_cache = {}
            all_suspicious_pairs = [(*pair, "text") for pair in suspicious_text_pairs] + [
                (*pair, "code") for pair in suspicious_code_pairs
            ]

            for i, (s1, s2, initial_score, content_type) in enumerate(all_suspicious_pairs, 1):
                content1 = separated_contents.get(s1, {}).get(content_type)
                content2 = separated_contents.get(s2, {}).get(content_type)
                if not content1 or not content2:
                    continue

                llm_analysis, _ = self.services.deepseek_service.analyze_plagiarism(
                    content1, content2, content_type
                )
                if llm_analysis:
                    llm_analysis_cache[(s1, s2, content_type)] = {
                        "initial_score": initial_score,
                        "llm_analysis": llm_analysis,
                    }

            final_plag_data = {sid: [] for sid in student_texts.keys()}
            for (s1, s2, content_type), analysis_result in llm_analysis_cache.items():
                llm_data = analysis_result.get("llm_analysis")
                if llm_data:
                    final_plag_data[s1].append(
                        {
                            "similar_to": s2,
                            "content_type": content_type,
                            "initial_score": float(analysis_result["initial_score"]),
                            "llm_analysis": llm_data,
                        }
                    )
                    final_plag_data[s2].append(
                        {
                            "similar_to": s1,
                            "content_type": content_type,
                            "initial_score": float(analysis_result["initial_score"]),
                            "llm_analysis": llm_data,
                        }
                    )

            return pb2.PlagiarismResponse(
                plagiarism_results_json=json.dumps(final_plag_data, ensure_ascii=False)
            )
        except Exception as e:
            return pb2.PlagiarismResponse(plagiarism_results_json="{}")

    def DetectAIGC(self, request, context):
        try:
            report = self.services.aigc_detector_service.detect(request.text_content)
            return pb2.AIGCResponse(aigc_report_json=json.dumps(report, ensure_ascii=False))
        except Exception as e:
            return pb2.AIGCResponse(aigc_report_json="{}")

    def GradeHomework(self, request, context):
        sid = request.student_id
        try:
            rubric = json.loads(request.rubric_json) if request.rubric_json else {}
            plag_reports_raw = (
                json.loads(request.plagiarism_reports_json)
                if request.plagiarism_reports_json
                else []
            )
            plag_reports = []
            for p in plag_reports_raw:
                try:
                    plag_reports.append(PlagiarismReport(**p))
                except:
                    pass
            aigc_report_raw = (
                json.loads(request.aigc_report_json) if request.aigc_report_json else None
            )
            aigc_report = None
            if aigc_report_raw and "ai_probability" in aigc_report_raw:
                try:
                    aigc_report = AIGCReport(**aigc_report_raw)
                except:
                    pass

            materials = decode_materials(request.student_text)
            if materials is not None:
                prose_content, code_content = analysis_parts(materials)
            else:
                prose_content, code_content = self.services.aigc_detector_service._separate_content(request.student_text)
            match_analysis = None
            if (code_content.strip() and prose_content.strip()
                    and (materials is None or len(code_content) <= 100000 and len(prose_content) <= 75000)):
                match_analysis, _ = self.services.deepseek_service.analyze_code_doc_match(
                    code_content=code_content,
                    doc_content=prose_content,
                    assignment_requirement=request.question,
                )
            code_doc_match_report = None
            if match_analysis and "score" in match_analysis:
                try:
                    code_doc_match_report = CodeDocMatchReport(**match_analysis)
                except:
                    pass

            ai_result = self.services.deepseek_service.grade_homework(
                question=request.question,
                rubric=rubric,
                student_answer=request.student_text,
                plagiarism_reports=plag_reports,
                aigc_report=aigc_report,
                code_doc_match_report=code_doc_match_report,
            )

            merged_content = request.student_text
            if materials is not None:
                try:
                    text = render_chunks(evidence_chunks(materials))
                except (ValueError, TypeError, KeyError):
                    text = '\n'.join(f.get('text', '') for f in materials.get('files', []))
                merged_content = json.dumps({"schema": "grading-result.v1", "text": text,
                                             "reading": ai_result.get('reading', {"status": "NEEDS_REVIEW", "warnings": ["缺少完整读取回执"]}),
                                             "score_details": ai_result.get('score_details', [])}, ensure_ascii=False)
            return pb2.GradeResponse(
                total_score=ai_result.get("total_score", -1.0),
                feedback=ai_result.get("overall_feedback", "评分失败"),
                merged_content=merged_content,
                code_doc_match_report_json=json.dumps(match_analysis, ensure_ascii=False)
                if match_analysis
                else "{}",
            )
        except Exception as e:
            return pb2.GradeResponse(total_score=-1.0, feedback=f"评分失败: {str(e)}")

    def PoolScores(self, request, context):
        start_time = time.time()
        try:
            logging.info(f"成绩池化节点: 开始对作业 {request.assignment_id} 进行成绩池化处理...")

            # 调用DeepSeek服务进行成绩池化
            pooled_results = asyncio.run(
                self.services.deepseek_service.pool_scores(
                    assignment_id=request.assignment_id, scores_data_json=request.scores_data_json
                )
            )

            if not pooled_results:
                logging.error(f"[PoolScores] 大模型未能返回有效的池化结果")
                return pb2.PoolScoresResponse(
                    success=False, pooled_results_json="{}", error_message="AI成绩池化处理失败"
                )

            logging.info(
                f"[成绩池化] 完成，处理了 {len(pooled_results) if isinstance(pooled_results, list) else '未知'} 条记录，耗时: {time.time() - start_time:.2f}s"
            )
            return pb2.PoolScoresResponse(
                success=True,
                pooled_results_json=json.dumps(pooled_results, ensure_ascii=False),
                error_message="",
            )

        except Exception as e:
            logging.error(f"[PoolScores] 失败: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.PoolScoresResponse(
                success=False, pooled_results_json="{}", error_message=f"成绩池化处理失败: {str(e)}"
            )

    def FetchPortalHomework(self, request, context):
        """教务系统作业抓取接口"""
        start_time = time.time()
        try:
            logging.info(f"[FetchPortalHomework] 开始执行教务系统作业抓取任务...")
            logging.info(f"  - 用户名: {request.username}")
            logging.info(f"  - 课程名称: {request.course_name}")
            logging.info(f"  - 作业名称: {request.assignment_name}")

            # 调用RPA工具执行作业下载
            result = fetch_homework_sync(
                username=request.username,
                password=request.password,
                course_name=request.course_name,
                assignment_name=request.assignment_name,
            )

            # 判断返回结果类型
            if isinstance(result, list):
                # 成功返回文件路径列表
                logging.info(f"[FetchPortalHomework] RPA执行成功，获取到 {len(result)} 个文件")
                for idx, file_path in enumerate(result, 1):
                    logging.info(f"  - 文件 {idx}: {file_path}")

                return pb2.FetchResponse(
                    success=True, message=f"成功下载 {len(result)} 个作业附件", file_paths=result
                )
            else:
                # 失败返回错误信息字符串
                logging.error(f"[FetchPortalHomework] RPA执行失败: {result}")
                return pb2.FetchResponse(success=False, message=str(result), file_paths=[])

        except Exception as e:
            logging.error(f"[FetchPortalHomework] 接口异常: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.FetchResponse(
                success=False, message=f"教务系统作业抓取失败: {str(e)}", file_paths=[]
            )
