"""Exercise a running deployment with synthetic data and real grading calls.

Requires a configured DeepSeek key. Tokens and passwords are never written to
the report. Only resources created by this run are removed after success.
"""
import argparse
import io
import json
import secrets
import subprocess
import time
import zipfile
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/deployment-check.json")
    parser.add_argument("--check-restart", action="store_true", help="Restart this WSL application's processes and verify persistence")
    args = parser.parse_args()
    client = requests.Session()
    client.trust_env = False  # Local service checks must bypass the system proxy.
    tag = secrets.token_hex(6)
    report = {"ok": False, "checks": {}, "resources": {}, "username": "deploy_check_" + tag}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def passed(name, value="passed"):
        report["checks"][name] = value
        save()
        print(name + ": " + str(value), flush=True)

    def api(method, path, expected=200, **kwargs):
        response = client.request(method, args.base_url.rstrip("/") + path, timeout=180, **kwargs)
        if response.status_code != expected:
            raise RuntimeError(f"{method} {path}: HTTP {response.status_code}, expected {expected}")
        return response

    def wait_job(job):
        deadline = time.monotonic() + 360
        previous = None
        while time.monotonic() < deadline:
            state = api("GET", "/api/jobs/" + job).json()
            status = state.get("status")
            if status != previous:
                print("job " + job + ": " + str(status), flush=True)
                previous = status
            if status == "SUCCESS":
                return state
            if status == "FAILED":
                raise RuntimeError("Grading job failed: " + str(state.get("message")))
            time.sleep(2)
        raise TimeoutError("Grading job did not finish: " + job)

    try:
        api("GET", "/api/profile", expected=401)
        password = secrets.token_urlsafe(32)
        api("POST", "/api/register", expected=201, json={
            "username": report["username"], "password": password,
            "name": "本机部署验收", "role": "teacher",
        })
        login = api("POST", "/api/login", json={"username": report["username"], "password": password}).json()
        report["user_id"] = login["user_id"]
        client.headers["Authorization"] = "Bearer " + login["access_token"]
        api("GET", "/api/profile")
        refreshed = api("POST", "/api/refresh", json={"refresh_token": login["refresh_token"]}).json()
        assert refreshed.get("access_token"), "Refresh did not return a token"
        client.headers["Authorization"] = "Bearer " + refreshed["access_token"]
        passed("registration_login_refresh")

        assignment = api("POST", "/assignments/", expected=201, json={
            "course_name": "部署验收", "class_name": "Synthetic", "task_name": "WSL 验收 " + tag,
            "question": "用 Python 实现 add(a, b)，返回两个数字之和，并给出测试和说明。",
            "rubric": {"total_score": 100, "正确实现加法": 80, "测试与说明": 20},
        }).json()["id"]
        report["resources"]["assignment_id"] = assignment
        student = "202610130001"
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(student + "-Smoke/add.py", "def add(a, b):\n    return a + b\n\nassert add(2, 3) == 5\nassert add(-2, 2) == 0\n")
            z.writestr(student + "-Smoke/report.txt", "本程序使用 Python 加法运算实现两个参数求和，不修改输入参数。测试覆盖正整数相加和负整数相加。add(2,3) 的结果为5，add(-2,2)的结果为0，两个断言均通过。\n")
        queued = api("POST", f"/assignments/{assignment}/submit", expected=202,
                     files={"batch_file": ("deployment-check.zip", archive.getvalue(), "application/zip")}).json()
        report["resources"]["homework_job"] = queued["job_id"]
        save()
        wait_job(queued["job_id"])
        result = api("GET", f"/assignments/{assignment}").json()["submissions"]
        assert len(result) == 1 and 0 <= result[0]["score"] <= 100, "Invalid homework score"
        assert result[0]["feedback"], "Missing homework feedback"
        passed("kafka_homework_grpc_deepseek", {"score": result[0]["score"]})

        # SUCCESS describes primary grading; pooling finishes separately.
        deadline = time.monotonic() + 210
        while time.monotonic() < deadline:
            result = api("GET", f"/assignments/{assignment}").json()["submissions"]
            if "【成绩池化处理】" in result[0]["feedback"]:
                assert 0 <= result[0]["score"] <= 100, "Invalid pooled score"
                passed("background_score_pooling", {"score": result[0]["score"]})
                break
            time.sleep(3)
        else:
            raise TimeoutError("Background score pooling did not finish")
        excel = api("GET", f"/assignments/{assignment}/export").content
        with zipfile.ZipFile(io.BytesIO(excel)) as book:
            assert "xl/workbook.xml" in book.namelist()
            assert any(name.startswith("xl/worksheets/sheet") for name in book.namelist())
        (args.output.parent / "deployment-export.xlsx").write_bytes(excel)
        passed("excel_export")

        exam = api("POST", "/exams/", expected=201, json={"name": "OCR 部署验收 " + tag}).json()["id"]
        report["resources"]["exam_id"] = exam
        api("POST", f"/exams/{exam}/questions", expected=201, json={
            "question_number": 1, "question_text": "计算 2 + 3。", "standard_answer": "5",
            "rubric": "答案为5得10分，否则0分。", "max_score": 10,
        })
        image = Image.new("RGB", (1000, 300), "white")
        fonts = [Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
        font = next((ImageFont.truetype(str(p), 48) for p in fonts if p.exists()), ImageFont.load_default())
        ImageDraw.Draw(image).text((40, 40), "Question 1\n2 + 3 = 5\nAnswer: 5", fill="black", font=font)
        page = io.BytesIO()
        image.save(page, format="PNG")
        queued = api("POST", f"/exams/{exam}/grade_submission", expected=202, data={"student_id": student},
                     files={"images": ("answer.png", page.getvalue(), "image/png")}).json()
        report["resources"]["exam_job"] = queued["job_id"]
        save()
        wait_job(queued["job_id"])
        results = api("GET", f"/exams/{exam}/results").json()
        assert len(results) == 1 and results[0]["total_score"] == 10, "OCR answer did not receive expected score"
        detail = api("GET", f"/exams/{exam}/results/{results[0]['student_exam_id']}").json()
        assert detail["answers"][0]["ocr_text"] and detail["report"]["summary"], "Missing exam OCR or summary"
        passed("kafka_exam_ocr_deepseek", {"score": results[0]["total_score"]})

        chat = api("POST", "/api/agent/chat", json={
            "message": "这是本机部署验收测试，请不要调用工具，只回复：部署验证成功。",
        }).json()
        report["resources"]["session_id"] = chat.get("session_id")
        assert "部署验证成功" in chat.get("reply", ""), "Agent did not return expected reply"
        passed("go_agent_deepseek_memory")

        if args.check_restart:
            for script in ("stop-wsl.sh", "start-wsl.sh"):
                subprocess.run(["bash", str(ROOT / "scripts" / script)], check=True, timeout=480)
            api("GET", "/api/profile")
            for job in (report["resources"]["homework_job"], report["resources"]["exam_job"]):
                assert api("GET", "/api/jobs/" + job).json()["status"] == "SUCCESS"
            assert api("GET", f"/assignments/{assignment}").json()["submissions"][0]["feedback"]
            assert api("GET", f"/exams/{exam}/results").json()[0]["total_score"] == 10
            passed("stop_start_jwt_results_job_persistence")

        api("DELETE", f"/assignments/{assignment}", expected=204)
        api("DELETE", f"/exams/{exam}", expected=204)
        if chat.get("session_id"):
            api("DELETE", "/api/sessions/" + chat["session_id"])
        passed("synthetic_resources_removed")
        report["ok"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        save()
        print("Deployment report: " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
