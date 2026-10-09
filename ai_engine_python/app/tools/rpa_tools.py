"""Compatibility entry point. Interactive tasks run in app.rpa.worker.

The former password/CAPTCHA automation has been retired. Platform navigation
and export semantics live in app.rpa.portal_adapter and skills/fetch-homework.
"""

def fetch_homework_sync(*args, **kwargs):
    return "旧版账号密码抓取接口已停用，请通过作业抓取控制台创建任务并人工登录"
