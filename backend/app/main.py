from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles # 导入 StaticFiles
import os
from .db.database import Base, engine
from .routers import assignments, exams 
import sys
import asyncio # 1. 导入 asyncio


# 确保上传目录存在，避免启动报错
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="智能化作业与试卷批改系统",
    description="一个基于FastAPI和LLM的AI助教系统，用于自动化批改作业和试卷。",
    version="2.2.0" 
)

#  挂载静态文件目录：可以在前端展示图片 
# 这样 /uploads/exams/xxx.png 就可以通过 http://localhost:8000/uploads/exams/xxx.png 访问
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册作业路由
app.include_router(assignments.router)
app.include_router(assignments.submission_router) 

# 注册新的试卷路由
app.include_router(exams.router)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "欢迎使用智能化作业与试卷批改系统API！请访问 /docs 查看详情。"}