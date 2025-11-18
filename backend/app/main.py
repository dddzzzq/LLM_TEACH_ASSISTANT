from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .db.database import Base, engine
from .routers import assignments, exams # 导入新的 exams 路由

app = FastAPI(
    title="智能化作业与试卷批改系统",
    description="一个基于FastAPI和LLM的AI助教系统，用于自动化批改作业和试卷。",
    version="2.2.0" # 版本升级
)

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
app.include_router(assignments.submission_router) # 增加提交路由

# 注册新的试卷路由
app.include_router(exams.router)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "欢迎使用智能化作业与试卷批改系统API！请访问 /docs 查看详情。"}