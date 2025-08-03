from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .db.database import Base, engine
from .routers import assignments

app = FastAPI(
    title="智能化作业与试卷批改系统",
    description="一个基于FastAPI和LLM的AI助教系统，用于自动化批改作业。",
    version="2.1.0"
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

app.include_router(assignments.router)
app.include_router(assignments.submission_router) # 新增此行

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "欢迎使用智能化作业与试卷批改系统API！请访问 /docs 查看详情。"}