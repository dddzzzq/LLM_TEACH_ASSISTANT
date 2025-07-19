from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# 定义数据库连接
DATABASE_URL = "sqlite+aiosqlite:///./grading_system.db"

# 创建异步数据库引擎
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# 获取数据库会话
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session