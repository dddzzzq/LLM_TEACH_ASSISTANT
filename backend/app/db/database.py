from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from ..core.config import settings

# 迁移到mysql数据库连接
DATABASE_URL = settings.DATABASE_URL 

# 2. 创建异步数据库引擎
# 添加 pool_recycle 参数，防止连接空闲过久断开
engine = create_async_engine(
    DATABASE_URL, 
    echo=True,
    pool_recycle=3600, # 1小时回收一次连接
    pool_pre_ping=True # 每次连接前检测是否存活
)

AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# 获取数据库会话
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session