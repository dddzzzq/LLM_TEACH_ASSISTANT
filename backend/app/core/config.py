# 引入配置
from pydantic_settings import BaseSettings
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# 定义项目根目录的路径 (config.py往上两级)
# backend/app/core/config.py -> backend/ -> ai_grading_assistant/
BASE_DIR = Path(__file__).resolve().parent.parent

# 定义属于我们项目的基础设置，包括项目名称、API路径、数据库URL和DeepSeek API密钥
class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Grading Assistant"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = "sqlite:///./sql_app.db"
    DEEPSEEK_API_KEY: str

    
    model_config = SettingsConfigDict(
        # 使用绝对路径确保可以正确找到config
        env_file=BASE_DIR / ".env",
        case_sensitive=True,
        env_file_encoding='utf-8'
    )

settings = Settings()
