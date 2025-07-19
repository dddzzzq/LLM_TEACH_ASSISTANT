# 引入配置
from pydantic_settings import BaseSettings
from pydantic_settings import BaseSettings, SettingsConfigDict

# 定义属于我们项目的基础设置，包括项目名称、API路径、数据库URL和DeepSeek API密钥
class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Grading Assistant"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = "sqlite:///./sql_app.db"
    DEEPSEEK_API_KEY: str
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        env_file_encoding='utf-8' 

settings = Settings()
