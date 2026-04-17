# 引入配置
from pydantic_settings import BaseSettings
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 定义属于我们项目的基础设置，包括项目名称、API路径、数据库URL和DeepSeek API密钥
class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Grading Assistant"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = "mysql+aiomysql://root:123456@localhost:3306/grading_system"
    DEEPSEEK_API_KEY: str
    
    # 截取长度配置
    MAX_STUDENT_ANSWER_LENGTH: int = 150000  # 学生答案最大长度（字符）
    MAX_CODE_CONTENT_LENGTH: int = 100000    # 代码内容最大长度（字符）
    MAX_DOC_CONTENT_LENGTH: int = 75000      # 文档内容最大长度（字符）
    MAX_PLAGIARISM_TEXT_LENGTH: int = 50000  # 抄袭分析文本长度（字符）
    USE_TOKEN_BASED_TRUNCATION: bool = False # 是否使用token截取（未来功能）

    
    model_config = SettingsConfigDict(
        # 使用绝对路径确保可以正确找到config
        env_file=BASE_DIR / ".env",
        case_sensitive=True,
        env_file_encoding='utf-8'
    )

settings = Settings()
