import sys
import os
import pandas as pd
from sqlalchemy import create_engine, text

# --- 路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.append(backend_dir)

try:
    from app.db.models import Base
except ImportError as e:
    print(f"错误: 无法导入应用模型。请确保你在 backend/ 目录下运行此脚本，或检查路径设置。\n详情: {e}")
    sys.exit(1)

# 1. 源数据库 (SQLite)
SQLITE_DB_PATH = os.path.join(backend_dir, "grading_system.db")
SQLITE_SYNC_URL = f"sqlite:///{SQLITE_DB_PATH}"

# 2. 目标数据库 (MySQL)
MYSQL_URL = "mysql+pymysql://root:123456@localhost:3306/grading_system?charset=utf8mb4"

def migrate():
    print("========================================")
    print("   AI 助教系统 - 数据库迁移工具")
    print("   (SQLite -> MySQL 8.0)")
    print("========================================")

    # 1. 检查 SQLite 文件是否存在
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"[Error] 未找到源数据库文件: {SQLITE_DB_PATH}")
        return

    print(f"[1/5] 连接源数据库 (SQLite)...")
    try:
        sqlite_engine = create_engine(SQLITE_SYNC_URL)
        # 测试连接
        with sqlite_engine.connect() as conn:
            pass
    except Exception as e:
        print(f"连接 SQLite 失败: {e}")
        return

    print(f"[2/5] 连接目标数据库 (MySQL)...")
    try:
        mysql_engine = create_engine(MYSQL_URL)
        with mysql_engine.connect() as conn:
            # 检查数据库是否为空（或是否存在）
            conn.execute(text("SELECT 1"))
    except Exception as e:
        print(f"[Error] 连接 MySQL 失败: {e}")
        print("请检查：\n1. MySQL 服务是否启动\n2. 数据库 'grading_system' 是否已创建 (CREATE DATABASE grading_system;)\n3. 账号密码是否正确")
        return

    print(f"[3/5] 初始化 MySQL 表结构...")
    try:
        # 使用 SQLAlchemy Base 元数据自动在 MySQL 中创建所有表
        Base.metadata.create_all(mysql_engine)
        print("表结构创建成功。")
    except Exception as e:
        print(f"[Error] 创建表结构失败: {e}")
        return

    print(f"[4/5] 开始数据迁移...")
    
    # 定义迁移顺序（处理外键依赖）
    # 顺序：无依赖表 -> 有依赖表
    migration_order = [
        "assignments",             # 基础表
        "exams",                   # 基础表
        "exam_questions",          # 依赖 exams
        "submissions",             # 依赖 assignments
        "student_exams",           # 依赖 exams
        "student_exam_images",     # 依赖 student_exams
        "student_question_answers",# 依赖 student_exams, exam_questions
        "student_exam_reports"     # 依赖 student_exams
    ]

    with mysql_engine.connect() as mysql_conn:
        # 临时关闭外键检查，防止因插入顺序或自引用导致的问题
        mysql_conn.execute(text("SET FOREIGN_KEY_CHECKS=0;"))
        
        for table_name in migration_order:
            print(f"  -> 正在迁移表 '{table_name}' ... ", end="")
            try:
                # 从 SQLite 读取数据
                try:
                    df = pd.read_sql_table(table_name, sqlite_engine)
                except ValueError:
                    # 如果 SQLite 中没有这个表（比如是新加的模型但没运行过），跳过
                    print("跳过 (源库中无此表)")
                    continue

                if df.empty:
                    print("跳过 (无数据)")
                    continue

                # 写入 MySQL
                # if_exists='append' 表示追加数据，因为表结构已经在第3步创建好了
                df.to_sql(table_name, mysql_conn, if_exists='append', index=False)
                print(f"成功 ({len(df)} 条记录)")
                
            except Exception as e:
                print(f"失败! \n错误信息: {e}")

        # 恢复外键检查
        mysql_conn.execute(text("SET FOREIGN_KEY_CHECKS=1;"))
        mysql_conn.commit()

    print("========================================")
    print("[5/5] 迁移完成！请验证 MySQL 数据。")
    print("========================================")

if __name__ == "__main__":
    migrate()