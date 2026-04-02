# 项目根目录变量
PROJECT_ROOT := /root/autodl-tmp/dzq/LLM_TEACH_ASSISTANT
KAFKA_DIR := /root/autodl-tmp/dzq/kafka_2.13-4.0.1
PY_DIR := $(PROJECT_ROOT)/ai_engine_python
GO_DIR := $(PROJECT_ROOT)/backend-go
VUE_DIR := $(PROJECT_ROOT)/vue-grading-frontend

# 你的大模型 API 密钥
API_KEY := "sk-bb23450a0e524194afbbe217bd1d49b9"

.PHONY: start stop status logs-go logs-py

start:
	@echo "=== 0. 启动 MySQL 数据库 ==="
	/etc/init.d/mysql start
	@echo "=== 1. 启动 Redis ==="
	nohup redis-server > redis.log 2>&1 &
	@echo "=== 2. 启动 Kafka (KRaft单机模式) ==="
	cd $(KAFKA_DIR) && nohup bin/kafka-server-start.sh config/server.properties > kafka.log 2>&1 &
	@sleep 3
	@echo "=== 3. 启动 Python gRPC AI引擎 ==="
	cd $(PY_DIR) && nohup conda run -n ai_grade_assistant python app/grpc_server.py > py_engine.log 2>&1 &
	@sleep 2
	@echo "=== 4. 启动 Go 后端网关 ==="
	cd $(GO_DIR) && export DEEPSEEK_API_KEY=$(API_KEY) && nohup go run main.go > go_backend.log 2>&1 &
	@echo "=== 5. 启动 vue 前端 ==="
	cd $(VUE_DIR) && nohup npm run dev > vue_fronted.log 2>&1 &
	@echo "🎉 AI 评分微服务系统已全部启动！"

stop:
	@echo "正在关闭所有服务..."
	-pkill -f "go run main.go"
	-pkill -f "python app/grpc_server.py"
	-pkill -f "kafka.Kafka"
	@echo "✅ 系统已完全停止。"

status:
	@echo "--- 端口监听状态 ---"
	@netstat -tulnp | grep -E "9092|6379|50051|8080|5173" || echo "未检测到相关端口"

logs-go:
	tail -f $(GO_DIR)/go_backend.log

logs-py:
	tail -f $(PY_DIR)/py_engine.log

logs-vue:
	tail -f $(PY_DIR)/vue_fronted.log

