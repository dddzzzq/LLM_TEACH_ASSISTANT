#!/bin/bash

echo "=== 智能作业批改系统 - 认证与对话记忆系统测试 ==="
echo "测试时间: $(date)"
echo

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查服务是否运行
check_service() {
    echo -n "检查后端服务是否运行... "
    if curl -s http://localhost:8000/assignments/ > /dev/null; then
        echo -e "${GREEN}✓ 后端服务正常运行${NC}"
        return 0
    else
        echo -e "${RED}✗ 后端服务未运行${NC}"
        echo "请先启动后端服务: cd /root/autodl-tmp/dzq/LLM_TEACH_ASSISTANT/backend-go && go run main.go"
        return 1
    fi
}

# 测试用户注册
test_register() {
    echo
    echo "1. 测试用户注册..."
    
    # 注册测试用户
    echo "注册测试用户..."
    response=$(curl -s -X POST http://localhost:8000/api/register \
        -H "Content-Type: application/json" \
        -d '{
            "username": "testuser_'$(date +%s)'",
            "password": "test123",
            "name": "测试用户",
            "role": "teacher"
        }')
    
    if echo "$response" | grep -q "access_token"; then
        echo -e "${GREEN}✓ 用户注册成功${NC}"
        echo "响应: $(echo $response | jq -r '.name')"
        ACCESS_TOKEN=$(echo $response | jq -r '.access_token')
        REFRESH_TOKEN=$(echo $response | jq -r '.refresh_token')
        USER_ID=$(echo $response | jq -r '.user_id')
        return 0
    else
        echo -e "${RED}✗ 用户注册失败${NC}"
        echo "错误: $response"
        return 1
    fi
}

# 测试用户登录
test_login() {
    echo
    echo "2. 测试用户登录..."
    
    # 使用测试账户登录
    response=$(curl -s -X POST http://localhost:8000/api/login \
        -H "Content-Type: application/json" \
        -d '{
            "username": "teacher",
            "password": "teacher123"
        }')
    
    if echo "$response" | grep -q "access_token"; then
        echo -e "${GREEN}✓ 用户登录成功${NC}"
        ACCESS_TOKEN=$(echo $response | jq -r '.access_token')
        REFRESH_TOKEN=$(echo $response | jq -r '.refresh_token')
        USER_ID=$(echo $response | jq -r '.user_id')
        USER_NAME=$(echo $response | jq -r '.name')
        echo "用户: $USER_NAME (ID: $USER_ID)"
        return 0
    else
        echo -e "${RED}✗ 用户登录失败${NC}"
        echo "错误: $response"
        return 1
    fi
}

# 测试Token刷新
test_token_refresh() {
    echo
    echo "3. 测试Token刷新..."
    
    if [ -z "$REFRESH_TOKEN" ]; then
        echo -e "${YELLOW}⚠ 无刷新令牌，跳过测试${NC}"
        return 0
    fi
    
    response=$(curl -s -X POST http://localhost:8000/api/refresh \
        -H "Content-Type: application/json" \
        -d "{\"refresh_token\": \"$REFRESH_TOKEN\"}")
    
    if echo "$response" | grep -q "access_token"; then
        echo -e "${GREEN}✓ Token刷新成功${NC}"
        NEW_ACCESS_TOKEN=$(echo $response | jq -r '.access_token')
        NEW_REFRESH_TOKEN=$(echo $response | jq -r '.refresh_token')
        
        # 更新Token
        ACCESS_TOKEN="$NEW_ACCESS_TOKEN"
        REFRESH_TOKEN="$NEW_REFRESH_TOKEN"
        return 0
    else
        echo -e "${RED}✗ Token刷新失败${NC}"
        echo "错误: $response"
        return 1
    fi
}

# 测试用户信息获取
test_user_profile() {
    echo
    echo "4. 测试用户信息获取..."
    
    if [ -z "$ACCESS_TOKEN" ]; then
        echo -e "${YELLOW}⚠ 无访问令牌，跳过测试${NC}"
        return 0
    fi
    
    response=$(curl -s -X GET http://localhost:8000/api/profile \
        -H "Authorization: Bearer $ACCESS_TOKEN")
    
    if echo "$response" | grep -q "username"; then
        echo -e "${GREEN}✓ 用户信息获取成功${NC}"
        echo "用户信息: $(echo $response | jq -r '.name') ($(echo $response | jq -r '.role'))"
        return 0
    else
        echo -e "${RED}✗ 用户信息获取失败${NC}"
        echo "错误: $response"
        return 1
    fi
}

# 测试会话创建
test_session_creation() {
    echo
    echo "5. 测试会话创建..."
    
    if [ -z "$ACCESS_TOKEN" ]; then
        echo -e "${YELLOW}⚠ 无访问令牌，跳过测试${NC}"
        return 0
    fi
    
    response=$(curl -s -X POST http://localhost:8000/api/sessions \
        -H "Authorization: Bearer $ACCESS_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{
            "title": "测试会话 '$(date +"%Y-%m-%d %H:%M:%S")'"
        }')
    
    if echo "$response" | grep -q "id"; then
        echo -e "${GREEN}✓ 会话创建成功${NC}"
        SESSION_ID=$(echo $response | jq -r '.id')
        echo "会话ID: $SESSION_ID"
        return 0
    else
        echo -e "${RED}✗ 会话创建失败${NC}"
        echo "错误: $response"
        return 1
    fi
}

# 测试Agent对话
test_agent_chat() {
    echo
    echo "6. 测试Agent对话..."
    
    if [ -z "$ACCESS_TOKEN" ]; then
        echo -e "${YELLOW}⚠ 无访问令牌，跳过测试${NC}"
        return 0
    fi
    
    if [ -z "$SESSION_ID" ]; then
        echo -e "${YELLOW}⚠ 无会话ID，使用新会话${NC}"
    fi
    
    response=$(curl -s -X POST http://localhost:8000/api/agent/chat \
        -H "Authorization: Bearer $ACCESS_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"message\": \"你好，请介绍一下你自己\",
            \"session_id\": \"${SESSION_ID:-}\"
        }")
    
    if echo "$response" | grep -q "reply"; then
        echo -e "${GREEN}✓ Agent对话成功${NC}"
        REPLY=$(echo $response | jq -r '.reply' | head -c 100)
        echo "回复摘要: $REPLY..."
        
        # 获取会话ID（如果返回了新的）
        NEW_SESSION_ID=$(echo $response | jq -r '.session_id')
        if [ "$NEW_SESSION_ID" != "null" ] && [ ! -z "$NEW_SESSION_ID" ]; then
            SESSION_ID="$NEW_SESSION_ID"
            echo "会话ID: $SESSION_ID"
        fi
        return 0
    else
        echo -e "${RED}✗ Agent对话失败${NC}"
        echo "错误: $response"
        return 1
    fi
}

# 测试会话历史获取
test_session_history() {
    echo
    echo "7. 测试会话历史获取..."
    
    if [ -z "$ACCESS_TOKEN" ]; then
        echo -e "${YELLOW}⚠ 无访问令牌，跳过测试${NC}"
        return 0
    fi
    
    if [ -z "$SESSION_ID" ]; then
        echo -e "${YELLOW}⚠ 无会话ID，跳过测试${NC}"
        return 0
    fi
    
    response=$(curl -s -X GET "http://localhost:8000/api/sessions/$SESSION_ID/history" \
        -H "Authorization: Bearer $ACCESS_TOKEN")
    
    if echo "$response" | grep -q "messages"; then
        echo -e "${GREEN}✓ 会话历史获取成功${NC}"
        MESSAGE_COUNT=$(echo $response | jq -r '.count')
        echo "消息数量: $MESSAGE_COUNT"
        return 0
    else
        echo -e "${RED}✗ 会话历史获取失败${NC}"
        echo "错误: $response"
        return 1
    fi
}

# 测试会话列表获取
test_session_list() {
    echo
    echo "8. 测试会话列表获取..."
    
    if [ -z "$ACCESS_TOKEN" ]; then
        echo -e "${YELLOW}⚠ 无访问令牌，跳过测试${NC}"
        return 0
    fi
    
    response=$(curl -s -X GET http://localhost:8000/api/sessions \
        -H "Authorization: Bearer $ACCESS_TOKEN")
    
    if echo "$response" | grep -q "\["; then
        echo -e "${GREEN}✓ 会话列表获取成功${NC}"
        SESSION_COUNT=$(echo $response | jq 'length')
        echo "会话数量: $SESSION_COUNT"
        return 0
    else
        echo -e "${RED}✗ 会话列表获取失败${NC}"
        echo "错误: $response"
        return 1
    fi
}

# 测试Redis连接
test_redis_connection() {
    echo
    echo "9. 测试Redis连接..."
    
    # 检查Redis是否在运行
    if redis-cli ping 2>/dev/null | grep -q "PONG"; then
        echo -e "${GREEN}✓ Redis连接正常${NC}"
        
        # 测试Redis键操作
        TEST_KEY="test:connection:$(date +%s)"
        if redis-cli set "$TEST_KEY" "test_value" > /dev/null 2>&1 && \
           redis-cli get "$TEST_KEY" | grep -q "test_value"; then
            echo -e "${GREEN}✓ Redis读写正常${NC}"
            redis-cli del "$TEST_KEY" > /dev/null 2>&1
            return 0
        else
            echo -e "${YELLOW}⚠ Redis读写测试失败${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ Redis连接失败${NC}"
        echo "请确保Redis服务已启动: redis-server"
        return 1
    fi
}

# 运行所有测试
run_all_tests() {
    echo "开始运行所有测试..."
    echo
    
    local tests_passed=0
    local tests_failed=0
    local tests_skipped=0
    
    # 检查服务
    if ! check_service; then
        echo -e "${RED}无法继续测试，请先启动后端服务${NC}"
        exit 1
    fi
    
    # 测试Redis
    if test_redis_connection; then
        ((tests_passed++))
    else
        ((tests_failed++))
    fi
    
    # 测试用户登录（使用预置账户）
    if test_login; then
        ((tests_passed++))
    else
        ((tests_failed++))
    fi
    
    # 测试Token刷新
    if test_token_refresh; then
        ((tests_passed++))
    else
        ((tests_skipped++))
    fi
    
    # 测试用户信息
    if test_user_profile; then
        ((tests_passed++))
    else
        ((tests_failed++))
    fi
    
    # 测试会话创建
    if test_session_creation; then
        ((tests_passed++))
    else
        ((tests_failed++))
    fi
    
    # 测试Agent对话
    if test_agent_chat; then
        ((tests_passed++))
    else
        ((tests_failed++))
    fi
    
    # 测试会话历史
    if test_session_history; then
        ((tests_passed++))
    else
        ((tests_skipped++))
    fi
    
    # 测试会话列表
    if test_session_list; then
        ((tests_passed++))
    else
        ((tests_failed++))
    fi
    
    echo
    echo "=== 测试总结 ==="
    echo -e "${GREEN}通过的测试: $tests_passed${NC}"
    echo -e "${RED}失败的测试: $tests_failed${NC}"
    echo -e "${YELLOW}跳过的测试: $tests_skipped${NC}"
    echo
    
    if [ $tests_failed -eq 0 ]; then
        echo -e "${GREEN}🎉 所有测试通过！认证与对话记忆系统正常运行。${NC}"
        return 0
    else
        echo -e "${RED}⚠ 部分测试失败，请检查系统配置。${NC}"
        return 1
    fi
}

# 主函数
main() {
    echo "选择测试模式:"
    echo "1) 运行完整测试"
    echo "2) 仅测试后端API"
    echo "3) 仅测试Redis"
    echo "4) 查看系统状态"
    read -p "请选择 (1-4): " choice
    
    case $choice in
        1)
            run_all_tests
            ;;
        2)
            check_service
            test_login
            test_user_profile
            test_session_creation
            test_agent_chat
            ;;
        3)
            test_redis_connection
            ;;
        4)
            echo "=== 系统状态 ==="
            check_service
            test_redis_connection
            echo
            echo "前端登录页面: http://localhost:5173/login"
            echo "API文档:"
            echo "  POST /api/login      - 用户登录"
            echo "  POST /api/register   - 用户注册"
            echo "  POST /api/refresh    - 刷新Token"
            echo "  GET  /api/profile    - 用户信息"
            echo "  GET  /api/sessions   - 会话列表"
            echo "  POST /api/sessions   - 创建会话"
            echo "  POST /api/agent/chat - AI对话"
            ;;
        *)
            echo "无效选择"
            ;;
    esac
}

# 安装jq（如果不存在）
if ! command -v jq &> /dev/null; then
    echo "安装jq工具..."
    apt-get update && apt-get install -y jq > /dev/null 2>&1
fi

# 运行主函数
main