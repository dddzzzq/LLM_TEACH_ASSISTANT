# 指定基础镜像
FROM base_image:tag

# 设置元数据
LABEL version="1.0"
LABEL description="部署项目"

# 设置工作目录
WORKDIR /app

# 复制文件到容器中
COPY . .

# 运行命令
RUN apt-get update && apt-get install -y \
    package1 \
    package2 \
    && rm -rf /var/lib/apt/lists/*

# 暴露端口
EXPOSE 80

# 设置环境变量
ENV NODE_ENV production
ENV PORT 8080

# 定义容器启动时执行的命令
CMD ["npm", "start"]
