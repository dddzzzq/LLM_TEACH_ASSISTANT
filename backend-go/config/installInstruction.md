## go配置命令
`
1. 安装 Gin Web 框架  
  go get -u github.com/gin-gonic/gin
2. 安装 GORM 及其 MySQL 驱动  
  go get -u gorm.io/gorm
  go get -u gorm.io/driver/mysql
3. 安装处理中文乱码的依赖库  
  go get golang.org/x/text
4. 安装grpc
  go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
  go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
  go get google.golang.org/grpc
  go get google.golang.org/protobuf
5. 安装用于解析文本内容的依赖库
  go get github.com/nwaples/rardecode   # 用于解析 rar
  go get github.com/ledongthuc/pdf      # 用于解析 pdf
  go get golang.org/x/text/encoding/simplifiedchinese # 用于处理 GBK 中文乱码
6. 安装用于导出excel的依赖库
  go get github.com/xuri/excelize/v2
`