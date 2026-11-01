package grpcclient

import (
	"grading-gateway/pb"
	"log"
	"os"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var Client pb.ComputeServiceClient
var conn *grpc.ClientConn

// InitGrpcClient 初始化与 Python AI 节点的 gRPC 连接
func InitGrpcClient() {
	var err error
	// 后续可考虑负载均衡策略
	address := os.Getenv("AI_GRPC_ADDR")
	if address == "" {
		address = "localhost:50051"
	}
	conn, err = grpc.Dial(address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(32<<20),
			grpc.MaxCallSendMsgSize(32<<20),
		),
	)
	if err != nil {
		log.Fatalf("无法连接至 Python AI 节点: %v", err)
	}
	Client = pb.NewComputeServiceClient(conn)
	log.Printf("Python gRPC 客户端已配置 (%s)，将在首次 RPC 时验证连接", address)
}

// CloseGrpcClient 关闭连接
func CloseGrpcClient() {
	if conn != nil {
		conn.Close()
	}
}
