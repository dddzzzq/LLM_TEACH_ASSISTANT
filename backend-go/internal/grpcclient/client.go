package grpcclient

import (
	"grading-gateway/pb"
	"log"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var Client pb.ComputeServiceClient
var conn *grpc.ClientConn

// InitGrpcClient 初始化与 Python AI 节点的 gRPC 连接
func InitGrpcClient() {
	var err error
	// 后续可考虑负载均衡策略
	conn, err = grpc.Dial("localhost:50051",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(1024*1024*50), // 50MB
			grpc.MaxCallSendMsgSize(1024*1024*50), // 50MB
		),
	)
	if err != nil {
		log.Fatalf("无法连接至 Python AI 节点: %v", err)
	}
	Client = pb.NewComputeServiceClient(conn)
	log.Println("成功连接到 Python gRPC 服务 (localhost:50051)")
}

// CloseGrpcClient 关闭连接
func CloseGrpcClient() {
	if conn != nil {
		conn.Close()
	}
}
