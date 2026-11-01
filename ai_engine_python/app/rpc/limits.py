"""Keep transport limits symmetric with the Go compute client."""
GRPC_OPTIONS = [
    ('grpc.max_receive_message_length', 32 << 20),
    ('grpc.max_send_message_length', 32 << 20),
]
