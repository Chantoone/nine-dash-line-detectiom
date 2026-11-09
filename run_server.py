#!/usr/bin/env python3
"""
Script khởi chạy server cho Nine Dash Line Detection Demo.
Sử dụng: python run_server.py [--host 0.0.0.0] [--port 8000] [--checkpoint PATH]
"""
import argparse
import os
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Khởi chạy Nine Dash Line Detection Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn checkpoint")
    parser.add_argument("--reload", action="store_true", help="Auto-reload khi thay đổi code")
    args = parser.parse_args()
    
    if args.checkpoint:
        os.environ["MODEL_CHECKPOINT"] = args.checkpoint
    
    print(f"🚀 Khởi chạy server tại http://{args.host}:{args.port}")
    print(f"📖 API docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == '__main__':
    main()
