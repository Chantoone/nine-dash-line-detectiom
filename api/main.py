#!/usr/bin/env python3
"""
FastAPI Backend cho Nine Dash Line Detection Demo.
Cung cấp API endpoint để upload ảnh và nhận kết quả phát hiện.
"""
import os
import io
import uuid
import time
import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.default import Config
from src.models.faster_rcnn import get_model
from src.data.dataset import get_transform
from src.utils.visualization import draw_detections, CLASS_NAMES

# =====================================================================================
# KHỞI TẠO ỨNG DỤNG
# =====================================================================================
app = FastAPI(
    title="Nine Dash Line Detection API",
    description="API phát hiện Đường Lưỡi Bò trong ảnh bản đồ sử dụng Faster R-CNN + Knowledge Distillation",
    version="1.0.0",
)

# CORS cho frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================================
# GLOBAL STATE
# =====================================================================================
config = Config()
model = None
device = None
transform = None

# Thư mục lưu kết quả
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "web" / "uploads"
RESULT_DIR = Path(__file__).resolve().parent.parent / "web" / "results"
STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"


def load_model(checkpoint_path: str = None):
    """Load mô hình từ checkpoint."""
    global model, device, transform
    
    device = config.get_device()
    print(f"Sử dụng thiết bị: {device}")
    
    model = get_model(config.model.num_classes, pretrained=False)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Đang tải checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Đã tải checkpoint thành công!")
    else:
        print("CẢNH BÁO: Không tìm thấy checkpoint. Sử dụng pretrained model.")
        model = get_model(config.model.num_classes, pretrained=True)
    
    model.to(device)
    model.eval()
    transform = get_transform(train=False)


@app.on_event("startup")
async def startup_event():
    """Khởi tạo khi server bắt đầu."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Tìm checkpoint
    checkpoint_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    checkpoint_path = None
    
    # Ưu tiên best_model.pth
    best_model = checkpoint_dir / "best_model.pth"
    if best_model.exists():
        checkpoint_path = str(best_model)
    else:
        # Tìm checkpoint mới nhất
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("model_epoch_*.pth"))
            if checkpoints:
                checkpoint_path = str(checkpoints[-1])
    
    # Kiểm tra biến môi trường
    env_checkpoint = os.environ.get("MODEL_CHECKPOINT")
    if env_checkpoint and os.path.exists(env_checkpoint):
        checkpoint_path = env_checkpoint
    
    load_model(checkpoint_path)


# =====================================================================================
# API ENDPOINTS
# =====================================================================================

@app.get("/")
async def root():
    """Trang chủ - phục vụ frontend."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Nine Dash Line Detection API", "docs": "/docs"}


@app.get("/api/health")
async def health_check():
    """Kiểm tra trạng thái server."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
    }


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Ngưỡng confidence")
):
    """
    Phát hiện Đường Lưỡi Bò trong ảnh.
    
    - **file**: File ảnh (JPEG, PNG)
    - **threshold**: Ngưỡng confidence (0.0 - 1.0)
    
    Returns: JSON chứa kết quả phát hiện và ảnh kết quả dạng base64.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa được tải")
    
    # Validate file
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file ảnh JPEG, PNG, WebP")
    
    try:
        start_time = time.time()
        
        # Đọc ảnh
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_image = np.array(image)
        
        # Tiền xử lý
        transformed = transform(image=original_image, bboxes=[], labels=[])
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            predictions = model(image_tensor)[0]
        
        # Lọc kết quả
        keep = predictions['scores'] >= threshold
        filtered = {
            'boxes': predictions['boxes'][keep],
            'labels': predictions['labels'][keep],
            'scores': predictions['scores'][keep],
        }
        
        inference_time = time.time() - start_time
        
        # Vẽ kết quả lên ảnh
        result_image = draw_detections(original_image, filtered, threshold)
        
        # Chuyển ảnh kết quả sang base64
        result_pil = Image.fromarray(result_image)
        buffer = io.BytesIO()
        result_pil.save(buffer, format="JPEG", quality=90)
        result_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Chuẩn bị response
        detections = []
        for box, label, score in zip(
            filtered['boxes'].cpu().numpy(),
            filtered['labels'].cpu().numpy(),
            filtered['scores'].cpu().numpy()
        ):
            detections.append({
                "bbox": {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                },
                "class_id": int(label),
                "class_name": CLASS_NAMES.get(int(label), f"class_{label}"),
                "confidence": float(score),
            })
        
        return JSONResponse({
            "success": True,
            "num_detections": len(detections),
            "detections": detections,
            "inference_time_ms": round(inference_time * 1000, 2),
            "image_size": {"width": original_image.shape[1], "height": original_image.shape[0]},
            "threshold": threshold,
            "result_image": f"data:image/jpeg;base64,{result_base64}",
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")


@app.post("/api/detect-batch")
async def detect_batch(
    files: list[UploadFile] = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    """Phát hiện trên nhiều ảnh cùng lúc."""
    if model is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa được tải")
    
    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            original_image = np.array(image)
            
            transformed = transform(image=original_image, bboxes=[], labels=[])
            image_tensor = transformed['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                predictions = model(image_tensor)[0]
            
            keep = predictions['scores'] >= threshold
            num_detections = keep.sum().item()
            
            has_nine_dash_line = num_detections > 0
            max_confidence = float(predictions['scores'][keep].max()) if has_nine_dash_line else 0.0
            
            results.append({
                "filename": file.filename,
                "has_nine_dash_line": has_nine_dash_line,
                "num_detections": num_detections,
                "max_confidence": max_confidence,
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
            })
    
    return JSONResponse({
        "success": True,
        "total_files": len(files),
        "results": results,
    })


@app.get("/api/model-info")
async def model_info():
    """Thông tin về mô hình."""
    return {
        "architecture": "Faster R-CNN (ResNet-50-FPN)",
        "knowledge_distillation": True,
        "num_classes": config.model.num_classes,
        "class_names": CLASS_NAMES,
        "input_normalization": {
            "mean": list(config.data.image_mean),
            "std": list(config.data.image_std),
        },
    }


# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
