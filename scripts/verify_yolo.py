
import sys
import torch
from pathlib import Path
from benchmark.engines.pytorch_engine import PyTorchEngine
from benchmark.models.yolo_adapter import YOLOAdapter

def verify_yolo():
    print("Verifying YOLO integration...")
    
    weights = [
        ("yolo11l", Path("weights/yolo11l/yolo11l.pt"), False),
        ("yolo26l", Path("weights/yolo26l/yolo26l.pt"), True),
    ]
    
    for name, path, nms_free in weights:
        if not path.exists():
            print(f"FAILED: {path} not found")
            return 1
            
        adapter = YOLOAdapter(is_nms_free=nms_free)
        engine = PyTorchEngine(name, adapter)
        engine.load_model(path)
        
        print(f"OK: {name} loaded successfully")
        
    print("YOLO integration verification passed!")
    return 0

if __name__ == "__main__":
    sys.exit(verify_yolo())
