from pathlib import Path

from benchmark.engines.tensorrt_engine import analyze_engine_precision

if __name__ == "__main__":
    engine_path = Path("engines/rtdetr_mixed_a_entropy.engine")
    if engine_path.exists():
        print(analyze_engine_precision(engine_path))
    else:
        print("Engine not found")
