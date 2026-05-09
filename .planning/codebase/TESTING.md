# TESTING.md — Testing Infrastructure

> Generated: 2026-05-09

## Current State

**No tests exist.** Zero test files, no `tests/` directory, no CI/CD pipeline.

## Test Framework Configuration

- **Ruff**: `PT` rules (flake8-pytest-style) are enabled in `pyproject.toml`, indicating pytest is the intended framework
- **pytest dependency**: Not listed in `pyproject.toml` dependencies — needs to be added as dev dependency
- **Per-file ignores**: `tests/**/*.py` already configured to ignore `S101` (assert) and `ANN` (annotations)

## Test Coverage

| Module | Coverage | Notes |
|--------|----------|-------|
| `benchmark/data/coco_loader.py` | 0% | Needs COCO fixtures or mocks |
| `benchmark/engines/base.py` | 0% | Abstract class — test via concrete subclass |
| `benchmark/engines/pytorch_engine.py` | 0% | Requires GPU or mocked CUDA |
| `benchmark/engines/onnx_export.py` | 0% | Requires model + onnx installed |
| `benchmark/utils/logger.py` | 0% | Pure data — easiest to test |

## Recommended Test Structure

```
tests/
├── conftest.py              # Shared fixtures (dummy models, sample images)
├── test_coco_loader.py      # Data loading, annotation parsing, ID mapping
├── test_logger.py           # BenchmarkResult, CSV/JSON output
├── test_base_engine.py      # Benchmark protocol via mock engine
├── test_pytorch_engine.py   # FP32 baseline (GPU-conditional)
├── test_onnx_export.py      # Export/simplify pipeline
└── test_tensorrt_engine.py  # (future) TensorRT engine tests
```

## Testing Considerations

### GPU-Conditional Tests
- Use `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")`
- Separate CPU-testable logic from GPU-dependent inference
- VRAM tests only meaningful on target hardware

### COCO Fixtures
- Create a minimal COCO annotation JSON with 2-3 images
- Use tiny synthetic images (e.g., 64x64) for fast tests
- Store in `tests/fixtures/`

### Priority Testing Targets
1. **`ResultLogger`** — pure I/O, no GPU needed, high ROI
2. **`COCODataLoader`** — data integrity is critical for valid benchmarks
3. **`BaseEngine.benchmark_latency`** — validate timing logic via mock engine
4. **`Detection` dataclass** — validate box format handling
5. **ONNX export pipeline** — validate export/simplify/validate chain

## Gaps & Risks

- No regression tests for mAP calculation correctness
- No validation of COCO ID mapping (91↔80 roundtrip)
- No integration tests for full benchmark pipeline
- Double inference bug in warm-up loop (`base.py:96-97`) untested
- No smoke test for ModelAdapter protocol compliance