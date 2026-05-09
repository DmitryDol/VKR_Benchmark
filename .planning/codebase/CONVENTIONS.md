# Coding Conventions

**Analysis Date:** 2026-05-09

## Naming Patterns

**Files:**
- Use `snake_case.py` for all module files: `coco_loader.py`, `pytorch_engine.py`, `onnx_export.py`
- Use short, descriptive names reflecting the single responsibility of the module

**Classes:**
- Use `PascalCase`: `COCODataLoader`, `BaseEngine`, `PyTorchEngine`, `BenchmarkResult`, `ResultLogger`
- Acronyms stay uppercase: `COCO`, `ONNX` (e.g., `COCOSample`, not `CocoSample`)
- Dataclasses for plain data containers: `Detection`, `COCOAnnotation`, `COCOSample`, `BenchmarkResult`
- ABC for abstract base classes: `BaseEngine(ABC)`
- Protocol for structural subtyping: `ModelAdapter(Protocol)` with `@runtime_checkable`

**Functions:**
- Use `snake_case`: `load_model()`, `benchmark_latency()`, `export_to_onnx()`, `simplify_onnx()`
- Private methods prefixed with underscore: `_load_sample()`, `_append_csv()`
- Verbs first for actions: `load_model`, `export_to_onnx`, `validate_onnx`, `measure_vram`
- Properties for computed attributes: `model_size_mb`, `coco`

**Variables:**
- Use `snake_case`: `image_id`, `model_name`, `engine_type`
- Private instance variables with underscore prefix: `self._model`, `self._coco`, `self._results`
- Constants in `UPPER_SNAKE_CASE` at module level: `WARMUP_RUNS`, `MEASURE_RUNS`, `IMAGENET_MEAN`, `IMAGENET_STD`, `DEFAULT_DYNAMIC_AXES`
- Dict-type constants fully typed: `COCO_91_TO_80: dict[int, int] = {...}`

**Type Aliases:**
- None currently used; built-in generics preferred (`dict[str, float]`, `list[int]`, `tuple[int, int]`)

## Type Annotations

**Strict enforcement:** Ruff rule `ANN` (flake8-annotations) is enabled. All functions must have full type annotations.

**Patterns:**
- All function parameters and return types annotated: `def load_model(self, weights_path: Path) -> None:`
- Use `from __future__ import annotations` in every module for PEP 604 union syntax (`X | None`)
- Use `TYPE_CHECKING` guard for import-only types to avoid circular imports and runtime overhead:
  ```python
  from __future__ import annotations
  from typing import TYPE_CHECKING

  if TYPE_CHECKING:
      from pathlib import Path
      from numpy.typing import NDArray
      from benchmark.data.coco_loader import COCOSample
  ```
- Dataclass fields fully typed with generics: `boxes: NDArray[np.float32]`, `labels: NDArray[np.int64]`
- Use `object` for polymorphic inputs/outputs in abstract methods: `def infer(self, inputs: object) -> object:`
- Optional fields use `X | None = None` syntax: `limit: int | None = None`, `macs: float | None = None`
- `# type: ignore[attr-defined]` used sparingly for torch backend attributes that lack stubs:
  ```python
  torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
  ```

## Import Organization

**Order (enforced by ruff `I` / isort):**
1. `__future__` imports (always first): `from __future__ import annotations`
2. Standard library: `import logging`, `import time`, `from abc import ABC, abstractmethod`, `from pathlib import Path`
3. Third-party: `import numpy as np`, `import torch`, `import onnx`, `from PIL import Image`
4. Local project: `from benchmark.engines.base import BaseEngine, Detection`, `from benchmark.utils.logger import BenchmarkResult`

**Conventions:**
- Use absolute imports from `benchmark.*` package, never relative imports
- Group `TYPE_CHECKING` imports in a single `if TYPE_CHECKING:` block after all runtime imports
- Import specific names, not entire modules: `from benchmark.engines.base import BaseEngine, Detection` (not `import benchmark.engines.base`)
- Alias numpy as `np`: `import numpy as np`
- Alias torchvision functional as `tvf`: `import torchvision.transforms.functional as tvf`

**`__init__.py` barrel exports:**
- Use explicit `__all__` lists in `__init__.py` files:
  ```python
  from benchmark.engines.base import BaseEngine
  from benchmark.engines.pytorch_engine import PyTorchEngine

  __all__ = ["BaseEngine", "PyTorchEngine"]
  ```

## Error Handling

**Patterns:**
- Use `msg` variable + raise pattern (enforced by ruff to avoid inline f-string in raise):
  ```python
  msg = f"Images directory not found: {self.images_dir}"
  raise FileNotFoundError(msg)
  ```
- Use `RuntimeError` for precondition violations (e.g., model not loaded):
  ```python
  if self._model is None:
      msg = "Model not loaded. Call load_model() first."
      raise RuntimeError(msg)
  ```
- Use `FileNotFoundError` for missing files/directories
- Let library exceptions propagate naturally (e.g., `onnx.checker.check_model` raises `ValidationError`)
- Use `logger.warning()` for non-fatal issues: `logger.warning("onnxsim validation failed for %s", model_path)`
- Guard empty results: `if not coco_results: ... return {"map_50": 0.0, "map_50_95": 0.0}`

## Logging

**Framework:** Python stdlib `logging`

**Setup:**
- Module-level logger: `logger = logging.getLogger(__name__)` at top of every module
- No root logger configuration in library code (left to CLI entry point)

**Patterns:**
- Use `%s` formatting (not f-strings) in log calls for lazy evaluation:
  ```python
  logger.info("COCODataLoader: %d images loaded", len(self._image_ids))
  logger.info("Model loaded: %s (%.1f MB)", self.model_name, self.model_size_mb)
  ```
- `logger.info()` for operational milestones: model loaded, export complete, measurement phases
- `logger.warning()` for recoverable issues: validation failures, empty results
- No `logger.debug()` calls currently used
- No `logger.error()` calls; errors are raised as exceptions instead

## Docstring Style

**Format:** NumPy-style docstrings

**Module-level:**
- Every module has a one-line docstring: `"""COCO val2017 DataLoader for single-image inference benchmarking."""`
- Multi-line for complex modules:
  ```python
  """ONNX export and optimization pipeline.

  Converts PyTorch models to ONNX format with onnx-simplifier
  optimization for subsequent TensorRT conversion.
  """
  ```

**Class-level:**
- Describe purpose and list subclass obligations for abstract classes
- Include `Parameters` section for classes with `__init__` parameters:
  ```python
  class PyTorchEngine(BaseEngine):
      """FP32 baseline inference engine using pure PyTorch.

      Parameters
      ----------
      model_name : str
          Human-readable model identifier (e.g. "rt-detr-l").
      adapter : ModelAdapter
          Model-specific adapter for loading and output parsing.
      """
  ```

**Function-level:**
- One-line summary for simple methods: `"""Load model weights and prepare for inference."""`
- Full NumPy-style with `Parameters`, `Returns`, `Raises` for public API functions:
  ```python
  def export_to_onnx(...) -> Path:
      """Export a PyTorch model to ONNX format.

      Parameters
      ----------
      model : nn.Module
          PyTorch model in eval mode.
      output_path : Path
          Destination path for the .onnx file.

      Returns
      -------
      Path
          Path to the exported ONNX model.
      """
  ```

**Protocol methods:**
- Use `...` (ellipsis) as body, with docstrings describing the contract

## Ruff Configuration

**Location:** `pyproject.toml` `[tool.ruff]` section

**Settings:**
- `target-version = "py313"` (Python 3.13)
- `line-length = 100`

**Enabled rule sets:**
| Code | Plugin | Purpose |
|------|--------|---------|
| `F` | Pyflakes | Undefined names, unused imports |
| `E`, `W` | pycodestyle | PEP 8 errors and warnings |
| `I` | isort | Import ordering |
| `N` | pep8-naming | Naming conventions |
| `UP` | pyupgrade | Modern Python syntax |
| `ANN` | flake8-annotations | Type annotation enforcement |
| `B` | flake8-bugbear | Common bugs and design problems |
| `A` | flake8-builtins | Shadowing built-in names |
| `SIM` | flake8-simplify | Simplifiable code |
| `TCH` | flake8-type-checking | TYPE_CHECKING import optimization |
| `RUF` | Ruff-specific | Ruff's own rules |
| `S` | flake8-bandit | Security checks |
| `PT` | flake8-pytest-style | Pytest best practices |
| `C4` | flake8-comprehensions | Comprehension improvements |
| `PIE` | flake8-pie | Misc. improvements |
| `T20` | flake8-print | Disallow print statements |
| `RET` | flake8-return | Return statement checks |
| `ARG` | flake8-unused-arguments | Unused function arguments |
| `PL` | pylint | Pylint rules subset |

**Ignored rules:**
- `ANN401` — Allow `Any` type in annotations
- `S101` — Allow `assert` (needed in scripts; fully ignored in `tests/**`)
- `T201` — Allow `print()` in scripts
- `PLR0913` — Allow many function arguments (necessary for dataclasses with many fields)

**Per-file overrides:**
- `tests/**/*.py`: Ignore `S101` (assert) and `ANN` (type annotations not required in tests)

**Formatter:**
- `quote-style = "double"` — Use double quotes for all strings
- `indent-style = "space"` — Spaces, not tabs (4-space indent implied by PEP 8)

## Code Style Patterns

**Dataclasses over plain classes for data:**
- Use `@dataclass` for all data containers: `Detection`, `COCOAnnotation`, `COCOSample`, `BenchmarkResult`, `COCODataLoader`
- Use `field(default_factory=...)` for mutable defaults
- Use `field(init=False, repr=False)` for computed/internal fields
- Use `__post_init__` for validation and initialization logic

**Protocol for structural subtyping:**
- Use `Protocol` + `@runtime_checkable` for adapter patterns (`ModelAdapter`)
- Prefer protocols over inheritance for defining interfaces consumed by engines

**Abstract base classes for engine hierarchy:**
- Use `ABC` + `@abstractmethod` for engine contracts (`BaseEngine`)
- Template Method pattern: `run_full_benchmark()` calls abstract `preprocess()`, `infer()`, `postprocess()`

**Context managers for resource safety:**
- Use `torch.no_grad()` context manager for inference
- Use `with` for file operations: `with csv_path.open("a", ...) as f:`

**pathlib over os.path:**
- Use `Path` throughout: `Path("data/val2017")`, `output_path.parent.mkdir(parents=True, exist_ok=True)`
- Convert to string only at library boundaries: `str(output_path)`, `str(model_path)`

**Numeric arrays:**
- Explicit dtype on all numpy array creation: `np.array(boxes, dtype=np.float32)`
- Use `.reshape(-1, 4)` for safety on empty arrays

**Module constants at top:**
- Place constants after imports, before class definitions
- Fully type-annotated: `WARMUP_RUNS: int = 50`

---

*Convention analysis: 2026-05-09*
