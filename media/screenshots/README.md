# Console Screenshots for Diploma Defense

This folder contains three manually captured PNG screenshots that provide visual evidence of the pipeline running end-to-end, as required for the diploma chapter on the optimization process.

## Expected files

- `trt_build_log.png` — terminal showing a TensorRT engine build log; the capture must include the `[I] Total Activation Memory`, `[I] Total Weights Memory`, and `[I] Total Host/Device Memory` lines plus a successful `Engine built in` line at the bottom.
- `benchmark_output.png` — terminal showing the output of `benchmark run --model rt-detr --stage 4_trt_fp16`, including the latency/mAP summary table printed at the end of the run.
- `pycocotools_output.png` — terminal showing the 12-line pycocotools COCOeval summary block (`Average Precision (AP) @ [...] = 0.xxx`) printed at the end of any `evaluate_accuracy` run.

## How to capture

### 1. trt_build_log.png

Run the following command in PowerShell (from the project root):

```powershell
uv run python -m benchmark run --model rt-detr --stage 4_trt_fp16
```

Bash users (WSL/Linux) can drop the `uv run python -m` prefix if the package is installed, and substitute `&&` for command chaining as needed.

Wait for the log line that reads `Engine built in X.X seconds` before taking the screenshot. The TRT build log appears early in the run, before warm-up begins.

Terminal configuration:
- At least 120 columns wide.
- Dark background, monospace font (Cascadia Mono 12 pt on Windows Terminal is recommended).
- Do NOT scroll the buffer upward — capture the bottom of the output where the memory-summary lines appear.

Save the screenshot as `media/screenshots/trt_build_log.png` using Windows `Win + Shift + S` (Snipping Tool), saved as PNG.

### 2. benchmark_output.png

Run the following command in PowerShell (from the project root):

```powershell
uv run python -m benchmark run --model rt-detr --stage 4_trt_fp16
```

Wait for the final output line `All stages complete.` and for the latency/mAP summary table to appear just before it.

Terminal configuration:
- At least 120 columns wide.
- Dark background, monospace font (Cascadia Mono 12 pt on Windows Terminal).
- Capture the bottom of the output showing the summary table — do NOT scroll up to intermediate progress output.

Save as `media/screenshots/benchmark_output.png` using `Win + Shift + S`, PNG format.

### 3. pycocotools_output.png

The pycocotools COCOeval block is printed automatically at the end of any `evaluate_accuracy` call. It appears as part of the same `benchmark run` invocation above. If you need to isolate it, run:

```powershell
uv run python -m benchmark run --model rt-detr --stage 4_trt_fp16
```

Wait for the 12-line block that begins with:

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.xxx
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.xxx
```

The block ends with:

```text
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.xxx
```

Terminal configuration:
- At least 120 columns wide.
- Dark background, monospace font (Cascadia Mono 12 pt on Windows Terminal).
- Capture the final screenful — the full 12-line block must be visible in the capture.

Save as `media/screenshots/pycocotools_output.png` using `Win + Shift + S`, PNG format.

## Verification

Run these commands to confirm all three PNG files are present.

PowerShell:

```powershell
foreach ($f in @("trt_build_log.png", "benchmark_output.png", "pycocotools_output.png")) {
    if (Test-Path -Path "media/screenshots/$f") {
        Write-Host "[OK] $f"
    } else {
        Write-Host "[MISSING] $f"
    }
}
```

Bash (WSL/Linux):

```bash
for f in trt_build_log.png benchmark_output.png pycocotools_output.png; do
  if [ -f "media/screenshots/$f" ]; then echo "[OK] $f"; else echo "[MISSING] $f"; fi
done
```

Expected output before capturing: three `[MISSING]` lines, exit code 0.
Expected output after capturing: three `[OK]` lines.

## Notes

- All three screenshots are MANUAL — there is no automation for this task. The screenshots must be captured during a human-attended terminal session.
- Screenshots should show the FINAL screenful at the end of the relevant run. Do NOT include intermediate progress bars or log lines from earlier in the run.
- File format MUST be PNG. JPEG is not acceptable — compression artifacts make log text unreadable and the advisor spec explicitly requests PNG.
- The screenshots are part of the diploma chapter on the optimization process. They will be referenced by figure number in the diploma text.
