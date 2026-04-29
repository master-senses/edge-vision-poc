"""
Edge Vision POC Lab — Deployment Readiness Analyzer
Ingests MP4 or RTSP, samples frames, runs Moondream inference,
and outputs a JSON Deployment Readiness Report.
"""

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
from PIL import Image

try:
    import moondream as md
except ImportError:
    print("ERROR: moondream SDK not installed. Run: pip install moondream", file=sys.stderr)
    sys.exit(1)


@dataclass
class RunConfig:
    source: str
    question: str
    sample_every_n: int
    max_frames: Optional[int]
    infer_timeout_s: float
    api_key: str
    output: Optional[str]


@dataclass
class IngestStats:
    source_fps: float = 0.0
    total_frames_in_source: int = 0
    frames_read: int = 0
    decode_errors: int = 0


@dataclass
class InferenceRecord:
    frame_index: int
    latency_ms: float
    answer: str
    success: bool
    error: Optional[str] = None


@dataclass
class FailureSummary:
    decode_errors: int = 0
    inference_timeouts: int = 0
    inference_errors: int = 0

    @property
    def total(self) -> int:
        return self.decode_errors + self.inference_timeouts + self.inference_errors


def _detect_device() -> dict:
    """Report GPU availability honestly."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            used_mb, total_mb = int(parts[0].strip()), int(parts[1].strip())
            return {"device": "cuda", "gpu_mem_used_mb": used_mb, "gpu_mem_total_mb": total_mb, "note": None}
    except Exception:
        pass
    return {
        "device": "cpu",
        "gpu_mem_used_mb": None,
        "gpu_mem_total_mb": None,
        "note": "No GPU detected; latency reflects CPU + network path only. "
                "For Jetson/bare-metal benchmarks, rerun on target hardware.",
    }


def _open_capture(source: str) -> cv2.VideoCapture:
    """Open VideoCapture for MP4 path or RTSP URL."""
    is_rtsp = source.lower().startswith("rtsp://")
    if is_rtsp:
        # Prefer TCP transport; avoids dropped-packet UDP issues on most LAN cameras
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source!r}. "
                           "Check the file path, codec, or RTSP URL/credentials.")
    return cap


def _read_source_meta(cap: cv2.VideoCapture) -> tuple[float, int]:
    """Return (source_fps, total_frame_count). Count is 0 for live RTSP."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return fps, total


def _frame_to_pil(bgr_frame: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR frame to PIL RGB image for Moondream."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _run_inference(model, pil_image: Image.Image, question: str, timeout_s: float) -> InferenceRecord:
    """Call Moondream query with timing; classify errors by type."""
    frame_index = -1  # set by caller
    t0 = time.perf_counter()
    try:
        result = model.query(pil_image, question)
        latency_ms = (time.perf_counter() - t0) * 1000
        answer = result.get("answer", "") if isinstance(result, dict) else str(result)
        return InferenceRecord(
            frame_index=frame_index,
            latency_ms=latency_ms,
            answer=answer,
            success=True,
        )
    except TimeoutError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return InferenceRecord(
            frame_index=frame_index,
            latency_ms=latency_ms,
            answer="",
            success=False,
            error=f"timeout:{exc}",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return InferenceRecord(
            frame_index=frame_index,
            latency_ms=latency_ms,
            answer="",
            success=False,
            error=f"error:{exc}",
        )


def run(cfg: RunConfig) -> dict:
    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    print(f"[{run_id[:8]}] Opening source: {cfg.source}", file=sys.stderr)
    try:
        cap = _open_capture(cfg.source)
    except RuntimeError as exc:
        return _error_report(run_id, started_at, cfg, "ingest_open_failed", str(exc))

    source_fps, total_frames_in_source = _read_source_meta(cap)
    device_info = _detect_device()

    print(f"[{run_id[:8]}] Source FPS={source_fps:.1f}  total_frames={total_frames_in_source or 'live'}  "
          f"device={device_info['device']}", file=sys.stderr)

    model = md.vl(api_key=cfg.api_key)

    ingest = IngestStats(
        source_fps=source_fps,
        total_frames_in_source=total_frames_in_source,
    )
    failures = FailureSummary()
    records: list[InferenceRecord] = []

    frame_idx = 0
    ingest_wall_start = time.perf_counter()

    while True:
        if cfg.max_frames and frame_idx >= cfg.max_frames:
            print(f"[{run_id[:8]}] Reached max_frames={cfg.max_frames}, stopping.", file=sys.stderr)
            break

        ret, raw_frame = cap.read()
        if not ret:
            if frame_idx == 0:
                failures.decode_errors += 1
                print(f"[{run_id[:8]}] DECODE ERROR on first frame — bad file or stream gone.", file=sys.stderr)
                break
            # Normal end-of-file for MP4
            break

        frame_idx += 1
        ingest.frames_read = frame_idx

        if frame_idx % cfg.sample_every_n != 0:
            continue

        try:
            pil_img = _frame_to_pil(raw_frame)
        except Exception as exc:
            failures.decode_errors += 1
            print(f"[{run_id[:8]}] Frame {frame_idx}: color conversion failed: {exc}", file=sys.stderr)
            continue

        print(f"[{run_id[:8]}] Frame {frame_idx}: running inference...", file=sys.stderr)
        rec = _run_inference(model, pil_img, cfg.question, cfg.infer_timeout_s)
        rec.frame_index = frame_idx

        if rec.success:
            print(f"[{run_id[:8]}] Frame {frame_idx}: {rec.latency_ms:.0f}ms → {rec.answer!r}", file=sys.stderr)
        else:
            err_lower = rec.error or ""
            if err_lower.startswith("timeout"):
                failures.inference_timeouts += 1
            else:
                failures.inference_errors += 1
            print(f"[{run_id[:8]}] Frame {frame_idx}: FAILED — {rec.error}", file=sys.stderr)

        records.append(rec)

    cap.release()
    ingest_wall_elapsed = time.perf_counter() - ingest_wall_start

    ingest.decode_errors = failures.decode_errors

    latencies = [r.latency_ms for r in records if r.success]
    latency_stats: dict = {}
    if latencies:
        latency_stats = {
            "p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "min_ms": round(float(np.min(latencies)), 2),
            "max_ms": round(float(np.max(latencies)), 2),
            "mean_ms": round(float(np.mean(latencies)), 2),
            "samples": len(latencies),
        }
    else:
        latency_stats = {"note": "No successful inference samples — cannot compute latency distribution."}

    actual_ingest_fps = ingest.frames_read / ingest_wall_elapsed if ingest_wall_elapsed > 0 else 0.0
    infer_throughput_fps = len(records) / ingest_wall_elapsed if ingest_wall_elapsed > 0 else 0.0

    # Dropped-frame calculation: how many sampled frames were skipped due to inference failures
    sampled_count = ingest.frames_read // cfg.sample_every_n
    failed_count = failures.inference_timeouts + failures.inference_errors + failures.decode_errors
    drop_pct = round((failed_count / sampled_count * 100) if sampled_count > 0 else 0.0, 2)

    verdict, verdict_notes = _compute_verdict(latency_stats, failures, drop_pct)

    docker_cmd = (
        f"docker run --rm \\\n"
        f"  -e MOONDREAM_API_KEY=$MOONDREAM_API_KEY \\\n"
        f"  -v \"$(pwd)/your_video.mp4:/data/input.mp4\" \\\n"
        f"  edge-vision-poc \\\n"
        f"  --input /data/input.mp4 \\\n"
        f"  --question \"{cfg.question}\" \\\n"
        f"  --sample-every-n {cfg.sample_every_n}"
    )

    report = {
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "source": cfg.source,
            "type": "rtsp" if cfg.source.lower().startswith("rtsp://") else "file",
            "question": cfg.question,
        },
        "sampling": {
            "policy": "every_nth_frame",
            "n": cfg.sample_every_n,
            "frames_read_total": ingest.frames_read,
            "frames_sampled": len(records),
            "source_fps_reported": round(source_fps, 2),
            "actual_ingest_fps": round(actual_ingest_fps, 2),
        },
        "inference": {
            **device_info,
            "model": "moondream-hosted-api",
            "latency": latency_stats,
            "throughput_fps": round(infer_throughput_fps, 3),
        },
        "failure_modes": {
            "decode_errors": failures.decode_errors,
            "inference_timeouts": failures.inference_timeouts,
            "inference_errors": failures.inference_errors,
            "total_failures": failures.total,
            "effective_drop_pct": drop_pct,
        },
        "deployment_readiness": {
            "verdict": verdict,
            "notes": verdict_notes,
        },
        "repro": {
            "docker_run": docker_cmd,
        },
    }

    return report


def _compute_verdict(latency_stats: dict, failures: FailureSummary, drop_pct: float) -> tuple[str, list[str]]:
    notes = []

    if failures.total > 0 and "samples" not in latency_stats:
        notes.append("All inference attempts failed — no latency data available.")
        return "FAIL", notes

    if failures.decode_errors > 0:
        notes.append(f"{failures.decode_errors} decode error(s) — check codec/container or stream health.")
    if failures.inference_timeouts > 0:
        notes.append(f"{failures.inference_timeouts} inference timeout(s) — network or API latency issue.")
    if failures.inference_errors > 0:
        notes.append(f"{failures.inference_errors} inference error(s) — check API key, rate limits, or payload size.")
    if drop_pct > 10:
        notes.append(f"Effective drop rate {drop_pct}% exceeds 10% threshold.")

    p95 = latency_stats.get("p95_ms")
    if p95 and p95 > 5000:
        notes.append(f"p95 latency {p95}ms is high for real-time use cases (>5000ms).")
    elif p95 and p95 > 2000:
        notes.append(f"p95 latency {p95}ms is acceptable for batch/async use but not real-time control.")

    if "note" in latency_stats:
        notes.append(latency_stats["note"])

    if not notes:
        notes.append("All checks passed on this hardware and sample. "
                     "Validate on target edge device (e.g. Jetson) for production SLA sign-off.")

    if failures.total == 0 and drop_pct == 0:
        verdict = "PASS"
    elif drop_pct > 10 or (failures.inference_timeouts + failures.inference_errors) > 0:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return verdict, notes


def _error_report(run_id: str, started_at: str, cfg: RunConfig, error_code: str, message: str) -> dict:
    return {
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "input": {"source": cfg.source},
        "failure_modes": {"error_code": error_code, "message": message},
        "deployment_readiness": {"verdict": "FAIL", "notes": [message]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Edge Vision POC Lab — Deployment Readiness Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MP4 input, every 10th frame
  python analyze.py --input video.mp4 --api-key sk-...

  # RTSP stream, cap at 300 frames
  python analyze.py --input rtsp://user:pass@192.168.1.10:554/stream --max-frames 300

  # Custom question and output file
  python analyze.py --input video.mp4 --question "Is there a vehicle in this image?" --output report.json
        """,
    )
    parser.add_argument("--input", required=True, help="MP4 file path or rtsp:// URL")
    parser.add_argument(
        "--question",
        default="Is there a person in this image?",
        help="Natural language question sent to Moondream per sampled frame",
    )
    parser.add_argument(
        "--sample-every-n",
        type=int,
        default=10,
        metavar="N",
        help="Run inference on every Nth frame (default: 10). "
             "At 30fps source this gives ~3 inference calls/sec.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Stop after reading N frames total (useful for live RTSP or quick smoke tests)",
    )
    parser.add_argument(
        "--infer-timeout",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Seconds before an inference call is considered timed out (default: 30)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MOONDREAM_API_KEY", ""),
        help="Moondream API key (or set MOONDREAM_API_KEY env var)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write JSON report to this file (default: stdout)",
    )

    args = parser.parse_args()

    if not args.api_key:
        print(
            "ERROR: Moondream API key required. Pass --api-key or set MOONDREAM_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = RunConfig(
        source=args.input,
        question=args.question,
        sample_every_n=args.sample_every_n,
        max_frames=args.max_frames,
        infer_timeout_s=args.infer_timeout,
        api_key=args.api_key,
        output=args.output,
    )

    report = run(cfg)
    output_json = json.dumps(report, indent=2)

    if cfg.output:
        with open(cfg.output, "w") as f:
            f.write(output_json)
        print(f"\nReport written to: {cfg.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
