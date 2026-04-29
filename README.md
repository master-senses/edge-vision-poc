# Edge Vision POC Lab

A CLI tool that ingests MP4 or RTSP video, runs [Moondream](https://moondream.ai) on sampled frames, and produces a **Deployment Readiness Report** — the artifact that answers a customer's real question: *"Will this work on our hardware, with our cameras, in our network?"*

## What it measures


| Field                              | Why it matters in a real deployment                        |
| ---------------------------------- | ---------------------------------------------------------- |
| `ingest.actual_ingest_fps`         | Is the decode pipeline keeping up with the source?         |
| `failure_modes.decode_errors`      | Corrupted frames, bad container, stream reconnect failures |
| `inference.latency.p50_ms`         | Typical user-visible latency                               |
| `inference.latency.p95_ms`         | Tail latency — what actually breaks SLAs                   |
| `inference.device` + `gpu_mem_`*   | Confirms hardware path; flags if GPU absent                |
| `failure_modes.inference_timeouts` | Network/API flakiness under load                           |
| `deployment_readiness.verdict`     | PASS / WARN / FAIL with explicit notes                     |
| `repro.docker_run`                 | One-command reproduction path for any operator             |


Inference is run on **sampled frames** (every Nth, configurable).   

Quick start (local Python)

```bash
pip install -r requirements.txt

export MOONDREAM_API_KEY=your_key_here

python analyze.py \
  --input path/to/video.mp4 \
  --question "Is there a person in this image?" \
  --sample-every-n 10 \
  --output report.json

cat report.json
```

## Quick start (Docker — reproducible path)

```bash
# Build once
docker build -t edge-vision-poc .

# Run against a local MP4
docker run --rm \
  -e MOONDREAM_API_KEY=$MOONDREAM_API_KEY \
  -v "$(pwd)/your_video.mp4:/data/input.mp4" \
  edge-vision-poc \
  --input /data/input.mp4 \
  --question "Is there a person in this image?" \
  --sample-every-n 10

# Run against an RTSP stream (cap at 300 frames for a smoke test)
docker run --rm \
  -e MOONDREAM_API_KEY=$MOONDREAM_API_KEY \
  --network host \
  edge-vision-poc \
  --input "rtsp://user:pass@192.168.1.10:554/stream" \
  --max-frames 300 \
  --question "Is there a vehicle in this image?"
```

## Sample output

```json
{
  "run_id": "3a7f1c2d-...",
  "started_at": "2026-04-28T03:14:00+00:00",
  "input": {
    "source": "video.mp4",
    "type": "file",
    "question": "Is there a person in this image?"
  },
  "sampling": {
    "policy": "every_nth_frame",
    "n": 10,
    "frames_read_total": 300,
    "frames_sampled": 30,
    "source_fps_reported": 30.0,
    "actual_ingest_fps": 28.7 // how fast we moved through the video in a run
  },
  "inference": {
    "device": "cpu",
    "gpu_mem_used_mb": null,
    "gpu_mem_total_mb": null,
    "note": "No GPU detected; latency reflects CPU + network path only. For Jetson/bare-metal benchmarks, rerun on target hardware.",
    "model": "moondream-hosted-api",
    "latency": {
      "p50_ms": 412.3,
      "p95_ms": 890.1,
      "min_ms": 310.5,
      "max_ms": 1104.2,
      "mean_ms": 445.7,
      "samples": 30
    },
    "throughput_fps": 0.84 // how fast am i hitting moondream per second
  },
  "failure_modes": {
    "decode_errors": 0,
    "inference_timeouts": 0,
    "inference_errors": 0,
    "total_failures": 0,
    "effective_drop_pct": 0.0
  },
  "deployment_readiness": {
    "verdict": "PASS",
    "notes": [
      "All checks passed on this hardware and sample. Validate on target edge device (e.g. Jetson) for production SLA sign-off."
    ]
  },
  "repro": {
    "docker_run": "docker run --rm \\\n  -e MOONDREAM_API_KEY=$MOONDREAM_API_KEY \\\n  -v \"$(pwd)/your_video.mp4:/data/input.mp4\" \\\n  edge-vision-poc \\\n  --input /data/input.mp4 \\\n  --question \"Is there a person in this image?\" \\\n  --sample-every-n 10"
  }
}
```

## CLI reference

```
--input           MP4 file path or rtsp:// URL
--question        Natural language question per sampled frame (default: "Is there a person in this image?")
--sample-every-n  Run inference on every Nth frame (default: 10 → ~3 infers/sec at 30fps source)
--max-frames      Stop after N frames total; required for live RTSP runs
--infer-timeout   Per-call cap on model.query (seconds; 0 = wait indefinitely)
--api-key         Moondream API key (or MOONDREAM_API_KEY env var)
--output          Write JSON report to file (default: stdout)
```

## Simulating failure modes without special hardware


| Failure class                | How to trigger locally                                   |
| ---------------------------- | -------------------------------------------------------- |
| Decode error (bad file)      | `truncate -s 50% video.mp4` then run against it          |
| Decode error (wrong format)  | Rename a `.txt` file as `.mp4` and feed it               |
| Inference timeout            | `--infer-timeout 0.001` (forces most calls to time out)  |
| Network / API instability    | Block outbound HTTPS briefly mid-run, or use a bad proxy |
| Stream disconnect (RTSP sim) | Point at a port that accepts TCP then closes             |
| Ingest backpressure          | `--sample-every-n 1` on a high-res source                |


## What "deployment ready" means

This report answers the **integration surface** (not **model accuracy)**:

- The inference backend is reachable and returns structured answers
- Ingest can decode the source format without errors
- Tail latency (p95) is measurable and consistent enough to reason about
- The full path is reproducible by another operator via `docker run`

**For production sign-off**, thresholds should be set jointly with the customer's SRE: typical targets are `drop_pct < 5%`, `p95 < 2000ms` for async pipelines, and `p95 < 150ms` for real-time control loops.

## Scope and honest limits

- **Hosted API only (Phase 1):** Latency numbers include network + TLS. On-device Jetson inference with local weights will be significantly faster.
- **No GPU on dev machine.**
- **RTSP support:** Uses OpenCV's `VideoCapture` with TCP transport.
- **No model accuracy eval.**

