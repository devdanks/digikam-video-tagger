# Project Context — digikam-video-tagger

> Long-term architectural memory. Update after significant structural changes.
> Reason for existence: any agent or contributor touching this repo reads this
> first to learn *what the system is*, *how it is built*, and *where the seams
> are* — without re-reading the source.

## Stack

- **Language / runtime:** Python 3.12 (src layout, Windows-first). Build backend: hatchling.
- **Package manager:** uv (locked). `uv sync --locked --dev` is the install contract.
- **Pinned libraries (`pyproject.toml`):**
  - `numpy>=2.0,<3` — face-feature vector math.
  - `opencv-python-headless>=4.10,<5` — YuNet/SFace/YOLOv11 DNN inference, frame decode.
  - `PyMySQL>=1.1,<2` — read-only digiKam MariaDB access.
  - `Send2Trash>=1.8,<2` — recoverable sidecar cleanup through the operating system's rubbish bin.
- **Dev:** `pytest>=8.3,<9` and `ruff>=0.12,<1`; CI runs lint and format checks.
- **External binaries (NOT pip — discovered at runtime):** FFmpeg/FFprobe, ExifTool, ONNX models (`face_detection_yunet_2023mar.onnx`, `face_recognition_sface_2021dec.onnx`, `yolo11n.onnx`, optional `yolo11x.onnx`, `coco.names`). CUDA is validated by default but frame sampling uses FFmpeg's software `select,scale` filter graph.
- **Downstream service:** digiKam + MariaDB. digiKam is the face-recognition authority, not this tool.

## Architecture

This is a **linear CLI metadata-bridge tool**, not a service or layered app. No web/API/ORM layer. Single entry point: `digikam-video-tagger` → `cli:main` → argparse with **six** subcommands. Three independent workflows share common primitives.

### The key inversion (read this first)

The *primary* workflow (`prepare`/`status`/`finalize`) deliberately does **not** run ML face recognition. It extracts proxy frames, hands them to digiKam's native People workflow, and reads back **confirmed** face regions from the digiKam DB. The ML pipeline (`models.py`, `pipeline.py`, `evidence.py`) powers only the *secondary*, experimental `tag` command. **digiKam is the recognition authority; this tool is a verified media-metadata bridge.**

### Shared primitives

| Module | Role |
| --- | --- |
| `process.py` | `run_command` — argv-form subprocess wrapper, `CREATE_NO_WINDOW`, raises `CommandError`. |
| `config.py` | Env-driven defaults (`DIGIKAM_VIDEO_TAGGER_*`), `ToolPaths`, and `DatabaseConfig`. |
| `ffmpeg.py` | `FFmpegSampler` — CPU `select,scale` frame extraction with optional CUDA prerequisite validation; `sampling_filter` builds CPU and standalone CUDA graphs. |
| `digikam_db.py` | `DigiKamCatalog` (confirmed faces) + `DigiKamFaceGallery` (SFace embeddings) — raw parameterized SQL, read-only. |
| `metadata.py` | `ExifToolMetadataWriter` — the **only** media-metadata write path; imports portable sidecar XMP absent from the media, merges tag lists, verifies supported media, restores timestamps, then calls `send2trash`. |
| `jobs.py` | `VideoFaceJob` manifest, `CompletedVideo` ledger, `prepare_job`, `discover_jobs`, `mark_job_completed`, manifest-owned `remove_generated_files`. |
| `evidence.py` | `EvidenceAccumulator` / `TagEvidence` — per-label hit + frame-ratio gating (`tag` only). |
| `models.py` | `YoloObjectTagger`, `FaceTagger`, `select_opencv_target` (OpenCL/CPU DNN). |
| `pipeline.py` | `VideoTaggingPipeline.analyze` — extract → detect per frame → evidence gate → write_tags (`tag` only). |
| `tags.py` | Central People and automatic-video tag vocabulary helpers. |

### Flow A — Two-pass People (`prepare` → digiKam → `finalize`/`status`)

```
cli.prepare → jobs.prepare_job → FFmpegSampler.extract_frames → manifest.json on disk
                                                         ↓ (human scans faces in digiKam UI)
cli.finalize → jobs.discover_jobs → DigiKamCatalog.confirmed_faces_for_frames (SQL)
            → ExifToolMetadataWriter.write_tags(<People paths>) → ExifTool read-back
            → jobs.mark_job_completed → (optional) job.remove_generated_files
```

`status` reuses `cli.finalize` with `apply=False, summary_only=True, status_mode=True` (set in `build_parser`). No separate handler.

### Flow B — Direct auto-tag (`tag`)

```
cli.tag → select_opencv_target → DigiKamFaceGallery.load (unless --no-people)
        → YoloObjectTagger + FaceTagger constructed
        → VideoTaggingPipeline.analyze(video, apply)
            → FFmpegSampler.extract_frames
            → per frame: YoloObjectTagger.detect + FaceTagger.detect → EvidenceAccumulator
            → tags = Auto Tags/Video/Objects/<label>, Auto Tags/Video/Contains Faces, People/<name>
            → ExifToolMetadataWriter.write_tags (only if apply and tags)
```

### Flow C — Existing sidecar migration (`embed`)

```
cli.embed → discover videos with adjacent <video>.xmp
          → preview READY, or under --apply:
          → ExifToolMetadataWriter.write_tags([]) → verify media → Recycle Bin
```

### Where logic lives

Business orchestration lives in `cli.py` — arg parsing, batch loops, JSON payload assembly, human output, and exit codes. Domain value objects are frozen dataclasses in their own modules. There is **no service/repository abstraction**: `cli.py` constructs concrete classes directly and `digikam_db.py` issues inline SQL.

## Conventions (Observed)

### Error handling

- `CommandError` (subclass of `RuntimeError`) is the only domain exception, raised by `run_command` on non-zero exit.
- No global handler. Each batch handler wraps per-item work in `try/except Exception`, increments `failures`, prints `[ERROR] …` to stderr, and **continues the batch**; returns exit code 1 if any item failed. Per-item errors do not abort the run.
- DB connections are short-lived `with … cursor()` blocks, `autocommit=True`, read/write/connect timeouts (3–10s), **never write**.
- Validation raises bare `ValueError`/`FileNotFoundError`/`RuntimeError` at boundaries (`extract_frames` arg checks, manifest version mismatches, unsafe proxy paths).

### Output shape (the "API")

- No HTTP API. Two output modes per command: human (`[INDEX/TOTAL STATE] path`) and `--json`.
- JSON: one object per video + a final `{"type": "summary", …}`. Keys are **snake_case**. `ensure_ascii=False`.
- stdout is reconfigured to line-buffering in `main()` for piped/PowerShell use.
- Exit codes: `0` success, `1` item failure(s), `2` no inputs found.

### Type system

- Strict-ish: `from __future__ import annotations`, explicit hints on all public surfaces, frozen dataclasses for value objects, tuples for immutable sequences. No `Any`. No `Optional` misuse.
- **No DIP / interfaces / ABCs.** Concrete classes are wired directly in `cli.py`. The codebase is typed but not abstracted — a deliberate Alpha-stage choice.

### Observability

- **Effectively none.** No `logging`, no metrics, no health endpoint. All output is `print()`.
- `doctor` is the de facto health check: FFmpeg build/CUDA, CUDA smoke test, OpenCV DNN target, YuNet/SFace and selected YOLO model inference, ExifTool runtime, and DB connection + `face_statistics()` (region/person/embedding counts).
- `--json` is the only machine-readable observability surface.

### Testing

- pytest tests live in `tests/test_<module>.py` with `addopts=-q`; there is no conftest/fixtures file.
- Strategy: pure unit tests with `tmp_path` + `monkeypatch`. External binaries and DB are mocked by `monkeypatch.setattr` on `cli` attributes (`discover_jobs`, `DigiKamCatalog`, `mark_job_completed`) or by exercising pure functions directly (`_relative_album`, `_tag_paths`, `sampling_filter`, `EvidenceAccumulator.accepted`).
- Assertions target stdout via `capsys`. **No GPU/DB integration tests by design.**
- CI (`.github/workflows/ci.yml`, windows-latest): `uv sync --locked --dev` → `compileall` → `ruff check`/`ruff format --check` → `pytest` → `uv build`. There is no type checker.

### Safety invariants (enforced in code)

- digiKam DB is query-only (no INSERT/UPDATE/DELETE anywhere in `digikam_db.py`).
- Only ExifTool-writable media types (`.3g2`, `.3gp`, `.m4v`, `.mov`, `.mp4`) are rewritten; original access and modification timestamps are restored exactly.
- Existing embedded and sidecar tags are merged case-insensitively; existing non-tag XMP takes precedence. All requested tag fields and imported sidecar XMP are read back from the media before `send2trash` recycles `<video>.xmp`.
- Any ExifTool, verification, timestamp-restoration, or Recycle Bin failure retains the sidecar and prevents completion-ledger/proxy cleanup.
- Cleanup is manifest-bound: `remove_generated_files` asserts each `resolved.parent == job_root` before unlink, then removes only manifest-named frames + their `.xmp` + the manifest.
- Rerun skip: `source_is_unchanged` compares `st_size` + `st_mtime_ns`; completion ledger keyed by `job_id` (sha256 of resolved path).

## Signals / Active Considerations

1. **`cli.py` remains an orchestration hotspot.** Parsing, batch loops, JSON payloads, and human output are co-located. A future `batch.py` extraction should preserve the `--json` contract.
2. **Sequential batch processing.** No threading/async exists; large `prepare`/`tag` trees process serially. This remains acceptable for Alpha.
3. **Album-root matching is intentionally conservative.** The catalog chooses the deepest matching absolute root, reports frames outside every root as `STAGING-NOT-CATALOGUED`, and does not attempt ambiguous remote-root reconstruction.
4. **Import-time side effect:** `__init__.py` sets `OPENCV_OCL4DNN_CONFIG_PATH` at import so OpenCV can use a persistent DNN cache.

## Verify

```bash
# Stack + build health
uv sync --locked --dev && uv run python -m compileall -q src tests && uv run ruff check . && uv run ruff format --check . && uv run pytest && uv build
uv run digikam-video-tagger --version

# Architectural invariants (grep-based structure checks)
grep -Lni "insert into\|update .*set\|delete from\|drop \|alter " src/digikam_video_tagger/digikam_db.py  # DB never writes (expect filename printed = no match)
grep -c "send2trash" src/digikam_video_tagger/metadata.py                              # verified sidecar recycling (expect 2)
grep -c "resolved.parent != job_root" src/digikam_video_tagger/jobs.py                 # manifest-bound cleanup guard (expect 1)
grep -cE "^def (doctor|prepare|finalize|embed|tag)\b" src/digikam_video_tagger/cli.py   # 5 handlers; status reuses finalize
grep -c "^## " specs/tech-architecture/tech-stack.md                                   # this doc has ≥5 sections
```
