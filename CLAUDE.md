# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`digikam-video-tagger` is a Windows-first Python 3.12 CLI that bridges video metadata into digiKam by writing adjacent XMP sidecars (`video.mp4.xmp`). It never rewrites video bytes and never writes to digiKam's database. digiKam is the face-recognition authority for confirmed tags; this tool is a safe sidecar-writing bridge.

Three independent workflows share common primitives:

- **Confirmed-People workflow (primary):** `prepare` → review faces in digiKam UI → `status` → `finalize --apply`. Extracts temporary JPEG proxy frames into a managed staging dir, lets digiKam do face detection/recognition, then reads back **confirmed** People tags and writes them to source-video sidecars. This flow does **not** run ML face recognition itself.
- **Automated People workflow:** `autofinalize` runs YuNet/SFace in-process, recognizes people from the read-only SFace gallery, clusters unknown faces into persistent `People/Unknown/Person_NNN` placeholders, and resolves those placeholders later when their centroid matches the gallery. It writes sidecars directly and does not need the staging Album Root, but still uses the staging directory for proxy frames, the cluster store, and the completion ledger.
- **Auto-tags workflow (secondary, experimental):** `tag` runs YOLOv11 + YuNet/SFace in-process and writes `Auto Tags/Video/...` sidecars. It does not create digiKam face regions.

## Commands

Run from repo root in PowerShell. `uv` is the package manager (`uv.lock` is the contract).

```powershell
uv sync                          # install locked runtime + dev deps
uv run pytest                    # full test suite
uv run pytest tests/test_jobs.py::test_finalize_can_complete_reviewed_job_without_people   # single test
uv run python -m compileall -q src tests          # syntax/import check
uv run ruff check .              # lint
uv run ruff format --check .     # format check (run `uv run ruff format .` to fix)
uv build                         # distribution artifacts
uv run digikam-video-tagger doctor    # validate FFmpeg/CUDA, OpenCV, ExifTool, models, sidecar setting, DB
uv run digikam-video-tagger status "G:\Videos"   # read-only batch readiness check
```

Exit codes: `0` success, `1` one-or-more item failures, `2` no inputs found.

## Local configuration

`config.local.ps1` (git-ignored) sets `DIGIKAM_VIDEO_TAGGER_*` env vars for the local digiKam DB, models, and digikamrc. **Dot-source it in every new PowerShell session** before running the tool: `. .\config.local.ps1`. See `config.example.ps1`.

The proxy staging location is fixed at `%LOCALAPPDATA%\digikam-video-tagger\staging`. It must be configured once in digiKam as an Album Root for `prepare`/`status`/`finalize`. The `autofinalize` workflow does not require the staging Album Root, but still uses the staging directory for proxy frames, the cluster store, and the completion ledger.

## Architecture (big picture)

`src/digikam_video_tagger/` — `cli.py` is the orchestration hotspot: argparse, batch loops, JSON payload assembly, human output, and exit codes. The `autofinalize` workflow introduces a focused service, `AutoFinalizeService`, in `autofinalize.py`; `cli.py` constructs it and formats its results. There is no broader service/repository layer; `cli.py` constructs concrete classes directly and `digikam_db.py` issues inline parameterized SQL.

- `cli.py` — six subcommands; `status` is **not a separate handler** — it reuses `finalize` with `apply=False, summary_only=True, status_mode=True` (set in `build_parser`). Changing `finalize` changes `status`.
- `jobs.py` — `VideoFaceJob` manifest on disk, `CompletedVideo` completion ledger (v2 with managed-placeholder ownership), `prepare_job`, `discover_jobs`, `mark_job_completed`, `rewrite_completed_entry`, manifest-owned cleanup, rerun-skip via `st_size`+`st_mtime_ns`, and `staging_apply_lock`.
- `ffmpeg.py` — `FFmpegSampler`; frame sampling uses FFmpeg's CPU `select,scale` filter graph; CUDA is validated as a prerequisite but not used for decoding.
- `models.py` / `pipeline.py` / `evidence.py` — power the `tag` command; `models.py` now also exposes gallery-independent `FaceTagger.detect_faces()` for `autofinalize`.
- `clustering.py` — `FaceClusterStore` / `ClusterSession`: validated persistent unknown-face clusters, provisional per-video assignment, accepted-only commits, SFace model fingerprint, and known-person resolution candidates.
- `autofinalize.py` — `AutoFinalizeService`: frame acquisition, per-frame evidence gating, cluster save before sidecar reference, completion, cleanup, and ledger-owned placeholder replacement.
- `digikam_db.py` — `DigiKamCatalog` (confirmed faces for `finalize`) + `DigiKamFaceGallery` (SFace embeddings for `tag` and `autofinalize`), read-only.
- `metadata.py` — `ExifToolSidecarWriter`, the **only** write path outside the job store; atomic tempfile + `os.replace`; merge-only tags plus exact managed-tag removal.

For the full module map, flow diagrams, conventions, and active considerations, read `specs/tech-architecture/tech-stack.md` (long-term architectural memory) and `AGENTS.md`.

## Safety invariants (do not violate)

- digiKam DB is query-only — never add INSERT/UPDATE/DELETE anywhere.
- Video bytes and timestamps are never modified; metadata goes to `<video>.xmp` only.
- Sidecar writes are atomic (tempfile + `os.replace`) and merge-only with dedup.
- Exact managed-tag removal (`ExifToolSidecarWriter.remove_tags`) is only allowed for tag paths proven tool-owned by the version-2 completion ledger.
- Proxy cleanup is manifest-bound: `remove_generated_files` asserts each `resolved.parent == job_root` before unlink. A changed source video is not finalized from stale proxies.
- `autofinalize` persists a cluster store keyed to the SFace model fingerprint; a changed model requires an explicit rebuild and is never silently reused.
- Only one `autofinalize --apply` operation may run against a staging directory at a time.
- Do not commit local paths, DB credentials, models, generated frames, or XMP sidecars.

## Style

Four-space indentation, PEP 8 naming, explicit type hints, `pathlib.Path` for filesystem ops. `from __future__ import annotations` is the norm. Frozen dataclasses for value objects; no `Any`, no interfaces/ABCs (deliberate Alpha-stage choice). Match existing style; keep imports grouped. Conventional Commit subjects (`fix(ffmpeg): ...`, `feat(cli): ...`).

## Testing

pytest in `tests/test_<module>.py`, `addopts=-q`, no conftest/fixtures. Pure unit tests with `tmp_path` + `monkeypatch`; mock external binaries and the digiKam catalog via `monkeypatch.setattr`. Assertions target stdout via `capsys`. **No GPU/DB integration tests by design** — the normal suite must pass without a workstation DB or GPU. Add a focused regression test for every bug fix; name tests descriptively (`test_<behavior>`).
