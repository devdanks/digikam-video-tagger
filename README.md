# digiKam Video Tagger

People tagging for videos using digiKam's existing face-recognition workflow, with optional GPU
backend validation for the supported workstation configuration.

The tool ships two workflows that share an FFmpeg frame sampler and an ExifTool sidecar writer,
and never write directly to digiKam's database or rewrite the video stream:

- **Two-pass People workflow** (`prepare` / `status` / `finalize`): extracts temporary JPEG proxy
  frames, lets digiKam's native face workflow detect and confirm People, then copies the confirmed
  People hierarchy to adjacent video XMP sidecars and removes the proxies.
- **Direct auto-tagging** (`tag`): runs YOLOv11 object detection and YuNet/SFace face recognition
  on extracted frames, accumulates per-label evidence, and writes `Auto Tags/Video` sidecars in one
  pass. It records object presence and face presence — it does **not** populate digiKam People
  regions.

> **Status:** Alpha. The supported workflow targets Windows and digiKam with MariaDB. CUDA and
> OpenCL are required by default, but can be independently relaxed for CPU-capable workstations.

## How it works

Two-pass People workflow:

```text
video tree -> sampled proxy frames -> digiKam face workflow -> video.ext.xmp -> proxy cleanup
```

Each video receives an isolated, manifest-backed proxy folder. Folder inputs recurse by default,
completed videos are tracked, and reruns skip unchanged work. Sidecars preserve existing metadata
while adding digiKam-compatible `TagsList`, `HierarchicalSubject`, and `Subject` values.

Direct auto-tagging (`tag`) skips the manual digiKam face-review workflow:

```text
video -> sampled frames -> YOLOv11 objects + YuNet/SFace faces -> evidence gating -> video.ext.xmp
```

A label is written only when it is seen in at least `--min-object-hits` (or `--min-person-hits`)
frames and in at least `--min-frame-ratio` of all sampled frames. Object tags land under
`Auto Tags/Video/Objects/<label>`, any frame with a face adds `Auto Tags/Video/Contains Faces`, and
recognized people add `People/<name>`.

## Requirements

- Windows with Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- digiKam with XMP sidecar reading enabled (`Use XMP Sidecar For Reading=true` in the
  `Metadata Settings` group); this is needed for adjacent video sidecars to override embedded
  metadata
- digiKam MariaDB reachable in read-only use for `finalize` and face-name recognition in `tag`
- FFmpeg and FFprobe; CUDA is validated by default and can be disabled with `--no-ffmpeg-cuda`
- OpenCV OpenCL is required by default for inference and can be disabled with `--no-opencl`
- ExifTool (the copy bundled with digiKam is suitable)
- digiKam face models (YuNet `face_detection_yunet_2023mar.onnx`, SFace
  `face_recognition_sface_2021dec.onnx`) and trained recognition data
- YOLOv11 ONNX weights (`yolo11n.onnx`, or `yolo11x.onnx` for `tag --object-model xl`) plus
  `coco.names` — required by `tag` and checked by `doctor`

## Install

```powershell
git clone https://github.com/devdanks/digikam-video-tagger.git
cd digikam-video-tagger
uv sync
```

FFmpeg is discovered from `PATH`. For the remaining local settings, copy and load the PowerShell
configuration template:

```powershell
Copy-Item config.example.ps1 config.local.ps1
# Edit config.local.ps1 for this workstation.
. .\config.local.ps1
uv run digikam-video-tagger doctor
```

`config.local.ps1` is ignored by Git. Every setting can also be supplied as a CLI option. Run
`uv run digikam-video-tagger --help` and the relevant subcommand's `--help` for details.

Use `doctor --object-model xl` before selecting the XL object model to validate `yolo11x.onnx`.
The legacy `--allow-cpu-fallback` permits CPU fallback for both backends; prefer the independent
`--no-ffmpeg-cuda` and `--no-opencl` controls when only one backend needs to be relaxed.

## Two-pass People workflow

Prepare an entire video tree:

```powershell
uv run digikam-video-tagger prepare "D:\Media\Videos"
```

In digiKam:

1. Scan for new items under `_digikam_video_faces`.
2. Detect and recognize faces on that album, including all subalbums.
3. Confirm or assign People names and click **Apply**.

The staging directory must be inside a registered digiKam Album Root. If `status` or `finalize`
reports `STAGING-NOT-CATALOGUED`, add the staging directory as a collection root and rescan before
continuing.

Inspect the batch without writing or deleting anything:

```powershell
uv run digikam-video-tagger status "D:\Media\Videos"
```

Apply confirmed People tags and remove only successfully processed proxies:

```powershell
uv run digikam-video-tagger finalize "D:\Media\Videos" --apply
```

After reviewing the whole proxy batch, complete scanned videos that contain no confirmed people and
remove their temporary frames:

```powershell
uv run digikam-video-tagger finalize "D:\Media\Videos" `
  --apply --complete-without-people --summary-only
```

Reread metadata for the source-video album in digiKam afterward. Use `--keep-frames` to retain
generated JPEGs, or `prepare --reprocess-completed` to deliberately scan completed videos again.

## Direct auto-tagging (`tag`)

Tag a video tree in a single pass without manual face review. Without `--apply` it is a dry run
that prints the tags it would write:

```powershell
uv run digikam-video-tagger tag "D:\Media\Videos"
```

Write the merged sidecars:

```powershell
uv run digikam-video-tagger tag "D:\Media\Videos" --apply
```

Skip detection stages or tune the evidence threshold:

```powershell
uv run digikam-video-tagger tag "D:\Media\Videos" --apply --no-people --min-object-hits 3
```

The `tag` command writes object-presence and face-presence tags. It does **not** create digiKam
People regions — use the two-pass workflow above for confirmed People assignments. Face
recognition reads digiKam's SFace training embeddings (`DigiKamFaceGallery`) and only emits
`People/<name>` for faces matching the trained gallery within `--person-distance`; use
`--no-people` to avoid the catalog lookup.

## Safety

- digiKam database connections are query-only.
- Video bytes and timestamps are not modified.
- Metadata is merged into `filename.ext.xmp` sidecars atomically.
- Existing embedded and sidecar tags are deduplicated, and all three digiKam-compatible tag fields
  (`TagsList`, `HierarchicalSubject`, and `Subject`) are verified before a sidecar is replaced.
- Cleanup removes only manifest-owned proxy files and their sidecars.
- Jobs stop if a source video changes after extraction.

## Development

```powershell
uv run pytest
uv run python -m compileall -q src tests
uv run ruff check .
uv run ruff format --check .
uv build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md). digiKam's relevant documentation
covers [metadata sidecars](https://docs.digikam.org/en/setup_application/metadata_settings.html) and
[database behavior](https://docs.digikam.org/en/getting_started/database_intro.html).
