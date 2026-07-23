# Contributing

Thanks for improving digiKam Video Tagger. Read [AGENTS.md](AGENTS.md) for repository structure,
style, tests, and data-safety rules.

## Development setup

```powershell
uv sync
uv run pytest
uv run python -m compileall -q src tests
uv build
```

Copy `config.example.ps1` to the ignored `config.local.ps1` only when running the real digiKam
workflow. Never commit local paths, database credentials, media, face models, proxy frames, or XMP
sidecars.

## Changes

Keep changes focused and add a regression test for bug fixes. Database access must remain read-only;
write People tags through adjacent XMP sidecars. Generated-file cleanup must be constrained by a
validated manifest.

Use Conventional Commit subjects such as:

```text
fix(ffmpeg): constrain portrait proxy dimensions
feat(cli): complete reviewed videos without people
```

Pull requests should explain the user-visible behavior, validation performed, and any digiKam or
filesystem safety implications.
