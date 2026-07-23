from pathlib import Path

from digikam_video_tagger.config import read_kconfig_boolean


def test_read_kconfig_boolean_is_section_scoped(tmp_path: Path) -> None:
    config = tmp_path / "digikamrc"
    config.write_text(
        "[Other]\nUse XMP Sidecar For Reading=false\n"
        "[Metadata Settings]\nUse XMP Sidecar For Reading=true\n",
        encoding="utf-8",
    )

    assert (
        read_kconfig_boolean(
            config,
            "Metadata Settings",
            "Use XMP Sidecar For Reading",
        )
        is True
    )
