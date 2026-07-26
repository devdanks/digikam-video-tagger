from pathlib import Path

from digikam_video_tagger.config import (
    DEFAULT_STAGING_DIR,
    LOCAL_APP_DATA,
    digikam_sidecar_reading_enabled,
    read_kconfig_boolean,
)


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


def test_read_kconfig_boolean_reads_digikams_actual_sidecar_key(tmp_path: Path) -> None:
    config = tmp_path / "digikamrc"
    config.write_text(
        "[Metadata Settings]\nUseXMPSidecar4Reading=true\n",
        encoding="utf-8",
    )

    assert (
        read_kconfig_boolean(config, "Metadata Settings", "UseXMPSidecar4Reading")
        is True
    )
    assert (
        read_kconfig_boolean(config, "Metadata Settings", "Use XMP Sidecar For Reading")
        is None
    )


def test_digikam_sidecar_reading_enabled_accepts_current_key(tmp_path: Path) -> None:
    config = tmp_path / "digikamrc"
    config.write_text(
        "[Metadata Settings]\nUse XMP Sidecar For Reading=true\n",
        encoding="utf-8",
    )

    assert digikam_sidecar_reading_enabled(config) is True


def test_default_staging_directory_is_managed_by_the_application() -> None:
    assert DEFAULT_STAGING_DIR == LOCAL_APP_DATA / "digikam-video-tagger" / "staging"
