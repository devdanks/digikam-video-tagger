from pathlib import Path

from digikam_video_tagger.digikam_db import DigiKamCatalog


def test_collection_path_maps_to_relative_album() -> None:
    frame = Path(r"G:\Pictures\_digikam_video_faces\job\frame.jpg")
    root = Path(r"G:\Pictures")

    assert DigiKamCatalog._relative_album(frame, root) == "/_digikam_video_faces/job"


def test_tag_hierarchy_is_reconstructed() -> None:
    paths = DigiKamCatalog._tag_paths(
        [
            (1, 0, "People"),
            (2, 1, "Family"),
            (3, 2, "Shelby"),
        ]
    )

    assert paths[3] == "People/Family/Shelby"
