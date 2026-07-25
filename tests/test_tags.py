import pytest

from digikam_video_tagger.tags import object_tag, people_tag


def test_people_tag_rejects_an_empty_or_hierarchical_name() -> None:
    with pytest.raises(ValueError, match="name"):
        people_tag(" / ")
    with pytest.raises(ValueError, match="name"):
        people_tag("A/B")


def test_object_tag_rejects_a_hierarchical_label() -> None:
    with pytest.raises(ValueError, match="label"):
        object_tag("Auto Tags/Video", "A/B")
