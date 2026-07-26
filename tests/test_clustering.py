from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from digikam_video_tagger.clustering import FaceClusterStore
from digikam_video_tagger.digikam_db import PersonEmbedding


def unit(index: int) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[index] = 1.0
    return vector


def make_store() -> FaceClusterStore:
    return FaceClusterStore.empty(
        model_fingerprint="a" * 64,
        distance_threshold=0.20,
        unknown_root="People/Unknown",
    )


def test_rejected_session_cluster_does_not_consume_id() -> None:
    store = make_store()
    session = store.begin_session()
    token = session.assign(unit(0))
    assert token.startswith("session:")
    assert session.commit(set()) == {}
    assert store.next_id == 1
    assert store.clusters == {}


def test_accepted_session_cluster_becomes_persistent() -> None:
    store = make_store()
    session = store.begin_session()
    first = session.assign(unit(0))
    second = session.assign(unit(0))
    tags = session.commit({first, second})
    assert set(tags.values()) == {"People/Unknown/Person_001"}
    assert store.next_id == 2
    assert store.clusters["Person_001"].count == 2
    assert np.linalg.norm(store.clusters["Person_001"].centroid) == pytest.approx(1.0)


def test_store_rejects_changed_model_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    store = make_store()
    store.save(path)
    with pytest.raises(ValueError, match="SFace model"):
        FaceClusterStore.load(
            path,
            model_fingerprint="b" * 64,
            distance_threshold=0.20,
            unknown_root="People/Unknown",
        )


def test_persistent_match_returns_existing_id() -> None:
    store = make_store()
    session = store.begin_session()
    token = session.assign(unit(0))
    session.commit({token})

    session2 = store.begin_session()
    token2 = session2.assign(unit(0))
    assert token2 == "persistent:Person_001"
    tags = session2.commit({token2})
    assert tags[token2] == "People/Unknown/Person_001"
    assert store.clusters["Person_001"].count == 2


def test_distant_identities_remain_separate() -> None:
    store = make_store()
    session = store.begin_session()
    t1 = session.assign(unit(0))
    t2 = session.assign(unit(1))
    tags = session.commit({t1, t2})
    assert set(tags.values()) == {
        "People/Unknown/Person_001",
        "People/Unknown/Person_002",
    }


def test_same_session_matching_groups_observations() -> None:
    store = make_store()
    session = store.begin_session()
    t1 = session.assign(unit(0))
    t2 = session.assign(unit(0))
    assert t1 == t2
    session.commit({t1})
    assert store.clusters["Person_001"].count == 2


def test_rejected_observation_does_not_update_persistent_centroid() -> None:
    store = make_store()
    session = store.begin_session()
    token = session.assign(unit(0))
    session.commit({token})
    original = store.clusters["Person_001"].centroid.copy()

    session2 = store.begin_session()
    session2.assign(unit(1))
    session2.commit(set())
    assert np.allclose(store.clusters["Person_001"].centroid, original)


def test_round_trip_persistence(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    store = make_store()
    session = store.begin_session()
    token = session.assign(unit(0))
    session.commit({token})
    store.save(path)

    loaded = FaceClusterStore.load(
        path,
        model_fingerprint="a" * 64,
        distance_threshold=0.20,
        unknown_root="People/Unknown",
    )
    assert loaded.next_id == store.next_id
    assert loaded.clusters["Person_001"].count == 1
    assert np.allclose(loaded.clusters["Person_001"].centroid, unit(0))


def test_atomic_save_cleans_up_temp_file_on_replace_failure(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    store = make_store()
    with (
        patch(
            "digikam_video_tagger.clustering.os.replace", side_effect=OSError("boom")
        ),
        pytest.raises(OSError, match="boom"),
    ):
        store.save(path)
    assert not path.exists()
    temps = list(tmp_path.glob("*.tmp*"))
    assert temps == []


def test_load_rejects_malformed_schema(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        FaceClusterStore.load(
            path,
            model_fingerprint="a" * 64,
            distance_threshold=0.20,
            unknown_root="People/Unknown",
        )


def test_load_rejects_non_finite_centroid(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    bad = unit(0)
    bad[0] = float("nan")
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "store_id": "x",
                "model_fingerprint": "a" * 64,
                "embedding_dimension": 128,
                "distance_threshold": 0.20,
                "unknown_root": "People/Unknown",
                "next_id": 2,
                "clusters": {"Person_001": {"count": 1, "centroid": bad.tolist()}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite"):
        FaceClusterStore.load(
            path,
            model_fingerprint="a" * 64,
            distance_threshold=0.20,
            unknown_root="People/Unknown",
        )


def test_load_rejects_wrong_dimension(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "store_id": "x",
                "model_fingerprint": "a" * 64,
                "embedding_dimension": 129,
                "distance_threshold": 0.20,
                "unknown_root": "People/Unknown",
                "next_id": 2,
                "clusters": {"Person_001": {"count": 1, "centroid": [0.0] * 129}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dimension"):
        FaceClusterStore.load(
            path,
            model_fingerprint="a" * 64,
            distance_threshold=0.20,
            unknown_root="People/Unknown",
        )


def test_load_rejects_invalid_cluster_id(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "store_id": "x",
                "model_fingerprint": "a" * 64,
                "embedding_dimension": 128,
                "distance_threshold": 0.20,
                "unknown_root": "People/Unknown",
                "next_id": 2,
                "clusters": {"Person_1": {"count": 1, "centroid": unit(0).tolist()}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cluster id"):
        FaceClusterStore.load(
            path,
            model_fingerprint="a" * 64,
            distance_threshold=0.20,
            unknown_root="People/Unknown",
        )


def test_load_rejects_invalid_next_id(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "store_id": "x",
                "model_fingerprint": "a" * 64,
                "embedding_dimension": 128,
                "distance_threshold": 0.20,
                "unknown_root": "People/Unknown",
                "next_id": 1,
                "clusters": {"Person_001": {"count": 1, "centroid": unit(0).tolist()}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="next_id"):
        FaceClusterStore.load(
            path,
            model_fingerprint="a" * 64,
            distance_threshold=0.20,
            unknown_root="People/Unknown",
        )


def test_load_rejects_non_normalized_centroid(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    vector = unit(0) * 2.0
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "store_id": "x",
                "model_fingerprint": "a" * 64,
                "embedding_dimension": 128,
                "distance_threshold": 0.20,
                "unknown_root": "People/Unknown",
                "next_id": 2,
                "clusters": {"Person_001": {"count": 1, "centroid": vector.tolist()}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="normalize"):
        FaceClusterStore.load(
            path,
            model_fingerprint="a" * 64,
            distance_threshold=0.20,
            unknown_root="People/Unknown",
        )


def test_resolution_requires_best_match_margin() -> None:
    store = make_store()
    session = store.begin_session()
    session.assign(unit(0))
    session.commit({"session:1"})

    mom = PersonEmbedding(1, "Mom", unit(0))

    # Dad is close enough to pass the recognition gate but not by a large margin,
    # so the cluster should not be considered resolved with a strict margin.
    dad_vector = np.zeros(128, dtype=np.float32)
    dad_vector[0] = 0.6
    dad_vector[1] = 0.8
    dad = PersonEmbedding(2, "Dad", dad_vector)

    # With margin too large, no candidate.
    assert (
        store.resolution_candidates(
            [mom, dad], recognition_distance=0.50, min_observations=1, min_margin=0.50
        )
        == {}
    )

    # With small margin, Mom wins.
    candidates = store.resolution_candidates(
        [mom, dad], recognition_distance=0.50, min_observations=1, min_margin=0.001
    )
    assert candidates == {"People/Unknown/Person_001": "Mom"}


def test_resolution_skips_resolved_clusters() -> None:
    store = make_store()
    session = store.begin_session()
    session.assign(unit(0))
    session.commit({"session:1"})
    store.clusters["Person_001"].resolved_name = "Mom"

    close = unit(0) * 0.99 + unit(1) * 0.01
    close /= np.linalg.norm(close)
    mom = PersonEmbedding(1, "Mom", close)
    assert (
        store.resolution_candidates(
            [mom], recognition_distance=0.50, min_observations=1, min_margin=0.001
        )
        == {}
    )


def test_resolution_requires_min_observations() -> None:
    store = make_store()
    session = store.begin_session()
    session.assign(unit(0))
    session.commit({"session:1"})

    close = unit(0) * 0.99 + unit(1) * 0.01
    close /= np.linalg.norm(close)
    mom = PersonEmbedding(1, "Mom", close)
    assert (
        store.resolution_candidates(
            [mom], recognition_distance=0.50, min_observations=2, min_margin=0.001
        )
        == {}
    )
