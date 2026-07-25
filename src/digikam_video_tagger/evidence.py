from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class TagEvidence:
    label: str
    hits: int
    frames: int
    max_confidence: float

    @property
    def frame_ratio(self) -> float:
        return self.hits / self.frames if self.frames else 0.0


class EvidenceAccumulator:
    def __init__(self) -> None:
        self.frames = 0
        self._hits: dict[str, int] = defaultdict(int)
        self._max_confidence: dict[str, float] = defaultdict(float)

    def add_frame(self, labels: dict[str, float]) -> None:
        self.frames += 1
        for label, confidence in labels.items():
            self._hits[label] += 1
            self._max_confidence[label] = max(self._max_confidence[label], confidence)

    def accepted(
        self,
        *,
        min_hits: int,
        min_frame_ratio: float,
        limit: int | None = None,
    ) -> list[TagEvidence]:
        required_hits = min(max(1, min_hits), self.frames)
        values = [
            TagEvidence(label, hits, self.frames, self._max_confidence[label])
            for label, hits in self._hits.items()
            if hits >= required_hits
            and (hits / self.frames if self.frames else 0.0) >= min_frame_ratio
        ]
        values.sort(
            key=lambda item: (-item.hits, -item.max_confidence, item.label.casefold())
        )
        return values if limit is None else values[:limit]
