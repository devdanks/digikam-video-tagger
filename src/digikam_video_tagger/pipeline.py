from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from .evidence import EvidenceAccumulator, TagEvidence
from .ffmpeg import FFmpegSampler, VideoInfo
from .metadata import ExifToolSidecarWriter, MetadataWriteResult
from .models import FaceTagger, YoloObjectTagger
from .tags import contains_faces_tag, object_tag, people_tag


@dataclass(frozen=True)
class AnalysisResult:
    video: Path
    info: VideoInfo
    frame_count: int
    unreadable_frames: int
    face_frames: int
    objects: tuple[TagEvidence, ...]
    people: tuple[TagEvidence, ...]
    tags: tuple[str, ...]
    metadata: MetadataWriteResult | None


class VideoTaggingPipeline:
    def __init__(
        self,
        sampler: FFmpegSampler,
        object_tagger: YoloObjectTagger | None,
        face_tagger: FaceTagger | None,
        sidecar_writer: ExifToolSidecarWriter,
        *,
        tag_root: str = "Auto Tags/Video",
        sample_seconds: float = 5.0,
        max_frames: int = 120,
        max_dimension: int = 1280,
        min_object_hits: int = 2,
        min_person_hits: int = 2,
        min_frame_ratio: float = 0.05,
        max_object_tags: int = 20,
    ) -> None:
        self.sampler = sampler
        self.object_tagger = object_tagger
        self.face_tagger = face_tagger
        self.sidecar_writer = sidecar_writer
        self.tag_root = tag_root.strip("/")
        self.sample_seconds = sample_seconds
        self.max_frames = max_frames
        self.max_dimension = max_dimension
        self.min_object_hits = min_object_hits
        self.min_person_hits = min_person_hits
        self.min_frame_ratio = min_frame_ratio
        self.max_object_tags = max_object_tags

    def analyze(self, video: Path, *, apply: bool = False) -> AnalysisResult:
        info, frames, temp_dir = self.sampler.extract_frames(
            video,
            sample_seconds=self.sample_seconds,
            max_frames=self.max_frames,
            max_dimension=self.max_dimension,
        )
        object_evidence = EvidenceAccumulator()
        person_evidence = EvidenceAccumulator()
        face_frames = 0
        unreadable_frames = 0
        try:
            for frame_path in frames:
                image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if image is None:
                    unreadable_frames += 1
                    if self.object_tagger is not None:
                        object_evidence.add_frame({})
                    if self.face_tagger is not None:
                        person_evidence.add_frame({})
                    continue
                if self.object_tagger is not None:
                    object_evidence.add_frame(self.object_tagger.detect(image))
                if self.face_tagger is not None:
                    face_count, people = self.face_tagger.detect(image)
                    if face_count:
                        face_frames += 1
                    person_evidence.add_frame(people)
        finally:
            temp_dir.cleanup()

        objects = object_evidence.accepted(
            min_hits=self.min_object_hits,
            min_frame_ratio=self.min_frame_ratio,
            limit=self.max_object_tags,
        )
        people = person_evidence.accepted(
            min_hits=self.min_person_hits,
            min_frame_ratio=self.min_frame_ratio,
        )
        tags = [object_tag(self.tag_root, item.label) for item in objects]
        if face_frames:
            tags.append(contains_faces_tag(self.tag_root))
        tags.extend(people_tag(item.label) for item in people)

        metadata = (
            self.sidecar_writer.write_tags(video, tags) if apply and tags else None
        )
        return AnalysisResult(
            video=video,
            info=info,
            frame_count=len(frames),
            unreadable_frames=unreadable_frames,
            face_frames=face_frames,
            objects=tuple(objects),
            people=tuple(people),
            tags=tuple(tags),
            metadata=metadata,
        )
