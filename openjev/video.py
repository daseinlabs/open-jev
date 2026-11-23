"""Video input: load bytes from a path, URL or base64 and sample N frames uniformly.

Gemma 3 has no native video, so a video becomes N still frames fed as images. Frame i
of N is the frame on screen at the midpoint of the i-th of N equal time bins, so
timestamps are ``(i + 0.5) * duration / N``. Audio is ignored. Decoding uses PyAV.

    video = load_video("clip.mp4", n_frames=16)
    video.frames       # list of RGB PIL images, native resolution unless max_side is set
    video.timestamps   # seconds from the start of the video, one per frame
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import os
import urllib.request
from dataclasses import dataclass, field

DEFAULT_FRAMES = 16


class VideoError(ValueError):
    """Bad or unsupported video input. The message is safe to return to API clients."""


@dataclass(frozen=True)
class VideoLimits:
    max_bytes: int = 512 * 1024 * 1024
    max_duration_s: float = 30 * 60.0
    max_frames: int = 64
    url_timeout_s: float = 30.0


@dataclass
class SampledVideo:
    frames: list  # PIL.Image.Image, RGB
    timestamps: list[float]
    duration_s: float
    width: int
    height: int
    sha256: str
    n_bytes: int
    meta: dict = field(default_factory=dict)


def sample_timestamps(duration_s: float, n_frames: int) -> list[float]:
    """Midpoints of ``n_frames`` equal bins over ``[0, duration_s)``."""
    if n_frames < 1:
        raise VideoError("n_frames must be at least 1")
    if not duration_s > 0:
        raise VideoError("video has no duration")
    step = duration_s / n_frames
    return [(i + 0.5) * step for i in range(n_frames)]


# ------------------------------------------------------------------ sources
def read_path(path: str, limits: VideoLimits = VideoLimits()) -> bytes:
    if not os.path.isfile(path):
        raise VideoError(f"video file not found: {path}")
    size = os.path.getsize(path)
    if size > limits.max_bytes:
        raise VideoError(f"video is {size} bytes; the limit is {limits.max_bytes}")
    with open(path, "rb") as f:
        return f.read()


def read_url(url: str, limits: VideoLimits = VideoLimits()) -> bytes:
    if not url.startswith(("http://", "https://")):
        raise VideoError("video URL must be http(s)")
    req = urllib.request.Request(url, headers={"User-Agent": "openjev"})
    try:
        with urllib.request.urlopen(req, timeout=limits.url_timeout_s) as resp:
            data = resp.read(limits.max_bytes + 1)
    except OSError as exc:
        raise VideoError(f"could not fetch video URL: {exc}") from exc
    if len(data) > limits.max_bytes:
        raise VideoError(f"video at URL exceeds the {limits.max_bytes} byte limit")
    return data


def read_base64(data: str, limits: VideoLimits = VideoLimits()) -> bytes:
    if data.startswith("data:"):
        head, _, data = data.partition(",")
        if not head.endswith(";base64"):
            raise VideoError("data URI must be base64 encoded")
    if len(data) * 3 // 4 > limits.max_bytes:
        raise VideoError(f"base64 video exceeds the {limits.max_bytes} byte limit")
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise VideoError("video is not valid base64") from exc


def read_source(source: str, limits: VideoLimits = VideoLimits()) -> bytes:
    """Path, http(s) URL or ``data:`` URI. Raw base64 needs :func:`read_base64`."""
    if source.startswith(("http://", "https://")):
        return read_url(source, limits)
    if source.startswith("data:"):
        return read_base64(source, limits)
    return read_path(source, limits)


# ------------------------------------------------------------------ decoding
def decode_frames(
    data: bytes,
    n_frames: int = DEFAULT_FRAMES,
    limits: VideoLimits = VideoLimits(),
    max_side: int | None = None,
) -> SampledVideo:
    """Decode ``data`` and return the frames shown at the N bin midpoints.

    Each target time seeks to the nearest keyframe before it and decodes forward, so
    long videos are not decoded end to end. The chosen frame is the last one whose
    presentation time is at or before the target (the frame on screen at that time).
    """
    import av
    from PIL import Image

    if n_frames > limits.max_frames:
        raise VideoError(f"n_frames={n_frames} exceeds the limit of {limits.max_frames}")
    if not data:
        raise VideoError("video is empty")
    try:
        container = av.open(io.BytesIO(data), mode="r")
    except (av.FFmpegError, OSError) as exc:
        raise VideoError(f"could not open video: {exc}") from exc
    with container:
        if not container.streams.video:
            raise VideoError("file has no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = float(stream.time_base) if stream.time_base else None
        start = float(stream.start_time * stream.time_base) if stream.start_time is not None and tb else 0.0
        if stream.duration is not None and tb:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / 1_000_000
        elif stream.frames and stream.average_rate:
            duration = stream.frames / float(stream.average_rate)
        else:
            duration = _scan_duration(container, stream, start)
        if duration > limits.max_duration_s:
            raise VideoError(f"video is {duration:.1f}s long; the limit is {limits.max_duration_s:.0f}s")
        targets = sample_timestamps(duration, n_frames)

        frames, frame_times, cache = [], [], {}
        try:
            for t in targets:
                frame_time, img = _frame_at(container, stream, start + t, cache)
                frames.append(img)
                frame_times.append(frame_time - start)
        except av.FFmpegError as exc:
            raise VideoError(f"could not decode video: {exc}") from exc
        width, height = stream.codec_context.width, stream.codec_context.height
        meta = {"fps": float(stream.average_rate) if stream.average_rate else None,
                "codec": stream.codec_context.name, "frame_times": frame_times}

    out = []
    for img in frames:
        if max_side and max(img.size) > max_side:
            scale = max_side / max(img.size)
            img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BICUBIC)
        out.append(img)
    return SampledVideo(
        frames=out, timestamps=targets, duration_s=duration, width=width, height=height,
        sha256=hashlib.sha256(data).hexdigest(), n_bytes=len(data),
        meta=meta,
    )


def _scan_duration(container, stream, start: float) -> float:
    last = None
    for frame in container.decode(stream):
        if frame.time is not None:
            last = frame.time
    if last is None:
        raise VideoError("video has no decodable frames")
    container.seek(0)
    rate = float(stream.average_rate) if stream.average_rate else 30.0
    return last - start + 1.0 / rate


def _frame_at(container, stream, target: float, cache: dict):
    """Last frame with time <= target (first frame if the target precedes it)."""
    if stream.time_base:
        container.seek(max(0, int(target / stream.time_base)), stream=stream, backward=True, any_frame=False)
    else:
        container.seek(0)
    chosen = None
    for frame in container.decode(stream):
        t = frame.time if frame.time is not None else 0.0
        if chosen is not None and t > target + 1e-6:
            break
        chosen = (t, frame)
        if t > target + 1e-6:  # first decoded frame is already past the target
            break
    if chosen is None:
        raise VideoError("video has no decodable frames")
    t, frame = chosen
    if t not in cache:
        cache[t] = frame.to_image().convert("RGB")
    return t, cache[t]


def load_video(
    source: str | bytes,
    n_frames: int = DEFAULT_FRAMES,
    limits: VideoLimits = VideoLimits(),
    max_side: int | None = None,
) -> SampledVideo:
    data = source if isinstance(source, bytes) else read_source(source, limits)
    return decode_frames(data, n_frames=n_frames, limits=limits, max_side=max_side)
