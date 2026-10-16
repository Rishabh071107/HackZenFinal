"""OpenCV-based frame extraction and annotated-video writing helpers."""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def get_video_metadata(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


def iter_frames(path: Path, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yields (frame_number, frame) for every `stride`-th frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    frame_number = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_number % stride == 0:
            yield frame_number, frame
        frame_number += 1
    cap.release()


def save_thumbnail(path: Path, output_path: Path, frame_index: int = 0) -> Path:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if ok:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
    return output_path


def _resolve_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _is_h264_mp4(path: Path) -> bool:
    """Quick sniff for H.264 tracks — HTML5 video needs avc1/avcC, not OpenCV mp4v."""
    try:
        head = path.read_bytes()[:16384]
    except OSError:
        return False
    return b"avc1" in head or b"avcC" in head


def ensure_browser_compatible_mp4(path: Path) -> Path:
    """
    Re-encode OpenCV mp4v output to H.264 so browsers can play it inline.
    No-op when the file is already H.264 or ffmpeg is unavailable.
    """
    path = Path(path)
    if not path.is_file() or _is_h264_mp4(path):
        return path

    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        logger.warning(
            "ffmpeg not found; annotated video may not play in browsers: %s", path
        )
        return path

    tmp = path.with_suffix(".web.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-preset",
        "fast",
        "-crf",
        "23",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(path)
    except subprocess.CalledProcessError as exc:
        logger.error(
            "ffmpeg transcode failed for %s: %s",
            path,
            exc.stderr.decode(errors="replace"),
        )
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path


class AnnotatedVideoWriter:
    """Draws bounding boxes, tracking IDs and team colors onto each frame."""

    def __init__(self, output_path: Path, width: int, height: int, fps: float):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path), fourcc, fps, (width, height)
        )

    def write_frame(self, frame: np.ndarray, boxes: list[dict]):
        """boxes: [{x1,y1,x2,y2,track_id,color_bgr,label,is_ball}, ...]"""
        annotated = frame.copy()
        for b in boxes:
            p1, p2 = (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"]))
            color = b.get("color_bgr", (0, 255, 0))
            is_ball = b.get("is_ball", False)

            if is_ball:
                center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                radius = max(6, max(p2[0] - p1[0], p2[1] - p1[1]) // 2)
                cv2.circle(annotated, center, radius, (0, 255, 255), 2)
                cv2.circle(annotated, center, 2, (0, 0, 255), -1)
                label = "BALL"
            else:
                cv2.rectangle(annotated, p1, p2, color, 2)
                label = b.get("label", str(b.get("track_id", "")))

            # Dark background fill badge behind label for crisp contrast
            (w_lbl, h_lbl), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2
            )
            bg_p1 = (p1[0], max(0, p1[1] - h_lbl - 6))
            bg_p2 = (p1[0] + w_lbl + 6, max(0, p1[1]))
            cv2.rectangle(annotated, bg_p1, bg_p2, (0, 0, 0), -1)

            cv2.putText(
                annotated,
                label,
                (p1[0] + 3, max(0, p1[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color if not is_ball else (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        self.writer.write(annotated)

    def release(self):
        self.writer.release()
        ensure_browser_compatible_mp4(self.output_path)
