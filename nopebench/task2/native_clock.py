"""Generated videos on the evaluator's 16 fps clock: a 24 fps video (121 frames: the hand-off frame and 120 generated) becomes 81 frames
at 16 fps by dropping every frame n with n mod 3 = 1, losslessly; a 16 fps video is taken as it is.

The evaluator reconstructs every second frame after the hand-off (8 fps), frame 3 j at 24 fps = frame 2 j at 16 fps, so those instants
are exact; the odd frames in between are 1/48 s late and only the tracker reads them. A video whose kept frames do not decode exactly
as their sources is refused. The new meta has the crop in whole pixels and the source's fps, frames and crop under "native".
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

NATIVE_FPS, EVALUATOR_FPS = 24, 16
NATIVE_FRAMES, EVALUATOR_FRAMES = 121, 81


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def kept(n):
    """Native frame n stays."""
    return n % 3 != 1


def whole_pixels(crop):
    """A box in whole pixels; None stays None."""
    return None if crop is None else [int(round(float(x))) for x in crop]


def encode(source, target):
    """The kept frames, in the source's own pixel format, lossless (the frames are chosen here, not by an ffmpeg filter)."""
    import cv2
    capture = cv2.VideoCapture(str(source))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    size = width * height * 3 // 2
    decode = subprocess.Popen([ffmpeg(), "-nostdin", "-loglevel", "error", "-i", str(source), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-"],
                              stdout=subprocess.PIPE)
    write = subprocess.Popen([ffmpeg(), "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}", "-r", str(EVALUATOR_FPS),
                              "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-qp", "0", "-pix_fmt", "yuv420p", "-threads", "4", "-movflags", "+faststart", str(target)],
                             stdin=subprocess.PIPE)
    n = 0
    while True:
        frame = decode.stdout.read(size)
        if len(frame) < size:
            break
        if kept(n):
            write.stdin.write(frame)
        n += 1
    write.stdin.close()
    codes = write.wait(), decode.wait()
    if any(codes):
        raise subprocess.CalledProcessError(max(codes), "ffmpeg")


def compare(source, target):
    """(frames of the source, frames of the target, largest difference of a kept frame), as OpenCV decodes them."""
    import cv2
    a, b = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(target))
    native, written, worst = 0, 0, 0
    while True:
        ok, frame = a.read()
        if not ok:
            break
        if kept(native):
            there, other = b.read()
            if not there:
                break
            written += 1
            worst = max(worst, int(cv2.absdiff(frame, other).max())) if frame.shape == other.shape else 255
        native += 1
    while b.read()[0]:
        written += 1
    a.release()
    b.release()
    return native, written, worst


def convert(video, out_dir):
    """One generated video (<route>.mp4 with <route>.meta.json beside it) onto the 16 fps clock in out_dir; its meta first, the video
    last. Raises ValueError when it must not be scored (another frame count or rate)."""
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(video.with_name(video.stem + ".meta.json").read_text())
    fps, frames = meta.get("fps"), meta.get("frames")
    target = out_dir / video.name
    written = dict(meta, crop=whole_pixels(meta.get("crop")))
    temporary = out_dir / "_tmp"
    temporary.mkdir(exist_ok=True)
    link = temporary / video.name
    link.unlink(missing_ok=True)
    if fps == EVALUATOR_FPS:
        if frames != EVALUATOR_FRAMES:
            raise ValueError(f"{video.name}: {frames} frames at {fps} fps")
        try:
            os.link(video, link)
        except OSError:
            shutil.copyfile(video, link)
    elif fps == NATIVE_FPS:
        if frames != NATIVE_FRAMES:
            raise ValueError(f"{video.name}: {frames} frames at {fps} fps, the evaluator's clock needs {NATIVE_FRAMES}")
        encode(video, link)
        native, count, worst = compare(video, link)
        if (native, count, worst) != (NATIVE_FRAMES, EVALUATOR_FRAMES, 0):
            link.unlink()
            raise ValueError(f"{video.name}: {native} native frames, {count} written, largest difference {worst}")
        written.update(fps=EVALUATOR_FPS, frames=EVALUATOR_FRAMES,
                       native=dict(fps=fps, frames=frames, crop=meta.get("crop"), dropped="n mod 3 = 1", largest_difference=worst))
    else:
        raise ValueError(f"{video.name}: {fps} fps")
    meta_tmp = temporary / (video.stem + ".meta.json")
    meta_tmp.write_text(json.dumps(written, indent=1))
    meta_tmp.replace(target.with_name(video.stem + ".meta.json"))
    link.replace(target)
    shutil.rmtree(temporary, ignore_errors=True)
    return target
