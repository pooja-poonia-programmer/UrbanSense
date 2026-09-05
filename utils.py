from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math, shutil, subprocess, uuid

@dataclass(frozen=True)
class LatLon:
    lat: float
    lon: float
    def valid(self):
        try:
            return -90 <= float(self.lat) <= 90 and -180 <= float(self.lon) <= 180
        except Exception:
            return False

def now_id(prefix="EVT"):
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"

def resize_keep_aspect(frame, max_dim):
    import cv2
    h, w = frame.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return frame
    scale = max_dim / float(m)
    return cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2, (y1+y2)/2)

def distance(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def format_time(sec):
    sec = max(0, int(sec or 0))
    return f"{sec//60:02d}:{sec%60:02d}"

def interpolate(start: LatLon, end: LatLon, t: float):
    if not start or not end or not start.valid() or not end.valid():
        return None
    t = max(0.0, min(1.0, float(t)))
    return LatLon(
        start.lat + (end.lat-start.lat)*t,
        start.lon + (end.lon-start.lon)*t,
    )

def find_ffmpeg():
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

def convert_video(src, dst):
    ff = find_ffmpeg()
    if not ff:
        return False, "FFmpeg executable unavailable."
    cmd = [
        ff, "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
        str(dst)
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if p.returncode == 0 and Path(dst).exists() and Path(dst).stat().st_size > 10000:
            return True, "H.264 browser-compatible output created."
        return False, (p.stderr or "FFmpeg failed")[-1000:]
    except Exception as e:
        return False, str(e)
