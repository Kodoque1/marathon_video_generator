"""
Marathon Reboot — Composite v2
================================
Minimal single-input ffmpeg encode.  Blender v2 produces all layers natively
(GP grid, targeting brackets, Line Art, sparklines, telemetry text) so there
is no HUD compositing step needed.

Usage:
    python composite_v2.py [--frames-dir DIR] [--output FILE]
                           [--fps 30] [--crf 18] [--preset slow]

    Or call compose() directly if integrating with a larger build script.
"""

import argparse
import os
import shutil
import subprocess
import sys


# ─────────────────────────────────────────────────────────────────────────────
# Default paths (mirrors config.yaml structure)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_FRAMES_DIR = "/tmp/blender_v2_full/"
DEFAULT_OUTPUT     = "marathon_v2.mp4"
DEFAULT_FPS        = 30
DEFAULT_CRF        = 18
DEFAULT_PRESET     = "slow"


# ─────────────────────────────────────────────────────────────────────────────

def find_ffmpeg():
    for name in ("ffmpeg", "ffmpeg5", "avconv"):
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError(
        "ffmpeg not found on PATH.  Install it: sudo apt install ffmpeg")


def compose(frames_dir: str = DEFAULT_FRAMES_DIR,
            output: str     = DEFAULT_OUTPUT,
            fps: int        = DEFAULT_FPS,
            crf: int        = DEFAULT_CRF,
            preset: str     = DEFAULT_PRESET) -> str:
    """
    Encode the PNG sequence in `frames_dir` to an MP4.
    Returns the path of the written file.

    ffmpeg input pattern: frame_%04d.png  (Blender default)
    Output: H.264, yuv420p, libx264, lossless-ish at crf=18.
    """
    frames_dir = os.path.abspath(frames_dir)
    output     = os.path.abspath(output)

    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"frames_dir not found: {frames_dir}")

    frame_pattern = os.path.join(frames_dir, "frame_%04d.png")

    ffmpeg = find_ffmpeg()

    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        # Flatten RGBA → opaque black background before encoding
        "-vf", "format=rgba,colorchannelmixer=aa=1,format=yuv420p",
        "-c:v", "libx264",
        "-crf",    str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        output,
    ]

    print(f"[composite_v2] Encoding: {frames_dir} → {output}")
    print("[composite_v2] cmd:", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode})")

    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"[composite_v2] Done → {output}  ({size_mb:.1f} MB)")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Blender render step  (optional — skip if you pre-rendered the frames)
# ─────────────────────────────────────────────────────────────────────────────

def render_blender(script: str,
                   frames_dir: str,
                   frames: int = 90,
                   width:  int = 1920,
                   height: int = 1080,
                   fps:    int = 30) -> None:
    """Call Blender headless to render the PNG sequence, then return."""
    blender = shutil.which("blender") or "/snap/bin/blender"
    if not os.path.exists(blender):
        raise FileNotFoundError(f"Blender not found at {blender}")

    os.makedirs(frames_dir, exist_ok=True)
    cmd = [
        blender, "--background", "--python", script,
        "--",
        "--output", frames_dir,
        "--frames", str(frames),
        "--width",  str(width),
        "--height", str(height),
        "--fps",    str(fps),
    ]
    print(f"[composite_v2] Blender render: {script}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"blender exited with code {result.returncode}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Marathon Reboot v2 — composite step")
    p.add_argument("--frames-dir", default=DEFAULT_FRAMES_DIR,
                   help="Directory containing frame_XXXX.png files")
    p.add_argument("--output",  default=DEFAULT_OUTPUT,
                   help="Output MP4 path")
    p.add_argument("--fps",     type=int, default=DEFAULT_FPS)
    p.add_argument("--crf",     type=int, default=DEFAULT_CRF,
                   help="H.264 CRF quality (0=lossless, 23=default, lower=better)")
    p.add_argument("--preset",  default=DEFAULT_PRESET,
                   help="libx264 preset (ultrafast … slow … veryslow)")
    p.add_argument("--render",  action="store_true",
                   help="Also run the Blender render step before encoding")
    p.add_argument("--blender-script",
                   default=os.path.join(os.path.dirname(__file__),
                                        "blender_scene_v2.py"),
                   help="Path to blender_scene_v2.py (only with --render)")
    p.add_argument("--frames",  type=int, default=90,
                   help="Number of frames (only with --render)")
    args = p.parse_args()

    if args.render:
        render_blender(
            script     = args.blender_script,
            frames_dir = args.frames_dir,
            frames     = args.frames,
            fps        = args.fps,
        )

    compose(
        frames_dir = args.frames_dir,
        output     = args.output,
        fps        = args.fps,
        crf        = args.crf,
        preset     = args.preset,
    )


if __name__ == "__main__":
    main()
