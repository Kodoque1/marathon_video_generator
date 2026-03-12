#!/usr/bin/env python3
"""
composite.py — Hybrid Blender + pycairo compositor
=====================================================

Pipeline:
  1. Run blender --background --python blender_scene.py
     → Renders N_LOOP frames (e.g. 90 = 3s @ 30fps) as RGBA PNGs
       with transparent background (film_transparent=True).

  2. Run generate_video.py --hud-layer
     → Renders all TOTAL_FRAMES RGBA HUD overlay PNGs
       (transparent background, opaque HUD elements).

  3. Expand the short Blender loop to full video length via symlinks.

  4. FFmpeg composite:
        lavfi color (base fill) ──┐
        Blender RGBA frames  ─────┼─ overlay ─ overlay ─ [+ audio] → MP4
        HUD RGBA frames      ─────┘

Usage:
    python composite.py [--config config.yaml] [--preview] [--skip-blender]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_blender(hint: Optional[str] = None) -> Optional[str]:
    """Resolve the Blender executable.

    *hint* can be a bare command name ('blender') or an absolute path.
    If it's a bare name, shutil.which() is used so PATH-installed Blender
    (snap, apt, AppImage on PATH, etc.) is found correctly.
    """
    if hint:
        if os.path.isabs(hint):
            return hint if Path(hint).exists() else None
        # Bare name: resolve via PATH first
        resolved = shutil.which(hint)
        if resolved:
            return resolved

    # Fallback: generic name + known Linux locations
    hit = shutil.which("blender")
    if hit:
        return hit
    for p in ["/usr/bin/blender", "/usr/local/bin/blender",
              "/snap/bin/blender", os.path.expanduser("~/blender/blender"),
              os.path.expanduser("~/.local/bin/blender")]:
        if Path(p).exists():
            return p
    return None


def find_python() -> str:
    """Return the Python interpreter that runs this script."""
    return sys.executable


def run_blender_scene(
    blender_exe: str,
    scene_script: Path,
    output_dir: Path,
    n_frames: int,
    width: int, height: int, fps: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        blender_exe, "--background",
        "--python", str(scene_script),
        "--",                          # Blender stops parsing; script gets the rest
        "--output",  str(output_dir) + "/",
        "--frames",  str(n_frames),
        "--width",   str(width),
        "--height",  str(height),
        "--fps",     str(fps),
    ]
    print(f"  Running Blender: {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Blender render failed (exit {proc.returncode})")


def run_hud_layer(
    python_exe: str,
    gen_script: Path,
    config_path: Path,
    hud_dir: Path,
    preview: bool,
) -> None:
    cmd = [
        python_exe, str(gen_script),
        "--config", str(config_path),
        "--hud-layer",
        "--hud-output", str(hud_dir),
    ]
    if preview:
        cmd.append("--preview")
    print(f"  Running HUD layer: {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"HUD layer render failed (exit {proc.returncode})")


def expand_blender_loop(
    blender_dir: Path,
    expanded_dir: Path,
    loop_frames: int,
    total_frames: int,
) -> None:
    """
    Symlink the short Blender loop (e.g. 90 frames) to fill total_frames.
    Blender names frames: frame_0001.png … frame_00NN.png
    """
    expanded_dir.mkdir(parents=True, exist_ok=True)

    for i in range(total_frames):
        loop_idx  = (i % loop_frames) + 1               # 1-based
        src_name  = f"frame_{loop_idx:04d}.png"
        src       = blender_dir / src_name
        dst       = expanded_dir / f"frame_{i:05d}.png"

        if dst.exists() or dst.is_symlink():
            dst.unlink()

        if src.exists():
            dst.symlink_to(src.resolve())
        else:
            # Blender might use different zero padding; try 3-digit
            alt = blender_dir / f"frame_{loop_idx:03d}.png"
            if alt.exists():
                dst.symlink_to(alt.resolve())
            else:
                print(f"  [warn] Missing blender frame: {src_name}")


def composite_final(
    blender_expanded_dir: Path,
    hud_dir: Path,
    output_path: Path,
    width: int, height: int, fps: int,
    base_color: str,     # hex e.g. "0B0C0F"
    audio_track: Optional[Path],
    codec: str, pix_fmt: str,
) -> None:
    """
    Three-layer ffmpeg composite:
      Layer 0: lavfi solid base colour (background canvas)
      Layer 1: Blender RGBA frames     (3D geometry, emissive bloom, glitch)
      Layer 2: HUD RGBA frames         (pycairo grid, panels, typography)
    """
    # ffmpeg color filter wants hex without #
    colour = base_color.lstrip("#")

    inputs = [
        # Base colour flood fill
        "-f", "lavfi", "-i", f"color=c=#{colour}:s={width}x{height}:r={fps}",
        # Blender expanded frames
        "-framerate", str(fps), "-i", str(blender_expanded_dir / "frame_%05d.png"),
        # HUD frames
        "-framerate", str(fps), "-i", str(hud_dir / "frame_%05d.png"),
    ]

    filter_complex = (
        "[0:v][1:v]overlay=0:0:format=auto[v1];"
        "[v1][2:v]overlay=0:0:format=auto[vout]"
    )

    cmd = ["ffmpeg", "-y"] + inputs

    if audio_track and audio_track.exists():
        cmd += ["-stream_loop", "-1", "-i", str(audio_track)]
        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "3:a",
            "-shortest",
            "-c:v", codec, "-pix_fmt", pix_fmt,
            "-c:a", "aac", "-b:a", "192k",
        ]
    else:
        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-c:v", codec, "-pix_fmt", pix_fmt,
        ]

    cmd.append(str(output_path))

    print(f"  Compositing → {output_path.name}")
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg composite failed (exit {proc.returncode})")


def find_audio(config: Dict, root: Path) -> Optional[Path]:
    audio_dir = root / config["project"]["assets_dir"] / "audio"
    for f in sorted(audio_dir.glob("track_*.mp3")) if audio_dir.exists() else []:
        return f
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Blender (3D) + pycairo (HUD) compositor"
    )
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--preview",       action="store_true",
                        help="854×480 @ 24fps / 3s quick composite")
    parser.add_argument("--skip-blender",  action="store_true",
                        help="Reuse existing Blender frames (skip re-rendering)")
    parser.add_argument("--skip-hud",      action="store_true",
                        help="Reuse existing HUD frames (skip re-rendering)")
    args = parser.parse_args()

    root        = Path(__file__).resolve().parent
    cfg         = load_config(root / args.config)

    vcfg        = cfg["video"]
    style       = cfg["style"]
    bcfg        = cfg.get("blender", {})

    width, height = int(vcfg["width"]), int(vcfg["height"])
    fps           = int(vcfg["fps"])
    duration_sec  = int(vcfg["duration_sec"])
    total_frames  = fps * duration_sec

    if args.preview:
        width, height, fps, duration_sec = 854, 480, 24, 3
        total_frames = fps * duration_sec
        print("[preview] 854×480 @ 24fps / 3 s")

    loop_frames   = int(bcfg.get("loop_frames", 90))
    base_color    = style["palette"]["base"]

    # Directory layout
    tmp           = root / cfg["project"]["tmp_dir"]
    blender_raw   = tmp / "blender_frames"
    blender_exp   = tmp / "blender_expanded"
    hud_dir       = tmp / "hud_frames"
    output_dir    = root / cfg["project"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    hybrid_name   = bcfg.get("hybrid_output", "marathon_hybrid.mp4")
    output_path   = output_dir / hybrid_name

    # ── Step 1: Blender 3D layer ──────────────────────────────────────────────
    blender_script = root / "blender_scene.py"
    blender_exe    = find_blender(bcfg.get("executable") or None)

    if not blender_exe:
        print("[warn] Blender not found — searched PATH + common Linux locations.")
        print("       Install Blender (e.g. sudo snap install blender --classic)")
        print("       or set blender.executable to the full path in config.yaml.")
        print("       Falling back to pycairo-only composite.")
        args.skip_blender = True
    else:
        print(f"[info] Blender found: {blender_exe}")

    if not args.skip_blender and blender_exe:
        print("\n── Step 1/3: Blender 3D layer ───────────────────────────────")
        run_blender_scene(
            blender_exe, blender_script, blender_raw,
            loop_frames, width, height, fps,
        )
    else:
        print("\n── Step 1/3: [skipped] using existing Blender frames")

    # ── Step 2: HUD layer ─────────────────────────────────────────────────────
    gen_script = root / "generate_video.py"
    python_exe = find_python()

    if not args.skip_hud:
        print("\n── Step 2/3: pycairo HUD layer ──────────────────────────────")
        run_hud_layer(python_exe, gen_script,
                      root / args.config, hud_dir, args.preview)
    else:
        print("\n── Step 2/3: [skipped] using existing HUD frames")

    # ── Step 3: Expand blender loop + composite ───────────────────────────────
    print("\n── Step 3/3: Expanding Blender loop + compositing ───────────")

    actual_loop = loop_frames
    if blender_raw.exists():
        pngs = sorted(blender_raw.glob("frame_*.png"))
        if pngs:
            actual_loop = len(pngs)

    if blender_raw.exists() and any(blender_raw.glob("frame_*.png")):
        expand_blender_loop(blender_raw, blender_exp, actual_loop, total_frames)
        blender_input = blender_exp
    else:
        # No Blender frames: composite HUD over solid background only
        print("  [info] No Blender frames found \u2014 compositing HUD only")
        blender_input = None

    audio = find_audio(cfg, root)

    if blender_input:
        composite_final(
            blender_input, hud_dir, output_path,
            width, height, fps, base_color, audio,
            vcfg["codec"], vcfg["pixel_format"],
        )
    else:
        # HUD-only fallback: overlay HUD RGBA frames over base colour
        colour = base_color.lstrip("#")
        filter_c = "[0:v][1:v]overlay=0:0:format=auto[vout]"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i",
                f"color=c=#{colour}:s={width}x{height}:r={fps}",
            "-framerate", str(fps), "-i", str(hud_dir / "frame_%05d.png"),
        ]
        if audio and audio.exists():
            cmd += ["-stream_loop", "-1", "-i", str(audio),
                    "-filter_complex", filter_c,
                    "-map", "[vout]", "-map", "2:a",
                    "-frames:v", str(total_frames),
                    "-shortest", "-c:v", vcfg["codec"],
                    "-pix_fmt", vcfg["pixel_format"],
                    "-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-filter_complex", filter_c,
                    "-map", "[vout]",
                    "-frames:v", str(total_frames),
                    "-c:v", vcfg["codec"],
                    "-pix_fmt", vcfg["pixel_format"]]
        cmd.append(str(output_path))
        subprocess.run(cmd, check=True)

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest = {
        "project":    cfg["project"]["name"],
        "pipeline":   "blender_3d + pycairo_hud + ffmpeg_composite",
        "blender":    str(blender_exe),
        "video":      output_path.name,
        "loop_frames": actual_loop,
        "total_frames": total_frames,
    }
    (output_dir / "manifest_hybrid.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print(f"\nGenerated: {output_path}")


if __name__ == "__main__":
    main()
