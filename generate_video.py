#!/usr/bin/env python3
"""
Marathon Reboot — Homage Video Generator  v2
─────────────────────────────────────────────
Stack
  pycairo     vector frame rendering (sub-pixel AA, true 1px hairlines)
  Shapely     boolean / subtractive geometry for tech-panel cut-outs
  numpy       vignette + specular highlight (no film grain)
  matplotlib  data-density HUD sparkline / bar panels (pre-baked per scene)
  ffmpeg      receives frames via stdin pipe — zero temp files on disk
"""
import argparse
import io
import json
import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cairo
import kiwisolver as kiwi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import yaml
from PIL import Image
from shapely.geometry import Polygon
from z3 import If, Optimize, Real, sat

# ─────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────

def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs(paths: List[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None

def try_download(url: str, target: Path, timeout: int = 20) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and r.content:
            target.write_bytes(r.content)
            return True
    except requests.RequestException:
        pass
    return False

# ─────────────────────────────────────────────
# Colour utilities
# ─────────────────────────────────────────────

def hex_to_rgb01(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def hex_to_rgb8(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _wcag_lum(rgb8: Tuple[int, int, int]) -> float:
    def _lin(c: int) -> float:
        v = c / 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (_lin(x) for x in rgb8)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def _contrast(a8, b8) -> float:
    la, lb = _wcag_lum(a8), _wcag_lum(b8)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)

def readable_on(bg8: Tuple[int, int, int],
                *candidates: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Return the candidate with highest WCAG contrast ratio against bg, as 0-1 floats."""
    best = max(candidates, key=lambda c: _contrast(bg8, c))
    return tuple(x / 255.0 for x in best)


def solve_palette_ratios(style: Dict) -> Dict[str, float]:
    """
    Use Z3 to stabilize palette ratios so they always form a coherent scheme
    (sum to 1.0, avoid accent overpowering, remain close to configured intent).
    """
    ratio_cfg = style.get("palette_ratio", {})
    desired_base = float(ratio_cfg.get("base", 0.80))
    desired_secondary = float(ratio_cfg.get("secondary", 0.15))
    desired_accent = float(ratio_cfg.get("accent", 0.05))

    base = Real("base_ratio")
    secondary = Real("secondary_ratio")
    accent = Real("accent_ratio")

    o = Optimize()
    o.add(base >= 0.65, base <= 0.90)
    o.add(secondary >= 0.08, secondary <= 0.28)
    o.add(accent >= 0.02, accent <= 0.12)
    o.add(base + secondary + accent == 1.0)
    o.add(base >= secondary)
    o.add(secondary >= accent)

    # L1 distance to requested profile (linearized with If/absolute value)
    o.minimize(If(base - desired_base >= 0, base - desired_base, desired_base - base))
    o.minimize(If(secondary - desired_secondary >= 0,
                  secondary - desired_secondary,
                  desired_secondary - secondary))
    o.minimize(If(accent - desired_accent >= 0,
                  accent - desired_accent,
                  desired_accent - accent))

    if o.check() != sat:
        return {
            "base": desired_base,
            "secondary": desired_secondary,
            "accent": desired_accent,
        }

    m = o.model()
    return {
        "base": float(m[base].as_decimal(10).replace("?", "")),
        "secondary": float(m[secondary].as_decimal(10).replace("?", "")),
        "accent": float(m[accent].as_decimal(10).replace("?", "")),
    }


def solve_layout_rects(gm: Dict, width: int, height: int, focus_left: bool) -> Dict[str, Tuple[float, float, float, float]]:
    """Use Kassowary constraints to enforce consistent panel placements."""
    margin = float(gm["margin"])
    cell_w = float(gm["cell_w"])
    cell_h = float(gm["cell_h"])

    s = kiwi.Solver()
    hx0 = kiwi.Variable("hx0")
    hy0 = kiwi.Variable("hy0")
    hx1 = kiwi.Variable("hx1")
    hy1 = kiwi.Variable("hy1")
    px0 = kiwi.Variable("px0")
    py0 = kiwi.Variable("py0")
    px1 = kiwi.Variable("px1")
    py1 = kiwi.Variable("py1")

    # Hard constraints: safe zones and minimum dimensions
    for c in [
        hx0 >= margin, hy0 >= margin, hx1 <= width - margin, hy1 <= height - margin,
        px0 >= margin, py0 >= margin, px1 <= width - margin, py1 <= height - margin,
        hx1 - hx0 >= 10 * cell_w, hy1 - hy0 >= 8 * cell_h,
        px1 - px0 >= 5 * cell_w, py1 - py0 >= 5 * cell_h,
    ]:
        s.addConstraint(c)

    # Preferred anchors (weak) to keep recognizable design while allowing adaptation.
    hero_c0 = 1 if focus_left else 12
    hero_c1 = 13 if focus_left else 23
    panel_c0, panel_r0, panel_c1, panel_r1 = ((17, 2, 23, 8) if focus_left else (1, 7, 6, 13))

    s.addConstraint((hx0 == margin + hero_c0 * cell_w) | kiwi.strength.weak)
    s.addConstraint((hx1 == margin + hero_c1 * cell_w) | kiwi.strength.weak)
    s.addConstraint((hy0 == margin + 2 * cell_h) | kiwi.strength.weak)
    s.addConstraint((hy1 == margin + 12 * cell_h) | kiwi.strength.weak)

    s.addConstraint((px0 == margin + panel_c0 * cell_w) | kiwi.strength.weak)
    s.addConstraint((px1 == margin + panel_c1 * cell_w) | kiwi.strength.weak)
    s.addConstraint((py0 == margin + panel_r0 * cell_h) | kiwi.strength.weak)
    s.addConstraint((py1 == margin + panel_r1 * cell_h) | kiwi.strength.weak)

    # Keep visual separation between focal panel and opposite info panel.
    if focus_left:
        s.addConstraint((px0 >= hx1 + 2 * cell_w) | kiwi.strength.medium)
    else:
        s.addConstraint((hx0 >= px1 + 2 * cell_w) | kiwi.strength.medium)

    s.updateVariables()
    return {
        "hero_rect": (hx0.value(), hy0.value(), hx1.value(), hy1.value()),
        "data_rect": (px0.value(), py0.value(), px1.value(), py1.value()),
    }

# ─────────────────────────────────────────────
# Shapely → Cairo path
# ─────────────────────────────────────────────

def poly_to_cairo(ctx: cairo.Context, poly: Polygon) -> None:
    if poly.is_empty:
        return
    coords = list(poly.exterior.coords)
    ctx.move_to(*coords[0])
    for pt in coords[1:]:
        ctx.line_to(*pt)
    ctx.close_path()
    for ring in poly.interiors:
        ic = list(ring.coords)
        ctx.move_to(*ic[0])
        for pt in ic[1:]:
            ctx.line_to(*pt)
        ctx.close_path()

# ─────────────────────────────────────────────
# Numpy post-processing (computed once per resolution)
# ─────────────────────────────────────────────

def make_vignette(w: int, h: int, strength: float = 0.62) -> np.ndarray:
    """Float32 [H,W]: 1.0 at centre, (1-strength) at corners."""
    cx, cy = w / 2.0, h / 2.0
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
    return np.clip(1.0 - dist * (strength / 1.414), 1.0 - strength, 1.0).astype(np.float32)

def make_spec_map(w: int, h: int) -> np.ndarray:
    """Float32 [H,W]: additive specular brightness (left-edge wedge + right-top sliver)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = x / w, y / h
    s = np.zeros((h, w), dtype=np.float32)
    s[xn < 0.22 * (1.0 - yn)] = 0.055
    s[(xn > 0.90) & (yn < 0.25)] = 0.035
    return s

def post_process(rgb: np.ndarray, vignette: np.ndarray, spec: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32) / 255.0
    out *= vignette[:, :, np.newaxis]
    out += spec[:, :, np.newaxis]
    np.clip(out, 0.0, 1.0, out=out)
    return (out * 255.0).astype(np.uint8)

# ─────────────────────────────────────────────
# Cairo surface ↔ numpy
# ─────────────────────────────────────────────

def surface_to_rgb(surface: cairo.ImageSurface) -> np.ndarray:
    """Cairo FORMAT_RGB24 (LE: BGRX per pixel) → uint8 [H,W,3] RGB numpy array."""
    w, h = surface.get_width(), surface.get_height()
    buf = surface.get_data()
    arr = np.ndarray(shape=(h, w, 4), dtype=np.uint8, buffer=buf)
    return np.ascontiguousarray(arr[:, :, [2, 1, 0]])


def surface_to_rgba(surface: cairo.ImageSurface) -> np.ndarray:
    """
    Cairo FORMAT_ARGB32 (LE: B, G, R, A per pixel) → uint8 [H,W,4] RGBA.
    Used for hud_layer transparent frames that composite over the Blender layer.
    """
    w, h = surface.get_width(), surface.get_height()
    buf = surface.get_data()
    arr = np.ndarray(shape=(h, w, 4), dtype=np.uint8, buffer=buf)
    return np.ascontiguousarray(arr[:, :, [2, 1, 0, 3]])  # BGRA → RGBA

def rgb_to_cairo_surface(arr: np.ndarray) -> cairo.ImageSurface:
    """uint8 [H,W,3] RGB → Cairo FORMAT_RGB24 surface."""
    h, w = arr.shape[:2]
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    dest = np.ndarray((h, w, 4), dtype=np.uint8, buffer=surf.get_data())
    dest[:, :, 0] = arr[:, :, 2]   # B
    dest[:, :, 1] = arr[:, :, 1]   # G
    dest[:, :, 2] = arr[:, :, 0]   # R
    dest[:, :, 3] = 0xFF
    surf.mark_dirty()
    return surf

# ─────────────────────────────────────────────
# Matplotlib HUD panels (pre-baked once per scene)
# ─────────────────────────────────────────────

def bake_hud_panel(pw: int, ph: int,
                   base01: Tuple, fg01: Tuple, scene: int) -> np.ndarray:
    """
    Render a sparkline + bar chart panel via matplotlib.
    Returns uint8 [ph, pw, 3] RGB.  Called once per scene — NOT per frame.
    """
    rng = random.Random(scene * 9127 + 42)
    line_data = [rng.gauss(0.5, 0.18) for _ in range(40)]
    bar_vals   = [max(0.05, rng.gauss(0.55, 0.2)) for _ in range(14)]

    dpi = 100
    fig, (ax_l, ax_b) = plt.subplots(2, 1, figsize=(pw / dpi, ph / dpi), dpi=dpi)
    bg = (base01[0], base01[1], base01[2])
    fg = (fg01[0],   fg01[1],   fg01[2])

    for ax in (ax_l, ax_b):
        ax.set_facecolor(bg)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    ax_l.plot(line_data, color=fg, linewidth=0.9, alpha=0.9)
    ax_l.fill_between(range(len(line_data)), line_data, 0, color=fg, alpha=0.12)
    ax_b.bar(range(len(bar_vals)), bar_vals, color=fg, width=0.7, alpha=0.85)

    fig.patch.set_facecolor(bg)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03, hspace=0.08)

    buf = io.BytesIO()
    fig.savefig(buf, format="rgba", dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    actual_w, actual_h = (int(round(v)) for v in (fig.get_size_inches() * dpi))
    buf_bytes = buf.read()
    expected = actual_h * actual_w * 4
    print(f"[DEBUG] bake_hud_panel: actual_w={actual_w}, actual_h={actual_h}, buffer_len={len(buf_bytes)}, expected={expected}")
    # Fix buffer length mismatch: pad or crop as needed
    if len(buf_bytes) < expected:
        print(f"[FIX] Padding buffer from {len(buf_bytes)} to {expected}")
        buf_bytes += b'\x00' * (expected - len(buf_bytes))
    elif len(buf_bytes) > expected:
        print(f"[FIX] Cropping buffer from {len(buf_bytes)} to {expected}")
        buf_bytes = buf_bytes[:expected]
    try:
        raw = np.frombuffer(buf_bytes, dtype=np.uint8).reshape(actual_h, actual_w, 4)
    except ValueError as e:
        print(f"[ERROR] bake_hud_panel reshape failed: {e}")
        print(f"[DEBUG] buffer_len={len(buf_bytes)}, actual_h={actual_h}, actual_w={actual_w}, expected={expected}")
        raise
    rgb  = raw[:, :, :3]

    if rgb.shape[:2] != (ph, pw):
        print(f"[DEBUG] Resizing rgb from {rgb.shape[:2]} to {(ph, pw)}")
        rgb = np.array(Image.fromarray(rgb).resize((pw, ph), Image.LANCZOS))

    return rgb

# ─────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────

def grid_metrics(w: int, h: int, cols: int, rows: int, margin: int) -> Dict:
    return {
        "margin": margin,
        "cell_w": (w - 2 * margin) / cols,
        "cell_h": (h - 2 * margin) / rows,
    }

def cell_xy(c: float, r: float, gm: Dict) -> Tuple[float, float]:
    return gm["margin"] + c * gm["cell_w"], gm["margin"] + r * gm["cell_h"]

def cell_rect(c0, r0, c1, r1, gm) -> Tuple[float, float, float, float]:
    x0, y0 = cell_xy(c0, r0, gm)
    x1, y1 = cell_xy(c1, r1, gm)
    return x0, y0, x1, y1

# ─────────────────────────────────────────────
# Cairo draw primitives
# ─────────────────────────────────────────────

def draw_corner_brackets(ctx: cairo.Context, rect, color01, arm: float = 22.0) -> None:
    ctx.set_source_rgb(*color01)
    ctx.set_line_width(1.5)
    x0, y0, x1, y1 = rect
    for (ox, oy), (dx, dy) in [
        ((x0, y0), (+arm, 0)), ((x0, y0), (0, +arm)),
        ((x1, y0), (-arm, 0)), ((x1, y0), (0, +arm)),
        ((x0, y1), (+arm, 0)), ((x0, y1), (0, -arm)),
        ((x1, y1), (-arm, 0)), ((x1, y1), (0, -arm)),
    ]:
        ctx.move_to(ox, oy)
        ctx.line_to(ox + dx, oy + dy)
    ctx.stroke()

def draw_45_connector(ctx: cairo.Context,
                      start: Tuple, end: Tuple, color01,
                      width: float = 1.0) -> None:
    """Keyline with a single right-angle elbow."""
    ctx.set_source_rgb(*color01)
    ctx.set_line_width(width)
    sx, sy = start
    ex, ey = end
    mid_x = sx + (ex - sx) * 0.5
    ctx.move_to(sx, sy)
    ctx.line_to(mid_x, sy)
    ctx.line_to(mid_x, ey)
    ctx.line_to(ex, ey)
    ctx.stroke()

def draw_hazard_stripes(ctx: cairo.Context, rect, color01,
                        spacing: float = 14.0) -> None:
    x0, y0, x1, y1 = rect
    rh = y1 - y0
    ctx.save()
    ctx.rectangle(x0, y0, x1 - x0, rh)
    ctx.clip()
    ctx.set_source_rgb(*color01)
    ctx.set_line_width(5.0)
    x = x0 - rh
    while x < x1:
        ctx.move_to(x, y1)
        ctx.line_to(x + rh, y0)
        x += spacing
    ctx.stroke()
    ctx.restore()

def draw_subtractive_panel(ctx: cairo.Context, rect,
                            fill_color01, bg_color01) -> None:
    """
    Shapely boolean: outer rect  minus  inset rect  minus  corner notch triangle.
    Produces the classic Marathon-style negative-space tech frame.
    """
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0

    outer = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    inset = Polygon([
        (x0 + 32, y0 + 32),
        (x1 - 80, y0 + 32),
        (x1 - 80, y1 - 44),
        (x0 + 32, y1 - 44),
    ])
    notch = Polygon([(x1 - 90, y1), (x1, y1), (x1, y1 - 90)])
    tab   = Polygon([
        (x1 - 24, y0), (x1, y0),
        (x1, y0 + h * 0.2), (x1 - 24, y0 + h * 0.2),
    ])

    frame = outer.difference(inset).difference(notch)

    ctx.set_source_rgb(*fill_color01)
    for geom in ([frame] if frame.geom_type == 'Polygon'
                 else list(frame.geoms)):
        poly_to_cairo(ctx, geom)
    ctx.fill()

    # Accent tab (protrusion on the right edge)
    ctx.set_source_rgb(*fill_color01)
    poly_to_cairo(ctx, tab)
    ctx.fill()

    # Inner cut-out: faint accent tint to simulate self-illuminated display
    ctx.set_source_rgba(*bg_color01, 1.0)   # first fill with base...
    ctx.rectangle(x0 + 32, y0 + 32, (x1 - 80) - (x0 + 32), (y1 - 44) - (y0 + 32))
    ctx.fill()

def draw_crosshair(ctx: cairo.Context,
                   cx: float, cy: float, r: float, color01) -> None:
    ctx.set_source_rgb(*color01)
    ctx.set_line_width(1.0)
    ctx.move_to(cx - r, cy); ctx.line_to(cx + r, cy)
    ctx.move_to(cx, cy - r); ctx.line_to(cx, cy + r)
    ctx.stroke()
    ctx.arc(cx, cy, r * 0.35, 0, 2 * math.pi)
    ctx.stroke()
    ctx.arc(cx, cy, r * 0.08, 0, 2 * math.pi)
    ctx.fill()

def draw_tracked_text(ctx: cairo.Context,
                      x: float, y: float,
                      text: str, size: float, bold: bool,
                      color01,
                      tracking: float = 0.0,
                      vertical: bool = False) -> None:
    ctx.set_font_face(cairo.ToyFontFace(
        "Inter",
        cairo.FontSlant.NORMAL,
        cairo.FontWeight.BOLD if bold else cairo.FontWeight.NORMAL,
    ))
    ctx.set_font_size(size)
    ctx.set_source_rgb(*color01)
    for ch in text:
        ctx.move_to(x, y)
        ctx.show_text(ch)
        ext = ctx.text_extents(ch)
        if vertical:
            y += ext.height * (1.0 + max(0.0, tracking))
        else:
            x += ext.x_advance * (1.0 + tracking)

def composite_numpy_panel(ctx: cairo.Context,
                           x0: float, y0: float,
                           panel_rgb: np.ndarray) -> None:
    surf = rgb_to_cairo_surface(panel_rgb)
    ctx.save()
    ctx.set_source_surface(surf, x0, y0)
    ctx.paint()
    ctx.restore()

def draw_machine_vision_overlay(
    ctx: cairo.Context,
    cx: float, cy: float,          # center of tracked 3-D object (screen space)
    panel_rect,                    # (hx0, hy0, hx1, hy1) focal panel
    scene: int, scene_t: float, global_t: float,
    accent01, secondary01,
) -> None:
    """
    Debug / machine-vision aesthetic layer.
    Draws a target acquisition box centered on the Blender 3-D object, with
    dashed telemetry lines routing back to the focal analysis panel.
    Intent: looks like render-engine debug output overlaid on live 3-D data.
    """

    # ── Target acquisition box ─────────────────────────────────────────────
    breathe  = 1.0 + 0.012 * math.sin(2 * math.pi * global_t * 0.9)
    box_r    = 140 * breathe        # half-size in px
    arm_len  = box_r * 0.22         # L-bracket arm length

    ctx.set_line_width(1.2)
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        bx, by = cx + sx * box_r, cy + sy * box_r
        ctx.set_source_rgba(*accent01, 0.90)
        ctx.move_to(bx, by);            ctx.line_to(bx - sx * arm_len, by)
        ctx.stroke()
        ctx.move_to(bx, by);            ctx.line_to(bx, by - sy * arm_len)
        ctx.stroke()
        # Inner tick
        ctx.set_source_rgba(*accent01, 0.35)
        ctx.set_line_width(0.6)
        ctx.move_to(bx - sx * arm_len * 0.55, by)
        ctx.line_to(bx - sx * arm_len * 0.85, by)
        ctx.stroke()
        ctx.move_to(bx, by - sy * arm_len * 0.55)
        ctx.line_to(bx, by - sy * arm_len * 0.85)
        ctx.stroke()
        ctx.set_line_width(1.2)

    # ── Centre reticle (gap crosshair + ring) ──────────────────────────────
    ctx.set_source_rgba(*accent01, 0.88)
    ctx.set_line_width(1.0)
    gap = 5
    for (mx, my), (dx, dy) in [
        ((cx - 22, cy), (1, 0)), ((cx + 5, cy), (1, 0)),
        ((cx, cy - 22), (0, 1)), ((cx, cy + 5), (0, 1)),
    ]:
        ctx.move_to(mx, my)
        ctx.line_to(mx + dx * 17, my + dy * 17)
    ctx.stroke()
    ctx.arc(cx, cy, 6.5, 0, 2 * math.pi);  ctx.stroke()
    ctx.arc(cx, cy, 1.8, 0, 2 * math.pi);  ctx.fill()

    # ── Scan sweep line ────────────────────────────────────────────────────
    sweep_frac = (global_t * 1.6) % 1.0
    sweep_y    = cy - box_r + 2 * box_r * sweep_frac
    ctx.save()
    ctx.rectangle(cx - box_r, cy - box_r, 2 * box_r, 2 * box_r)
    ctx.clip()
    ctx.set_source_rgba(*accent01, 0.06)
    ctx.rectangle(cx - box_r, sweep_y - 12, 2 * box_r, 24)
    ctx.fill()
    ctx.set_source_rgba(*accent01, 0.28)
    ctx.set_line_width(0.5)
    ctx.move_to(cx - box_r, sweep_y);  ctx.line_to(cx + box_r, sweep_y)
    ctx.stroke()
    ctx.restore()

    # ── Corner annotation labels ───────────────────────────────────────────
    conf      = 0.90 + 0.07 * math.sin(global_t * 1.7 + scene)
    ctx.set_font_face(cairo.ToyFontFace("monospace"))
    ctx.set_font_size(9)
    for lx, ly, ltxt in [
        (cx + box_r + 7,  cy - box_r + 10,  "OBJ_001"),
        (cx + box_r + 7,  cy - box_r + 21,  f"CONF:{conf:.2f}"),
        (cx + box_r + 7,  cy + box_r - 13,  "TRACK:ON"),
        (cx - box_r - 52, cy - box_r + 10,  f"W:{int(box_r*2)}px"),
        (cx - box_r - 52, cy - box_r + 21,  "SCAN_OK"),
    ]:
        ctx.set_source_rgba(*accent01, 0.70)
        ctx.move_to(lx, ly)
        ctx.show_text(ltxt)

    # ── Dashed telemetry line: box edge → focal panel ──────────────────────
    hx0, hy0, hx1, hy1 = panel_rect
    # Route from the tracking box side nearest the panel
    if (hx0 + hx1) / 2 < cx:
        box_ax, panel_ax = cx - box_r, hx1
    else:
        box_ax, panel_ax = cx + box_r, hx0
    box_ay    = cy
    panel_ay  = (hy0 + hy1) / 2
    mid_x     = (box_ax + panel_ax) / 2

    ctx.set_dash([3.5, 4.5])
    ctx.set_source_rgba(*secondary01, 0.40)
    ctx.set_line_width(0.8)
    ctx.move_to(box_ax, box_ay)
    ctx.line_to(mid_x,    box_ay)
    ctx.line_to(mid_x,    panel_ay)
    ctx.line_to(panel_ax, panel_ay)
    ctx.stroke()
    ctx.set_dash([])

    # Midpoint metric label
    ctx.set_font_size(8)
    ctx.set_source_rgba(*secondary01, 0.65)
    ctx.move_to(mid_x + 4, box_ay - 4)
    ctx.show_text("ΔPOS:0.00")
    ctx.move_to(mid_x + 4, panel_ay - 4)
    ctx.show_text("LATCH")

    # Anchor dot at panel connection point
    ctx.set_source_rgba(*accent01, 0.6)
    ctx.arc(panel_ax, panel_ay, 2.5, 0, 2 * math.pi)
    ctx.fill()
    ctx.arc(box_ax, box_ay, 2.5, 0, 2 * math.pi)
    ctx.fill()


# ─────────────────────────────────────────────
# Per-frame renderer  (pure cairo — no matplotlib here)
# ─────────────────────────────────────────────

def render_frame(
    config: Dict,
    scene: int,
    scene_t: float,       # 0→1 within scene
    global_t: float,      # 0→1 across whole video
    width: int, height: int,
    gm: Dict,
    hud_panels: Dict[int, np.ndarray],
    base01, secondary01, accent01,
    base8,  secondary8,  accent8,
    hud_layer: bool = False,  # True → transparent bg (ARGB32) for Blender compositing
) -> np.ndarray:
    style   = config["style"]
    content = config["content"]
    ratios = solve_palette_ratios(style)

    if hud_layer:
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    else:
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, width, height)
    ctx = cairo.Context(surface)

    # ── Background ────────────────────────────────────────
    if hud_layer:
        # Transparent — Blender 3D layer will show through negative space
        ctx.set_operator(cairo.Operator.CLEAR)
        ctx.paint()
        ctx.set_operator(cairo.Operator.OVER)
    else:
        ctx.set_source_rgb(*base01)
        ctx.paint()

    # ── Faint structural grid ─────────────────────────────
    ctx.set_source_rgba(secondary01[0], secondary01[1], secondary01[2], 0.03 + 0.08 * ratios["secondary"])
    ctx.set_line_width(0.5)
    for c in range(1, int(style["grid"]["cols"])):
        x, _ = cell_xy(c, 0, gm)
        ctx.move_to(x, 0); ctx.line_to(x, height)
    for r in range(1, int(style["grid"]["rows"])):
        _, y = cell_xy(0, r, gm)
        ctx.move_to(0, y); ctx.line_to(width, y)
    ctx.stroke()

    # Safe-zone border
    m = gm["margin"]
    ctx.set_source_rgba(secondary01[0], secondary01[1], secondary01[2], 0.06 + 0.12 * ratios["secondary"])
    ctx.set_line_width(0.5)
    ctx.rectangle(m, m, width - 2*m, height - 2*m)
    ctx.stroke()

    # ── Focal panel (alternates left / right per scene) ───
    focus_left = scene % 2 == 0
    c0 = 1 if focus_left else 12
    c1 = 13 if focus_left else 23
    layout_rects = solve_layout_rects(gm, width, height, focus_left)
    hero_rect = layout_rects["hero_rect"]
    hx0, hy0, hx1, hy1 = hero_rect

    draw_subtractive_panel(ctx, hero_rect, secondary01, base01)

    # Inner cut-out accent tint
    ctx.set_source_rgba(accent01[0], accent01[1], accent01[2], 0.02 + 0.20 * ratios["accent"])
    ctx.rectangle(hx0 + 32, hy0 + 32, (hx1 - 80) - (hx0 + 32), (hy1 - 44) - (hy0 + 32))
    ctx.fill()

    # ── Boundary corner brackets ──────────────────────────
    bracket_col = readable_on(secondary8, base8, accent8)
    draw_corner_brackets(ctx, hero_rect, bracket_col)

    # ── Keyline connectors ────────────────────────────────
    kx  = hx0 - 140 if focus_left else hx1 + 140
    kyt = hy0 - gm["cell_h"] * 0.8
    draw_45_connector(ctx, (hx0 + 10, hy0 + 50), (kx, kyt), secondary01)
    draw_45_connector(ctx,
                      ((hx0 + hx1) / 2, hy1),
                      ((hx0 + hx1) / 2 + 80, hy1 + gm["cell_h"] * 0.7),
                      secondary01)
    # Tab connector
    draw_45_connector(ctx, (hx1 - 12, hy0 + (hy1 - hy0) * 0.10),
                           (hx1 + 60, hy0 - 30), secondary01)

    # ── Hazard stripes (scenes 1 & 3 only) ───────────────
    if scene in (1, 3):
        haz_rect = (hx0, hy1 + 18, min(hx1, hx0 + 360), hy1 + 18 + gm["cell_h"])
        draw_hazard_stripes(ctx, haz_rect, accent01)

    # ── Pre-baked HUD data panel (opposite side) ──────────
    panel = hud_panels.get(scene)
    if panel is not None:
        dr = solve_layout_rects(gm, width, height, focus_left)["data_rect"]
        composite_numpy_panel(ctx, dr[0], dr[1], panel)
        serial = f"SYS_{(scene*3571+1009)%9000+1000}_{(scene*127)%90+10}"
        ctx.set_font_face(cairo.ToyFontFace("monospace"))
        ctx.set_font_size(11)
        ctx.set_source_rgb(*secondary01)
        ctx.move_to(dr[0], dr[3] + 14)
        ctx.show_text(serial)

    # ── Crosshair reticle — off-axis counterweight (small, scene-dependent) ──
    if focus_left:
        crit_x, crit_y = cell_xy(21.5, 12.5, gm)
    else:
        crit_x, crit_y = cell_xy(2.5,   3.5, gm)
    draw_crosshair(ctx, crit_x, crit_y, 10, secondary01)   # small / subdued

    # ── Machine-vision overlay (anchored to frame centre = 3-D object pos) ──
    draw_machine_vision_overlay(
        ctx,
        cx=width * 0.5, cy=height * 0.5,
        panel_rect=(hx0, hy0, hx1, hy1),
        scene=scene, scene_t=scene_t, global_t=global_t,
        accent01=accent01, secondary01=secondary01,
    )

    # ── Animated accent pulse square (5% role) ────────────
    pulse = 0.5 + 0.5 * math.sin(2 * math.pi * (global_t * 2.4 + scene * 0.18))
    ctx.set_source_rgba(*accent01, (0.08 + 0.30 * ratios["accent"]) + pulse * (0.20 + 0.20 * ratios["accent"]))
    ctx.rectangle(hx0 - 7, hy0 - 7, 30, 30)
    ctx.fill()

    # ── Hero title — renders on the dark inner cut-out (base-coloured region)
    hero_txt   = content["hero_titles"][scene % len(content["hero_titles"])]
    hero_color = readable_on(base8, secondary8, accent8)  # inner bg is base (dark)
    draw_tracked_text(ctx, hx0 + 44, hy0 + 170, hero_txt, 98,
                      bold=True, color01=hero_color,
                      tracking=style["typography"]["hero_tracking"])

    # ── Micro status label (stable per scene, WCAG-safe) ──
    label_txt   = content["micro_labels"][scene % len(content["micro_labels"])]
    label_color = readable_on(secondary8, base8, accent8)
    draw_tracked_text(ctx, hx0 + 48, hy0 + 28, label_txt, 15,
                      bold=False, color01=label_color,
                      tracking=style["typography"]["micro_tracking"])

    # ── Vertical channel label ────────────────────────────
    if style["layout_elements"].get("vertical_labels", True):
        draw_tracked_text(ctx, hx1 - 24, hy0 + 55, f"CH-{scene+1:02d}", 14,
                          bold=False, color01=label_color,
                          tracking=0.06, vertical=True)

    # ── Extract pixels ────────────────────────────────────
    if hud_layer:
        # Return RGBA — post-processing (vignette/specular) is applied
        # at composite stage on the merged image, not on the transparent overlay
        return surface_to_rgba(surface)

    rgb = surface_to_rgb(surface)

    # Scene fade-in / fade-out (blend with flat base colour)
    fade = (min(1.0, scene_t / 0.10)
            * (1.0 - max(0.0, (scene_t - 0.90) / 0.10)))
    if fade < 1.0:
        base_fill = np.full_like(rgb, [int(x * 255) for x in base01], dtype=np.uint8)
        rgb = (rgb * fade + base_fill * (1.0 - fade)).astype(np.uint8)

    return rgb

# ─────────────────────────────────────────────
# FFmpeg stdin-pipe writer (zero temp files)
# ─────────────────────────────────────────────

class VideoWriter:
    def __init__(self, output_path: Path,
                 width: int, height: int, fps: int,
                 codec: str, pix_fmt: str,
                 audio_track: Optional[Path] = None):
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0",
        ]
        if audio_track and audio_track.exists():
            cmd += ["-stream_loop", "-1", "-i", str(audio_track),
                    "-shortest", "-c:a", "aac", "-b:a", "192k"]
        cmd += ["-c:v", codec, "-pix_fmt", pix_fmt, str(output_path)]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

    def write(self, rgb: np.ndarray) -> None:
        self._proc.stdin.write(rgb.tobytes())

    def close(self) -> None:
        self._proc.stdin.close()
        stderr = self._proc.stderr.read()
        self._proc.wait()
        if self._proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{stderr.decode(errors='replace')}")

# ─────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────

def download_fonts(config: Dict, assets_dir: Path) -> None:
    if not config["assets"]["fonts"]["download_google_fonts"]:
        return
    fonts_dir = assets_dir / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bslnt,wght%5D.ttf"
    try_download(url, fonts_dir / "Inter-VariableFont_slnt,wght.ttf")

def download_audio(config: Dict, assets_dir: Path) -> List[Path]:
    cfg = config["assets"]["audio"]
    if not cfg.get("download", False):
        return []
    audio_dir = assets_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    urls = [
        "https://cdn.pixabay.com/download/audio/2022/10/25/audio_946f6f3f2d.mp3?filename=technology-cyberpunk-ambient-116199.mp3",
        "https://cdn.pixabay.com/download/audio/2022/03/15/audio_4a7908f82f.mp3?filename=deep-electronic-technology-110624.mp3",
        "https://cdn.pixabay.com/download/audio/2023/03/13/audio_89666efa34.mp3?filename=dark-techno-142930.mp3",
    ]
    tracks: List[Path] = []
    for i, url in enumerate(urls[:int(cfg.get("max_tracks", 3))]):
        t = audio_dir / f"track_{i+1}.mp3"
        if try_download(url, t):
            tracks.append(t)
    (audio_dir / "metadata.json").write_text(
        json.dumps({"query": cfg.get("query"), "downloaded": [p.name for p in tracks]}, indent=2)
    )
    return tracks

def download_vectors(config: Dict, assets_dir: Path) -> List[Path]:
    cfg = config["assets"]["vectors"]
    if not cfg.get("download", False):
        return []
    vdir = assets_dir / "vectors"
    vdir.mkdir(parents=True, exist_ok=True)
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/0/02/SVG_logo.svg",
        "https://upload.wikimedia.org/wikipedia/commons/8/84/Example.svg",
    ]
    out: List[Path] = []
    for i, url in enumerate(urls[:int(cfg.get("max_svgs", 2))]):
        t = vdir / f"vector_{i+1}.svg"
        if try_download(url, t):
            out.append(t)
    return out

# ─────────────────────────────────────────────
# Main generation pipeline
# ─────────────────────────────────────────────

def generate_hud_layer(
    config: Dict, root: Path,
    out_dir: Path,
    preview: bool = False,
) -> int:
    """
    Render the pycairo HUD as RGBA PNG frames (transparent background) to out_dir.
    Returns the number of frames written.
    Used by composite.py to composite over the Blender 3D layer.
    """
    from PIL import Image as _Image
    vcfg = config["video"]
    width, height = int(vcfg["width"]), int(vcfg["height"])
    fps, dur      = int(vcfg["fps"]),   int(vcfg["duration_sec"])
    scene_count   = int(vcfg["scene_count"])

    if preview:
        width, height, fps, dur = 854, 480, 24, 3
        print("  [hud-layer preview] 854×480 @ 24fps / 3 s")

    total_frames = fps * dur
    assets_dir   = root / config["project"]["assets_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    style       = config["style"]
    solved_ratios = solve_palette_ratios(style)
    base01      = hex_to_rgb01(style["palette"]["base"])
    secondary01 = hex_to_rgb01(style["palette"]["secondary"])
    accent01    = hex_to_rgb01(style["palette"]["accent"])
    base8       = hex_to_rgb8(style["palette"]["base"])
    secondary8  = hex_to_rgb8(style["palette"]["secondary"])
    accent8     = hex_to_rgb8(style["palette"]["accent"])

    print(f"  Palette ratios (Z3): base={solved_ratios['base']:.3f} sec={solved_ratios['secondary']:.3f} acc={solved_ratios['accent']:.3f}")

    gm = grid_metrics(width, height,
                      int(style["grid"]["cols"]),
                      int(style["grid"]["rows"]),
                      int(style["grid"]["safe_margin_px"]))

    print("  Baking HUD panels…")
    hud_panels: Dict[int, np.ndarray] = {}
    for s in range(scene_count):
        focus_left = s % 2 == 0
        dr = solve_layout_rects(gm, width, height, focus_left)["data_rect"]
        pw, ph = max(4, int(dr[2]-dr[0])), max(4, int(dr[3]-dr[1]))
        hud_panels[s] = bake_hud_panel(pw, ph, base01, secondary01, s)

    step   = total_frames // scene_count
    bounds = [(i * step, (i+1)*step - 1 if i < scene_count-1 else total_frames-1)
              for i in range(scene_count)]

    report_every = max(1, total_frames // 20)
    print("  Rendering HUD frames (RGBA)…")
    for fidx in range(total_frames):
        if fidx % report_every == 0:
            pct = int(100 * fidx / total_frames)
            print(f"    {pct:3d}%  ({fidx}/{total_frames})", end="\r", flush=True)

        scene    = next(i for i, (s, e) in enumerate(bounds) if s <= fidx <= e)
        s_start, s_end = bounds[scene]
        scene_t  = (fidx - s_start) / max(1, s_end - s_start)
        global_t = fidx / max(1, total_frames - 1)

        rgba = render_frame(
            config, scene, scene_t, global_t,
            width, height, gm, hud_panels,
            base01, secondary01, accent01,
            base8,  secondary8,  accent8,
            hud_layer=True,
        )
        _Image.fromarray(rgba, "RGBA").save(out_dir / f"frame_{fidx:05d}.png")

    print(f"    100%  ({total_frames}/{total_frames})")
    return total_frames


def generate(config: Dict, root: Path,
             preview: bool = False) -> Tuple[Path, List[Path], List[Path]]:
    vcfg = config["video"]
    width, height   = int(vcfg["width"]), int(vcfg["height"])
    fps, dur        = int(vcfg["fps"]),   int(vcfg["duration_sec"])
    scene_count     = int(vcfg["scene_count"])

    if preview:
        width, height, fps, dur = 854, 480, 24, 3
        print("  [preview] 854×480 @ 24fps / 3 s")

    total_frames = fps * dur
    assets_dir   = root / config["project"]["assets_dir"]
    output_dir   = root / config["project"]["output_dir"]
    output_path  = output_dir / config["render"]["export_name"]

    style       = config["style"]
    solved_ratios = solve_palette_ratios(style)
    base01      = hex_to_rgb01(style["palette"]["base"])
    secondary01 = hex_to_rgb01(style["palette"]["secondary"])
    accent01    = hex_to_rgb01(style["palette"]["accent"])
    base8       = hex_to_rgb8(style["palette"]["base"])
    secondary8  = hex_to_rgb8(style["palette"]["secondary"])
    accent8     = hex_to_rgb8(style["palette"]["accent"])

    print(f"  Palette ratios (Z3): base={solved_ratios['base']:.3f} sec={solved_ratios['secondary']:.3f} acc={solved_ratios['accent']:.3f}")

    gm = grid_metrics(width, height,
                      int(style["grid"]["cols"]),
                      int(style["grid"]["rows"]),
                      int(style["grid"]["safe_margin_px"]))

    # Pre-bake HUD panels (matplotlib runs scene_count times only)
    print("  Baking HUD panels…")
    hud_panels: Dict[int, np.ndarray] = {}
    for s in range(scene_count):
        focus_left = s % 2 == 0
        dr = solve_layout_rects(gm, width, height, focus_left)["data_rect"]
        pw, ph = max(4, int(dr[2]-dr[0])), max(4, int(dr[3]-dr[1]))
        hud_panels[s] = bake_hud_panel(pw, ph, base01, secondary01, s)

    # Numpy LUTs
    vignette = make_vignette(width, height)
    spec_map  = make_spec_map(width, height)

    # Downloads
    download_fonts(config, assets_dir)
    vectors = download_vectors(config, assets_dir)
    tracks  = download_audio(config, assets_dir)
    audio   = tracks[0] if tracks else None

    # Scene boundaries
    step   = total_frames // scene_count
    bounds = [(i * step, (i+1)*step - 1 if i < scene_count-1 else total_frames-1)
              for i in range(scene_count)]

    writer       = VideoWriter(output_path, width, height, fps,
                               vcfg["codec"], vcfg["pixel_format"], audio)
    report_every = max(1, total_frames // 20)

    print("  Rendering frames…")
    for fidx in range(total_frames):
        if fidx % report_every == 0:
            pct = int(100 * fidx / total_frames)
            print(f"    {pct:3d}%  ({fidx}/{total_frames})", end="\r", flush=True)

        scene    = next(i for i, (s, e) in enumerate(bounds) if s <= fidx <= e)
        s_start, s_end = bounds[scene]
        scene_t  = (fidx - s_start) / max(1, s_end - s_start)
        global_t = fidx / max(1, total_frames - 1)

        frame = render_frame(
            config, scene, scene_t, global_t,
            width, height, gm, hud_panels,
            base01, secondary01, accent01,
            base8,  secondary8,  accent8,
            hud_layer=False,
        )
        frame = post_process(frame, vignette, spec_map)
        writer.write(frame)

    print(f"    100%  ({total_frames}/{total_frames})")
    writer.close()
    return output_path, tracks, vectors

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Marathon reboot homage generator "
                    "(pycairo + Shapely + matplotlib + ffmpeg pipe)"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--preview", action="store_true",
                        help="Quick 3-second 854×480 test render")
    parser.add_argument("--hud-layer", action="store_true",
                        help="Output transparent RGBA HUD PNG frames for Blender compositing")
    parser.add_argument("--hud-output", default=None,
                        help="Directory to write HUD layer PNGs (used with --hud-layer)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg  = load_config(root / args.config)

    if not ffmpeg_exists() and not args.hud_layer:
        raise RuntimeError("ffmpeg not found in PATH.")

    ensure_dirs([root / cfg["project"]["output_dir"],
                 root / cfg["project"]["assets_dir"]])

    if args.hud_layer:
        # HUD layer mode: emit transparent RGBA PNGs for Blender compositing
        hud_out = Path(args.hud_output) if args.hud_output else (
            root / cfg["project"]["tmp_dir"] / "hud_frames"
        )
        n = generate_hud_layer(cfg, root, hud_out, preview=args.preview)
        print(f"HUD layer written: {n} frames → {hud_out}")
        return

    out_path, tracks, vectors = generate(cfg, root, preview=args.preview)

    manifest = {
        "project":      cfg["project"]["name"],
        "style_mode":   "very_close_homage",
        "stack":        ["pycairo", "shapely", "numpy", "matplotlib", "ffmpeg-pipe"],
        "soundtrack":   cfg["assets"]["audio"]["mood"],
        "video":        out_path.name,
        "audio_tracks": [p.name for p in tracks],
        "vectors":      [p.name for p in vectors],
        "license_note": "Verify downloaded resource licenses before commercial use.",
    }
    (root / cfg["project"]["output_dir"] / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    main()
