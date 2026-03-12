"""
Marathon Reboot — Blender 3D Layer v2
======================================
Full Blender-native 2D pipeline (no pycairo, no matplotlib, no external Python).

  · Grease Pencil 3 strokes — background tactical grid, fiducial markers,
                               machine-vision scan overlay (parented to ico)
  · Curve objects + bevel_depth — safe-zone border, targeting brackets,
                                   HUD panel borders
  · GP3 LINEART modifier — icosphere edge trace (cyan)
  · GP3 per-frame drawing — sparkline HUD panels (pre-computed for all frames)
  · TextCurve objects — all screen-overlay text, partly updated per-frame
  · frame_change_pre handler — dynamic telemetry labels (frame counter, hex, UTC)

Run headless:
    blender --background --python blender_scene_v2.py -- \\
            --output /path/to/blender_v2_frames/ \\
            --frames 90 --width 1920 --height 1080

Output: transparent RGBA PNG sequence, ready for single-input ffmpeg encode.

Blender API target: 5.0.x  (GP3 drawing API confirmed via probe 2025-06)
"""

import sys
import math
import argparse
import random
import os

import bpy
import mathutils
import kiwisolver as kiwi

# ─────────────────────────────────────────────────────────────────────────────
# CLI args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--output",  default="/tmp/blender_v2_frames/")
    p.add_argument("--frames",  type=int, default=90)
    p.add_argument("--width",   type=int, default=1920)
    p.add_argument("--height",  type=int, default=1080)
    p.add_argument("--fps",     type=int, default=30)
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# World constants (derived from camera ortho_scale=16.0, res 1920×1080)
# ─────────────────────────────────────────────────────────────────────────────

HALF_W    = 8.0         # half-width in world units  (ortho_scale / 2)
HALF_H    = 4.5         # half-height  (16 × 1080/1920 / 2)
GRID_COLS = 24
GRID_ROWS = 16

# Y-depth layers (camera at Y=-10, looking in +Y)
DEPTH_BG  =  3.0        # background elements (grid, fiducials)
DEPTH_GEO =  0.0        # main 3D focal geometry
DEPTH_HUD = -2.0        # closest HUD panels (in front of 3D geo)

# ── Two-colour palette ── black + acid lime ─────────────────────────────────
BASE        = (0.003, 0.003, 0.003, 1.0)   # near-pure black background
FG          = (0.698, 1.000, 0.000, 1.0)   # acid lime  (≈ #B3FF00 sRGB)
# All accent aliases → same lime so existing material calls need no change
SECONDARY   = FG
ACCENT_YEL  = FG
ACCENT_CYA  = FG
ACCENT_ORG  = FG
ACCENT_RED  = FG
ACCENT_MAG  = FG
ACCENT_LIM  = FG


def validate_palette_consistency():
    """
    Validate the black + acid-lime palette using Cassowary constraints.

    This keeps the design system deterministic for the Blender-only pipeline:
      * every accent alias must resolve to the same FG colour
      * background must remain darker than foreground
      * alpha channels stay fully opaque
    """
    solver = kiwi.Solver()

    # Variable groups: background + canonical foreground + alias accents.
    var_specs = {
        "base": BASE,
        "fg": FG,
        "secondary": SECONDARY,
        "accent_yel": ACCENT_YEL,
        "accent_cya": ACCENT_CYA,
        "accent_org": ACCENT_ORG,
        "accent_red": ACCENT_RED,
        "accent_mag": ACCENT_MAG,
        "accent_lim": ACCENT_LIM,
    }
    channels = ("r", "g", "b", "a")
    vars_by_name = {
        name: {ch: kiwi.Variable(f"{name}_{ch}") for ch in channels}
        for name in var_specs
    }

    # Lock all solver variables to the constants defined in this module.
    for name, rgba in var_specs.items():
        for idx, ch in enumerate(channels):
            solver.addConstraint(vars_by_name[name][ch] == rgba[idx])

    # Every accent is exactly the same as canonical FG.
    for alias in ("secondary", "accent_yel", "accent_cya", "accent_org",
                  "accent_red", "accent_mag", "accent_lim"):
        for ch in channels:
            solver.addConstraint(vars_by_name[alias][ch] == vars_by_name["fg"][ch])

    # Background must stay darker than foreground, channel-wise.
    solver.addConstraint(vars_by_name["base"]["r"] <= vars_by_name["fg"]["r"])
    solver.addConstraint(vars_by_name["base"]["g"] <= vars_by_name["fg"]["g"])
    solver.addConstraint(vars_by_name["base"]["b"] <= vars_by_name["fg"]["b"])

    # Opaque palette (alpha = 1).
    solver.addConstraint(vars_by_name["base"]["a"] == 1.0)
    solver.addConstraint(vars_by_name["fg"]["a"] == 1.0)

    solver.updateVariables()


def ensure_design_consistency(screen_w, screen_h):
    """Run solver checks for both palette and canonical live-text layout."""
    validate_palette_consistency()
    target_rect = {
        "x": screen_w / 2.0,
        "y": screen_h / 2.0,
        "w": 200,
        "h": 200,
    }
    _ = calculate_ui_layout(target_rect, screen_w, screen_h)

# ── Font ───────────────────────────────────────────────────────────────
FONT_PATH = "/usr/share/fonts/opentype/b612/B612Mono-Bold.otf"
_font_cache = {}   # path → bpy.types.VectorFont


def load_font(path=FONT_PATH):
    """Load a TrueType/OpenType font into Blender, caching by path."""
    global _font_cache
    if path in _font_cache:
        return _font_cache[path]
    import os
    if os.path.exists(path):
        f = bpy.data.fonts.load(path)
    else:
        f = bpy.data.fonts.load("<builtin>")  # fallback
    _font_cache[path] = f
    return f


def calculate_ui_layout(target_rect, screen_width, screen_height):
    """
    Calculate non-overlapping 2D center coordinates for 4 text boxes around
    a central target using kiwisolver. Inputs are in screen pixels with
    `target_rect` = {'x': cx, 'y': cy, 'w': w, 'h': h} (center + size).

    Returns dict: {'text_1': (x, y), ...} where x,y are pixel centers.
    Grid-snapped to nearest multiple of 20.
    """
    # Fixed box sizes (pixels)
    box_w, box_h = 150.0, 40.0
    margin = 50.0   # safe screen margin
    padding = 40.0  # distance from target rect

    # Kiwi variables: centers of the boxes
    cx1 = kiwi.Variable('cx1')  # Text 1 (above)
    cy1 = kiwi.Variable('cy1')
    cx2 = kiwi.Variable('cx2')  # Text 2 (below)
    cy2 = kiwi.Variable('cy2')
    cx3 = kiwi.Variable('cx3')  # Text 3 (left)
    cy3 = kiwi.Variable('cy3')
    cx4 = kiwi.Variable('cx4')  # Text 4 (right)
    cy4 = kiwi.Variable('cy4')

    s = kiwi.Solver()

    tx, ty, tw, th = target_rect['x'], target_rect['y'], target_rect['w'], target_rect['h']
    t_left   = tx - tw / 2.0
    t_right  = tx + tw / 2.0
    t_top    = ty - th / 2.0
    t_bottom = ty + th / 2.0

    # Screen bounds (centers must stay so full box stays within margin)
    min_cx = margin + box_w / 2.0
    max_cx = screen_width - margin - box_w / 2.0
    min_cy = margin + box_h / 2.0
    max_cy = screen_height - margin - box_h / 2.0

    # --- Topology constraints ---
    # Text 1: strictly above target
    s.addConstraint(cy1 + (box_h / 2.0) <= t_top - padding)
    # Text 2: strictly below target
    s.addConstraint(cy2 - (box_h / 2.0) >= t_bottom + padding)
    # Text 3: strictly left of target
    s.addConstraint(cx3 + (box_w / 2.0) <= t_left - padding)
    # Text 4: strictly right of target
    s.addConstraint(cx4 - (box_w / 2.0) >= t_right + padding)

    # All boxes within screen safe margin
    for (cxv, cyv) in ((cx1, cy1), (cx2, cy2), (cx3, cy3), (cx4, cy4)):
        s.addConstraint(cxv >= min_cx)
        s.addConstraint(cxv <= max_cx)
        s.addConstraint(cyv >= min_cy)
        s.addConstraint(cyv <= max_cy)

    # Avoid overlap between boxes (axis-aligned separation)
    # Keep vertical pair (1 & 2) separated from left/right pair via padding
    s.addConstraint(cy1 + box_h/2.0 <= cy2 - box_h/2.0 - padding)

    # Place left/right boxes vertically near target center (weak)
    s.addConstraint(cy3 == ty, strength=kiwi.Strength.weak)
    s.addConstraint(cy4 == ty, strength=kiwi.Strength.weak)

    # Weak alignment: center X of Text 1 & 2 align to target center
    s.addConstraint(cx1 == tx, strength=kiwi.Strength.weak)
    s.addConstraint(cx2 == tx, strength=kiwi.Strength.weak)

    # Prefer horizontal spacing for left/right boxes around target (weak)
    s.addConstraint(cx3 <= tx - tw/2.0 - padding, strength=kiwi.Strength.weak)
    s.addConstraint(cx4 >= tx + tw/2.0 + padding, strength=kiwi.Strength.weak)

    # Fall-back soft preferences to encourage pleasing layout (all weak)
    s.addConstraint(cx1 >= (screen_width/2.0 - 100), strength=kiwi.Strength.weak)
    s.addConstraint(cx1 <= (screen_width/2.0 + 100), strength=kiwi.Strength.weak)

    # Solve
    s.updateVariables()

    def snap20(v):
        return int(round(v / 20.0) * 20)

    coords = {
        'text_1': (snap20(cx1.value()), snap20(cy1.value())),
        'text_2': (snap20(cx2.value()), snap20(cy2.value())),
        'text_3': (snap20(cx3.value()), snap20(cy3.value())),
        'text_4': (snap20(cx4.value()), snap20(cy4.value())),
    }
    return coords

# ─────────────────────────────────────────────────────────────────────────────
# Scene setup
# ─────────────────────────────────────────────────────────────────────────────

def purge_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.curves,
                 bpy.data.lights, bpy.data.materials, bpy.data.node_groups):
        for blk in list(coll):
            try:
                coll.remove(blk)
            except Exception:
                pass


def get_action_fcurves(action):
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    curves = []
    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in layer.strips:
                if hasattr(strip, "channelbags"):
                    for cb in strip.channelbags:
                        curves.extend(cb.fcurves)
    return curves


def set_fcurve_interp(obj, data_path, interp="CONSTANT"):
    if obj.animation_data and obj.animation_data.action:
        for fc in get_action_fcurves(obj.animation_data.action):
            if fc.data_path == data_path:
                for kp in fc.keyframe_points:
                    kp.interpolation = interp


def set_fcurve_expo(obj, data_path):
    if obj.animation_data and obj.animation_data.action:
        for fc in get_action_fcurves(obj.animation_data.action):
            if fc.data_path == data_path:
                for kp in fc.keyframe_points:
                    kp.interpolation = "EXPO"
                    kp.easing = "EASE_OUT"


def setup_scene(args):
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end   = args.frames
    scene.render.fps  = args.fps

    scene.render.resolution_x           = args.width
    scene.render.resolution_y           = args.height
    scene.render.resolution_percentage  = 100

    scene.render.film_transparent                 = True
    scene.render.image_settings.file_format       = "PNG"
    scene.render.image_settings.color_mode        = "RGBA"
    scene.render.image_settings.compression       = 15

    os.makedirs(args.output, exist_ok=True)
    scene.render.filepath = args.output + "frame_"

    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"

    # ── Pixel-sharp rendering ────────────────────────────────────────────────
    # Box filter at exactly 1 pixel — no sub-pixel smearing
    scene.render.filter_size = 1.0
    if hasattr(scene.render, "filter_type"):
        try:
            scene.render.filter_type = "BOX"
        except Exception:
            pass

    # Color management: Standard view transform (no Filmic tone-mapping blur)
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look           = "None"
    except Exception:
        pass

    eevee = scene.eevee
    # Bloom disabled — saturated palette is self-contrasting without glow
    if hasattr(eevee, "use_bloom"):
        eevee.use_bloom = False
    # TAA = 1 sample → no temporal smearing; lines stay razor-sharp
    if hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = 1
    if hasattr(eevee, "taa_samples"):
        eevee.taa_samples = 1
    # Disable any SMAA / viewport AA bleed
    if hasattr(eevee, "use_taa_reprojection"):
        eevee.use_taa_reprojection = False

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value    = (*BASE[:3], 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    scene.world = world


def setup_camera():
    cam_data = bpy.data.cameras.new("OrthoCamera")
    cam_data.type         = "ORTHO"
    cam_data.ortho_scale  = 16.0
    cam_data.clip_start   = 0.1
    cam_data.clip_end     = 60.0

    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location       = mathutils.Vector((0, -10, 0))
    cam.rotation_euler = mathutils.Euler((math.radians(90), 0, 0), "XYZ")
    bpy.context.scene.camera = cam
    return cam


# ─────────────────────────────────────────────────────────────────────────────
# Material factories
# ─────────────────────────────────────────────────────────────────────────────

def mat_flat(name, color):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    t = m.node_tree
    t.nodes.clear()
    out = t.nodes.new("ShaderNodeOutputMaterial")
    em  = t.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value    = color
    em.inputs["Strength"].default_value = 1.0
    t.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


def mat_emissive(name, color, strength=6.0):
    """Without bloom, emissive just renders at strength*color brightness.
    Keep for API compatibility; strength clamped to 1.0 so colours stay true."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    t = m.node_tree
    t.nodes.clear()
    out = t.nodes.new("ShaderNodeOutputMaterial")
    em  = t.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value    = color
    em.inputs["Strength"].default_value = min(strength, 1.0)  # no bloom = no overexpose
    t.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


def gp_mat(name, stroke_rgba, fill_rgba=None):
    """Create a Grease Pencil 3 material."""
    mat = bpy.data.materials.new(name)
    bpy.data.materials.create_gpencil_data(mat)
    gp = mat.grease_pencil
    gp.show_stroke = True
    gp.color       = stroke_rgba
    gp.show_fill   = fill_rgba is not None
    if fill_rgba is not None:
        gp.fill_color = fill_rgba
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# GP3 drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def gp_stroke(drawing, points_xyz, mat_idx=0, radius=0.003, cyclic=False):
    """
    Append one stroke to a GreasePencilDrawing (Blender 5.0 GP3 API).

    points_xyz : list of (x, y, z) — in the GP object's local space.
                 For objects placed at (0, Y_depth, 0) with no rotation,
                 set y=0 for all points to draw in the camera-visible XZ plane.
    radius     : stroke half-width in Blender world units (~metres).
    Returns    : index of the new stroke.
    """
    n = len(points_xyz)
    assert n >= 2, "Stroke needs at least 2 points"

    # Offsets before adding
    pt_offset = (len(drawing.attributes["position"].data)
                 if "position" in drawing.attributes else 0)
    cu_offset = len(drawing.strokes)

    # ── Add stroke ─────────────────────────────────────────────────────────
    drawing.add_strokes([n])

    # ── Ensure expected attribute arrays ───────────────────────────────────
    for aname, dtype, domain in (
        ("radius",         "FLOAT",   "POINT"),
        ("opacity",        "FLOAT",   "POINT"),
        ("material_index", "INT",     "CURVE"),
        ("cyclic",         "BOOLEAN", "CURVE"),
    ):
        if aname not in drawing.attributes:
            drawing.attributes.new(aname, dtype, domain)

    # ── Write data ─────────────────────────────────────────────────────────
    pos = drawing.attributes["position"]
    r   = drawing.attributes["radius"]
    op  = drawing.attributes["opacity"]
    mi  = drawing.attributes["material_index"]
    cy  = drawing.attributes["cyclic"]

    for i, pt in enumerate(points_xyz):
        pos.data[pt_offset + i].vector = pt
        r.data[pt_offset + i].value    = radius
        op.data[pt_offset + i].value   = 1.0

    mi.data[cu_offset].value = mat_idx
    cy.data[cu_offset].value = cyclic

    return cu_offset


def gp_clear(drawing):
    """Remove all strokes from a GreasePencilDrawing."""
    n = len(drawing.strokes)
    if n:
        drawing.remove_strokes(list(range(n)))


def new_gp_obj(name, location=(0, 0, 0)):
    """
    Create a new Grease Pencil 3 object via operator (guaranteed Blender 5.0 compat).
    Returns (object, gpdata).
    """
    bpy.ops.object.grease_pencil_add(type="EMPTY")
    obj      = bpy.context.active_object
    obj.name = name
    obj.location = mathutils.Vector(location)
    gpd      = obj.data
    gpd.name = name + "_Data"
    return obj, gpd


def gp_new_layer_frame(gpd, layer_name, frame_no):
    """Either retrieve or create the layer+frame for a given render frame."""
    if layer_name in gpd.layers:
        layer = gpd.layers[layer_name]
    else:
        layer = gpd.layers.new(layer_name, set_active=True)
    frame = layer.frames.new(frame_no)
    return layer, frame


# ─────────────────────────────────────────────────────────────────────────────
# Curve object helper  (bevel_depth = line thickness)
# ─────────────────────────────────────────────────────────────────────────────

def new_curve_obj(name, poly_lines, location=(0, 0, 0),
                  bevel=0.015, material=None):
    """
    Create a Curve object (POLY splines) from a list of point-lists.
    Each inner list is a list of (x, y, z, w=1) tuples.

    All splines share the same object, which is placed at `location`.
    The visible rendered stroke thickness comes from bevel_depth.
    """
    cd = bpy.data.curves.new(name, type="CURVE")
    cd.dimensions     = "3D"
    cd.bevel_depth    = bevel
    cd.bevel_resolution = 0

    for pts in poly_lines:
        spl = cd.splines.new("POLY")
        spl.points.add(len(pts) - 1)       # starts with 1 point
        for j, (x, y, z, w) in enumerate(pts):
            spl.points[j].co = (x, y, z, w)

    obj = bpy.data.objects.new(name, cd)
    bpy.context.collection.objects.link(obj)
    obj.location = mathutils.Vector(location)
    if material:
        cd.materials.append(material)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# 1. Background tactical grid  (GP3 strokes)
# ─────────────────────────────────────────────────────────────────────────────

def create_gp_grid():
    """
    24×16 tactical grid + safe-zone border as GP3 strokes.
    Object placed at Y=DEPTH_BG so it renders behind the 3D focal geometry.

    Coordinate convention for this GP object:
      stroke points are at local (x, 0, z) → world (x, DEPTH_BG, z)
      camera (at Y=-10, ortho) sees the XZ plane correctly.
    """
    obj, gpd = new_gp_obj("GP_BgGrid", location=(0, DEPTH_BG, 0))

    m_grid   = gp_mat("gp_grid",   (0.000, 1.000, 0.898, 0.12))  # dim cyan grid
    m_border = gp_mat("gp_border", (0.914, 1.000, 0.137, 0.80))  # yellow safe-zone
    gpd.materials.append(m_grid)    # index 0
    gpd.materials.append(m_border)  # index 1

    layer = gpd.layers.new("Grid", set_active=True)
    frame = layer.frames.new(1)
    d = frame.drawing

    step_x = (HALF_W * 2) / GRID_COLS    # 16/24 ≈ 0.667 u
    step_z = (HALF_H * 2) / GRID_ROWS    #  9/16 = 0.5625 u

    # Vertical lines
    for col in range(GRID_COLS + 1):
        x = -HALF_W + col * step_x
        gp_stroke(d, [(x, 0, -HALF_H), (x, 0, HALF_H)],
                  mat_idx=0, radius=0.003)

    # Horizontal lines
    for row in range(GRID_ROWS + 1):
        z = -HALF_H + row * step_z
        gp_stroke(d, [(-HALF_W, 0, z), (HALF_W, 0, z)],
                  mat_idx=0, radius=0.003)

    # Safe-zone border (inset by camera-margin equivalent 0.333 u)
    mg = 0.333
    mx, mz = HALF_W - mg, HALF_H - mg
    border = [(-mx, 0, -mz), (mx, 0, -mz), (mx, 0, mz), (-mx, 0, mz)]
    gp_stroke(d, border, mat_idx=1, radius=0.002, cyclic=True)

    print(f"[v2] GP grid: {len(d.strokes)} strokes")
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# 2. Halftone dot matrix  (GP3 — solid square dots filling corners)
# ─────────────────────────────────────────────────────────────────────────────

def create_gp_fiducials():
    """
    Halftone dot matrix: a regular grid of small filled square dots that
    covers the full frame except the central focal zone.  This mirrors the
    dense halftone texture seen in the Marathon Update screenshot.
    Dot size is uniform; alpha fades linearly from 1.0 at corners to 0.0
    at the focal exclusion boundary.
    """
    obj, gpd = new_gp_obj("GP_Halftone", location=(0, DEPTH_BG - 0.05, 0))

    # Full-opacity lime material for dots
    m_dot = gp_mat("gp_dot", (*FG[:3], 1.0))
    gpd.materials.append(m_dot)

    layer = gpd.layers.new("Dots", set_active=True)
    frame = layer.frames.new(1)
    d = frame.drawing

    step  = 0.38     # grid spacing in world units (approx 46px at 1920px/16u)
    r_dot = 0.022    # dot radius (~2.6px half-width — renders as small filled square)
    r_excl = 3.6     # exclusion radius around focal geometry

    # Build a coarse grid across the full visible area
    cols = int(math.ceil((HALF_W * 2) / step)) + 2
    rows = int(math.ceil((HALF_H * 2) / step)) + 2

    for ci in range(cols):
        for ri in range(rows):
            wx = -HALF_W + ci * step
            wz = -HALF_H + ri * step
            # Skip focal zone
            if math.sqrt(wx * wx + wz * wz) < r_excl:
                continue
            # A "dot" is a zero-length stroke (two coincident points) with radius
            gp_stroke(d, [(wx, 0, wz), (wx, 0, wz)], mat_idx=0, radius=r_dot)

    print(f"[v2] GP halftone: {len(d.strokes)} dot strokes")
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# 3. Focal 3D geometry  (same core as v1)
# ─────────────────────────────────────────────────────────────────────────────

def create_focal_geometry(n_frames):
    """
    EEVEE torus + icosphere + inner glow with 45° snap rotation animation.
    Identical to v1 — the 3D core does not change in this migration.
    """
    mat_torus = mat_flat("mat_torus",  ACCENT_YEL)   # acid yellow ring
    mat_ico   = mat_flat("mat_ico",    ACCENT_CYA)   # cyan wireframe icosphere
    mat_inner = mat_flat("mat_inner",  ACCENT_ORG)   # orange inner core

    bpy.ops.mesh.primitive_torus_add(
        major_radius=3.2, minor_radius=0.08,
        major_segments=64, minor_segments=12,
        location=(0, 0, 0))
    torus      = bpy.context.object
    torus.name = "FocalTorus"
    # Rotate 90° on X so the major ring lies in the XZ plane (vertical circle
    # facing the camera, which looks in +Y).  Default is XY-plane (edge-on).
    torus.rotation_euler = mathutils.Euler((math.radians(90), 0, 0), "XYZ")
    torus.data.materials.append(mat_torus)

    bpy.ops.mesh.primitive_ico_sphere_add(
        radius=1.6, subdivisions=2, location=(0, 0, 0))
    ico      = bpy.context.object
    ico.name = "FocalIco"
    wire_mod = ico.modifiers.new("Wire", "WIREFRAME")
    wire_mod.thickness    = 0.022
    wire_mod.use_replace  = True
    ico.data.materials.append(mat_ico)

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.4, segments=12, ring_count=8, location=(0, 0, 0))
    inner      = bpy.context.object
    inner.name = "InnerGlow"
    inner.data.materials.append(mat_inner)

    snap_interval = 15

    def _snap_animate(obj, base_rot_x=0.0):
        """Keyframe 45° Z-snap rotation, preserving a fixed X rotation."""
        ca = 0.0
        frm = 1
        obj.rotation_euler = mathutils.Euler(
            (math.radians(base_rot_x), 0, math.radians(ca)), "XYZ")
        obj.keyframe_insert("rotation_euler", frame=1)
        while frm <= n_frames:
            obj.rotation_euler = mathutils.Euler(
                (math.radians(base_rot_x), 0, math.radians(ca)), "XYZ")
            obj.keyframe_insert("rotation_euler", frame=frm)
            nxt = frm + snap_interval
            if nxt > n_frames:
                break
            ca += 45.0
            obj.rotation_euler = mathutils.Euler(
                (math.radians(base_rot_x), 0, math.radians(ca)), "XYZ")
            obj.keyframe_insert("rotation_euler", frame=nxt)
            set_fcurve_expo(obj, "rotation_euler")
            frm = nxt

    # Torus already has 90° X rotation set above — preserve it in animation
    _snap_animate(torus, base_rot_x=90.0)
    _snap_animate(ico,   base_rot_x=0.0)

    return torus, ico, inner


# ─────────────────────────────────────────────────────────────────────────────
# 4. Targeting brackets  (Curve objects, Vector handles — same as v1)
# ─────────────────────────────────────────────────────────────────────────────

def create_targeting_brackets(scale=4.0):
    mat_b = mat_flat("mat_bracket", ACCENT_YEL)
    arm   = scale * 0.28

    # corners: (world_X, world_Z, horizontal_X_dir, vertical_Z_dir)
    corners = [
        (-scale, +scale, +1, -1),   # top-left
        (+scale, +scale, -1, -1),   # top-right
        (-scale, -scale, +1, +1),   # bottom-left
        (+scale, -scale, -1, +1),   # bottom-right
    ]
    for i, (cx, cz, hd, vd) in enumerate(corners):
        # Points live in world XZ — Y axis is depth (constant 0)
        pts = [
            (cx + hd * arm, 0, cz,           1.0),  # horizontal arm tip
            (cx,            0, cz,           1.0),  # corner
            (cx,            0, cz + vd * arm, 1.0),  # vertical arm tip
        ]
        obj = new_curve_obj(f"Bracket_{i}", [pts], bevel=0.012, material=mat_b)
        for frm, sz in [(1, 0.95), (45, 1.05), (90, 0.95)]:
            obj.scale = (1.0, 1.0, sz)
            obj.keyframe_insert("scale", frame=frm)

    # ── Solid filled square at each corner — like the Marathon screenshot ──
    # Implemented as a zero-length GP stroke with large radius = solid square cap
    sq_obj, sq_gpd = new_gp_obj("GP_BracketSquares", location=(0, DEPTH_HUD - 0.1, 0))
    m_sq = gp_mat("gp_brk_sq", (*FG[:3], 1.0))
    sq_gpd.materials.append(m_sq)
    sq_layer = sq_gpd.layers.new("Sq", set_active=True)
    sq_frame = sq_layer.frames.new(1)
    sq_d = sq_frame.drawing
    sq_r = 0.10   # square half-side ≈ 12px at 1920px
    for cx, cz, _hd, _vd in corners:
        gp_stroke(sq_d, [(cx, 0, cz), (cx, 0, cz)], mat_idx=0, radius=sq_r)


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Sidebar label strips  (left edge — rotated serial codes + identifier boxes)
# ─────────────────────────────────────────────────────────────────────────────

def create_sidebar_labels():
    """
    Marathon-style left-edge sidebar: identifier code boxes + rotated serial text.
    Two instances: one in the upper panel region, one in the lower panel region.
    Layout mirrors the AU-233 sidebar strips in the Marathon Update screenshot.
    """
    D = DEPTH_HUD - 0.15
    mat_fg  = mat_flat("mat_sb_fg",  FG)
    mat_dim = mat_flat("mat_sb_dim", (*FG[:3], 0.50))

    # GP3 object for the small filled-square dash bullets
    bullet_obj, bullet_gpd = new_gp_obj("GP_SidebarBullets", location=(0, D, 0))
    m_b = gp_mat("gp_sb_bullet", (*FG[:3], 1.0))
    bullet_gpd.materials.append(m_b)
    b_layer = bullet_gpd.layers.new("Bullets", set_active=True)
    b_frame = b_layer.frames.new(1)
    b_d = b_frame.drawing

    def _txt(name, body, x, z, size, mat, align="LEFT", rotation_z=0):
        bpy.ops.object.text_add(location=(x, D, z))
        obj = bpy.context.object
        obj.name = name
        obj.data.body    = body
        obj.data.size    = size
        obj.data.font    = load_font()
        obj.data.align_x = align
        obj.data.materials.append(mat)
        # 90° X rotation so text faces the orthographic camera (XZ plane)
        # plus optional Z rotation for vertical text (-90° rotates text to read bottom-up)
        obj.rotation_euler = mathutils.Euler(
            (math.radians(90), 0, math.radians(rotation_z)), "XYZ")
        return obj

    def _sidebar_strip(idx, z_centre):
        """One AU-style strip centred at world Z = z_centre, left edge."""
        bx   = -7.80          # bullet / box X position (left edge)
        box_w = 0.45           # width of the identifier box
        box_h = 0.28           # height

        # ─ short horizontal dash tick (two dashes + gap) above identifier box─
        dash_z = z_centre + 0.62
        for dx in (-0.22, 0.0):
            gp_stroke(b_d, [(bx + dx, 0, dash_z), (bx + dx + 0.14, 0, dash_z)],
                      mat_idx=0, radius=0.004)

        # ─ filled-square bullet dot─
        sq_z = z_centre + 0.35
        gp_stroke(b_d, [(bx + 0.05, 0, sq_z), (bx + 0.05, 0, sq_z)],
                  mat_idx=0, radius=0.075)

        # ─ identifier box border (open rectangle)─
        px = 0.03;  pz = 0.04
        box_pts = [
            (bx - px,        0, z_centre - box_h/2 - pz, 1.0),
            (bx + box_w + px, 0, z_centre - box_h/2 - pz, 1.0),
            (bx + box_w + px, 0, z_centre + box_h/2 + pz, 1.0),
            (bx - px,         0, z_centre + box_h/2 + pz, 1.0),
            (bx - px,         0, z_centre - box_h/2 - pz, 1.0),
        ]
        new_curve_obj(f"SB_{idx}_Box", [box_pts],
                      location=(0, D, 0), bevel=0.004, material=mat_fg)

        # ─ identifier text inside the box (e.g. “AU-233”)─
        serial = ["AU-127", "AU-233"][idx % 2]
        _txt(f"SB_{idx}_ID", serial,
             bx + 0.04, z_centre - 0.08, 0.16, mat_fg, align="LEFT")

        # ─ rotated serial code column (reads bottom-to-top, tight against box)─
        serials = [
            ("ZMRD F324-S234", -0.14),
            ("BK21-0023-RR44", -0.34),
            ("D654 .. 003",               -0.54),
        ]
        for body, dz in serials:
            _txt(f"SB_{idx}_Ser{dz:.0f}", body,
                 bx - 0.08, z_centre + dz, 0.085, mat_dim,
                 align="LEFT", rotation_z=-90)

    # Two strips on the left edge
    _sidebar_strip(0, z_centre= 0.80)   # upper (aligned with top HUD panel area)
    _sidebar_strip(1, z_centre=-2.20)   # lower (aligned with bottom of panels)


# ─────────────────────────────────────────────────────────────────────────────
# 5a. Logotype — MARATHON wordmark, subtitle rule, corner IDs
# ─────────────────────────────────────────────────────────────────────────────

def create_logotype():
    """
    Top-centre wordmark: 'MARATHON' in large white, subtitle below,
    thin separator rules, and corner identifier stamps.
    """
    mat_white = mat_flat("mat_logo_white", ACCENT_YEL)           # wordmark = yellow
    mat_yel   = mat_flat("mat_logo_yel",   ACCENT_YEL)
    mat_org   = mat_flat("mat_logo_org",   ACCENT_ORG)
    mat_cya   = mat_flat("mat_logo_cya",   ACCENT_CYA)
    mat_dim   = mat_flat("mat_logo_dim",   (*ACCENT_CYA[:3], 0.45))  # dim cyan

    # Depth slightly in front of text panel (but behind MV overlay)
    D = DEPTH_HUD - 0.3

    def _txt(name, body, x, z, size, mat, align="CENTER"):
        bpy.ops.object.text_add(location=(x, D, z))
        obj = bpy.context.object
        obj.name = name
        obj.data.body    = body
        obj.data.size    = size
        obj.data.font    = load_font()
        obj.data.align_x = align
        obj.data.materials.append(mat)
        obj.rotation_euler = mathutils.Euler((math.radians(90), 0, 0), "XYZ")
        return obj

    # ── Main wordmark ──────────────────────────────────────────────────────
    _txt("Logo_Main",   "MARATHON",                   0.0,  +4.10, 0.52, mat_white, "CENTER")
    _txt("Logo_Sub",    "TACTICAL OVERLAY  //  UNIT 041  //  SEC:BRAVO",
                                                       0.0,  +3.42, 0.115, mat_yel, "CENTER")

    # ── Separator rule under subtitle ─────────────────────────────────────
    mat_rule = mat_flat("mat_rule_dim", (*ACCENT_YEL[:3], 0.38))  # dim yellow rule
    rule_pts = [(-3.8, 0, +3.57, 1.0), (3.8, 0, +3.57, 1.0)]
    new_curve_obj("Logo_Rule", [rule_pts],
                  location=(0, D, 0), bevel=0.002, material=mat_rule)

    # ── Corner identifier stamps ───────────────────────────────────────────
    # Top-left bracket label
    _txt("Logo_TL_ID",  "MNRTN",          -7.55, +4.20, 0.17, mat_yel,  "LEFT")
    _txt("Logo_TL_Rev", "REV.4.1",        -7.55, +3.92, 0.14, mat_dim,  "LEFT")
    # Top-right bracket label
    _txt("Logo_TR_ID",  "UNIT:041",       +5.80, +4.20, 0.17, mat_org,  "LEFT")
    _txt("Logo_TR_Sub", "SEC:BRAVO",      +5.80, +3.92, 0.14, mat_dim,  "LEFT")
    # Bottom corner IDs
    _txt("Logo_BL",     "SYS:0x0B0C0F",  -7.55, -4.30, 0.13, mat_cya,  "LEFT")
    _txt("Logo_BR",     "LOCK:CONFIRMED", +4.00, -4.30, 0.15, mat_yel,  "LEFT")

    # ── Two stacked accent rules at top edge — yellow + magenta ─────────
    mat_topbar_y = mat_flat("mat_topbar_y", ACCENT_YEL)
    mat_topbar_m = mat_flat("mat_topbar_m", ACCENT_MAG)
    new_curve_obj("Logo_TopBar",  [[ (-7.65, 0, +4.38, 1.0), (7.65, 0, +4.38, 1.0) ]],
                  location=(0, D, 0), bevel=0.004, material=mat_topbar_y)
    new_curve_obj("Logo_TopBar2", [[ (-7.65, 0, +4.32, 1.0), (7.65, 0, +4.32, 1.0) ]],
                  location=(0, D, 0), bevel=0.002, material=mat_topbar_m)


# ─────────────────────────────────────────────────────────────────────────────
# 5b. Warning-stripe band  (bottom of frame, diagonal hazard stripes)
# ─────────────────────────────────────────────────────────────────────────────

def create_warning_stripes():
    """
    Classic construction-tape hazard stripes at the bottom frame edge,
    contained within a fully-bordered rectangle (all 4 sides closed).
    Alternating yellow / dark-navy diagonal stripes + CAUTION label.
    """
    mat_yel = mat_flat("mat_stripe_yel", ACCENT_YEL)
    mat_blk = mat_flat("mat_stripe_blk", (0.025, 0.030, 0.040, 1.0))  # dark navy stripe
    mat_bdr = mat_flat("mat_stripe_bdr", ACCENT_ORG)                   # orange border rules
    # Label uses ACCENT_MAG (magenta) — contrasts against BOTH yellow and dark navy
    mat_hz  = mat_flat("mat_hz_mag",    ACCENT_MAG)

    # Bounding rectangle for the hazard band
    x0, x1 = -HALF_W + 0.04,  HALF_W - 0.04   # left/right within safe zone
    z0, z1 =  -4.50, -3.82                      # band vertical extent
    height  =  z1 - z0                          # ≈ 0.68 world units
    step    =  0.68                             # stripe pitch
    bevel   =  step * 0.465                     # half-stripe fills the gap
    D = DEPTH_HUD + 0.8                         # behind HUD, in front of BG

    # ── Diagonal stripes (overscan then visually bounded by the border rect) ──
    x_start = x0 - height - step
    x_end   = x1 + step
    n_stripes = int((x_end - x_start) / step) + 2
    for i in range(n_stripes):
        x_bot = x_start + i * step
        x_top = x_bot + height    # 45° lean (rise = run = height)
        # Clip to bounding box: skip stripes entirely outside
        if x_top < x0 or x_bot > x1:
            continue
        pts = [(x_bot, 0, z0, 1.0), (x_top, 0, z1, 1.0)]
        mat = mat_yel if i % 2 == 0 else mat_blk
        new_curve_obj(f"Stripe_{i:02d}", [pts],
                      location=(0, D, 0), bevel=bevel, material=mat)

    # ── Bounding rectangle (all 4 sides — Curve POLY closed loop) ──────────
    rect_pts = [
        (x0, 0, z0, 1.0), (x1, 0, z0, 1.0),
        (x1, 0, z1, 1.0), (x0, 0, z1, 1.0),
        (x0, 0, z0, 1.0),   # close
    ]
    new_curve_obj("StripeBorderRect", [rect_pts],
                  location=(0, D + 0.12, 0), bevel=0.009, material=mat_bdr)

    # ── Interior corner notches (orange accent marks inside corners) ──────
    notch = 0.30
    for (xc, zc, xd, zd) in [( x0, z0, +1, +1), (x1, z0, -1, +1),
                               ( x0, z1, +1, -1), (x1, z1, -1, -1)]:
        pts_n = [(xc + xd * notch, 0, zc, 1.0),
                 (xc,              0, zc, 1.0),
                 (xc,              0, zc + zd * notch, 1.0)]
        new_curve_obj(f"StripeNotch_{xc:.0f}_{zc:.0f}", [pts_n],
                      location=(0, D + 0.13, 0), bevel=0.007, material=mat_bdr)

    # ── CAUTION label centred inside band ─────────────────────────────────
    # Text in magenta — contrasts strongly against yellow AND dark-navy stripes
    lz = (z0 + z1) / 2 - 0.08
    bpy.ops.object.text_add(location=(0, D + 0.16, lz))
    hz = bpy.context.object
    hz.name = "StripeLabel"
    hz.data.body    = "// CAUTION: RESTRICTED SECTOR //"
    hz.data.size    = 0.18
    hz.data.font    = load_font()
    hz.data.align_x = "CENTER"
    hz.data.materials.append(mat_hz)
    hz.rotation_euler = mathutils.Euler((math.radians(90), 0, 0), "XYZ")

    # ── Small sector identifier stamps at left/right ends of band ─────────
    mat_id = mat_flat("mat_stripe_id", ACCENT_YEL)
    for body, x, align in [("SEC-7", x0 + 0.08, "LEFT"),
                            ("SEC-7", x1 - 0.08, "RIGHT")]:
        bpy.ops.object.text_add(location=(x, D + 0.16, lz))
        t = bpy.context.object
        t.name = f"StripeID_{align}"
        t.data.body    = body
        t.data.size    = 0.14
        t.data.font    = load_font()
        t.data.align_x = align
        t.data.materials.append(mat_id)
        t.rotation_euler = mathutils.Euler((math.radians(90), 0, 0), "XYZ")


# ─────────────────────────────────────────────────────────────────────────────
# 5c. HUD panel decorations — header bars, dividers, data labels
# ─────────────────────────────────────────────────────────────────────────────

def create_panel_decorations():
    """
    Adds visual structure inside and around each sparkline HUD panel:
      - Header bar (yellow accent line at panel top, thicker than border)
      - Horizontal mid-divider
      - Corner cut marks (angled accent marks at panel corners)
      - Y-axis tick marks
    """
    mat_acc  = mat_flat("mat_pnl_acc",  ACCENT_YEL)
    mat_cya  = mat_flat("mat_pnl_cya",  ACCENT_CYA)
    mat_dim  = mat_flat("mat_pnl_dim",  (*ACCENT_CYA[:3], 0.55))  # cyan ticks — boosted opacity
    mat_wht  = mat_flat("mat_pnl_wht",  ACCENT_YEL)               # yellow panel titles

    D = DEPTH_HUD - 0.2   # slightly in front of panel borders

    # Panel geometry bounds (must match create_hud_panels and sparkline areas)
    PANELS = [
        # (label, x0, z0, x1, z1, header_color, data_color)
        ("LEFT",  -7.6, -4.10, -4.4, -0.8, mat_acc, mat_acc),
        ("RIGHT",  4.4, -4.10,  7.6, -0.8, mat_cya, mat_cya),
    ]

    def _rule(name, x0, z, x1, mat, bevel=0.0025):
        pts = [(x0, 0, z, 1.0), (x1, 0, z, 1.0)]
        new_curve_obj(name, [pts], location=(0, D, 0), bevel=bevel, material=mat)

    def _label(name, body, x, z, size, mat, align="LEFT"):
        bpy.ops.object.text_add(location=(x, D, z))
        obj = bpy.context.object
        obj.name = name
        obj.data.body    = body
        obj.data.size    = size
        obj.data.font    = load_font()
        obj.data.align_x = align
        obj.data.materials.append(mat)
        obj.rotation_euler = mathutils.Euler((math.radians(90), 0, 0), "XYZ")

    for side, x0, z0, x1, z1, hcol, dcol in PANELS:
        mid_z = z1 - (z1 - z0) * 0.195   # header strip height ≈ 19% of panel

        # ── Header accent rule (thick, top of panel) ───────────────────
        _rule(f"Pnl_{side}_HdrRule",   x0 + 0.04, z1, x1 - 0.04, hcol, bevel=0.005)

        # ── Sub-divider at ~20% from top (separates header from chart) ─
        _rule(f"Pnl_{side}_DivRule",   x0 + 0.04, mid_z, x1 - 0.04, mat_dim, bevel=0.0015)

        # ── Y-axis tick marks (4 ticks inside the chart area) ──────────
        chart_z0 = z0 + 0.12
        chart_z1 = mid_z - 0.06
        for ti in range(5):
            tz = chart_z0 + (chart_z1 - chart_z0) * (ti / 4)
            tick_x = x0 + 0.08
            pts = [(tick_x, 0, tz, 1.0), (tick_x + 0.18, 0, tz, 1.0)]
            new_curve_obj(f"Pnl_{side}_Tick{ti}", [pts],
                          location=(0, D, 0), bevel=0.0012, material=mat_dim)

        # ── Corner cut marks (45° notch at top-left and top-right) ─────
        notch = 0.22
        for xc, xd in [(x0, +1), (x1, -1)]:
            pts = [(xc + xd * notch, 0, z1, 1.0),
                   (xc,              0, z1, 1.0),
                   (xc,              0, z1 - notch, 1.0)]
            new_curve_obj(f"Pnl_{side}_Notch_{xc:.0f}", [pts],
                          location=(0, D, 0), bevel=0.0025, material=hcol)

        # ── Header text (inside header strip) ──────────────────────────
        label_text = ("NEURAL SYNC" if side == "LEFT" else "SIGNAL TRACE")
        _label(f"Pnl_{side}_Title", label_text,
               x0 + 0.14, z1 - 0.10, 0.155, mat_wht)

        # ── Sub-header: channel descriptor ─────────────────────────────
        sub_text = ("CH:04  LAYER_4  D.SYNC" if side == "LEFT"
                    else "CH:11  RF BAND  D.LOCK")
        _label(f"Pnl_{side}_Sub", sub_text,
               x0 + 0.14, mid_z + 0.04, 0.10, dcol)


# ─────────────────────────────────────────────────────────────────────────────
# 5d. Diegetic annotation markers — circle+cross on model, leader → callout
# ─────────────────────────────────────────────────────────────────────────────

def create_diegetic_markers():
    """
    Technical-drawing annotation overlay:
      · Circle + cross marker at precise model-geometry points
        (torus ring perimeter, ico node, inner core)
      · Thin angled leader line from marker to a horizontal arm
      · Small text callout alongside the arm (label box + dimension text)
    All geometry lives at DEPTH_HUD - 0.05 (in front of HUD panels).
    Colour rule: marker = ACCENT_CYA, arm = ACCENT_CYA, label = ACCENT_YEL.
    Second set of markers in ACCENT_ORG for orange-coded annotations.
    """
    D = DEPTH_HUD - 0.05
    mat_mk  = mat_flat("mat_ann_cya",  ACCENT_CYA)                  # cyan marker + leader
    mat_org = mat_flat("mat_ann_org",  ACCENT_ORG)                  # orange alt marker
    mat_lbl = mat_flat("mat_ann_lbl",  ACCENT_YEL)                  # yellow callout box
    mat_mag = mat_flat("mat_ann_mag",  ACCENT_MAG)                  # magenta box rule

    def _circle_cross(prefix, cx, cz, r, mat, n_pts=16):
        """GP3 circle + centre cross at (cx, cz) in the XZ plane."""
        gp_obj, gpd = new_gp_obj(prefix + "_gp", location=(0, D, 0))
        gpd.materials.append(mat)
        layer = gpd.layers.new("ann", set_active=True)
        fr    = layer.frames.new(1)
        d     = fr.drawing
        # Circle
        circle = [(cx + r * math.cos(2 * math.pi * i / n_pts), 0,
                   cz + r * math.sin(2 * math.pi * i / n_pts))
                  for i in range(n_pts + 1)]
        gp_stroke(d, circle, mat_idx=0, radius=0.004)
        # Inner cross (arms = 60 % of r)
        arm = r * 0.60
        gp_stroke(d, [(cx - arm, 0, cz), (cx + arm, 0, cz)], mat_idx=0, radius=0.002)
        gp_stroke(d, [(cx, 0, cz - arm), (cx, 0, cz + arm)], mat_idx=0, radius=0.002)
        return gp_obj

    def _leader(prefix, x0, z0, xm, zm, xend, arm_len, mat):
        """
        Angled leader:  (x0,z0) → elbow (xm,zm) → horizontal arm to (xend,zm).
        arm_len > 0 goes right, < 0 goes left.
        """
        # Angled segment from marker to elbow
        pts_a = [(x0, 0, z0, 1.0), (xm, 0, zm, 1.0)]
        new_curve_obj(prefix + "_seg", [pts_a],
                      location=(0, D, 0), bevel=0.0015, material=mat)
        # Horizontal arm at elbow
        pts_b = [(xm, 0, zm, 1.0), (xm + arm_len, 0, zm, 1.0)]
        new_curve_obj(prefix + "_arm", [pts_b],
                      location=(0, D, 0), bevel=0.0015, material=mat)

    def _callout_text(name, body, x, z, size=0.13, align="LEFT", mat=None):
        bpy.ops.object.text_add(location=(x, D, z))
        obj = bpy.context.object
        obj.name = name
        obj.data.body    = body
        obj.data.size    = size
        obj.data.font    = load_font()
        obj.data.align_x = align
        obj.data.materials.append(mat or mat_lbl)
        obj.rotation_euler = mathutils.Euler((math.radians(90), 0, 0), "XYZ")

    def _callout_box(prefix, x0, z0, x1, z1, mat):
        """Thin rectangle around a callout text block."""
        px = 0.04;  pz = 0.03
        pts = [(x0 - px, 0, z0 - pz, 1.0), (x1 + px, 0, z0 - pz, 1.0),
               (x1 + px, 0, z1 + pz, 1.0), (x0 - px, 0, z1 + pz, 1.0),
               (x0 - px, 0, z0 - pz, 1.0)]
        new_curve_obj(prefix + "_box", [pts],
                      location=(0, D, 0), bevel=0.0015, material=mat)

    # ══ Annotation A — Torus ring top  (world XZ: 0, +3.2) ════════════════
    _circle_cross("Ann_A", 0.0, 3.2, r=0.16, mat=mat_mk)
    _leader("Ann_A",  0.0, 3.2,   -2.0, 3.65,  -2.0, -1.10, mat_mk)
    _callout_box("Ann_A",  -3.15, 3.58, -2.02, 3.58 + 0.15, mat_mag)
    _callout_text("Ann_A_txt", "RING.R = 3.200u",  -3.10, 3.60, 0.13, mat=mat_lbl)

    # ══ Annotation B — Torus ring right  (world XZ: +3.2, 0) ══════════════
    _circle_cross("Ann_B", 3.2, 0.0, r=0.16, mat=mat_mk)
    _leader("Ann_B",  3.2, 0.0,   3.65, -1.40,  3.65, 0.90, mat_mk)
    _callout_box("Ann_B",   3.62, -1.50, 3.62 + 1.25, -1.50 + 0.15, mat_mag)
    _callout_text("Ann_B_txt", "ROT.W = 45 deg/f", 3.68, -1.47, 0.13, mat=mat_lbl)

    # ══ Annotation C — Ico node upper-right  (~+1.13, +1.13) ══════════════
    _circle_cross("Ann_C", 1.13, 1.13, r=0.14, mat=mat_org)
    _leader("Ann_C",  1.13, 1.13,  2.60, 2.10,  2.60, -0.90, mat_org)
    _callout_box("Ann_C",   1.67, 2.02, 1.67 + 1.10, 2.02 + 0.15, mat_org)
    _callout_text("Ann_C_txt", "NODE.D01",  1.73, 2.05, 0.13, mat=mat_org)

    # ══ Annotation D — Inner core centre  (0, 0) ══════════════════════════
    _circle_cross("Ann_D", 0.0, 0.0, r=0.10, mat=mat_org)
    _leader("Ann_D",  0.0, 0.0,  -2.20, -1.60,  -2.20, -1.05, mat_org)
    _callout_box("Ann_D",  -3.30, -1.68, -3.30 + 1.18, -1.68 + 0.15, mat_org)
    _callout_text("Ann_D_txt", "CORE.D=0.80u", -3.24, -1.65, 0.13, mat=mat_org)

    # ══ Dimension line — torus diameter (horizontal, across centre) ════════
    # Double-headed dimension line at Z = +3.55 between −3.2 and +3.2
    dim_mat = mat_flat("mat_ann_dim", (*ACCENT_CYA[:3], 0.70))
    dim_pts = [(-3.2, 0, 3.55, 1.0), (3.2, 0, 3.55, 1.0)]
    new_curve_obj("Ann_DimLine", [dim_pts],
                  location=(0, D, 0), bevel=0.0015, material=dim_mat)
    # Tick caps at each end
    for xc in (-3.2, 3.2):
        tick = [(xc, 0, 3.50, 1.0), (xc, 0, 3.60, 1.0)]
        new_curve_obj(f"Ann_DimTick_{xc:.0f}", [tick],
                      location=(0, D, 0), bevel=0.0015, material=dim_mat)
    # Dimension label above line
    _callout_text("Ann_Dim_txt", "DIA 6.400u", 0.0, 3.61, 0.13, align="CENTER", mat=dim_mat)

    # ══ Drop-line grid: vertical projection lines from ring to Z=0 axis ════
    # Two dashed reference drop lines from ring top/bottom to the axis
    for xc, name in [(0.0, "DropV"), (3.2, "DropH")]:
        ref_mat = mat_flat(f"mat_drop_{name}", (*ACCENT_CYA[:3], 0.22))
        pts_d = [(xc, 0, -3.2, 1.0), (xc, 0, +3.2, 1.0)]
        new_curve_obj(f"Ann_{name}_Drop", [pts_d],
                      location=(0, D + 0.05, 0), bevel=0.001, material=ref_mat)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Safe-zone border  (Curve object — thicker than GP grid variant)
# ─────────────────────────────────────────────────────────────────────────────

def create_safe_zone_curve():
    mat_sz = mat_flat("mat_safe_zone", (*ACCENT_ORG[:3], 0.7))  # orange border
    mg  = 0.333
    mx, mz = HALF_W - mg, HALF_H - mg

    pts = [(-mx,  0, -mz, 1.0),
           ( mx,  0, -mz, 1.0),
           ( mx,  0,  mz, 1.0),
           (-mx,  0,  mz, 1.0),
           (-mx,  0, -mz, 1.0)]    # close manually (POLY closed)

    cd = bpy.data.curves.new("SafeZone", type="CURVE")
    cd.dimensions   = "3D"
    cd.bevel_depth  = 0.002
    cd.bevel_resolution = 0
    spl = cd.splines.new("POLY")
    spl.points.add(len(pts) - 1)
    for j, co in enumerate(pts):
        spl.points[j].co = co
    spl.use_cyclic_u = True

    obj = bpy.data.objects.new("SafeZone", cd)
    bpy.context.collection.objects.link(obj)
    obj.location = mathutils.Vector((0, DEPTH_BG, 0))
    cd.materials.append(mat_sz)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# 6. HUD panel borders  (Curve objects)
# ─────────────────────────────────────────────────────────────────────────────

def _rect_curve_pts(x0, z0, x1, z1):
    """Return POLY points for a closed rectangle (in XZ, Y=0)."""
    return [(x0, 0, z0, 1), (x1, 0, z0, 1),
            (x1, 0, z1, 1), (x0, 0, z1, 1),
            (x0, 0, z0, 1)]


def create_hud_panels():
    """
    Two HUD panel rectangles (Curve objects) on the left and right flanks.
    Left  panel: neural-sync sparkline  (X: -7.6 → -4.4),  (Z: -4.1 → -0.8)
    Right panel: signal-trace sparkline (X: +4.4 → +7.6),  (Z: -4.1 → -0.8)
    """
    mat_p = mat_flat("mat_hud_panel", (*ACCENT_MAG[:3], 0.9))  # magenta borders

    panels = [
        ("HUD_PanelLeft",  -7.6, -4.1, -4.4, -0.8),
        ("HUD_PanelRight",  4.4, -4.1,  7.6, -0.8),
    ]
    objs = []
    for name, x0, z0, x1, z1 in panels:
        pts = _rect_curve_pts(x0, z0, x1, z1)
        obj = new_curve_obj(name, [pts],
                            location=(0, DEPTH_HUD, 0),
                            bevel=0.004, material=mat_p)
        objs.append(obj)
    return objs


# ─────────────────────────────────────────────────────────────────────────────
# 7. Line Art GP modifier  (icosphere edge trace in cyan)
# ─────────────────────────────────────────────────────────────────────────────

def create_lineart_gp(source_obj):
    """
    GP3 object with LINEART modifier tracing source_obj's edges.
    In Blender 5.0 the modifier type string for the unified GP modifier
    stack is 'LINEART' — added via obj.modifiers.new().
    """
    obj, gpd = new_gp_obj("GP_LineArt", location=(0, DEPTH_GEO, 0))

    m_la = gp_mat("gp_la_mag", (*ACCENT_MAG[:3], 1.0))  # magenta lineart trace
    gpd.materials.append(m_la)

    layer = gpd.layers.new("LA", set_active=True)
    layer.frames.new(1)   # Line Art modifier populates this

    mod = obj.modifiers.new("LineArt", "LINEART")
    if hasattr(mod, "source_type"):
        mod.source_type = "OBJECT"
    if hasattr(mod, "source_object") and source_obj is not None:
        mod.source_object = source_obj
    if hasattr(mod, "target_layer"):
        mod.target_layer = layer.name
    if hasattr(mod, "target_material"):
        mod.target_material = m_la
    if hasattr(mod, "thickness"):
        mod.thickness = 30
    if hasattr(mod, "stroke_depth_offset"):
        mod.stroke_depth_offset = 0.002

    return obj


# ─────────────────────────────────────────────────────────────────────────────
# 8. Machine-vision scan overlay  (GP3, parented to icosphere, per-frame data)
# ─────────────────────────────────────────────────────────────────────────────

def _mv_frame_strokes(drawing, frm, mats_count=2):
    """Write one frame's worth of machine-vision strokes to `drawing`."""
    box_r = 2.2 + 0.08 * math.sin(frm * 0.22)
    arm   = box_r * 0.26

    # Corner L-brackets (4 corners × 3-point polyline)
    dirs = [(+1, +1), (-1, +1), (-1, -1), (+1, -1)]
    for (dx, dz) in dirs:
        cx, cz = dx * box_r, dz * box_r
        gp_stroke(drawing,
                  [(cx + dx * arm, 0, cz),
                   (cx,            0, cz),
                   (cx,            0, cz + dz * arm)],
                  mat_idx=0, radius=0.008)

    # Horizontal scan sweep
    sweep_z = box_r * math.cos(frm * 0.19)
    gp_stroke(drawing,
              [(-box_r, 0, sweep_z), (box_r, 0, sweep_z)],
              mat_idx=1, radius=0.004)

    # Centre gap crosshair (two pairs of segments with gap)
    gap = 0.18
    far = 0.55
    gp_stroke(drawing, [(-far, 0, 0), (-gap, 0, 0)], mat_idx=0, radius=0.004)
    gp_stroke(drawing, [( gap, 0, 0), ( far, 0, 0)], mat_idx=0, radius=0.004)
    gp_stroke(drawing, [(0, 0, -far), (0, 0, -gap)], mat_idx=0, radius=0.004)
    gp_stroke(drawing, [(0, 0,  gap), (0, 0,  far)], mat_idx=0, radius=0.004)

    # Telemetry dashed line from centre toward lower HUD panel
    gp_stroke(drawing, [(0, 0, 0), (3.2, 0, -2.8)], mat_idx=1, radius=0.0025)
    # Confidence annotation tick mark at end of telemetry line
    gp_stroke(drawing, [(2.9, 0, -2.6), (3.5, 0, -3.0)], mat_idx=0, radius=0.004)


def create_machine_vision_gp(target_obj, n_frames):
    """
    GP3 object with per-frame drawings parented to target_obj.
    Strokes are in target_obj local space (centred on ico).
    The GP object is placed at Y=0 (same depth as ico) so the parent
    transform is identity — strokes render in front of the 3D geometry.
    """
    obj, gpd = new_gp_obj("GP_MachineVision", location=(0, 0, 0))
    obj.parent = target_obj
    obj.parent_type = "OBJECT"

    m_yel = gp_mat("gp_mv_yel", (*ACCENT_YEL[:3], 1.0))  # yellow brackets/crosshair
    m_cya = gp_mat("gp_mv_cya", (*ACCENT_CYA[:3], 0.90))  # cyan sweep line
    gpd.materials.append(m_yel)   # index 0
    gpd.materials.append(m_cya)   # index 1

    layer = gpd.layers.new("MV", set_active=True)

    for frm in range(1, n_frames + 1):
        gframe = layer.frames.new(frm)
        _mv_frame_strokes(gframe.drawing, frm)

    print(f"[v2] MV overlay: {n_frames} GP frames, ~7 strokes each")
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# 9. HUD sparkline panels  (GP3, pre-computed per-frame)
# ─────────────────────────────────────────────────────────────────────────────

def _gen_sparkline_data(n_frames, seed=7):
    """
    Generate two channels of pseudo-telemetry data for the sparkline panels.
    Returns (neural_vals, signal_vals) each of length n_frames,
    normalised to [0, 1].
    """
    rng = random.Random(seed)
    neural = [0.5]
    signal = [0.4]
    for _ in range(n_frames - 1):
        neural.append(max(0.0, min(1.0, neural[-1] + rng.uniform(-0.18, 0.18))))
        # Signal: occasional step drops/spikes for SNAP stepped graph
        raw = signal[-1] + rng.choice([-0.25, 0.0, 0.0, 0.25, 0.5, -0.5]) * rng.random()
        signal.append(max(0.0, min(1.0, raw)))
    return neural, signal


def _draw_sparkline(drawing, x0, z0, x1, z1, values, mat_idx, step_mode=False):
    """
    Draw a sparkline (line graph) within the 2D bounding box [x0,x1]×[z0,z1].
    In step_mode: each sample is a horizontal segment at integer-snapped height.
    values: list of normalised [0,1] floats.
    """
    n  = len(values)
    pw = x1 - x0    # panel width
    ph = z1 - z0    # panel height
    pad_x = pw * 0.05
    pad_z = ph * 0.08

    if step_mode:
        # SNAP stepped graph: each value floor-snapped to 8 levels
        levels = 8
        pts = []
        for i, v in enumerate(values):
            snap_v = math.floor(v * levels) / levels
            xi = x0 + pad_x + (pw - 2 * pad_x) * (i / (n - 1))
            zi = z0 + pad_z + (ph - 2 * pad_z) * snap_v
            if pts and abs(zi - pts[-1][2]) > 0.001:
                # Vertical jump: insert intermediate point at previous Z then new Z
                x_prev = pts[-1][0]
                pts.append((x_prev, 0, zi))
            pts.append((xi, 0, zi))
        gp_stroke(drawing, pts, mat_idx=mat_idx, radius=0.003)
    else:
        # Smooth line graph
        pts = []
        for i, v in enumerate(values):
            xi = x0 + pad_x + (pw - 2 * pad_x) * (i / (n - 1))
            zi = z0 + pad_z + (ph - 2 * pad_z) * v
            pts.append((xi, 0, zi))
        gp_stroke(drawing, pts, mat_idx=mat_idx, radius=0.003)


def _draw_bargraph(drawing, x0, z0, x1, z1, values, mat_idx):
    """Draw a bar graph: one vertical stroke per bar."""
    n   = len(values)
    pw  = x1 - x0
    ph  = z1 - z0
    pad_x = pw * 0.06
    slot  = (pw - 2 * pad_x) / n
    bw    = slot * 0.55     # bar width as radius
    for i, v in enumerate(values):
        xi  = x0 + pad_x + slot * (i + 0.5)
        z_t = z0 + ph * 0.05 + (ph * 0.90) * v
        gp_stroke(drawing,
                  [(xi, 0, z0 + ph * 0.03), (xi, 0, z_t)],
                  mat_idx=mat_idx, radius=max(0.0018, bw * 0.35))


def create_sparkline_panels_gp(n_frames):
    """
    Two GP3 objects (left: neural bar graph, right: signal step graph)
    each with one GP frame per render frame containing the sparkline strokes.

    Panel coordinate notes:
      Objects placed at Y=DEPTH_HUD.
      Stroke points at local (x, 0, z) → world (x, DEPTH_HUD, z).
    """
    neural_all, signal_all = _gen_sparkline_data(n_frames)

    # ── Left panel: neural sync bar graph ──────────────────────────────────
    obj_l, gpd_l = new_gp_obj("GP_SparkLeft", location=(0, DEPTH_HUD, 0))
    m_bar = gp_mat("gp_bar_yel",  (*ACCENT_YEL[:3], 1.0))   # solid yellow bars
    m_dim = gp_mat("gp_bar_dim",  (*ACCENT_ORG[:3], 0.55))   # dim orange ghost bars
    gpd_l.materials.append(m_bar)   # 0
    gpd_l.materials.append(m_dim)   # 1

    # Panel inner area — must match create_hud_panels() border rect exactly
    LPX0, LPZ0, LPX1, LPZ1 = -7.4, -3.9, -4.6, -1.0

    layer_l = gpd_l.layers.new("SL", set_active=True)

    for frm in range(1, n_frames + 1):
        # Show last 8 frames of values
        hist = neural_all[max(0, frm - 8): frm]
        gframe = layer_l.frames.new(frm)
        _draw_bargraph(gframe.drawing, LPX0, LPZ0, LPX1, LPZ1, hist, mat_idx=0)

    print(f"[v2] Sparkline left: {n_frames} GP frames")

    # ── Right panel: signal trace step graph ───────────────────────────────
    obj_r, gpd_r = new_gp_obj("GP_SparkRight", location=(0, DEPTH_HUD, 0))
    m_sig  = gp_mat("gp_sig_cya",  (*ACCENT_CYA[:3], 1.0))   # solid cyan step graph
    gpd_r.materials.append(m_sig)   # 0

    # Show last 24 frames of values — matches GRID_COLS for nice alignment
    RPX0, RPZ0, RPX1, RPZ1 = 4.6, -3.9, 7.4, -1.0

    layer_r = gpd_r.layers.new("SR", set_active=True)

    for frm in range(1, n_frames + 1):
        hist = signal_all[max(0, frm - 24): frm]
        # Pad to 24 if shorter (start of animation)
        while len(hist) < 24:
            hist = [hist[0]] + list(hist)
        gframe = layer_r.frames.new(frm)
        _draw_sparkline(gframe.drawing, RPX0, RPZ0, RPX1, RPZ1,
                        hist, mat_idx=0, step_mode=True)

    print(f"[v2] Sparkline right: {n_frames} GP frames")
    return obj_l, obj_r


# ─────────────────────────────────────────────────────────────────────────────
# 10. Telemetry TextCurve objects  (most static; dynamic ones via handler)
# ─────────────────────────────────────────────────────────────────────────────

def create_telemetry_text(n_frames):
    """
    TextCurve objects placed off-centre for the asymmetric grid layout.
    Static labels flash on/off with CONSTANT keyframes.
    Four "live" objects (frame counter, timestamp, hex hash, status) are
    later updated by the frame_change_pre handler.
    """
    mat_sec  = mat_flat("mat_txt_sec",  ACCENT_YEL)   # primary text = yellow
    mat_acc  = mat_flat("mat_txt_acc",  ACCENT_YEL)   # accent text = yellow
    mat_org  = mat_flat("mat_txt_org",  ACCENT_ORG)   # identifiers = orange
    mat_cya  = mat_flat("mat_txt_cya",  ACCENT_CYA)   # live data = cyan
    mat_red  = mat_flat("mat_txt_red",  ACCENT_RED)   # warnings = red

    # Tuples: (text, (world_X, world_Z_vertical, _unused), size, mat)
    # ALWAYS_ON entries never flash — they serve as persistent readouts.
    ALWAYS_ON = [
        # Left-column readouts
        ("[SYS_AUTH: ACTIVE]",  (-7.5, +3.20, 0), 0.17, mat_cya),   # cyan for status
        ("NEURAL SYNC",         (-7.5, -0.65, 0), 0.17, mat_sec),   # yellow panel label
        ("SIGNAL TRACE",        (+4.5, -0.65, 0), 0.17, mat_sec),   # yellow panel label
        # Right column
        (">>> DATA_STREAM",     (+4.5, +3.20, 0), 0.17, mat_cya),   # cyan
    ]

    # Flashing / intermittent readouts
    STATIC_STRINGS = [
        ("RUNNER_V4.1",         (+4.5, -2.85, 0), 0.20, mat_org),   # orange ID
        ("[STATUS: LOCKED]",    (+4.5, -3.35, 0), 0.17, mat_sec),   # yellow
        ("TRJ_8841 OK",         (+4.5, -3.75, 0), 0.15, mat_cya),   # cyan result
        ("SYS_BUS  082",        (-7.5, -3.25, 0), 0.17, mat_sec),   # yellow
        ("CHK: 0xE9FF23",       (+4.5, +2.65, 0), 0.15, mat_org),   # orange checksum
        ("!WARN SECTOR-7",      (-7.5, -2.55, 0), 0.17, mat_red),   # red alert
        ("VEL: 0.943c",         (-7.5, -2.95, 0), 0.15, mat_cya),   # cyan readout
    ]

    rng = random.Random(77)
    static_objs = []

    # Always-on labels — no keyframe animation
    for label, (lx, lz, _), size, mat in ALWAYS_ON:
        bpy.ops.object.text_add(location=(lx, DEPTH_HUD, lz))
        obj = bpy.context.object
        obj.name = f"Txt_{label[:12].strip()}"
        obj.data.body    = label
        obj.data.size    = size
        obj.data.font    = load_font()
        obj.data.align_x = "LEFT"
        obj.data.materials.append(mat)
        obj.rotation_euler = mathutils.Euler((math.radians(90), 0, 0), "XYZ")
        static_objs.append(obj)

    for label, (lx, lz, _), size, mat in STATIC_STRINGS:
        # lz = world Z (vertical); DEPTH_HUD = world Y (depth)
        bpy.ops.object.text_add(location=(lx, DEPTH_HUD, lz))
        obj = bpy.context.object
        obj.name = f"Txt_{label[:10]}"
        obj.data.body    = label
        obj.data.size    = size
        obj.data.font    = load_font()
        obj.data.align_x = "LEFT"
        obj.data.materials.append(mat)
        obj.rotation_euler = mathutils.Euler((math.radians(90), 0, 0), "XYZ")

        # Flash pattern — slow enough to be readable (~0.5–1 s visible)
        frame = 1
        visible = True
        while frame <= n_frames:
            obj.hide_render   = not visible
            obj.hide_viewport = not visible
            obj.keyframe_insert("hide_render",   frame=frame)
            obj.keyframe_insert("hide_viewport", frame=frame)
            set_fcurve_interp(obj, "hide_render")
            set_fcurve_interp(obj, "hide_viewport")
            duration = rng.randint(15, 30) if visible else rng.randint(8, 18)
            frame   += duration
            visible  = not visible

        static_objs.append(obj)

    # ── Live / dynamic labels ───────────────────────────────────────────────
    LIVE_SPECS = [
        ("Txt_LiveFrame",  (-7.5, +2.60, 0), 0.15, mat_org,  "FRM: 000"),          # orange
        ("Txt_LiveStamp",  (-7.5, +2.20, 0), 0.15, mat_cya,  "UTC 2093-03-11T04:22:00Z"),  # cyan
        ("Txt_LiveHash",   (-7.5, -3.75, 0), 0.15, mat_sec,  "0xFA39B2D7"),        # yellow
        ("Txt_LiveStatus", (-1.8, +3.30, 0), 0.17, mat_cya,  ">>> ACQUIRING TARGET"),  # cyan
    ]

    live_objs = {}
    for name, (lx, lz, _), size, mat, body in LIVE_SPECS:
        # lz = world Z (vertical)
        bpy.ops.object.text_add(location=(lx, DEPTH_HUD, lz))
        obj = bpy.context.object
        obj.name = name
        obj.data.body    = body
        obj.data.size    = size
        obj.data.font    = load_font()
        obj.data.align_x = "LEFT"
        obj.data.materials.append(mat)
        obj.rotation_euler = mathutils.Euler((math.radians(90), 0, 0), "XYZ")
        live_objs[name] = obj

    return static_objs, live_objs


# ─────────────────────────────────────────────────────────────────────────────
# 11. frame_change_pre handler  (updates live text each frame)
# ─────────────────────────────────────────────────────────────────────────────

_LIVE_OBJS   = {}   # populated in main(), read by handler
_STATUS_MSGS = [
    ">>> ACQUIRING TARGET",
    ">>> TARGET LOCKED",
    ">>> VELOCITY CALC",
    ">>> INTERCEPT PLOT",
    ">>> TRAJECTORY OK",
    ">>> DATALINK SYNC",
    ">>> SIGNAL CONFIRM",
    ">>> BUFFER FLUSH",
    ">>> STANDBY",
]
_HEX_RNG = random.Random(99)


def _frame_handler(scene):
    """Called before each frame is rendered."""
    global _LIVE_OBJS
    if not _LIVE_OBJS:
        return
    frm = scene.frame_current

    def _set(name, body):
        obj = _LIVE_OBJS.get(name)
        if obj:
            obj.data.body = body

    # Frame counter
    _set("Txt_LiveFrame",  f"FRM: {frm:03d}")

    # UTC timestamp (fake, evolving)
    secs = 22 * 3600 + frm * 2
    h, rem = divmod(secs, 3600)
    m, s   = divmod(rem, 60)
    _set("Txt_LiveStamp", f"UTC 2093-03-11T{h:02d}:{m:02d}:{s:02d}Z")

    # Hash value — flicker every ~9 frames for legibility
    if frm % 9 == 0:
        _set("Txt_LiveHash", f"0x{_HEX_RNG.randint(0, 0xFFFFFFFF):08X}")

    # Status message
    idx = (frm // 7) % len(_STATUS_MSGS)
    _set("Txt_LiveStatus", _STATUS_MSGS[idx])


def register_frame_handler(live_objs):
    global _LIVE_OBJS
    _LIVE_OBJS = live_objs
    # Remove duplicate registrations
    handlers = bpy.app.handlers.frame_change_pre
    for h in list(handlers):
        if getattr(h, "__name__", "") == "_frame_handler":
            handlers.remove(h)
    handlers.append(_frame_handler)
    print("[v2] frame_change_pre handler registered")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    n    = args.frames

    print("[v2] Purging scene …")
    purge_scene()

    print("[v2] Setting up scene + camera …")
    setup_scene(args)
    setup_camera()

    print("[v2] Building background GP grid …")
    create_gp_grid()
    create_gp_fiducials()
    create_safe_zone_curve()

    print("[v2] Building 3D focal geometry …")
    torus, ico, inner = create_focal_geometry(n)

    print("[v2] Building targeting brackets …")
    create_targeting_brackets(scale=4.0)

    print("[v2] Building sidebar labels …")
    create_sidebar_labels()

    print("[v2] Building HUD panel frames …")
    create_hud_panels()
    create_panel_decorations()

    print("[v2] Building diegetic annotation markers …")
    create_diegetic_markers()

    print("[v2] Building Line Art GP modifier …")
    create_lineart_gp(ico)

    print("[v2] Pre-computing machine-vision GP overlay …")
    create_machine_vision_gp(ico, n)

    print("[v2] Pre-computing sparkline panels …")
    create_sparkline_panels_gp(n)

    print("[v2] Building telemetry text …")
    _, live_objs = create_telemetry_text(n)

    # Canonical consistency checks (palette + layout) via Cassowary solver.
    screen_w = args.width
    screen_h = args.height
    ensure_design_consistency(screen_w, screen_h)

    # Compute a reasonable target_rect in screen pixels (centered on screen).
    target_rect = {"x": screen_w / 2.0, "y": screen_h / 2.0, "w": 200, "h": 200}

    try:
        ui_coords = calculate_ui_layout(target_rect, screen_w, screen_h)
        # Map pixel centers to world X/Z coordinates (camera ortho mapping)
        def pixel_to_world(px, py):
            # px,py: pixel coords with origin at top-left. Convert to normalized [-1,+1]
            nx = (px / screen_w - 0.5) * 2.0
            ny = (0.5 - py / screen_h) * 2.0
            wx = nx * HALF_W
            wz = ny * HALF_H
            return wx, wz

        # Map layout roles to the live object names
        mapping = {
            'text_1': 'Txt_LiveStatus',
            'text_2': 'Txt_LiveHash',
            'text_3': 'Txt_LiveFrame',
            'text_4': 'Txt_LiveStamp',
        }
        for key, name in mapping.items():
            if name in live_objs and key in ui_coords:
                px, py = ui_coords[key]
                wx, wz = pixel_to_world(px, py)
                obj = live_objs[name]
                obj.location.x = wx
                # Blender text objects in this script use X(world), DEPTH_HUD (Y), Z(world)
                obj.location.z = wz
    except Exception as exc:
        print(f"[v2] calculate_ui_layout failed: {exc}")

    register_frame_handler(live_objs)

    print(f"[v2] Rendering {n} frames → {args.output}")
    bpy.ops.render.render(animation=True)
    print("[v2] Done.")


if __name__ == "__main__":
    main()
