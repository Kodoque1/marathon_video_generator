"""
Marathon Reboot — Blender 3D Layer
===================================
Run headless:
    blender --background --python blender_scene.py -- \
            --output /path/to/blender_frames/ \
            --frames 90 \
            --width 1920 --height 1080

Produces a PNG sequence with transparent background (RGBA) so it can be
composited over the pycairo HUD layer by composite.py.

Requires Blender 4.x.  The script gracefully skips unavailable features
(e.g. Bloom moved to compositor in 4.0).
"""

import sys
import math
import argparse

import bpy
import mathutils

# ─────────────────────────────────────────────────────────────────────────────
# CLI args (everything after " -- " goes to this script, not to Blender)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--output",  default="/tmp/blender_frames/", help="Output dir for PNG frames")
    p.add_argument("--frames",  type=int, default=90,   help="Total frames to render (3s @ 30fps)")
    p.add_argument("--width",   type=int, default=1920)
    p.add_argument("--height",  type=int, default=1080)
    p.add_argument("--fps",     type=int, default=30)
    return p.parse_args(argv)

# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────

BASE        = (0.043, 0.047, 0.059, 1.0)   # #0B0C0F
SECONDARY   = (0.933, 0.933, 0.933, 1.0)   # #EEEEEE
ACCENT_YEL  = (0.914, 1.000, 0.137, 1.0)   # #E9FF23  (acid yellow)
ACCENT_CYA  = (0.000, 1.000, 0.898, 1.0)   # #00FFE5  (cyan)
ACCENT_ORG  = (1.000, 0.400, 0.000, 1.0)   # #FF6600  (safety orange trim)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def purge_scene():
    """Remove all default objects."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.meshes) + list(bpy.data.curves) + list(bpy.data.lights):
        bpy.data.batch_remove(ids=[block])


def get_action_fcurves(action):
    """Return all FCurves from an action: compatible with Blender ≤4.3 (action.fcurves)
    and Blender 4.4+ (layered slot / channelbag system)."""
    # Legacy API (Blender < 4.4)
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    # Blender 4.4+ layered action API
    curves = []
    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in layer.strips:
                if hasattr(strip, "channelbags"):
                    for cb in strip.channelbags:
                        curves.extend(cb.fcurves)
    return curves


def set_fcurve_constant(obj, data_path, index=-1):
    """Force all keyframes on a data_path to CONSTANT interpolation."""
    if obj.animation_data and obj.animation_data.action:
        for fc in get_action_fcurves(obj.animation_data.action):
            if fc.data_path == data_path and (index == -1 or fc.array_index == index):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"


def set_fcurve_expo(obj, data_path, index=-1):
    """EXPO ease-out for snap-in physical animation."""
    if obj.animation_data and obj.animation_data.action:
        for fc in get_action_fcurves(obj.animation_data.action):
            if fc.data_path == data_path and (index == -1 or fc.array_index == index):
                for kp in fc.keyframe_points:
                    kp.interpolation = "EXPO"
                    kp.easing = "EASE_OUT"


def mat_flat(name: str, color: tuple) -> bpy.types.Material:
    """Shadeless (unlit) flat colour material."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    tree = m.node_tree
    tree.nodes.clear()
    out  = tree.nodes.new("ShaderNodeOutputMaterial")
    em   = tree.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value   = color
    em.inputs["Strength"].default_value = 1.0
    tree.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


def mat_emissive(name: str, color: tuple, strength: float = 6.0) -> bpy.types.Material:
    """Self-illuminated emissive — drives bloom in Eevee."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    tree = m.node_tree
    tree.nodes.clear()
    out  = tree.nodes.new("ShaderNodeOutputMaterial")
    em   = tree.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value    = color
    em.inputs["Strength"].default_value  = strength
    tree.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


def new_obj(name: str, mesh_or_data, location=(0, 0, 0),
            material=None) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, mesh_or_data)
    bpy.context.collection.objects.link(obj)
    obj.location = mathutils.Vector(location)
    if material:
        if obj.data and hasattr(obj.data, "materials"):
            obj.data.materials.append(material)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Scene / render setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_scene(args):
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end   = args.frames
    scene.render.fps  = args.fps

    # Resolution
    scene.render.resolution_x     = args.width
    scene.render.resolution_y     = args.height
    scene.render.resolution_percentage = 100

    # Transparent background for compositing with cairo
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode  = "RGBA"
    scene.render.image_settings.compression = 15

    # Output path (Blender appends #### for frame number)
    import os
    os.makedirs(args.output, exist_ok=True)
    scene.render.filepath = args.output + "frame_"

    # Eevee engine
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"

    eevee = scene.eevee

    # Bloom: available directly in Eevee < 4.0; in 4.x it's a compositor Glare node
    if hasattr(eevee, "use_bloom"):
        eevee.use_bloom               = True
        eevee.bloom_threshold         = 0.5
        eevee.bloom_intensity         = 0.8
        eevee.bloom_radius            = 5.0
        eevee.bloom_color             = (1.0, 1.0, 1.0)

    # These properties were removed in EEVEE Next (Blender 4.x / 5.x)
    if hasattr(eevee, "use_ssr"):
        eevee.use_ssr             = False
    if hasattr(eevee, "use_gtao"):
        eevee.use_gtao            = False
    if hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples  = 16

    # World background — black alpha  (film_transparent covers this,
    # but set it explicitly for viewport previews)
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    scene.world = world


# ─────────────────────────────────────────────────────────────────────────────
# Orthographic camera
# ─────────────────────────────────────────────────────────────────────────────

def setup_camera(args):
    cam_data = bpy.data.cameras.new("OrthoCamera")
    cam_data.type         = "ORTHO"
    # Ortho scale: 1 unit = 1 metre; we set it to cover the visible canvas width
    # At 1080 wide with 1 unit ≈ width in world space:
    cam_data.ortho_scale  = 16.0

    cam = new_obj("Camera", cam_data, location=(0, -10, 0))
    cam.rotation_euler = mathutils.Euler((math.radians(90), 0, 0), "XYZ")
    bpy.context.scene.camera = cam
    return cam


# ─────────────────────────────────────────────────────────────────────────────
# Aesthetic Element 1 — Central 3D focal geometry
# ─────────────────────────────────────────────────────────────────────────────

def create_focal_geometry(n_frames: int):
    """
    A torus (outer frame) + icosphere (inner node) with an emissive accent material.
    Rotates in 45° SNAP increments via EXPO ease-out keyframes.
    """
    mat_torus  = mat_emissive("mat_torus_accent",  ACCENT_YEL, strength=7.0)
    mat_ico    = mat_flat("mat_ico_secondary",     SECONDARY)
    mat_inner  = mat_emissive("mat_inner_emissive", ACCENT_CYA, strength=9.0)

    # ── Torus (ring frame)
    bpy.ops.mesh.primitive_torus_add(
        major_radius=3.2, minor_radius=0.08,
        major_segments=64, minor_segments=12,
        location=(0, 0, 0)
    )
    torus = bpy.context.object
    torus.name = "FocalTorus"
    torus.data.materials.append(mat_torus)

    # ── Icosphere (centrepiece)
    bpy.ops.mesh.primitive_ico_sphere_add(radius=1.6, subdivisions=2, location=(0, 0, 0))
    ico = bpy.context.object
    ico.name = "FocalIco"
    # Subtractive: apply a wireframe modifier to expose the emissive inside
    wire_mod = ico.modifiers.new("Wire", "WIREFRAME")
    wire_mod.thickness = 0.05
    wire_mod.use_replace = True
    ico.data.materials.append(mat_ico)

    # ── Inner glow sphere (very small, pure emissive)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.4, segments=12, ring_count=8, location=(0, 0, 0))
    inner = bpy.context.object
    inner.name = "InnerGlow"
    inner.data.materials.append(mat_inner)

    # ── Snap rotation animation (45° increments, EXPO ease-out)
    # Pattern: hold → sudden snap every ~15 frames
    snap_interval = 15          # frames between snaps
    current_angle = 0.0

    for obj in (torus, ico):
        obj.rotation_euler = mathutils.Euler((0, 0, math.radians(current_angle)), "XYZ")
        obj.keyframe_insert("rotation_euler", frame=1)

        frame = 1
        while frame <= n_frames:
            # Hold pose
            obj.rotation_euler = mathutils.Euler((0, 0, math.radians(current_angle)), "XYZ")
            obj.keyframe_insert("rotation_euler", frame=frame)

            # Snap to next 45° step
            next_frame = frame + snap_interval
            if next_frame > n_frames:
                break
            current_angle += 45.0
            # Insert hold start (current) then immediate arrival (1 frame later)
            obj.rotation_euler = mathutils.Euler((0, 0, math.radians(current_angle)), "XYZ")
            obj.keyframe_insert("rotation_euler", frame=next_frame)
            set_fcurve_expo(obj, "rotation_euler")

            frame = next_frame

    return torus, ico, inner


# ─────────────────────────────────────────────────────────────────────────────
# Aesthetic Element 1b — Targeting brackets (L-shaped curves)
# ─────────────────────────────────────────────────────────────────────────────

def create_targeting_brackets(scale: float = 4.0):
    """
    Four L-shaped 2D curve objects forming corner brackets around the focal geo.
    Each bracket is a NURBS path with 3 control points (two arms + corner).
    """
    mat_bracket = mat_flat("mat_bracket", SECONDARY)
    arm = scale * 0.28

    corners = [
        # (corner_x, corner_y, horizontal_dir, vertical_dir)
        (-scale, +scale, +1, -1),
        (+scale, +scale, -1, -1),
        (-scale, -scale, +1, +1),
        (+scale, -scale, -1, +1),
    ]

    for i, (cx, cy, hd, vd) in enumerate(corners):
        name = f"Bracket_{i}"
        curve_data = bpy.data.curves.new(name, type="CURVE")
        curve_data.dimensions     = "3D"
        curve_data.resolution_u   = 2
        curve_data.bevel_depth    = 0.035      # line thickness
        curve_data.bevel_resolution = 0

        spline = curve_data.splines.new("POLY")
        spline.points.add(2)   # 3 total (already has 1)

        pts = [
            (cx + hd * arm, cy,           0, 1),
            (cx,            cy,           0, 1),
            (cx,            cy + vd * arm, 0, 1),
        ]
        for j, (x, y, z, w) in enumerate(pts):
            spline.points[j].co = (x, y, z, w)

        obj = new_obj(name, curve_data, material=mat_bracket)
        # Slow continuous pulse scale (Y-axis) simulating a radar sweep
        for frame, scale_y in [(1, 0.95), (45, 1.05), (90, 0.95)]:
            obj.scale = (1.0, scale_y, 1.0)
            obj.keyframe_insert("scale", frame=frame)


# ─────────────────────────────────────────────────────────────────────────────
# Aesthetic Element 1c — Fiducial markers on 12×8 grid
# ─────────────────────────────────────────────────────────────────────────────

def create_fiducial_markers():
    """
    Tiny crosshair empties anchored to a 12×8 grid.
    Only occupy the grid nodes OUTSIDE the central focal area.
    """
    mat_cross = mat_emissive("mat_fiducial", ACCENT_YEL, strength=3.0)

    cols, rows = 12, 8
    spacing_x  = 14.0 / cols
    spacing_y  = 9.0  / rows
    cx_range   = range(cols + 1)
    cy_range   = range(rows + 1)

    for gi in cx_range:
        for gj in cy_range:
            wx = -7.0 + gi * spacing_x
            wy = -4.5 + gj * spacing_y

            # Skip grid nodes that fall inside the focal area (radius 4.5)
            if math.sqrt(wx**2 + wy**2) < 4.5:
                continue

            # Crosshair: two perpendicular thin boxes
            arm = 0.18
            thick = 0.025

            for axis, dims, loc in [
                ("H", (arm * 2, thick, thick), (wx, wy, 0)),
                ("V", (thick, arm * 2, thick), (wx, wy, 0)),
            ]:
                bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
                marker = bpy.context.object
                marker.name    = f"Fid_{gi}_{gj}_{axis}"
                marker.scale   = dims
                marker.data.materials.append(mat_cross)


# ─────────────────────────────────────────────────────────────────────────────
# Aesthetic Element 2 — Diegetic debug telemetry (text objects)
# ─────────────────────────────────────────────────────────────────────────────

def create_telemetry_text(n_frames: int):
    """
    Floating text objects that flash CONSTANT on/off to simulate buffer refreshes.
    Placed off-centre per the asymmetric grid layout principle.
    """
    mat_text_sec = mat_flat("mat_text_secondary", SECONDARY)
    mat_text_acc = mat_emissive("mat_text_accent", ACCENT_YEL, strength=2.0)

    STRINGS = [
        ("[SYS_AUTH: ACTIVE]",   (-5.8, +3.6, 0), 0.20, mat_text_sec),
        ("0xFA39B2D7",           (-5.8, +3.0, 0), 0.15, mat_text_acc),
        ("UTC 2093-03-11T04:22Z",(-5.8, +2.5, 0), 0.13, mat_text_sec),
        ("RUNNER_V4.1",          (+4.2, -3.2, 0), 0.22, mat_text_acc),
        ("[STATUS: LOCKED]",     (+4.2, -3.8, 0), 0.15, mat_text_sec),
        ("TRJ_8841 OK",          (+4.2, -4.3, 0), 0.13, mat_text_sec),
        ("SYS_AUTH_082",         (-5.8, -3.5, 0), 0.16, mat_text_sec),
        ("0x003F:FF:AD",         (-5.8, -4.1, 0), 0.13, mat_text_acc),
        (">>> DATA_STREAM",      (+4.2, +3.6, 0), 0.16, mat_text_sec),
        ("CHK: 0xE9FF23",        (+4.2, +3.0, 0), 0.13, mat_text_acc),
    ]

    rng = __import__("random").Random(77)
    text_objs = []

    for label, loc, size, mat in STRINGS:
        bpy.ops.object.text_add(location=loc)
        obj = bpy.context.object
        obj.name = f"Telemetry_{label[:8]}"
        obj.data.body  = label
        obj.data.size  = size
        obj.data.font  = bpy.data.fonts.load("<builtin>")  # system monospace fallback
        obj.data.align_x = "LEFT"
        obj.data.materials.append(mat)
        obj.rotation_euler = mathutils.Euler((math.radians(90), 0, 0), "XYZ")

        # Flash pattern: 3–5 frames on, 4–8 frames off, CONSTANT interpolation
        frame = 1
        visible = True
        while frame <= n_frames:
            obj.hide_render   = not visible
            obj.hide_viewport = not visible
            obj.keyframe_insert("hide_render",   frame=frame)
            obj.keyframe_insert("hide_viewport", frame=frame)
            set_fcurve_constant(obj, "hide_render")
            set_fcurve_constant(obj, "hide_viewport")

            duration = rng.randint(3, 5) if visible else rng.randint(4, 8)
            frame   += duration
            visible  = not visible

        text_objs.append(obj)

    return text_objs


# ─────────────────────────────────────────────────────────────────────────────
# Aesthetic Element 3 — Clinical Glitch Compositor
# ─────────────────────────────────────────────────────────────────────────────

def setup_compositor(n_frames: int):
    """
    Node graph:
      Render → [Bloom Glare] → [Lens Distortion (CA)] → [Pixelate chain] → Composite
    Bloom handled here for Blender 4.x (where eevee.use_bloom was removed).
    CA dispersion and macro-block Factor are keyframed CONSTANT for glitch frames only.
    """
    scene = bpy.context.scene

    # ── Obtain the compositor node tree (API changed across Blender versions) ──
    tree = None
    try:
        # Blender < 4.4 / old path
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scene.use_nodes = True
        tree = scene.node_tree          # raises AttributeError on 5.0+
    except AttributeError:
        pass

    if tree is None:
        # Blender 4.4+ / 5.0: compositor uses a dedicated node group property
        try:
            if (not hasattr(scene, "compositing_node_group")
                    or scene.compositing_node_group is None):
                nt = bpy.data.node_groups.new("Compositor", "CompositorNodeTree")
                scene.compositing_node_group = nt
            tree = scene.compositing_node_group
        except Exception as e:
            print(f"[warn] Could not acquire compositor node tree: {e}")

    if tree is None:
        print("[warn] Compositor setup skipped — post-FX not available in this Blender build")
        return
    tree.nodes.clear()
    tree.links.clear()

    # ── Core nodes ────────────────────────────────────────────────────────────
    n_render   = tree.nodes.new("CompositorNodeRLayers")
    n_render.location = (-600, 0)

    # Bloom / glare — Blender 5.0 uses socket inputs instead of node properties
    n_bloom    = tree.nodes.new("CompositorNodeGlare")
    n_bloom.location = (-300, 100)
    try:
        # Blender 5.0+: type/threshold/quality/strength are socket inputs
        n_bloom.inputs["Type"].default_value      = "Bloom"
        n_bloom.inputs["Threshold"].default_value = 0.55
        n_bloom.inputs["Strength"].default_value  = 0.7
        try:
            n_bloom.inputs["Quality"].default_value = "High"
        except Exception:
            pass
    except KeyError:
        # Blender < 4.4 fallback: node properties
        if hasattr(n_bloom, "glare_type"):
            n_bloom.glare_type = "BLOOM"
        if hasattr(n_bloom, "threshold"):
            n_bloom.threshold  = 0.55
        if hasattr(n_bloom, "mix"):
            n_bloom.mix        = 0.7
        if hasattr(n_bloom, "quality"):
            n_bloom.quality    = "HIGH"

    # Chromatic aberration — node was renamed in Blender 5.0
    ca_type = ("CompositorNodeLensdist"
               if hasattr(bpy.types, "CompositorNodeLensdist")
               else "CompositorNodeLensDistortion")
    n_ca       = tree.nodes.new(ca_type)
    n_ca.inputs["Dispersion"].default_value = 0.0
    n_ca.location = (-300, -100)

    # Macroblock chain: scale down → pixelate → scale back up
    n_scale_dn = tree.nodes.new("CompositorNodeScale")
    # Blender 5.0: space is replaced by inputs["Type"]; default is already Relative
    if hasattr(n_scale_dn, "space"):
        n_scale_dn.space = "RENDER_SIZE"
    else:
        try:
            n_scale_dn.inputs["Type"].default_value = "Relative"
        except Exception:
            pass
    n_scale_dn.inputs["X"].default_value = 0.12
    n_scale_dn.inputs["Y"].default_value = 0.12
    n_scale_dn.location = (0, -200)

    n_pixelate = tree.nodes.new("CompositorNodePixelate")
    n_pixelate.location = (200, -200)

    n_scale_up = tree.nodes.new("CompositorNodeScale")
    if hasattr(n_scale_up, "space"):
        n_scale_up.space = "RENDER_SIZE"
    else:
        try:
            n_scale_up.inputs["Type"].default_value = "Relative"
        except Exception:
            pass
    n_scale_up.inputs["X"].default_value = 1.0 / 0.12
    n_scale_up.inputs["Y"].default_value = 1.0 / 0.12
    n_scale_up.location = (400, -200)

    # Mix / blend: in Blender 5.0, MixRGB and Mix nodes are gone from the
    # compositor. Use AlphaOver as a factor-controlled blend:
    #   Factor=0 → Background (clean), Factor=1 → Foreground (pixelated)
    n_mix = tree.nodes.new("CompositorNodeAlphaOver")
    fac_socket = "Factor"
    n_mix.inputs[fac_socket].default_value = 0.0
    n_mix.location = (600, -100)

    # Output node — in Blender 5.0 compositor node groups use NodeGroupOutput
    # with a socket named "Image" on the group interface.
    try:
        # Blender 4.0+ interface API
        if hasattr(tree, "interface"):
            tree.interface.new_socket("Image", in_out="OUTPUT",
                                      socket_type="NodeSocketColor")
        elif hasattr(tree, "outputs"):
            tree.outputs.new("NodeSocketColor", "Image")
        n_out = tree.nodes.new("NodeGroupOutput")
        out_image_input = "Image"
    except Exception:
        # Fallback: viewer node (always works, results shown in Blender viewport)
        n_out = tree.nodes.new("CompositorNodeViewer")
        out_image_input = "Image"
    n_out.location = (900, 0)

    # ── Links ──────────────────────────────────────────────────────────────────
    tree.links.new(n_render.outputs["Image"],   n_bloom.inputs["Image"])
    tree.links.new(n_bloom.outputs["Image"],    n_ca.inputs["Image"])

    # Pixelate node: Blender 5.0 uses "Color" socket instead of "Image"
    pixelate_in  = "Color" if "Color" in [i.name for i in n_pixelate.inputs]  else "Image"
    pixelate_out = "Color" if "Color" in [o.name for o in n_pixelate.outputs] else "Image"

    tree.links.new(n_ca.outputs["Image"],          n_scale_dn.inputs["Image"])
    tree.links.new(n_scale_dn.outputs["Image"],    n_pixelate.inputs[pixelate_in])
    tree.links.new(n_pixelate.outputs[pixelate_out], n_scale_up.inputs["Image"])

    # AlphaOver: Background=clean, Foreground=pixelated
    tree.links.new(n_ca.outputs["Image"],          n_mix.inputs["Background"])
    tree.links.new(n_scale_up.outputs["Image"],    n_mix.inputs["Foreground"])

    tree.links.new(n_mix.outputs["Image"],         n_out.inputs[out_image_input])

    # ── Animate glitch events ─────────────────────────────────────────────────
    # CA dispersion: normal = 0.0, spikes on specific glitch frames
    # Macro-block factor: normal = 0.0, jumps to 1.0 for 1-2 frames
    rng_g    = __import__("random").Random(13)
    glitch_frames = sorted(rng_g.sample(range(5, n_frames - 2), k=max(1, n_frames // 18)))

    def kf(node, input_name, frame, val, interp="CONSTANT"):
        node.inputs[input_name].default_value = val
        node.inputs[input_name].keyframe_insert("default_value", frame=frame)
        if node.id_data.animation_data and node.id_data.animation_data.action:
            for fc in get_action_fcurves(node.id_data.animation_data.action):
                if "default_value" in fc.data_path:
                    for kp in fc.keyframe_points:
                        kp.interpolation = interp

    # Baseline values
    kf(n_ca,  "Dispersion", 1,          0.0)
    kf(n_mix, fac_socket,   1,          0.0)

    for gf in glitch_frames:
        ca_val  = rng_g.uniform(0.03, 0.09)
        dur     = rng_g.randint(1, 2)
        kf(n_ca,  "Dispersion", gf,       ca_val)
        kf(n_ca,  "Dispersion", gf + dur, 0.0)
        kf(n_mix, fac_socket,   gf,       1.0)
        kf(n_mix, fac_socket,   gf + dur, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    n    = args.frames

    purge_scene()
    setup_scene(args)
    setup_camera(args)

    create_focal_geometry(n)
    create_targeting_brackets(scale=4.0)
    create_fiducial_markers()
    create_telemetry_text(n)
    # NOTE: setup_compositor() is intentionally NOT called.
    # In Blender 5.0 the compositing_node_group approach strips the RGBA alpha
    # channel from the PNG output, producing white backgrounds instead of
    # transparent ones.  film_transparent=True + EEVEE built-in bloom is
    # sufficient; CA / macroblock glitch is applied by ffmpeg post-process.

    print(f"[blender_scene] Rendering {n} frames → {args.output}")
    bpy.ops.render.render(animation=True)
    print("[blender_scene] Done.")


if __name__ == "__main__":
    main()
