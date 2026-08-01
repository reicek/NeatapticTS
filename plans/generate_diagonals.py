#!/usr/bin/env python3
"""
Generate the four diagonal yaw sprites for the 192x192 robot.

This renderer merges the best parts of the procedural variants:
  * orthographic box model tuned to reproduce the front reference exactly;
  * front/back faces sample the original front/back sprites;
  * visible side faces sample the existing left/right profile sprites;
  * per-yaw auto-centering and NEAREST-grid alignment so every diagonal uses
    the full 192x192 canvas;
  * small Tron-disk and gun-muzzle overlays reinforce back/right details.

Outputs (all in the directory containing this script):
  robot-proposal-192-{front-right,back-right,back-left,front-left}.png
  check-{front,right,back,left}.png
"""
from PIL import Image
import math
import os

SIZE = 192
SUPER = 4
SS = SIZE * SUPER

TRANS  = (0, 0, 0, 0)
BODY   = (18, 20, 24, 255)
DARK   = (10, 10, 12, 255)
LIGHT  = (30, 33, 40, 255)
RED    = (221, 34, 0, 255)
BRIGHT = (255, 44, 0, 255)
WHITE  = (255, 255, 255, 255)
BLUE   = (0, 120, 255, 255)


def project(x, y, z, yaw_deg, scale=1.0):
    rad = math.radians(yaw_deg)
    sx = (x * math.cos(rad) + z * math.sin(rad)) * scale
    sy = y * scale
    depth = -x * math.sin(rad) + z * math.cos(rad)
    return sx, sy, depth


def box(parts, x1, y1, z1, x2, y2, z2, color, texture_face='both'):
    """
    Add a cuboid.  texture_face controls sprite texturing:
      'front'  - only the -z face samples the front sprite
      'back'   - only the +z face samples the back sprite
      'both'   - -z uses front, +z uses back
      'none'   - use base color on all faces
    """
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    if z1 > z2: z1, z2 = z2, z1
    parts.append((x1, y1, z1, x2, y2, z2, color, texture_face))


def build_robot():
    """Build a blocky robot tuned to match the front reference sprite."""
    parts = []
    cx = 96
    cz = 96

    # Head: 32x56, depth 36.
    h_w, h_h, h_d = 32, 56, 36
    h_x1 = 80
    h_x2 = h_x1 + h_w - 1
    h_y1 = 0
    h_y2 = h_y1 + h_h - 1
    h_z1 = cz - h_d // 2
    h_z2 = h_z1 + h_d - 1
    box(parts, h_x1, h_y1, h_z1, h_x2, h_y2, h_z2, BODY, 'both')

    # Neck / shoulders.
    n_w, n_d = 48, 44
    n_x1 = cx - n_w // 2
    n_x2 = n_x1 + n_w - 1
    n_z1 = cz - n_d // 2
    n_z2 = n_z1 + n_d - 1
    box(parts, n_x1, 56, n_z1, n_x2, 63, n_z2, BODY, 'both')

    # Torso.
    t_w, t_h, t_d = 76, 48, 60
    t_x1 = 48
    t_x2 = t_x1 + t_w - 1
    t_y1 = 64
    t_y2 = t_y1 + t_h - 1
    t_z1 = cz - t_d // 2
    t_z2 = t_z1 + t_d - 1
    box(parts, t_x1, t_y1, t_z1, t_x2, t_y2, t_z2, BODY, 'both')

    # Pelvis / lower torso (keeps the front torso silhouette continuous).
    p_w, p_h, p_d = 48, 28, 60
    p_x1 = cx - p_w // 2
    p_x2 = p_x1 + p_w - 1
    p_y1 = 112
    p_y2 = p_y1 + p_h - 1
    p_z1 = cz - p_d // 2
    p_z2 = p_z1 + p_d - 1
    box(parts, p_x1, p_y1, p_z1, p_x2, p_y2, p_z2, BODY, 'both')

    # Left arm.
    la_w, la_h, la_d = 16, 92, 40
    la_x1 = 44
    la_x2 = la_x1 + la_w - 1
    la_y1 = 64
    la_y2 = la_y1 + la_h - 1
    la_z1 = cz - la_d // 2
    la_z2 = la_z1 + la_d - 1
    box(parts, la_x1, la_y1, la_z1, la_x2, la_y2, la_z2, BODY, 'both')

    # Right arm + gun as a single block covering the whole right-side
    # colored silhouette. This preserves the exact front reference match.
    ra_w, ra_h, ra_d = 60, 48, 60
    ra_x1 = 132
    ra_x2 = ra_x1 + ra_w - 1
    ra_y1 = 72
    ra_y2 = ra_y1 + ra_h - 1
    ra_z1 = cz - ra_d // 2
    ra_z2 = ra_z1 + ra_d - 1
    box(parts, ra_x1, ra_y1, ra_z1, ra_x2, ra_y2, ra_z2, BODY, 'both')

    # Legs.
    l_w, l_h, l_d = 20, 52, 40
    l_y1 = 140
    l_y2 = 191
    l_z1 = cz - l_d // 2
    l_z2 = l_z1 + l_d - 1

    ll_x2 = cx - 12
    ll_x1 = ll_x2 - l_w + 1
    box(parts, ll_x1, l_y1, l_z1, ll_x2, l_y2, l_z2, DARK, 'both')

    rl_x1 = cx + 12
    rl_x2 = rl_x1 + l_w - 1
    box(parts, rl_x1, l_y1, l_z1, rl_x2, l_y2, l_z2, DARK, 'both')

    return parts


def sprite_bbox(img):
    """Return the bounding box of opaque pixels in a sprite."""
    px = img.load()
    w, h = img.size
    xs = [x for x in range(w) for y in range(h) if px[x, y][3] > 0]
    ys = [y for x in range(w) for y in range(h) if px[x, y][3] > 0]
    if not xs:
        return (0, 0, w - 1, h - 1)
    return (min(xs), min(ys), max(xs), max(ys))


def sample_side(sprite, bbox, z, y, z_range, y_range, mirror=False):
    """
    Map a model (z, y) point into a side reference sprite.

    The side views are profile shots, so the horizontal sprite axis is the
    robot's front-to-back depth (z) and the vertical axis is height (y).
    ``mirror`` flips the horizontal axis so the same physical side can line up
    with the front/back handedness convention.
    """
    z0, z1 = z_range
    y0, y1 = y_range
    t = 0.0 if z1 == z0 else (z - z0) / (z1 - z0)
    if mirror:
        t = 1.0 - t
    u = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    bx0, by0, bx1, by1 = bbox
    sx = bx0 + int(round(t * (bx1 - bx0)))
    sy = by0 + int(round(u * (by1 - by0)))
    sx = max(0, min(SIZE - 1, sx))
    sy = max(0, min(SIZE - 1, sy))
    return sprite.load()[sx, sy]


def overlay_disk(img, yaw_deg):
    """Paint a subtle Tron-style disk on the back-facing diagonal views."""
    if abs(math.sin(math.radians(yaw_deg))) < 0.5:
        return
    px = img.load()
    cx = SS // 2 + int(4 * SUPER * math.sin(math.radians(yaw_deg)))
    cy = SS // 2 - int(10 * SUPER)
    rx = int(18 * SUPER)
    ry = int(24 * SUPER)
    for dy in range(-ry, ry + 1):
        for dx in range(-rx, rx + 1):
            if (dx / max(rx, 1)) ** 2 + (dy / max(ry, 1)) ** 2 <= 1.0:
                x = cx + dx
                y = cy + dy
                if 0 <= x < SS and 0 <= y < SS:
                    base = px[x, y]
                    if base[3] > 0 and base != BRIGHT and base != RED:
                        r = (base[0] + BLUE[0]) // 2
                        g = (base[1] + BLUE[1]) // 2
                        b = (base[2] + BLUE[2]) // 2
                        px[x, y] = (r, g, b, 255)


def overlay_gun(img, yaw_deg):
    """Highlight the gun muzzle on the viewer's-right diagonals."""
    if math.sin(math.radians(yaw_deg)) <= 0:
        return
    px = img.load()
    cx = SS // 2 + int(48 * SUPER)
    cy = SS // 2 + int(2 * SUPER)
    rx = int(8 * SUPER)
    ry = int(3 * SUPER)
    for dy in range(-ry, ry + 1):
        for dx in range(-rx, rx + 1):
            if (dx / max(rx, 1)) ** 2 + (dy / max(ry, 1)) ** 2 <= 1.0:
                x = cx + dx
                y = cy + dy
                if 0 <= x < SS and 0 <= y < SS:
                    base = px[x, y]
                    if base[3] > 0:
                        r = min(255, base[0] + (BRIGHT[0] - base[0]) // 3)
                        g = min(255, base[1] + (BRIGHT[1] - base[1]) // 3)
                        b = min(255, base[2] + (BRIGHT[2] - base[2]) // 3)
                        px[x, y] = (r, g, b, 255)


def compute_bbox(parts, yaw_deg):
    """Project every box corner to find the 2D bounding box at unit scale."""
    rad = math.radians(yaw_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    min_x = min_y = math.inf
    max_x = max_y = -math.inf
    for (x1, y1, z1, x2, y2, z2, _, _) in parts:
        for x in (x1, x2):
            for y in (y1, y2):
                for z in (z1, z2):
                    sx = x * c + z * s
                    sy = y
                    min_x = min(min_x, sx)
                    max_x = max(max_x, sx)
                    min_y = min(min_y, sy)
                    max_y = max(max_y, sy)
    return min_x, max_x, min_y, max_y


def render(yaw_deg, filename, front_sprite, back_sprite,
           left_sprite=None, right_sprite=None, overlays=False):
    """Render one yaw view and save it to ``filename``."""
    yaw = math.radians(yaw_deg)
    s = math.sin(yaw)
    c = math.cos(yaw)
    eps = 1e-9

    parts = build_robot()

    # Global depth / height ranges for side-sprite mapping.
    z_range = (min(p[2] for p in parts), max(p[5] for p in parts))
    y_range = (min(p[1] for p in parts), max(p[4] for p in parts))

    left_bbox = sprite_bbox(left_sprite) if left_sprite else None
    right_bbox = sprite_bbox(right_sprite) if right_sprite else None

    # Per-yaw framing.  yaw == 0 is kept at identity scale and offset so the
    # front check view matches the front reference pixel-for-pixel.
    if abs(s) < eps and c > 0:
        render_scale = float(SUPER)
        tx = 0.0
        ty = 0.0
        side_texturing = False
    else:
        min_sx, max_sx, min_sy, max_sy = compute_bbox(parts, yaw_deg)
        bbox_w = max_sx - min_sx + 1
        bbox_h = max_sy - min_sy + 1
        final_scale = min(1.0, SIZE / bbox_w)
        # The model is already 192 world units tall, so no vertical scaling up
        # is needed.  If a future model is shorter, uncomment the next line.
        # final_scale = min(final_scale, SIZE / bbox_h)
        render_scale = SUPER * final_scale
        output_left = (SIZE - bbox_w) // 2
        tx = output_left * SUPER + SUPER // 2 - min_sx * render_scale
        ty = SUPER // 2 - min_sy * render_scale
        side_texturing = bool(left_sprite and right_sprite)

    img = Image.new('RGBA', (SS, SS), TRANS)
    pixels = img.load()
    zbuf = [[1e9 for _ in range(SS)] for _ in range(SS)]
    front_px = front_sprite.load()
    back_px = back_sprite.load()

    # View ray direction for face culling.  A face is front-facing when its
    # normal dotted with the ray direction is negative.
    ray_x = -s
    ray_z = c

    def face_visible(axis, side):
        if not side_texturing:
            return True
        if axis == 'y':
            return True
        if axis == 'x':
            nx = -1 if side == 1 else 1
            return nx * ray_x < -1e-9
        if axis == 'z':
            nz = -1 if side == 1 else 1
            return nz * ray_z < -1e-9
        return True

    for part in parts:
        x1, y1, z1, x2, y2, z2, base_color, texture_face = part

        # Treat coordinates as pixel centers: expand by 0.5.
        X1 = x1 - 0.5
        X2 = x2 + 0.5
        Y1 = y1 - 0.5
        Y2 = y2 + 0.5
        Z1 = z1 - 0.5
        Z2 = z2 + 0.5

        # Skip parts whose vertical faces are all back-facing.
        if side_texturing:
            if not (face_visible('x', 1) or face_visible('x', 2) or
                    face_visible('z', 1) or face_visible('z', 2)):
                continue

        corners = []
        for x in (X1, X2):
            for y in (Y1, Y2):
                for z in (Z1, Z2):
                    sx, sy, _ = project(x, y, z, yaw_deg, 1.0)
                    corners.append((sx * render_scale + tx, sy * render_scale + ty))
        min_sx = max(0, int(math.floor(min(c[0] for c in corners))) - SUPER // 2)
        max_sx = min(SS - 1, int(math.ceil(max(c[0] for c in corners))) + SUPER // 2)
        min_sy = max(0, int(math.floor(min(c[1] for c in corners))) - SUPER // 2)
        max_sy = min(SS - 1, int(math.ceil(max(c[1] for c in corners))) + SUPER // 2)

        for py in range(min_sy, max_sy + 1):
            if (py - ty) / render_scale < Y1 or (py - ty) / render_scale > Y2:
                continue

            for px in range(min_sx, max_sx + 1):
                world_x = (px - tx) / render_scale
                ranges = []

                if abs(s) > eps:
                    d_low_x = (world_x * c - X2) / s
                    d_high_x = (world_x * c - X1) / s
                    if s < 0:
                        d_low_x, d_high_x = d_high_x, d_low_x
                    ranges.append((d_low_x, d_high_x, 'x'))
                else:
                    if not (X1 <= world_x * c <= X2):
                        continue

                if abs(c) > eps:
                    d_low_z = (Z1 - world_x * s) / c
                    d_high_z = (Z2 - world_x * s) / c
                    if c < 0:
                        d_low_z, d_high_z = d_high_z, d_low_z
                    ranges.append((d_low_z, d_high_z, 'z'))
                else:
                    if not (Z1 <= world_x * s <= Z2):
                        continue

                if len(ranges) == 2:
                    d_min = max(ranges[0][0], ranges[1][0])
                    d_max = min(ranges[0][1], ranges[1][1])
                    if d_min > d_max:
                        continue
                    hit_face = ranges[0][2] if d_min == ranges[0][0] else ranges[1][2]
                else:
                    d_min, d_max, hit_face = ranges[0][0], ranges[0][1], ranges[0][2]

                hx = world_x * c - d_min * s
                hy = (py - ty) / render_scale
                hz = world_x * s + d_min * c

                # Texture mapping.
                color = base_color
                if hit_face == 'z':
                    side = 1 if abs(hz - Z1) < 1.5 else 2
                    if not face_visible('z', side):
                        continue
                    if side == 1 and texture_face in ('front', 'both'):
                        fx = int(math.floor(hx))
                        fy = int(math.floor(hy))
                        if 0 <= fx < SIZE and 0 <= fy < SIZE:
                            sample = front_px[fx, fy]
                            if sample[3] > 128:
                                color = sample
                            else:
                                continue
                        else:
                            continue
                    elif side == 2 and texture_face in ('back', 'both'):
                        fx = int(math.floor(hx))
                        fy = int(math.floor(hy))
                        if 0 <= fx < SIZE and 0 <= fy < SIZE:
                            sample = back_px[fx, fy]
                            if sample[3] > 128:
                                color = sample
                            else:
                                continue
                        else:
                            continue
                elif hit_face == 'x' and side_texturing:
                    side = 1 if abs(hx - X1) < 1.5 else 2
                    if not face_visible('x', side):
                        continue
                    if side == 1:
                        tex = sample_side(left_sprite, left_bbox, hz, hy,
                                          z_range, y_range, mirror=False)
                        if tex[3] > 0:
                            color = tex
                    elif side == 2:
                        tex = sample_side(right_sprite, right_bbox, hz, hy,
                                          z_range, y_range, mirror=True)
                        if tex[3] > 0:
                            color = tex

                if d_min < zbuf[px][py]:
                    zbuf[px][py] = d_min
                    pixels[px, py] = color

    if overlays:
        overlay_disk(img, yaw_deg)
        overlay_gun(img, yaw_deg)

    final = img.resize((SIZE, SIZE), Image.Resampling.NEAREST)
    final.save(filename)
    return final


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate the four missing diagonal yaw sprites and optional cardinal check renders for the 192x192 robot.')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for output PNGs (default: the directory containing this script).')
    parser.add_argument('--checks', action=argparse.BooleanOptionalAction, default=True,
                        help='Also generate check-{front,right,back,left}.png reference renders.')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.output_dir if args.output_dir else script_dir
    os.makedirs(out_dir, exist_ok=True)

    front = Image.open(os.path.join(script_dir, 'robot-proposal-192-front.png')).convert('RGBA')
    back  = Image.open(os.path.join(script_dir, 'robot-proposal-192-back.png')).convert('RGBA')
    left  = Image.open(os.path.join(script_dir, 'robot-proposal-192-left.png')).convert('RGBA')
    right = Image.open(os.path.join(script_dir, 'robot-proposal-192-right.png')).convert('RGBA')

    for yaw, name in [
        (45,  'front-right'),
        (135, 'back-right'),
        (225, 'back-left'),
        (315, 'front-left'),
    ]:
        render(yaw, os.path.join(out_dir, f'robot-proposal-192-{name}.png'),
               front, back, left, right, overlays=True)
        print(f'Generated robot-proposal-192-{name}.png')

    if args.checks:
        for yaw, name in [
            (0,   'front'),
            (90,  'right'),
            (180, 'back'),
            (270, 'left'),
        ]:
            render(yaw, os.path.join(out_dir, f'check-{name}.png'),
                   front, back, left, right, overlays=False)
            print(f'Generated check-{name}.png')


if __name__ == '__main__':
    main()
