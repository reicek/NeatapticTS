#!/usr/bin/env python3
"""
Generate 8-directional robot sprite set for Neat Shooter.

Outputs:
  - plans/robot-sprite-data.js      (ES module with palette + frames)
  - plans/robot-sprite-data.json    (JSON verification copy)
  - plans/robot-proposal-192-<dir>-<pose>.png  (192x192 reference PNGs)

Logical sprite grid: 48x48. Display scale: 4x -> 192x192.
Palette indices are numeric so the renderer can remap team colors.
"""

from PIL import Image, ImageDraw
import json
import os

SCALE = 4
W, H = 48, 48

# ---------------------------------------------------------------------------
# Palette (must match ROBOT_SPRITE_PALETTE in JS output)
# ---------------------------------------------------------------------------
TRANS   = (0, 0, 0, 0)
BLK     = (10, 10, 12, 255)
SUIT    = (18, 20, 24, 255)
SUIT_L  = (30, 33, 40, 255)
WHITE   = (255, 255, 255, 255)
NEON    = (221, 34, 0, 255)       # 5 Ares Red default accent
NEON_G  = (255, 44, 0, 255)       # 6 opaque glow (cannon tip / core)
BLAST   = (255, 66, 22, 128)      # 7 semitransparent team-colored muzzle flash
BLAST_W = (255, 255, 255, 128)    # 8 semitransparent white muzzle flash core

PALETTE = [TRANS, BLK, SUIT, SUIT_L, WHITE, NEON, NEON_G, BLAST, BLAST_W]
PALETTE_INDEX = {c: i for i, c in enumerate(PALETTE)}

DEFAULT_NEON = NEON


def make_canvas():
    return Image.new('RGBA', (W, H), TRANS)


def save_scaled(img, path):
    img.resize((W * SCALE, H * SCALE), Image.Resampling.NEAREST).save(path)


def rect(d, x, y, w, h, c):
    if w <= 0 or h <= 0:
        return
    d.rectangle([x, y, x + w - 1, y + h - 1], fill=c)


def line_h(d, x, y, w, c):
    if w <= 0:
        return
    d.rectangle([x, y, x + w - 1, y], fill=c)


def line_v(d, x, y, h, c):
    if h <= 0:
        return
    d.rectangle([x, y, x, y + h - 1], fill=c)


def neon_glow(neon):
    return tuple(min(255, int(c * 1.3)) for c in neon[:3]) + (255,)


# ---------------------------------------------------------------------------
# Shared parts
# ---------------------------------------------------------------------------

def draw_head_front(d, neon, y_off=0):
    y = y_off
    rect(d, 20, y, 8, 14, BLK)
    rect(d, 21, y, 6, 15, SUIT)
    rect(d, 22, y, 4, 3, SUIT)
    rect(d, 21, y + 3, 6, 11, BLK)
    line_h(d, 21, y + 7, 6, neon)
    line_h(d, 21, y + 11, 1, neon)
    line_h(d, 26, y + 11, 1, neon)


def draw_head_back(d, neon, y_off=0):
    y = y_off
    rect(d, 20, y, 8, 14, BLK)
    rect(d, 21, y, 6, 15, SUIT)
    rect(d, 22, y, 4, 3, SUIT)
    rect(d, 21, y + 3, 6, 11, BLK)


def draw_head_diag(d, neon, y_off=0):
    y = y_off
    rect(d, 21, y, 7, 14, BLK)
    rect(d, 22, y, 5, 15, SUIT)
    rect(d, 23, y, 3, 3, SUIT)
    rect(d, 22, y + 3, 5, 10, BLK)
    line_h(d, 22, y + 7, 5, neon)


def draw_head_side(d, neon, facing='right', y_off=0):
    y = y_off
    if facing == 'right':
        rect(d, 21, y, 6, 15, BLK)
        rect(d, 22, y, 4, 16, SUIT)
        rect(d, 22, y, 4, 3, SUIT)
        rect(d, 23, y + 3, 3, 11, BLK)
        line_h(d, 23, y + 7, 3, neon)
    else:
        rect(d, 21, y, 6, 15, BLK)
        rect(d, 22, y, 4, 16, SUIT)
        rect(d, 22, y, 4, 3, SUIT)
        rect(d, 22, y + 3, 3, 11, BLK)
        line_h(d, 22, y + 7, 3, neon)


def draw_torso_front(d, neon, y_off=0):
    y = y_off
    NG = neon_glow(neon)
    rect(d, 18, y + 15, 12, 16, SUIT)
    rect(d, 17, y + 16, 1, 10, SUIT)
    rect(d, 30, y + 16, 1, 10, SUIT)
    line_v(d, 21, y + 16, 6, neon)
    line_v(d, 26, y + 16, 6, neon)
    rect(d, 22, y + 22, 4, 4, NG)
    rect(d, 23, y + 23, 2, 2, WHITE)
    rect(d, 20, y + 31, 8, 4, SUIT_L)


def draw_torso_back(d, neon, y_off=0):
    y = y_off
    NG = neon_glow(neon)
    rect(d, 18, y + 15, 12, 16, SUIT)
    rect(d, 17, y + 16, 1, 10, SUIT)
    rect(d, 30, y + 16, 1, 10, SUIT)
    line_v(d, 23, y + 16, 14, neon)
    line_v(d, 24, y + 16, 14, neon)
    rect(d, 20, y + 31, 8, 4, SUIT_L)
    # Identity disk on back, between shoulders, slightly high
    rect(d, 21, y + 13, 6, 10, BLK)
    rect(d, 22, y + 14, 4, 8, neon)
    rect(d, 23, y + 15, 2, 6, WHITE)
    rect(d, 23, y + 17, 2, 2, SUIT)


def draw_torso_diag(d, neon, y_off=0):
    y = y_off
    NG = neon_glow(neon)
    rect(d, 19, y + 15, 10, 16, SUIT)
    rect(d, 18, y + 16, 1, 10, SUIT)
    rect(d, 29, y + 16, 1, 8, SUIT)
    line_v(d, 22, y + 16, 6, neon)
    line_v(d, 26, y + 16, 6, neon)
    rect(d, 22, y + 22, 4, 4, NG)
    rect(d, 23, y + 23, 2, 2, WHITE)
    rect(d, 20, y + 31, 8, 4, SUIT_L)


def draw_torso_back_diag(d, neon, y_off=0):
    y = y_off
    NG = neon_glow(neon)
    rect(d, 19, y + 15, 10, 16, SUIT)
    rect(d, 18, y + 16, 1, 10, SUIT)
    rect(d, 29, y + 16, 1, 8, SUIT)
    line_v(d, 23, y + 16, 14, neon)
    line_v(d, 24, y + 16, 14, neon)
    rect(d, 20, y + 31, 8, 4, SUIT_L)
    rect(d, 20, y + 13, 6, 10, BLK)
    rect(d, 21, y + 14, 4, 8, neon)
    rect(d, 22, y + 15, 2, 6, WHITE)
    rect(d, 22, y + 17, 2, 2, SUIT)


def draw_torso_side(d, neon, facing='right', y_off=0):
    y = y_off
    NG = neon_glow(neon)
    rect(d, 20, y + 15, 7, 16, SUIT)
    rect(d, 19, y + 16, 1, 10, SUIT)
    rect(d, 27, y + 16, 1, 8, SUIT)
    line_h(d, 22, y + 16, 4, neon)
    rect(d, 22, y + 22, 4, 4, NG)
    rect(d, 23, y + 23, 2, 2, WHITE)
    rect(d, 21, y + 31, 6, 4, SUIT_L)
    # Back disk visible in profile
    if facing == 'right':
        rect(d, 17, y + 14, 3, 8, BLK)
        rect(d, 18, y + 15, 1, 6, neon)
        rect(d, 18, y + 16, 1, 4, WHITE)
        rect(d, 18, y + 18, 1, 1, SUIT)
    else:
        rect(d, 27, y + 14, 3, 8, BLK)
        rect(d, 28, y + 15, 1, 6, neon)
        rect(d, 28, y + 16, 1, 4, WHITE)
        rect(d, 28, y + 18, 1, 1, SUIT)


# ---------------------------------------------------------------------------
# Arms + cannon
# ---------------------------------------------------------------------------

def draw_arm_left_front(d, neon, y_off=0):
    y = y_off
    rect(d, 12, y + 16, 3, 10, SUIT)
    rect(d, 11, y + 26, 3, 9, SUIT)
    line_h(d, 12, y + 17, 3, neon)
    line_h(d, 11, y + 29, 2, neon)
    rect(d, 11, y + 35, 3, 4, BLK)


def draw_arm_right_front(d, neon, y_off=0, shoot=False):
    y = y_off
    NG = neon_glow(neon)
    # Right arm/hand supporting the cannon from below/right.
    rect(d, 30, y + 24, 3, 7, SUIT)
    rect(d, 31, y + 31, 3, 5, SUIT)
    line_h(d, 30, y + 25, 3, neon)
    line_h(d, 31, y + 32, 2, neon)
    rect(d, 30, y + 36, 3, 4, BLK)

    # Cannon pointing forward: barrel end visible as a circle above the hand.
    rect(d, 26, y + 17, 8, 7, BLK)
    rect(d, 28, y + 19, 4, 4, SUIT_L)
    line_h(d, 27, y + 18, 6, neon)
    line_h(d, 27, y + 23, 6, neon)
    line_v(d, 27, y + 18, 6, neon)
    line_v(d, 32, y + 18, 6, neon)
    rect(d, 29, y + 20, 2, 2, NG)
    rect(d, 29, y + 21, 1, 1, WHITE)

    if shoot:
        # Front muzzle flash is 100% opaque so the sprite never looks see-through.
        rect(d, 25, y + 16, 10, 9, BLK)
        rect(d, 26, y + 17, 8, 7, NG)
        rect(d, 28, y + 19, 4, 4, WHITE)
        rect(d, 29, y + 20, 2, 2, BLK)


def draw_arm_left_back(d, neon, y_off=0):
    y = y_off
    # In the back view the robot's left arm appears on the viewer's right side.
    rect(d, 33, y + 16, 3, 10, SUIT)
    rect(d, 34, y + 26, 3, 9, SUIT)
    line_h(d, 33, y + 17, 3, neon)
    line_h(d, 34, y + 29, 2, neon)
    rect(d, 34, y + 35, 3, 4, BLK)


def draw_arm_right_back(d, neon, y_off=0, shoot=False):
    """Gun-holding arm for the back view. The cannon is mostly hidden behind
    the robot; only a small grip/border is visible on the viewer's left side.
    The muzzle blast, when visible, is just an edge and stays semitransparent."""
    y = y_off
    # Robot's right arm/hand grip on the left edge of the torso.
    rect(d, 17, y + 21, 2, 7, BLK)
    rect(d, 18, y + 22, 1, 5, SUIT)
    line_h(d, 17, y + 23, 2, neon)

    if shoot:
        # Only the edge of the muzzle flash peeks out from behind the body.
        rect(d, 14, y + 19, 3, 8, BLAST_W)
        rect(d, 15, y + 21, 2, 4, BLAST)


def draw_arm_diag_left(d, neon, y_off=0):
    y = y_off
    rect(d, 14, y + 16, 3, 9, SUIT)
    rect(d, 13, y + 25, 3, 8, SUIT)
    line_h(d, 14, y + 17, 3, neon)
    line_h(d, 13, y + 28, 2, neon)
    rect(d, 13, y + 33, 3, 4, BLK)


def draw_arm_diag_right(d, neon, y_off=0, shoot=False):
    y = y_off
    NG = neon_glow(neon)
    recoil = 1 if shoot else 0
    rect(d, 27 - recoil, y + 18, 3, 7, SUIT)
    rect(d, 28 - recoil, y + 25, 3, 5, SUIT)
    line_h(d, 27 - recoil, y + 19, 3, neon)
    line_h(d, 28 - recoil, y + 27, 2, neon)
    rect(d, 28 - recoil, y + 22, 3, 3, BLK)
    rect(d, 30 - recoil, y + 21, 10, 4, BLK)
    rect(d, 31 - recoil, y + 22, 8, 2, SUIT_L)
    line_h(d, 31 - recoil, y + 21, 8, neon)
    line_h(d, 31 - recoil, y + 24, 8, neon)
    rect(d, 39 - recoil, y + 20, 3, 6, BLK)
    rect(d, 40 - recoil, y + 21, 1, 4, NG)
    if shoot:
        rect(d, 42 - recoil, y + 20, 3, 6, BLAST_W)
        rect(d, 43 - recoil, y + 22, 2, 2, BLAST)


def draw_arm_back_diag_left(d, neon, y_off=0):
    y = y_off
    rect(d, 14, y + 16, 3, 9, SUIT)
    rect(d, 13, y + 25, 3, 8, SUIT)
    line_h(d, 14, y + 17, 3, neon)
    line_h(d, 13, y + 28, 2, neon)
    rect(d, 13, y + 33, 3, 4, BLK)


def draw_arm_back_diag_right(d, neon, y_off=0, shoot=False):
    y = y_off
    NG = neon_glow(neon)
    recoil = 1 if shoot else 0
    rect(d, 27 - recoil, y + 18, 3, 7, SUIT)
    rect(d, 28 - recoil, y + 25, 3, 5, SUIT)
    line_h(d, 27 - recoil, y + 19, 3, neon)
    line_h(d, 28 - recoil, y + 27, 2, neon)
    rect(d, 28 - recoil, y + 22, 3, 3, BLK)
    rect(d, 30 - recoil, y + 21, 10, 4, BLK)
    rect(d, 31 - recoil, y + 22, 8, 2, SUIT_L)
    line_h(d, 31 - recoil, y + 21, 8, neon)
    line_h(d, 31 - recoil, y + 24, 8, neon)
    rect(d, 39 - recoil, y + 20, 3, 6, BLK)
    rect(d, 40 - recoil, y + 21, 1, 4, NG)
    if shoot:
        rect(d, 42 - recoil, y + 20, 3, 6, BLAST_W)
        rect(d, 43 - recoil, y + 22, 2, 2, BLAST)


def draw_arm_side_back(d, neon, facing='right', y_off=0):
    y = y_off
    if facing == 'right':
        rect(d, 16, y + 18, 2, 9, SUIT)
        rect(d, 15, y + 27, 2, 8, SUIT)
        line_h(d, 16, y + 19, 2, neon)
        line_h(d, 15, y + 30, 2, neon)
        rect(d, 15, y + 35, 2, 4, BLK)
    else:
        rect(d, 29, y + 18, 2, 9, SUIT)
        rect(d, 30, y + 27, 2, 8, SUIT)
        line_h(d, 29, y + 19, 2, neon)
        line_h(d, 30, y + 30, 2, neon)
        rect(d, 30, y + 35, 2, 4, BLK)


def draw_arm_side_front(d, neon, facing='right', y_off=0, shoot=False):
    y = y_off
    NG = neon_glow(neon)
    if facing == 'right':
        recoil = 1 if shoot else 0
        rect(d, 26 - recoil, y + 19, 3, 6, SUIT)
        rect(d, 27 - recoil, y + 25, 3, 4, SUIT)
        line_h(d, 26 - recoil, y + 20, 3, neon)
        line_h(d, 27 - recoil, y + 27, 2, neon)
        rect(d, 27 - recoil, y + 22, 3, 3, BLK)
        rect(d, 29 - recoil, y + 22, 13, 4, BLK)
        rect(d, 30 - recoil, y + 23, 11, 2, SUIT_L)
        line_h(d, 30 - recoil, y + 22, 10, neon)
        line_h(d, 30 - recoil, y + 25, 10, neon)
        rect(d, 41 - recoil, y + 21, 3, 6, BLK)
        rect(d, 42 - recoil, y + 22, 1, 4, NG)
        if shoot:
            rect(d, 44 - recoil, y + 21, 3, 6, BLAST_W)
            rect(d, 45 - recoil, y + 23, 2, 2, BLAST)
    else:
        recoil = 1 if shoot else 0
        rect(d, 18 - recoil, y + 19, 3, 6, SUIT)
        rect(d, 17 - recoil, y + 25, 3, 4, SUIT)
        line_h(d, 18 - recoil, y + 20, 3, neon)
        line_h(d, 17 - recoil, y + 27, 2, neon)
        rect(d, 17 - recoil, y + 22, 3, 3, BLK)
        rect(d, 5 - recoil, y + 22, 13, 4, BLK)
        rect(d, 6 - recoil, y + 23, 11, 2, SUIT_L)
        line_h(d, 7 - recoil, y + 22, 10, neon)
        line_h(d, 7 - recoil, y + 25, 10, neon)
        rect(d, 3 - recoil, y + 21, 3, 6, BLK)
        rect(d, 4 - recoil, y + 22, 1, 4, NG)
        if shoot:
            rect(d, 0 - recoil, y + 21, 3, 6, BLAST_W)
            rect(d, 1 - recoil, y + 23, 2, 2, BLAST)


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def draw_legs_front(d, neon, y_off=0, walk=0):
    y = y_off
    if walk == 0:
        rect(d, 17, y + 35, 3, 12, SUIT)
        rect(d, 28, y + 35, 3, 12, SUIT)
        line_h(d, 17, y + 37, 2, neon); line_h(d, 17, y + 45, 2, neon); rect(d, 16, y + 47, 5, 1, SUIT_L)
        line_h(d, 28, y + 37, 2, neon); line_h(d, 28, y + 45, 2, neon); rect(d, 27, y + 47, 5, 1, SUIT_L)
    elif walk == 1:
        # left leg up
        rect(d, 17, y + 34, 3, 11, SUIT)
        rect(d, 28, y + 35, 3, 12, SUIT)
        line_h(d, 17, y + 36, 2, neon); line_h(d, 17, y + 44, 2, neon); rect(d, 16, y + 46, 5, 1, SUIT_L)
        line_h(d, 28, y + 37, 2, neon); line_h(d, 28, y + 45, 2, neon); rect(d, 27, y + 47, 5, 1, SUIT_L)
    else:
        # right leg up
        rect(d, 17, y + 35, 3, 12, SUIT)
        rect(d, 28, y + 34, 3, 11, SUIT)
        line_h(d, 17, y + 37, 2, neon); line_h(d, 17, y + 45, 2, neon); rect(d, 16, y + 47, 5, 1, SUIT_L)
        line_h(d, 28, y + 36, 2, neon); line_h(d, 28, y + 44, 2, neon); rect(d, 27, y + 46, 5, 1, SUIT_L)


def draw_legs_back(d, neon, y_off=0, walk=0):
    y = y_off
    if walk == 0:
        rect(d, 17, y + 35, 3, 12, SUIT)
        rect(d, 28, y + 35, 3, 12, SUIT)
        line_h(d, 17, y + 37, 2, neon); line_h(d, 17, y + 45, 2, neon); rect(d, 16, y + 47, 5, 1, SUIT_L)
        line_h(d, 28, y + 37, 2, neon); line_h(d, 28, y + 45, 2, neon); rect(d, 27, y + 47, 5, 1, SUIT_L)
    elif walk == 1:
        rect(d, 17, y + 34, 3, 11, SUIT)
        rect(d, 28, y + 35, 3, 12, SUIT)
        line_h(d, 17, y + 36, 2, neon); line_h(d, 17, y + 44, 2, neon); rect(d, 16, y + 46, 5, 1, SUIT_L)
        line_h(d, 28, y + 37, 2, neon); line_h(d, 28, y + 45, 2, neon); rect(d, 27, y + 47, 5, 1, SUIT_L)
    else:
        rect(d, 17, y + 35, 3, 12, SUIT)
        rect(d, 28, y + 34, 3, 11, SUIT)
        line_h(d, 17, y + 37, 2, neon); line_h(d, 17, y + 45, 2, neon); rect(d, 16, y + 47, 5, 1, SUIT_L)
        line_h(d, 28, y + 36, 2, neon); line_h(d, 28, y + 44, 2, neon); rect(d, 27, y + 46, 5, 1, SUIT_L)


def draw_legs_diag(d, neon, y_off=0, walk=0):
    y = y_off
    if walk == 0:
        rect(d, 19, y + 35, 3, 12, SUIT)
        rect(d, 26, y + 35, 3, 12, SUIT)
        line_h(d, 19, y + 37, 2, neon); line_h(d, 19, y + 45, 2, neon); rect(d, 18, y + 47, 5, 1, SUIT_L)
        line_h(d, 26, y + 37, 2, neon); line_h(d, 26, y + 45, 2, neon); rect(d, 25, y + 47, 5, 1, SUIT_L)
    elif walk == 1:
        # left leg up
        rect(d, 19, y + 34, 3, 11, SUIT)
        rect(d, 26, y + 35, 3, 12, SUIT)
        line_h(d, 19, y + 36, 2, neon); line_h(d, 19, y + 44, 2, neon); rect(d, 18, y + 46, 5, 1, SUIT_L)
        line_h(d, 26, y + 37, 2, neon); line_h(d, 26, y + 45, 2, neon); rect(d, 25, y + 47, 5, 1, SUIT_L)
    else:
        # right leg up
        rect(d, 19, y + 35, 3, 12, SUIT)
        rect(d, 26, y + 34, 3, 11, SUIT)
        line_h(d, 19, y + 37, 2, neon); line_h(d, 19, y + 45, 2, neon); rect(d, 18, y + 47, 5, 1, SUIT_L)
        line_h(d, 26, y + 36, 2, neon); line_h(d, 26, y + 44, 2, neon); rect(d, 25, y + 46, 5, 1, SUIT_L)


def draw_legs_side(d, neon, facing='right', y_off=0, walk=0):
    y = y_off
    if walk == 0:
        rect(d, 20, y + 35, 3, 12, SUIT)
        rect(d, 24, y + 35, 3, 12, SUIT)
        line_h(d, 20, y + 37, 2, neon); line_h(d, 20, y + 45, 2, neon); rect(d, 19, y + 47, 5, 1, SUIT_L)
        line_h(d, 24, y + 37, 2, neon); line_h(d, 24, y + 45, 2, neon); rect(d, 23, y + 47, 5, 1, SUIT_L)
    elif walk == 1:
        # front leg up
        if facing == 'right':
            rect(d, 24, y + 34, 3, 11, SUIT)
            rect(d, 20, y + 35, 3, 12, SUIT)
            line_h(d, 24, y + 36, 2, neon); line_h(d, 24, y + 44, 2, neon); rect(d, 23, y + 46, 5, 1, SUIT_L)
            line_h(d, 20, y + 37, 2, neon); line_h(d, 20, y + 45, 2, neon); rect(d, 19, y + 47, 5, 1, SUIT_L)
        else:
            rect(d, 20, y + 34, 3, 11, SUIT)
            rect(d, 24, y + 35, 3, 12, SUIT)
            line_h(d, 20, y + 36, 2, neon); line_h(d, 20, y + 44, 2, neon); rect(d, 19, y + 46, 5, 1, SUIT_L)
            line_h(d, 24, y + 37, 2, neon); line_h(d, 24, y + 45, 2, neon); rect(d, 23, y + 47, 5, 1, SUIT_L)
    else:
        # back leg up
        if facing == 'right':
            rect(d, 20, y + 34, 3, 11, SUIT)
            rect(d, 24, y + 35, 3, 12, SUIT)
            line_h(d, 20, y + 36, 2, neon); line_h(d, 20, y + 44, 2, neon); rect(d, 19, y + 46, 5, 1, SUIT_L)
            line_h(d, 24, y + 37, 2, neon); line_h(d, 24, y + 45, 2, neon); rect(d, 23, y + 47, 5, 1, SUIT_L)
        else:
            rect(d, 24, y + 34, 3, 11, SUIT)
            rect(d, 20, y + 35, 3, 12, SUIT)
            line_h(d, 24, y + 36, 2, neon); line_h(d, 24, y + 44, 2, neon); rect(d, 23, y + 46, 5, 1, SUIT_L)
            line_h(d, 20, y + 37, 2, neon); line_h(d, 20, y + 45, 2, neon); rect(d, 19, y + 47, 5, 1, SUIT_L)


# ---------------------------------------------------------------------------
# Full direction compositors
# ---------------------------------------------------------------------------

def draw_front(neon, pose='stand'):
    img = make_canvas()
    dr = ImageDraw.Draw(img)
    y_off, walk, shoot = _pose_params(pose)
    draw_head_front(dr, neon, y_off)
    draw_torso_front(dr, neon, y_off)
    draw_legs_front(dr, neon, y_off, walk)
    draw_arm_left_front(dr, neon, y_off)
    draw_arm_right_front(dr, neon, y_off, shoot)
    return img


def draw_back(neon, pose='stand'):
    img = make_canvas()
    dr = ImageDraw.Draw(img)
    y_off, walk, shoot = _pose_params(pose)
    draw_head_back(dr, neon, y_off)
    draw_torso_back(dr, neon, y_off)
    draw_legs_back(dr, neon, y_off, walk)
    draw_arm_left_back(dr, neon, y_off)
    draw_arm_right_back(dr, neon, y_off, shoot)
    return img


def draw_right(neon, pose='stand'):
    img = make_canvas()
    dr = ImageDraw.Draw(img)
    y_off, walk, shoot = _pose_params(pose)
    draw_head_side(dr, neon, 'right', y_off)
    draw_torso_side(dr, neon, 'right', y_off)
    draw_legs_side(dr, neon, 'right', y_off, walk)
    draw_arm_side_back(dr, neon, 'right', y_off)
    draw_arm_side_front(dr, neon, 'right', y_off, shoot)
    return img


def draw_left(neon, pose='stand'):
    img = make_canvas()
    dr = ImageDraw.Draw(img)
    y_off, walk, shoot = _pose_params(pose)
    draw_head_side(dr, neon, 'left', y_off)
    draw_torso_side(dr, neon, 'left', y_off)
    draw_legs_side(dr, neon, 'left', y_off, walk)
    draw_arm_side_back(dr, neon, 'left', y_off)
    draw_arm_side_front(dr, neon, 'left', y_off, shoot)
    return img


def draw_front_right(neon, pose='stand'):
    img = make_canvas()
    dr = ImageDraw.Draw(img)
    y_off, walk, shoot = _pose_params(pose)
    draw_head_diag(dr, neon, y_off)
    draw_torso_diag(dr, neon, y_off)
    draw_legs_diag(dr, neon, y_off, walk)
    draw_arm_diag_left(dr, neon, y_off)
    draw_arm_diag_right(dr, neon, y_off, shoot)
    return img


def draw_front_left(neon, pose='stand'):
    img = draw_front_right(neon, pose)
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def draw_back_right(neon, pose='stand'):
    img = make_canvas()
    dr = ImageDraw.Draw(img)
    y_off, walk, shoot = _pose_params(pose)
    draw_head_diag(dr, neon, y_off)
    draw_torso_back_diag(dr, neon, y_off)
    draw_legs_diag(dr, neon, y_off, walk)
    draw_arm_back_diag_left(dr, neon, y_off)
    draw_arm_back_diag_right(dr, neon, y_off, shoot)
    return img


def draw_back_left(neon, pose='stand'):
    img = draw_back_right(neon, pose)
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def _pose_params(pose):
    if pose == 'stand':
        return 0, 0, False
    if pose == 'walk1':
        return -1, 1, False
    if pose == 'walk2':
        return -1, 2, False
    if pose == 'shoot':
        return 0, 0, True
    return 0, 0, False


DIRECTIONS = {
    'front': draw_front,
    'frontRight': draw_front_right,
    'right': draw_right,
    'backRight': draw_back_right,
    'back': draw_back,
    'backLeft': draw_back_left,
    'left': draw_left,
    'frontLeft': draw_front_left,
}

POSES = ['stand', 'walk1', 'walk2', 'shoot']


def encode_image(img):
    """Convert a 48x48 RGBA image to a 48x48 palette-index grid."""
    px = img.load()
    grid = []
    for y in range(H):
        row = []
        for x in range(W):
            c = px[x, y]
            # nearest palette index (handles accidental color drift)
            if c not in PALETTE_INDEX:
                best = min(PALETTE_INDEX, key=lambda p: sum((a - b) ** 2 for a, b in zip(c, p)))
                c = best
            row.append(PALETTE_INDEX[c])
        grid.append(row)
    return grid


def nearest_palette_index(c):
    if c in PALETTE_INDEX:
        return PALETTE_INDEX[c]
    return PALETTE_INDEX[min(PALETTE_INDEX, key=lambda p: sum((a - b) ** 2 for a, b in zip(c, p)))]


def main():
    base_dir = 'plans'
    frames = {}
    for dir_name, fn in DIRECTIONS.items():
        frames[dir_name] = {}
        for pose in POSES:
            img = fn(DEFAULT_NEON, pose)
            path = os.path.join(base_dir, f'robot-proposal-192-{dir_name}-{pose}.png')
            save_scaled(img, path)
            frames[dir_name][pose] = encode_image(img)
            print(f'Generated {dir_name}/{pose} -> {path}')

    # JS module
    js_path = os.path.join(base_dir, 'robot-sprite-data.js')
    with open(js_path, 'w') as f:
        f.write('// Auto-generated by plans/generate-robot-sprites.py\n')
        f.write('// Logical grid: 48x48. Render at ROBOT_SPRITE_SCALE for final output.\n')
        f.write('// Palette indices are numeric so the renderer can remap colors.\n\n')
        f.write('export const ROBOT_SPRITE_SCALE = 4;\n\n')
        f.write('export const ROBOT_SPRITE_PALETTE = [\n')
        for c in PALETTE:
            f.write(f'  [{c[0]}, {c[1]}, {c[2]}, {c[3]}],\n')
        f.write('];\n\n')
        f.write('// Upper/lower split: legs and lower torso start around row 35.\n')
        f.write('// Combine upper body of `shoot` with lower body of any walk frame\n')
        f.write('// to make the robot shoot while walking.\n\n')
        f.write('export const ROBOT_SPRITE_FRAMES = {\n')
        for dir_name in DIRECTIONS:
            f.write(f'  {dir_name}: {{\n')
            for pose in POSES:
                grid = frames[dir_name][pose]
                f.write(f'    {pose}: [\n')
                for row in grid:
                    f.write('      [' + ','.join(str(v) for v in row) + '],\n')
                f.write('    ],\n')
            f.write('  },\n')
        f.write('};\n')
    print(f'Wrote {js_path}')

    # JSON copy
    json_path = os.path.join(base_dir, 'robot-sprite-data.json')
    with open(json_path, 'w') as f:
        json.dump({
            'scale': SCALE,
            'palette': [list(c) for c in PALETTE],
            'frames': frames,
        }, f)
    print(f'Wrote {json_path}')


if __name__ == '__main__':
    main()
