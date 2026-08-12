#!/usr/bin/env python3
"""
Lightweight visual validation for the generated robot yaw sprites.

Checks:
  * reference   - check-front.png matches robot-proposal-192-front.png
  * diagonals   - the four diagonal outputs are 192x192 and use the full height
  * cardinals   - generated check-*.png files are non-empty and 192x192
  * all         - run reference + diagonals + cardinals
"""
from PIL import Image
import argparse
import os
import sys

SIZE = 192
PLANS = os.path.dirname(os.path.abspath(__file__))


def load(name):
    return Image.open(os.path.join(PLANS, name)).convert('RGBA')


def compare(ref_name, check_name):
    ref = load(ref_name)
    chk = load(check_name)
    same = both = ref_only = chk_only = 0
    for a, b in zip(ref.get_flattened_data(), chk.get_flattened_data()):
        if a == b:
            same += 1
        elif a[3] > 0 and b[3] > 0:
            both += 1
        elif a[3] > 0:
            ref_only += 1
        elif b[3] > 0:
            chk_only += 1
    total = SIZE * SIZE
    return {
        'same': same,
        'both': both,
        'ref_only': ref_only,
        'chk_only': chk_only,
        'match_pct': 100.0 * same / total,
    }


def measure(name):
    img = load(name)
    xs = [x for x in range(SIZE) for y in range(SIZE) if img.getpixel((x, y))[3] > 0]
    ys = [y for x in range(SIZE) for y in range(SIZE) if img.getpixel((x, y))[3] > 0]
    if not xs:
        return None
    return {
        'x0': min(xs),
        'y0': min(ys),
        'x1': max(xs),
        'y1': max(ys),
        'width': max(xs) - min(xs) + 1,
        'height': max(ys) - min(ys) + 1,
        'opaque': len(xs),
    }


def check_reference():
    ok = True
    stats = compare('robot-proposal-192-front.png', 'check-front.png')
    print(f"reference  check-front vs robot-proposal-192-front: "
          f"match={stats['match_pct']:.2f}%  same={stats['same']}  "
          f"ref_only={stats['ref_only']}  gen_only={stats['chk_only']}")
    if stats['match_pct'] < 99.0:
        print("  FAIL: front check should be an exact (or near-exact) reproduction")
        ok = False
    else:
        print("  PASS")
    return ok


def check_diagonals():
    ok = True
    for name in ['front-right', 'back-right', 'back-left', 'front-left']:
        m = measure(f'robot-proposal-192-{name}.png')
        if m is None:
            print(f"diagonals  {name}: EMPTY")
            ok = False
            continue
        print(f"diagonals  {name}: bbox=[{m['x0']},{m['y0']},{m['x1']},{m['y1']}] "
              f"w={m['width']} h={m['height']} opaque={m['opaque']}")
        if m['height'] != SIZE:
            print(f"  FAIL: height {m['height']} != {SIZE}")
            ok = False
        else:
            print("  PASS")
    return ok


def check_cardinals():
    ok = True
    for name in ['front', 'right', 'back', 'left']:
        m = measure(f'check-{name}.png')
        if m is None:
            print(f"cardinals  {name}: EMPTY")
            ok = False
            continue
        print(f"cardinals  {name}: bbox=[{m['x0']},{m['y0']},{m['x1']},{m['y1']}] "
              f"w={m['width']} h={m['height']} opaque={m['opaque']}")
        if m['height'] == 0:
            print("  FAIL: empty output")
            ok = False
        else:
            print("  PASS")
    return ok


def main():
    parser = argparse.ArgumentParser(description='Validate generated robot sprites')
    parser.add_argument('--mode', choices=['reference', 'diagonals', 'cardinals', 'all'],
                        default='all')
    args = parser.parse_args()

    ok = True
    if args.mode in ('reference', 'all'):
        ok = check_reference() and ok
    if args.mode in ('diagonals', 'all'):
        ok = check_diagonals() and ok
    if args.mode in ('cardinals', 'all'):
        ok = check_cardinals() and ok

    if ok:
        print("\nAll requested checks passed.")
        return 0
    else:
        print("\nSome checks failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
