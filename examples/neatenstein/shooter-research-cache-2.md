# Research Cache 2: Walking Animation Vertical Jump

## Root Cause Analysis
The whole enemy jumps vertically because the **walk sprite data itself is baked with a whole-body vertical offset**. In `generate-robot-sprites.py`, `_pose_params` assigns `y_off = -1` to both `walk1` and `walk2` poses. Every body part is drawn 1 logical pixel higher than in `stand`. At 4× scale, this produces a **4-screen-pixel upward jump** every time the animation switches from `stand` to a walk frame.

The runtime renderer does NOT add any vertical offset — it simply centers the decoded frame on the horizon. The jump is entirely in the generated frame data.

## Code Locations
- `examples/neatenstein/generate-robot-sprites.py:560-569` — `_pose_params` returns `y_off = -1` for walk1/walk2
- `examples/neatenstein/generate-robot-sprites.py:74-339` — all `draw_*` helpers add `y_off` to every Y coordinate
- `examples/neatenstein/generate-robot-sprites.py:380-398` — leg stride detail (walk1: left y+34, right y+35; walk2: swapped)
- `examples/neatenstein/robot-sprite-data.js` — generated data with baked offset
- `examples/neatenstein/browser-entry/renderer/sprites.ts:614-685` — frame resolution (no Y offset applied)
- `examples/neatenstein/browser-entry/renderer/sprites.ts:736-805` — projection centers on `height/2`
- `examples/neatenstein/scripts/enemy-controller.ts:454-455` — walkTick increment (no Y manipulation)
- `examples/neatenstein/robot-sprite-preview.html:97-115` — reference (masks the bug by compositing at fixed Y)

## Comparison with Reference
`robot-sprite-preview.html` composites the upper body from `shoot` pose (offset 0) and draws the lower-body crop at fixed canvas Y=140. The global `y_off = -1` only shifts leg pixels within the crop, not the whole sprite. The game renders the full decoded frame, so the offset shifts the entire robot.

## Key Findings
1. No runtime vertical offset exists — renderer centers every sprite on `height/2`
2. Jump is baked into sprite data: `y_off = -1` for walk1/walk2, `y_off = 0` for stand/shoot
3. At 4× scale, 1 logical pixel = 4 screen pixels of vertical jump
4. Walk cycle `stand→walk1→stand→walk2` amplifies perception — jumps up every time it leaves stand
5. Leg animation itself (stride swap) is correct WITHOUT the global offset
6. Composite shoot-walk frames also affected — 1-pixel mismatch between upper (offset 0) and lower (offset -1)

## Proposed Fix
Remove `y_off = -1` from walk poses in the sprite generator:
```python
def _pose_params(pose):
    if pose == 'walk1':
        return 0, 1, False   # was -1, 1, False
    if pose == 'walk2':
        return 0, 2, False   # was -1, 2, False
```
Then regenerate `robot-sprite-data.js` and `robot-sprite-data.json` from the updated script.

No changes needed in sprites.ts, display.worker.ts, or enemy-controller.ts — renderer and controller already handle pose selection correctly.

## Files to Change
1. `examples/neatenstein/generate-robot-sprites.py` — remove `y_off = -1` from walk1/walk2
2. `examples/neatenstein/robot-sprite-data.js` — regenerate
3. `examples/neatenstein/robot-sprite-data.json` — regenerate
4. `plans/robot-proposal-192-*.png` — regenerate reference PNGs (if they exist)