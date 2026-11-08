# editor

Editor undo stack for NeatRacing (slice P1-S6-B; spec section 8.3 /
AC-020 and AC-036 of `plans/neatracing-voxel-racer.spec.md`): the
count-bounded 3-step undo the editor shell drives while painting the
dense working volume (spec section 8.3, LD-28: "3-step undo/redo;
oldest evicted").

Snapshots are dense `Uint8Array` copies of the working volume taken at
{@linkcode EditorUndo.push} time, so a snapshot never aliases the live
volume and later paints never corrupt one. {@linkcode EditorUndo.undo}
restores the newest retained snapshot into the live volume in place —
the volume reference stays stable so renderers holding it keep seeing
the restored state — and reports `false` on an empty stack. The stack
retains at most {@linkcode NEATRACING_EDITOR_UNDO_MAX_STEPS} dense
snapshots and evicts the oldest one beyond that cap, keeping the total
snapshot bytes tracked by the stack within
{@linkcode NEATRACING_EDITOR_UNDO_BUDGET_BYTES} for any volume up to
the editor working volume's size.

The named budget derives structurally from the editor volume length
(3 × 2,097,152 = 6,291,456 bytes ≈ 6.3 MB), so a full cap of
editor-volume snapshots can never exceed the ≤ 8 MB editor undo
budget the spec names (appendix P7's dense-storage exception, separate
from the game runtime's sparse rule, AC-036).

The module stays pure data — no DOM, no WebGL context, and no RNG —
so the undo stack is testable in the plain Node Jest environment and
the editor shell binds it straight onto the painting workflow.

## editor/editor.undo.ts

### assertUndoVolumeWithinBudget

```ts
assertUndoVolumeWithinBudget(
  volume: Uint8Array<ArrayBufferLike>,
): void
```

Rejects volumes too large for the undo budget: a volume longer than
the editor working volume would let a full 3-step snapshot stack
exceed {@linkcode NEATRACING_EDITOR_UNDO_BUDGET_BYTES}.

Parameters:
- `volume` - Volume the undo stack would snapshot and restore into.

### createEditorUndo

```ts
createEditorUndo(
  volume: Uint8Array<ArrayBufferLike>,
): EditorUndo
```

Creates the editor undo stack bound to the working volume it snapshots
and restores into.

The volume must be the editor's dense palette-index working volume (or
a smaller plain Node `Uint8Array` for tests): the stack keeps dense
copies of it, restores into it in place on
{@linkcode EditorUndo.undo}, and never reorders or rebuilds the
caller's array reference.

Parameters:
- `volume` - Dense working volume to snapshot on push and restore
into on undo.

Returns: Ready-to-use undo manager implementing  {@linkcode EditorUndo} .

Example:

```ts
import { createEditorState } from './editor.state';
import { createEditorUndo } from './editor.undo';
const state = createEditorState();
const undo = createEditorUndo(state.volume);
undo.push();
state.paint(0, 0, 0, [255, 0, 0, 255, 0]);
undo.undo(); // the paint is rolled back in place
```

### EditorUndo

Public editor undo surface the editor shell drives.

### evictOldestSnapshotsOverCap

```ts
evictOldestSnapshotsOverCap(
  snapshots: Uint8Array<ArrayBufferLike>[],
): void
```

Evicts the oldest dense snapshot whenever the stack grows past the
3-step undo cap (spec section 8.3: oldest evicted), keeping the
retained byte total within the named budget.

Parameters:
- `snapshots` - Dense snapshots ordered oldest (index 0) first.

### NEATRACING_EDITOR_UNDO_BUDGET_BYTES

Named editor undo byte budget (spec section 8.3; AC-020, AC-036
partial): three dense snapshots of the full editor working volume —
3 × 2,097,152 = 6,291,456 bytes ≈ 6.3 MB — within the ≤ 8 MB editor
undo budget the spec names (appendix P7). Deriving the budget from
{@linkcode NEATRACING_EDITOR_VOLUME_LENGTH} keeps the guarantee
structural: a full {@linkcode NEATRACING_EDITOR_UNDO_MAX_STEPS}-step
stack of editor-volume snapshots can never exceed the named budget.

### NEATRACING_EDITOR_UNDO_MAX_STEPS

Maximum number of dense snapshots the editor undo stack retains
(spec section 8.3, LD-28: a count-bounded 3-step undo; the oldest
snapshot is evicted once the cap is reached).

### sumSnapshotBytes

```ts
sumSnapshotBytes(
  snapshots: readonly Uint8Array<ArrayBufferLike>[],
): number
```

Sums the byte lengths of the retained dense snapshots (AC-036 partial:
dense snapshot accounting asserted from live state, not assumed).

Parameters:
- `snapshots` - Dense snapshots ordered oldest (index 0) first.

Returns: Total snapshot bytes currently tracked by the stack.

## editor/editor-entry.ts

### createEditorPreviewBinsDraw

```ts
createEditorPreviewBinsDraw(
  gl: WebGL2RenderingContext,
  previewCanvas: Pick<HTMLCanvasElement, "width" | "height">,
): EditorPreviewBinsDraw
```

Build the production bins draw seam over the engine car pass.

The adapter lazily builds a {@link CarPass} keyed on the bins identity: the
first draw (or any new bins object) disposes the previous pass and compiles
a pass rendering the preview bins as the car body over the empty wheel bin,
so exactly the selected document's geometry reaches the framebuffer. Car
passes are compiled eagerly and throw on unusable contexts; such build
failures keep the RAF chain alive (LD-1 — the preview lane must never
throw from a RAF frame) but are surfaced observably with the preview
label instead of the batch 54 silent no-op, so a blank preview can be
distinguished from a coarse probe by its console evidence.

Every drawn frame also applies the pass-local-policy prelude the preview
host owns (the car pass never touches viewport or clear state — the game
composer path has a host prelude, the preview lane needs its own):
`viewport(0, 0, canvas.width, canvas.height)` plus a color AND depth clear.
Without the per-frame depth clear the translucent blob shadow keeps
re-blending into untouched depth, which burns the shadow in and ghosts
previous frames across every draw.

Parameters:
- `gl` - Live WebGL2 context the preview lane was gated on.
- `previewCanvas` - Structural canvas view supplying the per-frame
viewport extents (width/height only are read).

Returns: The draw-only seam; its `dispose` releases the current pass.

Example:

```ts
const binsDraw = createEditorPreviewBinsDraw(gl, canvas);
binsDraw.draw({ viewProjection, cameraWorld, bins });
binsDraw.dispose();
```

### createSessionStateView

```ts
createSessionStateView(
  toolframe: EditorToolframeSession,
): NeatRacingEditorState
```

Builds the session's live projection of the tool frame's editor state.

The projection reads the tool frame's state on every access, so the
session's model name follows the tool frame's selection (the initial
model load lands asynchronously) without the boot owning a second,
diverging copy of the selection.

Parameters:
- `toolframe` - Tool frame owning the editor state container.

Returns: The session's editor state container.

Example:

```ts
const view = createSessionStateView(toolframe);
console.log(view.model); // null until the initial model load lands
```

### NeatRacingEditorBootDeps

Optional boot dependencies for the editor boot.

The boot injects the model-store fetch seam through this object so the
initial model load chain (model-store list, read, decode into the working
volume) can be exercised end to end without a dev server. Production call
sites omit it and keep the ambient `globalThis.fetch`.

### NeatRacingEditorSession

Session handle returned by the editor boot.

Calling {@link NeatRacingEditorSession.stop} stops the AC-019 preview
render loop when one was started, then removes the mounted tool frame,
the preview canvas, and the tracked context-service fallback notice so an
embedding page or test can unmount the editor cleanly. The `state`
container is fresh per boot so successive boots never share state.

### neatRacingEditorStart

```ts
neatRacingEditorStart(
  previewHostId: string,
  deps: NeatRacingEditorBootDeps | undefined,
): NeatRacingEditorSession
```

Mount the WebGL2 preview canvas into the editor shell and return a session.

The boot mounts only what the shell needs in this slice: the preview canvas
plus the context service's fallback notice when WebGL2 is unavailable. The
canvas is sized to the preview host's client box at boot (clamped to a
one-pixel drawable minimum by {@link MIN_PREVIEW_CANVAS_SIZE}, since live
resize is a later slice's job). Live WebGL2 contexts then start the AC-019
preview render loop (the tool frame stays a later slice's job). The fresh
context is painted once at boot by a guarded clear (viewport, clearColor,
clear) so the preview never shows stale default-framebuffer noise before
the loop's first frame.

Parameters:
- `previewHostId` - Editor shell element id that owns the preview
canvas.
- `deps` - Optional boot dependencies (the model-store fetch seam).

Returns: A session with the fresh editor state container and a stop handle.

Example:

```ts
const session = window.neatRacingEditorStart('neatracing-editor-preview');
console.log(session.state); // { model: null }
session.stop(); // stops the preview loop and unmounts the canvas/notice
```

### NeatRacingEditorStart

```ts
NeatRacingEditorStart(
  previewHostId: string,
  deps: NeatRacingEditorBootDeps | undefined,
): NeatRacingEditorSession
```

Boot function installed on `window` by the editor bundle.

Parameters:
- `previewHostId` - Editor shell element id that owns the preview
canvas.
- `deps` - Optional boot dependencies (the model-store fetch seam).

Returns: A session with the fresh editor state and a stop handle.

### NeatRacingEditorState

Editor state container shared by the tool slices (AC-020 partial).

### pickPreviewRenderingContext

```ts
pickPreviewRenderingContext(
  rendering: RenderingContextInitResult,
): WebGL2RenderingContext | null
```

Pick the acquired rendering context when it is usable for the preview lane.

The preview needs more than a context container: `createCarPass` compiles
programs eagerly, so the gate checks `createShader` structurally. This keeps
jsdom-style stub contexts (`{}` cast to `WebGL2RenderingContext`) and any
`ok: false` acquisition from starting the loop, exactly matching the stub
boot contracts pinned by editor-entry.test.ts.

Parameters:
- `rendering` - Shared context-service acquisition result.

Returns: The live WebGL2 context, or null when the preview lane must not
start.

Example:

```ts
const gl = pickPreviewRenderingContext(rendering);
if (gl !== null) startPreview(gl);
```

## editor/editor.state.ts

Editor volume state for NeatRacing (slice P1-S6-A; spec section 8.3 /
AC-019 and AC-020 of `plans/neatracing-voxel-racer.spec.md`): the dense
128-cubed working volume the editor paints into before export.

The editor is an explicitly permitted exception to the sparse-storage
rule (R-07, appendix P7): it owns a single dense
{@linkcode NEATRACING_EDITOR_VOLUME_LENGTH} = 128³ = 2,097,152 byte
`Uint8Array` (~2.1 MB) where each voxel stores a palette index, and 0
stays reserved as the EMPTY sentinel entry (air; never rendered,
section 7.2). Voxels address as
`x + {@linkcode NEATRACING_EDITOR_VOLUME_ROW_STRIDE} * y +
{@linkcode NEATRACING_EDITOR_VOLUME_SLICE_STRIDE} * z`:
x = column, y = row (row 0 = down), z = slice (slice 0 = front).

{@linkcode EditorState.paint} delegates color resolution to the editor
palette (exact-else-nearest, AC-020) and writes the resolved palette
index, so painting the same color twice reuses its entry instead of
duplicating it, and {@linkcode EditorState.selectModel} records the
currently selected model name for the editor shell (null until the
model dropdown first resolves a selection).

The module stays pure data — no DOM and no WebGL context — so the
editor volume is testable in the plain Node Jest environment while the
shell and renderer consume its public reads.

### createEditorState

```ts
createEditorState(
  config: EditorStateConfig | undefined,
): EditorState
```

Creates the editor state with defaults plus partial overrides.

Parameters:
- `config` - Partial state configuration to spread over defaults.

Returns: Ready-to-use editor state implementing  {@linkcode EditorState} .

Example:

```ts
import { createEditorState } from './editor.state';
const state = createEditorState();
state.paint(15, 0, 15, [255, 0, 0, 255, 0]);
state.selectModel('car-f1');
```

### EditorState

Public editor state surface the editor shell drives.

### EditorStateConfig

Factory configuration for an editor state.

### EditorStateSnapshot

Immutable point-in-time view of the editor's document-level state.

`model` is the currently selected model name, or `null` while no
model selection has been made.

### NEATRACING_EDITOR_VOLUME_EDGE

Edge length, in voxels, of the editor's cubic working volume.

### NEATRACING_EDITOR_VOLUME_LENGTH

Total element count of the editor volume: 128 × 128 × 128 = 2,097,152
palette-index bytes (the editor's dense-storage exception; appendix P7).

### NEATRACING_EDITOR_VOLUME_ROW_STRIDE

Elements to skip per step along the y (row) axis.

### NEATRACING_EDITOR_VOLUME_SLICE_STRIDE

Elements to skip per step along the z (slice) axis.

### validateVoxelCoordinate

```ts
validateVoxelCoordinate(
  axisName: string,
  value: number,
): void
```

Validates one voxel coordinate against the volume edge, reporting the
failing axis name in the thrown range error.

Parameters:
- `axisName` - Axis label for error messages ('x', 'y', or 'z').
- `value` - Coordinate value to validate.

## editor/editor.camera.ts

Editor camera controls for NeatRacing (slice P1-S5-D; spec section 13
AC-019, editor navigation SL-28 of `plans/neatracing-voxel-racer.spec.md`):
orbit on drag, scroll zoom with clamped distance, CAD-style orthographic
front/side/top views, screen-plane pan, and inertial exponential smoothing
between ticks.

The module stays pure numeric state — it touches no DOM and no WebGL
context — so the state is testable inside the plain Node Jest environment
and the future editor shell (slice P1-S5-C) only feeds drag deltas, wheel
lines, and ticks into {@linkcode EditorCamera} and uploads the built
matrices into the game renderer's combined view-projection uniform slot.

Conventions match the game engine passes the editor re-implements nothing
from: matrices are column-major 4x4 `Float32Array(16)` and the world is
Y-up, so the look-at basis is right = normalize(forward × worldUp), the
camera up frame completes the right-handed basis, and view space looks
down −Z. Camera yaw/pitch orbit a focus `center` on a spherical shell of
radius `distance`, exactly like the CAD panels the spec calls out in
SL-28, and every mutation lands on a *target* pose that
{@linkcode EditorCamera.tick} eases the *current* pose toward — the
inertial smoothing the AC demands. CAD orthographic snaps bypass the
orbit pitch clamp BY DESIGN: the top view pitch (π/2) legitimately points
straight down, which free-drag orbiting must never reach.

### clampValue

```ts
clampValue(
  value: number,
  min: number,
  max: number,
): number
```

Clamps a value into an inclusive range.

Parameters:
- `value` - Number to clamp.
- `min` - Inclusive lower bound.
- `max` - Inclusive upper bound.

Returns: Clamped number.

### clonePose

```ts
clonePose(
  pose: EditorCameraPose,
): EditorCameraPose
```

Copies a pose deeply so snapshots and target writes never alias state.

Parameters:
- `pose` - Pose to copy.

Returns: Detached copy.

### createEditorCamera

```ts
createEditorCamera(
  overrides: Partial<EditorCameraConfig> | undefined,
): EditorCamera
```

Creates the editor camera state with defaults plus partial overrides.

Parameters:
- `overrides` - Partial camera config to spread over the defaults.

Returns: Ready-to-drive camera implementing  {@linkcode EditorCamera} .

Example:

```ts
import { createEditorCamera } from './editor.camera';
const camera = createEditorCamera({ distance: 64 });
camera.orbit(0.2, -0.1);
camera.tick(0.016);
```

### cross3

```ts
cross3(
  a: [number, number, number],
  b: [number, number, number],
): [number, number, number]
```

Crosses two 3-vectors into a new vector.

Parameters:
- `a` - First vector.
- `b` - Second vector.

Returns: `a × b`.

### dot3

```ts
dot3(
  a: [number, number, number],
  b: [number, number, number],
): number
```

Dots two 3-vectors.

Parameters:
- `a` - First vector.
- `b` - Second vector.

Returns: `a · b`.

### EditorCamera

Public editor camera surface the canvas shell drives per frame.

### EditorCameraConfig

Immutable camera tuning the factory spreads user overrides over.

### EditorCameraMode

Projection mode of the editor camera (AC-019 CAD views vs orbit view).

### EditorCameraPose

Orbit pose of the camera: focus center, spherical angles, and distance.

### EditorCameraSnapshot

Point-in-time view of both camera poses plus the active CAD state.

### EditorOrthoView

CAD orthographic projection the editor preview can snap to.

### NEATRACING_EDITOR_CAMERA_DEFAULTS

### NEATRACING_EDITOR_CAMERA_MAX_DISTANCE

Largest camera distance in world units before zoom clamps (AC-019).

### NEATRACING_EDITOR_CAMERA_MAX_ORBIT_PITCH_RAD

Free-drag orbit pitch clamp in radians (±80°) — CAD orthographic snaps
bypass this clamp on purpose so the top view can point straight down.

### NEATRACING_EDITOR_CAMERA_MIN_DISTANCE

Smallest camera distance in world units before zoom clamps (AC-019).

### NEATRACING_EDITOR_CAMERA_ORTHO_HEIGHT_PER_DISTANCE

Ortho frustum half-height as a fraction of the current orbit distance.

### NEATRACING_EDITOR_CAMERA_PAN_FRACTION_OF_DISTANCE

Pan speed as a fraction of the current orbit distance per screen unit.

### NEATRACING_EDITOR_CAMERA_SMOOTHING_TAU_S

Inertial smoothing time constant in seconds: each `tick(dtSeconds)` moves
the current pose by `ease = 1 − exp(−dt ÷ τ)` of the gap to the target
pose, the deterministic exponential ease the AC-019 tests lock in.

### NEATRACING_EDITOR_CAMERA_ZOOM_LOG_PER_LINE

Log-scale zoom per scroll-wheel line: the orbit distance multiplies by
`exp(k · lines)`, so positive lines zoom out and negative lines zoom in.

### normalize3

```ts
normalize3(
  vector: [number, number, number],
): [number, number, number]
```

Normalizes a 3-vector in place into a fresh array.

Parameters:
- `vector` - Vector to normalize.

Returns: Unit-length vector, or the input when the length is zero.

### snapTrigZero

```ts
snapTrigZero(
  value: number,
): number
```

Snaps trig values that fall within float noise of zero onto exact zero so
CAD axis-aligned views reproduce exact integer matrices instead of values
like `cos(π/2) ≈ 6.1e−17`.

Parameters:
- `value` - Trig evaluation result.

Returns: Exact zero when the magnitude is float-noise-small, else `value`.

## editor/editor.palette.ts

Editor palette state for NeatRacing (slice P1-S6-A; spec section 8.3 /
AC-020 of `plans/neatracing-voxel-racer.spec.md`): the pure palette
object that manages the on-disk `[r, g, b, a, glow]` entries behind the
editor's color picker.

Palette index 0 is reserved as the EMPTY sentinel entry
(`VOXEL_EMPTY_PALETTE_ENTRY` — air; never rendered, never counted,
section 7.2), so real color entries occupy palette indices
1..maxEntries over a default of
{@linkcode NEATRACING_EDITOR_PALETTE_MAX_ENTRIES} = 128 slots. Color
resolution is exact-else-nearest (AC-020):

1. An entry that already exists exactly (all five components) returns
   its existing index without allocating a slot; the all-zero entry
   resolves to the EMPTY sentinel index 0 itself.
2. A new entry claims the first free index.
3. A full palette resolves an unknown entry to the nearest existing
   color entry by Manhattan RGB distance, lowest palette index winning
   ties. Palette index 0 is never a nearest candidate because it is
   reserved air, not a paintable color.

The module stays pure data — no DOM, no WebGL context, and no RNG — so
the palette is testable in the plain Node Jest environment, and the
editor shell feeds it color picks while later phases serialize palette
entries into on-disk model JSON.

### buildSnapshotEntries

```ts
buildSnapshotEntries(
  colors: readonly VoxelPaletteEntry[],
): VoxelPaletteEntry[]
```

Builds the snapshot entries array: palette index 0 carries the EMPTY
sentinel entry on an empty palette, while on any palette holding colors
index 0 stays unfilled (a hole) because that slot is reserved EMPTY air
and is never a color entry — `Array.prototype.filter` skips the hole,
keeping `entries.filter(Boolean).length` equal to the color count.
Copies keep snapshots detached from live palette state.

Parameters:
- `colors` - Occupied color entries (palette index = position + 1).

Returns: Snapshot entries array indexed by palette index.

### clampToByteRange

```ts
clampToByteRange(
  value: number,
): number
```

Clamps a numeric component into the 0-255 palette byte range.

Parameters:
- `value` - Requested component value.

Returns: Value clamped into the byte range.

### cloneEntry

```ts
cloneEntry(
  entry: PaletteEntryView,
): VoxelPaletteEntry
```

Copies a palette entry so stored state never aliases caller input and
snapshots never alias living entries.

Parameters:
- `entry` - Palette entry to copy.

Returns: Detached `[r, g, b, a, glow]` copy.

### createEditorPalette

```ts
createEditorPalette(
  config: EditorPaletteConfig | undefined,
): EditorPalette
```

Creates the editor palette with defaults plus partial overrides.

Parameters:
- `config` - Partial palette configuration to spread over defaults.

Returns: Ready-to-use palette implementing  {@linkcode EditorPalette} .

Example:

```ts
import { createEditorPalette } from './editor.palette';
const palette = createEditorPalette({ maxEntries: 4 });
const redIndex = palette.addColor([255, 0, 0, 255, 0]);
```

### EditorPalette

Public editor palette surface the color picker and paint resolver drive.

### EditorPaletteConfig

Factory configuration for an editor palette.

### EditorPaletteSnapshot

Immutable point-in-time view of the palette entries and occupancy.

`entries` is indexed by palette index. An empty palette exposes exactly
the EMPTY sentinel entry at index 0. A palette holding colors leaves
index 0 as a hole (runtime `undefined`, despite the declared element
type) because that slot is reserved EMPTY air and is never a color
entry — `Array.prototype.filter` skips the hole, so
`snapshot.entries.filter(Boolean).length` always equals
{@linkcode EditorPaletteSnapshot.usedCount}.

### entriesEqual

```ts
entriesEqual(
  a: PaletteEntryView,
  b: PaletteEntryView,
): boolean
```

Compares two palette entries component by component (exact match rule).

Parameters:
- `a` - First entry.
- `b` - Second entry.

Returns: True when every one of the five components matches exactly.

### nearestColorIndex

```ts
nearestColorIndex(
  entry: VoxelPaletteEntry,
  colors: readonly VoxelPaletteEntry[],
): number
```

Finds the color entry nearest an entry by Manhattan RGB distance. Only
real color entries are candidates — palette index 0 is reserved EMPTY
air (section 7.2) and must never resolve a paint. Ties resolve to the
lowest palette index for determinism.

Parameters:
- `entry` - Entry to locate the nearest color for.
- `colors` - Occupied color entries (palette index = position + 1).

Returns: Palette index of the nearest existing color entry.

### NEATRACING_EDITOR_PALETTE_EMPTY_INDEX

Palette index reserved as EMPTY (air; never rendered, section 7.2).

### NEATRACING_EDITOR_PALETTE_MAX_ENTRIES

Maximum number of real color entries the editor palette holds (AC-020:
the 128-color palette). Palette index 0 is the reserved EMPTY sentinel
and does not count against this budget.

### PaletteEntryView

Any palette entry view, including the immutable
`VOXEL_EMPTY_PALETTE_ENTRY` sentinel constant from the schema.

### requireColorEntry

```ts
requireColorEntry(
  index: number,
  state: EditorPaletteState,
): VoxelPaletteEntry
```

Resolves a palette index to its stored color entry, rejecting the
reserved EMPTY sentinel and out-of-range indices.

Parameters:
- `index` - Palette index to resolve.
- `state` - Palette state the index resolves against.

Returns: The stored color entry (mutable for tuning).

### rgbDistance

```ts
rgbDistance(
  entry: VoxelPaletteEntry,
  r: number,
  g: number,
  b: number,
): number
```

Measures the Manhattan RGB distance from an entry to target channels.

Parameters:
- `entry` - Existing color entry to measure from.
- `r` - Target red channel.
- `g` - Target green channel.
- `b` - Target blue channel.

Returns: Manhattan distance over the RGB channels (0 by construction).

## editor/editor.controls.ts

### createEditorControls

```ts
createEditorControls(
  config: EditorControlsConfig,
): EditorControls
```

Creates the editor control behaviors over a mounted tool frame.

Parameters:
- `config` - Tool frame, camera, and the working-volume change hook.

Returns: Ready-to-drive controls implementing  {@linkcode EditorControls} .

Example:

```ts
const controls = createEditorControls({ toolframe, camera });
controls.selectPaletteEntry(controls.addPaletteEntry([255, 0, 0, 255, 0]));
```

### EditorControls

Public editor control surface the editor shell drives.

### EditorControlsConfig

Factory configuration for the editor control behaviors.

### NEATRACING_EDITOR_CONTROL_IDS

Ids of the affordances this module mounts next to the palette list.

### readPickedColorEntry

```ts
readPickedColorEntry(
  hexValue: string,
): VoxelPaletteEntry
```

Converts a color input's `#rrggbb` value into a palette entry.

Parameters:
- `hexValue` - Color input value (a six-digit hex triple).

Returns: The `[r, g, b, a, glow]` palette entry the value names.

## editor/editor.symmetry.ts

Mirror symmetry painting for the NeatRacing editor (slice P1-S6-C; spec
sections 7.2 / 8.3, AC-021, LD-28): deterministic expansion of one paint
voxel into every mirrored position across the enabled axes.

The mapping is identical on every axis: on the 128-cubed editor volume a
coordinate `c` mirrors to `{@linkcode NEATRACING_EDITOR_MIRROR_MAX_COORDINATE} - c`
(col c <-> col 127-c; spec section 7.2 stores only the canonical half,
cols 0..63, so a paint above the mirror line canonicalizes down first —
`map(127, y, z)` emits the same ordered set as `map(0, y, z)`).

{@linkcode EditorSymmetry.map} emits the canonical primary tuple first,
then exactly one tuple per enabled-axis mirror subset in ascending subset
order (x < y < xy < z < xz < yz < xyz), so combined modes compose into the
full axis-power set (1–8 tuples) in a stable, duplicate-free order.
Mirror-X symmetry is enabled by default (LD-28 — only half is stored);
mirror-Y and mirror-Z are opt-in, and an all-flags-off configuration
leaves only the primary voxel (disabled axes pass through unchanged).
Coordinates outside 0..127 are rejected with the same
{@linkcode RangeError} guard the editor volume paint path applies.

The module stays pure — no DOM, no WebGL, no random source — so the
expansion is exactly deterministic: same flags and same coordinates in,
the same ordered tuple list out on every call and runtime.

### applyMirrorSubset

```ts
applyMirrorSubset(
  axes: MirroredAxisName[],
  mask: number,
  base: EditorSymmetryTarget,
): EditorSymmetryTarget
```

Applies one mirror-subset bitmask to the canonical tuple.

Bit `i` mirrors across `axes[i]`, so iterating masks 1..2^n-1 in order
yields exactly the ascending subset order (x < y < xy < z < xz < yz <
xyz) the emission contract promises.

Parameters:
- `axes` - Enabled mirror axes in mask-bit order.
- `mask` - Subset bitmask selecting which axes to mirror.
- `base` - Canonical tuple to mirror from.

Returns: The tuple with every masked axis mirrored.

### canonicalizeMirrorCoordinate

```ts
canonicalizeMirrorCoordinate(
  value: number,
): number
```

Folds one coordinate into the stored mirror half of its axis pair.

Spec section 7.2 stores only cols 0..63 (the canonical half); a paint
above the mirror line canonicalizes down so the primary tuple is
volume-canonical before mirror subsets are emitted.

Parameters:
- `value` - Coordinate to fold.

Returns: The lower member of `{value, mirror(value)}`.

### canonicalizePaint

```ts
canonicalizePaint(
  x: number,
  y: number,
  z: number,
  mirrorX: boolean,
  mirrorY: boolean,
  mirrorZ: boolean,
): EditorSymmetryTarget
```

Canonicalizes a paint coordinate per enabled axis, leaving disabled axes
untouched.

Parameters:
- `x` - Painted x coordinate.
- `y` - Painted y coordinate.
- `z` - Painted z coordinate.
- `mirrorX` - Whether the x axis is mirrored.
- `mirrorY` - Whether the y axis is mirrored.
- `mirrorZ` - Whether the z axis is mirrored.

Returns: The canonical primary tuple for the paint.

### collectEnabledAxes

```ts
collectEnabledAxes(
  mirrorX: boolean,
  mirrorY: boolean,
  mirrorZ: boolean,
): MirroredAxisName[]
```

Collects the enabled mirror axes in ascending subset-bit order.

Parameters:
- `mirrorX` - Whether the x axis is mirrored.
- `mirrorY` - Whether the y axis is mirrored.
- `mirrorZ` - Whether the z axis is mirrored.

Returns: Enabled axis names ordered x, then y, then z (mask bit 0..2).

### collectMirrorSubsets

```ts
collectMirrorSubsets(
  base: EditorSymmetryTarget,
  axes: MirroredAxisName[],
): EditorSymmetryTarget[]
```

Emits the full ordered symmetric target list for a canonical paint.

Collects the canonical primary first, then walks every enabled-axis
mirror subset mask in ascending order, producing 2^n duplicate-free
tuples (n = enabled axis count; n = 0 leaves only the primary).

Parameters:
- `base` - Canonical primary tuple.
- `axes` - Enabled mirror axes in mask-bit order.

Returns: Ordered mirrored positions, primary first.

### createEditorSymmetry

```ts
createEditorSymmetry(
  config: EditorSymmetryConfig | undefined,
): EditorSymmetry
```

Creates the editor mirror-symmetry expansion policy.

The expansion is pure and exactly deterministic (Level 3 at the call
boundary): identical flags and coordinates always produce the identical
ordered tuple list on every runtime, with the mirrors emitted in
ascending subset order (x < y < xy < z < xz < yz < xyz).

Parameters:
- `config` - Partial axis configuration; mirror-X defaults to `true`
(LD-28 — only half the volume is stored), mirror-Y and mirror-Z default
to `false` unless explicitly overridden.

Returns: Ready-to-use symmetry expansion implementing
 *  {@linkcode EditorSymmetry} .

Example:

```ts
import { createEditorSymmetry } from './editor.symmetry';
const xy = createEditorSymmetry({ mirrorY: true });
xy.map(10, 20, 30); // [[10, 20, 30], [117, 20, 30], [10, 107, 30], [117, 107, 30]]
const off = createEditorSymmetry({ mirrorX: false });
off.map(10, 20, 30); // [[10, 20, 30]]
```

### EditorSymmetry

Public mirror-symmetry surface the editor paint path drives.

### EditorSymmetryConfig

Factory configuration for mirror symmetry painting.

### EditorSymmetryTarget

One mirrored voxel position, addressed as `[x, y, z]`.

### mirrorAlongAxis

```ts
mirrorAlongAxis(
  axis: MirroredAxisName,
  x: number,
  y: number,
  z: number,
): EditorSymmetryTarget
```

Mirrors one tuple across exactly one axis, leaving the others unchanged.

Parameters:
- `axis` - Axis to mirror ('x', 'y', or 'z').
- `x` - Current x coordinate.
- `y` - Current y coordinate.
- `z` - Current z coordinate.

Returns: The tuple with the named axis coordinate mirrored.

### mirrorCoordinate

```ts
mirrorCoordinate(
  value: number,
): number
```

Mirrors one coordinate across its axis under the 127 - c rule.

Parameters:
- `value` - Coordinate to mirror.

Returns: The mirrored coordinate (c and the result are always distinct on
the 128-volume, so mirroring never produces a self-duplicate).

### MirroredAxisName

Names of the axes mirror symmetry can expand a paint across.

### NEATRACING_EDITOR_DEFAULT_MIRROR_X

Default mirror-X flag: x symmetry paints are on by default (LD-28).

### NEATRACING_EDITOR_DEFAULT_MIRROR_Y

Default mirror-Y flag: y symmetry is opt-in.

### NEATRACING_EDITOR_DEFAULT_MIRROR_Z

Default mirror-Z flag: z symmetry is opt-in.

### NEATRACING_EDITOR_MIRROR_MAX_COORDINATE

Highest inclusive voxel coordinate on any editor axis: 128 - 1 = 127.
The mirror mapping is `c <-> NEATRACING_EDITOR_MIRROR_MAX_COORDINATE - c`,
so a paint at 0 expands across the mirror to 127 and vice versa.

### validateSymmetryCoordinate

```ts
validateSymmetryCoordinate(
  axisName: string,
  value: number,
): void
```

Validates one symmetry coordinate against the volume edge, reporting the
failing axis name in the thrown range error (mirrors the paint-path guard
in `editor.state.ts`).

Parameters:
- `axisName` - Axis label for the error message ('x', 'y', or 'z').
- `value` - Coordinate value to validate.

## editor/editor.edit.loop.ts

### buildCameraRay

```ts
buildCameraRay(
  camera: EditorCamera,
  viewport: EditorPickerViewport,
  position: EditorPointerPosition,
): EditorCameraRay
```

Builds the world-space camera ray under a canvas-local pointer position.

The ray follows the editor camera's conventions: the column-major view
matrix supplies the right/up/back basis and the eye point, the
projection matrix supplies the perspective near plane (or the ortho
half-extents), and the canvas-local position maps into normalized device
coordinates through the CSS box.

Parameters:
- `camera` - Editor camera the ray is cast through.
- `viewport` - Viewport metrics of the canvas.
- `position` - Pointer position inside the canvas box.

Returns: The world-space ray.

### buildDocumentPalette

```ts
buildDocumentPalette(
  palette: EditorPalette,
): VoxelPaletteEntry[]
```

Builds the section 7.2 document palette from the editor palette.

Palette index 0 is the reserved EMPTY sentinel (air); the editor
palette snapshot leaves that slot unfilled once colors exist, so the
document palette is rebuilt as `[EMPTY, ...colors]`.

Parameters:
- `palette` - Editor palette the document palette mirrors.

Returns: The document palette entries, index-aligned with the volume.

### buildEditorVolumeDocument

```ts
buildEditorVolumeDocument(
  config: EditorVolumeDocumentConfig,
): VoxelModelDocument
```

Projects the editor working volume into a section 7.2 model document.

The projection keeps the FULL width (the editor stores both mirror
halves), so the document declares `symmetric: false` and its rows cover
all 128 columns. The document palette is index-aligned with the editor
palette (index 0 is the EMPTY sentinel), so the volume's palette indices
address the document palette directly.

Parameters:
- `config` - Volume, palette, and optional model name.

Returns: The projected model document.

Example:

```ts
const document = buildEditorVolumeDocument({
  volume: state.volume,
  palette: state.palette,
  name: 'car-f1',
});
console.log(document.meta.dims); // [128, 128, 128]
```

### clampCellIndex

```ts
clampCellIndex(
  value: number,
): number
```

Clamps a cell index into the volume's inclusive 0..127 range.

Parameters:
- `value` - Raw cell index.

Returns: The clamped cell index.

### createEditorEditLoop

```ts
createEditorEditLoop(
  config: EditorEditLoopConfig,
): EditorEditLoop
```

Creates the pointer-pick edit loop over the editor canvas.

The loop owns the stroke semantics: the undo snapshot is captured once
per stroke (BEFORE the first paint of that stroke), every paint goes
through the symmetry expansion, every edit is routed through the
rebuild pipeline's `edit(sliceZ)` dirty window, and the stroke's end
flushes the pipeline with one full rebuild - but only when the stroke
actually painted something.

Parameters:
- `config` - Canvas, state, symmetry, undo, pipeline, and picker seams.

Returns: The edit loop implementing  {@linkcode EditorEditLoop} .

Example:

```ts
const loop = createEditorEditLoop({
  canvas,
  state: toolframe.state,
  symmetry: toolframe.symmetry,
  undo: toolframe.undo,
  pipeline,
  picker,
});
loop.attach();
```

### createEditorVoxelPicker

```ts
createEditorVoxelPicker(
  config: EditorVoxelPickerConfig,
): EditorVoxelPicker
```

Creates the editor voxel picker over the working volume.

The picker is pure math over the injected camera and volume: it casts
the camera ray under the pointer, maps it into the volume's grid space,
and returns the first occupied voxel along it. A ray that crosses no
occupied voxel falls back to the first cell the ray entered; a ray that
never enters the volume reports a miss.

Parameters:
- `config` - Volume, camera, and viewport accessor.

Returns: The picker implementing  {@linkcode EditorVoxelPicker} .

Example:

```ts
const picker = createEditorVoxelPicker({
  volume: state.volume,
  camera,
  readViewport: () => ({ cssWidth: 800, cssHeight: 600, aspect: 4 / 3 }),
});
const hit = picker.pick({ x: 400, y: 300 });
```

### EditorCameraRay

One world-space camera ray.

### EditorEditLoop

Public edit-loop surface the editor boot drives.

### EditorEditLoopConfig

Factory configuration for the editor edit loop.

### EditorGridSpan

The parameter span over which a ray stays inside the volume's box.

### EditorPickerViewport

Viewport metrics the picker needs to build the camera ray.

### EditorPointerPosition

Pointer position inside the canvas box, in CSS pixels.

### EditorVolumeDocumentConfig

Factory configuration for the working-volume document projection.

### EditorVoxelHit

One voxel of the editor working volume.

### EditorVoxelPicker

Picker seam the edit loop resolves pointer positions through.

### EditorVoxelPickerConfig

Factory configuration for the editor voxel picker.

### encodeRow

```ts
encodeRow(
  volume: Uint8Array<ArrayBufferLike>,
  z: number,
  y: number,
  paletteSize: number,
): string
```

Encodes one volume row into the section 7.2 RLE hex row string.

The row is a sequence of `<runLen:2hex><paletteIdx:2hex>` pairs; the
runs of one row sum exactly to the declared row width (128 cols).

Parameters:
- `volume` - Dense working volume.
- `z` - Voxel slice the row belongs to.
- `y` - Voxel row the row belongs to.
- `paletteSize` - Number of entries the document palette holds.

Returns: The RLE hex row string.

### encodeRun

```ts
encodeRun(
  runLength: number,
  paletteIndex: number,
): string
```

Encodes one RLE run into its hex pair.

Parameters:
- `runLength` - Run length in cells (1..255).
- `paletteIndex` - Palette index the run paints.

Returns: The `<runLen:2hex><paletteIdx:2hex>` hex pair.

### intersectGridBox

```ts
intersectGridBox(
  ray: EditorCameraRay,
): EditorGridSpan | null
```

Intersects a grid-space ray with the volume's bounding box.

Parameters:
- `ray` - Grid-space ray.

Returns: The parameter span inside the box, or null when the ray misses.

### isOccupied

```ts
isOccupied(
  volume: Uint8Array<ArrayBufferLike>,
  x: number,
  y: number,
  z: number,
): boolean
```

Reports whether one volume cell holds a painted voxel.

Parameters:
- `volume` - Dense working volume.
- `x` - Voxel column.
- `y` - Voxel row.
- `z` - Voxel slice.

Returns: true when the cell holds a non-EMPTY palette index.

### mapRayToGridSpace

```ts
mapRayToGridSpace(
  ray: EditorCameraRay,
): EditorCameraRay
```

Maps a world-space ray into the editor volume's grid space.

The grid space mirrors the engine's voxel expansion (the car pass maps
grid `x`/`z` to `(grid - dim * 0.5) * voxelSize` and grid `y` to
`grid * voxelSize`), so the inverse is `gridX = worldX / voxelSize + 64`,
`gridY = worldY / voxelSize`, and `gridZ = worldZ / voxelSize + 64`.

Parameters:
- `ray` - World-space ray.

Returns: The same ray expressed in grid cells.

### marchForOccupiedVoxel

```ts
marchForOccupiedVoxel(
  ray: EditorCameraRay,
  span: EditorGridSpan,
  entryCell: EditorVoxelHit,
  volume: Uint8Array<ArrayBufferLike>,
): EditorVoxelHit | null
```

Marches the grid cells along the ray and returns the first occupied
voxel.

The march is the classic Amanatides-Woo traversal: every step crosses
exactly one cell boundary, and the axes crossing at the same parameter
(a corner) step together so the ray never visits an off-ray cell.

Parameters:
- `ray` - Grid-space ray.
- `span` - The ray's parameter span inside the volume.
- `entryCell` - The cell the ray entered the volume through.
- `volume` - Dense working volume.

Returns: The first occupied voxel, or null when the ray crosses none.

### NEATRACING_EDITOR_DEFAULT_MODEL_NAME

Default model name the working-volume projection stamps into the
document metadata when the caller names none (the section 7.2 metadata
header requires a non-empty name).

### NEATRACING_EDITOR_DEFAULT_PAINT_ENTRY

Default paint entry the edit loop writes when the caller names no color:
an opaque white body color with no glow.

### nextBoundaryT

```ts
nextBoundaryT(
  ray: EditorCameraRay,
  axis: number,
  cell: number,
): number
```

Computes the ray parameter at which the ray leaves the current cell on
one axis.

Parameters:
- `ray` - Grid-space ray.
- `axis` - Axis index (0 = x, 1 = y, 2 = z).
- `cell` - Current cell index on that axis.

Returns: The boundary parameter, or infinity for a parallel axis.

### resolveEntryCell

```ts
resolveEntryCell(
  ray: EditorCameraRay,
  tEnter: number,
): EditorVoxelHit
```

Resolves the first cell the ray enters.

The entry point sits exactly on the volume boundary, so it is nudged a
hair along the ray before the cell index is taken - otherwise a ray
entering at the far boundary would resolve the out-of-range cell 128
instead of the cell 127 it actually enters.

Parameters:
- `ray` - Grid-space ray.
- `tEnter` - Ray parameter where the ray enters the volume.

Returns: The entered cell.

### resolvePaletteIndex

```ts
resolvePaletteIndex(
  value: number,
  paletteSize: number,
): number
```

Clamps a volume palette index into the document palette's range.

Parameters:
- `value` - Raw palette index stored in the volume.
- `paletteSize` - Number of entries the document palette holds.

Returns: The index to write, or 0 (EMPTY) when it is out of range.

## editor/editor.live.save.ts

### buildDocumentPalette

```ts
buildDocumentPalette(
  palette: EditorPalette,
): VoxelPaletteEntry[]
```

Builds the index-aligned document palette: the EMPTY sentinel at index 0
plus every real color entry, so the volume's palette indices address
the document palette directly.

Parameters:
- `palette` - Editor palette the volume's indices address.

Returns: The document palette (index 0 is the EMPTY sentinel).

### buildEditorLiveSaveDocument

```ts
buildEditorLiveSaveDocument(
  config: EditorLiveSaveDocumentConfig,
): VoxelModelDocument
```

Projects the editor working volume into a section 7.2 model document
for the model store.

The document palette is index-aligned with the editor palette (index 0
is the EMPTY sentinel), so the volume's palette indices address the
document palette directly. The stored width follows the requested
storage form: the full 128 columns, or the LEFT 64 columns the loader
mirrors back onto the full width.

Parameters:
- `config` - Volume, palette, model name, and storage form.

Returns: The projected model document.

Example:

```ts
const document = buildEditorLiveSaveDocument({
  volume: state.volume,
  palette: state.palette,
  name: 'car-f1',
  symmetric: true,
});
console.log(document.meta.symmetric); // true
```

### createEditorLiveSave

```ts
createEditorLiveSave(
  config: EditorLiveSaveConfig,
): EditorLiveSave
```

Creates the immediate live save.

The save serializes the working volume into a section 7.2 document,
PUTs it to the model store, and triggers the rebuild flight. The flight
is latest-wins: the saves enqueued while a request is in flight
coalesce into ONE follow-up request carrying the current volume.

Parameters:
- `config` - Model name, volume source, and the optional fetch seam.

Returns: The live-save surface the editor boot drives.

Example:

```ts
const liveSave = createEditorLiveSave({
  modelName: () => toolframe.state.snapshot().model,
  readSource: () => ({
    volume: toolframe.state.volume,
    palette: toolframe.palette,
  }),
  fetchImpl: deps?.saveFetch,
});
liveSave.enqueueSave();
```

### createEditorLiveSymmetry

```ts
createEditorLiveSymmetry(
  config: EditorLiveSymmetryConfig,
): EditorLiveSymmetry
```

Creates the live mirror-X symmetry policy.

The toggle is read at PAINT time, so flipping it mid-session changes
the next paint without rebuilding the policy. The mirror-Y and
mirror-Z flags stay frozen on the base policy (the slice owns the
mirror-X toggle only).

Parameters:
- `config` - The mirror-X toggle plus the optional frozen base
policy.

Returns: The live symmetry surface the edit loop drives.

Example:

```ts
const toggle = { checked: true };
const symmetry = createEditorLiveSymmetry({ toggle });
symmetry.map(10, 20, 30); // [[10, 20, 30], [117, 20, 30]]
toggle.checked = false;
symmetry.map(10, 20, 30); // [[10, 20, 30]]
```

### describeError

```ts
describeError(
  error: unknown,
): string
```

Describes a thrown value for a labeled failure message.

Parameters:
- `error` - The thrown value.

Returns: A short human-readable description.

### EditorLiveSave

Public live-save surface the editor boot drives.

### EditorLiveSaveConfig

Factory configuration for the live save.

### EditorLiveSaveDocumentConfig

Factory configuration for the live-save document projection.

### EditorLiveSaveFetch

```ts
EditorLiveSaveFetch(
  url: string,
  init: EditorLiveSaveRequestInit | undefined,
): Promise<EditorLiveSaveFetchResponse>
```

Fetch seam the live save drives.

Unlike the tool frame's GET-only seam, this one issues the write
methods the model store's save route requires.

### EditorLiveSaveFetchResponse

Minimal response shape the live save reads from its fetch seam. The
browser `Response` satisfies it structurally; the test doubles only
implement the members the live save actually consumes.

### EditorLiveSaveRebuild

Outcome kind of the rebuild flight that follows a stored document.

### EditorLiveSaveRequestInit

Request shape the live save issues through its fetch seam.

### EditorLiveSaveResult

Result of one live-save attempt.

### EditorLiveSaveSnapshot

Read-only view of the live save's state.

### EditorLiveSaveSource

The working volume the live save serializes.

### EditorLiveSaveStatus

Outcome kind of one live-save attempt.

### EditorLiveSymmetry

Public live symmetry surface the editor paint path drives.

The `mirrorX` flag and the expansion follow the toggle at call time, so
flipping the toggle mid-session takes effect on the NEXT paint without
rebuilding the policy.

### EditorLiveSymmetryConfig

Factory configuration for the live mirror-X symmetry policy.

### EditorMirrorXToggle

Minimal checkbox surface the live mirror-X flag reads.

The DOM checkbox (`HTMLInputElement`) satisfies it structurally; the
contract tests drive a plain object instead.

### encodeRow

```ts
encodeRow(
  volume: Uint8Array<ArrayBufferLike>,
  z: number,
  y: number,
  storedWidth: number,
  paletteSize: number,
): string
```

Encodes the stored part of one volume row into a section 7.2 RLE hex
row string.

The stored part is the FULL row for a full-width document and the LEFT
half (cols 0..63) for a mirror-X symmetric document - the half the
loader mirrors back onto the full width (LD-28).

Parameters:
- `volume` - Dense working volume.
- `z` - Voxel slice the row belongs to.
- `y` - Voxel row the row belongs to.
- `storedWidth` - Number of columns the row must cover.
- `paletteSize` - Number of entries the document palette holds.

Returns: The RLE hex row string.

### encodeRun

```ts
encodeRun(
  runLength: number,
  paletteIndex: number,
): string
```

Encodes one RLE run into its hex pair.

Parameters:
- `runLength` - Run length in cells (1..255).
- `paletteIndex` - Palette index the run paints.

Returns: The `<runLen:2hex><paletteIdx:2hex>` hex pair.

### isEditorVolumeMirrorSymmetric

```ts
isEditorVolumeMirrorSymmetric(
  volume: Uint8Array<ArrayBufferLike>,
): boolean
```

Reports whether the working volume is mirror-X symmetric.

The volume is symmetric when every row mirrors onto itself across the
x axis (`volume[x] === volume[127 - x]` for cols 0..63). The live save
folds only a verified symmetric volume into the half-width storage
form, because the loader mirrors the stored half back onto the full
width - folding an asymmetric volume would silently drop the right
half.

Parameters:
- `volume` - Dense working volume.

Returns: True when every row mirrors onto itself.

Example:

```ts
if (isEditorVolumeMirrorSymmetric(state.volume)) {
  // the half-width storage form is lossless for this volume
}
```

### NEATRACING_EDITOR_LIVE_SAVE_LABEL

Pass label prefixing every observable live-save failure surface.

### NEATRACING_EDITOR_MAX_SAVE_BODY_BYTES

Save body budget the editor enforces before it issues a save request.

Mirrors the dev server's `NEATRACING_MAX_SAVE_BODY_BYTES` (8 MiB): the
server rejects an over-budget body with `413 dev-server/body-too-large`,
and the editor refuses the same payload locally so no doomed request
leaves the browser. The server constant cannot be imported here (the
editor bundle is a browser build and the server module pulls in
`node:http`), so the contract test pins the two values together.

### readResponseCode

```ts
readResponseCode(
  response: EditorLiveSaveFetchResponse,
): Promise<string | null>
```

Reads the dev-server envelope code of one response.

Parameters:
- `response` - Response the seam returned.

Returns: The envelope code, or null when the body carries none.

### resolvePaletteIndex

```ts
resolvePaletteIndex(
  value: number,
  paletteSize: number,
): number
```

Clamps a volume palette index into the document palette's range.

Parameters:
- `value` - Raw palette index stored in the volume.
- `paletteSize` - Number of entries the document palette holds.

Returns: The index to write, or 0 (EMPTY) when it is out of range.

## editor/editor.toolframe.ts

### appendToolframeSection

```ts
appendToolframeSection(
  documentRef: Document,
  root: HTMLElement,
  group: string,
  title: string,
): HTMLElement
```

Creates one titled control-group section inside the tool frame.

Parameters:
- `documentRef` - Document the section is created in.
- `root` - Tool-frame root the section is appended to.
- `group` - Control-group name recorded on the section.
- `title` - Human-readable section title.

Returns: The created section element.

### applyDocumentToVolume

```ts
applyDocumentToVolume(
  state: EditorState,
  palette: EditorPalette,
  document: VoxelModelDocument,
): void
```

Decodes a model document into the editor working volume.

The document palette is mapped onto the editor palette first (the EMPTY
sentinel keeps index 0), then every decoded cell is written at the
volume's `x + row * y + slice * z` address. The volume is cleared first
so a reload never leaves stale voxels behind.

Parameters:
- `state` - Editor state container owning the working volume.
- `palette` - Editor palette the document palette is mapped onto.
- `document` - Validated model document to decode.

### createEditorToolframe

```ts
createEditorToolframe(
  config: EditorToolframeConfig | undefined,
): EditorToolframeSession
```

Creates the editor tool frame and mounts it into the reserved container.

The frame owns the editor state container (working volume plus palette),
the bounded undo stack, and the mirror-symmetry surface; it mounts the
palette list, the undo/redo buttons, the model dropdown, the CAD-view
buttons, and the symmetry toggle into the reserved container; and it
starts the initial model load (model-store list, read, and decode into
the working volume).

Parameters:
- `config` - Optional configuration (container id, document, fetch
seam).

Returns: The tool-frame session.

Example:

```ts
const toolframe = createEditorToolframe();
await toolframe.ready;
console.log(toolframe.state.snapshot().model);
```

### createToolframeButton

```ts
createToolframeButton(
  documentRef: Document,
  buttonId: string,
  label: string,
): HTMLButtonElement
```

Creates one labeled button inside a control-group section.

Parameters:
- `documentRef` - Document the button is created in.
- `buttonId` - Element id of the button.
- `label` - Human-readable button label.

Returns: The created button element.

### defaultToolframeFetch

```ts
defaultToolframeFetch(
  url: string,
): Promise<EditorToolframeFetchResponse>
```

Browser fetch seam: resolves a URL through the global `fetch`.

Parameters:
- `url` - Absolute or root-relative URL to request.

Returns: The JSON response.

### EditorToolframeConfig

Factory configuration for the editor tool frame.

### EditorToolframeControls

The controls the tool frame mounts into the reserved container.

### EditorToolframeFetch

```ts
EditorToolframeFetch(
  url: string,
): Promise<EditorToolframeFetchResponse>
```

Fetch seam the tool frame drives: it resolves a URL to a JSON response.

### EditorToolframeFetchResponse

Minimal response shape the tool frame reads from its fetch seam. The
browser `Response` satisfies it structurally; the test doubles only
implement the two members the tool frame actually consumes.

### EditorToolframeLoadResult

Outcome of a model load into the working volume.

### EditorToolframeSession

Public tool-frame surface the editor boot and the later slices drive.

### isRecord

```ts
isRecord(
  value: unknown,
): boolean
```

Reports whether a value is a plain object record.

Parameters:
- `value` - Value to inspect.

Returns: True when the value is a non-null object.

### listModelNames

```ts
listModelNames(
  fetchImpl: EditorToolframeFetch,
): Promise<readonly string[]>
```

Lists the model names the dev-server model store holds.

Parameters:
- `fetchImpl` - Fetch seam to drive.

Returns: The listed model names (empty when the store lists none).

### mountToolframeControls

```ts
mountToolframeControls(
  documentRef: Document,
  root: HTMLElement,
): EditorToolframeControls
```

Mounts the tool-frame controls into the tool-frame root element.

Parameters:
- `documentRef` - Document the controls are created in.
- `root` - Tool-frame root the control groups are appended to.

Returns: The mounted control elements.

### NEATRACING_EDITOR_MODEL_LIST_PATH

Dev-server model-store list route the tool frame reads model names from.

### NEATRACING_EDITOR_TOOLFRAME_CONTROL_IDS

Element ids of every control the tool frame mounts into the reserved
container. The ids are the tool frame's public DOM contract: the later
editing-loop slices bind their handlers to these elements.

### NEATRACING_EDITOR_TOOLS_ID

Element id of the reserved editor tool-frame container (the editor
shell's `neatracing-editor-tools` aside).

### readModelDocument

```ts
readModelDocument(
  fetchImpl: EditorToolframeFetch,
  modelName: string,
): Promise<VoxelModelDocument>
```

Reads one model document from the dev-server model store.

Parameters:
- `fetchImpl` - Fetch seam to drive.
- `modelName` - Name of the model to read.

Returns: The validated model document.

### renderModelOptions

```ts
renderModelOptions(
  modelSelect: HTMLSelectElement,
  modelNames: readonly string[],
): void
```

Renders the model-store names as dropdown options.

Parameters:
- `modelSelect` - Select element the options are rendered into.
- `modelNames` - Model names to render.

### renderPaletteList

```ts
renderPaletteList(
  paletteList: HTMLUListElement,
  palette: EditorPalette,
): void
```

Renders one swatch per used palette entry into the palette list.

Parameters:
- `paletteList` - List element the swatches are rendered into.
- `palette` - Palette whose snapshot is rendered.

### resolveToolframeFetch

```ts
resolveToolframeFetch(
  config: EditorToolframeConfig | undefined,
): EditorToolframeFetch | null
```

Resolves the fetch seam the tool frame drives.

Parameters:
- `config` - Optional tool-frame configuration.

Returns: The injected seam, the browser `fetch`, or null when no fetch
is available (a missing dev server, not a failure).

## editor/editor-preview.render.ts

### EditorPreviewBinsDraw

Draw-only seam the loop drives once per frame; the seam owns disposal.

### EditorPreviewBinsDrawState

Per-frame draw state handed to the bins draw seam.

### EditorPreviewCancelFrame

```ts
EditorPreviewCancelFrame(
  frameId: number,
): void
```

Frame cancellation seam mirroring `window.cancelAnimationFrame`.

### EditorPreviewCanvasView

Structural canvas view the loop consumes; only width and height matter.

### EditorPreviewRenderLoopConfig

Configuration for {@link startEditorPreviewRenderLoop}.

### EditorPreviewRenderLoopHandle

Handle returned by the loop factory; `stop` is idempotent.

### EditorPreviewRequestFrame

```ts
EditorPreviewRequestFrame(
  callback: (timeMs: number) => void,
): number
```

Frame scheduling seam mirroring `window.requestAnimationFrame`.

### multiplyViewProjection

```ts
multiplyViewProjection(
  out: Float32Array<ArrayBufferLike>,
  view: Float32Array<ArrayBufferLike>,
  projection: Float32Array<ArrayBufferLike>,
): void
```

Multiply a column-major projection onto a view and write the (projection *
view) product in place: `out[column * 4 + row] = sum_k projection
[k * 4 + row] * view[column * 4 + k]`.

The (projection * view) order is the seam contract because the consuming
car-pass GLSL applies the stored matrix as `gl_Position = uViewProjection *
world` (column-major, transpose = false): the view transform must apply to
world eye-space positions before the projection. A (view * projection)
product would corrupt the perspective row's `w_clip` and degenerate the
car draw.

Allocation-free: the 4x4 output buffer is owned by the caller and reused
every frame (frame-composer bag discipline, AC-042).

Parameters:
- `out` - Pre-allocated 16-float output buffer for the product.
- `view` - Column-major 4x4 view matrix.
- `projection` - Column-major 4x4 projection matrix.

Example:

```ts
const product = new Float32Array(16);
multiplyViewProjection(product, view, projection);
```

### orbitCameraWorld

```ts
orbitCameraWorld(
  pose: EditorCameraPose,
  out: [number, number, number],
): void
```

Compute the orbit pose eye point into the pre-owned camera-world triple.

Uses the documented plain-trig eye derivation (mirroring the reference
camera-eye formula in editor.camera JSDoc): `center + distance *
[cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw)]`. Trig-snapping
is deliberately NOT applied here — the camera applies its own snapping
inside `buildViewMatrix`, while the draw seam's eye must be the raw pose
value.

Parameters:
- `pose` - The camera's current (smoothed) orbit pose.
- `out` - Pre-owned triple refreshed in place.

Example:

```ts
const cameraWorld: [number, number, number] = [0, 0, 0];
orbitCameraWorld(camera.snapshot().current, cameraWorld);
```

### PreviewBinsMemoEntry

Memoized document -> bins resolution keyed by document reference.

### resolvePreviewBins

```ts
resolvePreviewBins(
  memo: PreviewBinsMemoEntry,
  document: VoxelModelDocument,
): VoxelBinModel | null
```

Resolve the selected document into greedy-meshed bins, memoized per
document identity.

The bridged bins pair the mesher's `VoxelQuad[]` (a structural subset of
`VoxelBinQuad`) with the document's validated palette. Decode failures are
cached as "no bins for this document" so an invalid selection is remembered
instead of re-thrown once per frame, and surfaced through the labeled
console surface exactly once per failing document identity.

Parameters:
- `memo` - Per-loop memo entry; updated in place.
- `document` - Currently selected document (never null; null selections
are handled by the caller and never reach this resolver).

Returns: The bridged bins for the document, or null when the document
failed to decode.

Example:

```ts
const memo = { document: null, bins: null };
const bins = document === null ? null : resolvePreviewBins(memo, document);
if (bins !== null) drawBins(bins);
```

### startEditorPreviewRenderLoop

```ts
startEditorPreviewRenderLoop(
  config: EditorPreviewRenderLoopConfig,
): EditorPreviewRenderLoopHandle
```

Start the editor model preview render loop (AC-019 preview lane).

Each scheduled frame eases the camera transform, resolves the selected
document through the decode -> greedy-mesher chain (memoized per document
identity), and — when bins exist — builds the column-major view-projection
product plus the orbit eye point and hands them to the bins draw seam.
With no document selected (or a document that fails to decode) the RAF
chain keeps ticking and the loop draws nothing. `stop` cancels the pending
frame and disposes the bins draw exactly once; repeated stops are no-ops.

Parameters:
- `config` - Loop configuration: structural canvas view, editor camera,
selected-document accessor, bins draw seam, and optional RAF seam
defaults (the window `requestAnimationFrame`/`cancelAnimationFrame`
pair).

Returns: A handle whose single `stop` cancels the pending frame and
disposes the bins draw; safe to call more than once.

Example:

```ts
const loop = startEditorPreviewRenderLoop({
  canvas,
  camera,
  getSelectedModelDocument,
  binsDraw,
});
loop.stop();
```

## editor/editor.rebuild.pipeline.ts

Save→rebuild pipeline for the NeatRacing editor (slice P1-S6-E; spec
section 8.3 policy P5, AC-041): every {@linkcode EditorRebuildPipeline.edit}
emits an immediate dirty-slice remesh covering the edited z-slice's clamped,
deduplicated AABB ± 1 window, then arms a
{@linkcode NEATRACING_REBUILD_DEBOUNCE_MS} trailing debounce that coalesces
the drag burst into exactly one full rebuild, and
{@linkcode EditorRebuildPipeline.pointerUp} flushes that pending debounce
into one immediate full rebuild.

The policy is latest-wins single-flight (spec P5 bullet 2): only one rebuild
is ever in flight, work requested mid-flight is queued behind the in-flight
rebuild (its superseded result is discarded; the
{@linkcode EditorRebuildPipeline.snapshot} `supersededCount` counter is
arrival-time), the queue drains strictly in FIFO order, and full-rebuild
durations measured through the injected `nowMs` clock yield the nearest-rank
`p95Ms` statistic the contract tests hold against the
{@linkcode NEATRACING_REBUILD_P95_BUDGET_MS} budget.

The module is pure scheduling/state — no DOM, no WebGL, and no RNG — so the
pipeline is exactly deterministic: the same edit/pointer-up sequence and
injected clock in, the same rebuild call order and snapshot out on every
run and runtime.

### buildDirtyRequest

```ts
buildDirtyRequest(
  sliceZ: number,
): EditorRebuildRequest
```

Builds the dirty-slice remesh request for one edited slice: the clamped
AABB ± 1 window `[z - 1, z, z + 1]`, deduplicated where the clamped edges
pin against the volume boundary (0 → `[0, 1]`, 127 → `[126, 127]`).

Parameters:
- `sliceZ` - Edited slice coordinate, clamped before windowing.

Returns: The dirty request carrying the ascending deduplicated window.

### clampSliceZ

```ts
clampSliceZ(
  sliceZ: number,
): number
```

Clamps one slice coordinate into the editor volume window 0..127
({@linkcode NEATRACING_EDITOR_VOLUME_EDGE} - 1 inclusive on both ends).

Parameters:
- `sliceZ` - Unclamped slice coordinate input.

Returns: The slice coordinate pinned into the editor volume window.

### createEditorRebuildPipeline

```ts
createEditorRebuildPipeline(
  options: EditorRebuildPipelineOptions,
): EditorRebuildPipeline
```

Pipeline factory: binds one rebuild executor and an optional monotonic
clock into a mutable edit→rebuild scheduler that serializes through a
single FIFO queue, so callers never observe two rebuilds in flight.

Parameters:
- `options` - Executor plus optional duration clock (see
 *  {@linkcode EditorRebuildPipelineOptions} ).

Returns: The `{@linkcode EditorRebuildPipeline.edit}`,
`{@linkcode EditorRebuildPipeline.pointerUp}`,
`{@linkcode EditorRebuildPipeline.snapshot}` scheduler surface.

Example:

```ts
const pipeline = createEditorRebuildPipeline({
  rebuild: async () => remeshDirtySlices(),
});
pipeline.edit(42); // immediate [41, 42, 43] remesh + 250 ms debounce
pipeline.pointerUp(); // pending debounce consumed → immediate full
const stats = pipeline.snapshot(); // in-flight/queued/p95 statistics
```

### defaultNowMs

```ts
defaultNowMs(): number
```

Ambient clock fallback for {@linkcode EditorRebuildPipelineOptions.nowMs}:
Node's high-resolution relative milliseconds stay monotonic across the
duration-measured requests, so recorded durations are order-independent.

Returns: Elapsed high-resolution milliseconds relative to process start.

### EditorRebuildClock

```ts
EditorRebuildClock(): number
```

Monotonic clock accessor used to measure full-rebuild durations.

### EditorRebuildExecutor

```ts
EditorRebuildExecutor(
  request: EditorRebuildRequest,
): Promise<void>
```

Rebuild executor injected into the pipeline by the editor integration.

### EditorRebuildPipeline

Scheduler surface returned by {@linkcode createEditorRebuildPipeline}.

### EditorRebuildPipelineOptions

Factory configuration for the editor rebuild pipeline.

### EditorRebuildRequest

One rebuild workload emitted or queued by the pipeline.

### EditorRebuildSnapshot

Read-only scheduling and timing statistics for one pipeline moment.

### NEATRACING_REBUILD_DEBOUNCE_MS

Trailing debounce window before a drag burst coalesces into one full rebuild (ms).

### NEATRACING_REBUILD_P95_BUDGET_MS

p95 budget ceiling for measured full-rebuild durations (ms).

### percentileNearestRank

```ts
percentileNearestRank(
  values: readonly number[],
): number
```

Nearest-rank percentile of one duration list: the smallest value whose
cumulative rank reaches `quantile * length`. An empty list reports 0 so a
parked-first-rebuild snapshot still yields a well-formed 0 ms statistic
before any rebuild completes.

Parameters:
- `values` - Completed durations in any order.

Returns: The nearest-rank quantile in the same millisecond units.
