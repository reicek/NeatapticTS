/// <reference lib="dom" />

/**
 * Pointer-pick edit loop for the Neat Racing voxel model editor (slice
 * P1-editor-editloop; spec section 8.3, AC-021 partial, LD-28 of
 * `plans/neatracing-voxel-racer.spec.md`): the canvas pointer chain that
 * turns a click into a painted voxel.
 *
 * The module owns three seams:
 *
 * 1. {@linkcode createEditorVoxelPicker} - the camera-ray picker. It casts
 *    the pointer position through the editor camera (perspective or CAD
 *    orthographic), maps the ray into the editor volume's grid space, and
 *    marches the grid cells (Amanatides-Woo). The pick returns the FIRST
 *    OCCUPIED voxel along the ray; when the ray crosses no occupied voxel
 *    it falls back to the FIRST CELL THE RAY ENTERED; when the ray never
 *    enters the volume it reports a miss (`null`).
 * 2. {@linkcode createEditorEditLoop} - the pointer chain. A pointerdown
 *    picks the voxel under the pointer, captures the stroke's undo
 *    snapshot, paints every symmetry-canonicalized target, and routes the
 *    edit through the rebuild pipeline's `edit(sliceZ)` dirty window. A
 *    pointermove continues the stroke; a pointerup flushes the pipeline
 *    with one full rebuild - but ONLY when the stroke actually painted
 *    (a stroke that hit nothing never flushes, so no spurious full
 *    rebuild fires).
 * 3. {@linkcode buildEditorVolumeDocument} - the working-volume
 *    projection into a section 7.2 model document (RLE hex rows, the EMPTY
 *    sentinel at palette index 0, the full 128-cubed dims). The editor
 *    stores the FULL width, so the projected document is declared
 *    `symmetric: false`; the live-save slice owns the symmetric on-disk
 *    storage.
 *
 * ORDERING DEVIATION (documented on purpose): the undo snapshot is
 * captured BEFORE the paint of the stroke it covers, not after it. The
 * editor undo stack restores the NEWEST retained snapshot
 * (`EditorUndo.undo` pops newest-first), so a post-paint push would make
 * the stroke's undo a no-op - the stroke would be unpaintable. The plan's
 * chain prose lists the push after the paint; this module implements the
 * ordering that keeps the undo meaningful, and the slice's PlanUpdate
 * records the deviation.
 *
 * The module stays free of WebGL and of any ambient timer: the picker is
 * pure math over the injected camera, the edit loop only talks to the
 * injected canvas/state/symmetry/undo/pipeline seams, and the projection
 * is pure data over the injected volume and palette.
 */

import type { EditorCamera } from './editor.camera';
import type { EditorPalette } from './editor.palette';
import type { EditorRebuildPipeline } from './editor.rebuild.pipeline';
import type { EditorState } from './editor.state';
import {
  NEATRACING_EDITOR_VOLUME_EDGE,
  NEATRACING_EDITOR_VOLUME_ROW_STRIDE,
  NEATRACING_EDITOR_VOLUME_SLICE_STRIDE,
} from './editor.state';
import type { EditorSymmetry } from './editor.symmetry';
import type { EditorUndo } from './editor.undo';
import { VOXEL_EMPTY_PALETTE_ENTRY } from '../voxel/model.schema';
import type {
  VoxelModelDocument,
  VoxelPaletteEntry,
} from '../voxel/model.schema';
import { NEATRACING_VOXEL_WORLD_SIZE } from '../voxel/lod.distance.utils';

/**
 * Default paint entry the edit loop writes when the caller names no color:
 * an opaque white body color with no glow.
 */
export const NEATRACING_EDITOR_DEFAULT_PAINT_ENTRY: VoxelPaletteEntry = [
  255, 255, 255, 255, 0,
];

/**
 * Default model name the working-volume projection stamps into the
 * document metadata when the caller names none (the section 7.2 metadata
 * header requires a non-empty name).
 */
export const NEATRACING_EDITOR_DEFAULT_MODEL_NAME = 'car-f1';

/** Smallest viewport edge (CSS pixels) the picker divides by. */
const PICK_MIN_VIEWPORT_PX = 1;

/** Direction magnitude below which a ray axis counts as parallel. */
const PICK_DIRECTION_EPSILON = 1e-9;

/** Ray-parameter nudge (world units) that moves the entry point inside. */
const PICK_ENTRY_NUDGE = 1e-6;

/** Largest number of grid cells the pick ray may cross. */
const PICK_MAX_MARCH_STEPS = 512;

/** Relative tolerance that treats two axis crossings as one corner. */
const PICK_CORNER_TIE_EPSILON = 1e-9;

/** Grid-space offset that centers the lattice on the world origin. */
const GRID_CENTER_OFFSET = NEATRACING_EDITOR_VOLUME_EDGE * 0.5;

/** Highest inclusive voxel index on every editor volume axis. */
const MAX_VOLUME_CELL_INDEX = NEATRACING_EDITOR_VOLUME_EDGE - 1;

/** Pointer position inside the canvas box, in CSS pixels. */
export interface EditorPointerPosition {
  /** Horizontal position inside the canvas box (0 = left edge). */
  readonly x: number;
  /** Vertical position inside the canvas box (0 = top edge). */
  readonly y: number;
}

/** One voxel of the editor working volume. */
export interface EditorVoxelHit {
  /** Voxel column, inclusive in 0..127. */
  readonly x: number;
  /** Voxel row, inclusive in 0..127. */
  readonly y: number;
  /** Voxel slice, inclusive in 0..127. */
  readonly z: number;
}

/**
 * Viewport metrics the picker needs to build the camera ray.
 *
 * @property cssWidth - Canvas CSS width in pixels (the pointer coordinate
 *   space).
 * @property cssHeight - Canvas CSS height in pixels.
 * @property aspect - Drawing-buffer width-over-height aspect ratio (the
 *   projection aspect the preview renders with).
 */
export interface EditorPickerViewport {
  readonly cssWidth: number;
  readonly cssHeight: number;
  readonly aspect: number;
}

/** Picker seam the edit loop resolves pointer positions through. */
export interface EditorVoxelPicker {
  /**
   * Picks the voxel under a canvas-local pointer position.
   *
   * @param position - Pointer position inside the canvas box, in CSS pixels.
   * @returns The picked voxel, or null when the ray misses the volume.
   */
  pick(position: EditorPointerPosition): EditorVoxelHit | null;
}

/** Public edit-loop surface the editor boot drives. */
export interface EditorEditLoop {
  /**
   * Attaches the pointer listeners to the canvas (idempotent).
   *
   * @returns Nothing.
   */
  attach(): void;

  /**
   * Detaches the pointer listeners and ends any in-flight stroke (the
   * detached loop never paints again).
   *
   * @returns Nothing.
   */
  detach(): void;

  /**
   * Reports whether a pointer stroke is currently in progress.
   *
   * @returns true while a stroke is open (pointerdown without pointerup).
   */
  isPainting(): boolean;
}

/** Factory configuration for the editor voxel picker. */
export interface EditorVoxelPickerConfig {
  /** Dense working volume the picker reads occupancy from. */
  readonly volume: Uint8Array;
  /** Editor camera the picker casts the ray through. */
  readonly camera: EditorCamera;
  /** Viewport metrics accessor the picker reads per pick. */
  readonly readViewport: () => EditorPickerViewport;
}

/** Factory configuration for the editor edit loop. */
export interface EditorEditLoopConfig {
  /** Canvas the pointer listeners attach to. */
  readonly canvas: HTMLCanvasElement;
  /** Editor state container the loop paints into. */
  readonly state: EditorState;
  /** Mirror-symmetry surface expanding one paint into its targets. */
  readonly symmetry: EditorSymmetry;
  /** Undo stack the loop captures the stroke snapshot on. */
  readonly undo: EditorUndo;
  /** Rebuild pipeline the loop routes the edits through. */
  readonly pipeline: EditorRebuildPipeline;
  /** Picker resolving the pointer position into a voxel. */
  readonly picker: EditorVoxelPicker;
  /** Paint entry; defaults to {@linkcode NEATRACING_EDITOR_DEFAULT_PAINT_ENTRY}. */
  readonly paintEntry?: VoxelPaletteEntry;
}

/** Factory configuration for the working-volume document projection. */
export interface EditorVolumeDocumentConfig {
  /** Dense working volume the projection reads. */
  readonly volume: Uint8Array;
  /** Editor palette the document palette is built from. */
  readonly palette: EditorPalette;
  /** Model name stamped into the document metadata; defaults to the editor default. */
  readonly name?: string;
}

/** One world-space camera ray. */
interface EditorCameraRay {
  /** Ray origin in world space. */
  readonly origin: readonly [number, number, number];
  /** Ray direction in world space (not normalized). */
  readonly direction: readonly [number, number, number];
}

/** The parameter span over which a ray stays inside the volume's box. */
interface EditorGridSpan {
  /** Ray parameter where the ray enters the volume. */
  readonly tEnter: number;
  /** Ray parameter where the ray leaves the volume. */
  readonly tExit: number;
}

/**
 * Clamps a cell index into the volume's inclusive 0..127 range.
 *
 * @param value - Raw cell index.
 * @returns The clamped cell index.
 */
function clampCellIndex(value: number): number {
  return Math.min(Math.max(value, 0), MAX_VOLUME_CELL_INDEX);
}

/**
 * Builds the world-space camera ray under a canvas-local pointer position.
 *
 * The ray follows the editor camera's conventions: the column-major view
 * matrix supplies the right/up/back basis and the eye point, the
 * projection matrix supplies the perspective near plane (or the ortho
 * half-extents), and the canvas-local position maps into normalized device
 * coordinates through the CSS box.
 *
 * @param camera - Editor camera the ray is cast through.
 * @param viewport - Viewport metrics of the canvas.
 * @param position - Pointer position inside the canvas box.
 * @returns The world-space ray.
 */
function buildCameraRay(
  camera: EditorCamera,
  viewport: EditorPickerViewport,
  position: EditorPointerPosition,
): EditorCameraRay {
  // Step 1: Read the camera basis and the eye point from the view matrix.
  const view = camera.buildViewMatrix();
  const right: readonly [number, number, number] = [
    view[0],
    view[4],
    view[8],
  ];
  const up: readonly [number, number, number] = [view[1], view[5], view[9]];
  const back: readonly [number, number, number] = [
    view[2],
    view[6],
    view[10],
  ];
  const eye: [number, number, number] = [
    -(view[12] * right[0] + view[13] * up[0] + view[14] * back[0]),
    -(view[12] * right[1] + view[13] * up[1] + view[14] * back[1]),
    -(view[12] * right[2] + view[13] * up[2] + view[14] * back[2]),
  ];

  // Step 2: Map the canvas-local pointer position into NDC (Y up).
  const cssWidth = Math.max(viewport.cssWidth, PICK_MIN_VIEWPORT_PX);
  const cssHeight = Math.max(viewport.cssHeight, PICK_MIN_VIEWPORT_PX);
  const ndcX = (position.x / cssWidth) * 2 - 1;
  const ndcY = 1 - (position.y / cssHeight) * 2;

  // Step 3: Build the orthographic or the perspective ray.
  const projection = camera.buildProjectionMatrix(viewport.aspect);
  if (camera.snapshot().mode === 'orthographic') {
    const viewX = ndcX / projection[0];
    const viewY = ndcY / projection[5];
    return {
      origin: [
        eye[0] + right[0] * viewX + up[0] * viewY,
        eye[1] + right[1] * viewX + up[1] * viewY,
        eye[2] + right[2] * viewX + up[2] * viewY,
      ],
      direction: [-back[0], -back[1], -back[2]],
    };
  }
  const nearPlane = projection[14] / (projection[10] - 1);
  const viewX = (ndcX * nearPlane) / projection[0];
  const viewY = (ndcY * nearPlane) / projection[5];
  return {
    origin: eye,
    direction: [
      right[0] * viewX + up[0] * viewY - back[0],
      right[1] * viewX + up[1] * viewY - back[1],
      right[2] * viewX + up[2] * viewY - back[2],
    ],
  };
}

/**
 * Maps a world-space ray into the editor volume's grid space.
 *
 * The grid space mirrors the engine's voxel expansion (the car pass maps
 * grid `x`/`z` to `(grid - dim * 0.5) * voxelSize` and grid `y` to
 * `grid * voxelSize`), so the inverse is `gridX = worldX / voxelSize + 64`,
 * `gridY = worldY / voxelSize`, and `gridZ = worldZ / voxelSize + 64`.
 *
 * @param ray - World-space ray.
 * @returns The same ray expressed in grid cells.
 */
function mapRayToGridSpace(ray: EditorCameraRay): EditorCameraRay {
  const inverseVoxelSize = 1 / NEATRACING_VOXEL_WORLD_SIZE;
  return {
    origin: [
      ray.origin[0] * inverseVoxelSize + GRID_CENTER_OFFSET,
      ray.origin[1] * inverseVoxelSize,
      ray.origin[2] * inverseVoxelSize + GRID_CENTER_OFFSET,
    ],
    direction: [
      ray.direction[0] * inverseVoxelSize,
      ray.direction[1] * inverseVoxelSize,
      ray.direction[2] * inverseVoxelSize,
    ],
  };
}

/**
 * Intersects a grid-space ray with the volume's bounding box.
 *
 * @param ray - Grid-space ray.
 * @returns The parameter span inside the box, or null when the ray misses.
 */
function intersectGridBox(ray: EditorCameraRay): EditorGridSpan | null {
  let tEnter = Number.NEGATIVE_INFINITY;
  let tExit = Number.POSITIVE_INFINITY;
  for (let axis = 0; axis < 3; axis += 1) {
    const direction = ray.direction[axis];
    const origin = ray.origin[axis];
    if (Math.abs(direction) < PICK_DIRECTION_EPSILON) {
      // A parallel axis only misses when the ray already left its slab.
      if (origin < 0 || origin > NEATRACING_EDITOR_VOLUME_EDGE) {
        return null;
      }
      continue;
    }
    const near = (0 - origin) / direction;
    const far = (NEATRACING_EDITOR_VOLUME_EDGE - origin) / direction;
    tEnter = Math.max(tEnter, Math.min(near, far));
    tExit = Math.min(tExit, Math.max(near, far));
  }
  if (tEnter > tExit || tExit < 0) {
    return null;
  }
  return { tEnter, tExit };
}

/**
 * Resolves the first cell the ray enters.
 *
 * The entry point sits exactly on the volume boundary, so it is nudged a
 * hair along the ray before the cell index is taken - otherwise a ray
 * entering at the far boundary would resolve the out-of-range cell 128
 * instead of the cell 127 it actually enters.
 *
 * @param ray - Grid-space ray.
 * @param tEnter - Ray parameter where the ray enters the volume.
 * @returns The entered cell.
 */
function resolveEntryCell(
  ray: EditorCameraRay,
  tEnter: number,
): EditorVoxelHit {
  const t = tEnter + PICK_ENTRY_NUDGE;
  return {
    x: clampCellIndex(Math.floor(ray.origin[0] + ray.direction[0] * t)),
    y: clampCellIndex(Math.floor(ray.origin[1] + ray.direction[1] * t)),
    z: clampCellIndex(Math.floor(ray.origin[2] + ray.direction[2] * t)),
  };
}

/**
 * Computes the ray parameter at which the ray leaves the current cell on
 * one axis.
 *
 * @param ray - Grid-space ray.
 * @param axis - Axis index (0 = x, 1 = y, 2 = z).
 * @param cell - Current cell index on that axis.
 * @returns The boundary parameter, or infinity for a parallel axis.
 */
function nextBoundaryT(
  ray: EditorCameraRay,
  axis: number,
  cell: number,
): number {
  const direction = ray.direction[axis];
  if (Math.abs(direction) < PICK_DIRECTION_EPSILON) {
    return Number.POSITIVE_INFINITY;
  }
  const boundary = direction > 0 ? cell + 1 : cell;
  return (boundary - ray.origin[axis]) / direction;
}

/**
 * Reports whether one volume cell holds a painted voxel.
 *
 * @param volume - Dense working volume.
 * @param x - Voxel column.
 * @param y - Voxel row.
 * @param z - Voxel slice.
 * @returns true when the cell holds a non-EMPTY palette index.
 */
function isOccupied(
  volume: Uint8Array,
  x: number,
  y: number,
  z: number,
): boolean {
  return (
    volume[
      x + NEATRACING_EDITOR_VOLUME_ROW_STRIDE * y + NEATRACING_EDITOR_VOLUME_SLICE_STRIDE * z
    ] !== 0
  );
}

/**
 * Marches the grid cells along the ray and returns the first occupied
 * voxel.
 *
 * The march is the classic Amanatides-Woo traversal: every step crosses
 * exactly one cell boundary, and the axes crossing at the same parameter
 * (a corner) step together so the ray never visits an off-ray cell.
 *
 * @param ray - Grid-space ray.
 * @param span - The ray's parameter span inside the volume.
 * @param entryCell - The cell the ray entered the volume through.
 * @param volume - Dense working volume.
 * @returns The first occupied voxel, or null when the ray crosses none.
 */
function marchForOccupiedVoxel(
  ray: EditorCameraRay,
  span: EditorGridSpan,
  entryCell: EditorVoxelHit,
  volume: Uint8Array,
): EditorVoxelHit | null {
  let cellX = entryCell.x;
  let cellY = entryCell.y;
  let cellZ = entryCell.z;

  for (let step = 0; step < PICK_MAX_MARCH_STEPS; step += 1) {
    // Step 1: Report the first occupied cell along the ray.
    if (isOccupied(volume, cellX, cellY, cellZ)) {
      return { x: cellX, y: cellY, z: cellZ };
    }

    // Step 2: Find the next cell boundary the ray crosses.
    const nextX = nextBoundaryT(ray, 0, cellX);
    const nextY = nextBoundaryT(ray, 1, cellY);
    const nextZ = nextBoundaryT(ray, 2, cellZ);
    const nextT = Math.min(nextX, nextY, nextZ);
    if (!Number.isFinite(nextT) || nextT > span.tExit) {
      break;
    }

    // Step 3: Step every axis crossing at that boundary (a corner steps
    // all of its axes together).
    const tie = Math.max(1, Math.abs(nextT)) * PICK_CORNER_TIE_EPSILON;
    if (nextX <= nextT + tie) {
      cellX += ray.direction[0] > 0 ? 1 : -1;
    }
    if (nextY <= nextT + tie) {
      cellY += ray.direction[1] > 0 ? 1 : -1;
    }
    if (nextZ <= nextT + tie) {
      cellZ += ray.direction[2] > 0 ? 1 : -1;
    }

    // Step 4: Stop once the march left the volume.
    if (
      cellX < 0 ||
      cellX > MAX_VOLUME_CELL_INDEX ||
      cellY < 0 ||
      cellY > MAX_VOLUME_CELL_INDEX ||
      cellZ < 0 ||
      cellZ > MAX_VOLUME_CELL_INDEX
    ) {
      break;
    }
  }
  return null;
}

/**
 * Creates the editor voxel picker over the working volume.
 *
 * The picker is pure math over the injected camera and volume: it casts
 * the camera ray under the pointer, maps it into the volume's grid space,
 * and returns the first occupied voxel along it. A ray that crosses no
 * occupied voxel falls back to the first cell the ray entered; a ray that
 * never enters the volume reports a miss.
 *
 * @param config - Volume, camera, and viewport accessor.
 * @returns The picker implementing {@linkcode EditorVoxelPicker}.
 * @example
 * ```ts
 * const picker = createEditorVoxelPicker({
 *   volume: state.volume,
 *   camera,
 *   readViewport: () => ({ cssWidth: 800, cssHeight: 600, aspect: 4 / 3 }),
 * });
 * const hit = picker.pick({ x: 400, y: 300 });
 * ```
 */
export function createEditorVoxelPicker(
  config: EditorVoxelPickerConfig,
): EditorVoxelPicker {
  const { volume, camera, readViewport } = config;

  return {
    pick(position: EditorPointerPosition): EditorVoxelHit | null {
      // Step 1: Build the world-space camera ray under the pointer.
      const ray = buildCameraRay(camera, readViewport(), position);

      // Step 2: Map the ray into the volume's grid space.
      const gridRay = mapRayToGridSpace(ray);

      // Step 3: Intersect the ray with the volume's bounding box.
      const span = intersectGridBox(gridRay);
      if (span === null) {
        return null;
      }

      // Step 4: March the cells; fall back to the entered cell.
      const entryCell = resolveEntryCell(gridRay, span.tEnter);
      const occupied = marchForOccupiedVoxel(
        gridRay,
        span,
        entryCell,
        volume,
      );
      return occupied ?? entryCell;
    },
  };
}

/**
 * Creates the pointer-pick edit loop over the editor canvas.
 *
 * The loop owns the stroke semantics: the undo snapshot is captured once
 * per stroke (BEFORE the first paint of that stroke), every paint goes
 * through the symmetry expansion, every edit is routed through the
 * rebuild pipeline's `edit(sliceZ)` dirty window, and the stroke's end
 * flushes the pipeline with one full rebuild - but only when the stroke
 * actually painted something.
 *
 * @param config - Canvas, state, symmetry, undo, pipeline, and picker seams.
 * @returns The edit loop implementing {@linkcode EditorEditLoop}.
 * @example
 * ```ts
 * const loop = createEditorEditLoop({
 *   canvas,
 *   state: toolframe.state,
 *   symmetry: toolframe.symmetry,
 *   undo: toolframe.undo,
 *   pipeline,
 *   picker,
 * });
 * loop.attach();
 * ```
 */
export function createEditorEditLoop(
  config: EditorEditLoopConfig,
): EditorEditLoop {
  const { canvas, state, symmetry, undo, pipeline, picker } = config;
  const paintEntry = config.paintEntry ?? NEATRACING_EDITOR_DEFAULT_PAINT_ENTRY;

  /** Whether the pointer listeners are currently attached. */
  let attached = false;
  /** Whether a pointer stroke is open. */
  let painting = false;
  /** Whether the open stroke already captured its undo snapshot. */
  let snapshotCaptured = false;
  /** Whether the open stroke painted anything. */
  let strokeEdited = false;

  /**
   * Maps a pointer event into a canvas-local position.
   *
   * @param event - Pointer event to read the client point from.
   * @returns The canvas-local pointer position.
   */
  function toCanvasPosition(event: Event): EditorPointerPosition {
    const pointerEvent = event as MouseEvent;
    const rect = canvas.getBoundingClientRect();
    return {
      x: pointerEvent.clientX - rect.left,
      y: pointerEvent.clientY - rect.top,
    };
  }

  /**
   * Paints one picked voxel and routes the edit through the pipeline.
   *
   * @param hit - Picked voxel to paint.
   */
  function paintHit(hit: EditorVoxelHit): void {
    // Step 1: Capture the stroke's undo snapshot BEFORE the paint (the
    // undo stack restores the NEWEST snapshot, so a post-paint push
    // would make the stroke's undo a no-op).
    if (!snapshotCaptured) {
      undo.push();
      snapshotCaptured = true;
    }

    // Step 2: Paint every symmetry-canonicalized target (mirror-X on by
    // default per NEATRACING_EDITOR_DEFAULT_MIRROR_X).
    const targets = symmetry.map(hit.x, hit.y, hit.z);
    for (const [x, y, z] of targets) {
      state.paint(x, y, z, paintEntry);
    }

    // Step 3: Route the edit through the rebuild pipeline's dirty window.
    pipeline.edit(hit.z);
    strokeEdited = true;
  }

  /**
   * Handles the pointerdown that opens a stroke.
   *
   * @param event - Pointer event.
   */
  function handlePointerDown(event: Event): void {
    if (!attached) {
      return;
    }
    // Step 1: Open the stroke and reset its per-stroke bookkeeping.
    painting = true;
    snapshotCaptured = false;
    strokeEdited = false;

    // Step 2: Pick the voxel under the pointer and paint it when hit.
    const hit = picker.pick(toCanvasPosition(event));
    if (hit !== null) {
      paintHit(hit);
    }

    // Step 3: Capture the pointer so the stroke survives leaving the
    // canvas box (guarded: jsdom-style hosts have no capture support).
    try {
      canvas.setPointerCapture?.((event as PointerEvent).pointerId);
    } catch {
      // A host without pointer capture simply ends the stroke on leave.
    }
  }

  /**
   * Handles the pointermove that continues an open stroke.
   *
   * @param event - Pointer event.
   */
  function handlePointerMove(event: Event): void {
    if (!attached || !painting) {
      return;
    }
    const hit = picker.pick(toCanvasPosition(event));
    if (hit !== null) {
      paintHit(hit);
    }
  }

  /**
   * Handles the pointerup/pointercancel/pointerleave that ends a stroke.
   */
  function handlePointerEnd(): void {
    if (!painting) {
      return;
    }
    painting = false;
    // Step 1: Flush the rebuild pipeline only when the stroke painted, so
    // an empty stroke never triggers a spurious full rebuild.
    if (strokeEdited) {
      pipeline.pointerUp();
    }
    strokeEdited = false;
    snapshotCaptured = false;
  }

  return {
    attach(): void {
      if (attached) {
        return;
      }
      attached = true;
      canvas.addEventListener('pointerdown', handlePointerDown);
      canvas.addEventListener('pointermove', handlePointerMove);
      canvas.addEventListener('pointerup', handlePointerEnd);
      canvas.addEventListener('pointercancel', handlePointerEnd);
      canvas.addEventListener('pointerleave', handlePointerEnd);
    },

    detach(): void {
      if (!attached) {
        return;
      }
      attached = false;
      painting = false;
      canvas.removeEventListener('pointerdown', handlePointerDown);
      canvas.removeEventListener('pointermove', handlePointerMove);
      canvas.removeEventListener('pointerup', handlePointerEnd);
      canvas.removeEventListener('pointercancel', handlePointerEnd);
      canvas.removeEventListener('pointerleave', handlePointerEnd);
    },

    isPainting(): boolean {
      return painting;
    },
  };
}

/**
 * Builds the section 7.2 document palette from the editor palette.
 *
 * Palette index 0 is the reserved EMPTY sentinel (air); the editor
 * palette snapshot leaves that slot unfilled once colors exist, so the
 * document palette is rebuilt as `[EMPTY, ...colors]`.
 *
 * @param palette - Editor palette the document palette mirrors.
 * @returns The document palette entries, index-aligned with the volume.
 */
function buildDocumentPalette(palette: EditorPalette): VoxelPaletteEntry[] {
  const entries = palette.snapshot().entries;
  const documentPalette: VoxelPaletteEntry[] = [
    [
      VOXEL_EMPTY_PALETTE_ENTRY[0],
      VOXEL_EMPTY_PALETTE_ENTRY[1],
      VOXEL_EMPTY_PALETTE_ENTRY[2],
      VOXEL_EMPTY_PALETTE_ENTRY[3],
      VOXEL_EMPTY_PALETTE_ENTRY[4],
    ],
  ];
  for (const entry of entries.slice(1)) {
    if (entry === undefined) {
      continue;
    }
    documentPalette.push([entry[0], entry[1], entry[2], entry[3], entry[4]]);
  }
  return documentPalette;
}

/**
 * Encodes one volume row into the section 7.2 RLE hex row string.
 *
 * The row is a sequence of `<runLen:2hex><paletteIdx:2hex>` pairs; the
 * runs of one row sum exactly to the declared row width (128 cols).
 *
 * @param volume - Dense working volume.
 * @param z - Voxel slice the row belongs to.
 * @param y - Voxel row the row belongs to.
 * @param paletteSize - Number of entries the document palette holds.
 * @returns The RLE hex row string.
 */
function encodeRow(
  volume: Uint8Array,
  z: number,
  y: number,
  paletteSize: number,
): string {
  const rowStart =
    NEATRACING_EDITOR_VOLUME_SLICE_STRIDE * z +
    NEATRACING_EDITOR_VOLUME_ROW_STRIDE * y;
  let row = '';
  let runIndex = resolvePaletteIndex(volume[rowStart], paletteSize);
  let runLength = 1;
  for (let x = 1; x < NEATRACING_EDITOR_VOLUME_EDGE; x += 1) {
    const value = resolvePaletteIndex(volume[rowStart + x], paletteSize);
    if (value === runIndex && runLength < 255) {
      runLength += 1;
      continue;
    }
    row += encodeRun(runLength, runIndex);
    runIndex = value;
    runLength = 1;
  }
  return row + encodeRun(runLength, runIndex);
}

/**
 * Clamps a volume palette index into the document palette's range.
 *
 * @param value - Raw palette index stored in the volume.
 * @param paletteSize - Number of entries the document palette holds.
 * @returns The index to write, or 0 (EMPTY) when it is out of range.
 */
function resolvePaletteIndex(value: number, paletteSize: number): number {
  return value < paletteSize ? value : 0;
}

/**
 * Encodes one RLE run into its hex pair.
 *
 * @param runLength - Run length in cells (1..255).
 * @param paletteIndex - Palette index the run paints.
 * @returns The `<runLen:2hex><paletteIdx:2hex>` hex pair.
 */
function encodeRun(runLength: number, paletteIndex: number): string {
  return (
    runLength.toString(16).padStart(2, '0') +
    paletteIndex.toString(16).padStart(2, '0')
  );
}

/**
 * Projects the editor working volume into a section 7.2 model document.
 *
 * The projection keeps the FULL width (the editor stores both mirror
 * halves), so the document declares `symmetric: false` and its rows cover
 * all 128 columns. The document palette is index-aligned with the editor
 * palette (index 0 is the EMPTY sentinel), so the volume's palette indices
 * address the document palette directly.
 *
 * @param config - Volume, palette, and optional model name.
 * @returns The projected model document.
 * @example
 * ```ts
 * const document = buildEditorVolumeDocument({
 *   volume: state.volume,
 *   palette: state.palette,
 *   name: 'car-f1',
 * });
 * console.log(document.meta.dims); // [128, 128, 128]
 * ```
 */
export function buildEditorVolumeDocument(
  config: EditorVolumeDocumentConfig,
): VoxelModelDocument {
  const { volume, palette } = config;
  const name = config.name ?? NEATRACING_EDITOR_DEFAULT_MODEL_NAME;

  // Step 1: Build the index-aligned document palette.
  const documentPalette = buildDocumentPalette(palette);

  // Step 2: Encode every row of every slice into RLE hex rows.
  const slices: string[][] = [];
  for (let z = 0; z < NEATRACING_EDITOR_VOLUME_EDGE; z += 1) {
    const rows: string[] = [];
    for (let y = 0; y < NEATRACING_EDITOR_VOLUME_EDGE; y += 1) {
      rows.push(encodeRow(volume, z, y, documentPalette.length));
    }
    slices.push(rows);
  }

  // Step 3: Stamp the section 7.2 metadata header.
  return {
    meta: {
      name,
      dims: [
        NEATRACING_EDITOR_VOLUME_EDGE,
        NEATRACING_EDITOR_VOLUME_EDGE,
        NEATRACING_EDITOR_VOLUME_EDGE,
      ],
      symmetric: false,
    },
    palette: documentPalette,
    slices,
  };
}
