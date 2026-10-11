/// <reference lib="dom" />

/**
 * RAF-driven editor model preview render loop (slice P1-editor-preview; AC-019 preview lane).
 *
 * The loop owns the per-frame sequencing only: it eases the Step 08 editor
 * camera transform (`camera.tick(dt)`), builds the column-major view and
 * projection matrices, feeds the orbit eye point plus the view-projection
 * product to an injected draw-only seam, and reschedules itself through the
 * injected RAF seam pair. The actual voxel-to-GPU bridge is deliberately NOT
 * in this module: the selected document is decoded and greedy-meshed here
 * (`decodeVoxelModel` -> `meshVoxelModel`) into a {@link VoxelBinModel}
 * bridge, but the GL draw itself belongs to the bins draw seam supplied by
 * the production wiring in `editor-entry` (preview-only lane, spec AC-019).
 *
 * Document identity memoization: resolve results are cached by document
 * reference (`===`) across frames so a steady selection decodes and meshes
 * exactly once. Decode failures are cached too (the bad document is
 * remembered with no bins), so an invalid selection cannot spam per-frame
 * errors — the loop stays alive and simply draws nothing. The failure
 * itself is surfaced observably through the labeled console surface exactly
 * once per failing document identity (resolution only re-runs on a new
 * document reference), so a blank preview exposes its reason instead of
 * being byte-identical to an empty selection.
 *
 * @module
 */

import { decodeVoxelModel } from '../voxel/model.decode.utils';
import { meshVoxelModel } from '../voxel/greedy-mesher';
import type { VoxelBinModel } from '../voxel/bin.codec';
import type { VoxelModelDocument } from '../voxel/model.schema';
import type { EditorCamera, EditorCameraPose } from './editor.camera';

/** Seconds elapsed on the first frame: the camera tick is a no-op there. */
const FIRST_FRAME_DT_SECONDS = 0;

/** Milliseconds per second: RAF timestamps are milliseconds since origin. */
const MS_PER_SECOND = 1000;

/** Pass label prefixing the preview loop's observable failure surface. */
const NEATRACING_EDITOR_PREVIEW_LABEL = 'neatRacing editor preview';

/** Component count of one column-major 4x4 matrix. */
const MATRIX_COMPONENT_COUNT = 16;

/** Structural canvas view the loop consumes; only width and height matter. */
export interface EditorPreviewCanvasView {
  /** Drawable width in CSS/drawing-buffer pixels. */
  readonly width: number;
  /** Drawable height in CSS/drawing-buffer pixels. */
  readonly height: number;
}

/** Per-frame draw state handed to the bins draw seam. */
export interface EditorPreviewBinsDrawState {
  /** Column-major view-projection product (projection * view), 16 floats. */
  readonly viewProjection: Float32Array;
  /** Orbit pose eye point in world units (the camera position). */
  readonly cameraWorld: readonly [number, number, number];
  /** Greedy-meshed voxel bins of the currently selected model document. */
  readonly bins: VoxelBinModel;
}

/** Draw-only seam the loop drives once per frame; the seam owns disposal. */
export interface EditorPreviewBinsDraw {
  /**
   * Draw one frame of the preview bins under the given camera state.
   *
   * @param state - View-projection product, orbit eye point, and bins.
   */
  draw(state: EditorPreviewBinsDrawState): void;

  /** Release every GL resource the draw implementation created. */
  dispose(): void;
}

/** Handle returned by the loop factory; `stop` is idempotent. */
export interface EditorPreviewRenderLoopHandle {
  /** Cancel the pending frame and dispose the bins draw seam exactly once. */
  stop(): void;
}

/** Frame scheduling seam mirroring `window.requestAnimationFrame`. */
export type EditorPreviewRequestFrame = (
  callback: (timeMs: number) => void,
) => number;

/** Frame cancellation seam mirroring `window.cancelAnimationFrame`. */
export type EditorPreviewCancelFrame = (frameId: number) => void;

/** Configuration for {@link startEditorPreviewRenderLoop}. */
export interface EditorPreviewRenderLoopConfig {
  /** Structural canvas view supplying the projection aspect ratio. */
  readonly canvas: EditorPreviewCanvasView;
  /** Step 08 editor camera the loop eases and samples per frame. */
  readonly camera: EditorCamera;
  /** Selected document accessor; null keeps the chain running with no draws. */
  readonly getSelectedModelDocument: () => VoxelModelDocument | null;
  /** Injected draw-only seam; the loop calls its `dispose` on stop. */
  readonly binsDraw: EditorPreviewBinsDraw;
  /** Optional frame scheduling seam; defaults to the window RAF pair. */
  readonly requestFrame?: EditorPreviewRequestFrame;
  /** Optional frame cancellation seam; defaults to the window RAF pair. */
  readonly cancelFrame?: EditorPreviewCancelFrame;
}

/** Memoized document -> bins resolution keyed by document reference. */
interface PreviewBinsMemoEntry {
  /** Last document the memo resolved, or null for "nothing selected". */
  document: VoxelModelDocument | null;
  /** Bins for the last resolved document, or null (none selected/failed). */
  bins: VoxelBinModel | null;
}

/** Default scheduling seam: the browser RAF (only invoked when no seam injected). */
const defaultRequestFrame: EditorPreviewRequestFrame = (callback) =>
  window.requestAnimationFrame(callback);

/** Default cancellation seam: the browser RAF cancel (lazy `window` access). */
const defaultCancelFrame: EditorPreviewCancelFrame = (frameId) => {
  window.cancelAnimationFrame(frameId);
};

/**
 * Resolve the selected document into greedy-meshed bins, memoized per
 * document identity.
 *
 * The bridged bins pair the mesher's `VoxelQuad[]` (a structural subset of
 * `VoxelBinQuad`) with the document's validated palette. Decode failures are
 * cached as "no bins for this document" so an invalid selection is remembered
 * instead of re-thrown once per frame, and surfaced through the labeled
 * console surface exactly once per failing document identity.
 *
 * @param memo - Per-loop memo entry; updated in place.
 * @param document - Currently selected document (never null; null selections
 *   are handled by the caller and never reach this resolver).
 * @returns The bridged bins for the document, or null when the document
 *   failed to decode.
 *
 * @example
 * ```ts
 * const memo = { document: null, bins: null };
 * const bins = document === null ? null : resolvePreviewBins(memo, document);
 * if (bins !== null) drawBins(bins);
 * ```
 */
function resolvePreviewBins(
  memo: PreviewBinsMemoEntry,
  document: VoxelModelDocument,
): VoxelBinModel | null {
  if (document === memo.document) {
    return memo.bins;
  }
  // No initializer: every path through the try/catch assigns `nextBins`
  // before the read below, so a default value would never be observed
  // (no-useless-assignment).
  let nextBins: VoxelBinModel | null;
  try {
    const decoded = decodeVoxelModel(document);
    const mesh = meshVoxelModel(decoded);
    nextBins = {
      dims: mesh.dims,
      palette: decoded.palette,
      quads: mesh.quads,
    };
  } catch (error) {
    // Cache the failure AND surface it observably: remember this document
    // with no bins so the loop keeps ticking without per-frame throw spam
    // (see module doc), but never leave the failure byte-identical to an
    // empty selection — the labeled console surface fires once per failing
    // document identity because the memo short-circuits repeat frames.
    console.error(
      `${NEATRACING_EDITOR_PREVIEW_LABEL}: selected model decode/mesh failed; preview renders nothing`,
      error,
    );
    nextBins = null;
  }
  memo.document = document;
  memo.bins = nextBins;
  return nextBins;
}

/**
 * Multiply a column-major projection onto a view and write the (projection *
 * view) product in place: `out[column * 4 + row] = sum_k projection
 * [k * 4 + row] * view[column * 4 + k]`.
 *
 * The (projection * view) order is the seam contract because the consuming
 * car-pass GLSL applies the stored matrix as `gl_Position = uViewProjection *
 * world` (column-major, transpose = false): the view transform must apply to
 * world eye-space positions before the projection. A (view * projection)
 * product would corrupt the perspective row's `w_clip` and degenerate the
 * car draw.
 *
 * Allocation-free: the 4x4 output buffer is owned by the caller and reused
 * every frame (frame-composer bag discipline, AC-042).
 *
 * @param out - Pre-allocated 16-float output buffer for the product.
 * @param view - Column-major 4x4 view matrix.
 * @param projection - Column-major 4x4 projection matrix.
 *
 * @example
 * ```ts
 * const product = new Float32Array(16);
 * multiplyViewProjection(product, view, projection);
 * ```
 */
function multiplyViewProjection(
  out: Float32Array,
  view: Float32Array,
  projection: Float32Array,
): void {
  for (let columnIndex = 0; columnIndex < 4; columnIndex += 1) {
    for (let rowIndex = 0; rowIndex < 4; rowIndex += 1) {
      let accumulator = 0;
      for (let termIndex = 0; termIndex < 4; termIndex += 1) {
        accumulator +=
          projection[termIndex * 4 + rowIndex] *
          view[columnIndex * 4 + termIndex];
      }
      out[columnIndex * 4 + rowIndex] = accumulator;
    }
  }
}

/**
 * Compute the orbit pose eye point into the pre-owned camera-world triple.
 *
 * Uses the documented plain-trig eye derivation (mirroring the reference
 * camera-eye formula in editor.camera JSDoc): `center + distance *
 * [cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw)]`. Trig-snapping
 * is deliberately NOT applied here — the camera applies its own snapping
 * inside `buildViewMatrix`, while the draw seam's eye must be the raw pose
 * value.
 *
 * @param pose - The camera's current (smoothed) orbit pose.
 * @param out - Pre-owned triple refreshed in place.
 *
 * @example
 * ```ts
 * const cameraWorld: [number, number, number] = [0, 0, 0];
 * orbitCameraWorld(camera.snapshot().current, cameraWorld);
 * ```
 */
function orbitCameraWorld(
  pose: EditorCameraPose,
  out: [number, number, number],
): void {
  out[0] =
    pose.center[0] + pose.distance * Math.cos(pose.pitch) * Math.sin(pose.yaw);
  out[1] = pose.center[1] + pose.distance * Math.sin(pose.pitch);
  out[2] =
    pose.center[2] + pose.distance * Math.cos(pose.pitch) * Math.cos(pose.yaw);
}

/**
 * Start the editor model preview render loop (AC-019 preview lane).
 *
 * Each scheduled frame eases the camera transform, resolves the selected
 * document through the decode -> greedy-mesher chain (memoized per document
 * identity), and — when bins exist — builds the column-major view-projection
 * product plus the orbit eye point and hands them to the bins draw seam.
 * With no document selected (or a document that fails to decode) the RAF
 * chain keeps ticking and the loop draws nothing. `stop` cancels the pending
 * frame and disposes the bins draw exactly once; repeated stops are no-ops.
 *
 * @param config - Loop configuration: structural canvas view, editor camera,
 *   selected-document accessor, bins draw seam, and optional RAF seam
 *   defaults (the window `requestAnimationFrame`/`cancelAnimationFrame`
 *   pair).
 * @returns A handle whose single `stop` cancels the pending frame and
 *   disposes the bins draw; safe to call more than once.
 *
 * @example
 * ```ts
 * const loop = startEditorPreviewRenderLoop({
 *   canvas,
 *   camera,
 *   getSelectedModelDocument,
 *   binsDraw,
 * });
 * loop.stop();
 * ```
 */
export function startEditorPreviewRenderLoop(
  config: EditorPreviewRenderLoopConfig,
): EditorPreviewRenderLoopHandle {
  const requestFrame = config.requestFrame ?? defaultRequestFrame;
  const cancelFrame = config.cancelFrame ?? defaultCancelFrame;
  const viewProjection = new Float32Array(MATRIX_COMPONENT_COUNT);
  const cameraWorld: [number, number, number] = [0, 0, 0];
  const memo: PreviewBinsMemoEntry = { document: null, bins: null };
  let pendingFrameId: number | null = null;
  let previousTimeMs: number | null = null;
  let stopped = false;

  const frame = (nowMs: number): void => {
    pendingFrameId = null;
    if (stopped) {
      return;
    }
    const deltaSeconds =
      previousTimeMs === null
        ? FIRST_FRAME_DT_SECONDS
        : (nowMs - previousTimeMs) / MS_PER_SECOND;
    previousTimeMs = nowMs;
    config.camera.tick(deltaSeconds);
    const document = config.getSelectedModelDocument();
    if (document !== null) {
      const bins = resolvePreviewBins(memo, document);
      if (bins !== null) {
        multiplyViewProjection(
          viewProjection,
          config.camera.buildViewMatrix(),
          config.camera.buildProjectionMatrix(
            config.canvas.width / config.canvas.height,
          ),
        );
        orbitCameraWorld(config.camera.snapshot().current, cameraWorld);
        config.binsDraw.draw({ viewProjection, cameraWorld, bins });
      }
    }
    pendingFrameId = requestFrame(frame);
  };

  pendingFrameId = requestFrame(frame);

  return {
    stop: (): void => {
      if (stopped) {
        return;
      }
      stopped = true;
      if (pendingFrameId !== null) {
        cancelFrame(pendingFrameId);
        pendingFrameId = null;
      }
      config.binsDraw.dispose();
    },
  };
}
