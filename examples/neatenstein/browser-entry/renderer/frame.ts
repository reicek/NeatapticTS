/**
 * Render frame protocol for the Neatenstein neon raycasting demo.
 *
 * A frame is a compact, versioned Structure-of-Arrays (SoA) payload that
 * carries per-column renderer outputs. The typed-array buffers can be
 * transferred to a worker via a `Transferable` list for zero-copy transport.
 *
 * @module
 */

import {
  NEATENSTEIN_CPU_COLUMN_COUNT,
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
} from '../constants';
import type { BoltState, GunState } from '../host/game/types';

/**
 * Simulation/render state snapshot needed to build a frame.
 *
 * The renderer consumes the canvas size and the current simulation tick so
 * downstream interpolation and transfer logic can stay synchronized with the
 * simulation clock.
 */
export interface NeatensteinRenderState {
  /** Canvas width in CSS pixels. */
  canvasWidth: number;
  /** Canvas height in CSS pixels. */
  canvasHeight: number;
  /** Current fixed-timestep simulation tick. */
  simTick: number;
  /** Camera world X position in grid units. */
  cameraX: number;
  /** Camera world Y position in grid units. */
  cameraY: number;
  /** Camera horizontal look angle in radians (0 = +X axis). */
  cameraYaw: number;
  /** Deterministic seed used to build the wall grid. */
  mapSeed: number;
  /** Optional movement intent forwarded from the input router. */
  movement?: {
    /** True while the forward key is held. */
    forward: boolean;
    /** True while the backward key is held. */
    backward: boolean;
    /** True while the strafe-left key is held. */
    left: boolean;
    /** True while the strafe-right key is held. */
    right: boolean;
  };
}

/**
 * Versioned render frame used to ship column-wise renderer results from the
 * producer to the consumer (host thread or worker).
 *
 * All typed arrays are sized exactly to {@link columnCount} so consumers can
 * reason about buffer lengths without extra metadata.
 */
export interface NeatensteinRenderFrame {
  /** Human-readable format identifier. */
  format: typeof NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION;
  /** Machine-readable format identifier (mirrors {@link format}). */
  version: typeof NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION;
  /** Monotonically increasing frame sequence number. */
  requestId: number;
  /** Canvas width in CSS pixels. */
  canvasWidth: number;
  /** Canvas height in CSS pixels. */
  canvasHeight: number;
  /** Number of renderer columns in this frame. */
  columnCount: number;
  /** Current fixed-timestep simulation tick at frame build time. */
  simTick: number;
  /** Perpendicular wall distance for each column. */
  wallDistances: Float32Array;
  /** Wall side (0 = X-side, 1 = Y-side) for each column. */
  wallSides: Uint8Array;
  /** Per-column depth buffer used for sprite/pulse occlusion. */
  zBuffer: Float32Array;
  /** Per-column enemy horizontal screen position (for the sprite pass). */
  enemyScreenX: Float32Array;
  /** Per-column enemy projected scale (for the sprite pass). */
  enemyScale: Float32Array;
  /** Per-column projectile horizontal screen position. */
  projectileScreenX: Float32Array;
  /** Current plasma-cannon gun state for the render overlay. */
  gun?: GunState;
  /** Active plasma bolts for the render overlay. */
  bolts?: BoltState[];
  /** Whether the teal dynamic world light is enabled this frame. */
  lightEnabled?: boolean;
}

let nextRequestId = 0;

/**
 * Build a fresh, versioned render frame with SoA typed arrays sized to the
 * requested column count.
 *
 * The returned frame includes only empty typed arrays. Downstream producers
 * (raycaster, wall/floor/sprite passes) fill the arrays before the frame is
 * serialized or transferred.
 *
 * @param state - Current canvas and simulation tick state.
 * @param columnCount - Number of renderer columns for this frame.
 * @returns A freshly allocated render frame.
 *
 * @example
 * ```ts
 * const frame = buildNeatensteinRenderFrame(
 *   {
 *     canvasWidth: 640,
 *     canvasHeight: 360,
 *     simTick: 7,
 *     cameraX: 12.5,
 *     cameraY: 12.5,
 *     cameraYaw: 0.25,
 *     mapSeed: 42,
 *   },
 *   NEATENSTEIN_CPU_COLUMN_COUNT,
 * );
 * ```
 */
export function buildNeatensteinRenderFrame(
  state: NeatensteinRenderState,
  columnCount: number = NEATENSTEIN_CPU_COLUMN_COUNT,
): NeatensteinRenderFrame {
  const id = nextRequestId;
  nextRequestId += 1;

  return {
    format: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
    version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
    requestId: id,
    canvasWidth: state.canvasWidth,
    canvasHeight: state.canvasHeight,
    columnCount,
    simTick: state.simTick,
    wallDistances: new Float32Array(columnCount),
    wallSides: new Uint8Array(columnCount),
    zBuffer: new Float32Array(columnCount),
    enemyScreenX: new Float32Array(columnCount),
    enemyScale: new Float32Array(columnCount),
    projectileScreenX: new Float32Array(columnCount),
  };
}

/**
 * Resolve the list of `Transferable` ArrayBuffers owned by a render frame.
 *
 * Passing this list to `postMessage` transfers ownership of the buffers to the
 * worker instead of cloning them. The producer must not touch the typed
 * arrays after transfer.
 *
 * @param frame - The frame whose buffers should be transferred.
 * @returns Array of `ArrayBuffer` instances suitable for `postMessage`.
 *
 * @example
 * ```ts
 * const frame = buildNeatensteinRenderFrame(state, columnCount);
 * const transfer = resolveNeatensteinRenderFrameTransferList(frame);
 * worker.postMessage({ frame }, transfer);
 * ```
 */
export function resolveNeatensteinRenderFrameTransferList(
  frame: NeatensteinRenderFrame,
): ArrayBuffer[] {
  return [
    frame.wallDistances.buffer as ArrayBuffer,
    frame.wallSides.buffer as ArrayBuffer,
    frame.zBuffer.buffer as ArrayBuffer,
    frame.enemyScreenX.buffer as ArrayBuffer,
    frame.enemyScale.buffer as ArrayBuffer,
    frame.projectileScreenX.buffer as ArrayBuffer,
  ];
}
