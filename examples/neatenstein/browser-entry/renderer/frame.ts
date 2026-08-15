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
import type {
  NeatensteinRenderFrame,
  NeatensteinRenderState,
} from './renderer.frame.types';

// Re-export types for external consumers.
export type {
  NeatensteinRenderFrame,
  NeatensteinRenderState,
} from './renderer.frame.types';

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
    // Step 1: copy scalar HUD fields from the render state when present.
    playerHealth: state.playerHealth,
    playerMaxHealth: state.playerMaxHealth,
    playerAmmo: state.playerAmmo,
    playerMaxAmmo: state.playerMaxAmmo,
    // Step 2: kills/deaths fall back to 0 until Phase 5 adds the deaths counter.
    playerKills: state.playerKills ?? 0,
    playerDeaths: state.playerDeaths ?? 0,
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
