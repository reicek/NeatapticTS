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
import type { AmmoPickupState, BoltState, GunState } from '../host/game/types';
import type { NeatensteinSprite } from './sprites';

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
  /** Current FPS-scaled simulation tick. */
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
  /** Optional enemy positions for the sprite pass. */
  enemies?: NeatensteinSprite[];
  /**
   * Delta-time in milliseconds derived from consecutive rAF timestamps and
   * clamped to MAX_DELTA_MS on the host. Consumed by the worker to drive
   * FPS-scaled simulation stepping; omitted/zero on the first frame.
   */
  deltaMs?: number;
  /**
   * Optional enemy population density in [0, 1] forwarded to the HIVE DENSITY
   * HUD overlay. When present, the host HUD updates its meter fill width,
   * threshold color, and percentage label from this value.
   */
  hiveDensity?: number;
  /**
   * Optional human-mode flag forwarded from the human-mode selector. When
   * `'human'`, downstream consumers (including the arms-race configuration)
   * enable replay-driven selection pressure.
   */
  humanMode?: 'auto' | 'human';
  /**
   * Optional scalar HUD fields forwarded from the host game state. The frame
   * builder copies them when present so the HUD status-bar overlay can render
   * health/ammo bars and kill/death labels on every render tier. Kills and
   * deaths fall back to 0 when the source state does not yet carry them.
   */
  playerHealth?: number;
  /** Optional maximum health forwarded from the host player state. */
  playerMaxHealth?: number;
  /** Optional current ammo forwarded from the host player state. */
  playerAmmo?: number;
  /** Optional maximum ammo forwarded from the host player state. */
  playerMaxAmmo?: number;
  /**
   * Optional confirmed-kill count forwarded from the host game state. Falls
   * back to 0 when omitted.
   */
  playerKills?: number;
  /**
   * Optional death count forwarded from the host game state. Falls back to 0
   * when omitted; Phase 5 introduces the deaths counter.
   */
  playerDeaths?: number;
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
  /** Current FPS-scaled simulation tick at frame build time. */
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
  /** Player current health for the health/ammo HUD overlay. */
  playerHealth?: number;
  /** Player maximum health for the health/ammo HUD overlay. */
  playerMaxHealth?: number;
  /** Player current ammo for the health/ammo HUD overlay. */
  playerAmmo?: number;
  /** Player maximum ammo for the health/ammo HUD overlay. */
  playerMaxAmmo?: number;
  /**
   * Total confirmed kills forwarded to the HUD status bar. Falls back to 0
   * when the source state does not carry a kills counter (e.g. before the
   * kill/death counter lands in Phase 5).
   */
  playerKills?: number;
  /**
   * Total deaths forwarded to the HUD status bar. Falls back to 0 until the
   * death/respawn counter is introduced in Phase 5.
   */
  playerDeaths?: number;
  /** Active ammo pickups for the ammo-drop render overlay. */
  ammoPickups?: AmmoPickupState[];
  /**
   * Monotonic spawn counter from the game state, used to derive the current
   * wave number for the centered "Wave N" announcement overlay.
   */
  spawnCount?: number;
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
