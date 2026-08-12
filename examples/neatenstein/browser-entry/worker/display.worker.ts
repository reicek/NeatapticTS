/**
 * Display worker entrypoint for the Neatenstein neon raycasting demo.
 *
 * This worker receives an `init` message that selects a renderer tier, then
 * either:
 *
 * - renders directly to an {@link OffscreenCanvas} for the `worker` tier, or
 * - builds a packed {@link NeatensteinRenderFrame} and posts it back to the
 *   host for the `cpu` and `gpu` tiers.
 *
 * The worker also owns the deterministic simulation step: it runs
 * {@link gameTick}, advances the NGE enemy population, drives enemy AI,
 * consumes enemy hitscan events to spawn return-fire bolts, and renders death
 * de-rez effects.
 *
 * The worker renders at the host-provided backing-store dimensions. It only
 * guards against non-finite or non-positive dimensions and synchronizes the
 * transferred OffscreenCanvas to the incoming state size for the `worker` tier;
 * it does not impose a maximum render resolution.
 *
 * @module
 */

/// <reference lib="webworker" />

import {
  NEATENSTEIN_CPU_COLUMN_COUNT,
  NEATENSTEIN_DEFAULT_SEED,
  NEATENSTEIN_GPU_COLUMN_COUNT,
  NEATENSTEIN_INPUT_MESSAGE_TYPE,
  NEATENSTEIN_MAP_SIZE,
  NEATENSTEIN_PULSE_ALPHA_MAX,
  NEATENSTEIN_PULSE_ALPHA_MIN,
  NEATENSTEIN_PULSE_GLOW_BLUR_RADIUS,
  NEATENSTEIN_PULSE_MAX_CONCURRENT,
  NEATENSTEIN_PULSE_SCREEN_DOT_RADIUS_PX,
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
} from '../constants';
import {
  buildNeatensteinRenderFrame,
  resolveNeatensteinRenderFrameTransferList,
  type NeatensteinRenderState,
} from '../renderer/frame';
import {
  drawNeatensteinCeiling,
  drawNeatensteinFloor,
  NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
  NEATENSTEIN_FLOOR_FOV_RADIANS,
  NEATENSTEIN_FLOOR_HORIZON_RATIO,
  projectNeatensteinCeilingPoint,
  projectNeatensteinFloorPoint,
  type NeatensteinFloorCamera,
} from '../renderer/floor';
import {
  depthTestPulse,
  emitNeatensteinAmbientPulse,
  NEATENSTEIN_PULSE_AMBIENT_LIFETIME_TICKS,
  NEATENSTEIN_PULSE_LAYER_CEILING,
  NEATENSTEIN_PULSE_LAYER_FLOOR,
  updateNeatensteinPulses,
  type NeatensteinPulse,
} from '../renderer/pulse';
import {
  buildNeatensteinMap,
  createCollisionMap,
  type CollisionMap,
} from '../renderer/map';
import { castRayDDAFromFlatMap } from '../renderer/raycast';
import {
  createGameState,
  gameTick,
  type GameTickInputSnapshot,
} from '../host/game/tick';
import { fireEnemyBolt } from '../host/game/combat';
import {
  NEATENSTEIN_ENEMY_MAX_CONCURRENT,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_KITING_APPROACH_DISTANCE_CELLS,
  NEATENSTEIN_KITING_BACKPEDAL_DISTANCE_CELLS,
  NEATENSTEIN_EXPLORATION_BOUNCE_ANGLE_RAD,
  NEATENSTEIN_EXPLORATION_WALL_BOUNCE_LOOKAHEAD_CELLS,
} from '../host/game/constants';
import type { EnemyState, GameState } from '../host/game/types';
import { advanceWave } from '../host/waves';
import {
  NEATENSTEIN_BACKGROUND_RGB,
  NEATENSTEIN_RENDER_DISTANCE_CAP,
} from '../renderer/framebuffer';
import { renderGunOverlay } from '../renderer/gun';
import {
  drawBolts,
  drawAmmoPickups,
  drawEnemyBolts,
  drawEnemyImpactSpots,
  drawImpactSpots,
} from '../renderer/bolt-render';
import {
  createEnemyControllerState,
  updateEnemyController,
  ENEMY_CONTROLLER_DE_REZ_DURATION_MS,
  type EnemyControllerState,
} from '../../scripts/enemy-controller';
import {
  extractSensors,
  findNearestVisibleEnemy,
} from '../../scripts/enemy-navigation';
import { createMlpEnemyPopulation } from '../harness/enemy-mlp';
import type { MlpEnemyPopulation } from '../harness/enemy-mlp';
import type { MlpSnapshot } from '../harness/types';
import {
  NEATENSTEIN_MAIN_NEAT_INPUTS,
  NEATENSTEIN_FALLBACK_TURN_RATE,
  NEATENSTEIN_MOVE_ACCEL_PER_TICK,
  NEATENSTEIN_MOVE_DECEL_PER_TICK,
  networkOutputToTickInput,
  applyFireGate,
  createFireGateState,
  smoothCommand,
  ENEMY_VISIBLE_SENSOR_INDEX,
  type FireGateState,
} from '../harness/neat-io-config';
import type { Network } from 'neataptic';
import {
  clipNeatensteinSprite,
  renderNeatensteinSprite,
  resolveNeatensteinEnemyFrame,
  type NeatensteinSprite,
  type NeatensteinSpriteRenderContext,
} from '../renderer/sprites';

type DisplayTier = 'worker' | 'cpu' | 'gpu';

let currentTier: DisplayTier | null = null;
let workerCanvas: OffscreenCanvas | null = null;
let workerContext: OffscreenCanvasRenderingContext2D | null = null;
let latestState: NeatensteinRenderState | null = null;

/**
 * Host resize dimensions received before the worker canvas is assigned.
 *
 * Resize messages can arrive before the `init` message finishes setting up
 * `workerCanvas`, so the last valid host dimension is stashed and applied once
 * the canvas is available.
 */
let pendingResizeDimensions: { width: number; height: number } | null = null;

/** Canonical flat deterministic map used by the worker raycaster. */
let wallMap: Uint8Array | null = null;

/** Reusable worker-tier z-buffer, resized only when the render width changes. */
let workerZBuffer: Float32Array | null = null;

/** Deterministic game state maintained and advanced by the worker. */
let gameState: GameState | null = null;

/** Collision map built from the init seed; reused for every movement tick. */
let collisionMap: CollisionMap | null = null;

/** Active ambient floor and ceiling pulses tracked across worker frames. */
let activePulses: NeatensteinPulse[] = [];

/**
 * Enemy AI controller state maintained across simulation ticks so fire
 * cooldowns, ammunition, and de-rez timing advance deterministically.
 */
let enemyControllerState: EnemyControllerState | null = null;

/**
 * Active enemy sprite positions computed by the controller this frame.
 *
 * Stored separately from {@link latestState} so the next slice can render
 * sprites after the wall pass without recomputing controller state.
 */
let activeEnemySprites: NeatensteinSprite[] = [];

/**
 * Wave-clear detection flag set when the last alive enemy is cleared in the
 * current wave.
 *
 * Tracks the `false → true` transition: the flag becomes `true` when the
 * enemy roster transitions from at least one alive enemy to none alive.
 * Dead/inactive enemies are kept in the array for index alignment and are
 * intentionally ignored. The flag is consumed by `advanceWave` (P3S1)
 * which increments `gameState.generation` and respawns the next wave.
 * P2S1 only detects and stores the flag; it does NOT increment
 * `generation`.
 *
 * @see AC-017
 */
let allEnemiesCleared = false;

/**
 * Previous-tick snapshot of {@link allEnemiesCleared} used to detect the
 * `false → true` transition that triggers {@link advanceWave}.
 *
 * @see AC-037
 */
let prevAllEnemiesCleared = false;

/**
 * MLP enemy population maintained by the worker for champion weight injection.
 *
 * Created on init from the game seed and advanced on each tick so that
 * refresh generations produce a new champion snapshot. The champion weights
 * are injected into every enemy controller pass via the optional `weights`
 * parameter of {@link updateEnemyController}.
 *
 * @see AC-024
 */
let enemyPopulation: MlpEnemyPopulation | null = null;

/**
 * Launch guard for the hoisted async Neat evaluation (P3S2).
 *
 * Stores the generation currently being evaluated, or `null` when no
 * evaluation is in flight. Before posting a new evaluation request to the
 * eval worker, the display worker checks this guard and skips if already
 * evaluating — concurrent wave-clears are dropped (not queued). On
 * completion, the eval worker posts back `result.generation` which must
 * equal `pendingGeneration + 1` before the result is applied.
 *
 * @see AC-043a
 */
let pendingGeneration: number | null = null;

/**
 * Champion main-agent network from the most recent arms-race generation.
 *
 * Stored for Phase 4's player auto-mode controller, which activates this
 * network to produce `GameTickInputSnapshot` from game-state sensors.
 *
 * @see AC-060a
 */
let championMainNetwork: Network | null = null;

/**
 * Input count the current champion network was evolved with.
 *
 * When the champion is stored (via {@link handleEvalComplete} or
 * {@link __testOnlySetChampionMainNetwork}), this is set to the current
 * {@link NEATENSTEIN_MAIN_NEAT_INPUTS} value. Before the champion is used
 * for auto-mode ticks, a guard compares this to the live constant; a
 * mismatch triggers a genome-extinction event that clears the champion.
 *
 * @see AC-P3S1c-002
 */
let lastChampionInputCount: number | null = null;

/**
 * Dedicated eval worker that handles NEAT population evaluation off the
 * render loop.
 *
 * Created lazily on the first wave-clear that triggers an arms-race
 * evaluation. In the browser, the worker URL is derived from the display
 * worker's own location. In tests, a mock worker is injected via
 * {@link __testOnlySetEvalWorker}.
 *
 * @see AC-P2S1b-001, AC-P2S1b-002
 */
let evalWorker: Worker | null = null;

/**
 * Input snapshot captured from the most recent `input` message.
 *
 * Movement/look use the latest value, while one-shot actions such as fire and
 * dash are latched until the next simulation tick consumes them.
 */
let pendingTickInput: GameTickInputSnapshot | null = null;

/**
 * Tracks the source of the most recent tick input for test introspection.
 *
 * `'auto'` — the champion NEAT network produced the tick input.
 * `'human'` — the pending human input queue (or default zero) was used.
 *
 * @internal
 */
let lastTickInputSource: 'auto' | 'human' = 'human';

/**
 * Monotonic tick counter for the fallback auto-mode AI.
 *
 * Incremented on every fallback tick and used to gate firing via
 * {@link NEATENSTEIN_FALLBACK_FIRE_INTERVAL}.  Reset to 0 on init.
 *
 * @see buildFallbackAutoTickInput
 */
let fallbackTickCounter = 0;

/**
 * Test-only capture of the last fallback input snapshot.
 *
 * Used by red/green tests to assert exact move and lookDelta values without
 * depending on physics simulation side effects.
 *
 * @internal
 */
let lastFallbackInputForTest: GameTickInputSnapshot | null = null;

/**
 * P5S1 — Hysteresis state for the soft fire gate.
 *
 * Maintained between ticks to prevent rapid on/off oscillation at the
 * vision boundary.  Reset on init.
 *
 * @see buildAutoTickInput
 * @see AC-P5S1a-003
 */
let fireGateState: FireGateState = createFireGateState();

/**
 * P1S1 — Smoothed move/look state for auto-mode inputs.
 *
 * Persisted between ticks so the AI ramps move and look commands instead
 * of snapping from 0 to max instantly.  Reset on init.
 *
 * @see buildAutoTickInput
 * @see buildFallbackAutoTickInput
 * @see AC-003, AC-004
 */
let smoothedMoveX = 0;
let smoothedMoveY = 0;
let smoothedLookDelta = 0;

/** RGB of the neon wall color for X-axis-side hits. */
const NEATENSTEIN_WALL_X_SIDE_RGB = { r: 0, g: 183, b: 255 } as const;

/** RGB of the neon wall color for Y-axis-side hits. */
const NEATENSTEIN_WALL_Y_SIDE_RGB = { r: 0, g: 164, b: 229 } as const;

/** CSS color string for the yellow ambient pulse dot. */
const NEATENSTEIN_PULSE_COLOR = '#B7FF00';

/** CSS shadow color string for the yellow ambient pulse glow. */
const NEATENSTEIN_PULSE_GLOW_COLOR = 'rgba(185, 255, 0, 0.42)';

/**
 * Format an RGB triple as a CSS `rgb(...)` string.
 *
 * @param color - RGB color object.
 * @returns CSS color string.
 */
function formatRgb(color: { r: number; g: number; b: number }): string {
  return `rgb(${Math.round(color.r)}, ${Math.round(color.g)}, ${Math.round(color.b)})`;
}

/** Worker clear color matching the dark neon void background. */
const NEATENSTEIN_WORKER_CLEAR_COLOR = formatRgb(NEATENSTEIN_BACKGROUND_RGB);

/**
 * Golden-angle rotation used to spread distinct enemy team hues across the
 * color wheel so multiple enemies remain visually separable.
 */
const NEATENSTEIN_ENEMY_TEAM_HUE_GOLDEN_STEP = 137.508;

/**
 * Resolve a deterministic enemy team color `[r, g, b]` from the enemy type
 * index by rotating the HSL hue wheel with the golden angle. Palette indices
 * 5/6/7 in the robot sprite atlas are swapped with this color at render time,
 * giving each enemy a stable neon tint while preserving the sprite alpha.
 *
 * @param typeIndex - Enemy type index (the source {@link GameState.enemies}
 *   array position). Negative indices fall back to index 0.
 * @returns RGB triple in the `[0, 255]` range.
 */
function resolveEnemyTeamColor(
  typeIndex: number,
): readonly [number, number, number] {
  const seed = typeIndex >= 0 ? typeIndex : 0;
  const hue = (seed * NEATENSTEIN_ENEMY_TEAM_HUE_GOLDEN_STEP) % 360;
  const saturation = 0.65;
  const lightness = 0.5;
  const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation;
  const huePrime = hue / 60;
  const intermediate = chroma * (1 - Math.abs((huePrime % 2) - 1));
  let r = 0;
  let g = 0;
  let b = 0;
  if (huePrime < 1) {
    r = chroma;
    g = intermediate;
  } else if (huePrime < 2) {
    r = intermediate;
    g = chroma;
  } else if (huePrime < 3) {
    g = chroma;
    b = intermediate;
  } else if (huePrime < 4) {
    g = intermediate;
    b = chroma;
  } else if (huePrime < 5) {
    r = intermediate;
    b = chroma;
  } else {
    r = chroma;
    b = intermediate;
  }
  const match = lightness - chroma / 2;
  return [
    Math.round((r + match) * 255),
    Math.round((g + match) * 255),
    Math.round((b + match) * 255),
  ] as const;
}

/**
 * Clamp a value to a `[min, max]` range.
 *
 * @param value - Value to clamp.
 * @param min - Inclusive minimum.
 * @param max - Inclusive maximum.
 * @returns The clamped value.
 */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Return whether a value is usable as a positive render dimension.
 *
 * @param value - Candidate dimension.
 * @returns Whether the dimension is positive and finite.
 */
function isPositiveFiniteDimension(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

/**
 * Synchronize the transferred OffscreenCanvas with the host-provided render
 * size.
 *
 * This is critical for direct worker rendering. Projection math, floor/ceiling
 * drawing, wall stripes, z-buffer columns, and the actual canvas backing store
 * must agree on the same dimensions.
 *
 * @param canvas - Transferred worker-owned canvas.
 * @param width - Host-provided backing-store width.
 * @param height - Host-provided backing-store height.
 */
function syncWorkerCanvasSize(
  canvas: OffscreenCanvas,
  width: number,
  height: number,
): void {
  if (canvas.width !== width) {
    canvas.width = width;
  }

  if (canvas.height !== height) {
    canvas.height = height;
  }
}

/**
 * Apply a host-resized CSS-box dimension to the worker canvas and latest state.
 *
 * The host owns the visible canvas CSS box, while the worker owns the backing
 * store. This helper updates both so the next frame renders at the new size
 * even if it arrives before the next `simState` tick.
 *
 * @param width - Host-derived render width in pixels.
 * @param height - Host-derived render height in pixels.
 */
function applyWorkerResize(width: number, height: number): void {
  if (!isPositiveFiniteDimension(width) || !isPositiveFiniteDimension(height)) {
    return;
  }

  if (workerCanvas !== null) {
    syncWorkerCanvasSize(workerCanvas, width, height);
  } else {
    pendingResizeDimensions = { width, height };
  }

  if (latestState !== null) {
    latestState.canvasWidth = width;
    latestState.canvasHeight = height;
  }
}

/**
 * Resolve the direct worker-tier raycast column count.
 *
 * The worker tier renders directly into the OffscreenCanvas, so its horizontal
 * ray density should match the host-provided backing-store width.
 *
 * @param canvasWidth - Canvas backing-store width in pixels.
 * @returns Number of direct worker raycast columns.
 */
function resolveWorkerCanvasColumnCount(canvasWidth: number): number {
  return Math.max(1, Math.floor(canvasWidth));
}

/**
 * Map a packed-frame tier to the column count used in typed frame payloads.
 *
 * Direct worker rendering does not use this helper; it raycasts at the
 * host-provided canvas width instead.
 *
 * @param tier - Active renderer tier.
 * @returns Packed-frame column count.
 */
function resolvePackedColumnCount(tier: DisplayTier): number {
  if (tier === 'gpu') {
    return NEATENSTEIN_GPU_COLUMN_COUNT;
  }
  return NEATENSTEIN_CPU_COLUMN_COUNT;
}

/**
 * Resolve and clear the reusable worker z-buffer.
 *
 * @param columnCount - Required number of z-buffer columns.
 * @returns A cleared z-buffer.
 */
/* istanbul ignore next -- pre-existing z-buffer reuse helper; only the allocation branch is exercised by this test suite */
function resolveWorkerZBuffer(columnCount: number): Float32Array {
  if (!workerZBuffer || workerZBuffer.length !== columnCount) {
    workerZBuffer = new Float32Array(columnCount);
  }

  workerZBuffer.fill(Number.POSITIVE_INFINITY);
  return workerZBuffer;
}

/**
 * Resolve a safe distance-fog interpolation factor.
 *
 * @param perpWallDist - Perpendicular wall distance.
 * @returns Fog factor in `[0, 1]`.
 */
function resolveWallFogFactor(perpWallDist: number): number {
  return perpWallDist >= NEATENSTEIN_RENDER_DISTANCE_CAP ? 1 : 0;
}

/**
 * Apply distance fog to a neon wall color and return a CSS color string.
 *
 * @param wallColor - Raw wall RGB.
 * @param perpWallDist - Perpendicular distance from camera to wall.
 * @returns CSS `rgb(...)` string for `context.fillStyle`.
 */
function applyWallFog(
  wallColor: { r: number; g: number; b: number },
  perpWallDist: number,
): string {
  const fogFactor = resolveWallFogFactor(perpWallDist);
  const bg = NEATENSTEIN_BACKGROUND_RGB;

  return formatRgb({
    r: wallColor.r + (bg.r - wallColor.r) * fogFactor,
    g: wallColor.g + (bg.g - wallColor.g) * fogFactor,
    b: wallColor.b + (bg.b - wallColor.b) * fogFactor,
  });
}

/**
 * Build a canvas-like context that does not flush per sprite.
 *
 * The worker tier renders all sprites into a single canvas snapshot and flushes
 * it back once after the sprite pass. The renderer's `putImageData` contract
 * still expects a context, so this no-op adapter satisfies the type without
 * redundant per-sprite copies.
 *
 * @returns Canvas-like context with a no-op `putImageData`.
 */
function buildNoOpSpriteRenderContext(): NeatensteinSpriteRenderContext {
  return {
    putImageData: () => {
      // Intentionally empty: the worker flushes the snapshot once after all
      // sprites are drawn.
    },
  };
}

/** Return type of the shared raycaster used to build every column. */
type RaycastHit = ReturnType<typeof castRayDDAFromFlatMap>;

/**
 * Cast a single camera ray for the given column index.
 *
 * @param column - Column index in `[0, columnCount)`.
 * @param columnCount - Total number of raycast columns.
 * @param cameraPositionX - Camera X position.
 * @param cameraPositionY - Camera Y position.
 * @param cameraDirectionX - Camera direction X.
 * @param cameraDirectionY - Camera direction Y.
 * @param cameraPlaneX - Camera plane X.
 * @param cameraPlaneY - Camera plane Y.
 * @returns DDA raycast hit or miss sentinel.
 */
function castColumnRay(
  column: number,
  columnCount: number,
  cameraPositionX: number,
  cameraPositionY: number,
  cameraDirectionX: number,
  cameraDirectionY: number,
  cameraPlaneX: number,
  cameraPlaneY: number,
): RaycastHit {
  // Map the column index to a -1..+1 offset on the camera plane.
  const cameraPlaneOffset = (2 * column) / columnCount - 1;

  // Combine camera forward direction with the camera plane offset.
  const rayDirectionX = cameraDirectionX + cameraPlaneX * cameraPlaneOffset;
  const rayDirectionY = cameraDirectionY + cameraPlaneY * cameraPlaneOffset;

  return castRayDDAFromFlatMap(
    wallMap!,
    NEATENSTEIN_MAP_SIZE,
    cameraPositionX,
    cameraPositionY,
    rayDirectionX,
    rayDirectionY,
  );
}

/**
 * Build a packed frame from the latest state or render directly to the worker
 * OffscreenCanvas.
 *
 * Worker-tier painter order:
 *
 * clear → floor grid → ceiling grid → fogged wall stripes → sprite snapshot
 * → encoded enemy sprites → sprite flush → pulses → impact spots → enemy
 * impact spots → bolts → gun overlay
 */
function buildAndPostFrame(): void {
  if (!latestState || !currentTier || !wallMap || !gameState || !collisionMap) {
    return;
  }

  const canvasWidth = latestState.canvasWidth;
  const canvasHeight = latestState.canvasHeight;

  if (
    !isPositiveFiniteDimension(canvasWidth) ||
    !isPositiveFiniteDimension(canvasHeight)
  ) {
    return;
  }

  // Use the worker-authoritative simulated player position for every render
  // pass (walls, floor, ceiling, pulses, bolts, and sprites). The host still
  // sends a camera snapshot, but it is intentionally not used for rendering so
  // that WASD movement and mouse rotation stay in sync across the whole frame.
  const cameraPositionX = gameState.player.position.x;
  const cameraPositionY = gameState.player.position.y;
  const cameraYaw = gameState.player.angleRad;

  // Sprite projection uses the same authoritative camera position as the wall
  // renderer to prevent enemies from appearing to move when the player walks.
  const spriteCameraX = cameraPositionX;
  const spriteCameraY = cameraPositionY;
  const spriteCameraYaw = cameraYaw;

  // Derive camera direction and projection plane from the player yaw.
  const cameraDirectionX = Math.cos(cameraYaw);
  const cameraDirectionY = Math.sin(cameraYaw);
  const planeScale =
    (canvasWidth / canvasHeight) * Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
  const cameraPlaneX = -cameraDirectionY * planeScale;
  const cameraPlaneY = cameraDirectionX * planeScale;

  // Sprite projection uses the authoritative simulated player transform so
  // enemy positions (advanced by the controller) stay in the same camera
  // space as the sprite renderer.
  const spriteDirectionX = Math.cos(spriteCameraYaw);
  const spriteDirectionY = Math.sin(spriteCameraYaw);
  const spritePlaneX = -spriteDirectionY * planeScale;
  const spritePlaneY = spriteDirectionX * planeScale;

  // The enemy controller is advanced before gameTick in the simState handler
  // so gameState.enemies already reflects the latest controlled positions.
  // Build the active sprite list from the persisted controller state.
  activeEnemySprites = enemyControllerState!.enemies
    .filter((enemy) => enemy.active)
    .map((enemy) => ({
      worldX: enemy.position.x,
      worldY: enemy.position.y,
      facing: enemy.yawRad,
      animationState: enemy.animationState,
      frameIndex: 0,
      type: enemy.index,
      walkTick: enemy.walkTick,
      shootBlinkTicks: enemy.shootBlinkTicks,
      teamColor: resolveEnemyTeamColor(enemy.index),
      deRezElapsedMs: enemy.deRezElapsedMs,
      deRezDurationMs: ENEMY_CONTROLLER_DE_REZ_DURATION_MS,
      seed: enemy.index,
    }));

  // Stash the active enemy sprites on the incoming render state so the next
  // slice's sprite pass can render them for any tier without recomputing the
  // controller state.
  latestState.enemies = activeEnemySprites;

  if (currentTier === 'worker') {
    if (!workerCanvas) {
      return;
    }

    // The worker owns the transferred canvas, so it must synchronize the
    // backing store to the host-provided dimensions before any drawing.
    syncWorkerCanvasSize(workerCanvas, canvasWidth, canvasHeight);

    if (!workerContext) {
      workerContext = workerCanvas.getContext('2d');
    }

    const context = workerContext;
    if (!context) {
      return;
    }

    const columnCount = resolveWorkerCanvasColumnCount(canvasWidth);
    const zBuffer = resolveWorkerZBuffer(columnCount);

    // Clear the full canvas to the dark neon void background.
    context.fillStyle = NEATENSTEIN_WORKER_CLEAR_COLOR;
    context.fillRect(0, 0, canvasWidth, canvasHeight);

    // Draw perspective floor and ceiling grids behind the walls.
    const floorCamera = {
      x: cameraPositionX,
      y: cameraPositionY,
      yaw: cameraYaw,
    };
    drawNeatensteinFloor(context, canvasWidth, canvasHeight, floorCamera);
    drawNeatensteinCeiling(context, canvasWidth, canvasHeight, floorCamera);

    const stripeWidth = canvasWidth / columnCount;
    const wallFocalLength =
      canvasHeight / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);

    for (let column = 0; column < columnCount; column += 1) {
      const hit = castColumnRay(
        column,
        columnCount,
        cameraPositionX,
        cameraPositionY,
        cameraDirectionX,
        cameraDirectionY,
        cameraPlaneX,
        cameraPlaneY,
      );

      const perpWallDist = hit.perpWallDist;
      const isCapped =
        !Number.isFinite(perpWallDist) ||
        perpWallDist >= NEATENSTEIN_RENDER_DISTANCE_CAP;

      zBuffer[column] =
        Number.isFinite(perpWallDist) && perpWallDist > 0
          ? perpWallDist
          : NEATENSTEIN_RENDER_DISTANCE_CAP;

      // Integer stripe bounds prevent subpixel gaps when the column count does
      // not divide the canvas width evenly.
      const xStart = Math.floor(column * stripeWidth);
      const xEnd = Math.floor((column + 1) * stripeWidth);
      const stripePixelWidth = Math.max(0, xEnd - xStart);

      if (!isCapped) {
        const lineHeight = wallFocalLength / hit.perpWallDist;
        const drawStart = clamp(
          (canvasHeight - lineHeight) / 2,
          0,
          canvasHeight,
        );
        const drawEnd = clamp((canvasHeight + lineHeight) / 2, 0, canvasHeight);

        const wallColor =
          hit.side === 0
            ? NEATENSTEIN_WALL_X_SIDE_RGB
            : NEATENSTEIN_WALL_Y_SIDE_RGB;

        context.fillStyle = applyWallFog(wallColor, hit.perpWallDist);
        context.fillRect(
          xStart,
          drawStart,
          stripePixelWidth,
          drawEnd - drawStart,
        );
      } else {
        // Render a fog wall at the render distance cap with the same
        // proportional height as a real wall at 30 cells
        // (wallFocalLength / NEATENSTEIN_RENDER_DISTANCE_CAP). A short
        // vertical gradient feather at the top and bottom edges blends the
        // fog wall smoothly with the floor/ceiling grid, avoiding a hard
        // cut line where the fog wall meets the floor or ceiling.
        const fogDist = NEATENSTEIN_RENDER_DISTANCE_CAP;
        const lineHeight = wallFocalLength / fogDist;
        const drawStart = clamp(
          (canvasHeight - lineHeight) / 2,
          0,
          canvasHeight,
        );
        const drawEnd = clamp((canvasHeight + lineHeight) / 2, 0, canvasHeight);

        const fogColor = formatRgb(NEATENSTEIN_BACKGROUND_RGB);
        context.fillStyle = fogColor;
        context.fillRect(
          xStart,
          drawStart,
          stripePixelWidth,
          drawEnd - drawStart,
        );

        // Feather the top edge: gradient from transparent (ceiling side)
        // to opaque background (fog wall side) so the fog wall blends
        // smoothly with the ceiling grid above it.
        const fogFeatherPixels = 6;
        const featherTopStart = Math.max(0, drawStart - fogFeatherPixels);
        /* istanbul ignore else -- lineHeight = wallFocalLength/30 is always << canvasHeight, so drawStart is always > 0 */
        if (drawStart > 0) {
          const topGrad = context.createLinearGradient(
            0,
            featherTopStart,
            0,
            drawStart,
          );
          const { r, g, b } = NEATENSTEIN_BACKGROUND_RGB;
          topGrad.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0)`);
          topGrad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 1)`);
          context.fillStyle = topGrad;
          context.fillRect(
            xStart,
            featherTopStart,
            stripePixelWidth,
            drawStart - featherTopStart,
          );
        }

        // Feather the bottom edge: gradient from opaque background (fog
        // wall side) to transparent (floor side) so the fog wall blends
        // smoothly with the floor grid below it.
        const featherBottomEnd = Math.min(
          canvasHeight,
          drawEnd + fogFeatherPixels,
        );
        /* istanbul ignore else -- lineHeight = wallFocalLength/30 is always << canvasHeight, so drawEnd is always < canvasHeight */
        if (drawEnd < canvasHeight) {
          const bottomGrad = context.createLinearGradient(
            0,
            drawEnd,
            0,
            featherBottomEnd,
          );
          const { r, g, b } = NEATENSTEIN_BACKGROUND_RGB;
          bottomGrad.addColorStop(0, `rgba(${r}, ${g}, ${b}, 1)`);
          bottomGrad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
          context.fillStyle = bottomGrad;
          context.fillRect(
            xStart,
            drawEnd,
            stripePixelWidth,
            featherBottomEnd - drawEnd,
          );
        }
      }
    }

    // Render encoded enemy sprites into a single canvas snapshot and flush it
    // once. The no-op render context prevents per-sprite putImageData calls.
    if (
      activeEnemySprites.length > 0 &&
      typeof context.getImageData === 'function'
    ) {
      const spriteSnapshot = context.getImageData(
        0,
        0,
        canvasWidth,
        canvasHeight,
      );
      const spriteContext = buildNoOpSpriteRenderContext();
      const spriteCamera = {
        posX: spriteCameraX,
        posY: spriteCameraY,
        dirX: spriteDirectionX,
        dirY: spriteDirectionY,
        planeX: spritePlaneX,
        planeY: spritePlaneY,
      };

      const sortedSprites = [...activeEnemySprites].sort((a, b) => {
        const distA =
          (a.worldX - spriteCameraX) ** 2 + (a.worldY - spriteCameraY) ** 2;
        const distB =
          (b.worldX - spriteCameraX) ** 2 + (b.worldY - spriteCameraY) ** 2;
        return distB - distA;
      });

      for (const sprite of sortedSprites) {
        const frame = resolveNeatensteinEnemyFrame(sprite, spriteCamera);
        if (!frame) {
          continue;
        }
        const projection = clipNeatensteinSprite(
          sprite,
          spriteCamera,
          canvasWidth,
          canvasHeight,
          zBuffer,
        );
        // Skip sprites culled by the render-distance cap (AC-10.3c-002).
        if (!projection.visible) {
          continue;
        }

        // Build the de-rez state for sprites in the death animation so the
        // column renderer can dissolve pixels and tint survivors.
        const derezState =
          sprite.animationState === 'death' &&
          sprite.deRezElapsedMs !== undefined &&
          sprite.deRezDurationMs !== undefined &&
          sprite.seed !== undefined
            ? {
                elapsedMs: sprite.deRezElapsedMs,
                durationMs: sprite.deRezDurationMs,
                seed: sprite.seed,
              }
            : undefined;

        renderNeatensteinSprite(
          spriteSnapshot.data,
          zBuffer,
          projection,
          frame,
          spriteContext,
          sprite.teamColor,
          derezState,
        );
      }

      context.putImageData(spriteSnapshot, 0, 0);
    }

    // Update and emit ambient pulses after the wall z-buffer exists.
    activePulses = updateNeatensteinPulses(activePulses, latestState.simTick);

    const newFloorPulse = emitNeatensteinAmbientPulse(
      latestState.simTick,
      gameState.seed,
      NEATENSTEIN_PULSE_LAYER_FLOOR,
    );

    /* istanbul ignore next -- ambient pulse emission is pre-existing and not triggered by Step 04 test fixtures */
    if (
      newFloorPulse &&
      activePulses.length < NEATENSTEIN_PULSE_MAX_CONCURRENT
    ) {
      activePulses.push(newFloorPulse);
    }

    const newCeilingPulse = emitNeatensteinAmbientPulse(
      latestState.simTick,
      gameState.seed,
      NEATENSTEIN_PULSE_LAYER_CEILING,
    );

    /* istanbul ignore next -- ambient pulse emission is pre-existing and not triggered by Step 04 test fixtures */
    if (
      newCeilingPulse &&
      activePulses.length < NEATENSTEIN_PULSE_MAX_CONCURRENT
    ) {
      activePulses.push(newCeilingPulse);
    }

    drawNeatensteinPulses(
      context,
      activePulses,
      zBuffer,
      { x: cameraPositionX, y: cameraPositionY, yaw: cameraYaw },
      canvasWidth,
      canvasHeight,
    );

    drawImpactSpots(
      context,
      gameState.impacts,
      zBuffer,
      { x: cameraPositionX, y: cameraPositionY, yaw: cameraYaw },
      canvasWidth,
      canvasHeight,
      gameState.simTimeMs,
    );

    drawEnemyImpactSpots(
      context,
      /* istanbul ignore next -- nullish fallback only reachable in worker-tier rendering mode with mock OffscreenCanvas */
      gameState.enemyImpacts ?? [],
      zBuffer,
      { x: cameraPositionX, y: cameraPositionY, yaw: cameraYaw },
      canvasWidth,
      canvasHeight,
      gameState.simTimeMs,
    );

    drawBolts(
      context,
      gameState.bolts!,
      { x: cameraPositionX, y: cameraPositionY, yaw: cameraYaw },
      canvasWidth,
      canvasHeight,
      gameState.simTimeMs,
    );

    drawEnemyBolts(
      context,
      gameState.enemyBolts!,
      { x: cameraPositionX, y: cameraPositionY, yaw: cameraYaw },
      canvasWidth,
      canvasHeight,
      gameState.simTimeMs,
    );

    drawAmmoPickups(
      context,
      gameState.ammoPickups ?? [],
      zBuffer,
      { x: cameraPositionX, y: cameraPositionY, yaw: cameraYaw },
      canvasWidth,
      canvasHeight,
      gameState.simTimeMs,
    );

    renderGunOverlay(context, gameState.gun!, canvasWidth, canvasHeight);

    const commitableContext = context as OffscreenCanvasRenderingContext2D & {
      commit?: () => void;
    };
    if (typeof commitableContext.commit === 'function') {
      commitableContext.commit();
    }

    // Post a frame acknowledgment so the host bridge can apply worker-busy
    // backpressure. The worker tier renders directly to the OffscreenCanvas,
    // so the frame payload only carries the request id and scalar HUD fields
    // for the status-bar overlay.
    self.postMessage({
      type: 'frame',
      frame: {
        requestId: latestState.simTick,
        playerHealth: gameState.player.health,
        playerMaxHealth: gameState.player.maxHealth,
        playerAmmo: gameState.player.ammo,
        playerMaxAmmo: gameState.player.maxAmmo,
        playerKills: gameState.kills,
        playerDeaths: gameState.deaths ?? 0,
        spawnCount: gameState.spawnCount,
        generation: gameState.generation,
      },
    });

    return;
  } else {
    // CPU/GPU tiers ship packed frame data back to the host.
    const columnCount = resolvePackedColumnCount(currentTier);
    const renderState = {
      ...latestState,
      canvasWidth,
      canvasHeight,
    };

    const frame = buildNeatensteinRenderFrame(renderState, columnCount);

    for (let column = 0; column < columnCount; column += 1) {
      const hit = castColumnRay(
        column,
        columnCount,
        cameraPositionX,
        cameraPositionY,
        cameraDirectionX,
        cameraDirectionY,
        cameraPlaneX,
        cameraPlaneY,
      );

      frame.wallDistances[column] = hit.perpWallDist;
      frame.wallSides[column] = hit.side;
      frame.zBuffer[column] =
        !Number.isFinite(hit.perpWallDist) ||
        hit.perpWallDist >= NEATENSTEIN_RENDER_DISTANCE_CAP
          ? NEATENSTEIN_RENDER_DISTANCE_CAP
          : hit.perpWallDist;
    }

    frame.gun = gameState.gun;
    frame.bolts = gameState.bolts;
    frame.playerHealth = gameState.player.health;
    frame.playerMaxHealth = gameState.player.maxHealth;
    frame.playerAmmo = gameState.player.ammo;
    frame.playerMaxAmmo = gameState.player.maxAmmo;
    frame.playerKills = gameState.kills;
    frame.playerDeaths = gameState.deaths ?? 0;
    frame.spawnCount = gameState.spawnCount;
    frame.ammoPickups = gameState.ammoPickups;

    self.postMessage(
      { type: 'frame', frame },
      resolveNeatensteinRenderFrameTransferList(frame),
    );
  }
}

/**
 * Draw active ambient floor and ceiling pulses that pass the z-buffer test.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param pulses - Active pulses.
 * @param zBuffer - Per-column depth buffer from the wall pass.
 * @param camera - Current camera state.
 * @param canvasWidth - Canvas width in pixels.
 * @param canvasHeight - Canvas height in pixels.
 */
/* istanbul ignore next -- pre-existing ambient-pulse renderer; untouched by Step 04 halo/gun-shadow/bolt changes */
function drawNeatensteinPulses(
  context: OffscreenCanvasRenderingContext2D,
  pulses: readonly NeatensteinPulse[],
  zBuffer: Float32Array,
  camera: NeatensteinFloorCamera,
  canvasWidth: number,
  canvasHeight: number,
): void {
  const safeX = Number.isFinite(camera.x) ? camera.x : 0;
  const safeY = Number.isFinite(camera.y) ? camera.y : 0;
  const safeYaw = Number.isFinite(camera.yaw) ? camera.yaw : 0;

  const horizonY = canvasHeight * NEATENSTEIN_FLOOR_HORIZON_RATIO;
  const halfWidth = canvasWidth / 2;
  const focalLength =
    canvasHeight / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
  const cosYaw = Math.cos(safeYaw);
  const sinYaw = Math.sin(safeYaw);

  for (const pulse of pulses) {
    if (!pulse.active) {
      continue;
    }

    const projected =
      pulse.layer === NEATENSTEIN_PULSE_LAYER_CEILING
        ? projectNeatensteinCeilingPoint(
            pulse.worldX,
            pulse.worldY,
            safeX,
            safeY,
            cosYaw,
            sinYaw,
            focalLength,
            halfWidth,
            horizonY,
            canvasHeight,
            NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
          )
        : projectNeatensteinFloorPoint(
            pulse.worldX,
            pulse.worldY,
            safeX,
            safeY,
            cosYaw,
            sinYaw,
            focalLength,
            halfWidth,
            horizonY,
            canvasHeight,
            NEATENSTEIN_FLOOR_CAMERA_HEIGHT_WORLD,
          );

    if (projected === null) {
      continue;
    }

    const depthTestPulseInput = {
      screenColumn: (projected.x / canvasWidth) * zBuffer.length,
      distance: projected.distance,
    };

    if (!depthTestPulse(depthTestPulseInput, zBuffer)) {
      continue;
    }

    const alpha = clamp(
      pulse.lifetimeTicks / NEATENSTEIN_PULSE_AMBIENT_LIFETIME_TICKS,
      NEATENSTEIN_PULSE_ALPHA_MIN,
      NEATENSTEIN_PULSE_ALPHA_MAX,
    );

    context.shadowColor = NEATENSTEIN_PULSE_GLOW_COLOR;
    context.shadowBlur = NEATENSTEIN_PULSE_GLOW_BLUR_RADIUS;
    context.fillStyle = NEATENSTEIN_PULSE_COLOR;
    context.globalAlpha = alpha;
    context.beginPath();
    context.arc(
      projected.x,
      projected.y,
      NEATENSTEIN_PULSE_SCREEN_DOT_RADIUS_PX,
      0,
      Math.PI * 2,
    );
    context.fill();
  }

  context.globalAlpha = 1;
  context.shadowBlur = 0;
}

/**
 * Merge a new input snapshot into the pending snapshot.
 *
 * Movement and look use the latest values. One-shot actions are latched so a
 * short fire/dash press cannot be overwritten before the next `simState` tick.
 *
 * @param previous - Existing pending input, if any.
 * @param next - Newly received input.
 * @returns Merged pending input.
 */
/* istanbul ignore next -- pre-existing input-merge helper; only the null-previous branch is exercised by this test suite */
function mergePendingTickInput(
  previous: GameTickInputSnapshot | null,
  next: GameTickInputSnapshot,
): GameTickInputSnapshot {
  if (previous === null) {
    return next;
  }

  return {
    move: next.move,
    lookDelta: next.lookDelta,
    fire: previous.fire || next.fire,
    dash: previous.dash || next.dash,
  };
}

/* istanbul ignore next -- input normalization is defensive pre-existing code; the test suite only exercises valid host input */
/**
 * Normalize a host input message into the snapshot shape expected by game tick.
 *
 * @param raw - Value attached to the input message by the host.
 * @returns Normalized tick input snapshot.
 */
function inputMessageToTickInput(raw: unknown): GameTickInputSnapshot {
  if (!raw || typeof raw !== 'object') {
    return {
      move: { x: 0, y: 0 },
      lookDelta: 0,
      fire: false,
      dash: false,
    };
  }

  const input = raw as Record<string, unknown>;

  const yawDelta =
    typeof input.yawDelta === 'number' && Number.isFinite(input.yawDelta)
      ? input.yawDelta
      : 0;

  let lookDelta = yawDelta;
  if (input.look && typeof input.look === 'object') {
    const lookRecord = input.look as Record<string, unknown>;
    const lookYaw = lookRecord.yawDelta;

    if (typeof lookYaw === 'number' && Number.isFinite(lookYaw)) {
      lookDelta = lookYaw;
    }
  }

  let moveX = 0;
  let moveY = 0;

  if (input.movement && typeof input.movement === 'object') {
    const movement = input.movement as Record<string, unknown>;

    if (movement.forward === true) moveY += 1;
    if (movement.backward === true) moveY -= 1;
    if (movement.left === true) moveX -= 1;
    if (movement.right === true) moveX += 1;
  } else if (input.move && typeof input.move === 'object') {
    const move = input.move as Record<string, unknown>;

    if (typeof move.x === 'number' && Number.isFinite(move.x)) {
      moveX = move.x;
    }

    if (typeof move.y === 'number' && Number.isFinite(move.y)) {
      moveY = move.y;
    }
  }

  return {
    move: { x: moveX, y: moveY },
    lookDelta,
    fire: input.fire === true,
    dash: input.dash === true,
  };
}

/**
 * Fire cooldown in ticks for the fallback auto-mode AI.
 *
 * At 16 ms per tick, 25 ticks ≈ 0.4 s between shots — enough to conserve the
 * 50-round starting ammo while still maintaining suppressive fire.
 *
 * @see buildFallbackAutoTickInput
 */
const NEATENSTEIN_FALLBACK_FIRE_INTERVAL = 25;

/**
 * Half-angle of the forward firing arc for the fallback auto-mode AI.
 *
 * The fallback AI only fires when an enemy's bearing is within ±30°
 * (π/6 rad) of the player's facing direction, so shots are more likely
 * to hit instead of wasting ammo into walls.
 *
 * @see buildFallbackAutoTickInput
 */
const NEATENSTEIN_FALLBACK_FIRE_ARC = Math.PI / 6;

/**
 * Outbound evaluation request payload sent to the eval worker.
 */
interface EvalRequestPayload {
  type: 'evaluate';
  seed: number;
  generation: number;
  enemySnapshot: MlpSnapshot;
  humanMode: boolean;
}

/**
 * Inbound evaluation result payload received from the eval worker.
 */
interface EvalCompletePayload {
  type: 'evalComplete';
  generation: number;
  championNetworkJSON: Record<string, unknown>;
}

/**
 * Resolve the eval worker URL from the display worker's own location.
 *
 * The display worker bundle is published as
 * `docs/assets/neatenstein.worker.js`; the eval worker bundle is published
 * alongside it as `docs/assets/neatenstein.eval-worker.js`. This helper
 * derives the eval worker URL by replacing the filename in the display
 * worker's `self.location.href`.
 *
 * @returns Absolute URL to the eval worker bundle, or `null` when the
 *   location cannot be resolved (e.g. in test environments).
 */
function resolveEvalWorkerUrl(): string | null {
  try {
    const href = self.location?.href;
    if (typeof href !== 'string' || !href) return null;
    return href.replace('neatenstein.worker.js', 'neatenstein.eval-worker.js');
  } catch {
    return null;
  }
}

/**
 * Get or create the dedicated eval worker.
 *
 * The worker is created lazily on the first call. In the browser, the URL
 * is derived from the display worker's location. In tests, a mock worker
 * is injected via {@link __testOnlySetEvalWorker}.
 *
 * @returns The eval worker instance, or `null` when no worker can be
 *   created (e.g. the `Worker` constructor is unavailable).
 */
function getOrCreateEvalWorker(): Worker | null {
  if (evalWorker) return evalWorker;

  const url = resolveEvalWorkerUrl();
  if (!url) return null;

  try {
    const workerCtor = (globalThis as { Worker?: typeof Worker }).Worker;
    if (!workerCtor) return null;
    evalWorker = new workerCtor(url);
    evalWorker.onmessage = handleEvalComplete;
  } catch {
    evalWorker = null;
  }

  return evalWorker;
}

/**
 * Handle the `evalComplete` message from the eval worker.
 *
 * Deserializes the champion network via `Network.fromJSON()`, applies the
 * advanced generation to `gameState`, stores the champion network, and
 * clears the launch guard.
 *
 * @param event - Message event from the eval worker.
 */
async function handleEvalComplete(event: MessageEvent): Promise<void> {
  const data = event.data as EvalCompletePayload | null;
  if (!data || typeof data !== 'object' || data.type !== 'evalComplete') {
    return;
  }

  // Lazy-load Network for deserialization (avoids pulling neataptic into
  // the static import chain, which triggers GPUDevice type errors in the
  // test environment).
  const { Network } = await import('neataptic');

  const championNetwork = Network.fromJSON(data.championNetworkJSON);

  // Apply the result (generation always matches pendingGeneration + 1).
  gameState = { ...gameState!, generation: data.generation };
  // Store the champion main-agent network for Phase 4's player controller.
  championMainNetwork = championNetwork;
  lastChampionInputCount = NEATENSTEIN_MAIN_NEAT_INPUTS;

  // Clear the launch guard.
  pendingGeneration = null;
}

/**
 * Delegate the arms-race generation evaluation to the eval worker.
 *
 * Posts an evaluate request to the eval worker via `postMessage`. The
 * evaluation runs entirely in the eval worker, off the display worker's
 * render loop — no blocking `await` on the main thread. When the eval
 * worker completes, it posts back an `evalComplete` message which is
 * handled by {@link handleEvalComplete}.
 *
 * @param seed - Game seed.
 * @param generation - Current generation (post-advanceWave).
 * @param enemySnapshot - Frozen enemy snapshot from advanceWave.
 * @param humanModeBool - Whether the game is in auto mode.
 *
 * @see AC-P2S1b-001, AC-P2S1b-002
 */
function delegateEvaluation(
  seed: number,
  generation: number,
  enemySnapshot: MlpSnapshot,
  humanModeBool: boolean,
): void {
  const worker = getOrCreateEvalWorker();
  if (!worker) return;

  const payload: EvalRequestPayload = {
    type: 'evaluate',
    seed,
    generation,
    enemySnapshot,
    humanMode: humanModeBool,
  };
  worker.postMessage(payload);
}

/**
 * Build a {@link GameTickInputSnapshot} from the champion NEAT network.
 *
 * Extracts real sensors from the current game state, activates the champion
 * network, and maps the output vector to a tick input via
 * {@link networkOutputToTickInput}.
 *
 * @param network - The champion main-agent network from the arms-race evaluation.
 * @param state - Current deterministic game state (non-null).
 * @param flatMap - Row-major wall map for raycast sensors.
 * @param mapSize - Width and height of the square grid.
 * @param collisionMap - Map queried for solid cells; used for ammo-pickup
 *   path-distance sensors (P2S1).
 * @returns A game-tick input snapshot derived from the network output.
 * @see AC-065, AC-066, AC-P2S1-001
 */
function buildAutoTickInput(
  network: Network,
  state: GameState,
  flatMap: Uint8Array,
  mapSize: number,
  collisionMap: CollisionMap,
): GameTickInputSnapshot {
  const sensors = extractSensors(state, flatMap, mapSize, collisionMap);
  const raw = network.activate(sensors);

  // P5S1: Apply soft fire gate with hysteresis on enemyVisible sensor.
  // When no enemy is visible (sensor below 0.15 floor), fire is suppressed
  // regardless of network output.  Hysteresis (0.15/0.18) prevents rapid
  // on/off oscillation at the vision boundary.
  // @see AC-P5S1a-001, AC-P5S1a-002, AC-P5S1a-003
  // istanbul ignore next -- defensive fallback; ENEMY_VISIBLE_SENSOR_INDEX
  // is always populated by the extractSensors vector above.
  const enemyVisible = sensors[ENEMY_VISIBLE_SENSOR_INDEX] ?? 0;

  const tick = networkOutputToTickInput(raw, {
    state: fireGateState,
    enemyVisible,
  });

  // P1S1: Apply per-tick accel/decel smoothing to move/look so the AI
  // does not snap from 0 to max in a single tick.
  smoothedMoveX = smoothCommand(smoothedMoveX, tick.move.x);
  smoothedMoveY = smoothCommand(smoothedMoveY, tick.move.y);
  smoothedLookDelta = smoothCommand(
    smoothedLookDelta,
    tick.lookDelta,
    NEATENSTEIN_MOVE_ACCEL_PER_TICK,
    NEATENSTEIN_MOVE_DECEL_PER_TICK,
  );

  return {
    move: { x: smoothedMoveX, y: smoothedMoveY },
    lookDelta: smoothedLookDelta,
    fire: tick.fire,
    dash: tick.dash,
  };
}

/**
 * Build a fallback auto-mode tick input when no champion network is available.
 *
 * **P6S2 — Chicken-and-egg deadlock resolution.**
 *
 * `championMainNetwork` starts as `null` and is only set after the first
 * wave-clear via the eval worker delegation.  But clearing the first
 * wave requires an active player, and the auto-mode player is paralyzed
 * (zero input) without a champion network — a circular dependency.
 *
 * This fallback breaks the deadlock by providing a deterministic hunter AI:
 *
 * - **Wall-bounce exploration** — when no enemy is visible, move forward in
 *   the current direction and only turn when a wall is detected within
 *   {@link NEATENSTEIN_EXPLORATION_WALL_BOUNCE_LOOKAHEAD_CELLS}.  The hunter
 *   casts angled rays at ±{@link NEATENSTEIN_EXPLORATION_BOUNCE_ANGLE_RAD},
 *   picks the most open direction, and turns toward it at the capped rate.
 * - **Distance-aware kiting** — when an enemy is visible, turn toward its
 *   bearing and choose movement based on the 15–20 cell band defined by
 *   {@link NEATENSTEIN_KITING_BACKPEDAL_DISTANCE_CELLS} and
 *   {@link NEATENSTEIN_KITING_APPROACH_DISTANCE_CELLS}.
 * - **Fire through the shared soft fire gate** — computes a raw fire intent
 *   when the visible enemy is within ±{@link NEATENSTEIN_FALLBACK_FIRE_ARC} and
 *   the cooldown interval has elapsed, then passes it through
 *   {@link applyFireGate} with the module-level {@link fireGateState} so the
 *   fallback respects the same hysteresis gate as the champion-network path.
 *
 * Once the first wave is cleared and the arms-race evaluation runs, the
 * evolved champion network replaces this fallback.
 *
 * @param state - Current deterministic game state (player + enemies).
 * @returns A game-tick input snapshot with exploration + kiting behavior.
 * @see AC-060a, AC-P5S1a-001, AC-P5S1a-002, AC-P5S1a-003, AC-P9S2-001
 */
function hasAliveEnemies(enemies: EnemyState[] | undefined | null): boolean {
  const list =
    enemies ??
    /* istanbul ignore next -- defensive fallback for malformed gameState.enemies */ [];
  return list.some((e) => e.health > 0 && e.active !== false);
}

function buildFallbackAutoTickInput(state: GameState): GameTickInputSnapshot {
  fallbackTickCounter++;

  let lookDelta = 0;
  let rawFire = 0;
  let moveY = 1;

  // Use the same vision query as the network path: nearest active enemy within
  // VISION_RANGE_CELLS and with a clear line of sight.
  const visibleEnemy =
    wallMap !== null
      ? findNearestVisibleEnemy(state, wallMap, NEATENSTEIN_MAP_SIZE)
      : null;
  const enemyVisible = visibleEnemy !== null ? 1 : 0;

  const px = state.player.position.x;
  const py = state.player.position.y;
  const pa = state.player.angleRad;

  if (visibleEnemy !== null) {
    const dx = visibleEnemy.position.x - px;
    const dy = visibleEnemy.position.y - py;
    const bearing = Math.atan2(dy, dx);
    const distanceCells = Math.hypot(dx, dy);

    // Normalize angle difference to [-π, π].
    let angleDiff = bearing - pa;
    while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
    while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;

    // Turn toward the enemy, capped by the fallback turn rate.
    lookDelta = Math.max(
      -NEATENSTEIN_FALLBACK_TURN_RATE,
      Math.min(NEATENSTEIN_FALLBACK_TURN_RATE, angleDiff),
    );

    // Kiting: backpedal if too close, hold in the 15–20 cell band, approach
    // if the enemy is beyond the preferred band.
    if (distanceCells < NEATENSTEIN_KITING_BACKPEDAL_DISTANCE_CELLS) {
      moveY = -1;
    } else if (distanceCells < NEATENSTEIN_KITING_APPROACH_DISTANCE_CELLS) {
      moveY = 0;
    } else {
      moveY = 1;
    }

    // Raw fire intent: only when the visible enemy is within the forward arc
    // AND the cooldown interval has elapsed.
    if (
      Math.abs(angleDiff) <= NEATENSTEIN_FALLBACK_FIRE_ARC &&
      fallbackTickCounter % NEATENSTEIN_FALLBACK_FIRE_INTERVAL === 0
    ) {
      rawFire = 1;
    }
  } else if (wallMap !== null) {
    // Wall-bounce exploration: move forward, but if a wall is imminent, turn
    // toward the more open of the two angled lookahead directions.
    const forwardHit = castRayDDAFromFlatMap(
      wallMap,
      NEATENSTEIN_MAP_SIZE,
      px,
      py,
      Math.cos(pa),
      Math.sin(pa),
    );

    const lookahead = NEATENSTEIN_EXPLORATION_WALL_BOUNCE_LOOKAHEAD_CELLS;
    if (
      Number.isFinite(forwardHit.perpWallDist) &&
      forwardHit.perpWallDist <= lookahead
    ) {
      const bounce = NEATENSTEIN_EXPLORATION_BOUNCE_ANGLE_RAD;
      const candidates = [
        {
          offset: bounce,
          hit: castRayDDAFromFlatMap(
            wallMap,
            NEATENSTEIN_MAP_SIZE,
            px,
            py,
            Math.cos(pa + bounce),
            Math.sin(pa + bounce),
          ),
        },
        {
          offset: -bounce,
          hit: castRayDDAFromFlatMap(
            wallMap,
            NEATENSTEIN_MAP_SIZE,
            px,
            py,
            Math.cos(pa - bounce),
            Math.sin(pa - bounce),
          ),
        },
      ];

      const best = candidates.reduce((chosen, current) =>
        current.hit.perpWallDist > chosen.hit.perpWallDist ? current : chosen,
      );

      let angleDiff = best.offset;
      /* istanbul ignore next -- defensive angle-normalisation loop; bounce
       * offsets are bounded by NEATENSTEIN_EXPLORATION_BOUNCE_ANGLE_RAD, so
       * this wrap is unreachable with current constants but kept for robustness
       * if offset sources change in the future. */
      while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
      /* istanbul ignore next -- defensive angle-normalisation loop (see above). */
      while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;

      lookDelta = Math.max(
        -NEATENSTEIN_FALLBACK_TURN_RATE,
        Math.min(NEATENSTEIN_FALLBACK_TURN_RATE, angleDiff),
      );
    }
  }

  // P1S1: Apply per-tick accel/decel smoothing to move/look so the AI
  // does not snap from 0 to max in a single tick.
  smoothedMoveX = smoothCommand(smoothedMoveX, 0);
  smoothedMoveY = smoothCommand(smoothedMoveY, moveY);
  smoothedLookDelta = smoothCommand(
    smoothedLookDelta,
    lookDelta,
    NEATENSTEIN_MOVE_ACCEL_PER_TICK,
    NEATENSTEIN_MOVE_DECEL_PER_TICK,
  );

  // Apply the same hysteresis fire gate used by buildAutoTickInput.
  const fire = applyFireGate(fireGateState, enemyVisible, rawFire);

  const result: GameTickInputSnapshot = {
    move: { x: smoothedMoveX, y: smoothedMoveY },
    lookDelta: smoothedLookDelta,
    fire,
    dash: false,
  };

  lastFallbackInputForTest = result;
  return result;
}

self.onmessage = (event: MessageEvent) => {
  const data = event.data;

  if (!data || typeof data !== 'object') {
    return;
  }

  if (data.type === 'init') {
    const tier =
      data.tier === 'worker' || data.tier === 'cpu' || data.tier === 'gpu'
        ? (data.tier as DisplayTier)
        : null;

    if (tier === null) {
      return;
    }

    currentTier = tier;
    latestState = null;
    activePulses = [];
    activeEnemySprites = [];
    pendingTickInput = null;
    workerZBuffer = null;
    workerContext = null;
    enemyControllerState = null;
    allEnemiesCleared = false;
    prevAllEnemiesCleared = false;
    enemyPopulation = null;
    pendingGeneration = null;
    championMainNetwork = null;
    lastChampionInputCount = null;
    evalWorker = null;
    lastTickInputSource = 'human';
    fallbackTickCounter = 0;
    lastFallbackInputForTest = null;
    fireGateState = createFireGateState();
    smoothedMoveX = 0;
    smoothedMoveY = 0;
    smoothedLookDelta = 0;

    if (data.canvas) {
      workerCanvas = data.canvas as OffscreenCanvas;
    } else {
      workerCanvas = null;
    }

    if (workerCanvas !== null && pendingResizeDimensions !== null) {
      syncWorkerCanvasSize(
        workerCanvas,
        pendingResizeDimensions.width,
        pendingResizeDimensions.height,
      );
      pendingResizeDimensions = null;
    }

    const seed =
      typeof data.mapSeed === 'number' && Number.isFinite(data.mapSeed)
        ? data.mapSeed
        : NEATENSTEIN_DEFAULT_SEED;

    // Build the deterministic map once, then share it between raycasting and
    // collision systems.
    wallMap = buildNeatensteinMap(seed);
    collisionMap = createCollisionMap(wallMap, NEATENSTEIN_MAP_SIZE);
    gameState = createGameState({ seed });
    enemyControllerState = createEnemyControllerState(gameState);
    enemyPopulation = createMlpEnemyPopulation({ seed: gameState.seed });

    const version = data.version ?? NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION;

    self.postMessage({
      type: 'initialized',
      tier,
      version,
    });

    return;
  }

  if (data.type === 'resize') {
    const width =
      typeof data.width === 'number' ? (data.width as number) : Number.NaN;
    const height =
      typeof data.height === 'number' ? (data.height as number) : Number.NaN;

    applyWorkerResize(width, height);
    return;
  }

  if (data.type === 'simState') {
    latestState = data.state as NeatensteinRenderState;

    if (!gameState || !wallMap || !collisionMap || !enemyControllerState) {
      buildAndPostFrame();
      return;
    }

    // P4S1: Branch on humanMode to select the tick input source.
    //   - 'auto'  → activate the champion NEAT network stored in worker scope
    //               from P3S2's hoisted arms-race evaluation.
    //   - 'human' (or undefined) → use the existing pendingTickInput path.
    // @see AC-055, AC-056, AC-057, AC-059, AC-060
    const humanMode = latestState?.humanMode;
    let tickInput: GameTickInputSnapshot;

    // P3S1c: Genome-extinction guard. If the champion was evolved with a
    // different input count than the current NEATENSTEIN_MAIN_NEAT_INPUTS,
    // the network topology is incompatible — clear it so the fallback AI
    // takes over until a new champion is evolved with the correct sensor
    // vector length.
    // @see AC-P3S1c-002
    if (
      championMainNetwork &&
      lastChampionInputCount !== null &&
      lastChampionInputCount !== NEATENSTEIN_MAIN_NEAT_INPUTS
    ) {
      championMainNetwork = null;
      lastChampionInputCount = null;
    }

    if (
      humanMode === 'auto' &&
      championMainNetwork &&
      hasAliveEnemies(gameState.enemies)
    ) {
      try {
        tickInput = buildAutoTickInput(
          championMainNetwork,
          gameState,
          wallMap,
          NEATENSTEIN_MAP_SIZE,
          collisionMap,
        );
        lastTickInputSource = 'auto';
      } catch {
        // If the network activation fails (e.g. corrupted network state),
        // use the fallback auto AI to keep the simulation alive.
        tickInput = buildFallbackAutoTickInput(gameState);
        lastTickInputSource = 'auto';
      }
    } else if (humanMode === 'auto') {
      // P6S2: No champion network yet — chicken-and-egg deadlock.
      // The champion is only set after the first wave-clear, but clearing
      // the first wave requires an active player.  Use the fallback AI
      // (move forward, steer toward nearest enemy, fire with cooldown) so
      // the auto-mode player can potentially clear the first wave and
      // trigger the first arms-race evaluation.
      // @see AC-060a
      tickInput = buildFallbackAutoTickInput(gameState);
      lastTickInputSource = 'auto';
    } else {
      tickInput = pendingTickInput ?? {
        move: { x: 0, y: 0 },
        lookDelta: 0,
        fire: false,
        dash: false,
      };
      lastTickInputSource = 'human';
    }

    // Fixed-timestep determinism: always use the authoritative
    // NEATENSTEIN_FIXED_TIMESTEP_MS regardless of the rAF delta-time posted by
    // the host. The rAF cadence only gates *when* a tick runs, not *how much*
    // simulation time advances. This is critical for replay determinism
    // (Determinism Contract §2 — Level 2 ordered deterministic).
    // @see AC-018
    const timestepMs = NEATENSTEIN_FIXED_TIMESTEP_MS;

    // Advance the enemy controller BEFORE the game tick so movement, bolt, and
    // contact damage subsystems see synced enemy positions/health/active states.
    // Inject champion MLP weights from the population snapshot so the MLP
    // re-ranking branch is active. The snapshot refresh is gated by generation.
    // @see AC-024, AC-025
    let enemyWeights = resolveEnemyWeights(enemyPopulation, gameState);
    const controlled = updateEnemyController(
      enemyControllerState,
      gameState,
      collisionMap,
      timestepMs,
      enemyWeights,
    );
    enemyControllerState = controlled;

    gameState = {
      ...gameState,
      enemies: gameState.enemies.map((enemy, index) => {
        const controlledEnemy = controlled.enemies[index];
        if (!controlledEnemy) {
          return enemy;
        }
        return {
          ...enemy,
          position: { ...controlledEnemy.position },
          controllerPosition: { ...controlledEnemy.position },
          health: controlledEnemy.health,
          active: controlledEnemy.active,
          stunTimerMs: controlledEnemy.stunTimerMs,
        };
      }),
    };

    // Capture whether any enemies are alive BEFORE the tick so wave-clear
    // detection only observes a true false → true transition when alive enemies
    // actually drop to zero (not when the roster already contained only dead/
    // inactive enemies for index alignment).
    const hasAliveEnemiesBeforeTick = hasAliveEnemies(gameState?.enemies);

    gameState = gameTick(gameState, tickInput, collisionMap, timestepMs);

    // Wave-clear detection (AC-017): track the false → true transition when all
    // enemies are dead. Dead enemies stay in the array for index alignment (per
    // waves.ts), so we check for *alive* enemies (health > 0 && active) rather
    // than array length. When alive enemies existed before the tick and no alive
    // enemies remain, set allEnemiesCleared so P3S1's advanceWave can consume
    // it to increment generation and spawn the next wave.
    // @see AC-017, AC-100
    const hasAliveEnemiesAfterTick = hasAliveEnemies(gameState?.enemies);
    if (hasAliveEnemiesBeforeTick && !hasAliveEnemiesAfterTick) {
      allEnemiesCleared = true;
    }
    if (hasAliveEnemiesAfterTick) {
      // Reset the flag when a new wave starts so the next clear can re-trigger.
      allEnemiesCleared = false;
    }

    // Consume hitscan events produced by the enemy controller and spawn
    // visible enemy bolts. Each bolt uses the HitscanEvent origin/direction
    // directly — no Math.random(), fully deterministic. Bolts are appended
    // after the tick so they start at the enemy position and are advanced on
    // the following tick, matching the player bolt spawn pattern.
    if (controlled.hitscanEvents.length > 0 && gameState) {
      const simTimeMs = gameState.simTimeMs;
      const newEnemyBolts = controlled.hitscanEvents.map((event) =>
        fireEnemyBolt(
          {
            origin: event.origin,
            direction: event.direction,
            damage: event.damage,
          },
          simTimeMs,
        ),
      );
      gameState = {
        ...gameState,
        enemyBolts: [...(gameState.enemyBolts ?? []), ...newEnemyBolts],
      };
    }

    // P3S1: On wave-clear transition (false → true), advance the wave using
    // the MLP enemy population. advanceWave clears the arena, increments
    // generation, evolves the population, and spawns the next wave. A fresh
    // controller state is created to match the new enemy roster, and the
    // evolved champion weights are injected for the next controller pass.
    // In human-played mode the population still exists (created on init), so
    // wave respawn works identically.
    // @see AC-037, AC-038
    if (
      !prevAllEnemiesCleared &&
      allEnemiesCleared &&
      enemyPopulation &&
      gameState
    ) {
      const advanceResult = advanceWave(gameState, {
        population: enemyPopulation,
        spawnCount: NEATENSTEIN_ENEMY_MAX_CONCURRENT,
      });
      gameState = advanceResult.state;
      // Refresh the controller state for the new wave's enemy roster.
      enemyControllerState = createEnemyControllerState(gameState);
      // Inject the evolved champion weights into the next controller pass.
      if (advanceResult.snapshot.kind === 'mlp') {
        enemyWeights = (advanceResult.snapshot as MlpSnapshot).weights;
      }

      // P2S1: Delegate the arms-race generation evaluation to the dedicated
      // eval worker via postMessage. The eval worker runs the NEAT population
      // evaluation asynchronously, off the display worker's render loop — no
      // blocking await on the main thread. Only one generation is in flight
      // at a time; concurrent wave-clears are dropped (not queued).
      // @see AC-P2S1b-001, AC-P2S1b-002, AC-039, AC-043a
      if (pendingGeneration === null && gameState) {
        pendingGeneration = gameState.generation;
        const armsRaceSeed = gameState.seed;
        const armsRaceGeneration = gameState.generation;
        const armsRaceSnapshot = advanceResult.snapshot;
        const humanModeBool = latestState?.humanMode === 'auto';

        // Delegate to eval worker: posts an evaluate request and returns
        // immediately. The evalComplete handler applies the result when the
        // eval worker finishes.
        if (armsRaceSnapshot.kind === 'mlp') {
          delegateEvaluation(
            armsRaceSeed,
            armsRaceGeneration,
            armsRaceSnapshot as MlpSnapshot,
            humanModeBool,
          );
        }
      }
    }
    prevAllEnemiesCleared = allEnemiesCleared;

    // Keep the worker authoritative. If an enemy was killed this tick, keep it
    // in the controller roster during the de-rez animation so
    // updateEnemyController advances deRezElapsedMs instead of re-creating the
    // enemy with a reset timer. Only prune once the de-rez duration elapses,
    // and respawn the slot in gameState so the subsequent controller pass does
    // not re-create a dead enemy with deRezElapsedMs reset to 0.
    const completedDeRezIndices: number[] = [];
    enemyControllerState = {
      ...enemyControllerState,
      enemies: enemyControllerState.enemies
        .map((controlled, index) => {
          if (!gameState) return { ...controlled, active: false, health: 0 };
          const live = gameState.enemies[index];
          if (!live) return { ...controlled, active: false, health: 0 };
          if ((live.health ?? 0) <= 0 || live.active === false) {
            // Enemy is dead or inactive. Only prune once the de-rez animation
            // has completed; otherwise keep the enemy so deRezElapsedMs
            // advances on the next controller pass.
            if (
              controlled.deRezElapsedMs >= ENEMY_CONTROLLER_DE_REZ_DURATION_MS
            ) {
              completedDeRezIndices.push(index);
              return {
                ...controlled,
                active: false,
                health: live?.health ?? 0,
              };
            }
            return { ...controlled, health: live?.health ?? 0 };
          }
          return controlled;
        })
        .filter((controlled) => controlled.active),
    };

    // Respawn policy fix: dead enemies that completed de-rez are removed from
    // gameState.enemies entirely so they do NOT respawn at the death location.
    // The wave spawner (spawnWaveTick) handles creating new enemies at map edges
    // on the next round, deferring respawn to a fresh spawn position.
    if (gameState && completedDeRezIndices.length > 0) {
      gameState = {
        ...gameState,
        enemies: gameState.enemies.filter(
          (_, index) => !completedDeRezIndices.includes(index),
        ),
      };
    }

    // Run a zero-timestep controller pass after the tick so newly spawned
    // enemies and any health changes from combat are reflected in the
    // controller state used for the next frame, without double-advancing enemy
    // movement.
    enemyControllerState = updateEnemyController(
      enemyControllerState,
      gameState,
      collisionMap,
      0,
      enemyWeights,
    );

    pendingTickInput = null;

    buildAndPostFrame();
    return;
  }

  if (data.type === NEATENSTEIN_INPUT_MESSAGE_TYPE) {
    const nextInput = inputMessageToTickInput(data.input);
    pendingTickInput = mergePendingTickInput(pendingTickInput, nextInput);
  }
};

/**
 * Test-only introspection hook: expose the current enemy controller state.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyGetEnemyControllerState =
  (): EnemyControllerState | null => enemyControllerState;

/**
 * Test-only introspection hook: expose the most recently rendered frame state.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyGetLatestState = (): NeatensteinRenderState | null =>
  latestState;

/**
 * Test-only hook: expose the worker canvas resize helper for direct testing.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlySyncWorkerCanvasSize = syncWorkerCanvasSize;

/**
 * Test-only hook: expose the enemy-team color resolver for direct testing.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyResolveEnemyTeamColor = resolveEnemyTeamColor;

/**
 * Test-only introspection hook: expose the pending generation guard value.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyGetPendingGeneration = (): number | null =>
  pendingGeneration;

/**
 * Test-only introspection hook: expose the champion main-agent network.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyGetChampionMainNetwork = (): Network | null =>
  championMainNetwork;

/**
 * Test-only hook: inject a mock eval worker for testing the delegation.
 *
 * When set, the display worker uses this worker instead of creating a real
 * one. The mock worker should have `postMessage` (jest.fn) and a settable
 * `onmessage` property so tests can simulate eval worker responses.
 *
 * @internal
 */
/* istanbul ignore next -- test-only hook to inject mock eval worker */
export const __testOnlySetEvalWorker = (
  worker: { postMessage: (msg: unknown) => void; onmessage: unknown } | null,
): void => {
  evalWorker = worker as Worker | null;
  if (evalWorker) {
    evalWorker.onmessage = handleEvalComplete;
  }
};

/**
 * Test-only hook: expose the eval worker for test introspection.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyGetEvalWorker = (): unknown => evalWorker;

/**
 * Test-only hook: inject a champion main-agent network for auto-mode testing.
 *
 * @internal
 */
/* istanbul ignore next -- test-only hook to inject champion network */
export const __testOnlySetChampionMainNetwork = (
  network: Network | null,
): void => {
  championMainNetwork = network;
  lastChampionInputCount =
    network !== null ? NEATENSTEIN_MAIN_NEAT_INPUTS : null;
};

/**
 * Test-only hook: override the champion input count for extinction testing.
 *
 * Simulates a champion evolved with a different input count (e.g. 12) so
 * the genome-extinction guard can be verified when the constant changes.
 *
 * @internal
 */
/* istanbul ignore next -- test-only hook for extinction testing */
export const __testOnlySetChampionInputCount = (count: number | null): void => {
  lastChampionInputCount = count;
};

/**
 * Test-only introspection hook: expose the champion input count.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyGetChampionInputCount = (): number | null =>
  lastChampionInputCount;

/**
 * Test-only introspection hook: expose the source of the most recent tick input.
 *
 * Returns `'auto'` when the champion NEAT network produced the last tick input,
 * or `'human'` when the human input queue (or default zero) was used.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyGetLastTickInputSource = (): 'auto' | 'human' =>
  lastTickInputSource;

/**
 * Test-only hook: get the last fallback AI input snapshot.
 *
 * @returns The raw {@link GameTickInputSnapshot} produced by the most recent
 *   call to {@link buildFallbackAutoTickInput}, or `null` before any fallback
 *   tick has run.
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyGetLastFallbackInput =
  (): GameTickInputSnapshot | null => lastFallbackInputForTest;

/**
 * Test-only hook: build a fallback tick input directly for a given state.
 *
 * This bypasses the simState handler so tests can inspect fallback behavior
 * with controlled wallMap/gameState combinations (e.g. wallMap === null).
 *
 * @internal
 */
/* istanbul ignore next -- test-only hook to drive buildFallbackAutoTickInput directly */
export const __testOnlyBuildFallbackAutoTickInput = (
  state: GameState,
): GameTickInputSnapshot => buildFallbackAutoTickInput(state);

/**
 * Test-only hook: reset the fire-gate hysteresis state.
 *
 * Allows tests to start from a known gate state (closed) without
 * re-initialising the entire worker.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyResetFireGateState = (): void => {
  fireGateState = createFireGateState();
};

/**
 * Test-only hook: get the current fire-gate hysteresis state.
 *
 * @returns A snapshot of the current `fireActive` flag.
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyGetFireGateState = (): FireGateState => fireGateState;

/**
 * Test-only hook: reset the P1S1 move/look smoothing state.
 *
 * Allows tests to start from a standstill without re-initialising the
 * entire worker.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyResetSmoothingState = (): void => {
  smoothedMoveX = 0;
  smoothedMoveY = 0;
  smoothedLookDelta = 0;
};

/**
 * Test-only hook: expose the wall fog-factor resolver for direct testing.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyResolveWallFogFactor = resolveWallFogFactor;

/**
 * Test-only hook: expose the wave-clear detection flag for direct testing.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyGetAllEnemiesCleared = (): boolean => allEnemiesCleared;

/**
 * Test-only hook: expose the current game state for direct testing.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyGetGameState = (): GameState | null => gameState;

/**
 * Test-only hook: place synthetic enemies near the player for deterministic
 * combat and rendering tests.
 *
 * @internal
 */
/* istanbul ignore next -- test-only hook to place enemies near the player */
export const __testOnlyInjectTestEnemies = (
  positions: { x: number; y: number }[],
): void => {
  if (!gameState || !enemyControllerState) return;
  gameState.enemies = positions.map((pos, i) => ({
    position: { ...pos },
    health: 100,
    index: i,
    active: true,
    controllerPosition: { ...pos },
    stunTimerMs: 0,
  }));
  enemyControllerState = {
    ...enemyControllerState,
    enemies: positions.map((pos, i) => ({
      index: i,
      position: { ...pos },
      health: 100,
      yawRad: 0,
      animationState: 'idle' as const,
      ammo: 10,
      fireCooldownMs: 0,
      deRezElapsedMs: 0,
      active: true,
      walkTick: 0,
      shootBlinkTicks: 0,
      flankStallTicks: 0,
      bfsStallTicks: 0,
      weights: undefined,
      variantId: 0,
      previousStepDistance: -1,
      stunTimerMs: 0,
    })),
  };
};

/**
 * Resolve the current champion MLP weights from the enemy population snapshot.
 *
 * Calls `update({ generation })` on the population to gate snapshot refresh,
 * then narrows the snapshot to `MlpSnapshot` and returns its `weights` field.
 * Returns `undefined` when the population is not initialized (e.g. before init
 * or in human-played mode without a population), preserving backward-
 * compatible BFS behavior.
 *
 * @param population - The MLP enemy population (or null).
 * @param state - Current game state (used for the generation counter).
 * @returns Champion weights, or `undefined` when no population is active.
 * @see AC-024
 */
function resolveEnemyWeights(
  population: MlpEnemyPopulation | null,
  state: GameState,
): Float32Array | undefined {
  if (!population) return undefined;
  const snapshot = population.update({ generation: state.generation });
  if (snapshot.kind === 'mlp') {
    return (snapshot as MlpSnapshot).weights;
  }
  return undefined;
}

/**
 * Test-only introspection hook: expose the enemy population for direct testing.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyGetEnemyPopulation = (): MlpEnemyPopulation | null =>
  enemyPopulation;
