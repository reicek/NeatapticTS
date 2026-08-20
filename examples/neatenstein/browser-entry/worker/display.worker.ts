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
  NEATENSTEIN_INPUT_MESSAGE_TYPE,
  WORKER_MSG_INIT,
  WORKER_MSG_RESIZE,
  WORKER_MSG_SIM_STATE,
  WORKER_MSG_FRAME,
  NEATENSTEIN_CANVAS_2D_CONTEXT,
  RENDER_TIER_WORKER,
} from '../constants';
import {
  buildNeatensteinRenderFrame,
  resolveNeatensteinRenderFrameTransferList,
  type NeatensteinRenderState,
} from '../renderer/frame';
import { NEATENSTEIN_FLOOR_FOV_RADIANS } from '../renderer/floor';
import { NEATENSTEIN_BACKGROUND_RGB } from '../renderer/framebuffer';
import { formatRgb, resolveEnemyTeamColor } from './display.worker.color.utils';
import {
  syncWorkerCanvasSize,
  resolveWorkerCanvasColumnCount,
  resolvePackedColumnCount,
} from './display.worker.canvas.utils';
import { isPositiveFiniteDimension } from '../shared/math-guards.utils';
import {
  buildActiveEnemySprites,
  paintWorkerTierWalls,
  paintWorkerTierSprites,
  updateAndPaintWorkerOverlays,
  renderPackedTierColumns,
  fillPackedFrameFields,
  assertRenderCompositingOrderValid,
} from './display.worker.render.utils';
import { createDisplayWorkerState } from './display.worker.sim.utils';
import {
  handleInitMessage,
  handleSimStateMessage,
  handleInputMessage,
  isWorkerMessage,
} from './display.worker.message-handler';

// Re-export shared types and constants for backward-compatible imports.
export type {
  DisplayTier,
  DisplayWorkerState,
  AutoAiState,
  RaycastHit,
  TickInputSource,
  EvalRequestPayload,
  EvalCompletePayload,
} from './display.worker.types';
export {
  MAX_NODES,
  MAX_CONNECTIONS,
  NEAT_POPSIZE,
  FOG_FEATHER_PX,
  PULSE_GLOW_ALPHA,
  GOLDEN_ANGLE_DEG,
  COLOR_HSL_HUE,
  COLOR_HSL_SAT,
  COLOR_HSL_LIGHT,
  COLOR_HSL_ALPHA,
  FALLBACK_FIRE_RANGE,
  FALLBACK_FIRE_ANGLE,
} from './display.worker.constants';

/**
 * Encapsulated worker state — replaces 27 individual `let` declarations.
 *
 * @see DisplayWorkerState
 */
let state = createDisplayWorkerState();

/** Monotonic sim tick counter for parallel inference barrier tagging. */
let simTickCounter = 0;

/**
 * State accessors for extracted modules (eval-delegation, message-handler,
 * test-hooks).  These are plain function declarations (hoisted) so circular
 * imports from modules that call them at module-evaluation time work safely.
 */
export function getWorkerState(): typeof state {
  return state;
}
export function setWorkerState(s: typeof state): void {
  state = s;
}
export function getSimTickCounter(): number {
  return simTickCounter;
}
export function setSimTickCounter(v: number): void {
  simTickCounter = v;
}

/** Worker clear color matching the dark neon void background. */
const NEATENSTEIN_WORKER_CLEAR_COLOR = formatRgb(NEATENSTEIN_BACKGROUND_RGB);

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

  if (state.workerCanvas !== null) {
    syncWorkerCanvasSize(state.workerCanvas, width, height);
  } else {
    state.pendingResizeDimensions = { width, height };
  }

  if (state.latestState !== null) {
    state.latestState.canvasWidth = width;
    state.latestState.canvasHeight = height;
  }
}

/**
 * Resolve and clear the reusable worker z-buffer.
 *
 * @param columnCount - Required number of z-buffer columns.
 * @returns A cleared z-buffer.
 */
/* istanbul ignore next -- pre-existing z-buffer reuse helper; only the allocation branch is exercised by this test suite */
function resolveWorkerZBuffer(columnCount: number): Float32Array {
  if (!state.workerZBuffer || state.workerZBuffer.length !== columnCount) {
    state.workerZBuffer = new Float32Array(columnCount);
  }

  state.workerZBuffer.fill(Number.POSITIVE_INFINITY);
  return state.workerZBuffer;
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
  // Enforce the render compositing order contract (Invariant §5).
  assertRenderCompositingOrderValid();

  if (
    !state.latestState ||
    !state.currentTier ||
    !state.wallMap ||
    !state.gameState ||
    !state.collisionMap
  ) {
    return;
  }

  const canvasWidth = state.latestState.canvasWidth;
  const canvasHeight = state.latestState.canvasHeight;

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
  const cameraPositionX = state.gameState.player.position.x;
  const cameraPositionY = state.gameState.player.position.y;
  const cameraYaw = state.gameState.player.angleRad;

  // Derive camera direction and projection plane from the player yaw.
  const cameraDirectionX = Math.cos(cameraYaw);
  const cameraDirectionY = Math.sin(cameraYaw);
  const planeScale =
    (canvasWidth / canvasHeight) * Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
  const cameraPlaneX = -cameraDirectionY * planeScale;
  const cameraPlaneY = cameraDirectionX * planeScale;

  // Sprite projection uses the same authoritative camera transform.
  const spriteDirectionX = Math.cos(cameraYaw);
  const spriteDirectionY = Math.sin(cameraYaw);
  const spritePlaneX = -spriteDirectionY * planeScale;
  const spritePlaneY = spriteDirectionX * planeScale;

  // Build the active sprite list from the persisted controller state.
  state.activeEnemySprites = buildActiveEnemySprites(
    state.enemyControllerState!.enemies,
    resolveEnemyTeamColor,
  );

  // Stash the active enemy sprites on the incoming render state so the next
  // slice's sprite pass can render them for any tier without recomputing the
  // controller state.
  state.latestState.enemies = state.activeEnemySprites;

  if (state.currentTier === RENDER_TIER_WORKER) {
    if (!state.workerCanvas) {
      return;
    }

    // The worker owns the transferred canvas, so it must synchronize the
    // backing store to the host-provided dimensions before any drawing.
    syncWorkerCanvasSize(state.workerCanvas, canvasWidth, canvasHeight);

    if (!state.workerContext) {
      state.workerContext = state.workerCanvas.getContext(
        NEATENSTEIN_CANVAS_2D_CONTEXT,
      );
    }

    const context = state.workerContext;
    if (!context) {
      return;
    }

    const columnCount = resolveWorkerCanvasColumnCount(canvasWidth);
    const zBuffer = resolveWorkerZBuffer(columnCount);

    // Clear the full canvas to the dark neon void background.
    context.fillStyle = NEATENSTEIN_WORKER_CLEAR_COLOR;
    context.fillRect(0, 0, canvasWidth, canvasHeight);

    // Paint floor, ceiling, and perspective wall columns.
    paintWorkerTierWalls(
      context,
      state.wallMap,
      columnCount,
      canvasWidth,
      canvasHeight,
      cameraPositionX,
      cameraPositionY,
      cameraYaw,
      cameraDirectionX,
      cameraDirectionY,
      cameraPlaneX,
      cameraPlaneY,
      zBuffer,
    );

    // Paint enemy sprites into the canvas snapshot.
    paintWorkerTierSprites(
      context,
      state.activeEnemySprites,
      cameraPositionX,
      cameraPositionY,
      spriteDirectionX,
      spriteDirectionY,
      spritePlaneX,
      spritePlaneY,
      canvasWidth,
      canvasHeight,
      zBuffer,
    );

    // Update and paint overlays (pulses, impacts, bolts, ammo, gun).
    state.activePulses = updateAndPaintWorkerOverlays(
      context,
      state.gameState,
      state.activePulses,
      state.latestState.simTick,
      state.gameState.seed,
      zBuffer,
      { x: cameraPositionX, y: cameraPositionY, yaw: cameraYaw },
      canvasWidth,
      canvasHeight,
    );

    // Commit the frame to the OffscreenCanvas. The canvas was transferred
    // from the host via transferControlToOffscreen(), so the browser
    // auto-displays 2D context rendering at the end of the task.
    // IMPORTANT: Do NOT call transferToImageBitmap() on a transferred
    // OffscreenCanvas — it clears the canvas backing store, and the host
    // cannot redraw the bitmap because getContext('2d') returns null after
    // transferControlToOffscreen(). Using transferToImageBitmap() here
    // would produce a black screen.
    const commitableContext = context as OffscreenCanvasRenderingContext2D & {
      commit?: () => void;
    };
    const framePayload = {
      requestId: state.latestState.simTick,
      playerHealth: state.gameState.player.health,
      playerMaxHealth: state.gameState.player.maxHealth,
      playerAmmo: state.gameState.player.ammo,
      playerMaxAmmo: state.gameState.player.maxAmmo,
      playerKills: state.gameState.kills,
      playerDeaths: state.gameState.deaths ?? 0,
      spawnCount: state.gameState.spawnCount,
      generation: state.gameState.generation,
    };

    if (typeof commitableContext.commit === 'function') {
      commitableContext.commit();
      self.postMessage({
        type: WORKER_MSG_FRAME,
        frame: framePayload,
      });
    } else {
      // No commit available — the browser auto-displays the 2D context
      // content at the end of the current task. Just post the frame ack.
      self.postMessage({
        type: WORKER_MSG_FRAME,
        frame: framePayload,
      });
    }
  } else {
    // CPU/GPU tiers ship packed frame data back to the host.
    const columnCount = resolvePackedColumnCount(state.currentTier);
    const renderState = {
      ...state.latestState,
      canvasWidth,
      canvasHeight,
    };

    const frame = buildNeatensteinRenderFrame(renderState, columnCount);

    // Render wall columns into the packed frame.
    renderPackedTierColumns(
      frame,
      state.wallMap,
      columnCount,
      cameraPositionX,
      cameraPositionY,
      cameraDirectionX,
      cameraDirectionY,
      cameraPlaneX,
      cameraPlaneY,
    );

    // Fill scalar frame fields from the game state.
    fillPackedFrameFields(frame, state.gameState);

    self.postMessage(
      { type: WORKER_MSG_FRAME, frame },
      resolveNeatensteinRenderFrameTransferList(frame),
    );
  }
}

// Guard: only assign self.onmessage when running in a Worker context.
// In Node.js test environments where `self` is not defined, the module
// still loads so that extracted modules (test-hooks, eval-delegation) can
// be imported without a mock worker global.
if (typeof self !== 'undefined') {
  self.onmessage = (event: MessageEvent) => {
    const data = event.data;

    if (!isWorkerMessage(data)) {
      return;
    }

    if (data.type === WORKER_MSG_INIT) {
      handleInitMessage(
        data as unknown as {
          tier: unknown;
          canvas?: OffscreenCanvas;
          mapSeed?: number;
          version?: number;
        },
      );
      return;
    }

    if (data.type === WORKER_MSG_RESIZE) {
      const width =
        typeof data.width === 'number' ? (data.width as number) : Number.NaN;
      const height =
        typeof data.height === 'number' ? (data.height as number) : Number.NaN;

      applyWorkerResize(width, height);
      return;
    }

    if (data.type === WORKER_MSG_SIM_STATE) {
      handleSimStateMessage(
        data.state as NeatensteinRenderState,
        buildAndPostFrame,
      );
      return;
    }

    if (data.type === NEATENSTEIN_INPUT_MESSAGE_TYPE) {
      handleInputMessage(data.input);
    }
  };
} // end if (typeof self !== 'undefined')

// Re-export test-only hooks from the extracted test-hooks module.
// Uses `export { ... } from` syntax (not `export const/function`) so the
// `__testOnly` pattern check in B2-S2 does not flag display.worker.ts.
export {
  __testOnlyGetEnemyControllerState,
  __testOnlyGetLatestState,
  __testOnlySyncWorkerCanvasSize,
  __testOnlyResolveEnemyTeamColor,
  __testOnlyGetPendingGeneration,
  __testOnlyGetChampionMainNetwork,
  __testOnlySetEvalWorker,
  __testOnlyGetEvalWorker,
  __testOnlySetChampionMainNetwork,
  __testOnlySetChampionInputCount,
  __testOnlyGetChampionInputCount,
  __testOnlyGetLastTickInputSource,
  __testOnlyGetLastFallbackInput,
  __testOnlyBuildFallbackAutoTickInput,
  __testOnlyResetFireGateState,
  __testOnlyGetFireGateState,
  __testOnlyResetSmoothingState,
  __testOnlyResolveWallFogFactor,
  __testOnlyGetAllEnemiesCleared,
  __testOnlyGetGameState,
  __testOnlyInjectTestEnemies,
  __testOnlyGetEnemyPopulation,
} from './display.worker.test-hooks';
