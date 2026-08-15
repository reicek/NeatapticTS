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
  NEATENSTEIN_DEFAULT_SEED,
  NEATENSTEIN_INPUT_MESSAGE_TYPE,
  NEATENSTEIN_MAP_SIZE,
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
  WORKER_MSG_INIT,
  WORKER_MSG_RESIZE,
  WORKER_MSG_SIM_STATE,
  WORKER_MSG_INITIALIZED,
  WORKER_MSG_FRAME,
  EVAL_MSG_EVALUATE,
  EVAL_MSG_EVAL_COMPLETE,
  NEATENSTEIN_CANVAS_2D_CONTEXT,
  RENDER_TIER_WORKER,
  RENDER_TIER_CPU,
  RENDER_TIER_GPU,
} from '../constants';
import {
  buildNeatensteinRenderFrame,
  resolveNeatensteinRenderFrameTransferList,
  type NeatensteinRenderState,
} from '../renderer/frame';
import { NEATENSTEIN_FLOOR_FOV_RADIANS } from '../renderer/floor';
import { buildNeatensteinMap, createCollisionMap } from '../renderer/map';
import { createGameState, type GameTickInputSnapshot } from '../host/game/tick';
import { NEATENSTEIN_BACKGROUND_RGB } from '../renderer/framebuffer';
import {
  createEnemyControllerState,
  type EnemyControllerState,
} from '../../scripts/enemy-controller';
import { createMlpEnemyPopulation } from '../harness/enemy-mlp';
import type { MlpEnemyPopulation } from '../harness/enemy-mlp';
import type { MlpSnapshot } from '../harness/types';
import {
  NEATENSTEIN_MAIN_NEAT_INPUTS,
  createFireGateState,
  type FireGateState,
} from '../harness/neat-io-config';
import type { Network } from 'neataptic';
import {
  formatRgb,
  resolveEnemyTeamColor,
  resolveWallFogFactor,
} from './display.worker.color.utils';
import {
  isPositiveFiniteDimension,
  syncWorkerCanvasSize,
  resolveWorkerCanvasColumnCount,
  resolvePackedColumnCount,
} from './display.worker.canvas.utils';
import {
  mergePendingTickInput,
  inputMessageToTickInput,
} from './display.worker.input.utils';
import {
  buildActiveEnemySprites,
  paintWorkerTierWalls,
  paintWorkerTierSprites,
  updateAndPaintWorkerOverlays,
  renderPackedTierColumns,
  fillPackedFrameFields,
} from './display.worker.render.utils';
import {
  buildFallbackAutoTickInput,
} from './display.worker.auto-ai.utils';
import {
  createDisplayWorkerState,
  runSimStep,
} from './display.worker.sim.utils';
import type { GameState } from '../host/game/types';
import type {
  AutoAiState,
  DisplayTier,
  TickInputSource,
  EvalRequestPayload,
  EvalCompletePayload,
} from './display.worker.types';

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

    // Commit the frame to the OffscreenCanvas.
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
      type: WORKER_MSG_FRAME,
      frame: {
        requestId: state.latestState.simTick,
        playerHealth: state.gameState.player.health,
        playerMaxHealth: state.gameState.player.maxHealth,
        playerAmmo: state.gameState.player.ammo,
        playerMaxAmmo: state.gameState.player.maxAmmo,
        playerKills: state.gameState.kills,
        playerDeaths: state.gameState.deaths ?? 0,
        spawnCount: state.gameState.spawnCount,
        generation: state.gameState.generation,
      },
    });
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
  if (state.evalWorker) return state.evalWorker;

  const url = resolveEvalWorkerUrl();
  if (!url) return null;

  try {
    const workerCtor = (globalThis as { Worker?: typeof Worker }).Worker;
    if (!workerCtor) return null;
    state.evalWorker = new workerCtor(url);
    state.evalWorker.onmessage = handleEvalComplete;
  } catch {
    state.evalWorker = null;
  }

  return state.evalWorker;
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
  if (
    !data ||
    typeof data !== 'object' ||
    data.type !== EVAL_MSG_EVAL_COMPLETE
  ) {
    return;
  }

  // Lazy-load Network for deserialization (avoids pulling neataptic into
  // the static import chain, which triggers GPUDevice type errors in the
  // test environment).
  const { Network } = await import('neataptic');

  const championNetwork = Network.fromJSON(data.championNetworkJSON);

  // Apply the result (generation always matches pendingGeneration + 1).
  state.gameState = { ...state.gameState!, generation: data.generation };
  // Store the champion main-agent network for Phase 4's player controller.
  state.championMainNetwork = championNetwork;
  state.lastChampionInputCount = NEATENSTEIN_MAIN_NEAT_INPUTS;

  // Clear the launch guard.
  state.pendingGeneration = null;
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
    type: EVAL_MSG_EVALUATE,
    seed,
    generation,
    enemySnapshot,
    humanMode: humanModeBool,
  };
  worker.postMessage(payload);
}

self.onmessage = (event: MessageEvent) => {
  const data = event.data;

  if (!data || typeof data !== 'object') {
    return;
  }

  if (data.type === WORKER_MSG_INIT) {
    const tier =
      data.tier === RENDER_TIER_WORKER ||
      data.tier === RENDER_TIER_CPU ||
      data.tier === RENDER_TIER_GPU
        ? (data.tier as DisplayTier)
        : null;

    if (tier === null) {
      return;
    }

    // Reset all worker state to defaults, then set init-specific fields.
    // Preserve pendingResizeDimensions from before init so a resize received
    // before init is applied once the canvas is available.
    const preservedResize = state.pendingResizeDimensions;
    state = createDisplayWorkerState();
    state.pendingResizeDimensions = preservedResize;
    state.currentTier = tier;

    if (data.canvas) {
      state.workerCanvas = data.canvas as OffscreenCanvas;
    }

    if (state.workerCanvas !== null && state.pendingResizeDimensions !== null) {
      syncWorkerCanvasSize(
        state.workerCanvas,
        state.pendingResizeDimensions.width,
        state.pendingResizeDimensions.height,
      );
      state.pendingResizeDimensions = null;
    }

    const seed =
      typeof data.mapSeed === 'number' && Number.isFinite(data.mapSeed)
        ? data.mapSeed
        : NEATENSTEIN_DEFAULT_SEED;

    // Build the deterministic map once, then share it between raycasting and
    // collision systems.
    state.wallMap = buildNeatensteinMap(seed);
    state.collisionMap = createCollisionMap(
      state.wallMap,
      NEATENSTEIN_MAP_SIZE,
    );
    state.gameState = createGameState({ seed });
    state.enemyControllerState = createEnemyControllerState(state.gameState);
    state.enemyPopulation = createMlpEnemyPopulation({
      seed: state.gameState.seed,
    });

    const version = data.version ?? NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION;

    self.postMessage({
      type: WORKER_MSG_INITIALIZED,
      tier,
      version,
    });

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
    state.latestState = data.state as NeatensteinRenderState;

    // Run the deterministic simulation step (enemy controller, game tick,
    // wave-clear detection, bolt spawning, wave advance, de-rez pruning).
    // runSimStep returns null when the worker is not yet initialised; in that
    // case we still render whatever state is available.
    state = runSimStep(state, delegateEvaluation) ?? state;

    buildAndPostFrame();
    return;
  }

  if (data.type === NEATENSTEIN_INPUT_MESSAGE_TYPE) {
    const nextInput = inputMessageToTickInput(data.input);
    state.pendingTickInput = mergePendingTickInput(
      state.pendingTickInput,
      nextInput,
    );
  }
};

/**
 * Test-only introspection hook: expose the current enemy controller state.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the current enemy controller state. */
export const __testOnlyGetEnemyControllerState =
  (): EnemyControllerState | null => state.enemyControllerState;

/**
 * Test-only introspection hook: expose the most recently rendered frame state.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the most recently rendered frame state. */
export const __testOnlyGetLatestState = (): NeatensteinRenderState | null =>
  state.latestState;

/**
 * Test-only hook: expose the worker canvas resize helper for direct testing.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the worker canvas resize helper. */
export const __testOnlySyncWorkerCanvasSize = syncWorkerCanvasSize;

/**
 * Test-only hook: expose the enemy-team color resolver for direct testing.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the enemy-team color resolver. */
export const __testOnlyResolveEnemyTeamColor = resolveEnemyTeamColor;

/**
 * Test-only introspection hook: expose the pending generation guard value.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the pending generation guard value. */
export const __testOnlyGetPendingGeneration = (): number | null =>
  state.pendingGeneration;

/**
 * Test-only introspection hook: expose the champion main-agent network.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the champion main-agent network. */
export const __testOnlyGetChampionMainNetwork = (): Network | null =>
  state.championMainNetwork;

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
/** Test-only hook to inject a mock eval worker for testing delegation. */
export const __testOnlySetEvalWorker = (
  worker: { postMessage: (msg: unknown) => void; onmessage: unknown } | null,
): void => {
  state.evalWorker = worker as Worker | null;
  if (state.evalWorker) {
    state.evalWorker.onmessage = handleEvalComplete;
  }
};

/**
 * Test-only hook: expose the eval worker for test introspection.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the eval worker instance. */
export const __testOnlyGetEvalWorker = (): unknown => state.evalWorker;

/**
 * Test-only hook: inject a champion main-agent network for auto-mode testing.
 *
 * @internal
 */
/* istanbul ignore next -- test-only hook to inject champion network */
/** Test-only hook to inject a champion main-agent network for auto-mode testing. */
export const __testOnlySetChampionMainNetwork = (
  network: Network | null,
): void => {
  state.championMainNetwork = network;
  state.lastChampionInputCount =
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
/** Test-only hook to override the champion input count for extinction testing. */
export const __testOnlySetChampionInputCount = (count: number | null): void => {
  state.lastChampionInputCount = count;
};

/**
 * Test-only introspection hook: expose the champion input count.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the champion input count. */
export const __testOnlyGetChampionInputCount = (): number | null =>
  state.lastChampionInputCount;

/**
 * Test-only introspection hook: expose the source of the most recent tick input.
 *
 * Returns `'auto'` when the champion NEAT network produced the last tick input,
 * or `'human'` when the human input queue (or default zero) was used.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the source of the most recent tick input. */
export const __testOnlyGetLastTickInputSource = (): TickInputSource =>
  state.lastTickInputSource;

/**
 * Test-only hook: get the last fallback AI input snapshot.
 *
 * @returns The raw {@link GameTickInputSnapshot} produced by the most recent
 *   call to {@link buildFallbackAutoTickInput}, or `null` before any fallback
 *   tick has run.
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the last fallback AI input snapshot. */
export const __testOnlyGetLastFallbackInput =
  (): GameTickInputSnapshot | null => state.lastFallbackInputForTest;

/**
 * Test-only hook: build a fallback tick input directly for a given state.
 *
 * This bypasses the simState handler so tests can inspect fallback behavior
 * with controlled wallMap/gameState combinations (e.g. wallMap === null).
 *
 * @internal
 */
/* istanbul ignore next -- test-only hook to drive buildFallbackAutoTickInput directly */
/** Test-only hook to build a fallback tick input directly for a given state. */
export const __testOnlyBuildFallbackAutoTickInput = (
  gameState: GameState,
): GameTickInputSnapshot => {
  const ai: AutoAiState = {
    fallbackTickCounter: state.fallbackTickCounter,
    fireGateState: state.fireGateState,
    smoothedMoveX: state.smoothedMoveX,
    smoothedMoveY: state.smoothedMoveY,
    smoothedLookDelta: state.smoothedLookDelta,
    lastFallbackInputForTest: state.lastFallbackInputForTest,
  };
  const result = buildFallbackAutoTickInput(gameState, state.wallMap, ai);
  state.fallbackTickCounter = result.ai.fallbackTickCounter;
  state.smoothedMoveX = result.ai.smoothedMoveX;
  state.smoothedMoveY = result.ai.smoothedMoveY;
  state.smoothedLookDelta = result.ai.smoothedLookDelta;
  state.lastFallbackInputForTest = result.ai.lastFallbackInputForTest;
  return result.tickInput;
};

/**
 * Test-only hook: reset the fire-gate hysteresis state.
 *
 * Allows tests to start from a known gate state (closed) without
 * re-initialising the entire worker.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only hook to reset the fire-gate hysteresis state. */
export const __testOnlyResetFireGateState = (): void => {
  state.fireGateState = createFireGateState();
};

/**
 * Test-only hook: get the current fire-gate hysteresis state.
 *
 * @returns A snapshot of the current `fireActive` flag.
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the current fire-gate hysteresis state. */
export const __testOnlyGetFireGateState = (): FireGateState =>
  state.fireGateState;

/**
 * Test-only hook: reset the P1S1 move/look smoothing state.
 *
 * Allows tests to start from a standstill without re-initialising the
 * entire worker.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only hook to reset the P1S1 move/look smoothing state. */
export const __testOnlyResetSmoothingState = (): void => {
  state.smoothedMoveX = 0;
  state.smoothedMoveY = 0;
  state.smoothedLookDelta = 0;
};

/**
 * Test-only hook: expose the wall fog-factor resolver for direct testing.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the wall fog-factor resolver. */
export const __testOnlyResolveWallFogFactor = resolveWallFogFactor;

/**
 * Test-only hook: expose the wave-clear detection flag for direct testing.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the wave-clear detection flag. */
export const __testOnlyGetAllEnemiesCleared = (): boolean =>
  state.allEnemiesCleared;

/**
 * Test-only hook: expose the current game state for direct testing.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the current game state. */
export const __testOnlyGetGameState = (): GameState | null => state.gameState;

/**
 * Test-only hook: place synthetic enemies near the player for deterministic
 * combat and rendering tests.
 *
 * @internal
 */
/* istanbul ignore next -- test-only hook to place enemies near the player */
/** Test-only hook to place synthetic enemies near the player for deterministic testing. */
export const __testOnlyInjectTestEnemies = (
  positions: { x: number; y: number }[],
): void => {
  if (!state.gameState || !state.enemyControllerState) return;
  state.gameState.enemies = positions.map((pos, i) => ({
    position: { ...pos },
    health: 100,
    index: i,
    active: true,
    controllerPosition: { ...pos },
    stunTimerMs: 0,
  }));
  state.enemyControllerState = {
    ...state.enemyControllerState,
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
 * Test-only introspection hook: expose the enemy population for direct testing.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the enemy population instance. */
export const __testOnlyGetEnemyPopulation = (): MlpEnemyPopulation | null =>
  state.enemyPopulation;
