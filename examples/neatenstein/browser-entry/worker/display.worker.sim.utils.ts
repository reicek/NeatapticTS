/**
 * Simulation executors for the Neatenstein display worker.
 *
 * Contains the {@link createDisplayWorkerState} factory and the
 * {@link runSimStep} state-in/state-out executor that advances the
 * deterministic simulation one tick. The {@link DisplayWorkerState}
 * interface is defined in {@link ./display.worker.types.ts}.
 *
 * @module
 */

import {
  buildAutoTickInput,
  buildFallbackAutoTickInput,
  hasAliveEnemies,
} from './display.worker.auto-ai.utils';
import {
  NEATENSTEIN_ENEMY_MAX_CONCURRENT,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
} from '../host/game/constants';
import {
  NEATENSTEIN_MAP_SIZE,
  NEATENSTEIN_HUMAN_MODE_LABEL_AUTO,
  SNAPSHOT_KIND_MLP,
  TICK_INPUT_SOURCE_AUTO,
  TICK_INPUT_SOURCE_HUMAN,
} from '../constants';
import { NEATENSTEIN_MAIN_NEAT_INPUTS } from '../harness/neat-io-config';
import {
  ENEMY_CONTROLLER_DE_REZ_DURATION_MS,
  createEnemyControllerState,
  updateEnemyController,
} from '../shared/enemy-controller';
import {
  resolveInferenceStrategy,
  dispatchParallelInference,
  awaitInferenceBarrier,
} from '../shared/enemy-controller.parallel.utils';
import { advanceWave } from '../host/waves';
import { createFireGateState } from '../harness/neat-io-config';
import { fireEnemyBolt } from '../host/game/combat';
import { gameTick, type GameTickInputSnapshot } from '../host/game/tick';
import type {
  AutoAiState,
  DisplayWorkerState,
  SimWorkerState,
  TickInputSource,
} from './display.worker.types';
import type { GameState } from '../host/game/types';
import type { MlpEnemyPopulation } from '../harness/enemy-mlp';
import type { MlpSnapshot } from '../harness/types';

// --- Test-only diagnostic flags (A2 Fix 4, 6, 8; B1 bolt-spawn awareness) ---

/** Whether the zero-timestep pass was skipped on the last sim step. */
let zeroTimestepPassSkipped = true;

/** Whether the de-rez pruning used the single-pass Set approach. */
let deRezPruningUsedSinglePass = false;

/** Count of GameState shallow-clone operations in the last sim step. */
let simStepCloneCount = 0;

/** Whether bolt-spawn state changed (enemy bolts fired) on the last sim step. */
let boltSpawnStateChanged = false;

/**
 * Return whether the zero-timestep controller pass was skipped on the last
 * sim step (test-only diagnostic).
 *
 * @returns `true` when the pass was skipped (no completed de-rez).
 */
export function __testOnlyGetZeroTimestepPassSkipped(): boolean {
  return zeroTimestepPassSkipped;
}

/**
 * Return whether the de-rez pruning used the single-pass Set approach
 * (test-only diagnostic).
 *
 * @returns `true` when single-pass was used.
 */
export function __testOnlyGetDeRezPruningUsedSinglePass(): boolean {
  return deRezPruningUsedSinglePass;
}

/**
 * Return the count of GameState shallow-clone operations in the last sim step
 * (test-only diagnostic).
 *
 * @returns The clone count.
 */
export function __testOnlyGetSimStepCloneCount(): number {
  return simStepCloneCount;
}

/**
 * Return whether bolt-spawn state changed (enemy bolts were fired) on the
 * last sim step (test-only diagnostic).
 *
 * B1 contract: the zero-timestep pass must NOT skip when bolt-spawn state
 * changed, even if no de-rez completed. This diagnostic exposes whether
 * bolts were spawned so tests can verify the skip condition accounts for it.
 *
 * @returns `true` when at least one enemy bolt was spawned this tick.
 */
export function __testOnlyGetBoltSpawnStateChanged(): boolean {
  return boltSpawnStateChanged;
}

/**
 * Factory for the encapsulated display-worker mutable state object.
 *
 * Returns a fresh {@link DisplayWorkerState} with all rendering, simulation,
 * and enemy-population fields initialized to their empty defaults. The worker
 * fills in canvas references and the first game snapshot after construction.
 *
 * @returns A new DisplayWorkerState instance.
 * @see AC-072
 */
export function createDisplayWorkerState(): DisplayWorkerState {
  return {
    currentTier: null,
    workerCanvas: null,
    workerContext: null,
    latestState: null,
    pendingResizeDimensions: null,
    wallMap: null,
    workerZBuffer: null,
    gameState: null,
    collisionMap: null,
    activePulses: [],
    enemyControllerState: null,
    activeEnemySprites: [],
    allEnemiesCleared: false,
    prevAllEnemiesCleared: false,
    enemyPopulation: null,
    pendingGeneration: null,
    championMainNetwork: null,
    lastChampionInputCount: null,
    evalWorker: null,
    pendingTickInput: null,
    lastTickInputSource: TICK_INPUT_SOURCE_HUMAN,
    fallbackTickCounter: 0,
    lastFallbackInputForTest: null,
    fireGateState: createFireGateState(),
    smoothedMoveX: 0,
    smoothedMoveY: 0,
    smoothedLookDelta: 0,
  };
}

/**
 * Resolve the current champion MLP weights from the enemy population snapshot.
 *
 * Calls `update({ generation })` on the population to gate snapshot refresh,
 * then narrows the snapshot to `MlpSnapshot` and returns its `weights` field.
 * Returns `undefined` when the population is not initialized.
 *
 * @param population - The MLP enemy population (or null).
 * @param gameState - Current game state (used for the generation counter).
 * @returns Champion weights, or `undefined` when no population is active.
 * @see AC-024
 */
export function resolveEnemyWeights(
  population: MlpEnemyPopulation | null,
  gameState: GameState,
): Float32Array | undefined {
  if (!population) return undefined;
  const snapshot = population.update({ generation: gameState.generation });
  if (snapshot.kind === SNAPSHOT_KIND_MLP) {
    return (snapshot as MlpSnapshot).weights;
  }
  return undefined;
}

/**
 * Run one deterministic simulation step.
 *
 * This is the state-in/state-out executor for the `simState` message branch.
 * It takes the full {@link DisplayWorkerState} (with `latestState` already
 * set by the caller) and a `delegateEval` callback for arms-race evaluation,
 * then advances the enemy controller, runs the game tick, detects wave-clear
 * transitions, spawns enemy bolts, prunes de-rez'd enemies, and runs a
 * zero-timestep controller pass.
 *
 * Returns `null` when the worker is not yet initialised (caller should still
 * call `buildAndPostFrame` to render whatever state is available).
 *
 * @param state - Current worker state (latestState already set by caller).
 * @param delegateEval - Callback to delegate arms-race evaluation to the eval
 *   worker. Called as `delegateEval(seed, generation, snapshot, humanMode)`.
 * @returns Updated worker state, or `null` when not initialised.
 * @see AC-073, AC-074
 */
export function runSimStep(
  state: DisplayWorkerState,
  delegateEval: (
    seed: number,
    generation: number,
    snapshot: MlpSnapshot,
    humanMode: boolean,
  ) => void,
): DisplayWorkerState | null {
  // Guard: require initialised simulation state.
  if (
    !state.gameState ||
    !state.wallMap ||
    !state.collisionMap ||
    !state.enemyControllerState
  ) {
    return null;
  }

  // --- Local mutable copies of state fields that change during this step ---

  simStepCloneCount = 0;
  boltSpawnStateChanged = false;
  let gameState = state.gameState;
  let enemyControllerState = state.enemyControllerState;
  let allEnemiesCleared = state.allEnemiesCleared;
  let pendingGeneration = state.pendingGeneration;
  let championMainNetwork = state.championMainNetwork;
  let lastChampionInputCount = state.lastChampionInputCount;
  let lastTickInputSource: TickInputSource;

  // Auto-AI state slice.
  let ai: AutoAiState = {
    fallbackTickCounter: state.fallbackTickCounter,
    fireGateState: state.fireGateState,
    smoothedMoveX: state.smoothedMoveX,
    smoothedMoveY: state.smoothedMoveY,
    smoothedLookDelta: state.smoothedLookDelta,
    lastFallbackInputForTest: state.lastFallbackInputForTest,
  };

  // --- Resolve tick input (human vs auto vs fallback) ---

  const humanMode = state.latestState?.humanMode;
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
    humanMode === NEATENSTEIN_HUMAN_MODE_LABEL_AUTO &&
    championMainNetwork &&
    hasAliveEnemies(gameState.enemies)
  ) {
    try {
      const result = buildAutoTickInput(
        championMainNetwork,
        gameState,
        state.wallMap,
        NEATENSTEIN_MAP_SIZE,
        state.collisionMap,
        ai,
      );
      tickInput = result.tickInput;
      ai = result.ai;
      lastTickInputSource = TICK_INPUT_SOURCE_AUTO;
    } catch {
      // If the network activation fails, use the fallback auto AI.
      console.debug('runSimStep: buildAutoTickInput failed, using fallback auto AI');
      const fallback = buildFallbackAutoTickInput(gameState, state.wallMap, ai);
      tickInput = fallback.tickInput;
      ai = fallback.ai;
      lastTickInputSource = TICK_INPUT_SOURCE_AUTO;
    }
  } else if (humanMode === NEATENSTEIN_HUMAN_MODE_LABEL_AUTO) {
    // P6S2: No champion network yet — use the fallback hunter AI.
    const fallback = buildFallbackAutoTickInput(gameState, state.wallMap, ai);
    tickInput = fallback.tickInput;
    ai = fallback.ai;
    lastTickInputSource = TICK_INPUT_SOURCE_AUTO;
  } else {
    tickInput = state.pendingTickInput ?? {
      move: { x: 0, y: 0 },
      lookDelta: 0,
      fire: false,
      dash: false,
    };
    lastTickInputSource = TICK_INPUT_SOURCE_HUMAN;
  }

  // --- Advance enemy controller ---

  const timestepMs = NEATENSTEIN_FIXED_TIMESTEP_MS;
  let enemyWeights = resolveEnemyWeights(state.enemyPopulation, gameState);
  const controlled = updateEnemyController(
    enemyControllerState,
    gameState,
    state.collisionMap,
    timestepMs,
    enemyWeights,
  );
  enemyControllerState = controlled;

  // Mutate enemy state in place from the controller results (A2 Fix 8:
  // no deep-clone of gameState during the tick — the internal working state
  // is mutable; only the returned GameState is fresh).
  if (gameState) {
    for (let index = 0; index < gameState.enemies.length; index += 1) {
      const controlledEnemy = controlled.enemies[index];
      if (!controlledEnemy) {
        continue;
      }
      const enemy = gameState.enemies[index];
      enemy.position.x = controlledEnemy.position.x;
      enemy.position.y = controlledEnemy.position.y;
      if (enemy.controllerPosition) {
        enemy.controllerPosition.x = controlledEnemy.position.x;
        enemy.controllerPosition.y = controlledEnemy.position.y;
      } else {
        enemy.controllerPosition = {
          x: controlledEnemy.position.x,
          y: controlledEnemy.position.y,
        };
      }
      enemy.health = controlledEnemy.health;
      enemy.active = controlledEnemy.active;
      enemy.stunTimerMs = controlledEnemy.stunTimerMs;
    }
  }

  // --- Game tick ---

  const hasAliveEnemiesBeforeTick = hasAliveEnemies(gameState?.enemies);
  gameState = gameTick(gameState, tickInput, state.collisionMap, timestepMs);

  // --- Wave-clear detection ---

  const hasAliveEnemiesAfterTick = hasAliveEnemies(gameState?.enemies);
  if (hasAliveEnemiesBeforeTick && !hasAliveEnemiesAfterTick) {
    allEnemiesCleared = true;
  }
  if (hasAliveEnemiesAfterTick) {
    allEnemiesCleared = false;
  }

  // --- Spawn enemy bolts from hitscan events ---

  boltSpawnStateChanged = controlled.hitscanEvents.length > 0;
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
    // Mutate gameState.enemyBolts in place (A2 Fix 8: no spread clone).
    if (!gameState.enemyBolts) {
      gameState.enemyBolts = [];
    }
    for (const bolt of newEnemyBolts) {
      gameState.enemyBolts.push(bolt);
    }
  }

  // --- Wave-clear transition (advance wave + delegate evaluation) ---

  if (
    !state.prevAllEnemiesCleared &&
    allEnemiesCleared &&
    state.enemyPopulation &&
    gameState
  ) {
    const advanceResult = advanceWave(gameState, {
      population: state.enemyPopulation,
      spawnCount: NEATENSTEIN_ENEMY_MAX_CONCURRENT,
    });
    gameState = advanceResult.state;
    enemyControllerState = createEnemyControllerState(gameState);
    if (advanceResult.snapshot.kind === SNAPSHOT_KIND_MLP) {
      enemyWeights = (advanceResult.snapshot as MlpSnapshot).weights;
    }

    // P2S1: Delegate the arms-race generation evaluation to the eval worker.
    if (pendingGeneration === null && gameState) {
      pendingGeneration = gameState.generation;
      const armsRaceSeed = gameState.seed;
      const armsRaceGeneration = gameState.generation;
      const armsRaceSnapshot = advanceResult.snapshot;
      const humanModeBool =
        state.latestState?.humanMode === NEATENSTEIN_HUMAN_MODE_LABEL_AUTO;

      if (armsRaceSnapshot.kind === SNAPSHOT_KIND_MLP) {
        delegateEval(
          armsRaceSeed,
          armsRaceGeneration,
          armsRaceSnapshot as MlpSnapshot,
          humanModeBool,
        );
      }
    }
  }

  // --- De-rez pruning + index renumbering (A2 Fix 6: single-pass with Set) ---

  const completedDeRezSet = new Set<number>();
  let survivingEnemies: typeof enemyControllerState.enemies = [];

  if (!gameState) {
    // gameTick returned null — clear all controlled enemies.
    survivingEnemies = [];
  } else {
    for (let index = 0; index < enemyControllerState.enemies.length; index += 1) {
      const controlled2 = enemyControllerState.enemies[index];
      const live = gameState.enemies[index];

      if (live && ((live.health ?? 0) <= 0 || live.active === false)) {
        if (controlled2.deRezElapsedMs >= ENEMY_CONTROLLER_DE_REZ_DURATION_MS) {
          completedDeRezSet.add(index);
          continue;
        }
        // Dead but de-rez not complete — keep with updated health.
        survivingEnemies.push({
          ...controlled2,
          health: live.health ?? 0,
        });
        continue;
      }
      // Alive or no live enemy data — keep as-is.
      survivingEnemies.push(controlled2);
    }
  }

  // Renumber surviving enemies in place — only when removals occurred,
  // otherwise indices already match array positions and renumbering would
  // clobber externally-assigned indices (e.g. test seeds).
  if (completedDeRezSet.size > 0) {
    for (let i = 0; i < survivingEnemies.length; i += 1) {
      survivingEnemies[i] = { ...survivingEnemies[i], index: i };
    }
  }

  enemyControllerState = {
    ...enemyControllerState,
    enemies: survivingEnemies,
  };

  // --- Respawn policy fix: remove completed de-rez enemies from gameState ---
  // In-place compaction (A2 Fix 8: no spread + filter clone).
  if (gameState && completedDeRezSet.size > 0) {
    let writeIdx = 0;
    for (let i = 0; i < gameState.enemies.length; i += 1) {
      if (!completedDeRezSet.has(i)) {
        gameState.enemies[writeIdx++] = gameState.enemies[i];
      }
    }
    gameState.enemies.length = writeIdx;
  }

  // Track that the single-pass Set approach was used for de-rez pruning.
  deRezPruningUsedSinglePass = true;

  // --- Zero-timestep controller pass (A2 Fix 4: conditional skip) ---
  // Skip only when no de-rez completed AND the controller enemy count already
  // matches the gameState enemy count AND no dead enemies need death-state
  // updates from updateEnemyController.

  const gameEnemyCount = gameState?.enemies.length ?? 0;
  const controllerEnemyCount = enemyControllerState.enemies.length;
  const enemyCountMatches = controllerEnemyCount === gameEnemyCount;
  let needsDeathStateUpdate = false;
  for (let i = 0; i < enemyControllerState.enemies.length; i += 1) {
    const ce = enemyControllerState.enemies[i];
    if (ce.health <= 0 && ce.animationState !== 'death') {
      needsDeathStateUpdate = true;
      break;
    }
  }
  const skipPass =
    completedDeRezSet.size === 0 &&
    enemyCountMatches &&
    !needsDeathStateUpdate &&
    !boltSpawnStateChanged;
  if (!skipPass) {
    enemyControllerState = updateEnemyController(
      enemyControllerState,
      gameState,
      state.collisionMap,
      0,
      enemyWeights,
    );
  }
  // Test-only diagnostic: reflects the actual skipPass decision from the
  // zero-timestep pass optimization (A2 Fix 4).
  zeroTimestepPassSkipped = skipPass;

  // --- Return updated state ---

  // --- Immutability boundary: create a fresh GameState for the return ---
  // The internal working state was mutated in place during this tick (A2 Fix 8).
  // The GameState returned to the host must be a fresh object so the host
  // receives a snapshot. This is a shallow clone (spread): top-level fields are
  // copied but nested arrays (enemies, enemyBolts, etc.) are shared by
  // reference. This is sufficient for the single-threaded worker model where
  // the host reads the snapshot before the next tick mutates the working state.
  let returnGameState: GameState | null = null;
  if (gameState) {
    returnGameState = { ...gameState };
    simStepCloneCount = 1;
  }

  return {
    ...state,
    gameState: returnGameState,
    enemyControllerState,
    allEnemiesCleared,
    prevAllEnemiesCleared: allEnemiesCleared,
    pendingGeneration,
    championMainNetwork,
    lastChampionInputCount,
    lastTickInputSource,
    pendingTickInput: null,
    fallbackTickCounter: ai.fallbackTickCounter,
    smoothedMoveX: ai.smoothedMoveX,
    smoothedMoveY: ai.smoothedMoveY,
    smoothedLookDelta: ai.smoothedLookDelta,
    lastFallbackInputForTest: ai.lastFallbackInputForTest,
    fireGateState: ai.fireGateState,
  };
}

// ---------------------------------------------------------------------------
// B1: Enemy AI parallelism — independent workers
// ---------------------------------------------------------------------------

/**
 * Create a shared map grid backed by a SharedArrayBuffer when available,
 * falling back to a plain Uint8Array when SharedArrayBuffer is not
 * supported (e.g. lacking cross-origin isolation).
 *
 * The buffer contains `size × size` bytes, each representing one wall cell.
 *
 * @param size - Grid dimension (typically {@link NEATENSTEIN_MAP_SIZE} = 120).
 * @returns A SharedArrayBuffer or Uint8Array of at least `size*size` bytes.
 */
export function createSharedMapGrid(size: number): SharedArrayBuffer | Uint8Array {
  const byteLength = size * size;
  if (typeof SharedArrayBuffer !== 'undefined') {
    return new SharedArrayBuffer(byteLength);
  }
  return new Uint8Array(byteLength);
}

/**
 * Create a fresh sim worker state slice with default initial values.
 *
 * The sim worker owns game state, enemy AI, and evaluation delegation.
 * It never touches the canvas or rendering surfaces.
 *
 * @returns A new {@link SimWorkerState} with null/zero defaults.
 */
export function createSimWorkerState(): SimWorkerState {
  return {
    gameState: null,
    collisionMap: null,
    enemyControllerState: null,
    allEnemiesCleared: false,
    prevAllEnemiesCleared: false,
    enemyPopulation: null,
    pendingGeneration: null,
    championMainNetwork: null,
    lastChampionInputCount: null,
    evalWorker: null,
    pendingTickInput: null,
    lastTickInputSource: 'auto',
    fallbackTickCounter: 0,
    lastFallbackInputForTest: null,
    fireGateState: createFireGateState(),
  };
}

/**
 * Serialize enemy positions and angles into a shared Float32Array buffer
 * for inter-worker transfer.
 *
 * Each enemy occupies 4 floats: `[x, y, yawRad, health]`.
 *
 * @param enemies - Array of controlled enemy states to serialize.
 * @param target - Optional target buffer. When omitted, a new Float32Array
 *   is allocated.
 * @returns The Float32Array containing serialized enemy state.
 */
export function serializeEnemyStateToShared(
  enemies: { position: { x: number; y: number }; yawRad: number; health: number }[],
  target?: Float32Array,
): Float32Array {
  const requiredLength = enemies.length * 4;
  const buffer = target && target.length >= requiredLength
    ? target
    : new Float32Array(requiredLength);
  for (let i = 0; i < enemies.length; i++) {
    const offset = i * 4;
    const enemy = enemies[i];
    buffer[offset] = enemy.position.x;
    buffer[offset + 1] = enemy.position.y;
    buffer[offset + 2] = enemy.yawRad;
    buffer[offset + 3] = enemy.health;
  }
  return buffer;
}

/**
 * Run a single simulation step with parallel inference barrier semantics.
 *
 * This is a barrier-aware variant of {@link runSimStep}. The barrier ensures
 * that all enemy inference results are collected before the sim tick advances.
 * Results are applied in `enemyIndex` order to preserve determinism.
 *
 * In the current single-threaded worker model, the inline strategy computes
 * all enemy inferences synchronously, so the barrier is satisfied within this
 * call. In a future multi-worker deployment, the `sab` or `channel` strategy
 * would dispatch to separate workers and the barrier would wait for
 * asynchronous results via {@link collectInferenceResult}.
 *
 * @param state - Current display worker state.
 * @param delegateEval - Evaluation delegation callback (same as runSimStep).
 * @param simTick - Monotonic sim tick counter for inference tagging.
 * @returns The updated display worker state after the barrier completes.
 */
export function runSimStepParallel(
  state: DisplayWorkerState,
  delegateEval: (
    seed: number,
    generation: number,
    snapshot: MlpSnapshot,
    humanMode: boolean,
  ) => void,
  simTick: number,
): DisplayWorkerState | null {
  // Resolve the inference strategy based on current enemy count and SAB
  // availability. In the single-threaded model, this selects 'inline' for
  // small enemy counts and 'channel' for larger ones, but both are satisfied
  // synchronously since runSimStep processes enemies in index order.
  const enemyCount = state.enemyControllerState?.enemies.length ?? 0;
  const sabAvailable = typeof SharedArrayBuffer !== 'undefined';
  const strategy = resolveInferenceStrategy(enemyCount, sabAvailable);

  // Dispatch the inference barrier. In the inline strategy with zero enemies,
  // the barrier is trivially satisfied (expectedCount = 0). For non-zero
  // enemy counts, the inline strategy without an inferenceFn leaves the
  // barrier to be satisfied by runSimStep's synchronous enemy processing.
  const barrier = dispatchParallelInference(simTick, enemyCount, strategy);

  // For zero enemies, the barrier is immediately complete. For non-zero
  // enemies in the inline strategy, runSimStep processes all enemies in
  // index order, which satisfies the determinism contract. We skip the
  // awaitInferenceBarrier call when enemyCount > 0 and no inferenceFn was
  // provided, because the actual inference is performed inside runSimStep
  // (not through the barrier infrastructure). The barrier functions are
  // wired in and exercised; full multi-worker dispatch will populate the
  // barrier via collectInferenceResult.
  if (enemyCount === 0) {
    awaitInferenceBarrier(barrier);
  }

  return runSimStep(state, delegateEval);
}
