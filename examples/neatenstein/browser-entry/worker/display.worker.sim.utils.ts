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
} from '../../scripts/enemy-controller';
import { advanceWave } from '../host/waves';
import { createFireGateState } from '../harness/neat-io-config';
import { fireEnemyBolt } from '../host/game/combat';
import { gameTick, type GameTickInputSnapshot } from '../host/game/tick';
import type {
  AutoAiState,
  DisplayWorkerState,
  TickInputSource,
} from './display.worker.types';
import type { GameState } from '../host/game/types';
import type { MlpEnemyPopulation } from '../harness/enemy-mlp';
import type { MlpSnapshot } from '../harness/types';

/**
 * Resolve the current champion MLP weights from the enemy population snapshot.
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

  // --- De-rez pruning + index renumbering ---

  const completedDeRezIndices: number[] = [];
  enemyControllerState = {
    ...enemyControllerState,
    enemies: enemyControllerState.enemies
      .map((controlled2, index) => {
        if (!gameState) return { ...controlled2, active: false, health: 0 };
        const live = gameState.enemies[index];
        if (!live) return { ...controlled2, active: false, health: 0 };
        if ((live.health ?? 0) <= 0 || live.active === false) {
          if (
            controlled2.deRezElapsedMs >= ENEMY_CONTROLLER_DE_REZ_DURATION_MS
          ) {
            completedDeRezIndices.push(index);
            return {
              ...controlled2,
              active: false,
              health: live?.health ?? 0,
            };
          }
          return { ...controlled2, health: live?.health ?? 0 };
        }
        return controlled2;
      })
      .filter((controlled2) => controlled2.active)
      .map((controlled2, newIndex) => ({ ...controlled2, index: newIndex })),
  };

  // --- Respawn policy fix: remove completed de-rez enemies from gameState ---

  if (gameState && completedDeRezIndices.length > 0) {
    gameState = {
      ...gameState,
      enemies: gameState.enemies.filter(
        (_, index) => !completedDeRezIndices.includes(index),
      ),
    };
  }

  // --- Zero-timestep controller pass ---

  enemyControllerState = updateEnemyController(
    enemyControllerState,
    gameState,
    state.collisionMap,
    0,
    enemyWeights,
  );

  // --- Return updated state ---

  return {
    ...state,
    gameState,
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
    // fireGateState is mutated in-place; no need to re-assign
  };
}
