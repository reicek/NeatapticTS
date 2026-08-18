/**
 * Test-only introspection and injection hooks for the Neatenstein display worker.
 *
 * Extracted from `display.worker.ts` as part of the B2 architecture-debt
 * refactoring.  These hooks are re-exported by `display.worker.ts` via
 * `export { ... } from` (which does not match the
 * `export\\s+(?:const|function)\\s+__testOnly` pattern) so the existing test
 * suite can continue to access them through the worker module.
 *
 * @module
 */

/// <reference lib="webworker" />

import type { Network } from 'neataptic';
import type { EnemyControllerState } from '../shared/enemy-controller';
import type { NeatensteinRenderState } from '../renderer/frame';
import type { MlpEnemyPopulation } from '../harness/enemy-mlp';
import type { GameState, EnemyState } from '../host/game/types';
import type { GameTickInputSnapshot } from '../host/game/tick';
import type { AutoAiState, TickInputSource } from './display.worker.types';
import type { FireGateState } from '../harness/neat-io-config';

import {
  NEATENSTEIN_MAIN_NEAT_INPUTS,
  createFireGateState,
} from '../harness/neat-io-config';
import { syncWorkerCanvasSize } from './display.worker.canvas.utils';
import {
  resolveEnemyTeamColor,
  resolveWallFogFactor,
} from './display.worker.color.utils';
import { buildFallbackAutoTickInput } from './display.worker.auto-ai.utils';
import { handleEvalComplete } from './display.worker.eval-delegation.utils';
import { getWorkerState } from './display.worker';

/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the current enemy controller state. */
export const __testOnlyGetEnemyControllerState =
  (): EnemyControllerState | null => getWorkerState().enemyControllerState;

/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the most recently rendered frame state. */
export const __testOnlyGetLatestState = (): NeatensteinRenderState | null =>
  getWorkerState().latestState;

/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the worker canvas resize helper. */
export const __testOnlySyncWorkerCanvasSize = syncWorkerCanvasSize;

/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the enemy-team color resolver. */
export const __testOnlyResolveEnemyTeamColor = resolveEnemyTeamColor;

/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the pending generation guard value. */
export const __testOnlyGetPendingGeneration = (): number | null =>
  getWorkerState().pendingGeneration;

/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the champion main-agent network. */
export const __testOnlyGetChampionMainNetwork = (): Network | null =>
  getWorkerState().championMainNetwork;

/* istanbul ignore next -- test-only hook to inject mock eval worker */
/** Test-only hook to inject a mock eval worker for testing delegation. */
export const __testOnlySetEvalWorker = (
  worker: { postMessage: (msg: unknown) => void; onmessage: unknown } | null,
): void => {
  const s = getWorkerState();
  s.evalWorker = worker as Worker | null;
  if (s.evalWorker) {
    s.evalWorker.onmessage = handleEvalComplete;
  }
};

/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the eval worker instance. */
export const __testOnlyGetEvalWorker = (): unknown => getWorkerState().evalWorker;

/* istanbul ignore next -- test-only hook to inject champion network */
/** Test-only hook to inject a champion main-agent network for auto-mode testing. */
export const __testOnlySetChampionMainNetwork = (
  network: Network | null,
): void => {
  const s = getWorkerState();
  s.championMainNetwork = network;
  s.lastChampionInputCount =
    network !== null ? NEATENSTEIN_MAIN_NEAT_INPUTS : null;
};

/* istanbul ignore next -- test-only hook for extinction testing */
/** Test-only hook to override the champion input count for extinction testing. */
export const __testOnlySetChampionInputCount = (count: number | null): void => {
  getWorkerState().lastChampionInputCount = count;
};

/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the champion input count. */
export const __testOnlyGetChampionInputCount = (): number | null =>
  getWorkerState().lastChampionInputCount;

/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the source of the most recent tick input. */
export const __testOnlyGetLastTickInputSource = (): TickInputSource =>
  getWorkerState().lastTickInputSource;

/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the last fallback AI input snapshot. */
export const __testOnlyGetLastFallbackInput =
  (): GameTickInputSnapshot | null => getWorkerState().lastFallbackInputForTest;

/* istanbul ignore next -- test-only hook to drive buildFallbackAutoTickInput directly */
/** Test-only hook to build a fallback tick input directly for a given state. */
export const __testOnlyBuildFallbackAutoTickInput = (
  gameState: GameState,
): GameTickInputSnapshot => {
  const s = getWorkerState();
  const ai: AutoAiState = {
    fallbackTickCounter: s.fallbackTickCounter,
    fireGateState: s.fireGateState,
    smoothedMoveX: s.smoothedMoveX,
    smoothedMoveY: s.smoothedMoveY,
    smoothedLookDelta: s.smoothedLookDelta,
    lastFallbackInputForTest: s.lastFallbackInputForTest,
  };
  const result = buildFallbackAutoTickInput(gameState, s.wallMap, ai);
  s.fallbackTickCounter = result.ai.fallbackTickCounter;
  s.smoothedMoveX = result.ai.smoothedMoveX;
  s.smoothedMoveY = result.ai.smoothedMoveY;
  s.smoothedLookDelta = result.ai.smoothedLookDelta;
  s.lastFallbackInputForTest = result.ai.lastFallbackInputForTest;
  return result.tickInput;
};

/* istanbul ignore next -- test-only introspection hook */
/** Test-only hook to reset the fire-gate hysteresis state. */
export const __testOnlyResetFireGateState = (): void => {
  getWorkerState().fireGateState = createFireGateState();
};

/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the current fire-gate hysteresis state. */
export const __testOnlyGetFireGateState = (): FireGateState =>
  getWorkerState().fireGateState;

/* istanbul ignore next -- test-only introspection hook */
/** Test-only hook to reset the P1S1 move/look smoothing state. */
export const __testOnlyResetSmoothingState = (): void => {
  const s = getWorkerState();
  s.smoothedMoveX = 0;
  s.smoothedMoveY = 0;
  s.smoothedLookDelta = 0;
};

/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the wall fog-factor resolver. */
export const __testOnlyResolveWallFogFactor = resolveWallFogFactor;

/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the wave-clear detection flag. */
export const __testOnlyGetAllEnemiesCleared = (): boolean =>
  getWorkerState().allEnemiesCleared;

/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the current game state. */
export const __testOnlyGetGameState = (): GameState | null =>
  getWorkerState().gameState;

/* istanbul ignore next -- test-only hook to place enemies near the player */
/**
 * Test-only hook to place synthetic enemies near the player for deterministic
 * combat and rendering tests.
 *
 * Returns the synthesised enemy array so callers can inspect the injected
 * state without re-reading the worker state.  The array is always returned,
 * even when the worker is not yet initialised, so tests can validate the
 * shape without a prior `init` message.
 *
 * @param positions - World-space positions for each synthetic enemy.
 * @returns The synthesised {@link EnemyState} array.
 */
export const __testOnlyInjectTestEnemies = (
  positions: { x: number; y: number }[],
): EnemyState[] => {
  const enemies: EnemyState[] = positions.map((pos, i) => ({
    position: { ...pos },
    health: 100,
    index: i,
    active: true,
    controllerPosition: { ...pos },
    stunTimerMs: 0,
  }));
  const s = getWorkerState();
  if (s.gameState && s.enemyControllerState) {
    s.gameState.enemies = enemies;
    s.enemyControllerState = {
      ...s.enemyControllerState,
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
  }
  return enemies;
};

/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the enemy population instance. */
export const __testOnlyGetEnemyPopulation = (): MlpEnemyPopulation | null =>
  getWorkerState().enemyPopulation;