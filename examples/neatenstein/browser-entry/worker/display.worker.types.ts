/**
 * Shared types for the Neatenstein display worker and eval worker.
 *
 * Extracted from inline type definitions scattered across `display.worker.ts`
 * and the `display.worker.*.utils.ts` files. This file is the single source of
 * truth for all worker-layer types.
 *
 * @module
 */

import type { castRayDDAFromFlatMap } from '../renderer/raycast';
import type { CollisionMap } from '../renderer/map';
import type { EnemyControllerState } from '../../scripts/enemy-controller';
import type { FireGateState } from '../harness/neat-io-config';
import type { GameTickInputSnapshot } from '../host/game/tick';
import type { GameState } from '../host/game/types';
import type { MlpEnemyPopulation } from '../harness/enemy-mlp';
import type { Network } from 'neataptic';
import type { NeatensteinPulse } from '../renderer/pulse';
import type { NeatensteinRenderState } from '../renderer/frame';
import type { NeatensteinSprite } from '../renderer/sprites';
import type { Snapshot } from '../harness/types';

/** Renderer tier selection for the display worker ('worker', 'cpu', or 'gpu'). */
export type DisplayTier = 'worker' | 'cpu' | 'gpu';

/** Source of the most recent tick input (champion NEAT network or human input). */
export type TickInputSource = 'auto' | 'human';

/** Return type of the shared DDA raycaster used to build every render column. */
export type RaycastHit = ReturnType<typeof castRayDDAFromFlatMap>;

/**
 * Mutable auto-AI state slice threaded through the champion and fallback
 * executors.
 *
 * Encapsulates the smoothing, fire-gate, and fallback-counter fields that
 * persist between ticks. The `fireGateState` object is mutated in-place by
 * `applyFireGate`; the primitive fields are replaced via the return value of
 * each executor.
 */
export interface AutoAiState {
  /** Monotonic tick counter for the fallback auto-mode AI. */
  fallbackTickCounter: number;
  /** Hysteresis state for the soft fire gate (mutated in-place). */
  fireGateState: FireGateState;
  /** Smoothed strafe input persisted between ticks. */
  smoothedMoveX: number;
  /** Smoothed forward/back input persisted between ticks. */
  smoothedMoveY: number;
  /** Smoothed look-delta persisted between ticks. */
  smoothedLookDelta: number;
  /** Test-only capture of the last fallback input snapshot. */
  lastFallbackInputForTest: GameTickInputSnapshot | null;
}

/**
 * Encapsulates all module-level mutable worker state.
 *
 * Replaces the 27 individual `let` declarations that were scattered across
 * the main worker module. The factory `createDisplayWorkerState` initialises
 * every field to its default value.
 *
 * @see AC-072
 */
export interface DisplayWorkerState {
  /** Active renderer tier ('worker', 'cpu', or 'gpu'). */
  currentTier: DisplayTier | null;
  /** Transferred OffscreenCanvas for the worker tier. */
  workerCanvas: OffscreenCanvas | null;
  /** 2D context for the worker canvas (lazily initialised). */
  workerContext: OffscreenCanvasRenderingContext2D | null;
  /** Most recent render state received from the host. */
  latestState: NeatensteinRenderState | null;
  /** Pending resize dimensions received before the canvas is assigned. */
  pendingResizeDimensions: { width: number; height: number } | null;
  /** Canonical flat deterministic wall map. */
  wallMap: Uint8Array | null;
  /** Reusable worker-tier z-buffer. */
  workerZBuffer: Float32Array | null;
  /** Deterministic game state maintained and advanced by the worker. */
  gameState: GameState | null;
  /** Collision map built from the init seed. */
  collisionMap: CollisionMap | null;
  /** Active ambient floor and ceiling pulses. */
  activePulses: NeatensteinPulse[];
  /** Enemy AI controller state. */
  enemyControllerState: EnemyControllerState | null;
  /** Active enemy sprite positions computed by the controller. */
  activeEnemySprites: NeatensteinSprite[];
  /** Wave-clear detection flag. */
  allEnemiesCleared: boolean;
  /** Previous-tick snapshot of allEnemiesCleared. */
  prevAllEnemiesCleared: boolean;
  /** MLP enemy population for champion weight injection. */
  enemyPopulation: MlpEnemyPopulation | null;
  /** Launch guard for the hoisted async NEAT evaluation. */
  pendingGeneration: number | null;
  /** Champion main-agent network from the most recent arms-race generation. */
  championMainNetwork: Network | null;
  /** Input count the current champion network was evolved with. */
  lastChampionInputCount: number | null;
  /** Dedicated eval worker. */
  evalWorker: Worker | null;
  /** Pending tick input from the most recent input message. */
  pendingTickInput: GameTickInputSnapshot | null;
  /** Source of the most recent tick input ('auto' or 'human'). */
  lastTickInputSource: TickInputSource;
  /** Monotonic tick counter for the fallback auto-mode AI. */
  fallbackTickCounter: number;
  /** Test-only capture of the last fallback input snapshot. */
  lastFallbackInputForTest: GameTickInputSnapshot | null;
  /** Hysteresis state for the soft fire gate. */
  fireGateState: FireGateState;
  /** Smoothed strafe input. */
  smoothedMoveX: number;
  /** Smoothed forward/back input. */
  smoothedMoveY: number;
  /** Smoothed look-delta. */
  smoothedLookDelta: number;
}

/**
 * Outbound evaluation request payload sent to the eval worker.
 *
 * Consolidated from duplicate definitions in `display.worker.ts` and
 * `eval.worker.ts`. Uses `Snapshot` (the broad union) so both the sender
 * (display worker sends `MlpSnapshot`) and receiver (eval worker) share the
 * same type.
 */
export interface EvalRequestPayload {
  /** Message type discriminator (always `'evaluate'`). */
  type: 'evaluate';
  /** Game seed for the evaluation episode. */
  seed: number;
  /** Current generation (post-advanceWave). */
  generation: number;
  /** Frozen enemy snapshot from advanceWave. */
  enemySnapshot: Snapshot;
  /** Whether the game is in auto mode. */
  humanMode: boolean;
}

/**
 * Inbound evaluation result payload received from the eval worker.
 */
export interface EvalCompletePayload {
  /** Message type discriminator (always `'evalComplete'`). */
  type: 'evalComplete';
  /** Generation number from the evaluation result. */
  generation: number;
  /** Serialized champion network JSON (deserialized via `Network.fromJSON`). */
  championNetworkJSON: Record<string, unknown>;
}