/**
 * @module enemy-controller.types
 *
 * Extracted type definitions for the enemy controller module.
 */

import type { CollisionMap } from '../browser-entry/renderer/map';
import type {
  EnemyState,
  GameState,
  Vector2,
} from '../browser-entry/host/game/types';
import type { DistanceMap } from './enemy-navigation';

/**
 * Animation and AI state for a single controlled enemy, produced by the
 * controller each tick and consumed by the sprite renderer.
 *
 * This type is intentionally local to the controller so the core
 * {@link EnemyState} can stay minimal while the renderer gets richer,
 * per-enemy behavior data.
 */
export interface ControlledEnemy {
  /** Index of this enemy in the source {@link GameState.enemies} array. */
  index: number;
  /** Current world position after movement and collision resolution. */
  position: Vector2;
  /** Current hit points, mirrored from the source {@link EnemyState}. */
  health: number;
  /** Facing angle in radians; 0 = +X axis. */
  yawRad: number;
  /** Animation state consumed by the sprite renderer. */
  animationState: 'idle' | 'move' | 'fire' | 'death' | 'damage';
  /** Remaining hitscan ammunition. Depletes to trigger de-rez. */
  ammo: number;
  /** Milliseconds until the enemy may fire again. */
  fireCooldownMs: number;
  /** Milliseconds spent in the death de-rez animation. */
  deRezElapsedMs: number;
  /** `true` while the enemy is still active (not fully de-rezzed). */
  active: boolean;
  /** Sim tick counter driving the walk cycle (stand → walk1 → stand → walk2). Increments each tick the enemy moves; resets to 0 when idle. */
  walkTick: number;
  /** Remaining sim ticks for the muzzle-flash shoot blink on the upper body. When > 0 the renderer composites the shoot upper body over the walk lower body. */
  shootBlinkTicks: number;
  /** Number of consecutive ticks the enemy has been stalled in flanking mode. When this exceeds 3, the enemy temporarily switches to BFS mode to avoid permanent stalls against walls. */
  flankStallTicks: number;
  /** Number of consecutive ticks the enemy has been stalled in BFS mode. When this exceeds 3, the enemy tries non-distance-reducing cardinal directions for one tick to escape diagonal-gap deadlocks. */
  bfsStallTicks: number;
  /**
   * Neural network weights for this enemy's MLP controller, or `undefined`
   * when the enemy uses the default BFS navigation fallback.
   */
  weights: Float32Array | undefined;
  /** Variant identifier for this enemy (0 = default/champion). */
  variantId: number;
  /**
   * BFS distance at the enemy's cell from the previous tick, used to compute
   * the progress delta in the vision vector. `-1` when no previous data is
   * available (first tick or respawn).
   */
  previousStepDistance: number;
  /** Remaining hit-stun time in milliseconds (0 when not stunned). */
  stunTimerMs: number;
}

/**
 * One hitscan fire event emitted by an enemy this tick.
 */
export interface HitscanEvent {
  /** Source enemy index in {@link GameState.enemies}. */
  enemyIndex: number;
  /** World-space origin of the hitscan ray. */
  origin: Vector2;
  /** Normalized direction toward the player. */
  direction: Vector2;
  /** Hit points removed from the player on a confirmed hit. */
  damage: number;
}

/**
 * Output of a single controller tick, bundling per-enemy AI/animation
 * descriptors with any hitscan fire events produced this tick.
 */
export interface EnemyControllerState {
  /** Per-enemy AI/animation descriptors. */
  enemies: ControlledEnemy[];
  /** Hitscan fire events produced this tick. */
  hitscanEvents: HitscanEvent[];
}

/**
 * Mutable context object threaded through the
 * {@link updateControlledEnemy} pipeline. Each branch executor reads from
 * and writes to this shared object, keeping the orchestrator declarative
 * and avoiding long parameter lists.
 */
export interface EnemyUpdateContext {
  // Immutable inputs (set once at context creation).
  /** Index in the source enemy array. */
  readonly index: number;
  /** Source enemy snapshot. */
  readonly enemyState: EnemyState;
  /** Previous controlled state, if any. */
  readonly previous: ControlledEnemy | undefined;
  /** Current game snapshot. */
  readonly gameState: GameState;
  /** Map queried for solid cells. */
  readonly collisionMap: CollisionMap;
  /** BFS distance map from the player position. */
  readonly distanceMap: DistanceMap;
  /** Tick duration in milliseconds. */
  readonly dtMs: number;
  /** Array to append any fire event to. */
  readonly hitscanEvents: HitscanEvent[];
  /** Champion MLP weights from the current population snapshot. */
  readonly injectedWeights: Float32Array | undefined;

  // Mutable pipeline state.
  /** Whether this tick is a respawn of a previously de-rezzed enemy. */
  isRespawn: boolean;
  /** Previous state or a fresh default. */
  previousOrDefault: ControlledEnemy;
  /** Current hitscan ammunition. */
  ammo: number;
  /** Remaining fire cooldown in ms. */
  fireCooldownMs: number;
  /** Elapsed de-rez animation time in ms. */
  deRezElapsedMs: number;
  /** Walk cycle tick counter. */
  walkTick: number;
  /** Remaining shoot blink ticks. */
  shootBlinkTicks: number;
  /** Consecutive flanking stall ticks. */
  flankStallTicks: number;
  /** Consecutive BFS stall ticks. */
  bfsStallTicks: number;
  /** MLP weights for this enemy. */
  weights: Float32Array | undefined;
  /** Variant identifier. */
  variantId: number;
  /** Current world position. */
  position: Vector2;
  /** Facing angle in radians. */
  yawRad: number;
  /** Remaining hit-stun time in ms. */
  stunTimerMs: number;
  /** Distance to the player. */
  distToPlayer: number;
  /** Whether the enemy moved this tick. */
  moved: boolean;
  /** Current animation state. */
  animationState: ControlledEnemy['animationState'];
  /** Whether BFS movement is active. */
  shouldMoveByBfs: boolean;
  /** Whether flanking movement is active. */
  shouldMoveByFlank: boolean;
  /** Flanking slot target position. */
  slotTarget: Vector2;
  /** Whether the enemy fired this tick. */
  isFiring: boolean;
}