/**
 * Shared type definitions for the Neatenstein asymmetric co-evolution harness.
 *
 * The harness keeps two populations in tension: a single main NEAT agent that
 * must survive an episode, and a small enemy population that refreshes on a
 * deterministic cadence. All harness state is plain data so it can be cloned,
 * snapshotted, and replayed from a seed.
 *
 * @module
 */

import type { Vector2 } from '../host/game/types';

/**
 * Genome description used by the main NEAT variant.
 *
 * This is intentionally a shallow, clone-safe representation of the phenotype
 * graph; the harness treats it as an opaque snapshot while the main runner
 * materializes it through the library construction pipeline.
 */
export interface Genome {
  /** Node genes or layer descriptors that define the network topology. */
  nodes: unknown[];
  /** Connection genes or weight matrices that define the network wiring. */
  connections: unknown[];
}

/**
 * One evaluated individual in either population.
 *
 * The `variant` payload is backend-specific (a {@link MainVariant} for the
 * main agent, or an {@link EnemyVariant} for an enemy).
 */
export interface Individual<TVariant> {
  /** Stable variant index within the population. */
  id: number;
  /** Backend-specific genotype/phenotype payload. */
  variant: TVariant;
  /** Optional cached fitness used by deterministic selection. */
  fitness?: FitnessScore;
}

/**
 * Numeric fitness value; higher is better.
 */
export type FitnessScore = number;

/**
 * Quality signal emitted by one main-agent episode.
 *
 * These raw telemetry fields are combined by the fitness composite into a
 * scalar fitness score. Positive fields are rewards; negative penalties are
 * applied by subtracting the weighted penalty fields.
 */
export interface CombatQualitySignal {
  /** Ticks (frames) the main agent survived in the episode. */
  survivalTicks: number;
  /** Total damage dealt to enemies. */
  damageDealt: number;
  /** Confirmed enemy kills. */
  kills: number;
  /** Damage taken from enemies (penalty magnitude). */
  damageTaken: number;
  /** Fraction of shots that missed (0 = perfect aim, 1 = never hit). */
  aimMissRate: number;
  /** Bonus for network complexity that improved performance. */
  complexityBonus: number;
  /** Penalty for excessive wiring density (parsimony pressure). */
  parsimonyDensityPenalty: number;
}

/**
 * Optional tuning knobs for the composite enemy fitness.
 *
 * Callers can override the default weights when comparing alternative
 * selection pressures (e.g., navigation-focused vs. combat-focused swarms).
 */
export interface EnemyTeamFitnessConfig {
  /** Weight for the navigation (progress + exploration + anti-stall) component. */
  navigationWeight?: number;
  /** Weight for the combat (damage + survival) component. */
  combatWeight?: number;
  /** Weight applied to collective damage dealt to the main agent. */
  damageWeight?: number;
  /** Weight applied to the number of enemies that survived the episode. */
  survivalWeight?: number;
}

/**
 * Per-step telemetry for one enemy episode rollout (AC-10.5e-002).
 *
 * Carries the per-step BFS distance array and aggregate metrics needed by the
 * composite navigation+combat fitness. The `bfsDistances` array records the
 * BFS distance from the enemy cell to the player goal at the start of each
 * tick, enabling the progress-reward computation (Σ prevDist − curDist).
 */
export interface EnemyEpisodeTelemetry {
  /** Final enemy position in world cells. */
  position: { x: number; y: number };
  /** Per-step BFS distances from the enemy cell to the player goal. */
  bfsDistances: number[];
  /** Total damage dealt to the static player across all ticks. */
  damageDealt: number;
  /** Number of enemies that survived (always 1 in the simplified rollout). */
  enemiesSurvived: number;
  /** Number of unique map cells entered by the enemy. */
  cellsVisited: number;
  /** Number of ticks where the enemy could not move (blocked or no step). */
  stagnationTicks: number;
  /** Final BFS distance from the enemy's final cell to the player goal. */
  finalDistance: number;
}

/**
 * Deterministic seed pack for a single generation.
 *
 * Every variant evaluated in a generation sees the same frozen seed stream so
 * that fitness differences reflect the variant, not environmental variance.
 */
export interface SeedPack {
  /** Generation this seed pack belongs to. */
  generation: number;
  /** Ordered seeds used by the episode RNG for this generation. */
  seeds: number[];
}

/**
 * One main-agent variant under evolution.
 */
export interface MainVariant {
  /** Stable variant id within the main population. */
  id: number;
  /** Genome snapshot that can be materialized into a Network. */
  genome: Genome;
}

/**
 * One enemy variant under evolution.
 *
 * Enemy populations are currently weight-only: the topology is fixed and the
 * harness evolves a small vector of neural weights.
 */
export interface EnemyVariant {
  /** Stable variant id within the enemy population. */
  id: number;
  /** Evolved neural weights for the fixed enemy topology. */
  weights: Float32Array;
}

/**
 * MLP enemy snapshot as stored in the rolling opponent pool.
 */
export interface MlpSnapshot {
  kind: 'mlp';
  /** Evolved weights for the fixed MLP topology. */
  weights: Float32Array;
}

/**
 * SWARM enemy snapshot as stored in the rolling opponent pool.
 */
export interface SwarmSnapshot {
  kind: 'swarm';
  /** Compact DNA string that deterministically regenerates the swarm. */
  dna: string;
  /** Stigmergic coordinates that make up the swarm body. */
  coordinates: Vector2[];
}

/**
 * Union of all enemy snapshots that can be stored in a barrier.
 */
export type Snapshot = MlpSnapshot | SwarmSnapshot;

/**
 * Abstract enemy population backend.
 *
 * Implementations (MLP, SWARM) expose the same surface so the harness can swap
 * backends without changing selection, barrier, or main-runner logic.
 */
export interface EnemyPopulation {
  /** Backend discriminator used by snapshot and refresh logic. */
  kind: 'mlp' | 'swarm';
  /** Number of variants maintained by this population. */
  size: number;
  /**
   * Return the variant at the given index.
   *
   * The exact shape is backend-specific; callers use this to seed an episode
   * without leaking backend details into the runner.
   */
  sample: (index: number) => unknown;
  /** Return a serializable snapshot of the current population champion. */
  snapshot: () => Snapshot;
}

/**
 * Frozen evaluation barrier.
 *
 * A barrier pairs one main-agent variant with one frozen enemy snapshot and a
 * deterministic seed so the same episode can be replayed exactly for fitness
 * evaluation.
 */
export interface BarrierState {
  /** Generation the barrier belongs to. */
  generation: number;
  /** Main-agent variant being evaluated. */
  mainSnapshot: MainVariant;
  /** Frozen enemy snapshot the main agent is evaluated against. */
  enemySnapshot: Snapshot;
  /** Deterministic seed used to run the episode. */
  seed: number;
}

/**
 * Configuration for one co-evolution population.
 */
export interface PopulationConfig {
  /** Number of variants to maintain. */
  size: number;
  /** Backend discriminator ('mlp' or 'swarm'). */
  kind: 'mlp' | 'swarm';
}

/**
 * Top-level harness configuration.
 */
export interface HarnessConfig {
  /** Maximum number of generations to run. */
  maxGenerations: number;
  /** Configuration for the enemy population. */
  enemy: PopulationConfig;
}

/**
 * Result emitted at the end of one generation.
 */
export interface GenerationResult {
  /** Generation number. */
  generation: number;
  /** Selected main-agent champion for this generation. */
  champion: MainVariant;
  /** Aggregated quality signal for the champion's episode. */
  quality: CombatQualitySignal;
}

/**
 * Death context captured at the moment the main agent dies in an episode.
 *
 * The context records the hero's final pose (position, angle, health), a
 * snapshot of the enemy state at death time, the damage source that caused the
 * death, and an optional simulation timestamp for temporal ordering.
 */
export interface DeathContext {
  /** Hero pose at the moment of death. */
  hero: {
    /** Hero position in world coordinates. */
    position: { x: number; y: number };
    /** Hero facing angle in radians. */
    angleRad: number;
    /** Hero health at death (typically 0). */
    health: number;
  };
  /** Snapshot of enemy state at death time. */
  enemies: unknown[];
  /** Identifier of the damage source that caused the death. */
  damageSource: string;
  /** Optional simulation tick when the death occurred. */
  simTimeMs?: number;
}

/**
 * One replay entry stored in the replay buffer.
 *
 * Pairs a {@link DeathContext} with the generation and seed it occurred in so
 * downstream replay logic can reproduce or bias selection pressure from
 * historical death states.
 */
export interface ReplayEntry {
  /** Death context captured at death time. */
  deathContext: DeathContext;
  /** Generation number when the death occurred. */
  generation: number;
  /** Deterministic seed for the generation. */
  seed: number;
}

/**
 * Replay buffer interface for death-context recording.
 *
 * The buffer stores death contexts in a bounded FIFO queue and exposes `push`,
 * `entries`, and `size` so both the harness replay logic and the arms-race
 * human-mode selector can query buffer state.
 */
export interface ReplayBuffer {
  /** Append a death context to the buffer, evicting the oldest entry when full. */
  push: (ctx: DeathContext) => void;
  /** Return all stored death contexts in insertion order. */
  entries: () => DeathContext[];
  /** Return the current number of stored death contexts. */
  size: () => number;
}

/**
 * Per-generation enemy behavior summary used by the death feedback loop.
 *
 * Captures three scalar dimensions of enemy behavior that the adaptation
 * signal compares across consecutive generations:
 * - `aggression` — how aggressively enemies press the main agent.
 * - `movementPattern` — diversity/complexity of enemy movement.
 * - `positioning` — spatial positioning quality relative to the main agent.
 */
export interface EnemyBehaviorMetrics {
  /** Aggression level of the enemy population, expected in [0, 1]. */
  aggression: number;
  /** Movement pattern diversity, expected in [0, 1]. */
  movementPattern: number;
  /** Positioning quality, expected in [0, 1]. */
  positioning: number;
}
