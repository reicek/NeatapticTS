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

import type { Network } from 'neataptic';
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
  /** Total shots fired during the episode (P4S1). Used for rate metrics. */
  shotsFired?: number;
  /** Total shots that struck an enemy (P4S1). Used for hit-rate computation. */
  shotsHit?: number;
  /** Shots fired with no active enemy in the world (P4S1). Penalized as blind fire. */
  shotsBlindFire?: number;
  /** Shots that hit a wall with no enemy nearby (P4S1). Penalized as poor aim. */
  shotsWallHit?: number;
  /** Total ticks elapsed in the episode (P4S1). Used for fire-rate computation. */
  ticksElapsed?: number;
  /**
   * Number of ammo pickups collected during the episode (P2S1).
   *
   * Used for a small fitness bonus; enemy combat metrics remain dominant.
   */
  ammoPickupsCollected?: number;
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
  /**
   * Optional live champion network produced by the hoisted Neat evaluation.
   *
   * When present (P3S2+), the worker passes the evaluated-and-evolved champion
   * network directly so downstream consumers (e.g. Phase 4's player
   * controller) can activate it without re-materializing the genome. The
   * `genome` field remains for backward compatibility with existing callers
   * that do not participate in the hoisted evaluation path.
   */
  network?: Network;
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

// ---------------------------------------------------------------------------
// Phase 5 — Consolidated types (moved from individual harness modules)
// ---------------------------------------------------------------------------

/**
 * Options accepted by {@link createMlpEnemyPopulation}.
 */
export interface CreateMlpEnemyPopulationOptions {
  /** Deterministic seed used to generate the initial variant weights. */
  seed?: number;
}

/**
 * MLP enemy population returned by {@link createMlpEnemyPopulation}.
 */
export interface MlpEnemyPopulation extends EnemyPopulation {
  /**
   * Advance the population snapshot on refresh generations.
   *
   * @param context - Current generation context.
   * @returns The population snapshot. The same reference is returned when no
   *   refresh happens; a new reference is returned on refresh generations.
   */
  update: (context: { generation: number }) => Snapshot;
}

/**
 * Options accepted by {@link createSwarmEnemyPopulation}.
 */
export interface CreateSwarmEnemyPopulationOptions {
  /** Deterministic seed used to generate the shared DNA and per-enemy coordinates. */
  seed?: number;
  /** Maximum cohort size. */
  size?: number;
}

/**
 * One member of the weight-shared cohort.
 */
export interface SwarmVariant {
  /** Stable enemy index within the cohort. */
  id: number;
  /** Shared DNA string that deterministically regenerates the swarm genotype. */
  dna: string;
  /** Shared weight vector for the cohort. */
  weights: Float32Array;
  /** Distinct stigmergic coordinates injected for this enemy member. */
  coordinates: Vector2[];
}

/**
 * Swarm enemy population returned by {@link createSwarmEnemyPopulation}.
 */
export interface SwarmEnemyPopulation extends EnemyPopulation {
  /**
   * Advance the cohort snapshot on refresh generations.
   *
   * @param context - Current generation context.
   * @returns The population snapshot. The same reference is returned when no
   *   refresh happens; a new reference is returned on refresh generations.
   */
  update: (context: { generation: number }) => Snapshot;
}

/**
 * Configuration accepted by {@link runArmsRaceGeneration}.
 */
export interface RunArmsRaceGenerationOptions {
  /** Deterministic seed for the generation. */
  seed: number;
  /** Current co-evolution generation (non-negative integer). */
  generation: number;
  /** Optional frozen enemy snapshot; when omitted the SWARM backend supplies one. */
  enemySnapshot?: Snapshot;
  /** When true, human-mode replay pressure is applied to generation selection. */
  humanMode?: boolean;
  /** Optional replay buffer of death contexts used as replay-driven selection pressure. */
  replayBuffer?: ReplayBuffer;
  /**
   * Optional champion network produced by the hoisted async Neat evaluation
   * (P3S2). When provided, the internal `runMainGeneration` call is skipped
   * (the worker has already evaluated the population) and `championQuality`
   * is used as the quality signal instead.
   */
  championNetwork?: Network;
  /**
   * Optional combat-quality signal for the champion network. When
   * `championNetwork` is provided, this quality signal is used directly
   * instead of running `runMainGeneration`.
   */
  championQuality?: CombatQualitySignal;
}

/**
 * Result emitted by one arms-race generation.
 */
export interface ArmsRaceGenerationResult {
  /** Generation number advanced by one step. */
  generation: number;
  /** Deterministic main-agent champion selected for this generation. */
  mainSnapshot: MainVariant;
  /** Frozen enemy snapshot the main agent was evaluated against. */
  enemySnapshot: Snapshot;
  /** Combat-quality signal for the champion's episode. */
  quality: CombatQualitySignal;
  /** Whether this generation was driven by replay-buffer selection pressure. */
  replayDriven: boolean;
  /** Replay-buffer selection pressure applied to this generation (0 when no replay). */
  replayPressure: number;
  /** Enemy behavior metrics summarising the generation's enemy population. */
  enemyBehaviorMetrics: EnemyBehaviorMetrics;
}

/**
 * One generation snapshot accepted by {@link computeAdaptationSignal}.
 *
 * Pairs a generation number with the enemy behavior metrics observed during
 * that generation so the adaptation signal can diff consecutive generations.
 */
export interface GenerationSnapshot {
  /** Generation number. */
  generation: number;
  /** Enemy behavior metrics observed during this generation. */
  enemyBehaviorMetrics: EnemyBehaviorMetrics;
}

/**
 * Adaptation signal emitted by {@link computeAdaptationSignal}.
 *
 * Summarises the behavioral delta between two consecutive generations into a
 * coarse `direction` label plus the raw numeric deltas for each behavior axis.
 */
export interface AdaptationSignal {
  /** Coarse direction label: `'stronger'`, `'weaker'`, or `'shifted'`. */
  direction: 'stronger' | 'weaker' | 'shifted';
  /** Change in enemy aggression between the two generations. */
  aggressionDelta: number;
  /** Change in enemy movement pattern between the two generations. */
  movementDelta: number;
  /** Change in enemy positioning between the two generations. */
  positioningDelta: number;
}

/**
 * An evaluated main-agent variant, extending {@link Individual} with the raw
 * combat-quality signal produced by its episode.
 */
export interface EvaluatedMainVariant extends Individual<MainVariant> {
  /** Raw combat-quality signal for the variant's episode. */
  signal: CombatQualitySignal;
}

/**
 * Configuration accepted by {@link runMainGeneration}.
 */
export interface RunMainGenerationOptions {
  /** Deterministic seed for the generation. */
  seed: number;
  /** Current co-evolution generation (non-negative integer). */
  generation: number;
  /**
   * Frozen enemy snapshot to evaluate against.
   *
   * Takes precedence over {@link enemy} when both are supplied.
   */
  enemySnapshot?: Snapshot;
  /**
   * Enemy backend selector.
   *
   * When `enemySnapshot` is omitted, the runner resolves a fresh frozen
   * snapshot from the requested enemy backend. Defaults to `'mlp'` to
   * preserve the original harness behavior.
   */
  enemy?: { kind: 'swarm' | 'mlp' };
}

/**
 * Result emitted by one main-agent generation.
 *
 * Extends the raw {@link CombatQualitySignal} with the champion genome produced
 * by the NGE main-agent pipeline and the enemy snapshot the generation was
 * evaluated against.
 */
export interface MainGenerationResult extends CombatQualitySignal {
  /** Champion main-agent genome built by the NGE pipeline. */
  championGenome: unknown;
  /** Frozen enemy snapshot the champion was evaluated against. */
  evaluatedEnemySnapshot: Snapshot;
}

/**
 * Mutable hysteresis state for the soft fire gate.
 *
 * Maintained between ticks to prevent rapid on/off oscillation at the
 * vision boundary. The `fireActive` flag tracks whether the gate is
 * currently open (enemy was recently visible).
 */
export interface FireGateState {
  /** Whether the fire gate is currently open. */
  fireActive: boolean;
}

/**
 * Optional fire-gate configuration passed to {@link networkOutputToTickInput}.
 */
export interface FireGateConfig {
  /** Mutable hysteresis state, persisted between ticks by the caller. */
  state: FireGateState;
  /** Current enemyVisible sensor value (sensor[12]). */
  enemyVisible: number;
}

/**
 * Configuration for {@link createSeedPack}.
 */
export interface CreateSeedPackOptions {
  /** Generation the seed pack belongs to (non-negative integer). */
  generation: number;
  /** Number of deterministic seeds to generate (defaults to the MLP variant count). */
  variantCount?: number;
  /**
   * Optional root seed. When provided, per-variant seeds are derived via
   * {@link hashSeed} so the pack is tied to the caller's seed rather than the
   * generation-only LCG. When omitted, the legacy generation-only LCG is used.
   */
  seed?: number;
}
