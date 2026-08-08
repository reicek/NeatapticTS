/**
 * NGE main-agent harness entry point for the Neatenstein co-evolution loop.
 *
 * This module wraps the lower-level {@link ./main-runner.ts} lifecycle runner
 * with the Phase 4 main-agent contract: a deterministic lifecycle stage,
 * a tier-capped topology budget, and the exact motif allowlist the Neon
 * Shooter demo is allowed to build from. The harness-level runner is kept
 * separate from the library `src/neat/nge-main-agent/` surface so the demo can
 * iterate on its fitness and episode semantics without touching core NGE.
 *
 * @module
 */

import type {
  NgeMainAgentEmbryo,
  NgeMainAgentLifecycleState,
} from '../../../../src/neat/nge-main-agent/neat.nge-main-agent.types';

import { runMainGeneration } from './main-runner';
import type { CombatQualitySignal, Snapshot } from './types';

/**
 * Lifecycle stages for the Neatenstein main agent.
 *
 * The stage advances deterministically with the generation number so the
 * harness can gate morphology policies (embryo construction, juvenile growth,
 * adult optimization, reproduction) without storing mutable state.
 */
export type NeatensteinMainAgentLifecycleStage =
  'embryo' | 'juvenile' | 'adult' | 'reproducing';

/**
 * Tier-capped topology budget for the main agent.
 *
 * These caps keep the evolved phenotype small enough for real-time inference
 * in the browser while leaving room for meaningful structural adaptation.
 */
export const NeatensteinMainAgentTierBudget = {
  /** Maximum number of nodes in the main-agent phenotype. */
  maxNodes: 64,
  /** Maximum number of edges in the main-agent phenotype. */
  maxEdges: 256,
} as const;

/**
 * Motif allowlist for the Neatenstein main agent.
 *
 * The main agent may only instantiate computation motifs that already exist in
 * the core catalogue. This restriction keeps the demo reproducible and prevents
 * the harness from depending on experimental motif prototypes.
 */
export const NeatensteinMainAgentMotifAllowlist: readonly string[] = [
  'AttentionHead',
  'GatedRecurrentCell',
  'EpisodicSlot',
];

/**
 * Configuration accepted by {@link runMainAgentGeneration}.
 */
export interface RunMainAgentGenerationOptions {
  /** Deterministic seed for the generation. */
  seed: number;
  /** Current co-evolution generation (non-negative integer). */
  generation: number;
  /** Frozen enemy snapshot to evaluate against. */
  enemySnapshot?: Snapshot;
}

/**
 * Result emitted by one main-agent generation.
 *
 * Extends the raw {@link CombatQualitySignal} with the deterministic lifecycle
 * stage, the champion NGE genome, the enemy snapshot used for evaluation, and
 * the internal assimilation report so callers can reason about morphology policy
 * gates.
 */
export interface MainAgentGenerationResult extends CombatQualitySignal {
  /** Lifecycle stage assigned to this generation. */
  stage: NeatensteinMainAgentLifecycleStage;
  /** Champion main-agent genome built by the NGE pipeline. */
  championGenome: NgeMainAgentEmbryo;
  /** Full NGE lifecycle state for the champion at this generation. */
  lifecycleState: NgeMainAgentLifecycleState;
  /** Frozen enemy snapshot the champion was evaluated against. */
  evaluatedEnemySnapshot: Snapshot;
  /** Internal assimilation report (no enemy-derived weights are written). */
  assimilation: { enemyWeightsIncorporated: false };
}

/**
 * Deterministic lifecycle stages ordered by generation modulo.
 *
 * The cycle repeats every four generations so the runner remains deterministic
 * and stateless from the caller's perspective.
 */
const LIFECYCLE_STAGES: readonly NeatensteinMainAgentLifecycleStage[] = [
  'embryo',
  'juvenile',
  'adult',
  'reproducing',
];

/**
 * Run one deterministic main-agent generation and emit the Phase 4 contract.
 *
 * The runner delegates fitness evaluation to the existing main-runner and then
 * attaches the deterministic lifecycle stage. The same `(seed, generation,
 * enemySnapshot)` tuple always produces the same result.
 *
 * @param options - Generation configuration.
 * @returns A {@link MainAgentGenerationResult} combining the combat-quality
 *   signal with the current lifecycle stage.
 *
 * @example
 * ```ts
 * const result = runMainAgentGeneration({
 *   seed: 1,
 *   generation: 0,
 *   enemySnapshot: { kind: 'mlp', weights: new Float32Array(80) },
 * });
 * console.log(result.stage, result.survivalTicks);
 * ```
 */
export function runMainAgentGeneration(
  options: RunMainAgentGenerationOptions,
): MainAgentGenerationResult {
  // Step 1: Evaluate the main-agent generation against the frozen enemy snapshot.
  const signal = runMainGeneration({
    seed: options.seed,
    generation: options.generation,
    enemySnapshot: options.enemySnapshot,
  });

  // Step 2: Assign the deterministic lifecycle stage for this generation.
  const stage = resolveLifecycleStage(options.generation);

  // Step 3: Attach the NGE main-agent contract fields required by Phase 4.
  const championGenome = signal.championGenome;
  const lifecycleState: NgeMainAgentLifecycleState = {
    stage,
    generation: options.generation,
    seed: options.seed,
    embryo: championGenome,
  } as unknown as NgeMainAgentLifecycleState;

  return {
    ...signal,
    stage,
    championGenome,
    lifecycleState,
    evaluatedEnemySnapshot: signal.evaluatedEnemySnapshot,
    assimilation: { enemyWeightsIncorporated: false },
  };
}

/**
 * Resolve the deterministic lifecycle stage for a generation.
 *
 * @param generation - Co-evolution generation.
 * @returns The lifecycle stage assigned to the generation.
 */
function resolveLifecycleStage(
  generation: number,
): NeatensteinMainAgentLifecycleStage {
  // `generation` is already validated as a non-negative integer by the runner.
  return LIFECYCLE_STAGES[generation % LIFECYCLE_STAGES.length];
}
