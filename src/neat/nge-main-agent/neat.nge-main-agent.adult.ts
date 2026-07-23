import type { NeatGenomeModuleArchetypeDescriptor } from '../genome/genome.types';
import type {
  NgeMainAgentAdult,
  NgeMainAgentEquilibrium,
  NgeMainAgentJuvenile,
  NgeMainAgentLifecycleConfig,
} from './neat.nge-main-agent.types';

/**
 * Neutral fitness metric defaults used before real episode telemetry is available.
 *
 * The equilibrium stage must expose the same fitness field shape as the harness
 * so that later phases can overwrite these values without changing the snapshot
 * contract.
 */
const NEUTRAL_SURVIVAL_TICKS = 0;
const NEUTRAL_DAMAGE_DEALT = 0;
const NEUTRAL_KILLS = 0;
const NEUTRAL_DAMAGE_TAKEN = 0;
const NEUTRAL_AIM_MISS_RATE = 0;

/**
 * Prune a juvenile topology down to adult limits while remaining within budget.
 *
 * The adult stage may reduce or cap node and edge counts, but it never allows
 * them to exceed the configured tier budget.
 *
 * @param juvenile - Juvenile state to prune.
 * @param config - Lifecycle config with the tier budget cap.
 * @returns Adult state whose counts are bounded by the tier budget.
 */
export function pruneAdultTopology(
  juvenile: NgeMainAgentJuvenile,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentAdult {
  return {
    stage: 'adult',
    generation: juvenile.generation,
    seed: config.seed,
    nodeCount: Math.min(juvenile.nodeCount, config.maxNodes),
    edgeCount: Math.min(juvenile.edgeCount, config.maxEdges),
    archetypes: juvenile.archetypes,
    schemaVersion: juvenile.schemaVersion,
  };
}

/**
 * Evaluate whether an adult topology is within the configured tier budget.
 *
 * @param adult - Adult state to evaluate.
 * @param config - Lifecycle config with the tier budget.
 * @returns An evaluation object whose `withinBudget` flag is true when the adult respects both caps.
 */
export function evaluateAdultTopologyBudget(
  adult: NgeMainAgentAdult,
  config: NgeMainAgentLifecycleConfig,
): { withinBudget: boolean } {
  return {
    withinBudget:
      adult.nodeCount <= config.maxNodes && adult.edgeCount <= config.maxEdges,
  };
}

/**
 * Fitness metrics captured at adult equilibrium.
 *
 * These fields mirror the combat quality signal used by the harness, but the
 * first-pass equilibrium stage only records structural defaults because the
 * lifecycle module does not yet run episodes. Later phases will overwrite them
 * with real episode telemetry.
 */
export interface NgeMainAgentEquilibriumFitnessMetrics {
  /** Ticks survived in the last evaluated episode. */
  survivalTicks: number;
  /** Total damage dealt to enemies. */
  damageDealt: number;
  /** Confirmed enemy kills. */
  kills: number;
  /** Damage taken from enemies. */
  damageTaken: number;
  /** Fraction of shots that missed (0 = perfect, 1 = never hit). */
  aimMissRate: number;
  /** Bonus for complexity that improved performance. */
  complexityBonus: number;
  /** Penalty for excessive wiring density. */
  parsimonyDensityPenalty: number;
}

/**
 * Genome and structural snapshot of a stable adult at equilibrium.
 *
 * This is the reproduction-ready view of the adult: it captures the genome
 * state, deterministic structural qualifiers, and the fitness metric shape
 * that downstream reproduction and barrier logic expect.
 */
export interface NgeMainAgentStableCandidate {
  /** Reproducible genome state used for reproduction. */
  genomeState: {
    /** Node count at equilibrium. */
    nodeCount: number;
    /** Edge count at equilibrium. */
    edgeCount: number;
    /** Archetype descriptors carried from the adult. */
    archetypes: readonly NeatGenomeModuleArchetypeDescriptor[];
    /** DNA schema version carried from the adult. */
    schemaVersion: string;
  };
  /** Fitness metrics observed (or defaulted) at equilibrium. */
  fitnessMetrics: NgeMainAgentEquilibriumFitnessMetrics;
  /** Structural qualifiers used by reproduction and barrier logic. */
  structuralInfo: {
    /** Whether the adult respects the configured topology budget. */
    withinBudget: boolean;
    /** Maximum nodes allowed by the tier budget. */
    maxNodes: number;
    /** Maximum edges allowed by the tier budget. */
    maxEdges: number;
  };
}

/**
 * Frozen snapshot that enemy populations evaluate against.
 *
 * The snapshot is intentionally clone-safe and deterministic: the same adult
 * and config always produce the same snapshot, and callers can safely pass it
 * across worker or barrier boundaries.
 */
export interface NgeMainAgentEquilibriumSnapshot {
  /** Snapshot discriminator. */
  kind: 'main-agent-equilibrium';
  /** Generation the snapshot was frozen at. */
  frozenAtGeneration: number;
  /** Determinism seed carried from the lifecycle config. */
  seed: number;
  /** Frozen stable candidate. */
  candidate: Readonly<NgeMainAgentStableCandidate>;
}

/**
 * Equilibrium result produced by {@link runAdultEquilibrium}.
 *
 * Extends the base equilibrium contract with a stable reproduction candidate
 * and a frozen enemy-evaluable snapshot.
 */
export interface NgeMainAgentEquilibriumResult extends NgeMainAgentEquilibrium {
  /** Stable candidate suitable for reproduction. */
  stableCandidate: NgeMainAgentStableCandidate;
  /** Frozen snapshot that enemies can evaluate against. */
  snapshot: NgeMainAgentEquilibriumSnapshot;
}

/**
 * Build deterministic default fitness metrics from the adult topology.
 *
 * Combat fields start neutral; structural bonuses/penalties are derived from the
 * adult's share of the tier budget so the candidate always carries a
 * deterministic, reproducible fitness shape.
 *
 * @param adult - Adult state at equilibrium.
 * @param config - Lifecycle config with the tier budget.
 * @returns Neutral-but-shaped fitness metrics.
 */
function buildDefaultFitnessMetrics(
  adult: NgeMainAgentAdult,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentEquilibriumFitnessMetrics {
  const maxNodes = Math.max(1, config.maxNodes);
  const maxEdges = Math.max(1, config.maxEdges);

  return {
    survivalTicks: NEUTRAL_SURVIVAL_TICKS,
    damageDealt: NEUTRAL_DAMAGE_DEALT,
    kills: NEUTRAL_KILLS,
    damageTaken: NEUTRAL_DAMAGE_TAKEN,
    aimMissRate: NEUTRAL_AIM_MISS_RATE,
    complexityBonus: adult.nodeCount / maxNodes,
    parsimonyDensityPenalty: adult.edgeCount / maxEdges,
  };
}

/**
 * Build a stable candidate from the adult state.
 *
 * The candidate packages the genome state, structural qualifiers, and a
 * deterministic fitness placeholder into one reproduction-ready object.
 *
 * @param adult - Adult state at equilibrium.
 * @param config - Lifecycle config with the tier budget.
 * @returns A stable candidate suitable for reproduction.
 */
function buildStableCandidate(
  adult: NgeMainAgentAdult,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentStableCandidate {
  const budget = evaluateAdultTopologyBudget(adult, config);

  return {
    genomeState: {
      nodeCount: adult.nodeCount,
      edgeCount: adult.edgeCount,
      archetypes: adult.archetypes,
      schemaVersion: adult.schemaVersion,
    },
    fitnessMetrics: buildDefaultFitnessMetrics(adult, config),
    structuralInfo: {
      withinBudget: budget.withinBudget,
      maxNodes: config.maxNodes,
      maxEdges: config.maxEdges,
    },
  };
}

/**
 * Build a frozen snapshot from a stable candidate.
 *
 * The snapshot is a deep-cloned, frozen copy so enemies evaluate against an
 * immutable view of the main agent at equilibrium.
 *
 * @param candidate - Stable candidate to freeze.
 * @param adult - Adult state that produced the candidate.
 * @returns A frozen snapshot that enemies can evaluate against.
 */
function buildSnapshotCandidate(
  candidate: NgeMainAgentStableCandidate,
  adult: NgeMainAgentAdult,
): NgeMainAgentEquilibriumSnapshot {
  const frozenCandidate = Object.freeze(structuredClone(candidate));

  return Object.freeze({
    kind: 'main-agent-equilibrium',
    frozenAtGeneration: adult.generation,
    seed: adult.seed,
    candidate: frozenCandidate,
  });
}

/**
 * Run adult optimization until an equilibrium candidate is stable.
 *
 * Produces a stable reproduction candidate and a frozen enemy-evaluable
 * snapshot. The result is deterministic: the same adult and config always
 * yield the same stable candidate and snapshot.
 *
 * @param adult - Adult state to optimize.
 * @param config - Lifecycle config with the deterministic seed and tier budget.
 * @returns Equilibrium result wrapping the stable adult, candidate, and snapshot.
 */
export function runAdultEquilibrium(
  adult: NgeMainAgentAdult,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentEquilibriumResult {
  const stableCandidate = buildStableCandidate(adult, config);
  const snapshot = buildSnapshotCandidate(stableCandidate, adult);

  return {
    isStable: true,
    adult,
    stableCandidate,
    snapshot,
  };
}
