import type { NgeDnaCanonicalEnvelope } from '../nge-dna/neat.nge-dna.types';
import type {
  NgeReproductionPolicy,
  NgeReproductionPolicyMode,
} from '../nge-dna/neat.nge-dna.types';
import {
  reproduceParthenogenesis,
  reproducePolyandric,
  reproduceSexual,
  type NgePolyandricDroneInput,
} from '../nge-evolution/neat.nge-evolution';
import {
  reproductionModeHysteresis,
  type ReproductionModePressureSignal,
} from '../nge-evolution/neat.nge-evolution.reproduction-mode';
import type {
  NgeMainAgentAdult,
  NgeMainAgentEquilibrium,
  NgeMainAgentLifecycleConfig,
  NgeMainAgentReproducing,
} from './neat.nge-main-agent.types';
import type { NgeMainAgentEquilibriumResult } from './neat.nge-main-agent.adult';

/**
 * Transition a stable adult equilibrium into the reproducing stage.
 *
 * The reproducing state inherits its topology from the stable equilibrium
 * adult. It is the final stage before the lifecycle runner loops back to embryo
 * for the next generation.
 *
 * @param adult - Adult state entering reproduction.
 * @param equilibrium - Stable equilibrium candidate produced by adult optimization.
 * @param config - Lifecycle config with the deterministic seed.
 * @returns Reproducing state ready to emit the next generation.
 */
export function transitionAdultToReproducing(
  adult: NgeMainAgentAdult,
  equilibrium: NgeMainAgentEquilibrium,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentReproducing {
  return {
    stage: 'reproducing',
    generation: adult.generation,
    seed: config.seed,
    nodeCount: equilibrium.adult.nodeCount,
    edgeCount: equilibrium.adult.edgeCount,
    archetypes: adult.archetypes,
    schemaVersion: adult.schemaVersion,
  };
}

/**
 * Result of running the main-agent reproduction stage.
 *
 * Carries the canonical offspring genome, the mode that produced it, the
 * fingerprints of every parent that contributed genetic material, and the
 * determinism seed so callers can replay or audit the generation.
 */
export interface NgeMainAgentReproductionStageResult {
  /** Canonical offspring DNA produced by the selected operator. */
  offspring: NgeDnaCanonicalEnvelope;
  /** Reproduction mode selected by hysteresis and used by the operator. */
  mode: NgeReproductionPolicyMode;
  /** Deterministic fingerprints of every parent that contributed genetic material. */
  parentFingerprints: string[];
  /** Determinism seed carried from the lifecycle config. */
  seed: number;
}

/**
 * Run the main-agent reproduction stage from a stable adult equilibrium.
 *
 * The stage consults the 3-generation combat-pressure hysteresis policy to
 * choose a reproduction mode, then dispatches to the matching NGE operator:
 * parthenogenesis, polyandric, or sexual crossover. The resulting offspring
 * always uses the canonical seed policy `{ siblingsDifferBySeed: true,
 * twinsAllowed: false }` so siblings diverge by seed and exact twins are
 * disallowed.
 *
 * @param equilibriumResult - Stable adult equilibrium carrying the reproduction-ready candidate.
 * @param pressureHistory - Last-generation combat-pressure window (oldest to newest).
 * @param parentDna - Canonical DNA envelope of the primary parent (queen/first parent).
 * @param config - Lifecycle config with the deterministic seed.
 * @param matePool - Optional secondary DNAs used as drones or sexual partners.
 * @returns Reproduction stage result with offspring, mode, parent fingerprints, and seed.
 */
export function runReproductionStage(
  equilibriumResult: NgeMainAgentEquilibriumResult,
  pressureHistory: readonly ReproductionModePressureSignal[],
  parentDna: NgeDnaCanonicalEnvelope,
  config: NgeMainAgentLifecycleConfig,
  matePool: readonly NgeDnaCanonicalEnvelope[] = [],
): NgeMainAgentReproductionStageResult {
  const modeSelection = reproductionModeHysteresis({
    policy: parentDna.reproductionPolicy,
    generationPressure: pressureHistory,
  });

  const policy = enforceMainAgentSeedPolicy(modeSelection.policy);

  switch (modeSelection.mode) {
    case 'parthenogenesis':
      return reproduceParthenogenesisPath(
        equilibriumResult,
        parentDna,
        policy,
        config,
      );
    case 'polyandric':
      return reproducePolyandricPath(
        equilibriumResult,
        parentDna,
        matePool,
        policy,
        config,
      );
    case 'sexual':
      return reproduceSexualPath(
        equilibriumResult,
        parentDna,
        matePool,
        policy,
        config,
      );
    default:
      return exhaustiveFallback(modeSelection.mode);
  }
}

/**
 * Canonical seed policy for the main-agent reproduction stage.
 *
 * Enforces deterministic divergence between siblings while forbidding exact
 * twins. The same seed policy is applied regardless of what the parent DNA
 * originally carried.
 */
const MAIN_AGENT_SEED_POLICY: {
  siblingsDifferBySeed: true;
  twinsAllowed: false;
} = {
  siblingsDifferBySeed: true,
  twinsAllowed: false,
};

/**
 * Neutral fitness score used for mates whose equilibrium metrics are unknown.
 *
 * The NGE reproduction operators only consume fitness for tie-breaking and
 * regional hash thresholds; a fixed neutral value keeps the main-agent stage
 * deterministic and independent of external telemetry.
 */
const NEUTRAL_MATE_FITNESS = 0.5;

/**
 * Enforce the main-agent seed policy on a hysteresis-selected policy.
 *
 * @param policy - Policy returned by the hysteresis selector.
 * @returns The same policy with the seed policy slot forced to the main-agent contract.
 */
function enforceMainAgentSeedPolicy(
  policy: NgeReproductionPolicy,
): NgeReproductionPolicy {
  return {
    ...policy,
    seedPolicy: MAIN_AGENT_SEED_POLICY,
  };
}

/**
 * Resolve the pool of secondary mates used by polyandric and sexual modes.
 *
 * When no external mate pool is supplied the primary parent is reused as a safe
 * fallback so the stage always produces deterministic output.
 *
 * @param matePool - Optional secondary DNAs supplied by the caller.
 * @param parentDna - Primary parent DNA to fall back to.
 * @returns A non-empty list of mate DNAs.
 */
function resolveMatePool(
  matePool: readonly NgeDnaCanonicalEnvelope[],
  parentDna: NgeDnaCanonicalEnvelope,
): NgeDnaCanonicalEnvelope[] {
  return matePool.length > 0 ? matePool.slice() : [parentDna];
}

/**
 * Extract a deterministic parent score from the stable candidate.
 *
 * Falls back to the neutral mate fitness when the candidate metrics are absent.
 *
 * @param equilibriumResult - Equilibrium result carrying the stable candidate.
 * @returns A numeric score suitable for reproduction operators.
 */
function resolveParentScore(
  equilibriumResult: NgeMainAgentEquilibriumResult,
): number {
  return (
    equilibriumResult.stableCandidate.fitnessMetrics.complexityBonus ??
    NEUTRAL_MATE_FITNESS
  );
}

/**
 * Run the parthenogenesis branch of the reproduction stage.
 *
 * @param equilibriumResult - Stable adult equilibrium for the primary parent.
 * @param parentDna - Canonical DNA of the primary parent.
 * @param policy - Hysteresis-selected policy with enforced seed policy.
 * @param config - Lifecycle config with the deterministic seed.
 * @returns Reproduction stage result for the asexual path.
 */
function reproduceParthenogenesisPath(
  equilibriumResult: NgeMainAgentEquilibriumResult,
  parentDna: NgeDnaCanonicalEnvelope,
  policy: NgeReproductionPolicy,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentReproductionStageResult {
  const result = reproduceParthenogenesis({
    ngeEnabled: true,
    parent: parentDna,
    parentId: parentDna.fingerprint,
    policy,
  });

  return {
    offspring: result.offspring,
    mode: 'parthenogenesis',
    parentFingerprints: [parentDna.fingerprint],
    seed: config.seed,
  };
}

/**
 * Run the polyandric branch of the reproduction stage.
 *
 * Selects up to `policy.polyandricDroneCount` mates from the pool, reusing the
 * primary parent as a fallback when the pool is empty.
 *
 * @param equilibriumResult - Stable adult equilibrium for the queen parent.
 * @param parentDna - Canonical DNA of the queen parent.
 * @param matePool - Optional secondary DNAs used as drones.
 * @param policy - Hysteresis-selected policy with enforced seed policy.
 * @param config - Lifecycle config with the deterministic seed.
 * @returns Reproduction stage result for the multi-parent path.
 */
function reproducePolyandricPath(
  equilibriumResult: NgeMainAgentEquilibriumResult,
  parentDna: NgeDnaCanonicalEnvelope,
  matePool: readonly NgeDnaCanonicalEnvelope[],
  policy: NgeReproductionPolicy,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentReproductionStageResult {
  const pool = resolveMatePool(matePool, parentDna);
  const drones: NgePolyandricDroneInput[] = pool
    .slice(0, policy.polyandricDroneCount)
    .map((dna) => ({
      dna,
      parentId: dna.fingerprint,
      fitness: NEUTRAL_MATE_FITNESS,
    }));

  const result = reproducePolyandric({
    ngeEnabled: true,
    queen: parentDna,
    queenId: parentDna.fingerprint,
    drones,
    policy,
  });

  return {
    offspring: result.offspring,
    mode: 'polyandric',
    parentFingerprints: [
      parentDna.fingerprint,
      ...drones.map((d) => d.dna.fingerprint),
    ],
    seed: config.seed,
  };
}

/**
 * Run the sexual crossover branch of the reproduction stage.
 *
 * Uses the first mate in the pool as the second parent, falling back to the
 * primary parent when no pool is supplied.
 *
 * @param equilibriumResult - Stable adult equilibrium for the first parent.
 * @param parentDna - Canonical DNA of the first parent.
 * @param matePool - Optional secondary DNAs used as the second parent.
 * @param policy - Hysteresis-selected policy with enforced seed policy.
 * @param config - Lifecycle config with the deterministic seed.
 * @returns Reproduction stage result for the sexual path.
 */
function reproduceSexualPath(
  equilibriumResult: NgeMainAgentEquilibriumResult,
  parentDna: NgeDnaCanonicalEnvelope,
  matePool: readonly NgeDnaCanonicalEnvelope[],
  policy: NgeReproductionPolicy,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentReproductionStageResult {
  const secondParent = resolveMatePool(matePool, parentDna)[0];
  const firstParentScore = resolveParentScore(equilibriumResult);

  const result = reproduceSexual({
    firstParent: parentDna,
    firstParentId: parentDna.fingerprint,
    firstParentScore,
    secondParent,
    secondParentId: secondParent.fingerprint,
    secondParentScore: NEUTRAL_MATE_FITNESS,
    policy,
  });

  return {
    offspring: result.offspring,
    mode: 'sexual',
    parentFingerprints: [parentDna.fingerprint, secondParent.fingerprint],
    seed: config.seed,
  };
}

/**
 * Fall back for unexpected reproduction modes.
 *
 * The switch above covers every known {@link NgeReproductionPolicyMode}, so a
 * runtime mismatch is treated as an implementation bug rather than a user error.
 *
 * @param mode - Unexpected mode value.
 * @returns Never; always throws.
 */
function exhaustiveFallback(mode: never): never {
  throw new Error(`Unsupported reproduction mode encountered: ${String(mode)}`);
}
