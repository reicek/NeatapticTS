import type { NgeEvolutionCompatibilityDistanceWeights } from './neat.nge-evolution.types';

/**
 * Default alpha weight for the classic NEAT topology-distance term used in speciation.
 */
export const NGE_EVOLUTION_DEFAULT_ALPHA_TOPOLOGY = 0.4;

/**
 * Default alpha weight for the NGE computation-motif distance term used in speciation.
 */
export const NGE_EVOLUTION_DEFAULT_ALPHA_COMPUTATION = 0.2;

/**
 * Default alpha weight for the NGE memory-tier distance term used in speciation.
 */
export const NGE_EVOLUTION_DEFAULT_ALPHA_MEMORY = 0.2;

/**
 * Default alpha weight for the NGE lifecycle-policy distance term used in speciation.
 */
export const NGE_EVOLUTION_DEFAULT_ALPHA_LIFECYCLE = 0.2;

/**
 * Default per-term alpha bag used when callers do not inject custom weights.
 */
export const NGE_EVOLUTION_DEFAULT_COMPATIBILITY_DISTANCE_WEIGHTS = {
  topology: NGE_EVOLUTION_DEFAULT_ALPHA_TOPOLOGY,
  computation: NGE_EVOLUTION_DEFAULT_ALPHA_COMPUTATION,
  memory: NGE_EVOLUTION_DEFAULT_ALPHA_MEMORY,
  lifecycle: NGE_EVOLUTION_DEFAULT_ALPHA_LIFECYCLE,
} as const satisfies NgeEvolutionCompatibilityDistanceWeights;

/**
 * Default weak-reference decay used by the optional epigenetic prior operator.
 */
export const NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY = 0.05;

/**
 * Default fraction of DNA regions that polyandric drone donors may patch.
 *
 * A value of `0.1` means only the first 10% of the queen's patchable regions
 * (rounded up) are exposed to drone contributions.
 */
export const NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION = 0.1;

/**
 * Default queen-bias multiplier for polyandric region merging.
 *
 * `1.0` means the queen data wins every conflict; `0.0` means the drone data
 * always wins; values in between act as a deterministic threshold keyed by the
 * FNV-1a hash of each region id. The same queen/drone pair and bias therefore
 * always produce the same offspring region.
 *
 * See the FNV-1a reference:
 * [Wikipedia — Fowler–Noll–Vo hash function](https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function).
 */
export const NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS = 1.0;
