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
 */
export const NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION = 0.1;

/**
 * Default queen-bias multiplier where `1.0` means queen data wins all conflicts.
 */
export const NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS = 1.0;
