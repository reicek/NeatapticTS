import {
  NGE_EVOLUTION_DEFAULT_ALPHA_COMPUTATION as NGE_EVOLUTION_DEFAULT_ALPHA_COMPUTATION_IMPL,
  NGE_EVOLUTION_DEFAULT_ALPHA_LIFECYCLE as NGE_EVOLUTION_DEFAULT_ALPHA_LIFECYCLE_IMPL,
  NGE_EVOLUTION_DEFAULT_ALPHA_MEMORY as NGE_EVOLUTION_DEFAULT_ALPHA_MEMORY_IMPL,
  NGE_EVOLUTION_DEFAULT_ALPHA_TOPOLOGY as NGE_EVOLUTION_DEFAULT_ALPHA_TOPOLOGY_IMPL,
  NGE_EVOLUTION_DEFAULT_COMPATIBILITY_DISTANCE_WEIGHTS as NGE_EVOLUTION_DEFAULT_COMPATIBILITY_DISTANCE_WEIGHTS_IMPL,
  NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY as NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY_IMPL,
  NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION as NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION_IMPL,
  NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS as NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS_IMPL,
} from './neat.nge-evolution.constants';
import { computeNgeEvolutionCompatibilityDistance as computeNgeEvolutionCompatibilityDistanceImpl } from './neat.nge-evolution.distance';
import { applyNgeEvolutionEpigeneticPrior as applyNgeEvolutionEpigeneticPriorImpl } from './neat.nge-evolution.epigenetic';
import {
  NgeEvolution_BudgetError as NgeEvolution_BudgetErrorImpl,
  NgeEvolution_ModeError as NgeEvolution_ModeErrorImpl,
  NgeEvolution_RegionError as NgeEvolution_RegionErrorImpl,
} from './neat.nge-evolution.errors';
import {
  reproduceParthenogenesis as reproduceParthenogenesisImpl,
  reproducePolyandric as reproducePolyandricImpl,
  reproduceSexual as reproduceSexualImpl,
} from './neat.nge-evolution.reproduction';

export type {
  NgeEvolutionCompatibilityComparisonInput,
  NgeEvolutionCompatibilityDistanceContext,
  NgeEvolutionCompatibilityDistanceResult,
  NgeEvolutionCompatibilityDistanceTerm,
  NgeEvolutionCompatibilityDistanceTermName,
  NgeEvolutionCompatibilityDistanceTerms,
  NgeEvolutionCompatibilityDistanceWeights,
  NgeEvolutionCompatibilityGenomeInput,
  NgeEvolutionCompatibilityWiringCostWeights,
  NgeEvolutionContributionKind,
  NgeEvolutionEpigeneticPriorInput,
  NgeEvolutionEpigeneticPriorResult,
  NgeEvolutionEpigeneticReference,
  NgeEvolutionParentContribution,
  NgeEvolutionParentRole,
  NgeEvolutionPolyandricAssignedRegion,
  NgeEvolutionPolyandricRegionAssignmentResult,
  NgeEvolutionReproductionOutcome,
  NgeEvolutionReproductionResult,
} from './neat.nge-evolution.types';

/**
 * Public Phase E compatibility-distance entrypoint exposed from one stable owner-local facade.
 */
export const computeNgeEvolutionCompatibilityDistance =
  computeNgeEvolutionCompatibilityDistanceImpl;

/**
 * Public birth-time epigenetic prior entrypoint exposed from one stable owner-local facade.
 */
export const applyNgeEvolutionEpigeneticPrior =
  applyNgeEvolutionEpigeneticPriorImpl;

/**
 * Public parthenogenesis reproduction operator exposed from one stable owner-local facade.
 */
export const reproduceParthenogenesis = reproduceParthenogenesisImpl;

/**
 * Public polyandric reproduction operator exposed from one stable owner-local facade.
 */
export const reproducePolyandric = reproducePolyandricImpl;

/**
 * Public sexual reproduction operator exposed from one stable owner-local facade.
 */
export const reproduceSexual = reproduceSexualImpl;

/**
 * Default alpha weight for the classic NEAT topology-distance term used in speciation.
 */
export const NGE_EVOLUTION_DEFAULT_ALPHA_TOPOLOGY =
  NGE_EVOLUTION_DEFAULT_ALPHA_TOPOLOGY_IMPL;

/**
 * Default alpha weight for the NGE computation-motif distance term used in speciation.
 */
export const NGE_EVOLUTION_DEFAULT_ALPHA_COMPUTATION =
  NGE_EVOLUTION_DEFAULT_ALPHA_COMPUTATION_IMPL;

/**
 * Default alpha weight for the NGE memory-tier distance term used in speciation.
 */
export const NGE_EVOLUTION_DEFAULT_ALPHA_MEMORY =
  NGE_EVOLUTION_DEFAULT_ALPHA_MEMORY_IMPL;

/**
 * Default alpha weight for the NGE lifecycle-policy distance term used in speciation.
 */
export const NGE_EVOLUTION_DEFAULT_ALPHA_LIFECYCLE =
  NGE_EVOLUTION_DEFAULT_ALPHA_LIFECYCLE_IMPL;

/**
 * Default per-term alpha bag for callers that want the Phase E compatibility defaults.
 */
export const NGE_EVOLUTION_DEFAULT_COMPATIBILITY_DISTANCE_WEIGHTS =
  NGE_EVOLUTION_DEFAULT_COMPATIBILITY_DISTANCE_WEIGHTS_IMPL;

/**
 * Default weak-reference decay used by the optional epigenetic prior operator.
 */
export const NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY =
  NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY_IMPL;

/**
 * Default fraction of DNA regions that polyandric drone donors may patch.
 */
export const NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION =
  NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION_IMPL;

/**
 * Default queen-bias multiplier where `1.0` means queen data wins all conflicts.
 */
export const NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS =
  NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS_IMPL;

/**
 * Public error class thrown when one requested reproduction mode is unavailable.
 */
export const NgeEvolution_ModeError = NgeEvolution_ModeErrorImpl;

/**
 * Public error class thrown when one region-assignment request is invalid.
 */
export const NgeEvolution_RegionError = NgeEvolution_RegionErrorImpl;

/**
 * Public error class thrown when one operator exceeds the configured budget.
 */
export const NgeEvolution_BudgetError = NgeEvolution_BudgetErrorImpl;

/**
 * Default bundle for the nge-evolution owner boundary.
 *
 * Import this object when a caller wants the full runtime shelf for Phase E
 * from one stable owner-local path instead of stitching together leaf modules.
 */
const ngeEvolution = {
  computeNgeEvolutionCompatibilityDistance,
  applyNgeEvolutionEpigeneticPrior,
  reproduceParthenogenesis,
  reproducePolyandric,
  reproduceSexual,
  NGE_EVOLUTION_DEFAULT_ALPHA_TOPOLOGY,
  NGE_EVOLUTION_DEFAULT_ALPHA_COMPUTATION,
  NGE_EVOLUTION_DEFAULT_ALPHA_MEMORY,
  NGE_EVOLUTION_DEFAULT_ALPHA_LIFECYCLE,
  NGE_EVOLUTION_DEFAULT_COMPATIBILITY_DISTANCE_WEIGHTS,
  NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY,
  NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION,
  NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS,
  NgeEvolution_ModeError,
  NgeEvolution_RegionError,
  NgeEvolution_BudgetError,
};

export default ngeEvolution;
