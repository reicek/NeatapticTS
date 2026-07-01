/**
 * Evolution operators for the NGE (Neuro-evolutionary Genesis Engine) extension.
 *
 * This boundary owns the three reproduction modes — parthenogenesis, polyandric,
 * and sexual crossover — plus the compatibility-distance and epigenetic-prior
 * helpers that sit next to them. Callers typically import the stable facade
 * exports rather than the leaf modules.
 *
 * The polyandric operator is the multi-parent recombination path: a queen DNA
 * template receives patch contributions from a small set of drone donors. The
 * {@link NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION} caps how
 * many regions may be patched, and the {@link NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS}
 * gate decides, per region, whether the queen or the drone wins the merge.
 * A deterministic FNV-1a hash of the region identifier converts the bias value
 * into a repeatable threshold, so the same queen/drone pair and policy always
 * produce the same offspring.
 *
 * ```mermaid
 * flowchart LR
 *   Queen["Queen DNA template"] --> Cap{"Cap patchable<br/>regions by fraction"}
 *   Cap --> Assign["Assign drones to regions"]
 *   Assign --> Gate{"FNV-1a hash of regionId<br/>vs queenBias"}
 *   Gate -->|queen wins| Keep["Keep queen region"]
 *   Gate -->|drone wins| Patch["Patch drone region"]
 *   Keep --> Offspring["Canonical offspring"]
 *   Patch --> Offspring
 * ```
 *
 * Background reading: the NEAT algorithm
 * ([Stanley & Miikkulainen (2002)](https://nn.cs.utexas.edu/?stanley:ec02)),
 * polyandry in evolutionary biology
 * ([Wikipedia — Polyandry](https://en.wikipedia.org/wiki/Polyandry)) and the
 * FNV-1a hash function
 * ([Wikipedia — Fowler–Noll–Vo hash function](https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function)).
 */

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

export type {
  NgePolyandricInput,
  NgePolyandricDroneInput,
} from './neat.nge-evolution.reproduction';

/**
 * Public NGE evolution compatibility-distance entrypoint exposed from one stable owner-local facade.
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
 *
 * Builds a queen-template offspring patched by a capped set of drone donors.
 * The per-region winner is decided by a deterministic FNV-1a hash of the
 * region id compared against the resolved `queenBias`; values below the bias
 * keep the queen region, values above it patch the drone region. See the
 * module introduction for the full pipeline diagram.
 *
 * @example
 * ```ts
 * const offspring = reproducePolyandric({
 *   ngeEnabled: true,
 *   queen: queenEnvelope,
 *   queenId: 'queen-1',
 *   drones: [{ dna: donorEnvelope, parentId: 'drone-a', fitness: 0.9 }],
 * });
 * ```
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
 * Default per-term alpha bag for callers that want the NGE evolution compatibility defaults.
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
 *
 * A value of `0.1` means only the first 10% of the queen's patchable regions
 * (rounded up) are exposed to drone contributions.
 */
export const NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION =
  NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION_IMPL;

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
 * Import this object when a caller wants the full runtime shelf for the NGE
 * evolution extension from one stable owner-local path instead of stitching
 * together leaf modules.
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
