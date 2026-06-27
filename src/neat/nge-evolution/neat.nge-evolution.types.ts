import type {
  NgeAssignedRegionStrategy,
  NgeDnaCanonicalEnvelope,
  NgeReproductionPolicy,
} from '../nge-dna/neat.nge-dna.types';

/**
 * Fixed term names used by the Phase E composite compatibility-distance calculator.
 */
export type NgeEvolutionCompatibilityDistanceTermName =
  'topology' | 'computation' | 'memory' | 'lifecycle';

/**
 * One weighted compatibility-distance term captured during Phase E speciation scoring.
 */
export interface NgeEvolutionCompatibilityDistanceTerm {
  /** Stable term label for telemetry and assertion surfaces. */
  name: NgeEvolutionCompatibilityDistanceTermName;
  /** Raw pre-normalization distance emitted by the term-specific calculator. */
  rawDistance: number;
  /** Min-max normalized distance contribution in the unit interval. */
  normalizedDistance: number;
  /** Alpha weight applied to the normalized distance. */
  weight: number;
  /** Final weighted contribution included in the composite distance. */
  weightedDistance: number;
}

/**
 * Alpha weights applied to the four Phase E compatibility-distance terms.
 */
export interface NgeEvolutionCompatibilityDistanceWeights {
  /** Alpha weight for the classic topology-distance term. */
  topology: number;
  /** Alpha weight for the computation-motif distance term. */
  computation: number;
  /** Alpha weight for the memory-tier distance term. */
  memory: number;
  /** Alpha weight for the lifecycle-policy distance term. */
  lifecycle: number;
}

/**
 * Wiring-cost preference knobs compared by the Phase E lifecycle-distance term.
 */
export interface NgeEvolutionCompatibilityWiringCostWeights {
  /** Relative penalty applied to realized module count. */
  nodeWeight?: number;
  /** Relative penalty applied to realized edge count. */
  edgeWeight?: number;
  /** Relative penalty applied to inter-zone or long-range edges. */
  interZonePenalty?: number;
}

/**
 * One genome-side input consumed by the Phase E composite compatibility calculator.
 *
 * The canonical DNA envelope does not yet own lifecycle cadence or wiring-preference
 * knobs, so the calculator accepts those traits as an owner-local sidecar.
 */
export interface NgeEvolutionCompatibilityGenomeInput {
  /** Canonical DNA envelope used for motif, memory, and reproduction comparisons. */
  dna: NgeDnaCanonicalEnvelope | null;
  /** Optional assimilation cadence sidecar compared by the lifecycle-distance term. */
  assimilationCadence?: number;
  /** Optional wiring-economy sidecar compared by the lifecycle-distance term. */
  wiringCostWeights?: NgeEvolutionCompatibilityWiringCostWeights;
}

/**
 * One pairwise comparison input evaluated by the Phase E composite compatibility calculator.
 */
export interface NgeEvolutionCompatibilityComparisonInput {
  /** Classic NEAT topology distance injected without recomputation. */
  topologyDistance: number;
  /** Whether the evaluated pair has NGE enabled for the current speciation path. */
  ngeEnabled: boolean;
  /** Left genome-side data for the comparison. */
  leftGenome: NgeEvolutionCompatibilityGenomeInput;
  /** Right genome-side data for the comparison. */
  rightGenome: NgeEvolutionCompatibilityGenomeInput;
}

/**
 * Context bag controlling one Phase E compatibility-distance computation and normalization scope.
 */
export interface NgeEvolutionCompatibilityDistanceContext {
  /** Optional population slice used for independent per-term min-max normalization. */
  populationSlice?: readonly NgeEvolutionCompatibilityComparisonInput[];
  /** Optional alpha weights overriding the Phase E defaults before normalization. */
  weights?: Partial<NgeEvolutionCompatibilityDistanceWeights>;
}

/**
 * Fully expanded term shelf returned by the Phase E compatibility-distance calculator.
 */
export interface NgeEvolutionCompatibilityDistanceTerms {
  /** Classic NEAT innovation-and-topology term. */
  topology: NgeEvolutionCompatibilityDistanceTerm;
  /** NGE computation-motif similarity term. */
  computation: NgeEvolutionCompatibilityDistanceTerm;
  /** NGE memory-tier similarity term. */
  memory: NgeEvolutionCompatibilityDistanceTerm;
  /** NGE lifecycle-policy similarity term. */
  lifecycle: NgeEvolutionCompatibilityDistanceTerm;
}

/**
 * Composite compatibility-distance result returned by the Phase E speciation helper.
 */
export interface NgeEvolutionCompatibilityDistanceResult {
  /** Weighted sum of the four normalized compatibility-distance terms. */
  distance: number;
  /** Whether the NGE-only term families were enabled for the evaluated pair. */
  ngeEnabled: boolean;
  /** Expanded per-term contributions used to build the composite distance. */
  terms: NgeEvolutionCompatibilityDistanceTerms;
}

/**
 * Parent-role labels used when Phase E operators report how one offspring was assembled.
 */
export type NgeEvolutionParentRole =
  'sole-parent' | 'queen' | 'drone' | 'primary' | 'secondary';

/**
 * High-level contribution kinds used to describe parent input at the reproduction boundary.
 */
export type NgeEvolutionContributionKind =
  'clone' | 'mutation' | 'patch' | 'crossover' | 'blend';

/**
 * One parent contribution reported by a Phase E reproduction operator.
 */
export interface NgeEvolutionParentContribution {
  /** Stable parent identifier used for lineage or telemetry lookups. */
  parentId: string;
  /** Role the parent played during offspring construction. */
  role: NgeEvolutionParentRole;
  /** High-level contribution kind attributed to the parent. */
  contributionKind: NgeEvolutionContributionKind;
  /** DNA region identifiers touched by the parent contribution. */
  regionIds: string[];
}

/**
 * Region-assignment record for one drone's patching contribution in polyandric offspring reproduction.
 */
export interface NgeEvolutionPolyandricAssignedRegion {
  /** Stable DNA-region identifier delegated to one drone. */
  regionId: string;
  /** Stable drone identifier assigned to patch the region. */
  droneId: string;
  /** Deterministic donor rank captured at assignment time. */
  droneRank: number;
  /** Optional donor fitness snapshot used by the `byFitness` strategy. */
  droneFitness?: number;
  /** Optional specialization tag used by the `bySpecialization` strategy. */
  specializationKey?: string;
}

/**
 * Polyandric region-assignment result reported before any drone patches are applied to offspring.
 */
export interface NgeEvolutionPolyandricRegionAssignmentResult {
  /** Strategy reused from the DNA reproduction-policy shelf. */
  strategy: NgeAssignedRegionStrategy;
  /** Region identifiers deemed patchable under the configured contribution cap. */
  patchableRegionIds: string[];
  /** Non-overlapping drone assignments resolved for the patchable regions. */
  assignedRegions: NgeEvolutionPolyandricAssignedRegion[];
  /** Patchable regions left untouched after assignment finishes. */
  unassignedRegionIds: string[];
}

/**
 * Two-parent weak reference captured for one birth-time epigenetic prior update.
 */
export interface NgeEvolutionEpigeneticReference<
  ParameterVector extends readonly number[] = readonly number[],
> {
  /** Parameter vector read from the first parent reference. */
  firstParentParameters: ParameterVector;
  /** Parameter vector read from the second parent reference. */
  secondParentParameters: ParameterVector;
  /** Blended reference vector derived from the two parents. */
  blendedReference: ParameterVector;
}

/**
 * Input contract consumed by the optional Phase E epigenetic prior operator.
 */
export interface NgeEvolutionEpigeneticPriorInput<
  ParameterVector extends readonly number[] = readonly number[],
> {
  /** Current child parameter vector before the epigenetic nudge applies. */
  childParameters: ParameterVector;
  /** Mutation delta already prepared for the child before the weak prior is applied. */
  mutationDelta: ParameterVector;
  /** Optional two-parent weak reference for birth-time nudging. */
  reference: NgeEvolutionEpigeneticReference<ParameterVector> | null;
  /** Optional decay override used instead of the module default when present. */
  decay?: number;
}

/**
 * Output contract returned by the optional Phase E epigenetic prior operator.
 */
export interface NgeEvolutionEpigeneticPriorResult<
  ParameterVector extends readonly number[] = readonly number[],
> {
  /** Final child parameter vector after mutation and any weak-reference decay. */
  outputParameters: ParameterVector;
  /** Decay factor actually applied during the update. */
  appliedDecay: number;
  /** Whether a weak epigenetic reference was present and used. */
  referenceApplied: boolean;
  /** Blended reference vector used for the nudge, or null when skipped. */
  blendedReference: ParameterVector | null;
}

/**
 * High-level offspring outcome labels surfaced by the Phase E reproduction operators.
 */
export type NgeEvolutionReproductionOutcome =
  'clone' | 'mutation-only' | 'queen-template-patched' | 'sexual-crossover';

/**
 * Shared reproduction result returned by all three Phase E reproduction modes.
 */
export interface NgeEvolutionReproductionResult<
  OffspringEnvelope = NgeDnaCanonicalEnvelope,
  ParameterVector extends readonly number[] = readonly number[],
> {
  /** Resolved reproduction policy that selected the operator path. */
  policy: NgeReproductionPolicy;
  /** High-level summary of how the offspring was assembled. */
  outcome: NgeEvolutionReproductionOutcome;
  /** Offspring envelope emitted by the operator. */
  offspring: OffspringEnvelope;
  /** Parent-level contribution records preserved for tests and telemetry. */
  parentContributions: NgeEvolutionParentContribution[];
  /** Polyandric region-assignment details when the operator used drone patching. */
  regionAssignment: NgeEvolutionPolyandricRegionAssignmentResult | null;
  /** Optional weak epigenetic reference prepared for the birth-time nudge step. */
  epigeneticReference: NgeEvolutionEpigeneticReference<ParameterVector> | null;
}
