/**
 * Canonical NGE DNA envelope schema.
 *
 * This file owns the TypeScript schema for the deterministic Neuro-Genesis
 * Engine (NGE) DNA envelope. Every exported type here describes a field that
 * is serialized inside the canonical envelope or accepted at a constructor/
 * runtime boundary.
 *
 * ## Input shorthand vs. canonical envelope
 *
 * The core NGE boundary accepts a small set of input shorthand values at
 * input time, but the canonical envelope always stores the expanded object
 * shape. This keeps external call sites terse while guaranteeing that
 * serialization, hashing, and round-trips always see the same canonical
 * structure.
 *
 * - {@link NgeAssignedRegionStrategy} accepts `'non-overlapping'` as input
 *   shorthand for the deterministic single-drone-per-region assignment that the
 *   core already implements under `'roundRobin'`.
 * - {@link NgeSeedPolicyShorthand} `'queen-weighted'` expands to the canonical
 *   `{ siblingsDifferBySeed: true, twinsAllowed: false }` object inside the
 *   `NGE_DNA` constructor.
 *
 * This design follows the envelope-normalization contract: core accepts
 * shorthand values at input and keeps a canonical shape internally.
 */
import type {
  NeatGenomeComputationType,
  NeatGenomeSubstrateCoordinate,
} from '../genome/genome.types';

/**
 * Branded schema version tag carried by the canonical NGE DNA envelope.
 */
export type NgeSchemaVersion = string & {
  readonly __brand: 'NgeSchemaVersion';
};

/**
 * Encoding modes supported by the canonical NGE DNA envelope during serialization.
 * `lossless` preserves all structural data; `lossy` is an opt-in flag for extreme-scale compression.
 */
export type NgeEncodingMode = 'lossless' | 'lossy';

/**
 * Identity fields that make one NGE DNA envelope self-describing and hashable.
 */
export interface NgeIdentityFields {
  /** Version tag for schema migration decisions. */
  schemaVersion: NgeSchemaVersion;
  /** Independent compatibility contract version for future runtime bridges. */
  compatibilityVersion: string;
  /** Serialization mode recorded by the envelope. */
  encodingMode: NgeEncodingMode;
  /** Deterministic SHA-256 hex fingerprint of the canonical envelope content. */
  fingerprint: string;
}

/**
 * Supported reproduction modes tracked by the NGE DNA policy shelf.
 * Governs whether offspring arise from a single parent, multiple drone donors, or sexual crossover.
 */
export type NgeReproductionPolicyMode =
  'parthenogenesis' | 'polyandric' | 'sexual';

/**
 * Region-assignment strategies used for polyandric drone patch selection among donors.
 * Determines how the queen distributes writable DNA regions among secondary drone contributors.
 *
 * ## Input shorthand compatibility
 *
 * `'non-overlapping'` is input shorthand for the deterministic
 * single-drone-per-region assignment that the core already implements under
 * `'roundRobin'`. Both values resolve to identical behavior; only the canonical
 * string stored in the envelope differs.
 */
export type NgeAssignedRegionStrategy =
  'roundRobin' | 'byFitness' | 'bySpecialization' | 'non-overlapping';

/**
 * Supported shorthand tokens for seed-governance policy at constructor/runtime boundaries.
 * The canonical envelope always stores the expanded object shape.
 *
 * `'queen-weighted'` expands to `{ siblingsDifferBySeed: true, twinsAllowed: false }`
 * during `NGE_DNA` construction. The normalized object is what the canonical
 * envelope serializes, so round-trips and fingerprints remain stable regardless of
 * whether the caller passed the shorthand or the full object.
 */
export type NgeSeedPolicyShorthand = 'queen-weighted';

/**
 * Seed governance toggles controlling sibling divergence and identical twin generation.
 * Determines whether offspring sharing the same DNA diverge by seed or remain exact replicas.
 */
export interface NgeSeedPolicy {
  /** Whether otherwise identical siblings diverge by seed by default. */
  siblingsDifferBySeed: boolean;
  /** Whether exact DNA-plus-seed twins are allowed. */
  twinsAllowed: boolean;
}

/**
 * Fully resolved reproduction policy stored in the canonical DNA envelope.
 */
export interface NgeReproductionPolicy {
  /** Active reproduction mode for the lineage. */
  mode: NgeReproductionPolicyMode;
  /** Mutation rate applied to parthenogenetic offspring where `0` means a true clone. */
  parthenogenesisMutationRate: number;
  /** Number of secondary drone donors in polyandric mode. */
  polyandricDroneCount: number;
  /** Fraction of DNA regions that drones may patch in polyandric mode. */
  polyandricDroneContributionFraction: number;
  /** Bias toward queen dominance where `1` means queen wins all conflicts. */
  queenBias: number;
  /** Deterministic strategy for assigning patchable DNA regions. */
  assignedRegionStrategy: NgeAssignedRegionStrategy;
  /** Whether the reproduction mode itself is evolvable. */
  modeIsEvolvable: boolean;
  /** Seed-governance policy for siblings and exact twins. */
  seedPolicy: NgeSeedPolicy;
}

/**
 * Policy input accepted at reproduction operator boundaries. Identical to the
 * canonical {@link NgeReproductionPolicy} but allows the seed-governance shelf
 * to be supplied as the {@link NgeSeedPolicyShorthand} `'queen-weighted'`, which
 * is expanded to the canonical object before the policy is stored in the
 * envelope.
 *
 * This follows the envelope-normalization contract: core accepts shorthand values
 * at input and keeps a canonical shape internally.
 */
export type NgeReproductionPolicyInput = Omit<
  NgeReproductionPolicy,
  'seedPolicy'
> & {
  seedPolicy: NgeSeedPolicy | NgeSeedPolicyShorthand;
};

/**
 * Optional substrate budget override for callers that supply stricter node/edge limits than the defaults.
 * When present, clamps the maximum node and edge counts allowed for one materialized substrate.
 */
export interface NgeSubstrateBudgetOverride {
  /** Maximum allowed module count for one materialized substrate. */
  maxNodes?: number;
  /** Maximum allowed edge count for one materialized substrate. */
  maxEdges?: number;
}

/**
 * Zone partitioning configuration for one substrate axis in the unit-cube grid.
 * Defines the equal-sized partition count used to assign deterministic zone cells during module placement.
 */
export interface NgeAxisPartitionConfig {
  /** Number of equal-sized partitions across one substrate axis. */
  count: number;
}

/**
 * Full zone partition configuration for all three substrate axes of the unit cube.
 * Determines how the unit-cube space is divided into deterministic cells for module placement.
 */
export interface NgeZonePartitionConfig {
  /** X-axis partition configuration. */
  x: NgeAxisPartitionConfig;
  /** Y-axis partition configuration. */
  y: NgeAxisPartitionConfig;
  /** Z-axis partition configuration. */
  z: NgeAxisPartitionConfig;
}

/**
 * Resolved zone descriptor for one deterministic unit-cube cell in the substrate grid.
 * Carries the integer-grid–derived `zoneId` and the inclusive coordinate bounds for the cell.
 */
export interface NgeSubstrateZone {
  /** Deterministic zone identifier derived from integer grid indices. */
  zoneId: string;
  /** Inclusive lower and upper bounds for the zone cell. */
  bounds: {
    /** Inclusive lower X bound for the zone. */
    xMin: number;
    /** Inclusive upper X bound for the zone. */
    xMax: number;
    /** Inclusive lower Y bound for the zone. */
    yMin: number;
    /** Inclusive upper Y bound for the zone. */
    yMax: number;
    /** Inclusive lower Z bound for the zone. */
    zMin: number;
    /** Inclusive upper Z bound for the zone. */
    zMax: number;
  };
}

/**
 * Substrate contract governing the deterministic development boundary for the NGE phenotype.
 * Fixes the three-axis unit-cube geometry and zone-partition scheme used during module placement.
 */
export interface NgeSubstrateConfig {
  /** Fixed three-axis dimensionality for the unit-cube substrate. */
  dimensions: 3;
  /** Fixed unit-cube normalization contract for development-time placement. */
  normalization: 'unit-cube';
  /** Zone-partition metadata used to assign deterministic unit-cube cells. */
  zonePartition: NgeZonePartitionConfig;
  /** Optional extension field reserved for future substrate budget governance. */
  budgetOverride?: NgeSubstrateBudgetOverride;
}

/**
 * Supported rule-pass kinds recognized by the deterministic NGE rule executor.
 * Each kind implies a distinct geometry strategy applied to the module placement list during development.
 */
export type NgeRulePassKind =
  'replicate' | 'symmetry' | 'hierarchy' | 'differentiate';

/**
 * One requested placement emitted by a DNA rule pass during deterministic development.
 * Specifies the unit-cube coordinate and computation motif assigned to one future virtual module.
 */
export interface NgeRulePlacement {
  /** Requested substrate coordinate for the future module. */
  coordinate: NeatGenomeSubstrateCoordinate;
  /** Computation motif assigned to the placed virtual module. */
  computationType: NeatGenomeComputationType;
}

/**
 * One deterministic rule-pass record carried by the NGE DNA envelope.
 * Encodes a family of module placements that the rule executor unfolds deterministically during development.
 */
export interface NgeRulePass {
  /** Rule-pass family whose later geometry semantics remain deferred. */
  kind: NgeRulePassKind;
  /** Stable archetype identity referenced by the placements. */
  archetypeId: string;
  /** Canonical execution priority where lower runs earlier. */
  priority: number;
  /** Requested module placements emitted by the pass. */
  placements: NgeRulePlacement[];
}

/**
 * CPPN activation families supported by the canonical deterministic phenotype evaluator.
 */
export type NgeCppnActivationKind =
  'linear' | 'tanh' | 'sigmoid' | 'gaussian' | 'sine';

/**
 * One explicit non-input CPPN node.
 *
 * Output-node descriptors may appear here to override the default linear,
 * zero-bias output nodes implied by the canonical output ids.
 */
export interface NgeCppnNode {
  /** Stable node identifier used by CPPN edges. */
  nodeId: string;
  /** Activation family applied after the weighted sum plus bias. */
  activationKind: NgeCppnActivationKind;
  /** Additive node bias applied before the activation function. */
  bias: number;
}

/**
 * Loose constructor input for one explicit non-input CPPN node in a program descriptor.
 * Omitted activation and bias fields resolve to conservative defaults during CPPN program construction.
 */
export type NgeCppnNodeInput = Pick<NgeCppnNode, 'nodeId'> &
  Partial<Omit<NgeCppnNode, 'nodeId'>>;

/**
 * One directed weighted connection inside the CPPN topology graph.
 * During adjacency realization, the weight is forwarded to the corresponding realized edge descriptor.
 */
export interface NgeCppnEdge {
  /** Source node id whose value feeds the target. */
  sourceNodeId: string;
  /** Target non-input node id whose accumulator receives the weighted source value. */
  targetNodeId: string;
  /** Connection weight applied during the forward pass. */
  weight: number;
}

/**
 * Loose constructor input for one directed weighted CPPN connection in a program descriptor.
 * Omitted `weight` fields resolve to zero during canonical CPPN edge construction.
 */
export type NgeCppnEdgeInput = Pick<
  NgeCppnEdge,
  'sourceNodeId' | 'targetNodeId'
> &
  Partial<Pick<NgeCppnEdge, 'weight'>>;

/**
 * Full canonical CPPN program descriptor carried by the NGE DNA envelope.
 * Evaluated during phenotype materialization to derive sparse adjacency between realized modules.
 */
export interface NgeCppnProgram {
  /** Stable program identifier inside the DNA envelope. */
  programId: string;
  /** Fixed canonical input ids for the NGE CPPN evaluator. */
  inputNodeIds: string[];
  /** Fixed canonical output ids for the NGE CPPN evaluator. */
  outputNodeIds: string[];
  /** Explicit non-input nodes keyed by `nodeId`. */
  hiddenNodes: NgeCppnNode[];
  /** Directed weighted topology connecting inputs and non-input nodes. */
  edges: NgeCppnEdge[];
}

/**
 * Loose constructor input for one canonical CPPN program descriptor for the NGE CPPN evaluator.
 * Omitted fields resolve to canonical defaults; omitted edge weights resolve to zero.
 */
export interface NgeCppnProgramInput {
  /** Stable program identifier inside the DNA envelope. */
  programId: string;
  /** Optional explicit input ids overriding the canonical defaults. */
  inputNodeIds?: readonly string[];
  /** Optional explicit output ids overriding the canonical defaults. */
  outputNodeIds?: readonly string[];
  /** Optional non-input node list whose omitted fields resolve conservatively. */
  hiddenNodes?: readonly NgeCppnNodeInput[];
  /** Optional weighted edge list whose omitted weights resolve to zero. */
  edges?: readonly NgeCppnEdgeInput[];
}

/**
 * DNA-level module archetype registry entry consumed during the phenotype realization pass.
 * Associates a computation motif with optional coordinate injection and weight-shared cohort membership.
 */
export interface NgeDnaModuleArchetype {
  /** Stable archetype identity referenced by rule-pass placements. */
  archetypeId: string;
  /** Computation motif associated with the archetype. */
  computationType: NeatGenomeComputationType;
  /** Whether materialized modules derived from this archetype receive substrate coordinates. */
  receivesCoordinates?: boolean;
  /** Stream identity when the archetype participates in one shared residual stream. */
  residualStreamId?: string;
  /** Cohort identity when the archetype participates in one weight-shared shelf. */
  weightSharedCohortId?: string;
  /** Optional archetype-local governance parameters forwarded into the realized descriptor. */
  parameterSchema?: Record<string, unknown>;
}

/**
 * Loose constructor input for one DNA-level module archetype registry entry.
 */
export type NgeDnaModuleArchetypeInput = Pick<
  NgeDnaModuleArchetype,
  'archetypeId' | 'computationType'
> &
  Partial<Omit<NgeDnaModuleArchetype, 'archetypeId' | 'computationType'>>;

/**
 * One placed module inside the in-memory deterministic virtual module plan.
 * Carries the archetype identity, computation motif, unit-cube coordinate, and zone assignment.
 */
export interface NgeVirtualModule {
  /** Stable deterministic module identifier derived from the canonical pass order. */
  moduleId: string;
  /** Stable archetype identity referenced by the source rule pass. */
  archetypeId: string;
  /** Computation motif assigned to the placed module. */
  computationType: NeatGenomeComputationType;
  /** Normalized unit-cube coordinate for the placed module. */
  coordinate: NeatGenomeSubstrateCoordinate;
  /** Deterministic zone identifier derived from the normalized coordinate. */
  zoneId: string;
  /** Canonical sorted rule-pass index for the source rule pass. */
  rulePassIndex: number;
  /** Placement index within the source rule pass. */
  placementOrdinal: number;
}

/**
 * Deterministic in-memory module-placement plan produced by NGE rule execution.
 * Consumed by the phenotype materialization pass and validated via a canonical plan fingerprint.
 */
export interface NgeVirtualModulePlan {
  /** Stable ordered list of virtual modules emitted by the rule executor. */
  modules: NgeVirtualModule[];
  /** SHA-256 fingerprint of the canonical substrate configuration. */
  substrateFingerprint: string;
  /** SHA-256 fingerprint of the canonical module list plus seed. */
  planFingerprint: string;
}

/**
 * One realized module emitted by the phenotype materialization pass for the genome.
 */
export interface NgeRealizedModule {
  /** Stable deterministic module identifier derived from the canonical pass order. */
  moduleId: string;
  /** Stable archetype identity referenced by the source rule pass. */
  archetypeId: string;
  /** Realized computation motif validated against the public computation-type catalogue. */
  computationType: NeatGenomeComputationType;
  /** Normalized unit-cube coordinate forwarded from the virtual module plan. */
  coordinate: NeatGenomeSubstrateCoordinate;
  /** Deterministic zone identifier derived from the normalized coordinate. */
  zoneId: string;
  /** Whether downstream runtime materialization should inject the coordinate vector. */
  receivesCoordinates: boolean;
  /** Stream identity when the realized module participates in one residual stream. */
  residualStreamId?: string;
  /** Cohort identity when the realized module participates in one weight-shared shelf. */
  weightSharedCohortId?: string;
  /** Optional archetype-local governance parameters forwarded into the descriptor. */
  archetypeParams?: Record<string, unknown>;
}

/**
 * One realized directed adjacency edge emitted by the CPPN evaluation pass.
 * Carries wiring cost, residual-tap, and broadcast flags for downstream budget-aware stages.
 */
export interface NgeRealizedEdge {
  /** Source realized-module identifier. */
  sourceModuleId: string;
  /** Target realized-module identifier. */
  targetModuleId: string;
  /** Raw CPPN output weight retained for downstream materialization. */
  weight: number;
  /** Whether the edge is exempt from wiring cost because the source taps a residual stream. */
  isResidualTap: boolean;
  /** Whether the edge is exempt from wiring cost because the source broadcasts within radius. */
  isModulatorBroadcast: boolean;
  /** Euclidean wiring cost retained for later budget-aware stages. */
  wiringCost: number;
}

/**
 * Fully JSON-serializable realized phenotype descriptor produced at the end of phenotype materialization.
 */
export interface NgeRealizedPhenotypeDescriptor {
  /** Stable ordered list of realized modules. */
  modules: NgeRealizedModule[];
  /** Stable ordered list of realized directed edges. */
  edges: NgeRealizedEdge[];
  /** Residual-stream assignments keyed by stream id. */
  residualStreamAssignments: Record<string, string[]>;
  /** Weight-shared cohort assignments keyed by cohort id. */
  weightSharedCohortAssignments: Record<string, string[]>;
  /** SHA-256 fingerprint of the canonical realized content plus seed. */
  phenotypeFingerprint: string;
  /** Fingerprint of the source DNA envelope used to build the descriptor. */
  dnaFingerprint: string;
  /** Deterministic seed folded into the realized fingerprint. */
  seed: number;
}

/**
 * Full canonical NGE DNA envelope serialized by the owner-local module.
 */
export type NgeDnaCanonicalEnvelope = NgeIdentityFields & {
  /** Fully resolved substrate metadata for deterministic development. */
  substrate: NgeSubstrateConfig;
  /** Resolved reproduction policy recorded by the DNA shelf. */
  reproductionPolicy: NgeReproductionPolicy;
  /** Deterministic rule-pass records used to build the virtual module plan. */
  rulePasses: NgeRulePass[];
  /** Canonical CPPN programs used to realize sparse adjacency. */
  cppnPrograms: NgeCppnProgram[];
  /** Canonical module-archetype registry used during realization. */
  moduleArchetypes: NgeDnaModuleArchetype[];
};
