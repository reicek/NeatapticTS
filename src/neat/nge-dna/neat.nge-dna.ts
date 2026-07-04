/**
 * Deterministic owner-local DNA boundary for NGE schema identity and canonical encoding.
 *
 * `NGE_DNA` builds and stores the canonical {@link NgeDnaCanonicalEnvelope}.
 * Its constructor accepts loose input where omitted fields resolve to
 * deterministic defaults, and it expands input shorthand values into the
 * canonical envelope shape before serialization.
 *
 * ## Shorthand normalization
 *
 * The constructor accepts `reproductionPolicy.seedPolicy: 'queen-weighted'` as
 * a shorthand for the canonical seed-governance object
 * `{ siblingsDifferBySeed: true, twinsAllowed: false }`. The normalized object
 * is what the envelope stores, so {@link NGE_DNA.toCanonical | toCanonical()}
 * and {@link NGE_DNA.serialize | serialize()} always emit the same shape
 * regardless of how the policy was originally expressed.
 *
 * This normalization follows the envelope-normalization contract: core accepts
 * shorthand values at input and keeps a canonical shape internally.
 */
import {
  NGE_DNA_DEFAULT_BUDGET_MAX_EDGES,
  NGE_DNA_DEFAULT_BUDGET_MAX_NODES,
  NGE_DNA_DEFAULT_RULE_PRIORITY,
  NGE_DNA_DEFAULT_ZONE_PARTITION_COUNT,
  NGE_DNA_SCHEMA_VERSION,
} from './neat.nge-dna.constants';
import {
  NGE_DNA_BudgetError,
  NGE_DNA_SchemaError,
  NGE_DNA_SubstrateError,
} from './neat.nge-dna.errors';
import { realizePhenotypeFromPlan } from './neat.nge-dna.realize';
import { executeRulePasses } from './neat.nge-dna.rules';
import {
  canonicalSerialize,
  computeFingerprint,
  validateIdentity,
} from './neat.nge-dna.utils';
import type {
  NgeAxisPartitionConfig,
  NgeCppnActivationKind,
  NgeCppnProgram,
  NgeCppnProgramInput,
  NgeDnaCanonicalEnvelope,
  NgeDnaModuleArchetype,
  NgeDnaModuleArchetypeInput,
  NgeEncodingMode,
  NgeRealizedPhenotypeDescriptor,
  NgeReproductionPolicy,
  NgeRulePass,
  NgeRulePlacement,
  NgeSeedPolicy,
  NgeSeedPolicyShorthand,
  NgeSubstrateConfig,
  NgeVirtualModulePlan,
  NgeZonePartitionConfig,
} from './neat.nge-dna.types';
import type { NeatGenomeSubstrateCoordinate } from '../genome/genome.types';

type NgeZonePartitionInput = Partial<
  Record<keyof NgeZonePartitionConfig, Partial<NgeAxisPartitionConfig>>
>;

type NgeSubstrateInput = Partial<Omit<NgeSubstrateConfig, 'zonePartition'>> & {
  zonePartition?: NgeZonePartitionInput;
};

type NgeRulePassInput = Omit<NgeRulePass, 'placements' | 'priority'> & {
  placements?: readonly NgeRulePlacement[];
  priority?: number;
};

type NgeDnaConstructorInput = Partial<
  Omit<
    NgeDnaCanonicalEnvelope,
    | 'cppnPrograms'
    | 'moduleArchetypes'
    | 'reproductionPolicy'
    | 'rulePasses'
    | 'substrate'
  >
> & {
  cppnPrograms?: readonly NgeCppnProgramInput[];
  moduleArchetypes?: readonly NgeDnaModuleArchetypeInput[];
  reproductionPolicy?: Partial<Omit<NgeReproductionPolicy, 'seedPolicy'>> & {
    seedPolicy?:
      Partial<NgeReproductionPolicy['seedPolicy']> | NgeSeedPolicyShorthand;
  };
  substrate?: NgeSubstrateInput;
  rulePasses?: readonly NgeRulePassInput[];
};

const DEFAULT_COMPATIBILITY_VERSION = '1.0.0';
const DEFAULT_PARTHENOGENESIS_MUTATION_RATE = 0.01;
const DEFAULT_POLYANDRIC_DRONE_COUNT = 2;
const DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION = 0.1;
const DEFAULT_QUEEN_BIAS = 1;
const DEFAULT_ASSIGNED_REGION_STRATEGY = 'roundRobin';
const DEFAULT_MODE_IS_EVOLVABLE = false;
const DEFAULT_SIBLINGS_DIFFER_BY_SEED = true;
const DEFAULT_TWINS_ALLOWED = false;
const DEFAULT_CPPN_INPUT_NODE_IDS = [
  'x1',
  'y1',
  'z1',
  'x2',
  'y2',
  'z2',
  'dist',
] as const;
const DEFAULT_CPPN_OUTPUT_NODE_IDS = ['weight', 'enableBias'] as const;
const DEFAULT_CPPN_ACTIVATION_KIND: NgeCppnActivationKind = 'linear';
const DEFAULT_CPPN_NODE_BIAS = 0;

/**
 * Deterministic owner-local DNA boundary for NGE schema identity and canonical encoding.
 */
export class NGE_DNA {
  #canonicalEnvelope: NgeDnaCanonicalEnvelope;

  /**
   * @param input - Partial DNA envelope whose omitted fields resolve to deterministic defaults.
   */
  constructor(input: NgeDnaConstructorInput = {}) {
    const resolvedEnvelope = resolveCanonicalEnvelope(input);
    this.#canonicalEnvelope = withComputedFingerprint(resolvedEnvelope);
  }

  /**
   * Schema version carried by the canonical envelope.
   */
  get schemaVersion() {
    return this.#canonicalEnvelope.schemaVersion;
  }

  /**
   * Compatibility contract version carried by the canonical envelope.
   */
  get compatibilityVersion() {
    return this.#canonicalEnvelope.compatibilityVersion;
  }

  /**
   * Encoding mode recorded in the canonical envelope.
   */
  get encodingMode() {
    return this.#canonicalEnvelope.encodingMode;
  }

  /**
   * Deterministic SHA-256 fingerprint of the canonical envelope content.
   */
  get fingerprint() {
    return this.#canonicalEnvelope.fingerprint;
  }

  /**
   * Fully resolved substrate configuration carried by the canonical envelope.
   */
  get substrate(): NgeSubstrateConfig {
    return structuredClone(this.#canonicalEnvelope.substrate);
  }

  /**
   * Canonical CPPN programs carried by the DNA envelope.
   */
  get cppnPrograms(): NgeCppnProgram[] {
    return structuredClone(this.#canonicalEnvelope.cppnPrograms);
  }

  /**
   * Canonical module archetype registry carried by the DNA envelope.
   */
  get moduleArchetypes(): NgeDnaModuleArchetype[] {
    return structuredClone(this.#canonicalEnvelope.moduleArchetypes);
  }

  /**
   * Materialize the full canonical envelope for serialization or inspection.
   *
   * @returns Deeply copied canonical DNA envelope.
   */
  toCanonical(): NgeDnaCanonicalEnvelope {
    return structuredClone(this.#canonicalEnvelope);
  }

  /**
   * Serialize the canonical envelope into key-sorted JSON.
   *
   * @returns Stable JSON representation of the DNA envelope.
   */
  serialize(): string {
    return canonicalSerialize(this.toCanonical());
  }

  /**
   * Recompute the fingerprint from the current canonical content.
   *
   * @returns Updated SHA-256 fingerprint.
   */
  recomputeFingerprint(): string {
    const nextFingerprint = computeFingerprint(
      serializeFingerprintSource(this.#canonicalEnvelope),
    );

    this.#canonicalEnvelope = {
      ...this.#canonicalEnvelope,
      fingerprint: nextFingerprint,
    };

    return nextFingerprint;
  }

  /**
   * Build the deterministic virtual module plan for one seed.
   *
   * @param seed - Deterministic seed folded into the plan fingerprint.
   * @returns Stable in-memory virtual module plan for the current DNA envelope.
   */
  buildVirtualPlan(seed: number): NgeVirtualModulePlan {
    return executeRulePasses(
      this.#canonicalEnvelope.rulePasses,
      this.substrate,
      seed,
    );
  }

  /**
   * Materialize one serializable phenotype descriptor from the current DNA envelope.
   *
   * @param plan - Deterministic virtual module plan to realize.
   * @param seed - Deterministic seed folded into the phenotype fingerprint.
   * @returns Fully serializable realized phenotype descriptor.
   */
  realizePhenotype(
    plan: NgeVirtualModulePlan,
    seed: number,
  ): NgeRealizedPhenotypeDescriptor {
    return realizePhenotypeFromPlan(plan, this.toCanonical(), seed);
  }

  /**
   * Construct one DNA instance from canonical envelope data.
   *
   * @param envelope - Canonical envelope whose identity and fingerprint must already be valid.
   * @returns New deterministic DNA instance.
   */
  static fromCanonical(envelope: NgeDnaCanonicalEnvelope): NGE_DNA {
    validateIdentity(envelope);
    validateFingerprintMatchesEnvelope(envelope);
    return new NGE_DNA(envelope);
  }

  /**
   * Deserialize one canonical JSON payload into a DNA instance.
   *
   * @param json - Canonical JSON payload.
   * @returns New deterministic DNA instance.
   */
  static deserialize(json: string): NGE_DNA {
    return NGE_DNA.fromCanonical(JSON.parse(json) as NgeDnaCanonicalEnvelope);
  }
}

function resolveCanonicalEnvelope(
  input: NgeDnaConstructorInput,
): NgeDnaCanonicalEnvelope {
  return {
    schemaVersion: input.schemaVersion ?? NGE_DNA_SCHEMA_VERSION,
    compatibilityVersion:
      input.compatibilityVersion ?? DEFAULT_COMPATIBILITY_VERSION,
    cppnPrograms: resolveCppnPrograms(input.cppnPrograms),
    encodingMode: resolveEncodingMode(input.encodingMode),
    fingerprint: '',
    moduleArchetypes: resolveModuleArchetypes(input.moduleArchetypes),
    reproductionPolicy: resolveReproductionPolicy(input.reproductionPolicy),
    rulePasses: resolveRulePasses(input.rulePasses),
    substrate: resolveSubstrateConfig(input.substrate),
  };
}

function resolveEncodingMode(
  encodingMode: NgeEncodingMode | undefined,
): NgeEncodingMode {
  return encodingMode === 'lossy' ? 'lossy' : 'lossless';
}

function resolveReproductionPolicy(
  reproductionPolicyInput: NgeDnaConstructorInput['reproductionPolicy'],
): NgeReproductionPolicy {
  const defaultSeedPolicy = {
    siblingsDifferBySeed: DEFAULT_SIBLINGS_DIFFER_BY_SEED,
    twinsAllowed: DEFAULT_TWINS_ALLOWED,
  };
  const resolvedSeedPolicy = resolveSeedPolicy(
    reproductionPolicyInput?.seedPolicy,
  );

  return {
    assignedRegionStrategy:
      reproductionPolicyInput?.assignedRegionStrategy ??
      DEFAULT_ASSIGNED_REGION_STRATEGY,
    mode: reproductionPolicyInput?.mode ?? 'sexual',
    modeIsEvolvable:
      reproductionPolicyInput?.modeIsEvolvable ?? DEFAULT_MODE_IS_EVOLVABLE,
    parthenogenesisMutationRate:
      reproductionPolicyInput?.parthenogenesisMutationRate ??
      DEFAULT_PARTHENOGENESIS_MUTATION_RATE,
    polyandricDroneContributionFraction:
      reproductionPolicyInput?.polyandricDroneContributionFraction ??
      DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION,
    polyandricDroneCount:
      reproductionPolicyInput?.polyandricDroneCount ??
      DEFAULT_POLYANDRIC_DRONE_COUNT,
    queenBias: reproductionPolicyInput?.queenBias ?? DEFAULT_QUEEN_BIAS,
    seedPolicy: {
      ...defaultSeedPolicy,
      ...resolvedSeedPolicy,
    },
  };
}

function resolveSeedPolicy(
  seedPolicyInput: Partial<NgeSeedPolicy> | NgeSeedPolicyShorthand | undefined,
): Partial<NgeSeedPolicy> {
  if (seedPolicyInput === 'queen-weighted') {
    return { siblingsDifferBySeed: true, twinsAllowed: false };
  }
  return seedPolicyInput ?? {};
}

function resolveSubstrateConfig(
  substrateInput: NgeDnaConstructorInput['substrate'],
): NgeSubstrateConfig {
  return {
    budgetOverride: {
      maxEdges: resolveBudgetValue(
        substrateInput?.budgetOverride?.maxEdges,
        NGE_DNA_DEFAULT_BUDGET_MAX_EDGES,
        'maxEdges',
      ),
      maxNodes: resolveBudgetValue(
        substrateInput?.budgetOverride?.maxNodes,
        NGE_DNA_DEFAULT_BUDGET_MAX_NODES,
        'maxNodes',
      ),
    },
    dimensions: 3,
    normalization: 'unit-cube',
    zonePartition: resolveZonePartition(substrateInput?.zonePartition),
  };
}

function resolveZonePartition(
  zonePartitionInput: NgeSubstrateInput['zonePartition'],
): NgeZonePartitionConfig {
  return {
    x: {
      count: resolveAxisPartitionCount(zonePartitionInput?.x?.count, 'x'),
    },
    y: {
      count: resolveAxisPartitionCount(zonePartitionInput?.y?.count, 'y'),
    },
    z: {
      count: resolveAxisPartitionCount(zonePartitionInput?.z?.count, 'z'),
    },
  };
}

function resolveAxisPartitionCount(
  partitionCount: number | undefined,
  axisLabel: keyof NgeZonePartitionConfig,
): number {
  const resolvedPartitionCount =
    partitionCount ?? NGE_DNA_DEFAULT_ZONE_PARTITION_COUNT;

  if (
    !Number.isInteger(resolvedPartitionCount) ||
    resolvedPartitionCount <= 0
  ) {
    throw new NGE_DNA_SubstrateError(
      `NGE_DNA substrate axis ${axisLabel} requires one positive integer partition count.`,
    );
  }

  return resolvedPartitionCount;
}

function resolveRulePasses(
  rulePassInputs: NgeDnaConstructorInput['rulePasses'],
): NgeRulePass[] {
  return (rulePassInputs ?? []).map((rulePassInput) => ({
    kind: rulePassInput.kind,
    archetypeId: rulePassInput.archetypeId,
    priority: resolveRulePriority(rulePassInput.priority),
    placements: (rulePassInput.placements ?? []).map((placement) => ({
      coordinate: cloneCoordinate(placement.coordinate),
      computationType: placement.computationType,
    })),
  }));
}

function resolveRulePriority(priority: number | undefined): number {
  return typeof priority === 'number' && Number.isFinite(priority)
    ? priority
    : NGE_DNA_DEFAULT_RULE_PRIORITY;
}

function resolveBudgetValue(
  overrideValue: number | undefined,
  maximumValue: number,
  label: 'maxEdges' | 'maxNodes',
): number {
  if (overrideValue === undefined) {
    return maximumValue;
  }

  if (overrideValue > maximumValue) {
    throw new NGE_DNA_BudgetError(
      `NGE_DNA ${label} budget ${overrideValue} exceeds the maximum ${maximumValue}.`,
    );
  }

  return overrideValue;
}

function withComputedFingerprint(
  envelope: NgeDnaCanonicalEnvelope,
): NgeDnaCanonicalEnvelope {
  return {
    ...envelope,
    fingerprint: computeFingerprint(serializeFingerprintSource(envelope)),
  };
}

function serializeFingerprintSource(envelope: NgeDnaCanonicalEnvelope): string {
  const cppnPrograms = Array.isArray(envelope.cppnPrograms)
    ? envelope.cppnPrograms
    : [];
  const moduleArchetypes = Array.isArray(envelope.moduleArchetypes)
    ? envelope.moduleArchetypes
    : [];

  return canonicalSerialize({
    ...envelope,
    cppnPrograms: cppnPrograms.length === 0 ? undefined : cppnPrograms,
    fingerprint: '',
    moduleArchetypes:
      moduleArchetypes.length === 0 ? undefined : moduleArchetypes,
  });
}

function validateFingerprintMatchesEnvelope(
  envelope: NgeDnaCanonicalEnvelope,
): void {
  const expectedFingerprint = computeFingerprint(
    serializeFingerprintSource(envelope),
  );

  if (envelope.fingerprint !== expectedFingerprint) {
    throw new NGE_DNA_SchemaError(
      `NGE_DNA fingerprint ${envelope.fingerprint} does not match the canonical envelope hash ${expectedFingerprint}.`,
    );
  }
}

function cloneCoordinate(
  coordinate: NeatGenomeSubstrateCoordinate,
): NeatGenomeSubstrateCoordinate {
  return [coordinate[0], coordinate[1], coordinate[2]];
}

function resolveCppnPrograms(
  cppnProgramInputs: NgeDnaConstructorInput['cppnPrograms'],
): NgeCppnProgram[] {
  return (cppnProgramInputs ?? []).map((cppnProgramInput) => ({
    programId: cppnProgramInput.programId,
    inputNodeIds: [
      ...(cppnProgramInput.inputNodeIds ?? DEFAULT_CPPN_INPUT_NODE_IDS),
    ],
    outputNodeIds: [
      ...(cppnProgramInput.outputNodeIds ?? DEFAULT_CPPN_OUTPUT_NODE_IDS),
    ],
    hiddenNodes: (cppnProgramInput.hiddenNodes ?? []).map((nodeInput) => ({
      activationKind: nodeInput.activationKind ?? DEFAULT_CPPN_ACTIVATION_KIND,
      bias: nodeInput.bias ?? DEFAULT_CPPN_NODE_BIAS,
      nodeId: nodeInput.nodeId,
    })),
    edges: (cppnProgramInput.edges ?? []).map((edgeInput) => ({
      sourceNodeId: edgeInput.sourceNodeId,
      targetNodeId: edgeInput.targetNodeId,
      weight: edgeInput.weight ?? 0,
    })),
  }));
}

function resolveModuleArchetypes(
  moduleArchetypeInputs: NgeDnaConstructorInput['moduleArchetypes'],
): NgeDnaModuleArchetype[] {
  return (moduleArchetypeInputs ?? []).map((moduleArchetypeInput) => ({
    archetypeId: moduleArchetypeInput.archetypeId,
    computationType: moduleArchetypeInput.computationType,
    receivesCoordinates: moduleArchetypeInput.receivesCoordinates,
    residualStreamId: moduleArchetypeInput.residualStreamId,
    weightSharedCohortId: moduleArchetypeInput.weightSharedCohortId,
    parameterSchema:
      moduleArchetypeInput.parameterSchema === undefined
        ? undefined
        : structuredClone(moduleArchetypeInput.parameterSchema),
  }));
}
