/**
 * NGE reproduction operators: parthenogenesis, polyandric, and sexual crossover.
 *
 * The polyandric operator merges a queen DNA template with patch contributions
 * from a small set of drone donors. Region assignment is controlled by
 * {@link NgeReproductionPolicy.assignedRegionStrategy}.
 *
 * ## Input shorthand compatibility
 *
 * The strategy `'non-overlapping'` is input shorthand for the deterministic
 * single-drone-per-region assignment that the core already implements under
 * `'roundRobin'`. Both values resolve to identical behavior; only the canonical
 * string stored in the envelope differs.
 *
 * This alias preserves the input shorthand while keeping the canonical strategy
 * string stored in the envelope.
 */
import { NGE_DNA } from '../nge-dna/neat.nge-dna';
import type {
  NgeDnaCanonicalEnvelope,
  NgeDnaModuleArchetype,
  NgeReproductionPolicy,
  NgeReproductionPolicyInput,
} from '../nge-dna/neat.nge-dna.types';

import { NgeEvolution_ModeError } from './neat.nge-evolution.errors';
import type {
  NgeEvolutionPolyandricAssignedRegion,
  NgeEvolutionPolyandricRegionAssignmentResult,
  NgeEvolutionReproductionResult,
} from './neat.nge-evolution.types';

type NgeEvolutionSexualSourceParent = 'first-parent' | 'second-parent';
type NgePolyandricRegionFamily =
  'cppnPrograms' | 'moduleArchetypes' | 'rulePasses';

interface NgeParthenogenesisInput {
  ngeEnabled: boolean;
  parent: NgeDnaCanonicalEnvelope;
  parentId: string;
  policy?: NgeReproductionPolicyInput;
}

/**
 * One drone donor offered to the polyandric reproduction operator.
 *
 * A drone carries a full DNA envelope plus optional bookkeeping: its fitness
 * ranking drives the `byFitness` assignment strategy, and `specializationKey`
 * drives the `bySpecialization` strategy. Only regions that are actually
 * assigned to this drone will be patched into the queen template.
 *
 * Background reading on multi-parent recombination:
 * [Wikipedia — Crossover (genetic algorithm)](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)).
 *
 * @example
 * ```ts
 * const drone: NgePolyandricDroneInput = {
 *   dna: donorEnvelope,
 *   fitness: 0.92,
 *   parentId: 'donor-a',
 *   specializationKey: 'moduleArchetypes',
 * };
 * ```
 */
export interface NgePolyandricDroneInput {
  dna: NgeDnaCanonicalEnvelope;
  fitness?: number;
  parentId: string;
  specializationKey?: string;
}

/**
 * Input contract for the polyandric reproduction operator.
 *
 * The queen DNA is the stable template. Up to
 * {@link NgeReproductionPolicy.polyandricDroneCount} drone donors compete to
 * patch a capped subset of the queen's regions, controlled by
 * {@link NgeReproductionPolicy.polyandricDroneContributionFraction}. The
 * {@link NgeReproductionPolicy.queenBias} gate then decides, per region,
 * whether queen or drone data wins the merge.
 *
 * Background reading on the biological metaphor:
 * [Wikipedia — Polyandry](https://en.wikipedia.org/wiki/Polyandry).
 *
 * @example
 * ```ts
 * const input: NgePolyandricInput = {
 *   ngeEnabled: true,
 *   queen: queenEnvelope,
 *   queenId: 'queen-1',
 *   drones: [
 *     {
 *       dna: donorA,
 *       parentId: 'drone-a',
 *       fitness: 0.9,
 *     },
 *     {
 *       dna: donorB,
 *       parentId: 'drone-b',
 *       fitness: 0.85,
 *       specializationKey: 'cppnPrograms',
 *     },
 *   ],
 * };
 * ```
 */
export interface NgePolyandricInput {
  drones: readonly NgePolyandricDroneInput[];
  ngeEnabled: boolean;
  policy?: NgeReproductionPolicyInput;
  queen: NgeDnaCanonicalEnvelope;
  queenId: string;
}

interface NgeSexualInput {
  firstParent: NgeDnaCanonicalEnvelope;
  firstParentId: string;
  firstParentScore: number;
  policy?: NgeReproductionPolicyInput;
  secondParent: NgeDnaCanonicalEnvelope;
  secondParentId: string;
  secondParentScore: number;
}

interface NgeSexualFamilySelectionResult<TFamilyItem> {
  firstParentRegionIds: string[];
  items: TFamilyItem[];
  secondParentRegionIds: string[];
}

interface NgeSexualMatchingRegionSelectionResult<TFamilyItem> {
  item: TFamilyItem;
  sourceParent: NgeEvolutionSexualSourceParent;
}

type NgeParthenogenesisMutationApplier = (
  canonicalEnvelope: NgeDnaCanonicalEnvelope,
  mutationRate: number,
) => NgeDnaCanonicalEnvelope;

const MATCHING_REGION_SELECTION_THRESHOLD = 0.5;
const DEFAULT_SEXUAL_RANDOM_SAMPLE = 0.75;

/**
 * Build one parthenogenetic offspring from a single NGE DNA parent.
 *
 * @param input - Operator context containing the source parent DNA and mode flags.
 * @param mutateOffspring - Optional mutation callback applied only when the configured rate is non-zero.
 * @returns Canonical offspring DNA plus parent-contribution metadata.
 */
export function reproduceParthenogenesis(
  input: NgeParthenogenesisInput,
  mutateOffspring: NgeParthenogenesisMutationApplier = passthroughMutation,
): NgeEvolutionReproductionResult {
  const resolvedPolicy = resolveOperatorPolicy(
    input.policy ?? input.parent.reproductionPolicy,
    'parthenogenesis',
  );

  if (!input.ngeEnabled) {
    throw new NgeEvolution_ModeError(
      'parthenogenesis is unavailable when NGE is disabled.',
    );
  }

  // Step 1: Canonicalize the parent DNA under the selected reproduction mode.
  const clonedOffspring = buildCanonicalEnvelope(input.parent, {
    reproductionPolicy: resolvedPolicy,
  });

  // Step 2: Apply mutation only when the parthenogenesis rate is non-zero.
  const offspring =
    resolvedPolicy.parthenogenesisMutationRate === 0
      ? clonedOffspring
      : buildCanonicalEnvelope(
          mutateOffspring(
            clonedOffspring,
            resolvedPolicy.parthenogenesisMutationRate,
          ),
          {
            reproductionPolicy: resolvedPolicy,
          },
        );

  // Step 3: Return the operator result with one sole-parent contribution record.
  return {
    epigeneticReference: null,
    offspring,
    outcome:
      resolvedPolicy.parthenogenesisMutationRate === 0
        ? 'clone'
        : 'mutation-only',
    parentContributions: [
      {
        contributionKind:
          resolvedPolicy.parthenogenesisMutationRate === 0
            ? 'clone'
            : 'mutation',
        parentId: input.parentId,
        regionIds: collectEnvelopeContributionRegionIds(offspring),
        role: 'sole-parent',
      },
    ],
    policy: resolvedPolicy,
    regionAssignment: null,
  };
}

/**
 * Build one polyandric offspring from a queen DNA template plus optional drone donors.
 *
 * Polyandric recombination is a multi-parent operator: the queen template keeps
 * most of its structure, while a small pool of drones patches a capped subset
 * of its DNA regions. The cap and drone count come from the resolved policy.
 * For every assigned region a deterministic FNV-1a hash of the region id is
 * compared against `queenBias`; when the hash is below the bias the queen wins,
 * otherwise the drone wins. This makes the merge deterministic and
 * reproducible for the same queen/drone/policy triple.
 *
 * See {@link NgePolyandricInput} for the input shape and
 * {@link NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS} for the default bias.
 *
 * Background reading:
 * - Polyandry in evolutionary biology:
 *   [Wikipedia — Polyandry](https://en.wikipedia.org/wiki/Polyandry).
 * - Multi-parent recombination in evolutionary computing:
 *   [Wikipedia — Crossover (genetic algorithm)](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)).
 * - The FNV-1a hash function:
 *   [Wikipedia — Fowler–Noll–Vo hash function](https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function).
 *
 * @param input - Operator context containing the queen DNA, drone donors, and policy overrides.
 * @returns Canonical offspring DNA plus the resolved region-assignment report.
 *
 * @example
 * ```ts
 * const result = reproducePolyandric({
 *   ngeEnabled: true,
 *   queen: queenEnvelope,
 *   queenId: 'queen-1',
 *   drones: [
 *     { dna: donorEnvelope, parentId: 'drone-a', fitness: 0.9 },
 *   ],
 * });
 * console.log(result.outcome); // 'queen-template-patched'
 * ```
 */
export function reproducePolyandric(
  input: NgePolyandricInput,
): NgeEvolutionReproductionResult {
  const resolvedPolicy = resolveOperatorPolicy(
    input.policy ?? input.queen.reproductionPolicy,
    'polyandric',
  );

  if (!input.ngeEnabled) {
    throw new NgeEvolution_ModeError(
      'polyandric reproduction is unavailable when NGE is disabled.',
    );
  }

  // Step 1: Compute the patchable queen-region shelf under the configured cap.
  const queenRegionIds = collectPolyandricRegionIds(input.queen);
  const patchableRegionIds = queenRegionIds.slice(
    0,
    Math.ceil(
      queenRegionIds.length *
        Math.max(0, resolvedPolicy.polyandricDroneContributionFraction),
    ),
  );
  const eligibleDrones = input.drones.slice(
    0,
    resolvedPolicy.polyandricDroneCount,
  );
  const regionAssignment = assignPolyandricRegions(
    patchableRegionIds,
    eligibleDrones,
    resolvedPolicy,
  );

  // Step 2: Patch the queen template with each assigned drone contribution.
  const offspring = buildCanonicalEnvelope(
    applyPolyandricAssignments(
      input.queen,
      eligibleDrones,
      regionAssignment,
      resolvedPolicy.queenBias,
    ),
    {
      reproductionPolicy: resolvedPolicy,
    },
  );

  // Step 3: Return the canonical offspring plus queen and drone contribution records.
  return {
    epigeneticReference: null,
    offspring,
    outcome: 'queen-template-patched',
    parentContributions: [
      {
        contributionKind: 'clone',
        parentId: input.queenId,
        regionIds: collectPolyandricRegionIds(input.queen),
        role: 'queen',
      },
      ...regionAssignment.assignedRegions.map((assignedRegion) => ({
        contributionKind: 'patch' as const,
        parentId: assignedRegion.droneId,
        regionIds: [assignedRegion.regionId],
        role: 'drone' as const,
      })),
    ],
    policy: resolvedPolicy,
    regionAssignment,
  };
}

/**
 * Build one sexual offspring using NEAT-aligned fitter-parent handling for disjoint regions.
 *
 * @param input - Operator context containing both parent DNAs and their relative fitness scores.
 * @param randomGenerator - Deterministic selector used for matching-region crossover choices.
 * @returns Canonical offspring DNA plus per-parent contribution records.
 */
export function reproduceSexual(
  input: NgeSexualInput,
  randomGenerator: () => number = createDefaultSexualRandomGenerator,
): NgeEvolutionReproductionResult {
  const resolvedPolicy = resolveOperatorPolicy(
    input.policy ?? input.firstParent.reproductionPolicy,
    'sexual',
  );

  // Step 1: Select each DNA family with fitter-parent disjoint inheritance.
  const selectedCppnPrograms = selectSexualFamilyRegions(
    input.firstParent.cppnPrograms,
    input.secondParent.cppnPrograms,
    ({ programId }) => programId,
    input.firstParentScore,
    input.secondParentScore,
    randomGenerator,
  );
  const selectedModuleArchetypes = selectSexualFamilyRegions(
    input.firstParent.moduleArchetypes,
    input.secondParent.moduleArchetypes,
    ({ archetypeId }) => archetypeId,
    input.firstParentScore,
    input.secondParentScore,
    randomGenerator,
  );
  const selectedRulePasses = selectSexualFamilyRegions(
    input.firstParent.rulePasses,
    input.secondParent.rulePasses,
    ({ archetypeId, kind, priority }) => `${kind}:${archetypeId}:${priority}`,
    input.firstParentScore,
    input.secondParentScore,
    randomGenerator,
  );

  // Step 2: Canonicalize the fitter-parent base envelope with the selected family shelves.
  const offspring = buildCanonicalEnvelope(
    selectSexualBaseEnvelope(
      input.firstParent,
      input.secondParent,
      input.firstParentScore,
      input.secondParentScore,
    ),
    {
      cppnPrograms: selectedCppnPrograms.items,
      moduleArchetypes: selectedModuleArchetypes.items,
      reproductionPolicy: resolvedPolicy,
      rulePasses: selectedRulePasses.items,
    },
  );

  // Step 3: Return the offspring plus one contribution record per parent.
  return {
    epigeneticReference: null,
    offspring,
    outcome: 'sexual-crossover',
    parentContributions: [
      {
        contributionKind: 'crossover',
        parentId: input.firstParentId,
        regionIds: [
          ...selectedCppnPrograms.firstParentRegionIds,
          ...selectedModuleArchetypes.firstParentRegionIds,
          ...selectedRulePasses.firstParentRegionIds,
        ],
        role:
          input.firstParentScore >= input.secondParentScore
            ? 'primary'
            : 'secondary',
      },
      {
        contributionKind: 'crossover',
        parentId: input.secondParentId,
        regionIds: [
          ...selectedCppnPrograms.secondParentRegionIds,
          ...selectedModuleArchetypes.secondParentRegionIds,
          ...selectedRulePasses.secondParentRegionIds,
        ],
        role:
          input.secondParentScore > input.firstParentScore
            ? 'primary'
            : 'secondary',
      },
    ],
    policy: resolvedPolicy,
    regionAssignment: null,
  };
}

/**
 * Apply every assigned drone patch to the queen template in region order.
 *
 * This is the fold that turns the region-assignment report into a concrete
 * offspring envelope. Each assigned region is passed to
 * {@link patchPolyandricRegion} with the same `queenBias`, so the queen/drone
 * winner gate is deterministic across the whole patch set.
 *
 * @param queenEnvelope - Queen DNA template.
 * @param drones - Eligible drone donors in the order supplied by the caller.
 * @param regionAssignment - Region-to-drone mapping produced by the assignment step.
 * @param queenBias - Per-region winner bias in [0, 1].
 * @returns The patched queen envelope ready for canonicalization.
 */
function applyPolyandricAssignments(
  queenEnvelope: NgeDnaCanonicalEnvelope,
  drones: readonly NgePolyandricDroneInput[],
  regionAssignment: NgeEvolutionPolyandricRegionAssignmentResult,
  queenBias: number,
): NgeDnaCanonicalEnvelope {
  const dronesById = new Map(
    drones.map((droneInput) => [droneInput.parentId, droneInput]),
  );

  return regionAssignment.assignedRegions.reduce(
    (currentEnvelope, assignedRegion) =>
      patchPolyandricRegion(
        currentEnvelope,
        dronesById.get(assignedRegion.droneId)!.dna,
        assignedRegion.regionId,
        queenBias,
      ),
    queenEnvelope,
  );
}

function assignPolyandricRegions(
  patchableRegionIds: readonly string[],
  drones: readonly NgePolyandricDroneInput[],
  policy: NgeReproductionPolicy,
): NgeEvolutionPolyandricRegionAssignmentResult {
  const assignedRegions: NgeEvolutionPolyandricAssignedRegion[] = [];
  const unassignedRegionIds: string[] = [];
  const orderedDrones =
    policy.assignedRegionStrategy === 'byFitness'
      ? drones.toSorted(comparePolyandricDronesByFitness)
      : [...drones];

  for (
    let patchableRegionIndex = 0;
    patchableRegionIndex < patchableRegionIds.length;
    patchableRegionIndex++
  ) {
    const regionId = patchableRegionIds[patchableRegionIndex];
    const assignedDrone = selectPolyandricDroneForRegion(
      regionId,
      patchableRegionIndex,
      orderedDrones,
      policy,
    );

    if (!assignedDrone) {
      unassignedRegionIds.push(regionId);
      continue;
    }

    assignedRegions.push({
      droneFitness: assignedDrone.fitness,
      droneId: assignedDrone.parentId,
      droneRank: orderedDrones.findIndex(
        ({ parentId }) => parentId === assignedDrone.parentId,
      ),
      regionId,
      specializationKey: assignedDrone.specializationKey,
    });
  }

  return {
    assignedRegions,
    patchableRegionIds: [...patchableRegionIds],
    strategy: policy.assignedRegionStrategy,
    unassignedRegionIds,
  };
}

function buildCanonicalEnvelope(
  baseEnvelope: NgeDnaCanonicalEnvelope,
  overrides: Partial<
    Pick<
      NgeDnaCanonicalEnvelope,
      | 'compatibilityVersion'
      | 'cppnPrograms'
      | 'encodingMode'
      | 'moduleArchetypes'
      | 'reproductionPolicy'
      | 'rulePasses'
      | 'schemaVersion'
      | 'substrate'
    >
  >,
): NgeDnaCanonicalEnvelope {
  return new NGE_DNA({
    ...baseEnvelope,
    ...overrides,
  }).toCanonical();
}

function collectEnvelopeContributionRegionIds(
  canonicalEnvelope: NgeDnaCanonicalEnvelope,
): string[] {
  const regionIds: string[] = [];

  for (
    let cppnProgramIndex = 0;
    cppnProgramIndex < canonicalEnvelope.cppnPrograms.length;
    cppnProgramIndex++
  ) {
    regionIds.push(canonicalEnvelope.cppnPrograms[cppnProgramIndex].programId);
  }

  for (
    let archetypeIndex = 0;
    archetypeIndex < canonicalEnvelope.moduleArchetypes.length;
    archetypeIndex++
  ) {
    regionIds.push(
      canonicalEnvelope.moduleArchetypes[archetypeIndex].archetypeId,
    );
  }

  for (
    let rulePassIndex = 0;
    rulePassIndex < canonicalEnvelope.rulePasses.length;
    rulePassIndex++
  ) {
    const rulePass = canonicalEnvelope.rulePasses[rulePassIndex];
    regionIds.push(
      `${rulePass.kind}:${rulePass.archetypeId}:${rulePass.priority}`,
    );
  }

  return regionIds;
}

function collectPolyandricRegionIds(
  canonicalEnvelope: NgeDnaCanonicalEnvelope,
): string[] {
  const regionIds: string[] = [];

  for (
    let programIndex = 0;
    programIndex < canonicalEnvelope.cppnPrograms.length;
    programIndex++
  ) {
    regionIds.push(`cppnPrograms:${programIndex}`);
  }

  for (
    let archetypeIndex = 0;
    archetypeIndex < canonicalEnvelope.moduleArchetypes.length;
    archetypeIndex++
  ) {
    regionIds.push(`moduleArchetypes:${archetypeIndex}`);
  }

  for (
    let rulePassIndex = 0;
    rulePassIndex < canonicalEnvelope.rulePasses.length;
    rulePassIndex++
  ) {
    regionIds.push(`rulePasses:${rulePassIndex}`);
  }

  return regionIds;
}

function comparePolyandricDronesByFitness(
  leftDrone: NgePolyandricDroneInput,
  rightDrone: NgePolyandricDroneInput,
): number {
  const leftFitness = leftDrone.fitness ?? Number.NEGATIVE_INFINITY;
  const rightFitness = rightDrone.fitness ?? Number.NEGATIVE_INFINITY;

  return (
    rightFitness - leftFitness ||
    Number(leftDrone.parentId > rightDrone.parentId) -
      Number(leftDrone.parentId < rightDrone.parentId)
  );
}

function createDefaultSexualRandomGenerator(): number {
  return DEFAULT_SEXUAL_RANDOM_SAMPLE;
}

/**
 * Deterministically map a DNA region identifier into the unit interval [0, 1).
 *
 * Uses the FNV-1a 32-bit hash so the same `regionId` always yields the same
 * value. The result is combined with `queenBias` to decide whether the queen or
 * drone wins a patched region.
 *
 * See the FNV-1a reference:
 * [Wikipedia — Fowler–Noll–Vo hash function](https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function).
 *
 * @param regionId - Stable region identifier emitted by {@link collectPolyandricRegionIds}.
 * @returns A deterministic number in [0, 1).
 */
function hashRegionIdToUnitInterval(regionId: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < regionId.length; i++) {
    hash ^= regionId.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0) / 4294967296;
}

/**
 * Merge one module-archetype region from queen and drone, respecting the queen bias.
 *
 * The per-region winner is decided by comparing
 * {@link hashRegionIdToUnitInterval}(regionId) to the clamped `queenBias`.
 * When the hash is below the bias the queen wins and its fields override the
 * drone's; otherwise the drone wins. Both cases shallow-merge the losing region
 * into the winning region and combine `parameterSchema` maps so no keys are
 * silently dropped.
 *
 * @param queenRegion - Module archetype taken from the queen template.
 * @param droneRegion - Module archetype taken from the assigned drone.
 * @param regionId - Stable region identifier used to seed the deterministic gate.
 * @param queenBias - Bias in [0, 1]; higher values make the queen more likely to win.
 * @returns The merged module archetype written back into the offspring envelope.
 */
function mergeModuleArchetypeWithQueenPriority(
  queenRegion: NgeDnaModuleArchetype,
  droneRegion: NgeDnaModuleArchetype,
  regionId: string,
  queenBias: number,
): NgeDnaModuleArchetype {
  const queenWins =
    hashRegionIdToUnitInterval(regionId) < Math.max(0, Math.min(1, queenBias));

  if (!queenWins) {
    return {
      ...structuredClone(queenRegion),
      ...structuredClone(droneRegion),
      parameterSchema: {
        ...(queenRegion.parameterSchema ?? {}),
        ...(droneRegion.parameterSchema ?? {}),
      },
    };
  }

  return {
    ...structuredClone(droneRegion),
    ...structuredClone(queenRegion),
    parameterSchema: {
      ...(droneRegion.parameterSchema ?? {}),
      ...(queenRegion.parameterSchema ?? {}),
    },
  };
}

function parsePolyandricRegionId(regionId: string): {
  family: NgePolyandricRegionFamily;
  index: number;
} {
  const [familyToken, indexToken] = regionId.split(':');

  return {
    family: familyToken as NgePolyandricRegionFamily,
    index: Number(indexToken),
  };
}

function passthroughMutation(
  canonicalEnvelope: NgeDnaCanonicalEnvelope,
): NgeDnaCanonicalEnvelope {
  return canonicalEnvelope;
}

/**
 * Patch one DNA region of the queen envelope with the matching drone region.
 *
 * Reads the region family (`cppnPrograms`, `moduleArchetypes`, or `rulePasses`)
 * and index from `regionId`, then writes back the merged value. Module
 * archetypes use {@link mergeModuleArchetypeWithQueenPriority}, which keeps the
 * queen-bias gate explicit and preserves both parameter schemas. Other families
 * replace the losing region with the winning region under the same FNV-1a gate.
 * If either side is missing the region, the queen envelope is returned unchanged.
 *
 * @param queenEnvelope - Queen DNA template being patched.
 * @param droneEnvelope - Drone DNA carrying the candidate replacement region.
 * @param regionId - Stable region identifier in `family:index` form.
 * @param queenBias - Bias in [0, 1] that controls how often the queen keeps the region.
 * @returns A new queen envelope with the region patched, or the original envelope when the region is absent.
 */
function patchPolyandricRegion(
  queenEnvelope: NgeDnaCanonicalEnvelope,
  droneEnvelope: NgeDnaCanonicalEnvelope,
  regionId: string,
  queenBias: number,
): NgeDnaCanonicalEnvelope {
  const { family, index } = parsePolyandricRegionId(regionId);
  const regionAccessors = {
    cppnPrograms: {
      read: (envelope: NgeDnaCanonicalEnvelope) => envelope.cppnPrograms[index],
      write: (
        envelope: NgeDnaCanonicalEnvelope,
        regionValue: NgeDnaCanonicalEnvelope['cppnPrograms'][number],
      ) => ({
        ...envelope,
        cppnPrograms: envelope.cppnPrograms.with(index, regionValue),
      }),
    },
    moduleArchetypes: {
      read: (envelope: NgeDnaCanonicalEnvelope) =>
        envelope.moduleArchetypes[index],
      write: (
        envelope: NgeDnaCanonicalEnvelope,
        regionValue: NgeDnaCanonicalEnvelope['moduleArchetypes'][number],
      ) => ({
        ...envelope,
        moduleArchetypes: envelope.moduleArchetypes.with(index, regionValue),
      }),
    },
    rulePasses: {
      read: (envelope: NgeDnaCanonicalEnvelope) => envelope.rulePasses[index],
      write: (
        envelope: NgeDnaCanonicalEnvelope,
        regionValue: NgeDnaCanonicalEnvelope['rulePasses'][number],
      ) => ({
        ...envelope,
        rulePasses: envelope.rulePasses.with(index, regionValue),
      }),
    },
  } as const;
  const resolvedRegionAccessor = regionAccessors[family];
  const queenRegion = resolvedRegionAccessor.read(queenEnvelope);
  const droneRegion = resolvedRegionAccessor.read(droneEnvelope);

  if (!queenRegion || !droneRegion) {
    return queenEnvelope;
  }

  if (family === 'moduleArchetypes') {
    return regionAccessors.moduleArchetypes.write(
      queenEnvelope,
      mergeModuleArchetypeWithQueenPriority(
        queenRegion as NgeDnaModuleArchetype,
        droneRegion as NgeDnaModuleArchetype,
        regionId,
        queenBias,
      ),
    );
  }

  const queenWins =
    hashRegionIdToUnitInterval(regionId) < Math.max(0, Math.min(1, queenBias));
  const [losingRegion, winningRegion] = queenWins
    ? [droneRegion, queenRegion]
    : [queenRegion, droneRegion];

  return resolvedRegionAccessor.write(queenEnvelope, {
    ...structuredClone(losingRegion),
    ...structuredClone(winningRegion),
  } as never);
}

/**
 * Expand input seed-governance shorthand into the canonical policy shape.
 *
 * Accepts the {@link NgeSeedPolicyShorthand} `'queen-weighted'` and returns the
 * canonical `{ siblingsDifferBySeed: true, twinsAllowed: false }` object. All
 * other already-canonical values pass through unchanged so the envelope always
 * stores a stable object for serialization, hashing, and round-trips.
 *
 * @param seedPolicy - Seed policy supplied at an operator input boundary.
 * @returns Canonical seed policy object for storage in the envelope.
 */
function expandSeedPolicy(
  seedPolicy: NgeReproductionPolicyInput['seedPolicy'],
): NgeReproductionPolicy['seedPolicy'] {
  if (seedPolicy === 'queen-weighted') {
    return { siblingsDifferBySeed: true, twinsAllowed: false };
  }
  return seedPolicy;
}

function resolveOperatorPolicy(
  policy: NgeReproductionPolicyInput,
  mode: NgeReproductionPolicy['mode'],
): NgeReproductionPolicy {
  return {
    ...policy,
    mode,
    seedPolicy: expandSeedPolicy(policy.seedPolicy),
  };
}

function selectPolyandricDroneForRegion(
  regionId: string,
  regionIndex: number,
  orderedDrones: readonly NgePolyandricDroneInput[],
  policy: NgeReproductionPolicy,
): NgePolyandricDroneInput | undefined {
  if (orderedDrones.length === 0) {
    return undefined;
  }

  if (policy.assignedRegionStrategy === 'bySpecialization') {
    return (
      orderedDrones.find(
        ({ specializationKey }) =>
          specializationKey === parsePolyandricRegionId(regionId).family,
      ) ?? orderedDrones[regionIndex % orderedDrones.length]
    );
  }

  // `roundRobin`, `non-overlapping`, and `byFitness` (after pre-sorting) all
  // resolve to one deterministic drone per patchable region.
  return orderedDrones[regionIndex % orderedDrones.length];
}

function selectSexualBaseEnvelope(
  firstParent: NgeDnaCanonicalEnvelope,
  secondParent: NgeDnaCanonicalEnvelope,
  firstParentScore: number,
  secondParentScore: number,
): NgeDnaCanonicalEnvelope {
  return firstParentScore >= secondParentScore ? firstParent : secondParent;
}

function selectSexualFamilyRegions<TFamilyItem>(
  firstParentItems: readonly TFamilyItem[],
  secondParentItems: readonly TFamilyItem[],
  getRegionId: (familyItem: TFamilyItem) => string,
  firstParentScore: number,
  secondParentScore: number,
  randomGenerator: () => number,
): NgeSexualFamilySelectionResult<TFamilyItem> {
  const secondParentItemsByRegionId = new Map(
    secondParentItems.map((familyItem) => [
      getRegionId(familyItem),
      familyItem,
    ]),
  );
  const selectedItems: TFamilyItem[] = [];
  const firstParentRegionIds: string[] = [];
  const secondParentRegionIds: string[] = [];
  const consumedSecondParentRegionIds = new Set<string>();

  for (
    let firstParentItemIndex = 0;
    firstParentItemIndex < firstParentItems.length;
    firstParentItemIndex++
  ) {
    const firstParentItem = firstParentItems[firstParentItemIndex];
    const regionId = getRegionId(firstParentItem);
    const secondParentItem = secondParentItemsByRegionId.get(regionId);

    if (secondParentItem) {
      const matchingSelection = chooseMatchingSexualRegion(
        firstParentItem,
        secondParentItem,
        randomGenerator,
      );

      selectedItems.push(matchingSelection.item);
      consumedSecondParentRegionIds.add(regionId);

      if (matchingSelection.sourceParent === 'first-parent') {
        firstParentRegionIds.push(regionId);
      } else {
        secondParentRegionIds.push(regionId);
      }

      continue;
    }

    if (firstParentScore >= secondParentScore) {
      selectedItems.push(structuredClone(firstParentItem));
      firstParentRegionIds.push(regionId);
    }
  }

  for (
    let secondParentItemIndex = 0;
    secondParentItemIndex < secondParentItems.length;
    secondParentItemIndex++
  ) {
    const secondParentItem = secondParentItems[secondParentItemIndex];
    const regionId = getRegionId(secondParentItem);

    if (consumedSecondParentRegionIds.has(regionId)) {
      continue;
    }

    if (secondParentScore >= firstParentScore) {
      selectedItems.push(structuredClone(secondParentItem));
      secondParentRegionIds.push(regionId);
    }
  }

  return {
    firstParentRegionIds,
    items: selectedItems,
    secondParentRegionIds,
  };
}

function chooseMatchingSexualRegion<TFamilyItem>(
  firstParentItem: TFamilyItem,
  secondParentItem: TFamilyItem,
  randomGenerator: () => number,
): NgeSexualMatchingRegionSelectionResult<TFamilyItem> {
  return randomGenerator() >= MATCHING_REGION_SELECTION_THRESHOLD
    ? {
        item: structuredClone(firstParentItem),
        sourceParent: 'first-parent',
      }
    : {
        item: structuredClone(secondParentItem),
        sourceParent: 'second-parent',
      };
}
