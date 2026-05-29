import { NGE_DNA } from '../nge-dna/neat.nge-dna';
import type {
  NgeDnaCanonicalEnvelope,
  NgeDnaModuleArchetype,
  NgeReproductionPolicy,
} from '../nge-dna/neat.nge-dna.types';

import { NgeEvolution_ModeError } from './neat.nge-evolution.errors';
import type {
  NgeEvolutionPolyandricAssignedRegion,
  NgeEvolutionPolyandricRegionAssignmentResult,
  NgeEvolutionReproductionResult,
} from './neat.nge-evolution.types';

type NgeEvolutionSexualSourceParent = 'first-parent' | 'second-parent';
type NgePolyandricRegionFamily =
  | 'cppnPrograms'
  | 'moduleArchetypes'
  | 'rulePasses';

interface NgeParthenogenesisInput {
  ngeEnabled: boolean;
  parent: NgeDnaCanonicalEnvelope;
  parentId: string;
  policy?: NgeReproductionPolicy;
}

interface NgePolyandricDroneInput {
  dna: NgeDnaCanonicalEnvelope;
  fitness?: number;
  parentId: string;
  specializationKey?: string;
}

interface NgePolyandricInput {
  drones: readonly NgePolyandricDroneInput[];
  ngeEnabled: boolean;
  policy?: NgeReproductionPolicy;
  queen: NgeDnaCanonicalEnvelope;
  queenId: string;
}

interface NgeSexualInput {
  firstParent: NgeDnaCanonicalEnvelope;
  firstParentId: string;
  firstParentScore: number;
  policy?: NgeReproductionPolicy;
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
 * @param input - Operator context containing the queen DNA, drone donors, and policy overrides.
 * @returns Canonical offspring DNA plus the resolved region-assignment report.
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
    applyPolyandricAssignments(input.queen, eligibleDrones, regionAssignment),
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

function applyPolyandricAssignments(
  queenEnvelope: NgeDnaCanonicalEnvelope,
  drones: readonly NgePolyandricDroneInput[],
  regionAssignment: NgeEvolutionPolyandricRegionAssignmentResult,
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

function mergeModuleArchetypeWithQueenPriority(
  queenRegion: NgeDnaModuleArchetype,
  droneRegion: NgeDnaModuleArchetype,
): NgeDnaModuleArchetype {
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

function patchPolyandricRegion(
  queenEnvelope: NgeDnaCanonicalEnvelope,
  droneEnvelope: NgeDnaCanonicalEnvelope,
  regionId: string,
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
      ),
    );
  }

  return resolvedRegionAccessor.write(queenEnvelope, {
    ...structuredClone(droneRegion),
    ...structuredClone(queenRegion),
  } as never);
}

function resolveOperatorPolicy(
  policy: NgeReproductionPolicy,
  mode: NgeReproductionPolicy['mode'],
): NgeReproductionPolicy {
  return {
    ...policy,
    mode,
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
