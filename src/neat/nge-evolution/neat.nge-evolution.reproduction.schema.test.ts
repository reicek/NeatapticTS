import type {
  NgeDnaCanonicalEnvelope,
  NgeReproductionPolicy,
} from '../nge-dna/neat.nge-dna.types';

import { reproducePolyandric } from './neat.nge-evolution.reproduction';

function createReproductionPolicy(
  overrides?: Partial<NgeReproductionPolicy>,
): NgeReproductionPolicy {
  return {
    mode: 'polyandric',
    parthenogenesisMutationRate: 0,
    polyandricDroneCount: 2,
    polyandricDroneContributionFraction: 1,
    queenBias: 1,
    assignedRegionStrategy: 'roundRobin',
    modeIsEvolvable: false,
    seedPolicy: {
      siblingsDifferBySeed: true,
      twinsAllowed: false,
    },
    ...overrides,
  };
}

function createDnaEnvelope(overrides?: {
  cppnPrograms?: NgeDnaCanonicalEnvelope['cppnPrograms'];
  moduleArchetypes?: NgeDnaCanonicalEnvelope['moduleArchetypes'];
  rulePasses?: NgeDnaCanonicalEnvelope['rulePasses'];
}): NgeDnaCanonicalEnvelope {
  return {
    schemaVersion: 'A.1.0',
    compatibilityVersion: '1.0.0',
    encodingMode: 'lossless',
    fingerprint: 'fingerprint:schema-test',
    substrate: {
      dimensions: 3,
      normalization: 'unit-cube',
      zonePartition: {
        x: { count: 1 },
        y: { count: 1 },
        z: { count: 1 },
      },
    },
    reproductionPolicy: createReproductionPolicy(),
    cppnPrograms: overrides?.cppnPrograms ?? [],
    moduleArchetypes: overrides?.moduleArchetypes ?? [],
    rulePasses: overrides?.rulePasses ?? [],
  } as unknown as NgeDnaCanonicalEnvelope;
}

function createModuleArchetype(
  overrides?: Partial<NgeDnaCanonicalEnvelope['moduleArchetypes'][number]>,
): NgeDnaCanonicalEnvelope['moduleArchetypes'][number] {
  return {
    archetypeId: 'archetype:base',
    computationType: 'DenseFeedForward',
    ...overrides,
  };
}

function createCppnProgram(
  overrides?: Partial<NgeDnaCanonicalEnvelope['cppnPrograms'][number]>,
): NgeDnaCanonicalEnvelope['cppnPrograms'][number] {
  return {
    edges: [],
    hiddenNodes: [],
    inputNodeIds: ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'dist'],
    outputNodeIds: ['weight', 'enableBias'],
    programId: 'cppn:base',
    ...overrides,
  };
}

function createRulePass(
  overrides?: Partial<NgeDnaCanonicalEnvelope['rulePasses'][number]>,
): NgeDnaCanonicalEnvelope['rulePasses'][number] {
  return {
    archetypeId: 'archetype:rule-pass',
    kind: 'replicate',
    placements: [],
    priority: 0,
    ...overrides,
  };
}

function buildSchemaQueen(): NgeDnaCanonicalEnvelope {
  return createDnaEnvelope({
    cppnPrograms: [createCppnProgram({ programId: 'cppn:queen' })],
    moduleArchetypes: [
      createModuleArchetype({ archetypeId: 'archetype:queen:0' }),
      createModuleArchetype({ archetypeId: 'archetype:queen:1' }),
      createModuleArchetype({ archetypeId: 'archetype:queen:2' }),
      createModuleArchetype({ archetypeId: 'archetype:queen:3' }),
    ],
    rulePasses: [createRulePass({ archetypeId: 'rule-pass:queen' })],
  });
}

function buildSchemaDrone(parentId: string): NgeDnaCanonicalEnvelope {
  return createDnaEnvelope({
    cppnPrograms: [createCppnProgram({ programId: `cppn:${parentId}` })],
    moduleArchetypes: [
      createModuleArchetype({ archetypeId: `archetype:${parentId}:0` }),
      createModuleArchetype({ archetypeId: `archetype:${parentId}:1` }),
      createModuleArchetype({ archetypeId: `archetype:${parentId}:2` }),
      createModuleArchetype({ archetypeId: `archetype:${parentId}:3` }),
    ],
    rulePasses: [createRulePass({ archetypeId: `rule-pass:${parentId}` })],
  });
}

describe('reproducePolyandric schema alignment (P5)', () => {
  describe('non-overlapping region strategy', () => {
    it('assigns every patchable region to exactly one drone with no unassigned regions', () => {
      // Arrange
      const queen = buildSchemaQueen();
      const drones = [
        { dna: buildSchemaDrone('drone-a'), parentId: 'drone-a' },
        { dna: buildSchemaDrone('drone-b'), parentId: 'drone-b' },
      ];

      // Act
      const result = reproducePolyandric({
        drones,
        ngeEnabled: true,
        policy: createReproductionPolicy({
          assignedRegionStrategy: 'non-overlapping' as const,
          polyandricDroneCount: 2,
          polyandricDroneContributionFraction: 1,
          queenBias: 1,
        }),
        queen,
        queenId: 'queen:schema',
      });

      // Assert
      expect({
        assignedCount: result.regionAssignment?.assignedRegions.length,
        patchableCount: result.regionAssignment?.patchableRegionIds.length,
        unassignedCount: result.regionAssignment?.unassignedRegionIds.length,
      }).toEqual({
        assignedCount: 6,
        patchableCount: 6,
        unassignedCount: 0,
      });
    });

    it('produces the same single-drone-per-region mapping as roundRobin for identical drone order', () => {
      // Arrange
      const queen = buildSchemaQueen();
      const drones = [
        { dna: buildSchemaDrone('drone-a'), parentId: 'drone-a' },
        { dna: buildSchemaDrone('drone-b'), parentId: 'drone-b' },
      ];

      // Act
      const nonOverlappingResult = reproducePolyandric({
        drones,
        ngeEnabled: true,
        policy: createReproductionPolicy({
          assignedRegionStrategy: 'non-overlapping' as const,
          polyandricDroneCount: 2,
          polyandricDroneContributionFraction: 1,
          queenBias: 1,
        }),
        queen,
        queenId: 'queen:schema',
      });

      const roundRobinResult = reproducePolyandric({
        drones,
        ngeEnabled: true,
        policy: createReproductionPolicy({
          assignedRegionStrategy: 'roundRobin',
          polyandricDroneCount: 2,
          polyandricDroneContributionFraction: 1,
          queenBias: 1,
        }),
        queen,
        queenId: 'queen:schema',
      });

      const nonOverlappingAssignments =
        nonOverlappingResult.regionAssignment?.assignedRegions.map(
          ({ regionId, droneId }) => ({ regionId, droneId }),
        );
      const roundRobinAssignments =
        roundRobinResult.regionAssignment?.assignedRegions.map(
          ({ regionId, droneId }) => ({ regionId, droneId }),
        );

      // Assert
      expect(nonOverlappingAssignments).toEqual(roundRobinAssignments);
    });
  });
});
