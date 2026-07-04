import type {
  NgeDnaCanonicalEnvelope,
  NgeReproductionPolicy,
} from '../nge-dna/neat.nge-dna.types';

import { reproducePolyandric } from './neat.nge-evolution.reproduction';
import {
  NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION,
  NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS,
} from './neat.nge-evolution.constants';

const baseReproductionPolicy: NgeReproductionPolicy = {
  mode: 'sexual',
  parthenogenesisMutationRate: 0,
  polyandricDroneCount: 3,
  polyandricDroneContributionFraction:
    NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION,
  queenBias: NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS,
  assignedRegionStrategy: 'roundRobin',
  modeIsEvolvable: true,
  seedPolicy: {
    siblingsDifferBySeed: true,
    twinsAllowed: false,
  },
};

function createReproductionPolicy(
  overrides?: Partial<NgeReproductionPolicy>,
): NgeReproductionPolicy {
  return {
    ...baseReproductionPolicy,
    ...overrides,
  };
}

function createDnaEnvelope(overrides?: {
  cppnPrograms?: NgeDnaCanonicalEnvelope['cppnPrograms'];
  moduleArchetypes?: NgeDnaCanonicalEnvelope['moduleArchetypes'];
  reproductionPolicy?: NgeReproductionPolicy;
  rulePasses?: NgeDnaCanonicalEnvelope['rulePasses'];
}): NgeDnaCanonicalEnvelope {
  return {
    schemaVersion: 'A.1.0',
    compatibilityVersion: '1.0.0',
    encodingMode: 'lossless',
    fingerprint: 'fingerprint:base',
    substrate: {
      dimensions: 3,
      normalization: 'unit-cube',
      zonePartition: {
        x: { count: 1 },
        y: { count: 1 },
        z: { count: 1 },
      },
    },
    reproductionPolicy:
      overrides?.reproductionPolicy ?? createReproductionPolicy(),
    rulePasses: overrides?.rulePasses ?? [],
    cppnPrograms: overrides?.cppnPrograms ?? [],
    moduleArchetypes: overrides?.moduleArchetypes ?? [],
  } as unknown as NgeDnaCanonicalEnvelope;
}

function createModuleArchetype(
  overrides?: Partial<NgeDnaCanonicalEnvelope['moduleArchetypes'][number]>,
): NgeDnaCanonicalEnvelope['moduleArchetypes'][number] {
  const baseArchetype: NgeDnaCanonicalEnvelope['moduleArchetypes'][number] = {
    archetypeId: 'archetype:base',
    computationType: 'DenseFeedForward',
  };
  return { ...baseArchetype, ...overrides };
}

function createRulePass(
  overrides?: Partial<NgeDnaCanonicalEnvelope['rulePasses'][number]>,
): NgeDnaCanonicalEnvelope['rulePasses'][number] {
  const baseRulePass: NgeDnaCanonicalEnvelope['rulePasses'][number] = {
    archetypeId: 'archetype:rule-pass',
    kind: 'replicate',
    placements: [],
    priority: 0,
  };
  return { ...baseRulePass, ...overrides };
}

function createCppnProgram(
  overrides?: Partial<NgeDnaCanonicalEnvelope['cppnPrograms'][number]>,
): NgeDnaCanonicalEnvelope['cppnPrograms'][number] {
  const baseCppnProgram: NgeDnaCanonicalEnvelope['cppnPrograms'][number] = {
    edges: [],
    hiddenNodes: [],
    inputNodeIds: ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'dist'],
    outputNodeIds: ['weight', 'enableBias'],
    programId: 'cppn:base',
  };
  return { ...baseCppnProgram, ...overrides };
}

/**
 * Shared small fixture: queen and drone with distinguishable values across
 * all three DNA families.  Region IDs and their FNV-1a hashes:
 *   cppnPrograms:0     → 0.605378
 *   moduleArchetypes:0 → 0.459978
 *   moduleArchetypes:1 → 0.456072
 *   rulePasses:0       → 0.514796
 */
function buildSharedQueen(): NgeDnaCanonicalEnvelope {
  return createDnaEnvelope({
    moduleArchetypes: [
      createModuleArchetype({
        archetypeId: 'archetype:queen:0',
        parameterSchema: { bias: 100, queenTrait: 200 },
      }),
      createModuleArchetype({
        archetypeId: 'archetype:queen:1',
        parameterSchema: { bias: 101, queenTrait: 201 },
      }),
    ],
    cppnPrograms: [createCppnProgram({ programId: 'cppn:queen' })],
    rulePasses: [createRulePass({ priority: 10 })],
  });
}

function buildSharedDrone(): NgeDnaCanonicalEnvelope {
  return createDnaEnvelope({
    moduleArchetypes: [
      createModuleArchetype({
        archetypeId: 'archetype:drone:0',
        parameterSchema: { bias: 900, droneTrait: 999 },
      }),
      createModuleArchetype({
        archetypeId: 'archetype:drone:1',
        parameterSchema: { bias: 901, droneTrait: 998 },
      }),
    ],
    cppnPrograms: [createCppnProgram({ programId: 'cppn:drone' })],
    rulePasses: [createRulePass({ priority: 99 })],
  });
}

function runSharedFixture(queenBias: number) {
  return reproducePolyandric({
    drones: [{ dna: buildSharedDrone(), parentId: 'drone:shared' }],
    ngeEnabled: true,
    policy: createReproductionPolicy({
      assignedRegionStrategy: 'roundRobin',
      mode: 'polyandric',
      polyandricDroneContributionFraction: 1,
      polyandricDroneCount: 1,
      queenBias,
    }),
    queen: buildSharedQueen(),
    queenId: 'queen:shared',
  });
}

describe('reproducePolyandric queenBias honoring (P5)', () => {
  it('queenBias=1.0 yields queen-wins-all offspring (regression anchor)', () => {
    const result = runSharedFixture(1.0);

    expect({
      firstArchBias: result.offspring.moduleArchetypes[0].parameterSchema?.bias,
      droneTrait:
        result.offspring.moduleArchetypes[0].parameterSchema?.droneTrait,
      cppnProgramId: result.offspring.cppnPrograms[0].programId,
      rulePassPriority: result.offspring.rulePasses[0].priority,
    }).toEqual({
      firstArchBias: 100,
      droneTrait: 999,
      cppnProgramId: 'cppn:queen',
      rulePassPriority: 10,
    });
  });

  it('queenBias=0.0 yields drone-wins-all offspring', () => {
    const result = runSharedFixture(0.0);

    expect({
      firstArchBias: result.offspring.moduleArchetypes[0].parameterSchema?.bias,
      queenTrait:
        result.offspring.moduleArchetypes[0].parameterSchema?.queenTrait,
      cppnProgramId: result.offspring.cppnPrograms[0].programId,
      rulePassPriority: result.offspring.rulePasses[0].priority,
    }).toEqual({
      firstArchBias: 900,
      queenTrait: 200,
      cppnProgramId: 'cppn:drone',
      rulePassPriority: 99,
    });
  });

  it('queenBias=0.5 produces deterministic per-region winner gate', () => {
    const result = runSharedFixture(0.5);

    expect({
      firstArchBias: result.offspring.moduleArchetypes[0].parameterSchema?.bias,
      secondArchBias:
        result.offspring.moduleArchetypes[1].parameterSchema?.bias,
      cppnProgramId: result.offspring.cppnPrograms[0].programId,
      rulePassPriority: result.offspring.rulePasses[0].priority,
    }).toEqual({
      firstArchBias: 100,
      secondArchBias: 101,
      cppnProgramId: 'cppn:drone',
      rulePassPriority: 99,
    });
  });

  it('queenBias=0.85 yields majority queen with minority drone regions', () => {
    const queenArchetypes = Array.from({ length: 101 }, (_, index) =>
      createModuleArchetype({
        archetypeId: `archetype:queen:${index}`,
        parameterSchema: { source: 'queen' },
      }),
    );
    const droneArchetypes = Array.from({ length: 101 }, (_, index) =>
      createModuleArchetype({
        archetypeId: `archetype:drone:${index}`,
        parameterSchema: { source: 'drone' },
      }),
    );

    const result = reproducePolyandric({
      drones: [
        {
          dna: createDnaEnvelope({ moduleArchetypes: droneArchetypes }),
          parentId: 'drone:archetypes',
        },
      ],
      ngeEnabled: true,
      policy: createReproductionPolicy({
        assignedRegionStrategy: 'roundRobin',
        mode: 'polyandric',
        polyandricDroneContributionFraction: 1,
        polyandricDroneCount: 1,
        queenBias: 0.85,
      }),
      queen: createDnaEnvelope({ moduleArchetypes: queenArchetypes }),
      queenId: 'queen:archetypes',
    });

    expect({
      arch0Source: result.offspring.moduleArchetypes[0].parameterSchema?.source,
      arch100Source:
        result.offspring.moduleArchetypes[100].parameterSchema?.source,
    }).toEqual({
      arch0Source: 'queen',
      arch100Source: 'drone',
    });
  });

  it('queenBias=-0.3 is clamped to 0.0 yielding drone-wins-all offspring', () => {
    const result = runSharedFixture(-0.3);

    expect({
      firstArchBias: result.offspring.moduleArchetypes[0].parameterSchema?.bias,
      cppnProgramId: result.offspring.cppnPrograms[0].programId,
    }).toEqual({
      firstArchBias: 900,
      cppnProgramId: 'cppn:drone',
    });
  });

  it('queenBias=1.5 is clamped to 1.0 yielding queen-wins-all offspring', () => {
    const result = runSharedFixture(1.5);

    expect({
      firstArchBias: result.offspring.moduleArchetypes[0].parameterSchema?.bias,
      cppnProgramId: result.offspring.cppnPrograms[0].programId,
    }).toEqual({
      firstArchBias: 100,
      cppnProgramId: 'cppn:queen',
    });
  });

  it('produces identical offspring for identical queen, drones, and queenBias', () => {
    const result1 = runSharedFixture(0.5);
    const result2 = runSharedFixture(0.5);

    expect({
      offspring: result1.offspring,
      regionAssignment: result1.regionAssignment,
    }).toEqual({
      offspring: result2.offspring,
      regionAssignment: result2.regionAssignment,
    });
  });

  it('falls back to empty parameterSchema when queen lacks one and drone wins', () => {
    const queen = createDnaEnvelope({
      moduleArchetypes: [createModuleArchetype()],
    });
    const drone = createDnaEnvelope({
      moduleArchetypes: [
        createModuleArchetype({ parameterSchema: { source: 'drone' } }),
      ],
    });

    const result = reproducePolyandric({
      drones: [{ dna: drone, parentId: 'drone:empty-queen' }],
      ngeEnabled: true,
      policy: createReproductionPolicy({
        mode: 'polyandric',
        polyandricDroneContributionFraction: 1,
        polyandricDroneCount: 1,
        queenBias: 0.0,
      }),
      queen,
      queenId: 'queen:empty',
    });

    expect(result.offspring.moduleArchetypes[0].parameterSchema).toEqual({
      source: 'drone',
    });
  });

  it('falls back to empty parameterSchema when drone lacks one and drone wins', () => {
    const queen = createDnaEnvelope({
      moduleArchetypes: [
        createModuleArchetype({ parameterSchema: { source: 'queen' } }),
      ],
    });
    const drone = createDnaEnvelope({
      moduleArchetypes: [createModuleArchetype()],
    });

    const result = reproducePolyandric({
      drones: [{ dna: drone, parentId: 'drone:no-schema' }],
      ngeEnabled: true,
      policy: createReproductionPolicy({
        mode: 'polyandric',
        polyandricDroneContributionFraction: 1,
        polyandricDroneCount: 1,
        queenBias: 0.0,
      }),
      queen,
      queenId: 'queen:with-schema',
    });

    expect(result.offspring.moduleArchetypes[0].parameterSchema).toEqual({
      source: 'queen',
    });
  });
});

describe('seed policy shorthand expansion (P5)', () => {
  it('expands queen-weighted shorthand to canonical seed policy in offspring', () => {
    const result = reproducePolyandric({
      drones: [{ dna: buildSharedDrone(), parentId: 'drone:seed' }],
      ngeEnabled: true,
      policy: {
        ...createReproductionPolicy({ mode: 'polyandric' }),
        seedPolicy: 'queen-weighted',
      },
      queen: buildSharedQueen(),
      queenId: 'queen:seed',
    });

    expect(result.offspring.reproductionPolicy.seedPolicy).toEqual({
      siblingsDifferBySeed: true,
      twinsAllowed: false,
    });
  });
});
