import type {
  NgeDnaCanonicalEnvelope,
  NgeReproductionPolicy,
} from '../nge-dna/neat.nge-dna.types';

import { computeNgeEvolutionCompatibilityDistance } from './neat.nge-evolution.distance';
import {
  NGE_EVOLUTION_DEFAULT_ALPHA_COMPUTATION,
  NGE_EVOLUTION_DEFAULT_ALPHA_LIFECYCLE,
  NGE_EVOLUTION_DEFAULT_ALPHA_MEMORY,
  NGE_EVOLUTION_DEFAULT_ALPHA_TOPOLOGY,
  NGE_EVOLUTION_DEFAULT_COMPATIBILITY_DISTANCE_WEIGHTS,
  NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY,
  NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION,
  NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS,
} from './neat.nge-evolution.constants';
import {
  NgeEvolution_BudgetError,
  NgeEvolution_ModeError,
  NgeEvolution_RegionError,
} from './neat.nge-evolution.errors';
import { applyNgeEvolutionEpigeneticPrior } from './neat.nge-evolution.epigenetic';
import {
  reproduceParthenogenesis,
  reproducePolyandric,
  reproduceSexual,
} from './neat.nge-evolution.reproduction';
import type {
  NgeEvolutionCompatibilityDistanceResult,
  NgeEvolutionEpigeneticPriorInput,
  NgeEvolutionPolyandricRegionAssignmentResult,
  NgeEvolutionReproductionResult,
} from './neat.nge-evolution.types';

const reproductionPolicyFixture: NgeReproductionPolicy = {
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

const distanceResultFixture: NgeEvolutionCompatibilityDistanceResult = {
  distance: 0.4,
  ngeEnabled: true,
  terms: {
    topology: {
      name: 'topology',
      rawDistance: 1,
      normalizedDistance: 0.4,
      weight: NGE_EVOLUTION_DEFAULT_ALPHA_TOPOLOGY,
      weightedDistance: 0.16,
    },
    computation: {
      name: 'computation',
      rawDistance: 0.5,
      normalizedDistance: 0.5,
      weight: NGE_EVOLUTION_DEFAULT_ALPHA_COMPUTATION,
      weightedDistance: 0.1,
    },
    memory: {
      name: 'memory',
      rawDistance: 0.3,
      normalizedDistance: 0.3,
      weight: NGE_EVOLUTION_DEFAULT_ALPHA_MEMORY,
      weightedDistance: 0.06,
    },
    lifecycle: {
      name: 'lifecycle',
      rawDistance: 0.4,
      normalizedDistance: 0.4,
      weight: NGE_EVOLUTION_DEFAULT_ALPHA_LIFECYCLE,
      weightedDistance: 0.08,
    },
  },
};

const regionAssignmentFixture: NgeEvolutionPolyandricRegionAssignmentResult = {
  strategy: 'roundRobin',
  patchableRegionIds: ['region:0', 'region:1'],
  assignedRegions: [
    {
      regionId: 'region:0',
      droneId: 'drone:0',
      droneRank: 0,
      droneFitness: 12,
    },
  ],
  unassignedRegionIds: ['region:1'],
};

const epigeneticInputFixture: NgeEvolutionEpigeneticPriorInput = {
  childParameters: [0.2, 0.4],
  mutationDelta: [0.05, -0.05],
  reference: {
    firstParentParameters: [0.1, 0.2],
    secondParentParameters: [0.3, 0.6],
    blendedReference: [0.2, 0.4],
  },
  decay: NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY,
};

const reproductionResultFixture: NgeEvolutionReproductionResult<{
  fingerprint: string;
}> = {
  policy: reproductionPolicyFixture,
  outcome: 'sexual-crossover',
  offspring: {
    fingerprint: 'offspring:fingerprint',
  },
  parentContributions: [
    {
      parentId: 'queen:0',
      role: 'primary',
      contributionKind: 'crossover',
      regionIds: ['region:0'],
    },
    {
      parentId: 'drone:0',
      role: 'secondary',
      contributionKind: 'blend',
      regionIds: ['region:1'],
    },
  ],
  regionAssignment: regionAssignmentFixture,
  epigeneticReference: epigeneticInputFixture.reference,
};

describe('nge evolution Step 02 shelf', () => {
  describe('constants', () => {
    it('exports the default composite compatibility-distance weights', () => {
      expect(NGE_EVOLUTION_DEFAULT_COMPATIBILITY_DISTANCE_WEIGHTS).toEqual({
        topology: 0.4,
        computation: 0.2,
        memory: 0.2,
        lifecycle: 0.2,
      });
    });

    it('exports the default epigenetic and polyandric knobs', () => {
      expect({
        decay: NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY,
        droneContribution:
          NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION,
        queenBias: NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS,
      }).toEqual({
        decay: 0.05,
        droneContribution: 0.1,
        queenBias: 1,
      });
    });
  });

  describe('type shelf', () => {
    it('keeps the downstream NGE result contracts assignable', () => {
      expect({
        decay: epigeneticInputFixture.decay,
        distance: distanceResultFixture.distance,
        mode: reproductionResultFixture.policy.mode,
        outcome: reproductionResultFixture.outcome,
        strategy: regionAssignmentFixture.strategy,
      }).toEqual({
        decay: 0.05,
        distance: 0.4,
        mode: 'sexual',
        outcome: 'sexual-crossover',
        strategy: 'roundRobin',
      });
    });
  });

  describe('NgeEvolution_ModeError', () => {
    it('preserves the configured message, name, and cause', () => {
      const cause = { reason: 'NGE disabled for parthenogenesis' };
      const error = new NgeEvolution_ModeError(
        'requested reproduction mode is unavailable',
        { cause },
      );

      expect({
        cause: error.cause,
        message: error.message,
        name: error.name,
      }).toEqual({
        cause,
        message: 'requested reproduction mode is unavailable',
        name: 'NgeEvolution_ModeError',
      });
    });

    it('keeps the cause undefined when no options are provided', () => {
      const error = new NgeEvolution_ModeError(
        'requested reproduction mode is unavailable',
      );

      expect({
        cause: error.cause,
        message: error.message,
        name: error.name,
      }).toEqual({
        cause: undefined,
        message: 'requested reproduction mode is unavailable',
        name: 'NgeEvolution_ModeError',
      });
    });
  });

  describe('NgeEvolution_RegionError', () => {
    it('preserves the configured message, name, and cause', () => {
      const cause = { reason: 'region assignment overflow' };
      const error = new NgeEvolution_RegionError(
        'polyandric region assignment failed',
        { cause },
      );

      expect({
        cause: error.cause,
        message: error.message,
        name: error.name,
      }).toEqual({
        cause,
        message: 'polyandric region assignment failed',
        name: 'NgeEvolution_RegionError',
      });
    });

    it('keeps the cause undefined when no options are provided', () => {
      const error = new NgeEvolution_RegionError(
        'polyandric region assignment failed',
      );

      expect({
        cause: error.cause,
        message: error.message,
        name: error.name,
      }).toEqual({
        cause: undefined,
        message: 'polyandric region assignment failed',
        name: 'NgeEvolution_RegionError',
      });
    });
  });

  describe('NgeEvolution_BudgetError', () => {
    it('preserves the configured message, name, and cause', () => {
      const cause = { reason: 'patch budget exceeded' };
      const error = new NgeEvolution_BudgetError(
        'evolution operator exceeded the configured budget',
        { cause },
      );

      expect({
        cause: error.cause,
        message: error.message,
        name: error.name,
      }).toEqual({
        cause,
        message: 'evolution operator exceeded the configured budget',
        name: 'NgeEvolution_BudgetError',
      });
    });

    it('keeps the cause undefined when no options are provided', () => {
      const error = new NgeEvolution_BudgetError(
        'evolution operator exceeded the configured budget',
      );

      expect({
        cause: error.cause,
        message: error.message,
        name: error.name,
      }).toEqual({
        cause: undefined,
        message: 'evolution operator exceeded the configured budget',
        name: 'NgeEvolution_BudgetError',
      });
    });
  });
});

describe('nge evolution reproduction operators', () => {
  describe('applyNgeEvolutionEpigeneticPrior', () => {
    it('returns the original child vector without parameter allocation when unconfigured', () => {
      const childParameters = [0.2, 0.4];
      const result = applyNgeEvolutionEpigeneticPrior({
        childParameters,
        mutationDelta: [0.05, -0.05],
        reference: null,
      });

      expect({
        appliedDecay: result.appliedDecay,
        blendedReference: result.blendedReference,
        outputParameters: result.outputParameters,
        referenceApplied: result.referenceApplied,
        sameReference: result.outputParameters === childParameters,
      }).toEqual({
        appliedDecay: 0,
        blendedReference: null,
        outputParameters: [0.2, 0.4],
        referenceApplied: false,
        sameReference: true,
      });
    });

    it('applies the same deterministic birth-time nudge for identical configured inputs', () => {
      const input = {
        childParameters: [1, 2],
        mutationDelta: [0, 0],
        reference: {
          firstParentParameters: [11, 22],
          secondParentParameters: [31, 42],
          blendedReference: [999, 999],
        },
      };
      const firstResult = applyNgeEvolutionEpigeneticPrior(input);
      const secondResult = applyNgeEvolutionEpigeneticPrior(input);

      expect({
        appliedDecay: firstResult.appliedDecay,
        firstOutput: firstResult.outputParameters.map((parameter) =>
          Number(parameter.toFixed(6)),
        ),
        referenceApplied: firstResult.referenceApplied,
        secondOutput: secondResult.outputParameters.map((parameter) =>
          Number(parameter.toFixed(6)),
        ),
      }).toEqual({
        appliedDecay: NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY,
        firstOutput: [2, 3.5],
        referenceApplied: true,
        secondOutput: [2, 3.5],
      });
    });

    it('scales the weak-parent pull by the injected decay override', () => {
      const result = applyNgeEvolutionEpigeneticPrior({
        childParameters: [1, 2],
        mutationDelta: [0.5, -0.25],
        reference: {
          firstParentParameters: [2, 4],
          secondParentParameters: [0, 6],
          blendedReference: [500, 500],
        },
        decay: 0.2,
      });

      expect({
        appliedDecay: result.appliedDecay,
        outputParameters: result.outputParameters.map((parameter) =>
          Number(parameter.toFixed(6)),
        ),
      }).toEqual({
        appliedDecay: 0.2,
        outputParameters: [1.5, 2.35],
      });
    });

    it('recomputes the blended anchor from both parent reference vectors', () => {
      const result = applyNgeEvolutionEpigeneticPrior({
        childParameters: [0, 0],
        mutationDelta: [0, 0],
        reference: {
          firstParentParameters: [0.2, 0.8],
          secondParentParameters: [0.6, 0.4],
          blendedReference: [9, 9],
        },
        decay: 0.1,
      });

      expect(
        result.blendedReference?.map((parameter) =>
          Number(parameter.toFixed(6)),
        ),
      ).toEqual([0.4, 0.6]);
    });

    it('falls back to child-owned defaults when optional reference shelves are shorter', () => {
      const result = applyNgeEvolutionEpigeneticPrior({
        childParameters: [2, 4],
        mutationDelta: [0.5],
        reference: {
          firstParentParameters: [6],
          secondParentParameters: [],
          blendedReference: [],
        },
        decay: 0.5,
      });

      expect({
        blendedReference: result.blendedReference?.map((parameter) =>
          Number(parameter.toFixed(6)),
        ),
        outputParameters: result.outputParameters.map((parameter) =>
          Number(parameter.toFixed(6)),
        ),
      }).toEqual({
        blendedReference: [4, 4],
        outputParameters: [3.5, 4],
      });
    });
  });

  describe('reproduceParthenogenesis', () => {
    it('returns a true clone when the mutation rate is zero', () => {
      const parent = createDnaEnvelope({
        cppnPrograms: [createCppnProgram({ programId: 'cppn:clone:0' })],
        moduleArchetypes: [
          createModuleArchetype({
            archetypeId: 'archetype:clone:0',
            parameterSchema: { bias: 1 },
          }),
        ],
        rulePasses: [createRulePass({ archetypeId: 'archetype:clone:rule:0' })],
      });
      const result = reproduceParthenogenesis({
        ngeEnabled: true,
        parent,
        parentId: 'parent:clone',
        policy: createReproductionPolicy({
          mode: 'parthenogenesis',
          parthenogenesisMutationRate: 0,
        }),
      });

      expect({
        contributionKind: result.parentContributions[0].contributionKind,
        moduleArchetypes: result.offspring.moduleArchetypes,
        outcome: result.outcome,
        policyMode: result.policy.mode,
      }).toEqual({
        contributionKind: 'clone',
        moduleArchetypes: parent.moduleArchetypes,
        outcome: 'clone',
        policyMode: 'parthenogenesis',
      });
    });

    it('applies only the mutation callback when the rate is non-zero', () => {
      let observedMutationRate = -1;
      const parent = createDnaEnvelope({
        moduleArchetypes: [
          createModuleArchetype({
            archetypeId: 'archetype:mutation:0',
            parameterSchema: { bias: 1 },
          }),
        ],
      });
      const result = reproduceParthenogenesis(
        {
          ngeEnabled: true,
          parent,
          parentId: 'parent:mutation',
          policy: createReproductionPolicy({
            mode: 'parthenogenesis',
            parthenogenesisMutationRate: 0.25,
          }),
        },
        (canonicalEnvelope: NgeDnaCanonicalEnvelope, mutationRate: number) => {
          observedMutationRate = mutationRate;

          return createDnaEnvelope({
            cppnPrograms: canonicalEnvelope.cppnPrograms,
            moduleArchetypes: [
              createModuleArchetype({
                ...canonicalEnvelope.moduleArchetypes[0],
                parameterSchema: { bias: 9 },
              }),
            ],
            reproductionPolicy: canonicalEnvelope.reproductionPolicy,
            rulePasses: canonicalEnvelope.rulePasses,
          });
        },
      );

      expect({
        contributionKind: result.parentContributions[0].contributionKind,
        moduleBias: result.offspring.moduleArchetypes[0].parameterSchema?.bias,
        mutationRate: observedMutationRate,
        parentCount: result.parentContributions.length,
      }).toEqual({
        contributionKind: 'mutation',
        moduleBias: 9,
        mutationRate: 0.25,
        parentCount: 1,
      });
    });

    it('throws a mode error when NGE is disabled', () => {
      expect(() =>
        reproduceParthenogenesis({
          ngeEnabled: false,
          parent: createDnaEnvelope(),
          parentId: 'parent:disabled',
          policy: createReproductionPolicy({ mode: 'parthenogenesis' }),
        }),
      ).toThrow(NgeEvolution_ModeError);
    });

    it('falls back to identity mutation when no callback is supplied', () => {
      const parent = createDnaEnvelope({
        moduleArchetypes: [
          createModuleArchetype({ archetypeId: 'archetype:identity:0' }),
        ],
        reproductionPolicy: createReproductionPolicy({
          mode: 'parthenogenesis',
          parthenogenesisMutationRate: 0.2,
        }),
      });
      const result = reproduceParthenogenesis({
        ngeEnabled: true,
        parent,
        parentId: 'parent:identity',
      });

      expect({
        outcome: result.outcome,
        offspringIds: result.offspring.moduleArchetypes.map(
          ({
            archetypeId,
          }: NgeDnaCanonicalEnvelope['moduleArchetypes'][number]) =>
            archetypeId,
        ),
      }).toEqual({
        outcome: 'mutation-only',
        offspringIds: ['archetype:identity:0'],
      });
    });
  });

  describe('reproducePolyandric', () => {
    it('keeps queen conflicts while patching non-overlapping round-robin regions', () => {
      const queen = createDnaEnvelope({
        moduleArchetypes: [
          createModuleArchetype({
            archetypeId: 'archetype:queen:0',
            parameterSchema: { bias: 1, preserved: 7 },
          }),
          createModuleArchetype({
            archetypeId: 'archetype:queen:1',
            parameterSchema: { bias: 2 },
          }),
        ],
        rulePasses: [
          createRulePass({
            archetypeId: 'archetype:queen:rule:0',
            priority: 3,
          }),
        ],
      });
      const result = reproducePolyandric({
        drones: [
          {
            dna: createDnaEnvelope({
              moduleArchetypes: [
                createModuleArchetype({
                  archetypeId: 'archetype:drone:0',
                  parameterSchema: { bias: 99, drift: 5 },
                }),
              ],
              rulePasses: [
                createRulePass({
                  archetypeId: 'archetype:drone:rule:0',
                  priority: 99,
                }),
              ],
            }),
            parentId: 'drone:round-robin',
          },
        ],
        ngeEnabled: true,
        policy: createReproductionPolicy({
          assignedRegionStrategy: 'roundRobin',
          mode: 'polyandric',
          polyandricDroneContributionFraction: 1,
          polyandricDroneCount: 1,
        }),
        queen,
        queenId: 'queen:round-robin',
      });

      expect({
        assignedRegionIds: result.regionAssignment?.assignedRegions.map(
          ({
            regionId,
          }: NgeEvolutionPolyandricRegionAssignmentResult['assignedRegions'][number]) =>
            regionId,
        ),
        firstBias: result.offspring.moduleArchetypes[0].parameterSchema?.bias,
        firstDrift: result.offspring.moduleArchetypes[0].parameterSchema?.drift,
        outcome: result.outcome,
        rulePassPriority: result.offspring.rulePasses[0].priority,
        secondBias: result.offspring.moduleArchetypes[1].parameterSchema?.bias,
      }).toEqual({
        assignedRegionIds: [
          'moduleArchetypes:0',
          'moduleArchetypes:1',
          'rulePasses:0',
        ],
        firstBias: 1,
        firstDrift: 5,
        outcome: 'queen-template-patched',
        rulePassPriority: 3,
        secondBias: 2,
      });
    });

    it('assigns donors by descending fitness when requested', () => {
      const result = reproducePolyandric({
        drones: [
          {
            dna: createDnaEnvelope({
              moduleArchetypes: [createModuleArchetype()],
            }),
            parentId: 'drone:fitness:low',
          },
          {
            dna: createDnaEnvelope({
              moduleArchetypes: [createModuleArchetype()],
            }),
            fitness: 10,
            parentId: 'drone:fitness:high',
          },
        ],
        ngeEnabled: true,
        policy: createReproductionPolicy({
          assignedRegionStrategy: 'byFitness',
          mode: 'polyandric',
          polyandricDroneContributionFraction: 1,
          polyandricDroneCount: 2,
        }),
        queen: createDnaEnvelope({
          moduleArchetypes: [
            createModuleArchetype({ archetypeId: 'archetype:fitness:0' }),
            createModuleArchetype({ archetypeId: 'archetype:fitness:1' }),
          ],
        }),
        queenId: 'queen:fitness',
      });

      expect(
        result.regionAssignment?.assignedRegions.map(
          ({
            droneId,
            droneRank,
          }: NgeEvolutionPolyandricRegionAssignmentResult['assignedRegions'][number]) => ({
            droneId,
            droneRank,
          }),
        ),
      ).toEqual([
        {
          droneId: 'drone:fitness:high',
          droneRank: 0,
        },
        {
          droneId: 'drone:fitness:low',
          droneRank: 1,
        },
      ]);
    });

    it('breaks equal-fitness ties by donor id', () => {
      const result = reproducePolyandric({
        drones: [
          {
            dna: createDnaEnvelope({
              moduleArchetypes: [createModuleArchetype()],
            }),
            parentId: 'drone:tie:b',
          },
          {
            dna: createDnaEnvelope({
              moduleArchetypes: [createModuleArchetype()],
            }),
            parentId: 'drone:tie:a',
          },
        ],
        ngeEnabled: true,
        policy: createReproductionPolicy({
          assignedRegionStrategy: 'byFitness',
          mode: 'polyandric',
          polyandricDroneContributionFraction: 1,
          polyandricDroneCount: 2,
        }),
        queen: createDnaEnvelope({
          moduleArchetypes: [
            createModuleArchetype({ archetypeId: 'archetype:tie:0' }),
          ],
        }),
        queenId: 'queen:tie',
      });

      expect(result.regionAssignment?.assignedRegions[0].droneId).toEqual(
        'drone:tie:a',
      );
    });

    it('prefers specialization matches and falls back when none exist', () => {
      const result = reproducePolyandric({
        drones: [
          {
            dna: createDnaEnvelope({
              cppnPrograms: [
                createCppnProgram({ programId: 'cppn:specialist' }),
              ],
            }),
            parentId: 'drone:specialist',
            specializationKey: 'cppnPrograms',
          },
          {
            dna: createDnaEnvelope({
              moduleArchetypes: [
                createModuleArchetype({
                  archetypeId: 'archetype:fallback',
                  parameterSchema: { bias: 4 },
                }),
              ],
            }),
            parentId: 'drone:fallback',
            specializationKey: 'residual',
          },
        ],
        ngeEnabled: true,
        policy: createReproductionPolicy({
          assignedRegionStrategy: 'bySpecialization',
          mode: 'polyandric',
          polyandricDroneContributionFraction: 1,
          polyandricDroneCount: 2,
        }),
        queen: createDnaEnvelope({
          cppnPrograms: [createCppnProgram({ programId: 'cppn:queen' })],
          moduleArchetypes: [
            createModuleArchetype({
              archetypeId: 'archetype:queen:specialization',
              parameterSchema: { bias: 1 },
            }),
          ],
        }),
        queenId: 'queen:specialization',
      });

      expect({
        assignedDroneIds: result.regionAssignment?.assignedRegions.map(
          ({
            droneId,
          }: NgeEvolutionPolyandricRegionAssignmentResult['assignedRegions'][number]) =>
            droneId,
        ),
        cppnProgramId: result.offspring.cppnPrograms[0]?.programId,
        moduleBias: result.offspring.moduleArchetypes[0].parameterSchema?.bias,
      }).toEqual({
        assignedDroneIds: ['drone:specialist', 'drone:fallback'],
        cppnProgramId: 'cppn:queen',
        moduleBias: 1,
      });
    });

    it('throws a mode error when NGE is disabled', () => {
      expect(() =>
        reproducePolyandric({
          drones: [],
          ngeEnabled: false,
          policy: createReproductionPolicy({ mode: 'polyandric' }),
          queen: createDnaEnvelope(),
          queenId: 'queen:disabled',
        }),
      ).toThrow(NgeEvolution_ModeError);
    });

    it('reports unassigned regions when no drones are available', () => {
      const result = reproducePolyandric({
        drones: [],
        ngeEnabled: true,
        queen: createDnaEnvelope({
          moduleArchetypes: [
            createModuleArchetype({ archetypeId: 'archetype:unassigned:0' }),
          ],
          reproductionPolicy: createReproductionPolicy({
            assignedRegionStrategy: 'roundRobin',
            mode: 'polyandric',
            polyandricDroneContributionFraction: 1,
            polyandricDroneCount: 0,
          }),
        }),
        queenId: 'queen:unassigned',
      });

      expect({
        offspringArchetypeId: result.offspring.moduleArchetypes[0].archetypeId,
        unassignedRegionIds: result.regionAssignment?.unassignedRegionIds,
      }).toEqual({
        offspringArchetypeId: 'archetype:unassigned:0',
        unassignedRegionIds: ['moduleArchetypes:0'],
      });
    });
  });

  describe('reproduceSexual', () => {
    it('keeps fitter-parent disjoint regions and uses the injected RNG for matches', () => {
      const result = reproduceSexual(
        {
          firstParent: createDnaEnvelope({
            cppnPrograms: [createCppnProgram({ programId: 'cppn:shared' })],
            moduleArchetypes: [
              createModuleArchetype({
                archetypeId: 'shared',
                parameterSchema: { bias: 1 },
              }),
              createModuleArchetype({ archetypeId: 'first-only' }),
            ],
            rulePasses: [
              createRulePass({
                archetypeId: 'first-only:rule',
                priority: 1,
              }),
            ],
          }),
          firstParentId: 'parent:first',
          firstParentScore: 5,
          policy: createReproductionPolicy({ mode: 'sexual' }),
          secondParent: createDnaEnvelope({
            cppnPrograms: [createCppnProgram({ programId: 'cppn:shared' })],
            moduleArchetypes: [
              createModuleArchetype({
                archetypeId: 'shared',
                parameterSchema: { bias: 9 },
              }),
              createModuleArchetype({ archetypeId: 'second-only' }),
            ],
            rulePasses: [
              createRulePass({
                archetypeId: 'second-only:rule',
                priority: 2,
              }),
            ],
          }),
          secondParentId: 'parent:second',
          secondParentScore: 2,
        },
        () => 0.25,
      );

      expect({
        selectedCppnProgramIds: result.offspring.cppnPrograms.map(
          ({ programId }: NgeDnaCanonicalEnvelope['cppnPrograms'][number]) =>
            programId,
        ),
        outcome: result.outcome,
        parentIds: result.parentContributions.map(({ parentId }) => parentId),
        selectedRulePassIds: result.offspring.rulePasses.map(
          ({ archetypeId }: NgeDnaCanonicalEnvelope['rulePasses'][number]) =>
            archetypeId,
        ),
        selectedBias:
          result.offspring.moduleArchetypes[0].parameterSchema?.bias,
        selectedIds: result.offspring.moduleArchetypes.map(
          ({
            archetypeId,
          }: NgeDnaCanonicalEnvelope['moduleArchetypes'][number]) =>
            archetypeId,
        ),
      }).toEqual({
        selectedCppnProgramIds: ['cppn:shared'],
        outcome: 'sexual-crossover',
        parentIds: ['parent:first', 'parent:second'],
        selectedRulePassIds: ['first-only:rule'],
        selectedBias: 9,
        selectedIds: ['shared', 'first-only'],
      });
    });

    it('keeps second-parent disjoint regions when the second parent is fitter', () => {
      const result = reproduceSexual({
        firstParent: createDnaEnvelope({
          moduleArchetypes: [
            createModuleArchetype({
              archetypeId: 'shared',
              parameterSchema: { bias: 3 },
            }),
            createModuleArchetype({ archetypeId: 'first-only:weaker' }),
          ],
        }),
        firstParentId: 'parent:first:weaker',
        firstParentScore: 1,
        secondParent: createDnaEnvelope({
          moduleArchetypes: [
            createModuleArchetype({
              archetypeId: 'shared',
              parameterSchema: { bias: 8 },
            }),
            createModuleArchetype({ archetypeId: 'second-only:fitter' }),
          ],
        }),
        secondParentId: 'parent:second:fitter',
        secondParentScore: 4,
      });

      expect({
        roles: result.parentContributions.map(({ role }) => role),
        selectedBias:
          result.offspring.moduleArchetypes[0].parameterSchema?.bias,
        selectedIds: result.offspring.moduleArchetypes.map(
          ({
            archetypeId,
          }: NgeDnaCanonicalEnvelope['moduleArchetypes'][number]) =>
            archetypeId,
        ),
      }).toEqual({
        roles: ['secondary', 'primary'],
        selectedBias: 3,
        selectedIds: ['shared', 'second-only:fitter'],
      });
    });
  });
});

describe('computeNgeEvolutionCompatibilityDistance', () => {
  it('isolates the computation term from motif composition drift', () => {
    const comparison = createCompatibilityComparison({
      leftDna: createDnaEnvelope({
        moduleArchetypes: [
          {
            archetypeId: 'archetype:attention:0',
            computationType: 'AttentionHead',
          },
          {
            archetypeId: 'archetype:dense:0',
            computationType: 'DenseFeedForward',
          },
        ],
      }),
      rightDna: createDnaEnvelope({
        moduleArchetypes: [
          {
            archetypeId: 'archetype:dense:1',
            computationType: 'DenseFeedForward',
          },
          {
            archetypeId: 'archetype:dense:2',
            computationType: 'DenseFeedForward',
          },
        ],
      }),
    });
    const result = computeNgeEvolutionCompatibilityDistance(comparison, {
      populationSlice: [createCompatibilityComparison()],
    });

    expect(roundCompatibilityDistanceResult(result)).toEqual({
      distance: 0.2,
      ngeEnabled: true,
      terms: {
        topology: {
          name: 'topology',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.4,
          weightedDistance: 0,
        },
        computation: {
          name: 'computation',
          rawDistance: 0.5,
          normalizedDistance: 1,
          weight: 0.2,
          weightedDistance: 0.2,
        },
        memory: {
          name: 'memory',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.2,
          weightedDistance: 0,
        },
        lifecycle: {
          name: 'lifecycle',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.2,
          weightedDistance: 0,
        },
      },
    });
  });

  it('isolates the memory term from tier presence and capacity bins', () => {
    const comparison = createCompatibilityComparison({
      leftDna: createDnaEnvelope({
        moduleArchetypes: [
          {
            archetypeId: 'archetype:recurrent:0',
            computationType: 'GatedRecurrentCell',
            parameterSchema: {
              hiddenDim: 4,
            },
          },
          {
            archetypeId: 'archetype:episodic:0',
            computationType: 'EpisodicSlot',
            parameterSchema: {
              slotCount: 4,
            },
          },
        ],
      }),
      rightDna: createDnaEnvelope({
        moduleArchetypes: [
          {
            archetypeId: 'archetype:recurrent:1',
            computationType: 'GatedRecurrentCell',
          },
          {
            archetypeId: 'archetype:episodic:1',
            computationType: 'EpisodicSlot',
            parameterSchema: {
              slotCount: 16,
            },
          },
        ],
      }),
    });
    const result = computeNgeEvolutionCompatibilityDistance(comparison, {
      populationSlice: [createCompatibilityComparison()],
    });

    expect(roundCompatibilityDistanceResult(result)).toEqual({
      distance: 0.2,
      ngeEnabled: true,
      terms: {
        topology: {
          name: 'topology',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.4,
          weightedDistance: 0,
        },
        computation: {
          name: 'computation',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.2,
          weightedDistance: 0,
        },
        memory: {
          name: 'memory',
          rawDistance: 0.417,
          normalizedDistance: 1,
          weight: 0.2,
          weightedDistance: 0.2,
        },
        lifecycle: {
          name: 'lifecycle',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.2,
          weightedDistance: 0,
        },
      },
    });
  });

  it('isolates the lifecycle term from reproduction and governance drift', () => {
    const comparison = createCompatibilityComparison({
      leftDna: createDnaEnvelope({
        reproductionPolicy: createReproductionPolicy({
          mode: 'sexual',
        }),
      }),
      rightDna: createDnaEnvelope({
        reproductionPolicy: createReproductionPolicy({
          mode: 'polyandric',
        }),
      }),
      leftLifecycleTraits: {
        assimilationCadence: 2,
        wiringCostWeights: {
          nodeWeight: 1,
        },
      },
      rightLifecycleTraits: {
        assimilationCadence: 4,
        wiringCostWeights: {
          nodeWeight: 2,
        },
      },
    });
    const result = computeNgeEvolutionCompatibilityDistance(comparison, {
      populationSlice: [createCompatibilityComparison()],
    });

    expect(roundCompatibilityDistanceResult(result)).toEqual({
      distance: 0.2,
      ngeEnabled: true,
      terms: {
        topology: {
          name: 'topology',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.4,
          weightedDistance: 0,
        },
        computation: {
          name: 'computation',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.2,
          weightedDistance: 0,
        },
        memory: {
          name: 'memory',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.2,
          weightedDistance: 0,
        },
        lifecycle: {
          name: 'lifecycle',
          rawDistance: 0.667,
          normalizedDistance: 1,
          weight: 0.2,
          weightedDistance: 0.2,
        },
      },
    });
  });

  it('handles computation motifs that appear only on the right genome', () => {
    const comparison = createCompatibilityComparison({
      leftDna: createDnaEnvelope({
        moduleArchetypes: [
          {
            archetypeId: 'archetype:dense:right-only:0',
            computationType: 'DenseFeedForward',
          },
        ],
      }),
      rightDna: createDnaEnvelope({
        moduleArchetypes: [
          {
            archetypeId: 'archetype:dense:right-only:1',
            computationType: 'DenseFeedForward',
          },
          {
            archetypeId: 'archetype:attention:right-only:0',
            computationType: 'AttentionHead',
          },
        ],
      }),
    });
    const result = computeNgeEvolutionCompatibilityDistance(comparison, {
      populationSlice: [createCompatibilityComparison()],
    });

    expect(roundCompatibilityDistanceResult(result)).toEqual({
      distance: 0.2,
      ngeEnabled: true,
      terms: {
        topology: {
          name: 'topology',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.4,
          weightedDistance: 0,
        },
        computation: {
          name: 'computation',
          rawDistance: 0.5,
          normalizedDistance: 1,
          weight: 0.2,
          weightedDistance: 0.2,
        },
        memory: {
          name: 'memory',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.2,
          weightedDistance: 0,
        },
        lifecycle: {
          name: 'lifecycle',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.2,
          weightedDistance: 0,
        },
      },
    });
  });

  it('maps overflow memory capacities into the top comparison bin', () => {
    const comparison = createCompatibilityComparison({
      leftDna: createDnaEnvelope({
        moduleArchetypes: [
          {
            archetypeId: 'archetype:recurrent:overflow:0',
            computationType: 'GatedRecurrentCell',
            parameterSchema: {
              hiddenDim: 32,
            },
          },
          {
            archetypeId: 'archetype:episodic:overflow:0',
            computationType: 'EpisodicSlot',
            parameterSchema: {
              slotCount: 32,
            },
          },
        ],
      }),
      rightDna: createDnaEnvelope({
        moduleArchetypes: [
          {
            archetypeId: 'archetype:recurrent:overflow:1',
            computationType: 'GatedRecurrentCell',
            parameterSchema: {
              hiddenDim: 4,
            },
          },
          {
            archetypeId: 'archetype:episodic:overflow:1',
            computationType: 'EpisodicSlot',
            parameterSchema: {
              slotCount: 4,
            },
          },
        ],
      }),
    });
    const result = computeNgeEvolutionCompatibilityDistance(comparison, {
      populationSlice: [createCompatibilityComparison()],
    });

    expect(roundCompatibilityDistanceResult(result)).toEqual({
      distance: 0.2,
      ngeEnabled: true,
      terms: {
        topology: {
          name: 'topology',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.4,
          weightedDistance: 0,
        },
        computation: {
          name: 'computation',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.2,
          weightedDistance: 0,
        },
        memory: {
          name: 'memory',
          rawDistance: 0.375,
          normalizedDistance: 1,
          weight: 0.2,
          weightedDistance: 0.2,
        },
        lifecycle: {
          name: 'lifecycle',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.2,
          weightedDistance: 0,
        },
      },
    });
  });

  it('compares sparse wiring-preference keys across both directions of absence', () => {
    const comparison = createCompatibilityComparison({
      leftDna: createDnaEnvelope({
        reproductionPolicy: createReproductionPolicy({
          mode: 'sexual',
        }),
      }),
      rightDna: createDnaEnvelope({
        reproductionPolicy: createReproductionPolicy({
          mode: 'sexual',
        }),
      }),
      leftLifecycleTraits: {
        wiringCostWeights: {
          nodeWeight: 2,
        },
      },
      rightLifecycleTraits: {
        wiringCostWeights: {
          edgeWeight: 4,
        },
      },
    });
    const result = computeNgeEvolutionCompatibilityDistance(comparison, {
      populationSlice: [createCompatibilityComparison()],
    });

    expect(roundCompatibilityDistanceResult(result)).toEqual({
      distance: 0.2,
      ngeEnabled: true,
      terms: {
        topology: {
          name: 'topology',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.4,
          weightedDistance: 0,
        },
        computation: {
          name: 'computation',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.2,
          weightedDistance: 0,
        },
        memory: {
          name: 'memory',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.2,
          weightedDistance: 0,
        },
        lifecycle: {
          name: 'lifecycle',
          rawDistance: 0.333,
          normalizedDistance: 1,
          weight: 0.2,
          weightedDistance: 0.2,
        },
      },
    });
  });

  it('collapses the NGE-only terms to zero when NGE is disabled', () => {
    const result = computeNgeEvolutionCompatibilityDistance(
      createCompatibilityComparison({
        ngeEnabled: false,
        topologyDistance: 2.5,
      }),
    );

    expect(roundCompatibilityDistanceResult(result)).toEqual({
      distance: 2.5,
      ngeEnabled: false,
      terms: {
        topology: {
          name: 'topology',
          rawDistance: 2.5,
          normalizedDistance: 2.5,
          weight: 1,
          weightedDistance: 2.5,
        },
        computation: {
          name: 'computation',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0,
          weightedDistance: 0,
        },
        memory: {
          name: 'memory',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0,
          weightedDistance: 0,
        },
        lifecycle: {
          name: 'lifecycle',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0,
          weightedDistance: 0,
        },
      },
    });
  });

  it('normalizes injected weights so the composite sum stays bounded by one', () => {
    const result = computeNgeEvolutionCompatibilityDistance(
      createCompatibilityComparison({
        topologyDistance: 3,
        leftDna: createDnaEnvelope({
          moduleArchetypes: [
            {
              archetypeId: 'archetype:attention:1',
              computationType: 'AttentionHead',
            },
            {
              archetypeId: 'archetype:dense:4',
              computationType: 'DenseFeedForward',
            },
          ],
        }),
        rightDna: createDnaEnvelope({
          moduleArchetypes: [
            {
              archetypeId: 'archetype:dense:5',
              computationType: 'DenseFeedForward',
            },
            {
              archetypeId: 'archetype:dense:6',
              computationType: 'DenseFeedForward',
            },
          ],
        }),
      }),
      {
        weights: {
          topology: 1,
          computation: 1,
          memory: 0,
          lifecycle: 0,
        },
      },
    );

    expect(roundCompatibilityDistanceResult(result)).toEqual({
      distance: 1,
      ngeEnabled: true,
      terms: {
        topology: {
          name: 'topology',
          rawDistance: 3,
          normalizedDistance: 1,
          weight: 0.5,
          weightedDistance: 0.5,
        },
        computation: {
          name: 'computation',
          rawDistance: 0.5,
          normalizedDistance: 1,
          weight: 0.5,
          weightedDistance: 0.5,
        },
        memory: {
          name: 'memory',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0,
          weightedDistance: 0,
        },
        lifecycle: {
          name: 'lifecycle',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0,
          weightedDistance: 0,
        },
      },
    });
  });

  it('keeps every weight at zero when the injected bag sums to zero', () => {
    const result = computeNgeEvolutionCompatibilityDistance(
      createCompatibilityComparison({
        topologyDistance: 3,
        leftDna: createDnaEnvelope({
          moduleArchetypes: [
            {
              archetypeId: 'archetype:attention:zero-weight:0',
              computationType: 'AttentionHead',
            },
            {
              archetypeId: 'archetype:dense:zero-weight:0',
              computationType: 'DenseFeedForward',
            },
          ],
        }),
        rightDna: createDnaEnvelope({
          moduleArchetypes: [
            {
              archetypeId: 'archetype:dense:zero-weight:1',
              computationType: 'DenseFeedForward',
            },
            {
              archetypeId: 'archetype:dense:zero-weight:2',
              computationType: 'DenseFeedForward',
            },
          ],
        }),
      }),
      {
        weights: {
          topology: 0,
          computation: 0,
          memory: 0,
          lifecycle: 0,
        },
      },
    );

    expect(roundCompatibilityDistanceResult(result)).toEqual({
      distance: 0,
      ngeEnabled: true,
      terms: {
        topology: {
          name: 'topology',
          rawDistance: 3,
          normalizedDistance: 1,
          weight: 0,
          weightedDistance: 0,
        },
        computation: {
          name: 'computation',
          rawDistance: 0.5,
          normalizedDistance: 1,
          weight: 0,
          weightedDistance: 0,
        },
        memory: {
          name: 'memory',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0,
          weightedDistance: 0,
        },
        lifecycle: {
          name: 'lifecycle',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0,
          weightedDistance: 0,
        },
      },
    });
  });

  it('keeps a zero-distance enabled comparison at zero when the slice is degenerate', () => {
    const result = computeNgeEvolutionCompatibilityDistance(
      createCompatibilityComparison(),
    );

    expect(roundCompatibilityDistanceResult(result)).toEqual({
      distance: 0,
      ngeEnabled: true,
      terms: {
        topology: {
          name: 'topology',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.4,
          weightedDistance: 0,
        },
        computation: {
          name: 'computation',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.2,
          weightedDistance: 0,
        },
        memory: {
          name: 'memory',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.2,
          weightedDistance: 0,
        },
        lifecycle: {
          name: 'lifecycle',
          rawDistance: 0,
          normalizedDistance: 0,
          weight: 0.2,
          weightedDistance: 0,
        },
      },
    });
  });
});

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

  return {
    ...baseArchetype,
    ...overrides,
  };
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

  return {
    ...baseRulePass,
    ...overrides,
  };
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

  return {
    ...baseCppnProgram,
    ...overrides,
  };
}

function createReproductionPolicy(
  overrides?: Partial<NgeReproductionPolicy>,
): NgeReproductionPolicy {
  return {
    ...reproductionPolicyFixture,
    ...overrides,
  };
}

function createCompatibilityComparison(overrides?: {
  topologyDistance?: number;
  ngeEnabled?: boolean;
  leftDna?: NgeDnaCanonicalEnvelope | null;
  rightDna?: NgeDnaCanonicalEnvelope | null;
  leftLifecycleTraits?: {
    assimilationCadence?: number;
    wiringCostWeights?: {
      nodeWeight?: number;
      edgeWeight?: number;
      interZonePenalty?: number;
    };
  };
  rightLifecycleTraits?: {
    assimilationCadence?: number;
    wiringCostWeights?: {
      nodeWeight?: number;
      edgeWeight?: number;
      interZonePenalty?: number;
    };
  };
}) {
  return {
    topologyDistance: overrides?.topologyDistance ?? 0,
    ngeEnabled: overrides?.ngeEnabled ?? true,
    leftGenome: {
      dna: overrides?.leftDna ?? createDnaEnvelope(),
      ...overrides?.leftLifecycleTraits,
    },
    rightGenome: {
      dna: overrides?.rightDna ?? createDnaEnvelope(),
      ...overrides?.rightLifecycleTraits,
    },
  };
}

function roundCompatibilityDistanceResult(
  result: NgeEvolutionCompatibilityDistanceResult,
) {
  return {
    distance: roundScalar(result.distance),
    ngeEnabled: result.ngeEnabled,
    terms: {
      topology: roundCompatibilityDistanceTerm(result.terms.topology),
      computation: roundCompatibilityDistanceTerm(result.terms.computation),
      memory: roundCompatibilityDistanceTerm(result.terms.memory),
      lifecycle: roundCompatibilityDistanceTerm(result.terms.lifecycle),
    },
  };
}

function roundCompatibilityDistanceTerm(
  term: NgeEvolutionCompatibilityDistanceResult['terms']['topology'],
) {
  return {
    name: term.name,
    rawDistance: roundScalar(term.rawDistance),
    normalizedDistance: roundScalar(term.normalizedDistance),
    weight: roundScalar(term.weight),
    weightedDistance: roundScalar(term.weightedDistance),
  };
}

function roundScalar(value: number): number {
  return Number(value.toFixed(3));
}
