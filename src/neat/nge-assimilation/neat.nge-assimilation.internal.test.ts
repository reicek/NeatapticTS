/**
 * Red-phase test contracts for Phase 4 Step 01 — internal assimilation prior
 * write-back.
 *
 * Covers AC-405: assimilation writes weak/decaying structural priors derived
 * from the main agent's own equilibrium candidate only; no enemy weights or
 * structure flow into the main agent.
 *
 * All tests fail because the imported source modules do not exist yet. The
 * expected failure reason is TS2307 "Cannot find module".
 *
 * Single-expect rule enforced. AAA structure in every test.
 */

import type { EquilibriumCandidate } from '../nge-adult/neat.nge-adult.types';
import type {
  NgeMainAgentEquilibriumFitnessMetrics,
  NgeMainAgentStableCandidate,
} from '../nge-main-agent/neat.nge-main-agent.adult';

import { AssimilationSchemaError } from './neat.nge-assimilation.errors';
import type { NgeAssimilationCandidate } from './neat.nge-assimilation.types';
import {
  writeInternalAssimilationPriors,
  type InternalAssimilationInput,
} from './neat.nge-assimilation.internal';

function createEquilibriumCandidate(
  overrides?: Partial<EquilibriumCandidate>,
): EquilibriumCandidate {
  return {
    zoneId: 'zone:main-1',
    isGainStable: true,
    isPlateau: true,
    ...overrides,
  };
}

function createAssimilationCandidate(
  overrides?: Partial<NgeAssimilationCandidate>,
): NgeAssimilationCandidate {
  return {
    equilibriumCandidate: createEquilibriumCandidate(),
    sourceDnaFingerprint: 'fp:main-agent-equilibrium',
    sourceSchemaVersion: 'A.1.0',
    moduleDelta: {
      moduleId: 'module:main-1',
      zoneId: 'zone:main-1',
      archetypeId: 'archetype:sensor',
      ruleParameters: {
        connectionDensity: {
          currentValue: 0.2,
          targetValue: 0.8,
        },
      },
    },
    ...overrides,
  };
}

function createStableCandidate(
  overrides?: Partial<NgeMainAgentStableCandidate>,
): NgeMainAgentStableCandidate {
  const defaultFitnessMetrics: NgeMainAgentEquilibriumFitnessMetrics = {
    survivalTicks: 100,
    damageDealt: 50,
    kills: 2,
    damageTaken: 10,
    aimMissRate: 0.2,
    complexityBonus: 0.1,
    parsimonyDensityPenalty: 0.05,
  };

  return {
    genomeState: {
      nodeCount: 16,
      edgeCount: 32,
      archetypes: [
        {
          archetypeId: 'archetype:sensor',
          computationType: 'DenseFeedForward',
        },
      ],
      schemaVersion: 'A.1.0',
    },
    fitnessMetrics: defaultFitnessMetrics,
    structuralInfo: {
      withinBudget: true,
      maxNodes: 64,
      maxEdges: 128,
    },
    ...overrides,
  };
}

describe('writeInternalAssimilationPriors', () => {
  describe('single-source constraint', () => {
    it('writes priors derived from the main agent equilibrium candidate', () => {
      // Arrange
      const equilibrium = createEquilibriumCandidate();
      const candidate = createAssimilationCandidate({
        equilibriumCandidate: equilibrium,
      });
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 0.9,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.sourceDnaFingerprint).toBe('fp:main-agent-equilibrium');
    });

    it('does not accept enemy-derived weights in the input', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 0.9,
        enemyWeightTensor: new Float32Array([0.1, 0.2, 0.3]),
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert — enemy weights must be rejected or ignored
      expect(result.enemyWeightsIncorporated).toBe(false);
    });

    it('does not accept enemy-derived structure in the input', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 0.9,
        enemyStructureChecksum: 'enemy:structure:123',
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert — enemy structure must be rejected or ignored
      expect(result.enemyStructureIncorporated).toBe(false);
    });
  });

  describe('weak/decaying prior update', () => {
    it('applies the configured write-back rate to move current toward target', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(
        result.updatedModuleDelta.ruleParameters?.connectionDensity
          .currentValue,
      ).toBe(0.26);
    });

    it('decays the applied delta by the configured decay factor', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 0.5,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.appliedDecay).toBe(0.5);
    });

    it('keeps the target value unchanged after write-back', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(
        result.updatedModuleDelta.ruleParameters?.connectionDensity.targetValue,
      ).toBe(0.8);
    });
  });

  describe('result contract', () => {
    it('returns a non-null updated module delta', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.updatedModuleDelta).not.toBeNull();
    });

    it('preserves the module identifier through write-back', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.updatedModuleDelta.moduleId).toBe('module:main-1');
    });
  });

  describe('cppnParameterBlock update', () => {
    it('updates every Float32Array parameter toward its target with weak decay', () => {
      // Arrange
      const currentValue = new Float32Array([0.1, 0.5]);
      const targetValue = new Float32Array([0.3, 0.7]);
      const candidate = createAssimilationCandidate({
        moduleDelta: {
          moduleId: 'module:main-1',
          zoneId: 'zone:main-1',
          cppnParameterBlock: {
            currentValue,
            targetValue,
          },
        },
      });
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };
      const expected = Float32Array.from(targetValue, (target, index) =>
        Number(
          (currentValue[index] + 0.1 * (target - currentValue[index])).toFixed(
            12,
          ),
        ),
      );

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(
        result.updatedModuleDelta.cppnParameterBlock?.currentValue,
      ).toEqual(expected);
    });

    it('copies the target Float32Array so the result does not alias the input', () => {
      // Arrange
      const targetValue = new Float32Array([0.3, 0.7]);
      const candidate = createAssimilationCandidate({
        moduleDelta: {
          moduleId: 'module:main-1',
          zoneId: 'zone:main-1',
          cppnParameterBlock: {
            currentValue: new Float32Array([0.1, 0.5]),
            targetValue,
          },
        },
      });
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(
        result.updatedModuleDelta.cppnParameterBlock?.targetValue,
      ).not.toBe(targetValue);
    });
  });

  describe('numeric delta edge cases', () => {
    it('preserves undefined numeric deltas inside an otherwise defined record', () => {
      // Arrange
      const candidate = createAssimilationCandidate({
        moduleDelta: {
          moduleId: 'module:main-1',
          zoneId: 'zone:main-1',
          ruleParameters: {
            connectionDensity: {
              currentValue: 0.2,
              targetValue: 0.8,
            },
            reservedGain: undefined,
          },
        } as unknown as NgeAssimilationCandidate['moduleDelta'],
      });
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(
        result.updatedModuleDelta.ruleParameters?.reservedGain,
      ).toBeUndefined();
    });
  });

  describe('candidate envelope validation', () => {
    it('throws when the source DNA fingerprint is empty', () => {
      // Arrange
      const candidate = createAssimilationCandidate({
        sourceDnaFingerprint: '',
      });
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act + Assert
      expect(() => writeInternalAssimilationPriors(input)).toThrow(
        AssimilationSchemaError,
      );
    });

    it('throws when the equilibrium zone identifier is empty', () => {
      // Arrange
      const candidate = createAssimilationCandidate({
        equilibriumCandidate: createEquilibriumCandidate({ zoneId: '' }),
      });
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act + Assert
      expect(() => writeInternalAssimilationPriors(input)).toThrow(
        AssimilationSchemaError,
      );
    });

    it('throws when the module delta identifiers are empty', () => {
      // Arrange
      const candidate = createAssimilationCandidate({
        moduleDelta: {
          moduleId: '',
          zoneId: '',
        } as NgeAssimilationCandidate['moduleDelta'],
      });
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act + Assert
      expect(() => writeInternalAssimilationPriors(input)).toThrow(
        AssimilationSchemaError,
      );
    });

    it('throws when equilibrium and module delta zones mismatch', () => {
      // Arrange
      const candidate = createAssimilationCandidate({
        equilibriumCandidate: createEquilibriumCandidate({
          zoneId: 'zone:other',
        }),
      });
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act + Assert
      expect(() => writeInternalAssimilationPriors(input)).toThrow(
        AssimilationSchemaError,
      );
    });
  });

  describe('stable candidate validation', () => {
    it('marks stableCandidateValidated false when no stable candidate is supplied', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.stableCandidateValidated).toBe(false);
    });

    it('marks stableCandidateValidated true when the stable candidate matches', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        stableCandidate: createStableCandidate(),
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.stableCandidateValidated).toBe(true);
    });

    it('marks stableCandidateValidated false when schema versions mismatch', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        stableCandidate: createStableCandidate({
          genomeState: {
            schemaVersion: 'B.2.0',
          } as unknown as NgeMainAgentStableCandidate['genomeState'],
        }),
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.stableCandidateValidated).toBe(false);
    });

    it('marks stableCandidateValidated false when the module archetype is missing', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const baseGenomeState = createStableCandidate().genomeState;
      const input: InternalAssimilationInput = {
        candidate,
        stableCandidate: createStableCandidate({
          genomeState: {
            ...baseGenomeState,
            archetypes: [
              {
                archetypeId: 'archetype:other',
                computationType: 'DenseFeedForward',
              },
            ],
          },
        }),
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.stableCandidateValidated).toBe(false);
    });

    it('marks stableCandidateValidated false when genome counts exceed the budget', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const baseGenomeState = createStableCandidate().genomeState;
      const input: InternalAssimilationInput = {
        candidate,
        stableCandidate: createStableCandidate({
          genomeState: {
            ...baseGenomeState,
            nodeCount: 200,
            edgeCount: 200,
          },
        }),
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.stableCandidateValidated).toBe(false);
    });

    it('marks stableCandidateValidated false when fitness metrics are out of bounds', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        stableCandidate: createStableCandidate({
          fitnessMetrics: {
            aimMissRate: 1.5,
          } as unknown as NgeMainAgentStableCandidate['fitnessMetrics'],
        }),
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.stableCandidateValidated).toBe(false);
    });

    it('marks stableCandidateValidated false when the budget flag is false', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        stableCandidate: createStableCandidate({
          structuralInfo: {
            withinBudget: false,
          } as unknown as NgeMainAgentStableCandidate['structuralInfo'],
        }),
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.stableCandidateValidated).toBe(false);
    });

    it('marks stableCandidateValidated false when archetype list is empty', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const baseGenomeState = createStableCandidate().genomeState;
      const input: InternalAssimilationInput = {
        candidate,
        stableCandidate: createStableCandidate({
          genomeState: { ...baseGenomeState, archetypes: [] },
        }),
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.stableCandidateValidated).toBe(false);
    });

    it('marks stableCandidateValidated false when genome counts are negative', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const baseGenomeState = createStableCandidate().genomeState;
      const input: InternalAssimilationInput = {
        candidate,
        stableCandidate: createStableCandidate({
          genomeState: { ...baseGenomeState, nodeCount: -1, edgeCount: -1 },
        }),
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.stableCandidateValidated).toBe(false);
    });

    it('marks stableCandidateValidated false when structural info is malformed', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        stableCandidate: createStableCandidate({
          structuralInfo: {
            maxNodes: Number.NaN,
          } as unknown as NgeMainAgentStableCandidate['structuralInfo'],
        }),
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.stableCandidateValidated).toBe(false);
    });

    it('marks stableCandidateValidated false when fitness metrics are missing', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        stableCandidate: createStableCandidate({
          fitnessMetrics:
            undefined as unknown as NgeMainAgentStableCandidate['fitnessMetrics'],
        }),
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.stableCandidateValidated).toBe(false);
    });

    it('marks stableCandidateValidated false when a fitness metric is non-finite', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        stableCandidate: createStableCandidate({
          fitnessMetrics: {
            damageDealt: Number.NaN,
          } as unknown as NgeMainAgentStableCandidate['fitnessMetrics'],
        }),
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.stableCandidateValidated).toBe(false);
    });
  });

  describe('hyperparameter sanitization', () => {
    it('clamps a negative writeBackRate to zero', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: -0.5,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(
        result.updatedModuleDelta.ruleParameters?.connectionDensity
          .currentValue,
      ).toBe(0.2);
    });

    it('clamps a writeBackRate above one to one', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 2,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(
        result.updatedModuleDelta.ruleParameters?.connectionDensity
          .currentValue,
      ).toBe(0.8);
    });

    it('treats NaN writeBackRate as zero', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: Number.NaN,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(
        result.updatedModuleDelta.ruleParameters?.connectionDensity
          .currentValue,
      ).toBe(0.2);
    });

    it('treats positive Infinity writeBackRate as one', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: Number.POSITIVE_INFINITY,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(
        result.updatedModuleDelta.ruleParameters?.connectionDensity
          .currentValue,
      ).toBe(0.8);
    });

    it('treats negative Infinity decay as zero', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: Number.NEGATIVE_INFINITY,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.appliedDecay).toBe(0);
    });
  });

  describe('non-finite numeric delta guards', () => {
    it('skips updating a numeric delta with a non-finite current value', () => {
      // Arrange
      const candidate = createAssimilationCandidate({
        moduleDelta: {
          moduleId: 'module:main-1',
          zoneId: 'zone:main-1',
          ruleParameters: {
            connectionDensity: {
              currentValue: Number.NaN,
              targetValue: 0.8,
            },
          },
        },
      });
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(
        result.updatedModuleDelta.ruleParameters?.connectionDensity
          .currentValue,
      ).toBe(Number.NaN);
    });

    it('skips updating a numeric delta with a non-finite target value', () => {
      // Arrange
      const candidate = createAssimilationCandidate({
        moduleDelta: {
          moduleId: 'module:main-1',
          zoneId: 'zone:main-1',
          ruleParameters: {
            connectionDensity: {
              currentValue: 0.2,
              targetValue: Number.POSITIVE_INFINITY,
            },
          },
        },
      });
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(
        result.updatedModuleDelta.ruleParameters?.connectionDensity
          .currentValue,
      ).toBe(0.2);
    });
  });

  describe('cppn parameter block non-finite guards', () => {
    it('skips non-finite current values in the cppn parameter block', () => {
      // Arrange
      const candidate = createAssimilationCandidate({
        moduleDelta: {
          moduleId: 'module:main-1',
          zoneId: 'zone:main-1',
          cppnParameterBlock: {
            currentValue: new Float32Array([Number.NaN, 0.5]),
            targetValue: new Float32Array([0.3, 0.7]),
          },
        },
      });
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(
        result.updatedModuleDelta.cppnParameterBlock?.currentValue[0],
      ).toBe(Number.NaN);
    });

    it('skips non-finite target values in the cppn parameter block', () => {
      // Arrange
      const candidate = createAssimilationCandidate({
        moduleDelta: {
          moduleId: 'module:main-1',
          zoneId: 'zone:main-1',
          cppnParameterBlock: {
            currentValue: new Float32Array([0.1, 0.5]),
            targetValue: new Float32Array([Number.POSITIVE_INFINITY, 0.7]),
          },
        },
      });
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors;
      const updated =
        result(input).updatedModuleDelta.cppnParameterBlock?.currentValue;

      // Assert
      expect(updated?.[0]).toBeCloseTo(0.1, 7);
    });
  });

  describe('enemy topology rejection', () => {
    it('does not incorporate an enemy topology hash', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
        enemyTopologyHash: 'enemy:topology:abc',
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.enemyTopologyIncorporated).toBe(false);
    });

    it('produces the same output whether enemy inputs are present or not', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const baseInput: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };
      const enemyInput: InternalAssimilationInput = {
        ...baseInput,
        enemyWeightTensor: new Float32Array([0.1, 0.2, 0.3]),
        enemyStructureChecksum: 'enemy:structure:123',
        enemyTopologyHash: 'enemy:topology:abc',
      };

      // Act
      const baseResult = writeInternalAssimilationPriors(baseInput);
      const enemyResult = writeInternalAssimilationPriors(enemyInput);

      // Assert
      expect(baseResult.updatedModuleDelta).toEqual(
        enemyResult.updatedModuleDelta,
      );
    });
  });

  describe('determinism', () => {
    it('returns identical results for identical inputs', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        stableCandidate: createStableCandidate(),
        writeBackRate: 0.1,
        decay: 0.9,
      };

      // Act + Assert
      expect(writeInternalAssimilationPriors(input)).toEqual(
        writeInternalAssimilationPriors(input),
      );
    });
  });
});
