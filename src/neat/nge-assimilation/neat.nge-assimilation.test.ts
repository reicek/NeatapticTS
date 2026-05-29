import {
  ASSIMILATION_ENCODING_MODES,
  DEFAULT_ASSIMILATION_WRITE_BACK_RATE,
  DEFAULT_BUDGET_GUARD_ENABLED,
} from './neat.nge-assimilation.constants';
import {
  AssimilationBudgetError,
  AssimilationSchemaError,
} from './neat.nge-assimilation.errors';
import { assimilateEquilibriumCandidate } from './neat.nge-assimilation';
import {
  buildAssimilationResult,
  buildAssimilationTelemetry,
  validateAssimilationCandidate,
} from './neat.nge-assimilation.utils';
import { applyAssimilationWriteback } from './neat.nge-assimilation.writeback';
import type {
  NgeAssimilationCandidate,
  NgeAssimilationModuleDelta,
  NgeAssimilationPolicy,
} from './neat.nge-assimilation.types';

describe('nge assimilation chapter', () => {
  describe('assimilateEquilibriumCandidate', () => {
    describe('given one valid equilibrium candidate envelope', () => {
      it('returns the normalized accepted result through the orchestration facade', () => {
        // Arrange
        const candidate = createCandidate({
          ruleParameters: {
            replicationDepth: {
              currentValue: 10,
              targetValue: 30,
            },
          },
        });
        const policy = createPolicy({ writeBackRate: 0.25 });

        // Act
        const assimilationResult = assimilateEquilibriumCandidate(
          candidate,
          policy,
        );

        // Assert
        expect(assimilationResult).toEqual({
          moduleId: 'module:alpha',
          status: 'accepted',
          telemetry: {
            budgetGuardEnabled: true,
            lossy: false,
          },
          updatedModuleDelta: {
            moduleId: 'module:alpha',
            zoneId: 'zone:alpha',
            ruleParameters: {
              replicationDepth: {
                currentValue: 15,
                targetValue: 30,
              },
            },
          },
        });
      });
    });

    describe('given an invalid equilibrium candidate envelope', () => {
      it('returns schema-invalid before write-back runs', () => {
        // Arrange
        const candidate = createCandidate({
          zoneId: 'zone:module',
          ruleParameters: {
            replicationDepth: {
              currentValue: 1,
              targetValue: 2,
            },
          },
        });
        candidate.equilibriumCandidate.zoneId = 'zone:equilibrium';

        // Act
        const assimilationResult = assimilateEquilibriumCandidate(
          candidate,
          createPolicy(),
        );

        // Assert
        expect(assimilationResult).toEqual({
          moduleId: 'module:alpha',
          status: 'schema-invalid',
          telemetry: {
            budgetGuardEnabled: true,
            lossy: false,
          },
          updatedModuleDelta: null,
        });
      });
    });
  });

  describe('validateAssimilationCandidate', () => {
    describe('given a valid owner-local envelope', () => {
      it('returns null', () => {
        // Arrange
        const candidate = createCandidate({
          cppnParameterBlock: {
            currentValue: new Float32Array([0.1, 0.2]),
            targetValue: new Float32Array([0.3, 0.4]),
          },
        });

        // Act
        const validationError = validateAssimilationCandidate(candidate);

        // Assert
        expect(validationError).toBeNull();
      });
    });

    describe('given a candidate missing the source fingerprint', () => {
      it('returns the matching schema error message', () => {
        // Arrange
        const candidate = createCandidate({});
        candidate.sourceDnaFingerprint = '   ';

        // Act
        const validationError = validateAssimilationCandidate(candidate);

        // Assert
        expect(validationError?.message).toBe(
          'Assimilation candidate requires a source DNA fingerprint.',
        );
      });
    });

    describe('given a candidate missing the schema version', () => {
      it('returns the matching schema error message', () => {
        // Arrange
        const candidate = createCandidate({});
        candidate.sourceSchemaVersion = '  ';

        // Act
        const validationError = validateAssimilationCandidate(candidate);

        // Assert
        expect(validationError?.message).toBe(
          'Assimilation candidate requires a source schema version.',
        );
      });
    });

    describe('given a candidate whose equilibrium and module zones differ', () => {
      it('returns the zone-mismatch schema error', () => {
        // Arrange
        const candidate = createCandidate({ zoneId: 'zone:module' });
        candidate.equilibriumCandidate.zoneId = 'zone:equilibrium';

        // Act
        const validationError = validateAssimilationCandidate(candidate);

        // Assert
        expect(validationError?.message).toBe(
          'Assimilation candidate zone identifiers must match across the equilibrium and module shelves.',
        );
      });
    });

    describe('given a candidate with mismatched CPPN parameter lengths', () => {
      it('returns the parameter-length schema error', () => {
        // Arrange
        const candidate = createCandidate({
          cppnParameterBlock: {
            currentValue: new Float32Array([0.1]),
            targetValue: new Float32Array([0.2, 0.3]),
          },
        });

        // Act
        const validationError = validateAssimilationCandidate(candidate);

        // Assert
        expect(validationError?.message).toBe(
          'Assimilation candidate CPPN parameter arrays must have matching lengths.',
        );
      });
    });
  });

  describe('assimilation shared helpers', () => {
    describe('buildAssimilationTelemetry', () => {
      it('mirrors the budget flag and lossy override into one telemetry packet', () => {
        // Arrange
        const policy = createPolicy({ budgetGuardEnabled: false });

        // Act
        const telemetry = buildAssimilationTelemetry(policy, true);

        // Assert
        expect(telemetry).toEqual({
          budgetGuardEnabled: false,
          lossy: true,
        });
      });
    });

    describe('buildAssimilationResult', () => {
      it('folds the canonical result shape with a default non-lossy telemetry flag', () => {
        // Arrange
        const candidate = createCandidate({});
        const policy = createPolicy();

        // Act
        const assimilationResult = buildAssimilationResult(
          candidate,
          'schema-invalid',
          policy,
          null,
        );

        // Assert
        expect(assimilationResult).toEqual({
          moduleId: 'module:alpha',
          status: 'schema-invalid',
          telemetry: {
            budgetGuardEnabled: true,
            lossy: false,
          },
          updatedModuleDelta: null,
        });
      });
    });
  });

  describe('assimilation scaffold exports', () => {
    describe('constants', () => {
      it('expose the documented default write-back and encoding policy', () => {
        // Assert
        expect({
          budgetGuardEnabled: DEFAULT_BUDGET_GUARD_ENABLED,
          encodingModes: ASSIMILATION_ENCODING_MODES,
          writeBackRate: DEFAULT_ASSIMILATION_WRITE_BACK_RATE,
        }).toEqual({
          budgetGuardEnabled: true,
          encodingModes: ['lossless', 'lossy'],
          writeBackRate: 0.1,
        });
      });
    });

    describe('errors', () => {
      it('name the budget error for diagnostic routing', () => {
        // Act
        const errorName = new AssimilationBudgetError('budget exceeded').name;

        // Assert
        expect(errorName).toBe('AssimilationBudgetError');
      });

      it('name the schema error for diagnostic routing', () => {
        // Act
        const errorName = new AssimilationSchemaError('schema invalid').name;

        // Assert
        expect(errorName).toBe('AssimilationSchemaError');
      });
    });
  });

  describe('applyAssimilationWriteback', () => {
    describe('given a candidate with structural-prior deltas across the local DNA shelf', () => {
      it('applies the configured write-back fraction to the current DNA-facing values', () => {
        // Arrange
        const candidate = createCandidate({
          ruleParameters: {
            replicationDepth: {
              currentValue: 10,
              targetValue: 30,
            },
          },
          cppnTopology: {
            edgeCount: {
              currentValue: 8,
              targetValue: 12,
            },
            enableThreshold: {
              currentValue: 0.2,
              targetValue: 0.6,
            },
          },
          wiringCostWeights: {
            edgeWeight: {
              currentValue: 0.4,
              targetValue: 1,
            },
          },
          lifecycleKnobs: {
            adultWindow: {
              currentValue: 4,
              targetValue: 8,
            },
          },
        });
        const policy = createPolicy({ writeBackRate: 0.25 });

        // Act
        const assimilationResult = applyAssimilationWriteback(
          candidate,
          policy,
        );

        // Assert
        expect(assimilationResult).toEqual({
          moduleId: 'module:alpha',
          status: 'accepted',
          telemetry: {
            budgetGuardEnabled: true,
            lossy: false,
          },
          updatedModuleDelta: {
            moduleId: 'module:alpha',
            zoneId: 'zone:alpha',
            ruleParameters: {
              replicationDepth: {
                currentValue: 15,
                targetValue: 30,
              },
            },
            cppnTopology: {
              edgeCount: {
                currentValue: 9,
                targetValue: 12,
              },
              enableThreshold: {
                currentValue: 0.3,
                targetValue: 0.6,
              },
            },
            wiringCostWeights: {
              edgeWeight: {
                currentValue: 0.55,
                targetValue: 1,
              },
            },
            lifecycleKnobs: {
              adultWindow: {
                currentValue: 5,
                targetValue: 8,
              },
            },
          },
        });
      });
    });

    describe('given a candidate with memory-tier capacity shifts', () => {
      it('updates the DNA memory-tier values while clamping hiddenDim to the node-cap bound', () => {
        // Arrange
        const candidate = createCandidate({
          memoryTier: {
            hiddenDim: {
              currentValue: 6,
              targetValue: 18,
            },
            slotCount: {
              currentValue: 2,
              targetValue: 6,
            },
            decayRate: {
              currentValue: 0.4,
              targetValue: 1.4,
            },
          },
        });
        const policy = createPolicy({
          writeBackRate: 0.5,
          maxNodes: 10,
        });

        // Act
        const assimilationResult = applyAssimilationWriteback(
          candidate,
          policy,
        );

        // Assert
        expect(assimilationResult.updatedModuleDelta).toEqual({
          moduleId: 'module:alpha',
          zoneId: 'zone:alpha',
          memoryTier: {
            hiddenDim: {
              currentValue: 10,
              targetValue: 18,
            },
            slotCount: {
              currentValue: 4,
              targetValue: 6,
            },
            decayRate: {
              currentValue: 0.9,
              targetValue: 1.4,
            },
          },
        });
      });
    });

    describe('given a candidate with stabilized neuromodulator-zone deltas', () => {
      it('writes the updated gain range and broadcast radius back into the zone shelf', () => {
        // Arrange
        const candidate = createCandidate({
          neuromodulatorZone: {
            gainMin: {
              currentValue: 0.2,
              targetValue: 0.6,
            },
            gainMax: {
              currentValue: 1,
              targetValue: 1.8,
            },
            broadcastRadius: {
              currentValue: 1,
              targetValue: 3,
            },
          },
        });
        const policy = createPolicy({ writeBackRate: 0.5 });

        // Act
        const assimilationResult = applyAssimilationWriteback(
          candidate,
          policy,
        );

        // Assert
        expect(assimilationResult.updatedModuleDelta).toEqual({
          moduleId: 'module:alpha',
          zoneId: 'zone:alpha',
          neuromodulatorZone: {
            gainMin: {
              currentValue: 0.4,
              targetValue: 0.6,
            },
            gainMax: {
              currentValue: 1.4,
              targetValue: 1.8,
            },
            broadcastRadius: {
              currentValue: 2,
              targetValue: 3,
            },
          },
        });
      });
    });

    describe('given an accepted structural-prior write-back result', () => {
      it('does not introduce raw weight fields onto the candidate or result boundary', () => {
        // Arrange
        const candidate = createCandidate({
          ruleParameters: {
            hierarchyLevels: {
              currentValue: 2,
              targetValue: 6,
            },
          },
        });

        // Act
        const assimilationResult = applyAssimilationWriteback(
          candidate,
          createPolicy(),
        );

        // Assert
        expect({
          candidateHasRawWeights:
            'rawWeights' in candidate.moduleDelta ||
            'weights' in candidate.moduleDelta,
          resultHasRawWeights:
            assimilationResult.updatedModuleDelta !== null &&
            ('rawWeights' in assimilationResult.updatedModuleDelta ||
              'weights' in assimilationResult.updatedModuleDelta),
        }).toEqual({
          candidateHasRawWeights: false,
          resultHasRawWeights: false,
        });
      });
    });

    describe('given the same candidate and policy twice', () => {
      it('returns the same deterministic write-back result for both calls', () => {
        // Arrange
        const candidate = createCandidate({
          ruleParameters: {
            symmetryUsage: {
              currentValue: 0.1,
              targetValue: 0.7,
            },
          },
        });
        const policy = createPolicy({ writeBackRate: 0.5 });

        // Act
        const firstResult = applyAssimilationWriteback(candidate, policy);
        const secondResult = applyAssimilationWriteback(candidate, policy);

        // Assert
        expect(firstResult).toEqual(secondResult);
      });
    });

    describe('given a candidate whose projected budget impact exceeds the configured cap', () => {
      it('returns budget-exceeded without mutating the candidate module delta', () => {
        // Arrange
        const candidate = createCandidate({
          ruleParameters: {
            symmetryUsage: {
              currentValue: 0.2,
              targetValue: 0.8,
            },
          },
          estimatedBudgetImpact: {
            nodeCountDelta: 20,
            edgeCountDelta: 4,
            byteDelta: 64,
          },
        });
        const originalModuleDelta = structuredClone(candidate.moduleDelta);

        // Act
        const assimilationResult = applyAssimilationWriteback(
          candidate,
          createPolicy({ maxNodes: 16 }),
        );

        // Assert
        expect({
          status: assimilationResult.status,
          telemetry: assimilationResult.telemetry,
          updatedModuleDelta: assimilationResult.updatedModuleDelta,
          candidateModuleDelta: candidate.moduleDelta,
        }).toEqual({
          status: 'budget-exceeded',
          telemetry: {
            budgetGuardEnabled: true,
            lossy: false,
          },
          updatedModuleDelta: null,
          candidateModuleDelta: originalModuleDelta,
        });
      });
    });

    describe('given a candidate whose edge-budget impact exceeds the configured cap', () => {
      it('returns budget-exceeded before write-back applies any edge-facing update', () => {
        // Arrange
        const candidate = createCandidate({
          cppnTopology: {
            edgeCount: {
              currentValue: 12,
              targetValue: 18,
            },
          },
          estimatedBudgetImpact: {
            nodeCountDelta: 4,
            edgeCountDelta: 40,
            byteDelta: 64,
          },
        });

        // Act
        const assimilationResult = applyAssimilationWriteback(
          candidate,
          createPolicy(),
        );

        // Assert
        expect({
          status: assimilationResult.status,
          updatedModuleDelta: assimilationResult.updatedModuleDelta,
        }).toEqual({
          status: 'budget-exceeded',
          updatedModuleDelta: null,
        });
      });
    });

    describe('given a candidate whose byte-budget impact exceeds the configured cap', () => {
      it('returns budget-exceeded before write-back applies any byte-facing update', () => {
        // Arrange
        const candidate = createCandidate({
          cppnTopology: {
            connectionDensity: {
              currentValue: 0.2,
              targetValue: 0.7,
            },
          },
          estimatedBudgetImpact: {
            nodeCountDelta: 4,
            edgeCountDelta: 8,
            byteDelta: 512,
          },
        });

        // Act
        const assimilationResult = applyAssimilationWriteback(
          candidate,
          createPolicy(),
        );

        // Assert
        expect({
          status: assimilationResult.status,
          updatedModuleDelta: assimilationResult.updatedModuleDelta,
        }).toEqual({
          status: 'budget-exceeded',
          updatedModuleDelta: null,
        });
      });
    });

    describe('given lossy encoding mode with a CPPN parameter block', () => {
      it('quantizes the updated block to int8 output and marks telemetry as lossy', () => {
        // Arrange
        const candidate = createCandidate({
          cppnParameterBlock: {
            currentValue: new Float32Array([0, 0.5, -0.5, 0.25]),
            targetValue: new Float32Array([0.5, 1, -1, -0.25]),
          },
        });

        // Act
        const assimilationResult = applyAssimilationWriteback(
          candidate,
          createPolicy({
            encodingMode: 'lossy',
            writeBackRate: 0.5,
          }),
        );

        // Assert
        expect(summarizeCppnParameterBlockResult(assimilationResult)).toEqual({
          status: 'accepted',
          telemetry: {
            budgetGuardEnabled: true,
            lossy: true,
          },
          currentValueKind: 'Int8Array',
          currentValues: [32, 95, -95, 0],
          targetValues: [0.5, 1, -1, -0.25],
        });
      });
    });

    describe('given lossless encoding mode with a CPPN parameter block', () => {
      it('keeps the updated block in float32 form', () => {
        // Arrange
        const candidate = createCandidate({
          cppnParameterBlock: {
            currentValue: new Float32Array([0, 0.5, -0.5, 0.25]),
            targetValue: new Float32Array([0.5, 1, -1, -0.25]),
          },
        });

        // Act
        const assimilationResult = applyAssimilationWriteback(
          candidate,
          createPolicy({
            encodingMode: 'lossless',
            writeBackRate: 0.5,
          }),
        );

        // Assert
        expect(summarizeCppnParameterBlockResult(assimilationResult)).toEqual({
          status: 'accepted',
          telemetry: {
            budgetGuardEnabled: true,
            lossy: false,
          },
          currentValueKind: 'Float32Array',
          currentValues: [0.25, 0.75, -0.75, 0],
          targetValues: [0.5, 1, -1, -0.25],
        });
      });
    });

    describe('given budget guard disabled for an over-budget candidate', () => {
      it('accepts the update even when the projected impact exceeds the cap', () => {
        // Arrange
        const candidate = createCandidate({
          cppnTopology: {
            enableThreshold: {
              currentValue: 0.2,
              targetValue: 0.6,
            },
          },
          estimatedBudgetImpact: {
            nodeCountDelta: 20,
            edgeCountDelta: 4,
            byteDelta: 64,
          },
        });

        // Act
        const assimilationResult = applyAssimilationWriteback(
          candidate,
          createPolicy({
            budgetGuardEnabled: false,
            maxNodes: 16,
            writeBackRate: 0.5,
          }),
        );

        // Assert
        expect({
          status: assimilationResult.status,
          telemetry: assimilationResult.telemetry,
          updatedEnableThreshold:
            assimilationResult.updatedModuleDelta?.cppnTopology
              ?.enableThreshold,
        }).toEqual({
          status: 'accepted',
          telemetry: {
            budgetGuardEnabled: false,
            lossy: false,
          },
          updatedEnableThreshold: {
            currentValue: 0.4,
            targetValue: 0.6,
          },
        });
      });
    });
  });
});

function summarizeCppnParameterBlockResult(
  assimilationResult: ReturnType<typeof applyAssimilationWriteback>,
) {
  const parameterBlock =
    assimilationResult.updatedModuleDelta?.cppnParameterBlock;

  return {
    status: assimilationResult.status,
    telemetry: assimilationResult.telemetry,
    currentValueKind: parameterBlock?.currentValue.constructor.name ?? null,
    currentValues:
      parameterBlock === undefined
        ? null
        : Array.from(parameterBlock.currentValue),
    targetValues:
      parameterBlock === undefined
        ? null
        : Array.from(parameterBlock.targetValue),
  };
}

function createCandidate(
  moduleDelta: AssimilationModuleDeltaInput,
): NgeAssimilationCandidate {
  const zoneId = moduleDelta.zoneId ?? 'zone:alpha';

  return {
    equilibriumCandidate: {
      isGainStable: true,
      isPlateau: true,
      zoneId,
    },
    moduleDelta: {
      ...moduleDelta,
      moduleId: moduleDelta.moduleId ?? 'module:alpha',
      zoneId,
    },
    sourceDnaFingerprint: 'fingerprint:alpha',
    sourceSchemaVersion: 'A.1.0',
  };
}

function createPolicy(
  overrides: Partial<NgeAssimilationPolicy> = {},
): NgeAssimilationPolicy {
  return {
    budgetGuardEnabled: true,
    encodingMode: 'lossless',
    maxBytes: 256,
    maxEdges: 32,
    maxNodes: 16,
    writeBackRate: 0.1,
    ...overrides,
  };
}

type AssimilationModuleDeltaInput = Omit<
  NgeAssimilationModuleDelta,
  'moduleId' | 'zoneId'
> &
  Partial<Pick<NgeAssimilationModuleDelta, 'moduleId' | 'zoneId'>>;
