/**
 * Test contracts for the NGE score-gated commit/rollback loop.
 *
 * These tests define the expected contract for `adapt()` in
 * `src/neat/nge-juvenile/neat.nge-juvenile.adapt.ts`. `adapt()` snapshots the
 * network + global connection innovation counter, evaluates a baseline score,
 * applies the candidate mutation via the evaluator, evaluates the candidate
 * score, and either commits the mutation or rolls back to the snapshot based on
 * `config.overrides.improvementThreshold`.
 *
 * First-growth behavior is config-driven via
 * `config.overrides.firstGrowthExemption`, not hardcoded.
 *
 * Single-expect rule enforced throughout; each `it()` block contains exactly
 * one top-level `expect(...)` call.
 */

import mutation from '../../methods/mutation/mutation';
import Network from '../../architecture/network';
import Connection from '../../architecture/connection/connection';
import { adapt } from './neat.nge-juvenile.adapt';
import type {
  NgeAdaptOptions,
  NgeCandidateEvaluator,
  NgeModuleMetricsSnapshot,
} from './neat.nge-juvenile.types';

const DEFAULT_IMPROVEMENT_THRESHOLD = 0.05;

/**
 * Build a deterministic evaluator whose candidate mutation always adds one
 * hidden node and returns the supplied candidate score.
 */
function buildNodeAddEvaluator(candidateScore: number): NgeCandidateEvaluator {
  return {
    baseline: () => 0.5,
    apply: (network) => {
      network.mutate(mutation.ADD_NODE);
    },
    candidate: () => candidateScore,
  };
}

/**
 * Build default red-phase options, deterministic with `{ seed: 42 }`.
 */
function buildAdaptOptions(
  candidateScore: number,
  partial: Partial<NgeAdaptOptions> = {},
): NgeAdaptOptions {
  return {
    network: new Network(4, 2, { seed: 42 }),
    scoreHistory: [0.5, 0.55, 0.52],
    evaluator: buildNodeAddEvaluator(candidateScore),
    hasGrownBefore: true,
    ...partial,
    config: {
      overrides: {
        improvementThreshold: DEFAULT_IMPROVEMENT_THRESHOLD,
        firstGrowthExemption: false,
        ...(partial.config?.overrides ?? {}),
      },
    },
  };
}

/**
 * Build adapt options without forcing config overrides.
 *
 * Unlike {@link buildAdaptOptions}, this helper does not inject a config with
 * default overrides. This lets `resolveAdaptConfig` exercise its `??` fallback
 * branches when no config or a partial config is supplied.
 */
function buildAdaptOptionsWithoutConfig(
  candidateScore: number,
  partial: Partial<NgeAdaptOptions> = {},
): NgeAdaptOptions {
  return {
    network: new Network(4, 2, { seed: 42 }),
    scoreHistory: [0.5, 0.55, 0.52],
    evaluator: buildNodeAddEvaluator(candidateScore),
    hasGrownBefore: true,
    ...partial,
  };
}

describe('nge juvenile adaptation', () => {
  describe('adapt', () => {
    let capturedInnovation: number;

    beforeEach(() => {
      capturedInnovation = Connection.nextInnovation;
    });

    afterEach(() => {
      Connection.resetInnovationCounter(capturedInnovation);
    });

    describe('commit path', () => {
      it('returns accepted true when candidate score exceeds baseline plus threshold', () => {
        // Arrange — candidate score (0.7) improves well beyond baseline (0.5) + threshold (0.05).
        const options = buildAdaptOptions(0.7);

        // Act
        const result = adapt(options);

        // Assert
        expect(result.accepted).toBe(true);
      });

      it('keeps the structural mutation when the candidate is accepted', () => {
        // Arrange — same commit scenario, captured before any mutation is applied.
        const options = buildAdaptOptions(0.7);
        const initialNodeCount = options.network.nodes.length;

        // Act
        adapt(options);

        // Assert — the ADD_NODE mutation should persist after commit.
        expect(options.network.nodes.length).toBe(initialNodeCount + 1);
      });
    });

    describe('rollback path', () => {
      it('returns accepted false when candidate score does not improve enough', () => {
        // Arrange — candidate score (0.51) is only +0.01 above baseline (0.5), below threshold (0.05).
        const options = buildAdaptOptions(0.51);

        // Act
        const result = adapt(options);

        // Assert
        expect(result.accepted).toBe(false);
      });

      it('restores the original network state when the candidate is rejected', () => {
        // Arrange — reject scenario with the original node count captured.
        const options = buildAdaptOptions(0.51);
        const initialNodeCount = options.network.nodes.length;

        // Act
        adapt(options);

        // Assert — rolled-back ADD_NODE leaves node count unchanged.
        expect(options.network.nodes.length).toBe(initialNodeCount);
      });
    });

    describe('first-growth exemption', () => {
      it('returns accepted true for a non-improving candidate when first growth exemption applies', () => {
        // Arrange — first growth, non-improving candidate, exemption enabled.
        const options = buildAdaptOptions(0.51, {
          hasGrownBefore: false,
          config: { overrides: { firstGrowthExemption: true } },
        });

        // Act
        const result = adapt(options);

        // Assert
        expect(result.accepted).toBe(true);
      });
    });

    describe('config override improvementThreshold', () => {
      it('rejects a marginal improvement that does not meet a high custom threshold', () => {
        // Arrange — candidate score (0.52) is +0.02 above baseline, below custom threshold (0.1).
        const options = buildAdaptOptions(0.52, {
          config: { overrides: { improvementThreshold: 0.1 } },
        });

        // Act
        const result = adapt(options);

        // Assert
        expect(result.accepted).toBe(false);
      });
    });

    describe('config override disabling firstGrowthExemption', () => {
      it('requires improvement even when hasGrownBefore is false if firstGrowthExemption is disabled', () => {
        // Arrange — first growth, non-improving candidate, exemption explicitly disabled.
        const options = buildAdaptOptions(0.51, {
          hasGrownBefore: false,
          config: { overrides: { firstGrowthExemption: false } },
        });

        // Act
        const result = adapt(options);

        // Assert
        expect(result.accepted).toBe(false);
      });
    });
    describe('config default fallbacks', () => {
      it('uses default improvementThreshold when config is omitted', () => {
        // Arrange — no config, so resolveAdaptConfig falls back to DEFAULT_IMPROVEMENT_THRESHOLD (0.05).
        // Candidate (0.51) is only +0.01 above baseline (0.5), below the default threshold.
        const options = buildAdaptOptionsWithoutConfig(0.51);

        // Act
        const result = adapt(options);

        // Assert
        expect(result.accepted).toBe(false);
      });

      it('uses default firstGrowthExemption when config is omitted', () => {
        // Arrange — no config, so resolveAdaptConfig falls back to DEFAULT_FIRST_GROWTH_EXEMPTION (true).
        // hasGrownBefore is false, so the exemption applies and the candidate commits regardless of score.
        const options = buildAdaptOptionsWithoutConfig(0.51, {
          hasGrownBefore: false,
        });

        // Act
        const result = adapt(options);

        // Assert
        expect(result.accepted).toBe(true);
      });

      it('uses default firstGrowthExemption when only improvementThreshold is overridden', () => {
        // Arrange — partial config sets improvementThreshold but not firstGrowthExemption.
        // The default firstGrowthExemption (true) should apply with hasGrownBefore false.
        const options = buildAdaptOptionsWithoutConfig(0.51, {
          hasGrownBefore: false,
          config: { overrides: { improvementThreshold: 0.05 } },
        });

        // Act
        const result = adapt(options);

        // Assert
        expect(result.accepted).toBe(true);
      });

      it('uses default improvementThreshold when only firstGrowthExemption is overridden', () => {
        // Arrange — partial config sets firstGrowthExemption but not improvementThreshold.
        // hasGrownBefore is true so the exemption is irrelevant; the default threshold (0.05) applies.
        // Candidate (0.51) is only +0.01 above baseline (0.5), below the default threshold.
        const options = buildAdaptOptionsWithoutConfig(0.51, {
          hasGrownBefore: true,
          config: { overrides: { firstGrowthExemption: false } },
        });

        // Act
        const result = adapt(options);

        // Assert
        expect(result.accepted).toBe(false);
      });
    });

    describe('hasGrownBefore default', () => {
      it('defaults hasGrownBefore to true when omitted, preventing first-growth exemption', () => {
        // Arrange — hasGrownBefore is omitted, so it defaults to true.
        // firstGrowthExemption is explicitly true, but hasGrownBefore=true means the exemption does not apply.
        // Candidate (0.51) is only +0.01 above baseline (0.5), below the threshold (0.05).
        const options: NgeAdaptOptions = {
          network: new Network(4, 2, { seed: 42 }),
          scoreHistory: [0.5, 0.55, 0.52],
          evaluator: buildNodeAddEvaluator(0.51),
          config: {
            overrides: {
              firstGrowthExemption: true,
              improvementThreshold: 0.05,
            },
          },
          // hasGrownBefore intentionally omitted — should default to true
        };

        // Act
        const result = adapt(options);

        // Assert
        expect(result.accepted).toBe(false);
      });
    });
  });
});
// B4-red: pluggable adapt() API
//
// These tests define the expected contract that `adapt()` accepts optional
// pluggable provider interfaces: metricsProvider, cadencePolicy,
// observationEncoder, and lifecycleRunner. They fail because `adapt()`
// currently only uses `evaluator` and pre-computed metrics/cadence/lifecycle
// fields — it never calls provider-style interfaces.
//
// The provider interface types (NgeMetricsProvider, NgeCadencePolicy,
// NgeLifecycleRunner) do not yet exist; NgeObservationEncoder already
// exists. B4-impl will add the missing types and wire them into adapt().
// Tests use `as unknown as NgeAdaptOptions` casts to compile against the
// current types while expressing the future contract.
// ──────────────────────────────────────────────────────────────────────

describe('B4-red: pluggable adapt() API', () => {
  let capturedInnovation: number;

  beforeEach(() => {
    capturedInnovation = Connection.nextInnovation;
  });

  afterEach(() => {
    Connection.resetInnovationCounter(capturedInnovation);
  });

  it('calls custom metricsProvider when supplied', () => {
    // Arrange — mock metricsProvider that records it was called
    let metricsCalled = false;
    const mockMetricsProvider = {
      getMetrics: () => {
        metricsCalled = true;
        return {
          moduleId: 'test-module',
          utilization: 0.5,
          rewardDelta: 0.1,
          novelty: 0.2,
          stabilityAge: 1,
          wiringCost: 0.3,
        } as NgeModuleMetricsSnapshot;
      },
    };

    // Act — metricsProvider is not yet on NgeAdaptOptions
    adapt(
      buildAdaptOptions(0.7, {
        metricsProvider: mockMetricsProvider,
      } as unknown as Partial<NgeAdaptOptions>),
    );

    // Assert — adapt() should call metricsProvider.getMetrics()
    expect(metricsCalled).toBe(true);
  });

  it('calls custom cadencePolicy when supplied', () => {
    // Arrange — mock cadencePolicy that records it was called
    let cadenceCalled = false;
    const mockCadencePolicy = {
      decideCadence: () => {
        cadenceCalled = true;
        return { runAdapt: true, reason: 'test' };
      },
    };

    // Act — cadencePolicy is not yet on NgeAdaptOptions
    adapt(
      buildAdaptOptions(0.7, {
        cadencePolicy: mockCadencePolicy,
      } as unknown as Partial<NgeAdaptOptions>),
    );

    // Assert — adapt() should call cadencePolicy.decideCadence()
    expect(cadenceCalled).toBe(true);
  });

  it('calls custom observationEncoder when supplied', () => {
    // Arrange — mock observationEncoder that records it was called
    let encoderCalled = false;
    const mockObservationEncoder = {
      encode: () => {
        encoderCalled = true;
        return new Float32Array([0.1, 0.2, 0.3]);
      },
    };

    // Act — observationEncoder is not yet on NgeAdaptOptions
    adapt(
      buildAdaptOptions(0.7, {
        observationEncoder: mockObservationEncoder,
      } as unknown as Partial<NgeAdaptOptions>),
    );

    // Assert — adapt() should call observationEncoder.encode()
    expect(encoderCalled).toBe(true);
  });

  it('calls custom lifecycleRunner when supplied', () => {
    // Arrange — mock lifecycleRunner that records it was called
    let lifecycleCalled = false;
    const mockLifecycleRunner = () => {
      lifecycleCalled = true;
      return {
        stage: 'juvenile',
        applyOutcomes: [{ status: 'applied', kind: 'nodeAdd' }],
        hysteresis: {
          growthPositiveWindowCount: 0,
          pruneUnderuseWindowCount: 0,
          lastMorphKind: 'none' as const,
          cooldownWindowsRemaining: 0,
        },
      };
    };

    // Act — lifecycleRunner is not yet on NgeAdaptOptions
    adapt(
      buildAdaptOptions(0.7, {
        lifecycleRunner: mockLifecycleRunner,
      } as unknown as Partial<NgeAdaptOptions>),
    );

    // Assert — adapt() should call lifecycleRunner()
    expect(lifecycleCalled).toBe(true);
  });
});
