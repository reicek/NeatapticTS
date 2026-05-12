import type { Neat } from '../../../src/neataptic';
import type Network from '../../../src/architecture/network';
import { FLAPPY_NETWORK_INPUT_SIZE } from '../constants/constants';
import {
  resolveHeuristicTeacherFlapDecisionFromObservationVector,
  resolveWarmStartEvaluationScore,
  resolveWarmStartRolloutOptimizationPlan,
  resolveWorkerWarmStartTeacherStrategy,
  warmStartWorkerGenerationZeroIfNeeded,
  type WorkerWarmStartDeadline,
  type WorkerWarmStartDependencies,
} from './flappy-evolution-worker.warm-start.service';

const RECURRENT_WARM_START_TIME_LIMIT_MS = 10_000;

describe('warmStartWorkerGenerationZeroIfNeeded', () => {
  it('seeds generation-zero genomes from the rollout-optimized template', async () => {
    const optimizedTemplateNetwork = createMockNetwork({
      nodeBiases: [7],
      connectionWeights: [11],
    });
    const optimizeWarmStartTemplateNetwork = jest
      .fn<Network, [Network, number]>()
      .mockReturnValue(optimizedTemplateNetwork);
    const dependencies: WorkerWarmStartDependencies = {
      buildHeuristicPretrainSet: () => [
        { input: [0, 0], output: [1, 0] },
        { input: [1, 1], output: [0, 1] },
      ],
      optimizeWarmStartTemplateNetwork,
    };
    const firstGenome = createMockNetwork();
    const secondGenome = createMockNetwork();
    const neatController = {
      generation: 0,
      population: [firstGenome, secondGenome],
    } as unknown as Neat;

    await warmStartWorkerGenerationZeroIfNeeded(
      neatController,
      {
        architectureProfileId: 'mlp',
        workerInitSeed: 123,
        generationZeroWarmStartApplied: false,
      },
      dependencies,
    );

    expect({
      optimizeCalls: optimizeWarmStartTemplateNetwork.mock.calls.length,
      firstGenomeBiasImproved: firstGenome.nodes[0].bias > 6.5,
      firstGenomeWeightImproved: firstGenome.connections[0].weight > 10.5,
      secondGenomeBiasImproved: secondGenome.nodes[0].bias > 6.5,
      secondGenomeWeightImproved: secondGenome.connections[0].weight > 10.5,
      firstGenomeScoreCleared: firstGenome.score === undefined,
      secondGenomeScoreCleared: secondGenome.score === undefined,
    }).toEqual({
      optimizeCalls: 1,
      firstGenomeBiasImproved: true,
      firstGenomeWeightImproved: true,
      secondGenomeBiasImproved: true,
      secondGenomeWeightImproved: true,
      firstGenomeScoreCleared: true,
      secondGenomeScoreCleared: true,
    });
  });

  it('skips heuristic teacher fitting for the NARX profile before rollout refinement', async () => {
    const templateTelemetry = createMockNetworkTelemetry();
    const templateGenome = createMockNetwork({
      telemetry: templateTelemetry,
    });
    const optimizeWarmStartTemplateNetwork = jest
      .fn<Network, [Network, number]>()
      .mockReturnValue(createMockNetwork());
    const dependencies: WorkerWarmStartDependencies = {
      buildHeuristicPretrainSet: () => [
        { input: [0, 0], output: [1, 0] },
        { input: [1, 1], output: [0, 1] },
      ],
      optimizeWarmStartTemplateNetwork,
    };
    const neatController = {
      generation: 0,
      population: [templateGenome],
    } as unknown as Neat;

    await warmStartWorkerGenerationZeroIfNeeded(
      neatController,
      {
        architectureProfileId: 'narx',
        workerInitSeed: 123,
        generationZeroWarmStartApplied: false,
      },
      dependencies,
    );

    expect({
      optimizeCalls: optimizeWarmStartTemplateNetwork.mock.calls.length,
      trainCalls: templateTelemetry.trainCalls,
      clearCalls: templateTelemetry.clearCalls,
      activateCalls: templateTelemetry.activateCalls,
      propagateCalls: templateTelemetry.propagateCalls,
    }).toEqual({
      optimizeCalls: 1,
      trainCalls: 0,
      clearCalls: 0,
      activateCalls: 0,
      propagateCalls: 0,
    });
  });

  it('skips heuristic teacher fitting for the GRU profile before rollout refinement', async () => {
    const templateTelemetry = createMockNetworkTelemetry();
    const templateGenome = createMockNetwork({
      telemetry: templateTelemetry,
    });
    const optimizeWarmStartTemplateNetwork = jest
      .fn<Network, [Network, number]>()
      .mockReturnValue(createMockNetwork());
    const dependencies: WorkerWarmStartDependencies = {
      buildHeuristicPretrainSet: () => [
        { input: [0, 0], output: [1, 0] },
        { input: [1, 1], output: [0, 1] },
      ],
      optimizeWarmStartTemplateNetwork,
    };
    const neatController = {
      generation: 0,
      population: [templateGenome],
    } as unknown as Neat;

    await warmStartWorkerGenerationZeroIfNeeded(
      neatController,
      {
        architectureProfileId: 'gru',
        workerInitSeed: 123,
        generationZeroWarmStartApplied: false,
      },
      dependencies,
    );

    expect({
      optimizeCalls: optimizeWarmStartTemplateNetwork.mock.calls.length,
      trainCalls: templateTelemetry.trainCalls,
      clearCalls: templateTelemetry.clearCalls,
      activateCalls: templateTelemetry.activateCalls,
      propagateCalls: templateTelemetry.propagateCalls,
    }).toEqual({
      optimizeCalls: 1,
      trainCalls: 0,
      clearCalls: 0,
      activateCalls: 0,
      propagateCalls: 0,
    });
  });

  it('passes a 10s deadline to recurrent rollout refinement without waiting for the full limit', async () => {
    const optimizedTemplateNetwork = createMockNetwork();
    let capturedWarmStartDeadline: WorkerWarmStartDeadline | undefined;
    const dependencies: WorkerWarmStartDependencies = {
      buildHeuristicPretrainSet: jest.fn(() => []),
      optimizeWarmStartTemplateNetwork: (
        templateNetwork,
        _workerInitSeed,
        _architectureProfileId,
        warmStartDeadline,
      ) => {
        capturedWarmStartDeadline = warmStartDeadline;
        return optimizedTemplateNetwork ?? templateNetwork;
      },
      resolveCurrentTimeMs: jest.fn(() => 1_000),
    };
    const neatController = {
      generation: 0,
      population: [createMockNetwork()],
    } as unknown as Neat;

    await warmStartWorkerGenerationZeroIfNeeded(
      neatController,
      {
        architectureProfileId: 'gru',
        workerInitSeed: 123,
        generationZeroWarmStartApplied: false,
      },
      dependencies,
    );

    expect({
      deadlineMs: capturedWarmStartDeadline?.expiresAtMs,
      clockCalls: (dependencies.resolveCurrentTimeMs as jest.Mock).mock.calls
        .length,
      heuristicBuildCalls: (dependencies.buildHeuristicPretrainSet as jest.Mock)
        .mock.calls.length,
    }).toEqual({
      deadlineMs: 11_000,
      clockCalls: 1,
      heuristicBuildCalls: 0,
    });
  });

  it('passes the recurrent warm-start deadline into active rollouts as a hard stop hook', async () => {
    let currentTimeMs = 1_000;
    const shouldStopSnapshots: boolean[] = [];
    const rolloutEpisode = jest.fn((_templateNetwork, rolloutOptions) => {
      shouldStopSnapshots.push(rolloutOptions.shouldStop?.() ?? false);
      currentTimeMs = 11_000;
      shouldStopSnapshots.push(rolloutOptions.shouldStop?.() ?? false);

      return {
        done: true,
        doneReason: 'timeout' as const,
        fitness: 0,
        fitnessBreakdown: {
          denseShaping: 0,
          pipeProgress: 0,
          survival: 0,
          terminalShaping: 0,
        },
        framesSurvived: 1,
        pipesPassed: 0,
      };
    });
    const dependencies: WorkerWarmStartDependencies = {
      buildHeuristicPretrainSet: jest.fn(() => []),
      resolveCurrentTimeMs: jest.fn(() => currentTimeMs),
      rolloutEpisode,
    };

    await warmStartWorkerGenerationZeroIfNeeded(
      {
        generation: 0,
        population: [createMockNetwork()],
      } as unknown as Neat,
      {
        architectureProfileId: 'lstm',
        workerInitSeed: 123,
        generationZeroWarmStartApplied: false,
      },
      dependencies,
    );

    expect({
      rolloutCallCount: rolloutEpisode.mock.calls.length,
      shouldStopSnapshots,
    }).toEqual({
      rolloutCallCount: 1,
      shouldStopSnapshots: [false, true],
    });
  });

  it('treats warm-start optimizer failures as best-effort and marks the assist complete', async () => {
    const warmStartState = {
      architectureProfileId: 'lstm' as const,
      workerInitSeed: 123,
      generationZeroWarmStartApplied: false,
    };
    const dependencies: WorkerWarmStartDependencies = {
      buildHeuristicPretrainSet: jest.fn(() => []),
      optimizeWarmStartTemplateNetwork: () => {
        throw new Error('synthetic warm-start failure');
      },
    };

    await warmStartWorkerGenerationZeroIfNeeded(
      {
        generation: 0,
        population: [createMockNetwork()],
      } as unknown as Neat,
      warmStartState,
      dependencies,
    );

    expect(warmStartState.generationZeroWarmStartApplied).toBe(true);
  });

  it('skips heuristic teacher fitting for the LSTM profile before rollout refinement', async () => {
    const templateTelemetry = createMockNetworkTelemetry();
    const templateGenome = createMockNetwork({
      telemetry: templateTelemetry,
    });
    const optimizeWarmStartTemplateNetwork = jest
      .fn<Network, [Network, number]>()
      .mockReturnValue(createMockNetwork());
    const dependencies: WorkerWarmStartDependencies = {
      buildHeuristicPretrainSet: () => [
        { input: [0, 0], output: [1, 0] },
        { input: [1, 1], output: [0, 1] },
      ],
      optimizeWarmStartTemplateNetwork,
    };
    const neatController = {
      generation: 0,
      population: [templateGenome],
    } as unknown as Neat;

    await warmStartWorkerGenerationZeroIfNeeded(
      neatController,
      {
        architectureProfileId: 'lstm',
        workerInitSeed: 123,
        generationZeroWarmStartApplied: false,
      },
      dependencies,
    );

    expect({
      optimizeCalls: optimizeWarmStartTemplateNetwork.mock.calls.length,
      trainCalls: templateTelemetry.trainCalls,
      clearCalls: templateTelemetry.clearCalls,
      activateCalls: templateTelemetry.activateCalls,
      propagateCalls: templateTelemetry.propagateCalls,
    }).toEqual({
      optimizeCalls: 1,
      trainCalls: 0,
      clearCalls: 0,
      activateCalls: 0,
      propagateCalls: 0,
    });
  });

  it('resolves a GRU-specific bounded rollout optimization plan for browser-safe warm-start refinement', () => {
    expect(resolveWarmStartRolloutOptimizationPlan('gru')).toEqual({
      optimizationStepCount: 8,
      rolloutSeedCount: 4,
      timeLimitMs: RECURRENT_WARM_START_TIME_LIMIT_MS,
    });
  });

  it('resolves a stronger LSTM-specific rollout optimization plan for browser-safe warm-start refinement', () => {
    expect(resolveWarmStartRolloutOptimizationPlan('lstm')).toEqual({
      optimizationStepCount: 16,
      rolloutSeedCount: 6,
      timeLimitMs: RECURRENT_WARM_START_TIME_LIMIT_MS,
    });
  });

  it('scores GRU warm-start rollouts with pipe progress as the dominant signal', () => {
    expect(
      resolveWarmStartEvaluationScore(
        {
          seedCount: 4,
          meanFitness: 11,
          medianFitness: 10,
          p90Fitness: 12,
          fitnessStdDev: 8,
          robustFitness: 7.5,
          meanPipesPassed: 2,
          meanFramesSurvived: 180,
        },
        'gru',
      ),
    ).toBe(20_176);
  });

  it('scores LSTM warm-start rollouts with pipe progress as the dominant signal', () => {
    expect(
      resolveWarmStartEvaluationScore(
        {
          seedCount: 4,
          meanFitness: 11,
          medianFitness: 10,
          p90Fitness: 12,
          fitnessStdDev: 8,
          robustFitness: 7.5,
          meanPipesPassed: 2,
          meanFramesSurvived: 180,
        },
        'lstm',
      ),
    ).toBe(20_176);
  });

  it('resolves a stronger rollout optimization plan for NARX warm-start refinement', () => {
    expect(resolveWarmStartRolloutOptimizationPlan('narx')).toEqual({
      optimizationStepCount: 20,
      rolloutSeedCount: 7,
      timeLimitMs: RECURRENT_WARM_START_TIME_LIMIT_MS,
    });
  });

  it('scores NARX warm-start rollouts with pipe progress as the dominant signal', () => {
    expect(
      resolveWarmStartEvaluationScore(
        {
          seedCount: 7,
          meanFitness: 10,
          medianFitness: 9,
          p90Fitness: 12,
          fitnessStdDev: 8,
          robustFitness: 7.2,
          meanPipesPassed: 2,
          meanFramesSurvived: 180,
        },
        'narx',
      ),
    ).toBe(20_176);
  });

  it('uses rollout-only refinement for the NARX warm-start strategy', () => {
    expect(resolveWorkerWarmStartTeacherStrategy('narx')).toBe('rollout-only');
  });

  it('uses rollout-only refinement for the GRU warm-start strategy', () => {
    expect(resolveWorkerWarmStartTeacherStrategy('gru')).toBe('rollout-only');
  });

  it('uses rollout-only refinement for the LSTM warm-start strategy', () => {
    expect(resolveWorkerWarmStartTeacherStrategy('lstm')).toBe('rollout-only');
  });
});

describe('resolveHeuristicTeacherFlapDecisionFromObservationVector', () => {
  it('labels warm-start samples from the compact live controller shelf', () => {
    expect(
      resolveHeuristicTeacherFlapDecisionFromObservationVector([
        0.62, -0.12, 0.31, 0.18, 0.3, 0.66, 0.12, 0.04, -0.1,
      ]),
    ).toBe(true);
  });

  it('ignores trailing legacy-style extras beyond the live controller input width', () => {
    const baseObservationVector = [
      0.62, -0.12, 0.31, 0.18, 0.3, 0.66, 0.12, 0.04, -0.1,
    ];

    expect({
      baseDecision: resolveHeuristicTeacherFlapDecisionFromObservationVector(
        baseObservationVector,
      ),
      extendedDecision:
        resolveHeuristicTeacherFlapDecisionFromObservationVector([
          ...baseObservationVector,
          0.99,
          -0.99,
          0.77,
          -0.77,
        ]),
      expectedInputWidth: FLAPPY_NETWORK_INPUT_SIZE,
    }).toEqual({
      baseDecision: true,
      extendedDecision: true,
      expectedInputWidth: FLAPPY_NETWORK_INPUT_SIZE,
    });
  });

  it('refuses incomplete vectors so warm-start cannot silently invent missing channels', () => {
    expect(
      resolveHeuristicTeacherFlapDecisionFromObservationVector([
        0.62, -0.12, 0.31, 0.18, 0.3,
      ]),
    ).toBe(false);
  });
});

type MockNetworkTelemetry = {
  activateCalls: number;
  clearCalls: number;
  propagateCalls: number;
  trainCalls: number;
};

/**
 * Creates the call-count telemetry bag shared by mock networks in this suite.
 *
 * @returns Mutable telemetry counters for one mock network family.
 */
function createMockNetworkTelemetry(): MockNetworkTelemetry {
  return {
    activateCalls: 0,
    clearCalls: 0,
    propagateCalls: 0,
    trainCalls: 0,
  };
}

/**
 * Creates a narrow mock `Network` that exposes the warm-start methods under test.
 *
 * @param options - Optional parameters and shared telemetry for the mock network.
 * @returns Mock network cast to the production `Network` shape.
 */
function createMockNetwork(options?: {
  nodeBiases?: number[];
  connectionWeights?: number[];
  telemetry?: MockNetworkTelemetry;
}): Network {
  const nodeBiases = [...(options?.nodeBiases ?? [0])];
  const connectionWeights = [...(options?.connectionWeights ?? [0])];
  const telemetry = options?.telemetry ?? createMockNetworkTelemetry();

  return {
    nodes: nodeBiases.map((biasValue) => ({ bias: biasValue })),
    connections: connectionWeights.map((weightValue) => ({
      weight: weightValue,
    })),
    score: 5,
    clone() {
      return createMockNetwork({
        nodeBiases,
        connectionWeights,
        telemetry,
      });
    },
    activate: jest.fn((inputValues: number[]) => {
      telemetry.activateCalls += 1;
      return inputValues;
    }),
    clear: jest.fn(() => {
      telemetry.clearCalls += 1;
    }),
    propagate: jest.fn(() => {
      telemetry.propagateCalls += 1;
    }),
    train: jest.fn(() => {
      telemetry.trainCalls += 1;
    }),
  } as unknown as Network;
}
