import type { Neat } from '../../../src/neataptic';
import type Network from '../../../src/architecture/network';
import { FLAPPY_NETWORK_INPUT_SIZE } from '../constants/constants';
import {
  resolveHeuristicTeacherFlapDecisionFromObservationVector,
  resolveWarmStartEvaluationScore,
  resolveWarmStartRolloutOptimizationPlan,
  resolveWorkerWarmStartTeacherStrategy,
  warmStartWorkerGenerationZeroIfNeeded,
  type WorkerWarmStartDependencies,
} from './flappy-evolution-worker.warm-start.service';

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

  it('applies heuristic teacher fitting to the NARX profile before rollout refinement', async () => {
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
      trainCalls: 1,
      clearCalls: 0,
      activateCalls: 0,
      propagateCalls: 0,
    });
  });

  it('applies heuristic teacher fitting to the GRU profile before rollout refinement', async () => {
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
      trainCalls: 1,
      clearCalls: 0,
      activateCalls: 0,
      propagateCalls: 0,
    });
  });

  it('applies heuristic teacher fitting to the LSTM profile before rollout refinement', async () => {
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
      trainCalls: 1,
      clearCalls: 0,
      activateCalls: 0,
      propagateCalls: 0,
    });
  });

  it('resolves a GRU-specific bounded rollout optimization plan for browser-safe warm-start refinement', () => {
    expect(resolveWarmStartRolloutOptimizationPlan('gru')).toEqual({
      optimizationStepCount: 8,
      rolloutSeedCount: 4,
    });
  });

  it('resolves a stronger LSTM-specific rollout optimization plan for browser-safe warm-start refinement', () => {
    expect(resolveWarmStartRolloutOptimizationPlan('lstm')).toEqual({
      optimizationStepCount: 16,
      rolloutSeedCount: 6,
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

  it('uses feed-forward teacher fitting for the NARX warm-start strategy', () => {
    expect(resolveWorkerWarmStartTeacherStrategy('narx')).toBe('feed-forward-teacher-fit');
  });

  it('uses feed-forward teacher fitting for the GRU warm-start strategy', () => {
    expect(resolveWorkerWarmStartTeacherStrategy('gru')).toBe('feed-forward-teacher-fit');
  });

  it('uses feed-forward teacher fitting for the LSTM warm-start strategy', () => {
    expect(resolveWorkerWarmStartTeacherStrategy('lstm')).toBe('feed-forward-teacher-fit');
  });
});

describe('resolveHeuristicTeacherFlapDecisionFromObservationVector', () => {
  it('labels warm-start samples from the compact live controller shelf', () => {
    expect(
      resolveHeuristicTeacherFlapDecisionFromObservationVector([
        0.62, -0.12, 0.31, 0.18, 0.3, 0.66,
      ]),
    ).toBe(true);
  });

  it('ignores trailing legacy-style extras beyond the live controller input width', () => {
    const baseObservationVector = [0.62, -0.12, 0.31, 0.18, 0.3, 0.66];

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
      expectedInputWidth: 6,
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

function createMockNetworkTelemetry(): MockNetworkTelemetry {
  return {
    activateCalls: 0,
    clearCalls: 0,
    propagateCalls: 0,
    trainCalls: 0,
  };
}

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
