import type { Neat } from '../../../src/neataptic';
import type Network from '../../../src/architecture/network';
import {
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
      secondGenomeWeightImproved:
        secondGenome.connections[0].weight > 10.5,
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

  it('keeps the NARX profile on rollout-only warm-start instead of heuristic teacher fitting', async () => {
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

  it('keeps the GRU profile on rollout-only refinement while using stronger profile-specific rollout tuning', async () => {
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

  it('keeps the heavier LSTM profile on rollout-only refinement', async () => {
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

  it('keeps the NARX warm-start strategy on rollout-only refinement', () => {
    expect(resolveWorkerWarmStartTeacherStrategy('narx')).toBe('rollout-only');
  });

  it('keeps the GRU warm-start strategy on rollout-only refinement', () => {
    expect(resolveWorkerWarmStartTeacherStrategy('gru')).toBe('rollout-only');
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
