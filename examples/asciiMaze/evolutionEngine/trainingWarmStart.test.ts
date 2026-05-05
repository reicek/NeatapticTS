import { describe, expect, it, jest } from '@jest/globals';
import type { Neat, Network } from '../../../src/neataptic';
import { createEngineState, reseedRngState } from './engineState';
import { pretrainPopulationWarmStart } from './trainingWarmStart';

describe('pretrainPopulationWarmStart', () => {
  it('trains one cloned template then copies its tuned parameters across the population when noise is disabled', () => {
    const state = createEngineState();
    const firstGenome = createMockTrainableNetwork({
      initialBiases: [0.1, -0.2],
      initialWeights: [0.3, -0.4],
      trainedBiases: [1.5, -1.25],
      trainedWeights: [2.25, -2.5],
      initialScore: 11,
    });
    const secondGenome = createMockTrainableNetwork({
      initialBiases: [5, 6],
      initialWeights: [7, 8],
      trainedBiases: [9, 10],
      trainedWeights: [11, 12],
      initialScore: 13,
    });

    pretrainPopulationWarmStart(
      { population: [firstGenome.network, secondGenome.network] } as Neat,
      [{ input: [1, 0, 0, 0, 0, 0], output: [1, 0, 0, 0] }],
      {
        PRETRAIN_MAX_ITER: 60,
        PRETRAIN_BASE_ITER: 8,
        DEFAULT_TRAIN_ERROR: 0.01,
        DEFAULT_PRETRAIN_RATE: 0.002,
        DEFAULT_PRETRAIN_MOMENTUM: 0.1,
        DEFAULT_TRAIN_BATCH_SMALL: 2,
        TEMPLATE_WEIGHT_NOISE_STDDEV: 0,
        TEMPLATE_BIAS_NOISE_STDDEV: 0,
      },
      state,
      () => {},
      () => {},
    );

    expect({
      firstBiases: firstGenome.network.nodes.map((node) => node.bias),
      firstScore: firstGenome.network.score,
      firstTrainCalls: firstGenome.train.mock.calls.length,
      firstWeights: firstGenome.network.connections.map(
        (connection) => connection.weight,
      ),
      secondBiases: secondGenome.network.nodes.map((node) => node.bias),
      secondScore: secondGenome.network.score,
      secondTrainCalls: secondGenome.train.mock.calls.length,
      secondWeights: secondGenome.network.connections.map(
        (connection) => connection.weight,
      ),
    }).toEqual({
      firstBiases: [1.5, -1.25],
      firstScore: undefined,
      firstTrainCalls: 0,
      firstWeights: [2.25, -2.5],
      secondBiases: [1.5, -1.25],
      secondScore: undefined,
      secondTrainCalls: 0,
      secondWeights: [2.25, -2.5],
    });
  });

  it('adds deterministic copy noise so genomes do not stay exact template clones by default', () => {
    const state = createEngineState();
    reseedRngState(1234, state);
    const firstGenome = createMockTrainableNetwork({
      initialBiases: [0.1, -0.2],
      initialWeights: [0.3, -0.4],
      trainedBiases: [1.5, -1.25],
      trainedWeights: [2.25, -2.5],
    });
    const secondGenome = createMockTrainableNetwork({
      initialBiases: [5, 6],
      initialWeights: [7, 8],
      trainedBiases: [9, 10],
      trainedWeights: [11, 12],
    });

    pretrainPopulationWarmStart(
      { population: [firstGenome.network, secondGenome.network] } as Neat,
      [{ input: [1, 0, 0, 0, 0, 0], output: [1, 0, 0, 0] }],
      {
        PRETRAIN_MAX_ITER: 60,
        PRETRAIN_BASE_ITER: 8,
        DEFAULT_TRAIN_ERROR: 0.01,
        DEFAULT_PRETRAIN_RATE: 0.002,
        DEFAULT_PRETRAIN_MOMENTUM: 0.1,
        DEFAULT_TRAIN_BATCH_SMALL: 2,
        TEMPLATE_WEIGHT_NOISE_STDDEV: 0.08,
        TEMPLATE_BIAS_NOISE_STDDEV: 0.03,
      },
      state,
      () => {},
      () => {},
    );

    expect({
      firstBiasesMatchTemplate:
        JSON.stringify(firstGenome.network.nodes.map((node) => node.bias)) ===
        JSON.stringify([1.5, -1.25]),
      firstWeightsMatchTemplate:
        JSON.stringify(
          firstGenome.network.connections.map(
            (connection) => connection.weight,
          ),
        ) === JSON.stringify([2.25, -2.5]),
      genomesMatchExactly:
        JSON.stringify(firstGenome.network.nodes.map((node) => node.bias)) ===
          JSON.stringify(secondGenome.network.nodes.map((node) => node.bias)) &&
        JSON.stringify(
          firstGenome.network.connections.map(
            (connection) => connection.weight,
          ),
        ) ===
          JSON.stringify(
            secondGenome.network.connections.map(
              (connection) => connection.weight,
            ),
          ),
    }).toEqual({
      firstBiasesMatchTemplate: false,
      firstWeightsMatchTemplate: false,
      genomesMatchExactly: false,
    });
  });
});

function createMockTrainableNetwork(options: {
  initialBiases: number[];
  initialWeights: number[];
  trainedBiases: number[];
  trainedWeights: number[];
  initialScore?: number;
}): {
  network: Network;
  train: ReturnType<typeof jest.fn>;
} {
  const train = jest.fn();
  const network = {
    nodes: options.initialBiases.map((bias) => ({ bias })),
    connections: options.initialWeights.map((weight) => ({ weight })),
    score: options.initialScore,
    train,
    clone: jest.fn(() => {
      const cloneTrain = jest.fn((_dataset: unknown, _config: unknown) => {
        clonedNetwork.nodes.forEach((node, nodeIndex) => {
          node.bias = options.trainedBiases[nodeIndex] ?? node.bias;
        });
        clonedNetwork.connections.forEach((connection, connectionIndex) => {
          connection.weight =
            options.trainedWeights[connectionIndex] ?? connection.weight;
        });
      });
      const clonedNetwork = {
        nodes: options.initialBiases.map((bias) => ({ bias })),
        connections: options.initialWeights.map((weight) => ({ weight })),
        train: cloneTrain,
      };
      return clonedNetwork;
    }),
  } as unknown as Network;

  return {
    network,
    train,
  };
}
