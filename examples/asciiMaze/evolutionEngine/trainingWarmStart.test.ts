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
      connectionPairs: [
        [0, 1],
        [1, 0],
      ],
      trainedBiases: [1.5, -1.25],
      trainedWeights: [2.25, -2.5],
      initialScore: 11,
    });
    const secondGenome = createMockTrainableNetwork({
      initialBiases: [5, 6],
      initialWeights: [7, 8],
      connectionPairs: [
        [0, 1],
        [1, 0],
      ],
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
      connectionPairs: [
        [0, 1],
        [1, 0],
      ],
      trainedBiases: [1.5, -1.25],
      trainedWeights: [2.25, -2.5],
    });
    const secondGenome = createMockTrainableNetwork({
      initialBiases: [5, 6],
      initialWeights: [7, 8],
      connectionPairs: [
        [0, 1],
        [1, 0],
      ],
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

  it('uses the first trainable genome in the population instead of aborting warm-start when population[0] is not trainable', () => {
    const state = createEngineState();
    const untrainableGenome = {
      nodes: [
        { bias: 100, index: 0 },
        { bias: 200, index: 1 },
      ],
      connections: [{ from: { index: 0 }, to: { index: 1 }, weight: 300 }],
    } as unknown as Network;
    const trainableGenome = createMockTrainableNetwork({
      initialBiases: [0.1, -0.2],
      initialWeights: [0.3],
      connectionPairs: [[0, 1]],
      trainedBiases: [1.5, -1.25],
      trainedWeights: [2.25],
    });
    const copiedGenome = createMockTrainableNetwork({
      initialBiases: [5, 6],
      initialWeights: [7],
      connectionPairs: [[0, 1]],
      trainedBiases: [9, 10],
      trainedWeights: [11],
    });

    pretrainPopulationWarmStart(
      {
        population: [
          untrainableGenome,
          trainableGenome.network,
          copiedGenome.network,
        ],
      } as Neat,
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
      copiedBiases: copiedGenome.network.nodes.map((node) => node.bias),
      copiedWeights: copiedGenome.network.connections.map(
        (connection) => connection.weight,
      ),
      trainableGenomeTrainCalls: trainableGenome.train.mock.calls.length,
      untrainableBiases: untrainableGenome.nodes.map((node) => node.bias),
    }).toEqual({
      copiedBiases: [1.5, -1.25],
      copiedWeights: [2.25],
      trainableGenomeTrainCalls: 0,
      untrainableBiases: [100, 200],
    });
  });

  it('adds template-only compass connections to matching genomes instead of copying by raw connection index', () => {
    const state = createEngineState();
    const templateGenome = createMockTrainableNetwork({
      initialBiases: [0.1, -0.2],
      initialWeights: [0.3],
      connectionPairs: [[0, 1]],
      trainedBiases: [1.5, -1.25],
      trainedWeights: [2.25],
    });
    const copiedGenome = createMockTrainableNetwork({
      initialBiases: [5, 6],
      initialWeights: [7],
      connectionPairs: [[0, 1]],
      trainedBiases: [9, 10],
      trainedWeights: [11],
    });

    pretrainPopulationWarmStart(
      { population: [templateGenome.network, copiedGenome.network] } as Neat,
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
      (network) => {
        const fromNode = network.nodes[1];
        const toNode = network.nodes[0];
        network.connect?.(fromNode, toNode, 3.5);
      },
      () => {},
    );

    expect(
      copiedGenome.network.connections.map((connection) => ({
        fromIndex: connection.from.index,
        toIndex: connection.to.index,
        weight: connection.weight,
      })),
    ).toEqual([
      { fromIndex: 0, toIndex: 1, weight: 2.25 },
      { fromIndex: 1, toIndex: 0, weight: 3.5 },
    ]);
  });
});

function createMockTrainableNetwork(options: {
  initialBiases: number[];
  initialWeights: number[];
  connectionPairs: Array<readonly [number, number]>;
  trainedBiases: number[];
  trainedWeights: number[];
  initialScore?: number;
}): {
  network: Network;
  train: ReturnType<typeof jest.fn>;
} {
  type MockNode = { bias: number; index: number };
  type MockConnection = { from: MockNode; to: MockNode; weight: number };
  type MockTrainableNetwork = {
    clone: ReturnType<typeof jest.fn>;
    connect: ReturnType<typeof jest.fn>;
    connections: MockConnection[];
    nodes: MockNode[];
    score?: number;
    train: ReturnType<typeof jest.fn>;
  };

  const train = jest.fn();
  const createNodes = (biases: number[]) =>
    biases.map((bias, nodeIndex) => ({ bias, index: nodeIndex }));
  const createConnections = (
    weights: number[],
    nodes: MockNode[],
  ): MockConnection[] =>
    weights.map((weight, connectionIndex) => {
      const [fromIndex, toIndex] =
        options.connectionPairs[connectionIndex] ?? options.connectionPairs[0];

      return {
        from: nodes[fromIndex],
        to: nodes[toIndex],
        weight,
      };
    });
  const nodes = createNodes(options.initialBiases);
  const network = {} as MockTrainableNetwork;
  network.nodes = nodes;
  network.connections = createConnections(options.initialWeights, nodes);
  network.score = options.initialScore;
  network.train = train;
  network.connect = jest.fn(
    (fromNode: MockNode, toNode: MockNode, weight = 0) => {
      const createdConnection: MockConnection = {
        from: fromNode,
        to: toNode,
        weight,
      };
      network.connections.push(createdConnection);
      return [createdConnection];
    },
  );
  network.clone = jest.fn(() => {
    const cloneTrain = jest.fn(() => {
      clonedNetwork.nodes.forEach((node, nodeIndex) => {
        node.bias = options.trainedBiases[nodeIndex] ?? node.bias;
      });
      clonedNetwork.connections.forEach((connection, connectionIndex) => {
        connection.weight =
          options.trainedWeights[connectionIndex] ?? connection.weight;
      });
    });
    const clonedNodes = createNodes(options.initialBiases);
    const clonedNetwork = {
      nodes: clonedNodes,
      connections: createConnections(options.initialWeights, clonedNodes),
      connect: jest.fn((fromNode: MockNode, toNode: MockNode, weight = 0) => {
        const createdConnection: MockConnection = {
          from: fromNode,
          to: toNode,
          weight,
        };
        clonedNetwork.connections.push(createdConnection);
        return [createdConnection];
      }),
      train: cloneTrain,
    } as MockTrainableNetwork;
    return clonedNetwork;
  });

  return {
    network: network as unknown as Network,
    train,
  };
}
