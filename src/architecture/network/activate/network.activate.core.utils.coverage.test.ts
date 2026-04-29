import type Network from '../network';
import { activationArrayPool } from '../../activationArrayPool/activationArrayPool';
import { NetworkActivateCorruptedStructureError } from './network.activate.errors';
import { activate, gaussianRand } from './network.activate.core.utils';
import {
  type ActivateRuntimeNetworkProps,
  type ActivationStats,
} from './network.activate.utils.types';
import {
  resolveActivationTraversalNodes,
  resolveInputValuesByNodeId,
  resolveOrderedOutputNodes,
} from './network.activate.schedule.utils';

jest.mock('../../activationArrayPool/activationArrayPool', () => ({
  activationArrayPool: {
    acquire: jest.fn((outputSize: number) => new Array(outputSize).fill(0)),
    release: jest.fn(),
  },
}));

jest.mock('./network.activate.schedule.utils', () => ({
  resolveActivationTraversalNodes: jest.fn(),
  resolveInputValuesByNodeId: jest.fn(),
  resolveOrderedOutputNodes: jest.fn(),
}));

type MockNodeRole = 'input' | 'hidden' | 'output';

type MockNode = {
  type: MockNodeRole;
  geneId: number;
  activation: number;
  mask: number;
  activate: (value?: number) => void;
};

type MockLayer = {
  nodes: MockNode[];
  activate: (
    inputVector: number[] | undefined,
    isTraining: boolean,
  ) => number[] | undefined;
};

type MockConnection = {
  from: MockNode;
  to: MockNode;
  weight: number;
  _origWeightNoise?: number;
  _origWeight?: number;
  _wnLast?: number;
  dcMask?: number;
};

type MockActivationNetwork = ActivateRuntimeNetworkProps & {
  input: number;
  output: number;
  dropout: number;
  nodes: MockNode[];
  connections: MockConnection[];
  layers?: MockLayer[];
  _lastSkippedLayers?: number[];
};

type StatsSnapshot = ActivationStats;

const mockedAcquire = jest.mocked(activationArrayPool.acquire);
const mockedRelease = jest.mocked(activationArrayPool.release);
const mockedResolveActivationTraversalNodes = jest.mocked(
  resolveActivationTraversalNodes,
);
const mockedResolveInputValuesByNodeId = jest.mocked(
  resolveInputValuesByNodeId,
);
const mockedResolveOrderedOutputNodes = jest.mocked(resolveOrderedOutputNodes);

describe('network activate core utilities coverage chapter', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('gaussianRand()', () => {
    describe('given the random source yields zero samples before a usable pair', () => {
      it('retries until it can compute the Box-Muller sample', () => {
        // Arrange
        const randomSource = createRandomSequenceGenerator([0, 0.25, 0, 0.5]);

        // Act
        const gaussianSample = gaussianRand(randomSource);

        // Assert
        expect(roundNumber(gaussianSample)).toBe(-1.665109222315);
      });
    });

    describe('given the caller relies on the default random source', () => {
      it('uses Math.random when no generator override is provided', () => {
        // Arrange
        const mathRandomSpy = jest
          .spyOn(Math, 'random')
          .mockReturnValueOnce(0.5)
          .mockReturnValueOnce(0.5);

        // Act
        const gaussianSample = gaussianRand();
        mathRandomSpy.mockRestore();

        // Assert
        expect(roundNumber(gaussianSample)).toBe(-1.177410022515);
      });
    });
  });

  describe('activate()', () => {
    describe('given the input vector is missing', () => {
      describe('when activation validation formats the size mismatch message', () => {
        it('reports the input length as undefined', () => {
          // Arrange
          const activationNetwork = createMockNetwork({
            input: 2,
          });

          // Act
          const activateWithMissingInput = () =>
            activate.call(
              asNetwork(activationNetwork),
              undefined as unknown as number[],
            );

          // Assert
          expect(activateWithMissingInput).toThrow(
            'Input size mismatch: expected 2, got undefined',
          );
        });
      });
    });

    describe('given the network has no node collection', () => {
      describe('when activation reaches the structure guard', () => {
        it('throws the corrupted-structure error', () => {
          // Arrange
          const activationNetwork = createMockNetwork({
            nodes: [],
          });

          // Act
          const activateWithCorruptedStructure = () =>
            activate.call(asNetwork(activationNetwork), [0.5]);

          // Assert
          expect(activateWithCorruptedStructure).toThrow(
            NetworkActivateCorruptedStructureError,
          );
        });
      });
    });

    describe('given the fast slab path throws before activation continues', () => {
      describe('when activate falls back to the regular node traversal path', () => {
        it('returns the fallback output instead of propagating the fast slab error', () => {
          // Arrange
          const inputNode = createMockNode({ type: 'input', geneId: 1 });
          const outputNode = createMockNode({
            type: 'output',
            geneId: 2,
            activation: 0.6,
          });
          const fastSlabActivate = jest.fn(() => {
            throw new Error('forced fast slab failure');
          });
          const activationNetwork = createMockNetwork({
            nodes: [inputNode, outputNode],
            _canUseFastSlab: jest.fn(() => true),
            _fastSlabActivate: fastSlabActivate,
          });

          configureFallbackTraversal({
            activationNodes: [inputNode, outputNode],
            inputValuesByNodeId: new Map([[1, 0.25]]),
            orderedOutputNodes: [outputNode],
          });

          // Act
          const activationResult = activate.call(
            asNetwork(activationNetwork),
            [0.25],
            false,
          );

          // Assert
          expect({
            activationResult,
            fastSlabCalls: fastSlabActivate.mock.calls.length,
          }).toEqual({
            activationResult: [0.6],
            fastSlabCalls: 1,
          });
        });
      });
    });

    describe('given layered weight noise uses the scheduled fallback deviation', () => {
      describe('when training activation completes', () => {
        it('records the sampled weight-noise statistics and preserves the original weight snapshot', () => {
          // Arrange
          const inputNodeOne = createMockNode({ type: 'input', geneId: 1 });
          const inputNodeTwo = createMockNode({ type: 'input', geneId: 2 });
          const hiddenNodeOne = createMockNode({ type: 'hidden', geneId: 3 });
          const hiddenNodeTwo = createMockNode({ type: 'hidden', geneId: 4 });
          const outputNode = createMockNode({
            type: 'output',
            geneId: 5,
            activation: 10,
          });
          const stochasticDepthSchedule = jest.fn(() => [0.9]);
          const fallbackNoiseConnection = createMockConnection({
            from: hiddenNodeTwo,
            to: outputNode,
            weight: 1,
          });
          const activationNetwork = createMockNetwork({
            input: 2,
            output: 1,
            nodes: [
              inputNodeOne,
              inputNodeTwo,
              hiddenNodeOne,
              hiddenNodeTwo,
              outputNode,
            ],
            connections: [fallbackNoiseConnection],
            layers: [
              createMockLayer([inputNodeOne, inputNodeTwo], [2, 4]),
              createMockLayer([hiddenNodeOne], [6]),
              createMockLayer([hiddenNodeTwo], [8]),
              createMockLayer([outputNode], [10]),
            ],
            _rand: createRandomSequenceGenerator([0.5, 0.5]),
            _trainingStep: 3,
            _weightNoiseStd: 0.4,
            _weightNoisePerHidden: [0.05],
            _weightNoiseSchedule: jest.fn(() => 0.2),
            _stochasticDepth: [],
            _stochasticDepthSchedule: stochasticDepthSchedule,
          });
          const expectedNoiseMagnitude = Math.abs(
            0.2 * gaussianRand(createRandomSequenceGenerator([0.5, 0.5])),
          );

          // Act
          const activationResult = activate.call(
            asNetwork(activationNetwork),
            [0.25, 0.75],
            true,
          );

          // Assert
          expect({
            activationResult,
            connection: summarizeConnection(fallbackNoiseConnection),
            stochasticDepthScheduleCalls:
              stochasticDepthSchedule.mock.calls.length,
            trainingStep: activationNetwork._trainingStep,
            weightNoise: summarizeStats(activationNetwork)?.weightNoise,
          }).toEqual({
            activationResult: [10],
            connection: {
              dcMask: undefined,
              origWeight: undefined,
              origWeightNoise: 1,
              weight: roundNumber(1 - expectedNoiseMagnitude),
              sampledNoise: roundNumber(-expectedNoiseMagnitude),
            },
            stochasticDepthScheduleCalls: 0,
            trainingStep: 4,
            weightNoise: {
              count: 1,
              maxAbs: roundNumber(expectedNoiseMagnitude),
              meanAbs: roundNumber(expectedNoiseMagnitude),
              sumAbs: roundNumber(expectedNoiseMagnitude),
            },
          });
        });
      });
    });

    describe('given the stochastic-depth schedule returns a non-array value', () => {
      describe('when training activation completes', () => {
        it('keeps the existing stochastic-depth probabilities unchanged', () => {
          // Arrange
          const layeredScenario = createSimpleStochasticDepthScenario({
            scheduleResult: null as unknown as number[],
            stochasticDepth: [0.5],
            training: true,
          });

          // Act
          activate.call(
            asNetwork(layeredScenario.activationNetwork),
            layeredScenario.inputVector,
            true,
          );

          // Assert
          expect({
            skippedLayers: layeredScenario.activationNetwork._lastSkippedLayers,
            stochasticDepth: layeredScenario.activationNetwork._stochasticDepth,
          }).toEqual({
            skippedLayers: [1],
            stochasticDepth: [0.5],
          });
        });
      });
    });

    describe('given the stochastic-depth schedule returns invalid probabilities', () => {
      describe('when training activation completes', () => {
        it('rejects the updated probability vector', () => {
          // Arrange
          const layeredScenario = createSimpleStochasticDepthScenario({
            scheduleResult: [0],
            stochasticDepth: [0.5],
            training: true,
          });

          // Act
          activate.call(
            asNetwork(layeredScenario.activationNetwork),
            layeredScenario.inputVector,
            true,
          );

          // Assert
          expect({
            skippedLayers: layeredScenario.activationNetwork._lastSkippedLayers,
            stochasticDepth: layeredScenario.activationNetwork._stochasticDepth,
          }).toEqual({
            skippedLayers: [1],
            stochasticDepth: [0.5],
          });
        });
      });
    });

    describe('given the first hidden layer cannot reuse the previous activations', () => {
      describe('when stochastic-depth training activation completes', () => {
        it('updates the configured probabilities without skipping any hidden layers', () => {
          // Arrange
          const inputNodeOne = createMockNode({ type: 'input', geneId: 1 });
          const inputNodeTwo = createMockNode({ type: 'input', geneId: 2 });
          const hiddenNodeOne = createMockNode({ type: 'hidden', geneId: 3 });
          const hiddenNodeTwo = createMockNode({ type: 'hidden', geneId: 4 });
          const outputNode = createMockNode({
            type: 'output',
            geneId: 5,
            activation: 9,
          });
          const activationNetwork = createMockNetwork({
            input: 2,
            output: 1,
            nodes: [
              inputNodeOne,
              inputNodeTwo,
              hiddenNodeOne,
              hiddenNodeTwo,
              outputNode,
            ],
            connections: [],
            layers: [
              createMockLayer([inputNodeOne, inputNodeTwo], [1, 2]),
              createMockLayer([hiddenNodeOne], [3]),
              createMockLayer([hiddenNodeTwo], [4]),
              createMockLayer([outputNode], [9]),
            ],
            _rand: createRandomSequenceGenerator([0.9]),
            _stochasticDepth: [0.2],
            _stochasticDepthSchedule: jest.fn(() => [0.8]),
          });

          // Act
          const activationResult = activate.call(
            asNetwork(activationNetwork),
            [0.5, 0.25],
            true,
          );

          // Assert
          expect({
            activationResult,
            skippedLayers: activationNetwork._lastSkippedLayers,
            stochasticDepth: activationNetwork._stochasticDepth,
          }).toEqual({
            activationResult: [9],
            skippedLayers: [],
            stochasticDepth: [0.8],
          });
        });
      });
    });

    describe('given the hidden-layer override resolves to zero noise', () => {
      describe('when training activation inspects layered connections', () => {
        it('keeps the fresh connection weight unchanged while recording a zero sampled noise value', () => {
          // Arrange
          const inputNode = createMockNode({ type: 'input', geneId: 1 });
          const hiddenNode = createMockNode({ type: 'hidden', geneId: 2 });
          const outputNode = createMockNode({
            type: 'output',
            geneId: 3,
            activation: 5,
          });
          const preservedSnapshotConnection = createMockConnection({
            from: hiddenNode,
            to: outputNode,
            weight: 5,
            _origWeightNoise: 5,
          });
          const zeroNoiseConnection = createMockConnection({
            from: hiddenNode,
            to: outputNode,
            weight: 2,
          });
          const activationNetwork = createMockNetwork({
            input: 1,
            output: 1,
            nodes: [inputNode, hiddenNode, outputNode],
            connections: [preservedSnapshotConnection, zeroNoiseConnection],
            layers: [
              createMockLayer([inputNode], [1]),
              createMockLayer([hiddenNode], [2]),
              createMockLayer([outputNode], [5]),
            ],
            _weightNoisePerHidden: [0],
            _weightNoiseSchedule: jest.fn(() => 0),
          });

          // Act
          const activationResult = activate.call(
            asNetwork(activationNetwork),
            [0.5],
            true,
          );

          // Assert
          expect({
            activationResult,
            preservedSnapshotConnection: summarizeConnection(
              preservedSnapshotConnection,
            ),
            zeroNoiseConnection: summarizeConnection(zeroNoiseConnection),
          }).toEqual({
            activationResult: [5],
            preservedSnapshotConnection: {
              dcMask: undefined,
              origWeight: undefined,
              origWeightNoise: 5,
              sampledNoise: undefined,
              weight: 5,
            },
            zeroNoiseConnection: {
              dcMask: undefined,
              origWeight: undefined,
              origWeightNoise: 2,
              sampledNoise: 0,
              weight: 2,
            },
          });
        });
      });
    });

    describe('given the connection source is not part of any declared layer', () => {
      describe('when layered weight noise resolves the per-hidden deviation', () => {
        it('falls back to the scheduled global deviation', () => {
          // Arrange
          const inputNode = createMockNode({ type: 'input', geneId: 1 });
          const hiddenNode = createMockNode({ type: 'hidden', geneId: 2 });
          const orphanNode = createMockNode({ type: 'hidden', geneId: 99 });
          const outputNode = createMockNode({
            type: 'output',
            geneId: 3,
            activation: 4,
          });
          const fallbackNoiseConnection = createMockConnection({
            from: orphanNode,
            to: outputNode,
            weight: 1,
          });
          const activationNetwork = createMockNetwork({
            input: 1,
            output: 1,
            nodes: [inputNode, hiddenNode, orphanNode, outputNode],
            connections: [fallbackNoiseConnection],
            layers: [
              createMockLayer([inputNode], [1]),
              createMockLayer([hiddenNode], [2]),
              createMockLayer([outputNode], [4]),
            ],
            _rand: createRandomSequenceGenerator([0.5, 0.5]),
            _weightNoisePerHidden: [0.1],
            _weightNoiseSchedule: jest.fn(() => 0.3),
          });
          const expectedNoiseMagnitude = Math.abs(
            0.3 * gaussianRand(createRandomSequenceGenerator([0.5, 0.5])),
          );

          // Act
          activate.call(asNetwork(activationNetwork), [0.25], true);

          // Assert
          expect(summarizeConnection(fallbackNoiseConnection)).toEqual({
            dcMask: undefined,
            origWeight: undefined,
            origWeightNoise: 1,
            sampledNoise: roundNumber(-expectedNoiseMagnitude),
            weight: roundNumber(1 - expectedNoiseMagnitude),
          });
        });
      });
    });

    describe('given layer definitions disappear before source-layer lookup completes', () => {
      describe('when training activation falls back to raw traversal', () => {
        it('still applies the scheduled fallback noise without crashing the source lookup', () => {
          // Arrange
          const inputNode = createMockNode({ type: 'input', geneId: 1 });
          const hiddenNode = createMockNode({ type: 'hidden', geneId: 2 });
          const outputNode = createMockNode({
            type: 'output',
            geneId: 3,
            activation: 0.8,
          });
          const transientLayerConnection = createMockConnection({
            from: hiddenNode,
            to: outputNode,
            weight: 1,
          });
          const stableLayers = [
            createMockLayer([inputNode], [1]),
            createMockLayer([hiddenNode], [2]),
            createMockLayer([outputNode], [3]),
          ];
          const activationNetwork = createMockNetwork({
            input: 1,
            output: 1,
            nodes: [inputNode, hiddenNode, outputNode],
            connections: [transientLayerConnection],
            _rand: createRandomSequenceGenerator([0.5, 0.5]),
            _weightNoisePerHidden: [0.1],
            _weightNoiseSchedule: jest.fn(() => 0.2),
          });
          const expectedNoiseMagnitude = Math.abs(
            0.2 * gaussianRand(createRandomSequenceGenerator([0.5, 0.5])),
          );

          assignLayerSequence(activationNetwork, [
            stableLayers,
            undefined,
            undefined,
            undefined,
            undefined,
          ]);

          configureFallbackTraversal({
            activationNodes: [inputNode, hiddenNode, outputNode],
            inputValuesByNodeId: new Map([[1, 0.25]]),
            orderedOutputNodes: [outputNode],
          });

          // Act
          const activationResult = activate.call(
            asNetwork(activationNetwork),
            [0.25],
            true,
          );

          // Assert
          expect({
            activationResult,
            connection: summarizeConnection(transientLayerConnection),
          }).toEqual({
            activationResult: [0.8],
            connection: {
              dcMask: 1,
              origWeight: undefined,
              origWeightNoise: undefined,
              sampledNoise: roundNumber(-expectedNoiseMagnitude),
              weight: 1,
            },
          });
        });
      });
    });

    describe('given the stochastic-depth schedule returns the wrong number of probabilities', () => {
      describe('when training activation completes', () => {
        it('keeps the previous probability vector unchanged', () => {
          // Arrange
          const layeredScenario = createSimpleStochasticDepthScenario({
            scheduleResult: [0.1, 0.2],
            stochasticDepth: [0.5],
            training: true,
          });

          // Act
          activate.call(
            asNetwork(layeredScenario.activationNetwork),
            layeredScenario.inputVector,
            true,
          );

          // Assert
          expect({
            skippedLayers: layeredScenario.activationNetwork._lastSkippedLayers,
            stochasticDepth: layeredScenario.activationNetwork._stochasticDepth,
          }).toEqual({
            skippedLayers: [1],
            stochasticDepth: [0.5],
          });
        });
      });
    });

    describe('given the previous layer returns no activations', () => {
      describe('when stochastic-depth activation reaches the first hidden layer', () => {
        it('keeps the hidden layer active and leaves the output buffer unchanged', () => {
          // Arrange
          const inputNode = createMockNode({ type: 'input', geneId: 1 });
          const hiddenNode = createMockNode({ type: 'hidden', geneId: 2 });
          const outputNode = createMockNode({ type: 'output', geneId: 3 });
          const activationNetwork = createMockNetwork({
            input: 1,
            output: 1,
            nodes: [inputNode, hiddenNode, outputNode],
            connections: [],
            layers: [
              createMockLayer([inputNode], undefined),
              createMockLayer([hiddenNode], [3]),
              createMockLayer([outputNode], undefined),
            ],
            _rand: createRandomSequenceGenerator([0.9]),
            _stochasticDepth: [0.5],
          });

          // Act
          const activationResult = activate.call(
            asNetwork(activationNetwork),
            [0.25],
            true,
          );

          // Assert
          expect({
            activationResult,
            skippedLayers: activationNetwork._lastSkippedLayers,
          }).toEqual({
            activationResult: [0],
            skippedLayers: [],
          });
        });
      });
    });

    describe('given stochastic-depth activation runs outside training mode', () => {
      describe('when activation completes', () => {
        it('keeps every layer active and returns the output layer activations', () => {
          // Arrange
          const layeredScenario = createSimpleStochasticDepthScenario({
            scheduleResult: [0.7],
            stochasticDepth: [0.5],
            training: false,
          });

          // Act
          const activationResult = activate.call(
            asNetwork(layeredScenario.activationNetwork),
            layeredScenario.inputVector,
            false,
          );

          // Assert
          expect({
            activationResult,
            skippedLayers: layeredScenario.activationNetwork._lastSkippedLayers,
            stochasticDepth: layeredScenario.activationNetwork._stochasticDepth,
          }).toEqual({
            activationResult: [6],
            skippedLayers: [],
            stochasticDepth: [0.5],
          });
        });
      });
    });

    describe('given primary weight noise is disabled by the schedule in node traversal mode', () => {
      describe('when fallback weight noise runs', () => {
        it('initializes the fallback snapshot array and annotates the noisy connection', () => {
          // Arrange
          const inputNode = createMockNode({ type: 'input', geneId: 1 });
          const hiddenNode = createMockNode({ type: 'hidden', geneId: 2 });
          const outputNode = createMockNode({
            type: 'output',
            geneId: 3,
            activation: 0.75,
          });
          const preservedConnection = createMockConnection({
            from: inputNode,
            to: hiddenNode,
            weight: 2,
            _origWeightNoise: 2,
          });
          const noisyConnection = createMockConnection({
            from: hiddenNode,
            to: outputNode,
            weight: 1,
          });
          const activationNetwork = createMockNetwork({
            input: 1,
            output: 1,
            nodes: [inputNode, hiddenNode, outputNode],
            connections: [preservedConnection, noisyConnection],
            _rand: createRandomSequenceGenerator([0.5, 0.5]),
            _weightNoiseStd: 0.3,
            _weightNoiseSchedule: jest.fn(() => 0),
          });
          const expectedNoiseMagnitude = Math.abs(
            0.3 * gaussianRand(createRandomSequenceGenerator([0.5, 0.5])),
          );

          configureFallbackTraversal({
            activationNodes: [inputNode, hiddenNode, outputNode],
            inputValuesByNodeId: new Map([[1, 0.25]]),
            orderedOutputNodes: [outputNode],
          });

          // Act
          const activationResult = activate.call(
            asNetwork(activationNetwork),
            [0.25],
            true,
          );

          // Assert
          expect({
            activationResult,
            preservedConnection: summarizeConnection(preservedConnection),
            noisyConnection: summarizeConnection(noisyConnection),
            weightNoise: summarizeStats(activationNetwork)?.weightNoise,
            weightNoiseSnapshotSize: activationNetwork._wnOrig?.length,
          }).toEqual({
            activationResult: [0.75],
            preservedConnection: {
              dcMask: 1,
              origWeight: undefined,
              origWeightNoise: 2,
              sampledNoise: undefined,
              weight: 2,
            },
            noisyConnection: {
              dcMask: 1,
              origWeight: undefined,
              origWeightNoise: 1,
              sampledNoise: undefined,
              weight: roundNumber(1 - expectedNoiseMagnitude),
            },
            weightNoise: {
              count: 1,
              maxAbs: roundNumber(expectedNoiseMagnitude),
              meanAbs: roundNumber(expectedNoiseMagnitude),
              sumAbs: roundNumber(expectedNoiseMagnitude),
            },
            weightNoiseSnapshotSize: 2,
          });
        });
      });
    });

    describe('given fallback dropout keeps one hidden node active', () => {
      describe('when one hidden node survives the dropout mask', () => {
        it('tracks the dropped-node statistics without forcing a replacement node', () => {
          // Arrange
          const inputNode = createMockNode({ type: 'input', geneId: 1 });
          const hiddenNodeOne = createMockNode({ type: 'hidden', geneId: 2 });
          const hiddenNodeTwo = createMockNode({ type: 'hidden', geneId: 3 });
          const outputNode = createMockNode({
            type: 'output',
            geneId: 4,
            activation: 1,
          });
          const activationNetwork = createMockNetwork({
            input: 1,
            output: 1,
            dropout: 0.5,
            nodes: [inputNode, hiddenNodeOne, hiddenNodeTwo, outputNode],
            connections: [],
            _rand: createRandomSequenceGenerator([0.1, 0.9]),
          });

          configureFallbackTraversal({
            activationNodes: [
              inputNode,
              hiddenNodeOne,
              hiddenNodeTwo,
              outputNode,
            ],
            inputValuesByNodeId: new Map([[1, 0.5]]),
            orderedOutputNodes: [outputNode],
          });

          // Act
          activate.call(asNetwork(activationNetwork), [0.5], true);

          // Assert
          expect({
            hiddenMasks: [hiddenNodeOne.mask, hiddenNodeTwo.mask],
            stats: summarizeStats(activationNetwork),
          }).toEqual({
            hiddenMasks: [0, 1],
            stats: {
              droppedConnections: 0,
              droppedHiddenNodes: 1,
              skippedLayers: [],
              totalConnections: 0,
              totalHiddenNodes: 2,
              weightNoise: {
                count: 0,
                maxAbs: 0,
                meanAbs: 0,
                sumAbs: 0,
              },
            },
          });
        });
      });
    });

    describe('given fallback dropout drops every hidden node', () => {
      describe('when activation completes', () => {
        it('reactivates one hidden node to preserve a traversal path', () => {
          // Arrange
          const inputNode = createMockNode({ type: 'input', geneId: 1 });
          const hiddenNodeOne = createMockNode({ type: 'hidden', geneId: 2 });
          const hiddenNodeTwo = createMockNode({ type: 'hidden', geneId: 3 });
          const outputNode = createMockNode({
            type: 'output',
            geneId: 4,
            activation: 1,
          });
          const activationNetwork = createMockNetwork({
            input: 1,
            output: 1,
            dropout: 1,
            nodes: [inputNode, hiddenNodeOne, hiddenNodeTwo, outputNode],
            connections: [],
            _rand: createRandomSequenceGenerator([0.1, 0.2, 0.75]),
          });

          configureFallbackTraversal({
            activationNodes: [
              inputNode,
              hiddenNodeOne,
              hiddenNodeTwo,
              outputNode,
            ],
            inputValuesByNodeId: new Map([[1, 0.5]]),
            orderedOutputNodes: [outputNode],
          });

          // Act
          activate.call(asNetwork(activationNetwork), [0.5], true);

          // Assert
          expect({
            hiddenMasks: [hiddenNodeOne.mask, hiddenNodeTwo.mask],
            stats: summarizeStats(activationNetwork),
          }).toEqual({
            hiddenMasks: [0, 1],
            stats: {
              droppedConnections: 0,
              droppedHiddenNodes: 2,
              skippedLayers: [],
              totalConnections: 0,
              totalHiddenNodes: 2,
              weightNoise: {
                count: 0,
                maxAbs: 0,
                meanAbs: 0,
                sumAbs: 0,
              },
            },
          });
        });
      });
    });

    describe('given layered dropout keeps one hidden node active', () => {
      describe('when activation completes', () => {
        it('tracks the dropped hidden layer node without forcing a replacement activation', () => {
          // Arrange
          const inputNode = createMockNode({ type: 'input', geneId: 1 });
          const hiddenNodeOne = createMockNode({ type: 'hidden', geneId: 2 });
          const hiddenNodeTwo = createMockNode({ type: 'hidden', geneId: 3 });
          const outputNode = createMockNode({
            type: 'output',
            geneId: 4,
            activation: 6,
          });
          const activationNetwork = createMockNetwork({
            input: 1,
            output: 1,
            dropout: 0.5,
            nodes: [inputNode, hiddenNodeOne, hiddenNodeTwo, outputNode],
            connections: [],
            layers: [
              createMockLayer([inputNode], [1]),
              createMockLayer([hiddenNodeOne, hiddenNodeTwo], [7, 8]),
              createMockLayer([outputNode], [6]),
            ],
            _rand: createRandomSequenceGenerator([0.1, 0.9]),
          });

          // Act
          activate.call(asNetwork(activationNetwork), [0.5], true);

          // Assert
          expect({
            hiddenActivations: [
              hiddenNodeOne.activation,
              hiddenNodeTwo.activation,
            ],
            hiddenMasks: [hiddenNodeOne.mask, hiddenNodeTwo.mask],
            stats: summarizeStats(activationNetwork),
          }).toEqual({
            hiddenActivations: [0, 8],
            hiddenMasks: [0, 1],
            stats: {
              droppedConnections: 0,
              droppedHiddenNodes: 1,
              skippedLayers: [],
              totalConnections: 0,
              totalHiddenNodes: 2,
              weightNoise: {
                count: 0,
                maxAbs: 0,
                meanAbs: 0,
                sumAbs: 0,
              },
            },
          });
        });
      });
    });

    describe('given layered dropout drops every hidden node', () => {
      describe('when activation completes', () => {
        it('restores one hidden node activation from the raw layer output', () => {
          // Arrange
          const inputNode = createMockNode({ type: 'input', geneId: 1 });
          const hiddenNodeOne = createMockNode({ type: 'hidden', geneId: 2 });
          const hiddenNodeTwo = createMockNode({ type: 'hidden', geneId: 3 });
          const outputNode = createMockNode({
            type: 'output',
            geneId: 4,
            activation: 6,
          });
          const activationNetwork = createMockNetwork({
            input: 1,
            output: 1,
            dropout: 1,
            nodes: [inputNode, hiddenNodeOne, hiddenNodeTwo, outputNode],
            connections: [],
            layers: [
              createMockLayer([inputNode], [1]),
              createMockLayer([hiddenNodeOne, hiddenNodeTwo], [7, 8]),
              createMockLayer([outputNode], [6]),
            ],
            _rand: createRandomSequenceGenerator([0.1, 0.2, 0.75]),
          });

          // Act
          activate.call(asNetwork(activationNetwork), [0.5], true);

          // Assert
          expect({
            hiddenActivations: [
              hiddenNodeOne.activation,
              hiddenNodeTwo.activation,
            ],
            hiddenMasks: [hiddenNodeOne.mask, hiddenNodeTwo.mask],
            stats: summarizeStats(activationNetwork),
          }).toEqual({
            hiddenActivations: [0, 8],
            hiddenMasks: [0, 1],
            stats: {
              droppedConnections: 0,
              droppedHiddenNodes: 2,
              skippedLayers: [],
              totalConnections: 0,
              totalHiddenNodes: 2,
              weightNoise: {
                count: 0,
                maxAbs: 0,
                meanAbs: 0,
                sumAbs: 0,
              },
            },
          });
        });
      });
    });

    describe('given node traversal activation applies primary weight noise', () => {
      describe('when finalization restores original connection weights', () => {
        it('restores only the connections that still keep their original weight snapshot', () => {
          // Arrange
          const inputNode = createMockNode({ type: 'input', geneId: 1 });
          const hiddenNode = createMockNode({ type: 'hidden', geneId: 2 });
          const restorableOutputConnection = createMockConnection({
            from: inputNode,
            to: hiddenNode,
            weight: 1,
          });
          const deletedSnapshotConnection = createMockConnection({
            from: hiddenNode,
            to: createMockNode({ type: 'output', geneId: 3, activation: 0.5 }),
            weight: 2,
          });
          const outputNode = createMockNode({
            type: 'output',
            geneId: 3,
            activation: 0.5,
            onActivate: () => {
              delete deletedSnapshotConnection._origWeightNoise;
            },
          });
          deletedSnapshotConnection.to = outputNode;

          const activationNetwork = createMockNetwork({
            input: 1,
            output: 1,
            nodes: [inputNode, hiddenNode, outputNode],
            connections: [
              restorableOutputConnection,
              deletedSnapshotConnection,
            ],
            _rand: createRandomSequenceGenerator([0.5, 0.5, 0.5, 0.5]),
            _weightNoiseStd: 0.2,
          });
          const expectedNoiseMagnitude = Math.abs(
            0.2 * gaussianRand(createRandomSequenceGenerator([0.5, 0.5])),
          );

          configureFallbackTraversal({
            activationNodes: [inputNode, hiddenNode, outputNode],
            inputValuesByNodeId: new Map([[1, 0.25]]),
            orderedOutputNodes: [outputNode],
          });

          // Act
          const activationResult = activate.call(
            asNetwork(activationNetwork),
            [0.25],
            true,
          );

          // Assert
          expect({
            activationResult,
            restorableOutputConnection: summarizeConnection(
              restorableOutputConnection,
            ),
            deletedSnapshotConnection: summarizeConnection(
              deletedSnapshotConnection,
            ),
          }).toEqual({
            activationResult: [0.5],
            restorableOutputConnection: {
              dcMask: 1,
              origWeight: undefined,
              origWeightNoise: undefined,
              sampledNoise: roundNumber(-expectedNoiseMagnitude),
              weight: 1,
            },
            deletedSnapshotConnection: {
              dcMask: 1,
              origWeight: undefined,
              origWeightNoise: undefined,
              sampledNoise: roundNumber(-expectedNoiseMagnitude),
              weight: roundNumber(2 - expectedNoiseMagnitude),
            },
          });
        });
      });
    });

    describe('given node traversal activation applies drop-connect masks', () => {
      describe('when some connections drop and others restore their original weights', () => {
        it('tracks dropped connections while restoring the kept connection snapshots', () => {
          // Arrange
          const inputNode = createMockNode({ type: 'input', geneId: 1 });
          const outputNode = createMockNode({
            type: 'output',
            geneId: 2,
            activation: 0.4,
          });
          const droppedFreshConnection = createMockConnection({
            from: inputNode,
            to: outputNode,
            weight: 5,
          });
          const droppedSnapshotConnection = createMockConnection({
            from: inputNode,
            to: outputNode,
            weight: 6,
            _origWeight: 8,
          });
          const keptRestoredConnection = createMockConnection({
            from: inputNode,
            to: outputNode,
            weight: 9,
            _origWeight: 7,
          });
          const keptPlainConnection = createMockConnection({
            from: inputNode,
            to: outputNode,
            weight: 4,
          });
          const activationNetwork = createMockNetwork({
            input: 1,
            output: 1,
            nodes: [inputNode, outputNode],
            connections: [
              droppedFreshConnection,
              droppedSnapshotConnection,
              keptRestoredConnection,
              keptPlainConnection,
            ],
            _rand: createRandomSequenceGenerator([0.1, 0.2, 0.9, 0.9]),
            _dropConnectProb: 0.5,
          });

          configureFallbackTraversal({
            activationNodes: [inputNode, outputNode],
            inputValuesByNodeId: new Map([[1, 0.1]]),
            orderedOutputNodes: [outputNode],
          });

          // Act
          activate.call(asNetwork(activationNetwork), [0.1], true);

          // Assert
          expect({
            connections: activationNetwork.connections.map(summarizeConnection),
            droppedConnections:
              summarizeStats(activationNetwork)?.droppedConnections,
          }).toEqual({
            connections: [
              {
                dcMask: 0,
                origWeight: 5,
                origWeightNoise: undefined,
                sampledNoise: undefined,
                weight: 0,
              },
              {
                dcMask: 0,
                origWeight: 8,
                origWeightNoise: undefined,
                sampledNoise: undefined,
                weight: 0,
              },
              {
                dcMask: 1,
                origWeight: undefined,
                origWeightNoise: undefined,
                sampledNoise: undefined,
                weight: 7,
              },
              {
                dcMask: 1,
                origWeight: undefined,
                origWeightNoise: undefined,
                sampledNoise: undefined,
                weight: 4,
              },
            ],
            droppedConnections: 2,
          });
        });
      });
    });

    describe('given fallback weight noise already owns snapshot storage', () => {
      describe('when fallback noise is sampled again', () => {
        it('reuses the existing snapshot array instead of allocating a replacement', () => {
          // Arrange
          const inputNode = createMockNode({ type: 'input', geneId: 1 });
          const outputNode = createMockNode({
            type: 'output',
            geneId: 2,
            activation: 0.3,
          });
          const noisyConnection = createMockConnection({
            from: inputNode,
            to: outputNode,
            weight: 1,
          });
          const existingSnapshotArray: number[] = [];
          const activationNetwork = createMockNetwork({
            input: 1,
            output: 1,
            nodes: [inputNode, outputNode],
            connections: [noisyConnection],
            _rand: createRandomSequenceGenerator([0.5, 0.5]),
            _weightNoiseStd: 0.3,
            _weightNoiseSchedule: jest.fn(() => 0),
            _wnOrig: existingSnapshotArray,
          });

          configureFallbackTraversal({
            activationNodes: [inputNode, outputNode],
            inputValuesByNodeId: new Map([[1, 0.1]]),
            orderedOutputNodes: [outputNode],
          });

          // Act
          activate.call(asNetwork(activationNetwork), [0.1], true);

          // Assert
          expect({
            sameReference: activationNetwork._wnOrig === existingSnapshotArray,
            snapshotLength: activationNetwork._wnOrig?.length,
          }).toEqual({
            sameReference: true,
            snapshotLength: 0,
          });
        });
      });
    });

    describe('given stochastic-depth activation loses its layer list before execution starts', () => {
      describe('when the layered stochastic-depth helper begins', () => {
        it('returns the untouched pooled output buffer', () => {
          // Arrange
          const inputNode = createMockNode({ type: 'input', geneId: 1 });
          const stableLayers = [createMockLayer([inputNode], [4])];
          const activationNetwork = createMockNetwork({
            input: 1,
            output: 1,
            nodes: [inputNode],
            connections: [],
            _stochasticDepth: [0.4],
          });

          assignLayerSequence(activationNetwork, [
            stableLayers,
            stableLayers,
            undefined,
          ]);

          // Act
          const activationResult = activate.call(
            asNetwork(activationNetwork),
            [0.4],
            true,
          );

          // Assert
          expect(activationResult).toEqual([0]);
        });
      });
    });

    describe('given the stochastic-depth layer list disappears during skip resolution', () => {
      describe('when the helper decides whether to skip the current layer', () => {
        it('returns the current layer activations without skipping', () => {
          // Arrange
          const inputNode = createMockNode({ type: 'input', geneId: 1 });
          const stableLayers = [createMockLayer([inputNode], [4])];
          const activationNetwork = createMockNetwork({
            input: 1,
            output: 1,
            nodes: [inputNode],
            connections: [],
            _stochasticDepth: [0.4],
          });

          assignLayerSequence(activationNetwork, [
            stableLayers,
            stableLayers,
            stableLayers,
            stableLayers,
            stableLayers,
            undefined,
            stableLayers,
          ]);

          // Act
          const activationResult = activate.call(
            asNetwork(activationNetwork),
            [0.4],
            true,
          );

          // Assert
          expect(activationResult).toEqual([4]);
        });
      });
    });

    describe('given layered dropout loses its layer list before execution starts', () => {
      describe('when the layered dropout helper begins', () => {
        it('returns the untouched pooled output buffer', () => {
          // Arrange
          const inputNode = createMockNode({ type: 'input', geneId: 1 });
          const stableLayers = [createMockLayer([inputNode], [4])];
          const activationNetwork = createMockNetwork({
            input: 1,
            output: 1,
            nodes: [inputNode],
            connections: [],
          });

          assignLayerSequence(activationNetwork, [
            stableLayers,
            stableLayers,
            stableLayers,
            stableLayers,
            undefined,
          ]);

          // Act
          const activationResult = activate.call(
            asNetwork(activationNetwork),
            [0.4],
            true,
          );

          // Assert
          expect(activationResult).toEqual([0]);
        });
      });
    });
  });
});

function createMockNode(options: {
  type: MockNodeRole;
  geneId: number;
  activation?: number;
  onActivate?: (value?: number) => void;
}): MockNode {
  const mockNode: MockNode = {
    activation: options.activation ?? 0,
    geneId: options.geneId,
    mask: 1,
    type: options.type,
    activate: jest.fn((value?: number) => {
      if (value !== undefined) {
        mockNode.activation = value;
      }

      options.onActivate?.(value);
    }),
  };

  return mockNode;
}

function createMockLayer(
  nodes: MockNode[],
  layerActivations: number[] | undefined,
): MockLayer {
  return {
    nodes,
    activate: jest.fn((inputVector: number[] | undefined) => {
      if (inputVector) {
        nodes.forEach((node, nodeIndex) => {
          node.activation = inputVector[nodeIndex] ?? node.activation;
        });
      }

      if (layerActivations) {
        nodes.forEach((node, nodeIndex) => {
          node.activation = layerActivations[nodeIndex] ?? node.activation;
        });
      }

      return layerActivations;
    }),
  };
}

function createMockConnection(options: {
  from: MockNode;
  to: MockNode;
  weight: number;
  _origWeightNoise?: number;
  _origWeight?: number;
}): MockConnection {
  return {
    _origWeight: options._origWeight,
    _origWeightNoise: options._origWeightNoise,
    from: options.from,
    to: options.to,
    weight: options.weight,
  };
}

function createMockNetwork(
  overrides: Partial<MockActivationNetwork> = {},
): MockActivationNetwork {
  const inputNode = createMockNode({ type: 'input', geneId: 1 });
  const outputNode = createMockNode({
    type: 'output',
    geneId: 2,
    activation: 1,
  });

  return {
    _canUseFastSlab: jest.fn(() => false),
    _computeTopoOrder: jest.fn(),
    _dropConnectProb: 0,
    _fastSlabActivate: jest.fn((inputVector: number[]) => inputVector),
    _lastSkippedLayers: [],
    _rand: createRandomSequenceGenerator([]),
    _stochasticDepth: [],
    _topoDirty: false,
    _trainingStep: 0,
    _weightNoisePerHidden: [],
    _weightNoiseStd: 0,
    connections: [],
    dropout: 0,
    input: 1,
    nodes: [inputNode, outputNode],
    output: 1,
    ...overrides,
  };
}

function createSimpleStochasticDepthScenario(options: {
  scheduleResult: number[];
  stochasticDepth: number[];
  training: boolean;
}): {
  activationNetwork: MockActivationNetwork;
  inputVector: number[];
} {
  const inputNode = createMockNode({ type: 'input', geneId: 1 });
  const hiddenNode = createMockNode({ type: 'hidden', geneId: 2 });
  const outputNode = createMockNode({
    type: 'output',
    geneId: 3,
    activation: 6,
  });
  const activationNetwork = createMockNetwork({
    input: 1,
    output: 1,
    nodes: [inputNode, hiddenNode, outputNode],
    connections: [],
    layers: [
      createMockLayer([inputNode], [1]),
      createMockLayer([hiddenNode], [2]),
      createMockLayer([outputNode], [6]),
    ],
    _rand: createRandomSequenceGenerator([0.9]),
    _stochasticDepth: options.stochasticDepth.slice(),
    _stochasticDepthSchedule: jest.fn(() => options.scheduleResult),
  });

  return {
    activationNetwork,
    inputVector: [0.25],
  };
}

function configureFallbackTraversal(options: {
  activationNodes: MockNode[];
  inputValuesByNodeId: Map<number, number>;
  orderedOutputNodes: MockNode[];
}): void {
  mockedResolveActivationTraversalNodes.mockReturnValue(
    options.activationNodes as unknown as Network['nodes'],
  );
  mockedResolveInputValuesByNodeId.mockReturnValue(options.inputValuesByNodeId);
  mockedResolveOrderedOutputNodes.mockReturnValue(
    options.orderedOutputNodes as unknown as Network['nodes'],
  );
}

function createRandomSequenceGenerator(
  sequence: number[],
  fallbackValue = 0.5,
): () => number {
  let sequenceIndex = 0;

  return jest.fn(() => sequence[sequenceIndex++] ?? fallbackValue);
}

function summarizeConnection(connection: MockConnection): {
  dcMask: number | undefined;
  origWeight: number | undefined;
  origWeightNoise: number | undefined;
  sampledNoise: number | undefined;
  weight: number;
} {
  return {
    dcMask: connection.dcMask,
    origWeight: connection._origWeight,
    origWeightNoise: connection._origWeightNoise,
    sampledNoise: roundNumber(connection._wnLast),
    weight: roundNumber(connection.weight) ?? 0,
  };
}

function summarizeStats(network: MockActivationNetwork):
  | {
      droppedConnections: number;
      droppedHiddenNodes: number;
      skippedLayers: number[];
      totalConnections: number;
      totalHiddenNodes: number;
      weightNoise: {
        count: number;
        maxAbs: number;
        meanAbs: number;
        sumAbs: number;
      };
    }
  | undefined {
  const statsSnapshot = network._lastStats as StatsSnapshot | undefined;

  if (!statsSnapshot) {
    return undefined;
  }

  return {
    droppedConnections: statsSnapshot.droppedConnections,
    droppedHiddenNodes: statsSnapshot.droppedHiddenNodes,
    skippedLayers: [...statsSnapshot.skippedLayers],
    totalConnections: statsSnapshot.totalConnections,
    totalHiddenNodes: statsSnapshot.totalHiddenNodes,
    weightNoise: {
      count: statsSnapshot.weightNoise.count,
      maxAbs: roundNumber(statsSnapshot.weightNoise.maxAbs) ?? 0,
      meanAbs: roundNumber(statsSnapshot.weightNoise.meanAbs) ?? 0,
      sumAbs: roundNumber(statsSnapshot.weightNoise.sumAbs) ?? 0,
    },
  };
}

function roundNumber(value: number | undefined): number | undefined {
  if (value === undefined) {
    return undefined;
  }

  return Number(value.toFixed(12));
}

function assignLayerSequence(
  network: MockActivationNetwork,
  layerSequence: Array<MockLayer[] | undefined>,
): void {
  let accessIndex = 0;

  Object.defineProperty(network, 'layers', {
    configurable: true,
    get() {
      const currentLayers =
        accessIndex < layerSequence.length
          ? layerSequence[accessIndex]
          : layerSequence.at(-1);

      accessIndex++;
      return currentLayers;
    },
  });
}

function asNetwork(network: MockActivationNetwork): Network {
  return network as unknown as Network;
}
