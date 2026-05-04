import Network from './network';
import { NetworkConstructorDimensionRequiredError } from './network.errors';

type GradientClipConfig = {
  mode: 'norm' | 'percentile' | 'layerwiseNorm' | 'layerwisePercentile';
  maxNorm?: number;
  percentile?: number;
};

type RootNetworkInternals = {
  _applyGradientClipping: (config: GradientClipConfig) => void;
  _currentGradClip?: GradientClipConfig;
  _hasPath: (from: unknown, to: unknown) => boolean;
  _rand: () => number;
  _stochasticDepthSchedule?: (step: number, current: number[]) => number[];
  _weightNoiseSchedule?: (step: number) => number;
};

type GaussianNetworkClass = {
  _gaussianRand: (rng?: () => number) => number;
};

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('network root coverage chapter', () => {
  describe('constructor', () => {
    describe('given the input size is omitted', () => {
      it('throws the dimension-required constructor error', () => {
        // Arrange
        const constructNetwork = () =>
          new Network(undefined as unknown as number, 1);

        // Act / Assert
        expect(constructNetwork).toThrow(
          NetworkConstructorDimensionRequiredError,
        );
      });
    });

    describe('given the output size is omitted', () => {
      it('throws the dimension-required constructor error', () => {
        // Arrange
        const constructNetwork = () =>
          new Network(1, undefined as unknown as number);

        // Act / Assert
        expect(constructNetwork).toThrow(
          NetworkConstructorDimensionRequiredError,
        );
      });
    });
  });

  describe('private bridge helpers', () => {
    describe('given a feed-forward path exists between two nodes', () => {
      it('reports that the path is reachable', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);
        const networkInternals = network as unknown as RootNetworkInternals;
        const sourceNode = network.nodes[0];
        const targetNode = network.nodes.at(-1);

        // Act
        const hasPath = networkInternals._hasPath(sourceNode, targetNode);

        // Assert
        expect(hasPath).toBe(true);
      });
    });

    describe('given gradient clipping is configured through the root bridge', () => {
      it('delegates the clipping request without throwing', () => {
        // Arrange
        const network = new Network(2, 1, { enforceAcyclic: true });
        const networkInternals = network as unknown as RootNetworkInternals;
        const gradientClipConfig: GradientClipConfig = {
          mode: 'norm',
          maxNorm: 0.75,
        };

        const applyGradientClipping = () =>
          networkInternals._applyGradientClipping(gradientClipConfig);

        // Act / Assert
        expect(applyGradientClipping).not.toThrow();
      });
    });

    describe('given the gaussian helper uses its default RNG source', () => {
      it('returns a finite sampled value', () => {
        // Arrange
        const networkClass = Network as unknown as GaussianNetworkClass;

        // Act
        const gaussianSample = networkClass._gaussianRand();

        // Assert
        expect(Number.isFinite(gaussianSample)).toBe(true);
      });
    });
  });

  describe('root runtime delegates', () => {
    describe('given the connection slab is rebuilt through the root API', () => {
      it('publishes a slab snapshot sized to the current connection count', () => {
        // Arrange
        const network = new Network(3, 2, { enforceAcyclic: true });

        // Act
        network.rebuildConnectionSlab();
        const slabUsed = network.getConnectionSlab()?.used;

        // Assert
        expect(slabUsed).toBe(network.connections.length);
      });
    });

    describe('given evolutionary pruning is requested through the root API', () => {
      it('delegates pruning with the default magnitude mode without throwing', () => {
        // Arrange
        const network = new Network(3, 2, { enforceAcyclic: true });
        const pruneNetwork = () => network.pruneToSparsity(0.5);

        // Act / Assert
        expect(pruneNetwork).not.toThrow();
      });
    });

    describe('given a dynamic weight-noise schedule is registered', () => {
      it('stores the schedule function on the runtime state', () => {
        // Arrange
        const network = new Network(2, 1, { enforceAcyclic: true });
        const networkInternals = network as unknown as RootNetworkInternals;
        const weightNoiseSchedule = (step: number) => step * 0.1;

        // Act
        network.setWeightNoiseSchedule(weightNoiseSchedule);

        // Assert
        expect(networkInternals._weightNoiseSchedule?.(2)).toBe(0.2);
      });
    });

    describe('given a dynamic weight-noise schedule is cleared', () => {
      it('removes the stored schedule function from runtime state', () => {
        // Arrange
        const network = new Network(2, 1, { enforceAcyclic: true });
        const networkInternals = network as unknown as RootNetworkInternals;
        network.setWeightNoiseSchedule((step: number) => step * 0.1);

        // Act
        network.clearWeightNoiseSchedule();

        // Assert
        expect(networkInternals._weightNoiseSchedule).toBeUndefined();
      });
    });

    describe('given the root RNG is replaced explicitly', () => {
      it('uses the replacement RNG for later stochastic work', () => {
        // Arrange
        const network = new Network(2, 1, { enforceAcyclic: true });
        const networkInternals = network as unknown as RootNetworkInternals;
        const replacementRandomFunction = () => 0.25;

        // Act
        network.setRandom(replacementRandomFunction);

        // Assert
        expect(networkInternals._rand()).toBe(0.25);
      });
    });

    describe('given the caller reads the public training step getter', () => {
      it('returns the initial training-step counter', () => {
        // Arrange
        const network = new Network(2, 1, { enforceAcyclic: true });

        // Act
        const trainingStep = network.trainingStep;

        // Assert
        expect(trainingStep).toBe(0);
      });
    });

    describe('given a stochastic-depth schedule is registered', () => {
      it('stores the schedule function on the runtime state', () => {
        // Arrange
        const network = new Network(2, 1, { enforceAcyclic: true });
        const networkInternals = network as unknown as RootNetworkInternals;
        const stochasticDepthSchedule = (
          _step: number,
          currentSchedule: number[],
        ) => currentSchedule.toReversed();

        // Act
        network.setStochasticDepthSchedule(stochasticDepthSchedule);

        // Assert
        expect(
          networkInternals._stochasticDepthSchedule?.(1, [0.2, 0.8]),
        ).toEqual([0.8, 0.2]);
      });
    });

    describe('given a stochastic-depth schedule is cleared', () => {
      it('removes the stored schedule function from runtime state', () => {
        // Arrange
        const network = new Network(2, 1, { enforceAcyclic: true });
        const networkInternals = network as unknown as RootNetworkInternals;
        network.setStochasticDepthSchedule(
          (_step: number, currentSchedule: number[]) => currentSchedule,
        );

        // Act
        network.clearStochasticDepthSchedule();

        // Assert
        expect(networkInternals._stochasticDepthSchedule).toBeUndefined();
      });
    });
  });

  describe('static helpers', () => {
    describe('given accumulation keeps average reduction semantics', () => {
      it('returns the original rate unchanged', () => {
        // Arrange
        const rate = 0.2;

        // Act
        const adjustedRate = Network.adjustRateForAccumulation(
          rate,
          4,
          'average',
        );

        // Assert
        expect(adjustedRate).toBe(rate);
      });
    });
  });

  describe('bulk node updates', () => {
    describe('given bias and squash values are provided together', () => {
      it('applies both values to every node in the network', () => {
        // Arrange
        const network = new Network(2, 1, { enforceAcyclic: true });
        const nextSquash = (value: number) => value;

        // Act
        network.set({ bias: 1.25, squash: nextSquash });
        const allNodesUpdated = network.nodes.every(
          (nodeEntry) =>
            nodeEntry.bias === 1.25 && nodeEntry.squash === nextSquash,
        );

        // Assert
        expect(allNodesUpdated).toBe(true);
      });
    });

    describe('given no bulk node updates are provided', () => {
      it('preserves the original node bias and squash assignments', () => {
        // Arrange
        const network = new Network(2, 1, { enforceAcyclic: true });
        const nodeSnapshotBeforeUpdate = network.nodes.map((nodeEntry) => ({
          bias: nodeEntry.bias,
          squash: nodeEntry.squash,
        }));

        // Act
        network.set({});
        const nodeSnapshotAfterUpdate = network.nodes.map((nodeEntry) => ({
          bias: nodeEntry.bias,
          squash: nodeEntry.squash,
        }));

        // Assert
        expect(nodeSnapshotAfterUpdate).toEqual(nodeSnapshotBeforeUpdate);
      });
    });
  });

  describe('public ONNX bridge', () => {
    describe('given a compact layered network is exported through the root API', () => {
      it('returns an ONNX graph with one output tensor entry', () => {
        // Arrange
        const network = Network.createMLP(2, [1], 1);

        // Act
        const onnxModel = network.toONNX();

        // Assert
        expect(onnxModel.graph.outputs.length).toBe(1);
      });
    });
  });

  describe('temporal diagnostics bridge', () => {
    describe('given a plain feed-forward network has no temporal extensions', () => {
      it('returns an empty recurrent-module and gated-block descriptor', () => {
        // Arrange
        const network = new Network(2, 1, { enforceAcyclic: true });

        // Act
        const temporalDescriptor = network.describeTemporalStructure();

        // Assert
        expect(temporalDescriptor).toEqual({
          recurrentModules: [],
          gatedBlocks: [],
        });
      });
    });
  });
});
