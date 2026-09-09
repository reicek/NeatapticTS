import { config, type PrecisionConfig } from '../../../config';
import { activationArrayPool } from '../../activationArrayPool/activationArrayPool';
import Node from '../../node/node';
import * as nodePoolModule from '../../nodePool/nodePool';
import { NetworkBootstrapTopologyIntentConflictError } from './network.bootstrap.errors';
import {
  bootstrapNetwork,
  resolveAcyclicEnforcement,
  resolveTopologyIntent,
  validateTopologyIntentConfiguration,
} from './network.bootstrap.utils';

type BootstrapNetworkFixture = {
  _activationPrecision?: 'f32' | 'f64';
  _precisionConfig?: PrecisionConfig;
  _enforceAcyclic?: boolean;
  _rand: () => number;
  _returnTypedActivations?: boolean;
  _reuseActivationArrays?: boolean;
  _reuseSequenceBuffers?: boolean;
  _topologyIntent?: 'feed-forward' | 'unconstrained';
  addNodeBetween: jest.Mock;
  connect: jest.Mock;
  connectBatch: jest.Mock;
  connections: unknown[];
  dropout: number;
  gates: unknown[];
  input: number;
  nodes: Node[];
  output: number;
  refreshExplicitIORoles: jest.Mock;
  selfconns: unknown[];
  setSeed: jest.Mock;
};

function createBootstrapNetwork(): BootstrapNetworkFixture {
  const network = {
    _rand: () => 0.5,
    addNodeBetween: jest.fn(() => {
      network.nodes.push(new Node('hidden', undefined, network._rand));
    }),
    connect: jest.fn(),
    connectBatch: jest.fn(),
    connections: [],
    dropout: 0,
    gates: [],
    input: 0,
    nodes: [],
    output: 0,
    refreshExplicitIORoles: jest.fn(),
    selfconns: [],
    setSeed: jest.fn(),
  } as BootstrapNetworkFixture;

  return network;
}

describe('network bootstrap utility chapter', () => {
  const originalEnableNodePooling = config.enableNodePooling;
  const originalFloat32Mode = config.float32Mode;
  const originalPoolMaxPerBucket = config.poolMaxPerBucket;
  const originalPoolPrewarmCount = config.poolPrewarmCount;

  afterEach(() => {
    config.enableNodePooling = originalEnableNodePooling;
    config.float32Mode = originalFloat32Mode;
    config.poolMaxPerBucket = originalPoolMaxPerBucket;
    config.poolPrewarmCount = originalPoolPrewarmCount;
    jest.restoreAllMocks();
  });

  describe('validateTopologyIntentConfiguration', () => {
    describe('given feed-forward intent explicitly disables acyclic enforcement', () => {
      it('throws the topology-intent conflict error', () => {
        // Arrange
        const validateConflict = () => {
          validateTopologyIntentConfiguration({
            enforceAcyclic: false,
            topologyIntent: 'feed-forward',
          });
        };

        // Act
        const conflictAction = validateConflict;

        // Assert
        expect(conflictAction).toThrow(
          NetworkBootstrapTopologyIntentConflictError,
        );
      });
    });

    describe('given unconstrained intent explicitly enables acyclic enforcement', () => {
      it('throws the topology-intent conflict error', () => {
        // Arrange
        const validateConflict = () => {
          validateTopologyIntentConfiguration({
            enforceAcyclic: true,
            topologyIntent: 'unconstrained',
          });
        };

        // Act
        const conflictAction = validateConflict;

        // Assert
        expect(conflictAction).toThrow(
          NetworkBootstrapTopologyIntentConflictError,
        );
      });
    });

    describe('given unconstrained intent keeps legacy acyclic enforcement disabled', () => {
      it('does not throw the topology-intent conflict error', () => {
        // Arrange
        const validateConfiguration = () => {
          validateTopologyIntentConfiguration({
            enforceAcyclic: false,
            topologyIntent: 'unconstrained',
          });
        };

        // Act
        const validationAction = validateConfiguration;

        // Assert
        expect(validationAction).not.toThrow();
      });
    });
  });

  describe('resolveTopologyIntent', () => {
    describe('given constructor options explicitly provide a topology intent', () => {
      it('returns the explicit semantic topology contract', () => {
        // Arrange
        const options = {
          enforceAcyclic: false,
          topologyIntent: 'feed-forward',
        } as const;

        // Act
        const topologyIntent = resolveTopologyIntent(options);

        // Assert
        expect(topologyIntent).toBe('feed-forward');
      });
    });

    describe('given constructor options only provide the legacy acyclic flag', () => {
      it('falls back to the feed-forward topology intent', () => {
        // Arrange
        const options = {
          enforceAcyclic: true,
        };

        // Act
        const topologyIntent = resolveTopologyIntent(options);

        // Assert
        expect(topologyIntent).toBe('feed-forward');
      });
    });

    describe('given constructor options provide neither semantic intent nor legacy acyclic enablement', () => {
      it('falls back to the unconstrained topology intent', () => {
        // Arrange
        const options = {
          enforceAcyclic: false,
        };

        // Act
        const topologyIntent = resolveTopologyIntent(options);

        // Assert
        expect(topologyIntent).toBe('unconstrained');
      });
    });
  });

  describe('resolveAcyclicEnforcement', () => {
    describe('given constructor options explicitly provide the legacy acyclic flag', () => {
      it('returns the explicit legacy enforcement value', () => {
        // Arrange
        const options = {
          enforceAcyclic: false,
        };

        // Act
        const enforceAcyclic = resolveAcyclicEnforcement(
          options,
          'feed-forward',
        );

        // Assert
        expect(enforceAcyclic).toBe(false);
      });
    });

    describe('given constructor options omit the legacy acyclic flag', () => {
      it('falls back to the semantic feed-forward topology intent', () => {
        // Arrange
        const options = undefined;

        // Act
        const enforceAcyclic = resolveAcyclicEnforcement(
          options,
          'feed-forward',
        );

        // Assert
        expect(enforceAcyclic).toBe(true);
      });
    });
  });

  describe('bootstrapNetwork', () => {
    describe('given constructor options explicitly configure activation precision and pool warmup', () => {
      it('publishes the explicit runtime flags, warms the pool with configured values, seeds deterministically, and grows the requested hidden width', () => {
        // Arrange
        config.enableNodePooling = false;
        config.float32Mode = false;
        config.poolMaxPerBucket = 7;
        config.poolPrewarmCount = 3;
        const setMaxPerBucketSpy = jest
          .spyOn(activationArrayPool, 'setMaxPerBucket')
          .mockImplementation(() => undefined);
        const prewarmSpy = jest
          .spyOn(activationArrayPool, 'prewarm')
          .mockImplementation(() => undefined);
        const network = createBootstrapNetwork();

        // Act
        bootstrapNetwork(network as never, {
          enforceAcyclic: true,
          input: 1,
          options: {
            activationPrecision: 'f64',
            minHidden: 2,
            reuseActivationArrays: true,
            reuseSequenceBuffers: true,
            returnTypedActivations: true,
            seed: 42,
          },
          output: 1,
          topologyIntent: 'feed-forward',
        });

        // Assert
        expect({
          activationPrecision: network._activationPrecision,
          addNodeBetweenCalls: network.addNodeBetween.mock.calls.length,
          batchRequestCount: network.connectBatch.mock.calls[0]?.[0]?.length,
          connectCalls: network.connect.mock.calls.length,
          nodeCount: network.nodes.length,
          poolMaxCall: setMaxPerBucketSpy.mock.calls[0]?.[0],
          prewarmCall: prewarmSpy.mock.calls[0],
          reuseActivationArrays: network._reuseActivationArrays,
          reuseSequenceBuffers: network._reuseSequenceBuffers,
          seededWith: network.setSeed.mock.calls[0]?.[0],
          typedActivations: network._returnTypedActivations,
        }).toEqual({
          activationPrecision: 'f64',
          addNodeBetweenCalls: 2,
          batchRequestCount: 1,
          connectCalls: 0,
          nodeCount: 4,
          poolMaxCall: 7,
          prewarmCall: [1, 3],
          reuseActivationArrays: true,
          reuseSequenceBuffers: true,
          seededWith: 42,
          typedActivations: true,
        });
      });
    });

    describe('given float32 mode is enabled without explicit activation precision or pool overrides', () => {
      it('falls back to float32 activation precision and the default prewarm count', () => {
        // Arrange
        config.enableNodePooling = false;
        config.float32Mode = true;
        config.poolMaxPerBucket = undefined;
        config.poolPrewarmCount = undefined;
        const setMaxPerBucketSpy = jest
          .spyOn(activationArrayPool, 'setMaxPerBucket')
          .mockImplementation(() => undefined);
        const prewarmSpy = jest
          .spyOn(activationArrayPool, 'prewarm')
          .mockImplementation(() => undefined);
        const network = createBootstrapNetwork();

        // Act
        bootstrapNetwork(network as never, {
          enforceAcyclic: false,
          input: 1,
          options: undefined,
          output: 1,
          topologyIntent: 'unconstrained',
        });

        // Assert
        expect({
          activationPrecision: network._activationPrecision,
          poolMaxCalls: setMaxPerBucketSpy.mock.calls.length,
          prewarmCall: prewarmSpy.mock.calls[0],
        }).toEqual({
          activationPrecision: 'f32',
          poolMaxCalls: 0,
          prewarmCall: [1, 2],
        });
      });
    });

    describe('given node pooling is enabled without explicit precision and float32 mode stays disabled', () => {
      it('acquires pooled nodes and leaves activation precision unchanged', () => {
        // Arrange
        config.enableNodePooling = true;
        config.float32Mode = false;
        const acquireNodeSpy = jest
          .spyOn(nodePoolModule, 'acquireNode')
          .mockImplementation(
            (request) =>
              new Node(
                request?.type ?? 'hidden',
                undefined,
                request?.rng ?? (() => 0.5),
              ),
          );
        const network = createBootstrapNetwork();

        // Act
        bootstrapNetwork(network as never, {
          enforceAcyclic: false,
          input: 1,
          options: undefined,
          output: 1,
          topologyIntent: 'unconstrained',
        });

        // Assert
        expect({
          activationPrecision: network._activationPrecision,
          precisionConfigActivationPrecision:
            network._precisionConfig?.activationPrecision,
          acquiredNodeTypes: acquireNodeSpy.mock.calls.map(
            ([request]) => request?.type ?? 'missing',
          ),
        }).toEqual({
          activationPrecision: undefined,
          precisionConfigActivationPrecision: 'f64',
          acquiredNodeTypes: ['input', 'output'],
        });
      });
    });
  });
});
