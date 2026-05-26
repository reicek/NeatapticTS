import { jest } from '@jest/globals';

import type Network from '../../network/network';
import { config } from '../../../config';
import { activationArrayPool } from '../../activationArrayPool/activationArrayPool';
import type { NetworkSlabProps } from './network.slab.utils.types';
import {
  _canUseFastSlab,
  _collectFastSlabOutput,
  _ensureFastSlabBuffers,
  _prepareFastSlabRuntime,
  _resolveFastTopoOrder,
} from './network.slab.fast-path.helpers.utils';

function createFastSlabEligibleNetwork(
  overrides: Partial<
    Pick<Network, 'dropout' | 'gates' | 'selfconns'> &
      Pick<
        NetworkSlabProps,
        | '_enforceAcyclic'
        | '_stochasticDepth'
        | '_topoDirty'
        | '_weightNoisePerHidden'
        | '_weightNoiseStd'
      >
  > = {},
): Network & NetworkSlabProps {
  return {
    _enforceAcyclic: true,
    _stochasticDepth: undefined,
    _topoDirty: false,
    _weightNoisePerHidden: undefined,
    _weightNoiseStd: 0,
    dropout: 0,
    gates: [],
    selfconns: [],
    ...overrides,
  } as unknown as Network & NetworkSlabProps;
}

describe('network slab fast-path helper chapter', () => {
  afterEach(() => {
    config.float32Mode = false;
    activationArrayPool.clear();
  });

  describe('_prepareFastSlabRuntime', () => {
    describe('given topology and node indices are both marked dirty', () => {
      it('recomputes topology order and reindexes the nodes', () => {
        // Arrange
        const computeTopoOrder = jest.fn();
        const reindexNodes = jest.fn();
        const network = {
          _computeTopoOrder: computeTopoOrder,
          nodes: [],
        } as unknown as Network;
        const internalNet = {
          _nodeIndexDirty: true,
          _topoDirty: true,
        } as NetworkSlabProps;

        // Act
        _prepareFastSlabRuntime(network, internalNet, reindexNodes);

        // Assert
        expect({
          computeTopoOrderCalls: computeTopoOrder.mock.calls.length,
          reindexCalls: reindexNodes.mock.calls,
        }).toEqual({
          computeTopoOrderCalls: 1,
          reindexCalls: [[network]],
        });
      });
    });
  });

  describe('_canUseFastSlab', () => {
    describe('given activation is running in training mode', () => {
      it('rejects the fast path before evaluating structural predicates', () => {
        const network = createFastSlabEligibleNetwork();

        expect(_canUseFastSlab.call(network, true)).toBe(false);
      });
    });

    describe('given the cached topology order is dirty', () => {
      it('rejects the fast path', () => {
        const network = createFastSlabEligibleNetwork({ _topoDirty: true });

        expect(_canUseFastSlab.call(network, false)).toBe(false);
      });
    });

    describe('given acyclic enforcement is disabled', () => {
      it('rejects the fast path', () => {
        const network = createFastSlabEligibleNetwork({
          _enforceAcyclic: false,
        });

        expect(_canUseFastSlab.call(network, false)).toBe(false);
      });
    });

    describe('given dropout is enabled', () => {
      it('rejects the fast path immediately', () => {
        const network = createFastSlabEligibleNetwork({ dropout: 0.1 });

        expect(_canUseFastSlab.call(network, false)).toBe(false);
      });
    });

    describe('given gated connections are present', () => {
      it('rejects the fast path', () => {
        const network = createFastSlabEligibleNetwork({
          gates: [{}] as unknown as Network['gates'],
        });

        expect(_canUseFastSlab.call(network, false)).toBe(false);
      });
    });

    describe('given self-connections are present', () => {
      it('rejects the fast path', () => {
        const network = createFastSlabEligibleNetwork({
          selfconns: [{}] as unknown as Network['selfconns'],
        });

        expect(_canUseFastSlab.call(network, false)).toBe(false);
      });
    });

    describe('given global weight noise is enabled', () => {
      it('rejects the fast path', () => {
        const network = createFastSlabEligibleNetwork({ _weightNoiseStd: 0.2 });

        expect(_canUseFastSlab.call(network, false)).toBe(false);
      });
    });

    describe('given per-hidden weight noise is configured', () => {
      it('rejects the fast path', () => {
        const network = createFastSlabEligibleNetwork({
          _weightNoisePerHidden: [0.3],
        });

        expect(_canUseFastSlab.call(network, false)).toBe(false);
      });
    });

    describe('given optional regularization arrays are absent', () => {
      it('keeps the fast path eligible', () => {
        const network = createFastSlabEligibleNetwork();

        expect(_canUseFastSlab.call(network, false)).toBe(true);
      });
    });
  });

  describe('_resolveFastTopoOrder', () => {
    describe('given no cached topological order is available', () => {
      it('falls back to the network node order', () => {
        // Arrange
        const fallbackNodeOrder = [{ index: 1 }, { index: 2 }];
        const network = {
          nodes: fallbackNodeOrder,
        } as unknown as Network;
        const internalNet = {
          _topoOrder: undefined,
        } as NetworkSlabProps;

        // Act
        const topoOrder = _resolveFastTopoOrder(network, internalNet);

        // Assert
        expect(topoOrder).toBe(fallbackNodeOrder);
      });
    });
  });

  describe('_ensureFastSlabBuffers', () => {
    describe('given float32 activation precision is requested but cached buffers use float64', () => {
      it('replaces both reusable buffers with float32 arrays', () => {
        // Arrange
        const internalNet = {
          _activationPrecision: 'f32',
          _fastA: new Float64Array(3),
          _fastS: new Float64Array(3),
        } as NetworkSlabProps;

        // Act
        _ensureFastSlabBuffers(internalNet, 3);

        // Assert
        expect({
          activationType: internalNet._fastA?.constructor.name,
          stateType: internalNet._fastS?.constructor.name,
        }).toEqual({
          activationType: 'Float32Array',
          stateType: 'Float32Array',
        });
      });
    });

    describe('given float64 activation precision is requested but cached buffers use float32', () => {
      it('replaces both reusable buffers with float64 arrays', () => {
        // Arrange
        const internalNet = {
          _activationPrecision: 'f64',
          _fastA: new Float32Array(3),
          _fastS: new Float32Array(3),
        } as NetworkSlabProps;

        // Act
        _ensureFastSlabBuffers(internalNet, 3);

        // Assert
        expect({
          activationType: internalNet._fastA?.constructor.name,
          stateType: internalNet._fastS?.constructor.name,
        }).toEqual({
          activationType: 'Float64Array',
          stateType: 'Float64Array',
        });
      });
    });

    describe('given shared precision config requests float32 while the raw alias is absent', () => {
      it('uses the shared precision config for reusable fast slab buffers', () => {
        // Arrange
        const internalNet = {
          _precisionConfig: {
            activationPrecision: 'f32',
          },
          _fastA: new Float64Array(3),
          _fastS: new Float64Array(3),
        } as NetworkSlabProps;

        // Act
        _ensureFastSlabBuffers(internalNet, 3);

        // Assert
        expect({
          activationType: internalNet._fastA?.constructor.name,
          stateType: internalNet._fastS?.constructor.name,
        }).toEqual({
          activationType: 'Float32Array',
          stateType: 'Float32Array',
        });
      });
    });
  });

  describe('_collectFastSlabOutput', () => {
    describe('given global float32 mode is enabled but the network requests f64 activation precision', () => {
      it('preserves the explicit float64 output value', () => {
        // Arrange
        const previousFloat32Mode = config.float32Mode;
        const expectedOutputValue = Math.PI / 13;
        config.float32Mode = true;

        try {
          const network = {
            output: 1,
            _activationPrecision: 'f64',
          } as unknown as Network;
          const activationBuffer = new Float64Array([expectedOutputValue]);

          // Act
          const output = _collectFastSlabOutput(network, activationBuffer, 1);

          // Assert
          expect(output[0]).toBe(expectedOutputValue);
        } finally {
          config.float32Mode = previousFloat32Mode;
        }
      });
    });

    describe('given the legacy raw alias requests float32 against a shared f64 config', () => {
      it('preserves the raw float32 compatibility override for detached slab output', () => {
        // Arrange
        const expectedOutputValue = Math.PI / 13;
        const network = {
          output: 1,
          _precisionConfig: {
            activationPrecision: 'f64',
          },
          _activationPrecision: 'f32',
        } as unknown as Network;
        const activationBuffer = new Float64Array([expectedOutputValue]);

        // Act
        const output = _collectFastSlabOutput(network, activationBuffer, 1);

        // Assert
        expect(output[0]).toBe(Math.fround(expectedOutputValue));
      });
    });

    describe('given shared precision config requests float32 while the raw alias is absent', () => {
      it('quantizes the detached slab output through the shared precision config', () => {
        // Arrange
        const expectedOutputValue = Math.PI / 13;
        const network = {
          output: 1,
          _precisionConfig: {
            activationPrecision: 'f32',
          },
        } as unknown as Network;
        const activationBuffer = new Float64Array([expectedOutputValue]);

        // Act
        const output = _collectFastSlabOutput(network, activationBuffer, 1);

        // Assert
        expect(output[0]).toBe(Math.fround(expectedOutputValue));
      });
    });
  });
});
