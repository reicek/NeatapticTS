import type Network from '../../network/network';
import type { NetworkSlabProps } from './network.slab.utils.types';
import {
  _ensureFastSlabBuffers,
  _prepareFastSlabRuntime,
  _resolveFastTopoOrder,
} from './network.slab.fast-path.helpers.utils';

describe('network slab fast-path helper chapter', () => {
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
  });
});
