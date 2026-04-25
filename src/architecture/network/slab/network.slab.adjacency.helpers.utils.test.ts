import type Network from '../../network/network';
import { _buildAdjacency } from './network.slab.adjacency.helpers.utils';

describe('network slab adjacency helper chapter', () => {
  describe('_buildAdjacency', () => {
    describe('given required slab arrays are missing', () => {
      it('returns early without mutating published adjacency state', () => {
        // Arrange
        const network = {
          _adjDirty: true,
          _connFrom: undefined,
          _connTo: undefined,
          _outOrder: undefined,
          _outStart: undefined,
          connections: [],
          nodes: [{ index: 0 }],
        } as unknown as Network;

        // Act
        _buildAdjacency(network);

        // Assert
        expect({
          adjacencyDirty: (network as unknown as { _adjDirty?: boolean })._adjDirty,
          outOrder: (network as unknown as { _outOrder?: Uint32Array })._outOrder,
          outStart: (network as unknown as { _outStart?: Uint32Array })._outStart,
        }).toEqual({
          adjacencyDirty: true,
          outOrder: undefined,
          outStart: undefined,
        });
      });
    });
  });
});
