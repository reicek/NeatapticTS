import type Network from '../../network/network';
import type { NetworkSlabProps } from './network.slab.utils.types';
import {
  _createConnectionSlabView,
  _readSlabVersion,
} from './network.slab.view.utils';

type SlabInternals = Omit<Partial<NetworkSlabProps>, '_connWeights'> & {
  _connWeights?: Float32Array | Float64Array | null;
};

function createNetworkDouble(overrides: SlabInternals): Network {
  return overrides as unknown as Network;
}

describe('network slab chapter', () => {
  describe('view utility helpers', () => {
    describe('given slab metadata is omitted', () => {
      describe('when a slab view and version are read directly', () => {
        it('returns zero-based fallback metadata and a synthesized float64 gain view', () => {
          // Arrange
          const network = createNetworkDouble({
            _connFlags: new Uint8Array(0),
            _connFrom: new Uint32Array(0),
            _connTo: new Uint32Array(0),
            _connWeights: null,
            _useFloat32Weights: false,
          });

          // Act
          const slabSnapshot = _createConnectionSlabView(network);
          const slabVersion = _readSlabVersion(network);

          // Assert
          expect({
            capacity: slabSnapshot.capacity,
            gainLength: slabSnapshot.gain?.length ?? -1,
            gainType: slabSnapshot.gain?.constructor.name,
            plastic: slabSnapshot.plastic,
            used: slabSnapshot.used,
            version: slabSnapshot.version,
            versionReader: slabVersion,
          }).toStrictEqual({
            capacity: 0,
            gainLength: 0,
            gainType: 'Float64Array',
            plastic: null,
            used: 0,
            version: 0,
            versionReader: 0,
          });
        });
      });
    });

    describe('given the retained capacity is missing but a float32 weight slab exists', () => {
      describe('when a slab view is created directly', () => {
        it('falls back to the weight length and fills the synthesized float32 gain view for active connections', () => {
          // Arrange
          const network = createNetworkDouble({
            _connCapacity: 0,
            _connCount: 2,
            _connFlags: new Uint8Array(4),
            _connFrom: new Uint32Array(4),
            _connGain: null,
            _connPlastic: null,
            _connTo: new Uint32Array(4),
            _connWeights: new Float32Array(4),
            _useFloat32Weights: true,
          });

          // Act
          const slabSnapshot = _createConnectionSlabView(network);

          // Assert
          expect({
            capacity: slabSnapshot.capacity,
            gainType: slabSnapshot.gain?.constructor.name,
            neutralPrefix: Array.from(slabSnapshot.gain?.slice(0, 4) ?? []),
          }).toStrictEqual({
            capacity: 4,
            gainType: 'Float32Array',
            neutralPrefix: [1, 1, 0, 0],
          });
        });
      });
    });
  });
});
