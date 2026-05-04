import { config } from '../../../config';
import type {
  NetworkSlabProps,
  SlabBuildContext,
} from './network.slab.utils.types';
import {
  _applyPlasticPolicyAsync,
  _createSlabBuildContext,
  _ensureSlabCapacityAsync,
  _populateSlabConnectionsAsync,
  _resolveAsyncChunkSize,
} from './network.slab.rebuild.helpers.utils';

type RebuildConnectionFixture = {
  _flags: number;
  from: { index: number };
  gain?: number;
  to: { index: number };
  weight: number;
};

function createConnection(
  input?: Partial<RebuildConnectionFixture>,
): RebuildConnectionFixture {
  return {
    _flags: 0,
    from: { index: 0 },
    gain: 1,
    to: { index: 1 },
    weight: 0.5,
    ...input,
  };
}

function createBuildContext(input?: {
  capacity?: number;
  connections?: RebuildConnectionFixture[];
  growthFactor?: number;
  internalNet?: Partial<NetworkSlabProps>;
}): SlabBuildContext {
  const connections = input?.connections ?? [createConnection()];
  const network = {
    _connCapacity: input?.capacity ?? 0,
    connections,
    ...input?.internalNet,
  } as unknown as Parameters<typeof _createSlabBuildContext>[0];

  return _createSlabBuildContext(network, input?.growthFactor ?? 2);
}

describe('network slab rebuild helper chapter', () => {
  const originalBrowserSlabChunkTargetMs = config.browserSlabChunkTargetMs;

  afterEach(() => {
    config.browserSlabChunkTargetMs = originalBrowserSlabChunkTargetMs;
  });

  describe('_ensureSlabCapacityAsync', () => {
    describe('given async rebuild capacity already covers the active connections', () => {
      it('reuses the existing slab capacity without reallocating the core slabs', () => {
        // Arrange
        const existingWeights = new Float64Array(4);
        const buildContext = createBuildContext({
          capacity: 4,
          connections: [createConnection(), createConnection()],
          internalNet: {
            _connWeights: existingWeights,
          },
        });

        // Act
        _ensureSlabCapacityAsync(buildContext);

        // Assert
        expect({
          capacity: buildContext.capacity,
          reusedWeights:
            buildContext.internalNet._connWeights === existingWeights,
        }).toEqual({
          capacity: 4,
          reusedWeights: true,
        });
      });
    });

    describe('given async rebuild grows a float32 slab that still retains gain and plastic arrays', () => {
      it('reallocates float32 core slabs, replaces the gain slab, and clears the retained plastic slab', () => {
        // Arrange
        const previousGainArray = new Float32Array(1);
        const previousPlasticArray = new Float32Array(1);
        const buildContext = createBuildContext({
          capacity: 1,
          connections: [createConnection(), createConnection()],
          internalNet: {
            _connFlags: new Uint8Array(1),
            _connFrom: new Uint32Array(1),
            _connGain: previousGainArray,
            _connPlastic: previousPlasticArray,
            _connTo: new Uint32Array(1),
            _connWeights: new Float32Array(1),
            _useFloat32Weights: true,
          },
        });

        // Act
        _ensureSlabCapacityAsync(buildContext);

        // Assert
        expect({
          connCapacity: buildContext.internalNet._connCapacity,
          plasticCleared: buildContext.internalNet._connPlastic,
          replacedGainArray:
            buildContext.internalNet._connGain !== previousGainArray,
          weightBytes: buildContext.weightBytes,
          weightCtor: buildContext.weightCtor.name,
          weightType: buildContext.internalNet._connWeights?.constructor.name,
        }).toEqual({
          connCapacity: buildContext.capacity,
          plasticCleared: null,
          replacedGainArray: true,
          weightBytes: 4,
          weightCtor: 'Float32Array',
          weightType: 'Float32Array',
        });
      });
    });
  });

  describe('_populateSlabConnectionsAsync', () => {
    describe('given the first non-neutral gain appears in a later async chunk', () => {
      it('yields between chunks and backfills earlier neutral gain entries with one', async () => {
        // Arrange
        const buildContext = createBuildContext({
          capacity: 2,
          connections: [
            createConnection({ from: { index: 0 }, gain: 1, to: { index: 1 } }),
            createConnection({
              from: { index: 1 },
              gain: 1.5,
              to: { index: 2 },
            }),
          ],
          internalNet: {
            _connCapacity: 2,
            _connFlags: new Uint8Array(2),
            _connFrom: new Uint32Array(2),
            _connGain: null,
            _connPlastic: null,
            _connTo: new Uint32Array(2),
            _connWeights: new Float64Array(2),
          },
        });

        // Act
        const populateResult = await _populateSlabConnectionsAsync(
          buildContext,
          1,
        );

        // Assert
        expect({
          gainValues: Array.from(populateResult.gainArray ?? []).slice(0, 2),
          weightValues: Array.from(buildContext.internalNet._connWeights ?? []),
        }).toEqual({
          gainValues: [1, 1.5],
          weightValues: [0.5, 0.5],
        });
      });
    });
  });

  describe('_applyPlasticPolicyAsync', () => {
    describe('given the async pass still has plastic edges and already owns one plastic slab', () => {
      it('keeps the published plastic slab references unchanged', () => {
        // Arrange
        const retainedPlasticArray = new Float64Array(2);
        const buildContext = createBuildContext({
          capacity: 2,
          internalNet: {
            _connCapacity: 2,
            _connPlastic: retainedPlasticArray,
          },
        });
        const populateResult = {
          anyNonNeutralGain: false,
          anyPlastic: true,
          gainArray: null,
          plasticArray: retainedPlasticArray,
        };

        // Act
        _applyPlasticPolicyAsync(buildContext, populateResult);

        // Assert
        expect({
          internalPlastic: buildContext.internalNet._connPlastic,
          resultPlastic: populateResult.plasticArray,
        }).toEqual({
          internalPlastic: retainedPlasticArray,
          resultPlastic: retainedPlasticArray,
        });
      });
    });

    describe('given the async pass sees no plastic edges but still holds one retained plastic slab', () => {
      it('releases the retained async plastic slab and clears the published references', () => {
        // Arrange
        const retainedPlasticArray = new Float64Array(2);
        const buildContext = createBuildContext({
          capacity: 2,
          internalNet: {
            _connCapacity: 2,
            _connPlastic: retainedPlasticArray,
          },
        });
        const populateResult = {
          anyNonNeutralGain: false,
          anyPlastic: false,
          gainArray: null,
          plasticArray: retainedPlasticArray,
        };

        // Act
        _applyPlasticPolicyAsync(buildContext, populateResult);

        // Assert
        expect({
          internalPlastic: buildContext.internalNet._connPlastic,
          resultPlastic: populateResult.plasticArray,
        }).toEqual({
          internalPlastic: null,
          resultPlastic: null,
        });
      });
    });
  });

  describe('_resolveAsyncChunkSize', () => {
    describe('given a large async graph and a positive browser chunk budget', () => {
      it('caps the requested chunk size with the adaptive budget estimate', () => {
        // Arrange
        config.browserSlabChunkTargetMs = 1;

        // Act
        const chunkSize = _resolveAsyncChunkSize(200_001, 20_000);

        // Assert
        expect(chunkSize).toBe(15_000);
      });
    });

    describe('given a large async graph without a positive browser chunk budget', () => {
      it('falls back to the conservative global chunk cap', () => {
        // Arrange
        config.browserSlabChunkTargetMs = 0;

        // Act
        const chunkSize = _resolveAsyncChunkSize(200_001, 60_000);

        // Assert
        expect(chunkSize).toBe(50_000);
      });
    });
  });
});
