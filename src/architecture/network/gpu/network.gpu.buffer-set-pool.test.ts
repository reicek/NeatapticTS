/**
 * Red-phase failing tests for GPUBufferSetPool.
 *
 * These tests define the expected behavior of the GPUBufferSetPool class
 * before any implementation exists. They fail because the module
 * `network.gpu.buffer-set-pool` does not exist yet — a clean red failure,
 * not a syntax or fixture error.
 *
 * The pool is designed to recycle GPU buffers across evaluateWeightVariants
 * calls so that variant evaluation does not pay per-iteration allocation
 * and destruction overhead.
 *
 * GPU real-device gate justification: GPUBufferSetPool is a pure data
 * structure for managing buffer set lifecycle (acquire/release/destroy). It
 * makes NO GPU API calls directly — no navigator.gpu, no requestGPUDevice, no
 * GPUBuffer creation. It only tracks metadata (keys, sizes, inUse flags).
 * Therefore mock-only Jest validation IS sufficient for this class; there is
 * no GPU code to test on a real device.
 */

import { GPUBufferSetPool } from './network.gpu.buffer-set-pool';
import { createMockGPUDevice } from './__mocks__/gpu.mock';
import * as capability from './network.gpu.capability';
import Network from '../network';
import {
  resolveBufferPoolMaxPooledBytes,
  MIN_BUFFER_POOL_BYTES,
  DEFAULT_BUFFER_POOL_AVG_DEGREE,
  DEFAULT_BUFFER_POOL_BUFFER_COUNT,
  DEFAULT_BUFFER_POOL_FLOAT32_BYTES,
  DEFAULT_BUFFER_POOL_SAFETY_FACTOR,
} from '../../../acceleration/acceleration.constants';

function createEligibleMLP(): Network {
  const network = Network.createMLP(2, [3], 1);
  network.activate([0.1, 0.2]);
  return network;
}

describe('network.gpu.buffer-set-pool', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('GPUBufferSetPool construction', () => {
    it('creates a pool with default options when no config is supplied', () => {
      const pool = new GPUBufferSetPool();

      expect(pool).toBeDefined();
    });

    it('accepts config-overridable maxPooledBytes option', () => {
      const pool = new GPUBufferSetPool({ maxPooledBytes: 8 * 1024 * 1024 });

      expect(pool.maxPooledBytes).toBe(8 * 1024 * 1024);
    });

    it('does not hardcode the default cap to the legacy 16 MB constant', () => {
      const pool = new GPUBufferSetPool();

      expect(pool.maxPooledBytes).not.toBe(16 * 1024 * 1024);
    });
  });

  describe('dynamic buffer pool cap resolution', () => {
    it('returns MIN_BUFFER_POOL_BYTES for very small workloads', () => {
      const cap = resolveBufferPoolMaxPooledBytes({
        nodeCount: 10,
        variantCount: 2,
      });

      expect(cap).toBe(MIN_BUFFER_POOL_BYTES);
    });

    it('scales the cap upward as nodeCount increases', () => {
      const small = resolveBufferPoolMaxPooledBytes({
        nodeCount: 100,
        variantCount: 2,
      });
      const large = resolveBufferPoolMaxPooledBytes({
        nodeCount: 8_000,
        variantCount: 16,
      });

      expect(large).toBeGreaterThan(small);
    });

    it('scales the cap linearly with avgDegree', () => {
      const nodeCount = 10_000;
      const base = resolveBufferPoolMaxPooledBytes(
        { nodeCount, variantCount: 2 },
        { avgDegree: DEFAULT_BUFFER_POOL_AVG_DEGREE },
      );
      const doubled = resolveBufferPoolMaxPooledBytes(
        { nodeCount, variantCount: 2 },
        { avgDegree: DEFAULT_BUFFER_POOL_AVG_DEGREE * 2 },
      );

      expect(doubled).toBe(base * 2);
    });

    it('uses a caller-supplied minBytes floor', () => {
      const customFloor = 1_048_576;

      const cap = resolveBufferPoolMaxPooledBytes(
        { nodeCount: 10, variantCount: 2 },
        { minBytes: customFloor },
      );

      expect(cap).toBe(customFloor);
    });

    it('respects a caller-supplied hard cap', () => {
      const cap = resolveBufferPoolMaxPooledBytes(
        { nodeCount: 8_000, variantCount: 16 },
        { maxPooledBytes: 4 * 1024 * 1024 },
      );

      expect(cap).toBe(4 * 1024 * 1024);
    });

    it('caps at one quarter of the device maxBufferSize', () => {
      const deviceMaxBufferSize = 64 * 1024 * 1024;
      const cap = resolveBufferPoolMaxPooledBytes(
        { nodeCount: 100_000, variantCount: 16 },
        { maxBufferSize: deviceMaxBufferSize },
      );

      expect(cap).toBe(deviceMaxBufferSize / 4);
    });

    it('matches the Config Defaults Catalog formula', () => {
      const nodeCount = 2_000;
      const expected = Math.max(
        MIN_BUFFER_POOL_BYTES,
        nodeCount *
          DEFAULT_BUFFER_POOL_AVG_DEGREE *
          DEFAULT_BUFFER_POOL_FLOAT32_BYTES *
          DEFAULT_BUFFER_POOL_BUFFER_COUNT *
          DEFAULT_BUFFER_POOL_SAFETY_FACTOR,
      );

      const cap = resolveBufferPoolMaxPooledBytes(
        { nodeCount, variantCount: 8 },
        { avgDegree: DEFAULT_BUFFER_POOL_AVG_DEGREE },
      );

      expect(cap).toBe(expected);
    });

    it('floors the cap to minBytes when nodeCount is omitted', () => {
      // Arrange — workload omits nodeCount, so the heuristic estimate collapses
      // to zero and must be clamped to the configured floor.
      const cap = resolveBufferPoolMaxPooledBytes({ variantCount: 2 });

      expect(cap).toBe(MIN_BUFFER_POOL_BYTES);
    });
  });

  describe('acquire — buffer allocation', () => {
    it('returns a GPUBufferSet with correct capacity for the given topology', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);
      const pool = new GPUBufferSetPool();
      const topologyKey = 'test-topo-2-3-1';
      const variantCount = 4;

      const bufferSet = pool.acquire(
        device,
        topologyKey,
        network,
        variantCount,
      );

      expect(bufferSet.nodeCount).toBe(network.nodes.length);
    });
  });

  describe('release — buffer deallocation and recycling', () => {
    it('returns buffers to the pool so a subsequent acquire reuses them', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);
      const pool = new GPUBufferSetPool();
      const topologyKey = 'test-topo-recycle';
      const variantCount = 4;

      const first = pool.acquire(device, topologyKey, network, variantCount);
      pool.release(topologyKey);
      const second = pool.acquire(device, topologyKey, network, variantCount);

      expect(second).toBe(first);
    });
  });

  describe('pool keying by topology and variant count', () => {
    it('returns distinct buffer sets for different topology keys', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);
      const pool = new GPUBufferSetPool();
      const variantCount = 4;

      const setA = pool.acquire(device, 'topo-A', network, variantCount);
      const setB = pool.acquire(device, 'topo-B', network, variantCount);

      expect(setA).not.toBe(setB);
    });

    it('returns distinct buffer sets for different variant counts on the same topology', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);
      const pool = new GPUBufferSetPool();

      const setA = pool.acquire(device, 'topo-same', network, 4);
      const setB = pool.acquire(device, 'topo-same', network, 8);

      expect(setA).not.toBe(setB);
    });
  });

  describe('reuse across evaluateWeightVariants calls', () => {
    it('does not reallocate buffers for the same topology and variant count across calls', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);
      const pool = new GPUBufferSetPool();
      const topologyKey = 'eval-variants-topo';
      const variantCount = 4;

      const first = pool.acquire(device, topologyKey, network, variantCount);
      pool.release(topologyKey);
      // Simulate a second evaluateWeightVariants call
      const second = pool.acquire(device, topologyKey, network, variantCount);
      pool.release(topologyKey);

      expect(first).toBe(second);
    });
  });

  describe('destroy — pool teardown', () => {
    it('destroys all pooled buffers when destroy is called', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);
      const pool = new GPUBufferSetPool();
      const topologyKey = 'destroy-topo';
      const variantCount = 4;

      pool.acquire(device, topologyKey, network, variantCount);
      pool.destroy();

      expect(pool.size).toBe(0);
    });
  });

  describe('maxPooledBytes cap eviction', () => {
    it('evicts free entries when a new acquisition would exceed the byte cap', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);
      // Small cap so two buffer sets exceed it (~332 bytes each for 2-3-1 MLP)
      const pool = new GPUBufferSetPool({ maxPooledBytes: 500 });

      // Acquire setA, then release it (marks it as free in the pool)
      pool.acquire(device, 'topo-A', network, 4);
      pool.release('topo-A');

      // Acquire setB with a different key — should evict the free setA entry
      pool.acquire(device, 'topo-B', network, 4);

      // Pool contains only setB; setA was evicted to stay under the byte cap
      expect(pool.size).toBe(1);
    });

    it('does not evict in-use entries when no free entries exist', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);
      // Small cap so subsequent acquisitions trigger the eviction path
      const pool = new GPUBufferSetPool({ maxPooledBytes: 500 });

      // Acquire two sets with different keys (both remain in-use)
      pool.acquire(device, 'topo-A', network, 4);
      pool.acquire(device, 'topo-B', network, 4);

      // Acquire a third set — eviction runs but finds no free entries to evict
      pool.acquire(device, 'topo-C', network, 4);

      // All three entries remain because none were free to evict
      expect(pool.size).toBe(3);
    });

    it('stops evicting once enough free entries are reclaimed to fit under the cap', () => {
      const device = createMockGPUDevice();
      const network = createEligibleMLP();
      jest.spyOn(capability, 'canUseGPU').mockReturnValue(true);
      // Cap large enough for two entries (~332 bytes each = 664) but not three
      // (664 + 332 = 996 > 700)
      const pool = new GPUBufferSetPool({ maxPooledBytes: 700 });

      // Acquire two different topologies and release both (both become free)
      pool.acquire(device, 'topo-A', network, 4);
      pool.release('topo-A');
      pool.acquire(device, 'topo-B', network, 4);
      pool.release('topo-B');

      // Pool now has 2 free entries totaling ~664 bytes
      expect(pool.size).toBe(2);

      // Acquire topo-C: 664 + 332 = 996 > 700 triggers eviction.
      // After evicting topo-A (664 - 332 = 332), 332 + 332 = 664 <= 700 → break.
      // Only topo-A is evicted; topo-B remains free in the pool.
      pool.acquire(device, 'topo-C', network, 4);

      // Pool retains topo-B (free) and topo-C (in use) = 2 entries.
      // If the break did not fire, both free entries would be evicted,
      // leaving only topo-C (size = 1).
      expect(pool.size).toBe(2);
    });
  });
});
