/**
 * Red tests for the generic acceleration config/status surface.
 *
 * These tests define the expected public shape of the acceleration layer before
 * the implementation exists. They will fail with TS2307 "Cannot find module" or
 * missing-export errors until `src/acceleration/` is implemented.
 */

import {
  resolveAccelerationConfig,
  DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
  DEFAULT_ACCELERATION_WORKER_MIN_CORES,
  DEFAULT_ACCELERATION_MAX_WORKERS,
  DEFAULT_ACCELERATION_GPU_BATCH_PARALLEL_THRESHOLD,
  DEFAULT_ACCELERATION_PARALLEL_VARIANT_COUNT,
  type AccelerationConfig,
  type AccelerationStatus,
  type AccelerationMode,
  type BackendMode,
} from './acceleration.config';

import {
  resolveBufferPoolMaxPooledBytes,
  MIN_BUFFER_POOL_BYTES,
} from './acceleration.constants';

import type {
  BufferPoolMaxPooledBytesOptions,
  BufferPoolWorkload,
} from './acceleration.types';

describe('acceleration.config', () => {
  describe('DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD', () => {
    it('equals 1024', () => {
      expect(DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD).toBe(1024);
    });
  });

  describe('DEFAULT_ACCELERATION_GPU_BATCH_PARALLEL_THRESHOLD', () => {
    it('equals 8', () => {
      expect(DEFAULT_ACCELERATION_GPU_BATCH_PARALLEL_THRESHOLD).toBe(8);
    });
  });

  describe('DEFAULT_ACCELERATION_WORKER_MIN_CORES', () => {
    it('equals 2', () => {
      expect(DEFAULT_ACCELERATION_WORKER_MIN_CORES).toBe(2);
    });
  });

  describe('DEFAULT_ACCELERATION_MAX_WORKERS', () => {
    it('equals 4', () => {
      expect(DEFAULT_ACCELERATION_MAX_WORKERS).toBe(4);
    });
  });

  describe('DEFAULT_ACCELERATION_PARALLEL_VARIANT_COUNT', () => {
    it('equals 16', () => {
      expect(DEFAULT_ACCELERATION_PARALLEL_VARIANT_COUNT).toBe(16);
    });
  });

  describe('resolveAccelerationConfig', () => {
    it('fills in defaults for an empty partial config', () => {
      const resolved = resolveAccelerationConfig({});

      expect(resolved).toMatchObject({
        gpuNodeThreshold: 1024,
        gpuBatchParallelThreshold: 8,
        workerMinCores: 2,
        maxWorkers: 4,
        disableGPU: false,
        disableWorkers: false,
        backend: 'auto',
      });
    });

    it('fills in defaults when called with no arguments', () => {
      const resolved = resolveAccelerationConfig();

      expect(resolved).toEqual({
        gpuNodeThreshold: 1024,
        gpuBatchParallelThreshold: 8,
        workerMinCores: 2,
        maxWorkers: 4,
        disableGPU: false,
        disableWorkers: false,
        backend: 'auto',
        parallelVariantCount: 16,
      });
    });

    it('preserves caller overrides', () => {
      const resolved = resolveAccelerationConfig({
        gpuNodeThreshold: 512,
        workerMinCores: 4,
        maxWorkers: 2,
        disableGPU: true,
        backend: 'cpu',
      });

      expect(resolved).toMatchObject({
        gpuNodeThreshold: 512,
        workerMinCores: 4,
        maxWorkers: 2,
        disableGPU: true,
        backend: 'cpu',
      });
    });

    it('fills in the default parallelVariantCount of 16 when omitted', () => {
      const resolved = resolveAccelerationConfig({});

      expect(resolved.parallelVariantCount).toBe(
        DEFAULT_ACCELERATION_PARALLEL_VARIANT_COUNT,
      );
    });

    it('resolves parallelVariantCount to 16 for an empty partial config', () => {
      const resolved = resolveAccelerationConfig({});

      expect(resolved.parallelVariantCount).toBe(16);
    });

    it('preserves an explicit parallelVariantCount override', () => {
      const resolved = resolveAccelerationConfig({ parallelVariantCount: 4 });

      expect(resolved.parallelVariantCount).toBe(4);
    });

    it('forwards an explicit stageVariantCounts override', () => {
      const resolved = resolveAccelerationConfig({
        stageVariantCounts: { baby: 256 },
      });

      expect(resolved).toMatchObject({
        stageVariantCounts: { baby: 256 },
      });
    });

    it('allows a backend mode of auto', () => {
      const config: AccelerationConfig = resolveAccelerationConfig({
        backend: 'auto',
      });

      expect(config.backend).toBe('auto');
    });

    it('allows a backend mode of gpu', () => {
      const config: AccelerationConfig = resolveAccelerationConfig({
        backend: 'gpu',
      });

      expect(config.backend).toBe('gpu');
    });

    it('allows a backend mode of cpu', () => {
      const config: AccelerationConfig = resolveAccelerationConfig({
        backend: 'cpu',
      });

      expect(config.backend).toBe('cpu');
    });
  });

  describe('AccelerationStatus type', () => {
    it('accepts a minimal CPU-only status object', () => {
      const status: AccelerationStatus = {
        mode: 'cpu',
        gpu: {
          available: false,
          reason: 'GPU disabled by config',
        },
        worker: {
          available: false,
          count: 0,
          reason: 'Workers disabled by config',
        },
      };

      expect(status.mode).toBe('cpu');
    });
  });

  describe('AccelerationMode type', () => {
    it('accepts the three legal mode literals', () => {
      const modes: AccelerationMode[] = ['cpu', 'gpu', 'worker'];

      expect(modes).toEqual(['cpu', 'gpu', 'worker']);
    });
  });

  describe('BackendMode type', () => {
    it('accepts the three legal backend literals', () => {
      const backends: BackendMode[] = ['auto', 'gpu', 'cpu'];

      expect(backends).toEqual(['auto', 'gpu', 'cpu']);
    });
  });

  describe('resolveBufferPoolMaxPooledBytes', () => {
    it('returns the explicit maxPooledBytes when provided', () => {
      const cap = resolveBufferPoolMaxPooledBytes(
        { nodeCount: 1_024, variantCount: 1 },
        { maxPooledBytes: 512 * 1024 },
      );

      expect(cap).toBe(512 * 1024);
    });

    it('computes the default heuristic cap from nodeCount', () => {
      const nodeCount = 1_024;
      const cap = resolveBufferPoolMaxPooledBytes({
        nodeCount,
        variantCount: 1,
      });

      expect(cap).toBe(MIN_BUFFER_POOL_BYTES);
    });

    it('floors the heuristic to the minimum byte cap', () => {
      const cap = resolveBufferPoolMaxPooledBytes({
        nodeCount: 0,
        variantCount: 1,
      });

      expect(cap).toBe(MIN_BUFFER_POOL_BYTES);
    });

    it('clamps the cap to one quarter of maxBufferSize', () => {
      const maxBufferSize = 1_048_576;
      const cap = resolveBufferPoolMaxPooledBytes(
        { nodeCount: 10_000, variantCount: 1 },
        { maxBufferSize },
      );

      expect(cap).toBe(Math.floor(maxBufferSize / 4));
    });

    it('honors custom heuristic knobs', () => {
      const cap = resolveBufferPoolMaxPooledBytes(
        { nodeCount: 100, variantCount: 1 },
        {
          avgDegree: 5,
          float32Bytes: 8,
          bufferCount: 2,
          safetyFactor: 2,
          minBytes: 1_000,
        },
      );

      expect(cap).toBe(100 * 5 * 8 * 2 * 2);
    });

    it('lets explicit maxPooledBytes override the maxBufferSize clamp', () => {
      const cap = resolveBufferPoolMaxPooledBytes(
        { nodeCount: 10_000, variantCount: 1 },
        { maxPooledBytes: 2_000_000, maxBufferSize: 1_048_576 },
      );

      expect(cap).toBe(2_000_000);
    });

    it('accepts a workload with all optional fields omitted', () => {
      const cap = resolveBufferPoolMaxPooledBytes({} as BufferPoolWorkload);

      expect(cap).toBe(MIN_BUFFER_POOL_BYTES);
    });

    it('accepts options with all heuristic overrides typed', () => {
      const options: BufferPoolMaxPooledBytesOptions = {
        maxPooledBytes: 128 * 1024,
      };
      const cap = resolveBufferPoolMaxPooledBytes(
        { nodeCount: 1_024, variantCount: 1 },
        options,
      );

      expect(cap).toBe(128 * 1024);
    });
  });
});
