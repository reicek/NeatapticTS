/**
 * Red tests for the generic acceleration type surface.
 *
 * These tests define the expected type shapes exported by the acceleration layer
 * before the implementation exists. They will fail with TS2307 "Cannot find
 * module" or missing-export errors until `src/acceleration/acceleration.types.ts`
 * is implemented.
 */

import type {
  AccelerationConfig,
  AccelerationStatus,
  AccelerationMode,
  BackendMode,
  GPUStatus,
  WorkerStatus,
  AccelerationCapabilities,
} from './acceleration.types';

describe('acceleration.types', () => {
  describe('AccelerationConfig', () => {
    it('accepts an empty object as a valid partial config', () => {
      const config: AccelerationConfig = {};

      expect(Object.keys(config)).toHaveLength(0);
    });

    it('accepts all optional backend override fields', () => {
      const config: AccelerationConfig = {
        gpuNodeThreshold: 1024,
        gpuBatchParallelThreshold: 8,
        workerMinCores: 2,
        maxWorkers: 4,
        disableGPU: false,
        disableWorkers: false,
        backend: 'auto',
      };

      expect(config.backend).toBe('auto');
    });

    it('accepts backend mode gpu', () => {
      const config: AccelerationConfig = { backend: 'gpu' };

      expect(config.backend).toBe('gpu');
    });

    it('accepts backend mode cpu', () => {
      const config: AccelerationConfig = { backend: 'cpu' };

      expect(config.backend).toBe('cpu');
    });
  });

  describe('AccelerationStatus', () => {
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

    it('accepts a GPU-enabled status object', () => {
      const status: AccelerationStatus = {
        mode: 'gpu',
        gpu: {
          available: true,
          reason: 'GPU ready',
        },
        worker: {
          available: false,
          count: 0,
          reason: 'No workers requested',
        },
      };

      expect(status.gpu.available).toBe(true);
    });

    it('accepts a worker-enabled status object', () => {
      const status: AccelerationStatus = {
        mode: 'worker',
        gpu: {
          available: false,
          reason: 'Below GPU threshold',
        },
        worker: {
          available: true,
          count: 4,
          reason: 'Workers ready',
        },
      };

      expect(status.worker.count).toBe(4);
    });
  });

  describe('AccelerationMode', () => {
    it('accepts the three legal mode literals', () => {
      const modes: AccelerationMode[] = ['cpu', 'gpu', 'worker'];

      expect(modes).toEqual(['cpu', 'gpu', 'worker']);
    });
  });

  describe('BackendMode', () => {
    it('accepts the three legal backend literals', () => {
      const backends: BackendMode[] = ['auto', 'gpu', 'cpu'];

      expect(backends).toEqual(['auto', 'gpu', 'cpu']);
    });
  });

  describe('GPUStatus', () => {
    it('records availability and a reason', () => {
      const gpu: GPUStatus = {
        available: true,
        reason: 'adapter found',
      };

      expect(gpu.available).toBe(true);
    });

    it('records unavailability with a reason', () => {
      const gpu: GPUStatus = {
        available: false,
        reason: 'no adapter',
      };

      expect(gpu.reason).toBe('no adapter');
    });
  });

  describe('WorkerStatus', () => {
    it('records availability, count, and a reason', () => {
      const worker: WorkerStatus = {
        available: true,
        count: 4,
        reason: 'cores available',
      };

      expect(worker.count).toBe(4);
    });

    it('records zero count when workers are unavailable', () => {
      const worker: WorkerStatus = {
        available: false,
        count: 0,
        reason: 'single core',
      };

      expect(worker.available).toBe(false);
    });
  });

  describe('AccelerationCapabilities', () => {
    it('describes GPU and worker availability together', () => {
      const capabilities: AccelerationCapabilities = {
        gpu: {
          available: false,
          reason: 'no adapter',
        },
        worker: {
          available: false,
          count: 0,
          reason: 'single core',
        },
      };

      expect(capabilities.gpu.available).toBe(false);
    });

    it('describes available GPU and worker backends', () => {
      const capabilities: AccelerationCapabilities = {
        gpu: {
          available: true,
          reason: 'ready',
        },
        worker: {
          available: true,
          count: 4,
          reason: 'ready',
        },
      };

      expect(capabilities.worker.available).toBe(true);
    });
  });
});
