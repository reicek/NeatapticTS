/**
 * Red tests for the new network acceleration API surface.
 *
 * These tests define the expected behavior of `Network.activate` with the new
 * `{ backend: 'auto' | 'gpu' | 'cpu' }` option, plus the new inspection methods
 * `getAccelerationStatus`, `isGPUReady`, `getGPUEligibility`, and the
 * `lastActivationBackend` property. The implementation does not exist yet, so
 * most of the suite fails with missing-method or missing-property errors. A
 * small number of regression guards intentionally pass under the current code.
 */

import Network from './network';
import { createMockGPUDevice } from './gpu/__mocks__/gpu.mock';

describe('network acceleration API', () => {
  const originalNavigator = (globalThis as unknown as Record<string, unknown>)
    .navigator;

  afterEach(() => {
    jest.restoreAllMocks();
    (globalThis as unknown as Record<string, unknown>).navigator =
      originalNavigator;
  });

  describe('activate with backend option', () => {
    it('activates via CPU when backend is cpu', () => {
      const network = Network.createMLP(2, [3], 1);
      const input = [0.5, -0.5];

      network.activate(input, { backend: 'cpu' });

      expect(network.lastActivationBackend).toBe('cpu');
    });

    it('returns a Float32Array promise when backend is gpu and a device is present', async () => {
      const network = Network.createMLP(2, [3], 1);
      network.gpuDevice = createMockGPUDevice();
      const input = [0.5, -0.5];

      const result = network.activate(input, { backend: 'gpu' });

      expect(result).toBeInstanceOf(Promise);
    });

    it('falls back to CPU when backend is gpu but no device is present', () => {
      const events: unknown[] = [];
      const observer = {
        onFallback: (event: unknown) => events.push(event),
      };
      const network = Network.createMLP(2, [3], 1);

      network.activate([0.5, -0.5], { backend: 'gpu', observer });

      expect(events.length).toBe(1);
    });

    it('chooses GPU for auto backend when a device is attached and eligible', async () => {
      const network = Network.createMLP(2, [3], 1);
      network.gpuDevice = createMockGPUDevice();

      const result = network.activate([0.5, -0.5], { backend: 'auto' });

      expect(result).toBeInstanceOf(Promise);
    });

    it('chooses CPU for auto backend when no device is attached', () => {
      const network = Network.createMLP(2, [3], 1);
      const input = [0.5, -0.5];

      network.activate(input, { backend: 'auto' });

      expect(network.lastActivationBackend).toBe('cpu');
    });

    it('preserves backward compatibility with the boolean training flag', () => {
      const network = Network.createMLP(2, [3], 1);
      const input = [0.5, -0.5];

      const result = network.activate(input, false);

      expect(Array.isArray(result)).toBe(true);
    });
  });

  describe('getAccelerationStatus', () => {
    it('returns a status object for the network', () => {
      const network = Network.createMLP(2, [3], 1);

      const status = network.getAccelerationStatus();

      expect(status).toMatchObject({
        mode: expect.any(String),
        gpu: expect.objectContaining({ available: expect.any(Boolean) }),
        worker: expect.objectContaining({ available: expect.any(Boolean) }),
      });
    });
  });

  describe('isGPUReady', () => {
    it('returns false when no gpuDevice is assigned', () => {
      const network = Network.createMLP(2, [3], 1);

      expect(network.isGPUReady()).toBe(false);
    });

    it('returns true when a mock GPU device is assigned and ready', () => {
      const network = Network.createMLP(2, [3], 1);
      network.gpuDevice = createMockGPUDevice();

      expect(network.isGPUReady()).toBe(true);
    });

    it('returns false when the assigned device is lost', async () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();
      network.gpuDevice = device;
      device.fakeLose();
      await device.lost;

      expect(network.isGPUReady()).toBe(false);
    });
  });

  describe('getGPUEligibility', () => {
    it('reports not eligible when no device is assigned', () => {
      const network = Network.createMLP(2, [3], 1);

      const eligibility = network.getGPUEligibility();

      expect(eligibility.eligible).toBe(false);
    });

    it('reports eligible when a device is assigned to a simple feed-forward network', () => {
      const network = Network.createMLP(2, [3], 1);
      network.gpuDevice = createMockGPUDevice();

      const eligibility = network.getGPUEligibility();

      expect(eligibility.eligible).toBe(true);
    });

    it('includes a human-readable reason', () => {
      const network = Network.createMLP(2, [3], 1);

      const eligibility = network.getGPUEligibility();

      expect(eligibility.reason).toEqual(expect.any(String));
    });
  });

  describe('lastActivationBackend', () => {
    it('defaults to undefined before any activation', () => {
      const network = Network.createMLP(2, [3], 1);

      expect(network.lastActivationBackend).toBeUndefined();
    });

    it('records cpu after a CPU activation', () => {
      const network = Network.createMLP(2, [3], 1);

      network.activate([0.5, -0.5], { backend: 'cpu' });

      expect(network.lastActivationBackend).toBe('cpu');
    });
  });

  describe('observer propagation', () => {
    it('notifies observer when backend changes from cpu to gpu', async () => {
      const events: unknown[] = [];
      const observer = {
        onBackendChange: (event: unknown) => events.push(event),
      };
      const network = Network.createMLP(2, [3], 1);
      network.gpuDevice = createMockGPUDevice();

      await network.activate([0.5, -0.5], { backend: 'gpu', observer });

      expect(events.length).toBe(1);
    });

    it('notifies observer when gpu falls back to cpu', async () => {
      const events: unknown[] = [];
      const observer = {
        onFallback: (event: unknown) => events.push(event),
      };
      const network = Network.createMLP(2, [3], 1);

      await network.activate([0.5, -0.5], { backend: 'gpu', observer });

      expect(events.length).toBe(1);
    });
  });

  describe('legacy useGPU deprecation', () => {
    it('warns once per Network instance when useGPU is used', async () => {
      const warnSpy = jest
        .spyOn(console, 'warn')
        .mockImplementation(() => undefined);
      const network = Network.createMLP(2, [3], 1);
      network.gpuDevice = createMockGPUDevice();

      await network.activate([0.5, -0.5], { useGPU: true });
      await network.activate([0.5, -0.5], { useGPU: true });

      expect(warnSpy).toHaveBeenCalledTimes(1);
    });
  });
});
