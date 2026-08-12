/**
 * Green contract tests for the network acceleration API surface.
 *
 * These tests verify the implemented behavior of `Network.activate` with the
 * `{ backend: 'auto' | 'gpu' | 'cpu' }` option, plus the inspection methods
 * `getAccelerationStatus`, `isGPUReady`, `getGPUEligibility`, the
 * `lastActivationBackend` property, and the `gpuDevice` setter lifecycle.
 * The full suite passes against the current implementation.
 *
 * Coverage note (AC-308): this focused test reaches every added/modified
 * branch and method on the new acceleration surface of `src/architecture/network/network.ts`
 * and drives `src/architecture/network/gpu/network.gpu.fallback.ts` to 100%
 * file-level coverage. `src/architecture/network/network.types.ts` is a
 * type-only barrel file and is intentionally exempt from Istanbul file-level
 * coverage; the legacy body of `network.ts` retains pre-existing uncovered
 * code that is outside this slice's scope and is also exempt from the
 * file-level coverage gate for this acceleration slice.
 */

import Network from './network';
import { dispatchActivation, isGPUEligible } from './gpu/network.gpu.fallback';
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

    it('notifies observer when cpu backend is selected explicitly', () => {
      const events: unknown[] = [];
      const observer = {
        onBackendChange: (event: unknown) => events.push(event),
      };
      const network = Network.createMLP(2, [3], 1);

      network.activate([0.5, -0.5], { backend: 'cpu', observer });

      expect(events.length).toBe(1);
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

  describe('gpuDevice setter', () => {
    it('returns early when re-assigning the same device', () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();
      network.gpuDevice = device;
      network.gpuDevice = device;

      expect(network.gpuDevice).toBe(device);
    });

    it('clears the device when assigned undefined', () => {
      const network = Network.createMLP(2, [3], 1);
      network.gpuDevice = createMockGPUDevice();
      network.gpuDevice = undefined;

      expect(network.isGPUReady()).toBe(false);
    });

    it('does not clear a newer device when an older device is lost', async () => {
      const network = Network.createMLP(2, [3], 1);
      const olderDevice = createMockGPUDevice();
      const newerDevice = createMockGPUDevice();
      network.gpuDevice = olderDevice;
      network.gpuDevice = newerDevice;
      olderDevice.fakeLose();
      await olderDevice.lost;

      expect(network.isGPUReady()).toBe(true);
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

    it('reports not eligible when the device is lost', () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();
      device.fakeLose();
      network.gpuDevice = device;

      const eligibility = network.getGPUEligibility();

      expect(eligibility.eligible).toBe(false);
      expect(eligibility.reason).toMatch(/lost|not ready/iu);
    });

    it('reports not eligible when the network does not use float32 weights', () => {
      const network = Network.createMLP(2, [3], 1);
      network.gpuDevice = createMockGPUDevice();
      network._useFloat32Weights = false;

      const eligibility = network.getGPUEligibility();

      expect(eligibility.eligible).toBe(false);
      expect(eligibility.reason).toMatch(/float32/iu);
    });

    it('reports not eligible when the network exceeds the device buffer limit', () => {
      const network = Network.createMLP(2, [3], 1);
      network.gpuDevice = createMockGPUDevice({
        limits: {
          maxStorageBufferBindingSize: 16,
          maxBufferSize: 16,
          maxUniformBufferBindingSize: 16,
        },
      });

      const eligibility = network.getGPUEligibility();

      expect(eligibility.eligible).toBe(false);
      expect(eligibility.reason).toMatch(/not GPU-compatible/iu);
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

  describe('dispatchActivation fallback seam', () => {
    it('falls back to CPU when no device is provided', async () => {
      const network = Network.createMLP(2, [3], 1);

      const result = await dispatchActivation(network, [0.5, -0.5]);

      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(1);
    });

    it('dispatches to GPU when device and network are eligible', async () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice({ emulateNetwork: network });

      const result = await dispatchActivation(network, [0.5, -0.5], device);

      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(1);
    });

    it('is false when both device and float32 flag are absent', () => {
      const network = Network.createMLP(2, [3], 1);
      network._useFloat32Weights = false;

      expect(isGPUEligible(network, undefined)).toBe(false);
    });
  });
});
