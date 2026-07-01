import {
  createMockGPUDevice,
  createMockNavigatorGPU,
} from './__mocks__/gpu.mock';
import { isDeviceReady, requestGPUDevice } from './network.gpu.device';

type MutableNavigator = { navigator?: Navigator };

describe('network.gpu.device', () => {
  let originalNavigator: Navigator | undefined;

  beforeEach(() => {
    originalNavigator = globalThis.navigator as Navigator | undefined;
  });

  afterEach(() => {
    (globalThis as unknown as MutableNavigator).navigator = originalNavigator;
  });

  describe('requestGPUDevice', () => {
    it('requests a high-performance adapter', async () => {
      const requestAdapter = jest.fn(async () => null);
      const gpu = { requestAdapter } as unknown as GPU;
      (globalThis as unknown as MutableNavigator).navigator = {
        gpu,
      } as unknown as Navigator;

      await requestGPUDevice();

      expect(requestAdapter).toHaveBeenCalledWith({
        powerPreference: 'high-performance',
      });
    });

    it('forwards requiredLimits derived from adapter limits', async () => {
      const requestDevice = jest.fn(async () => createMockGPUDevice());
      const adapter = {
        limits: {
          maxStorageBufferBindingSize: 128 * 1024 * 1024,
          maxBufferSize: 256 * 1024 * 1024,
        } as GPUSupportedLimits,
        requestDevice,
      } as unknown as GPUAdapter;

      (globalThis as unknown as MutableNavigator).navigator =
        createMockNavigatorGPU({ adapter });

      await requestGPUDevice();

      expect(requestDevice).toHaveBeenCalledWith(
        expect.objectContaining({
          requiredLimits: {
            maxStorageBufferBindingSize: 128 * 1024 * 1024,
            maxBufferSize: 256 * 1024 * 1024,
          },
        }),
      );
    });

    it('reports a requested device as ready', async () => {
      const requestDevice = jest.fn(async () => createMockGPUDevice());
      const adapter = {
        limits: {} as GPUSupportedLimits,
        requestDevice,
      } as unknown as GPUAdapter;

      (globalThis as unknown as MutableNavigator).navigator =
        createMockNavigatorGPU({ adapter });

      const device = await requestGPUDevice();

      expect(device !== null && isDeviceReady(device)).toBe(true);
    });

    it('returns null when navigator.gpu is undefined', async () => {
      (globalThis as unknown as MutableNavigator).navigator =
        {} as unknown as Navigator;

      expect(await requestGPUDevice()).toBeNull();
    });

    it('returns null when no adapter is available', async () => {
      (globalThis as unknown as MutableNavigator).navigator =
        createMockNavigatorGPU({ adapter: null });

      expect(await requestGPUDevice()).toBeNull();
    });

    it('returns null when device creation is rejected', async () => {
      const requestDevice = jest.fn(async () => {
        throw new Error('mock device creation failure');
      });
      const adapter = {
        limits: {
          maxStorageBufferBindingSize: 128 * 1024 * 1024,
        } as GPUSupportedLimits,
        requestDevice,
      } as unknown as GPUAdapter;

      (globalThis as unknown as MutableNavigator).navigator =
        createMockNavigatorGPU({ adapter });

      expect(await requestGPUDevice()).toBeNull();
    });
  });

  describe('isDeviceReady', () => {
    it('returns false for a null device', () => {
      expect(isDeviceReady(null)).toBe(false);
    });

    it('returns false after the device reports lost', () => {
      const device = createMockGPUDevice();
      device.fakeLose();

      expect(isDeviceReady(device)).toBe(false);
    });

    it('remains ready when checked repeatedly', () => {
      const device = createMockGPUDevice();
      isDeviceReady(device);

      expect(isDeviceReady(device)).toBe(true);
    });
  });
});
