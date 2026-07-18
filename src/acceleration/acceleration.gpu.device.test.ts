/**
 * Red tests for the generic GPU device bootstrap primitives.
 *
 * `requestGPUDevice` requests a WebGPU device from the host and rejects when
 * WebGPU is unavailable or the adapter/device request fails. `isDeviceReady`
 * reports whether the module's current device is still usable. The tests use a
 * fake `navigator` so they run in Node/Jest without a real browser environment.
 *
 * This file imports from `./acceleration.gpu.device`, which does not exist yet,
 * so the suite fails with TS2307 until the implementation slice lands.
 */

/** Per-test module handle so the module-level device cache is isolated. */
let requestGPUDevice: (typeof import('./acceleration.gpu.device'))['requestGPUDevice'];
let isDeviceReady: (typeof import('./acceleration.gpu.device'))['isDeviceReady'];

/** Minimal navigator fixture used to mock the WebGPU surface. */
interface NavigatorFixture {
  /** Mock WebGPU API surface, or omitted to simulate a missing GPU. */
  gpu?: {
    requestAdapter: jest.MockedFunction<() => Promise<GPUAdapter | null>>;
  };
}

/** Install a deterministic navigator-like object on globalThis. */
const setNavigator = (fixture: NavigatorFixture): void => {
  const navigatorLike: Record<string, unknown> = {
    hardwareConcurrency: 8,
  };

  if (fixture.gpu) {
    navigatorLike.gpu = fixture.gpu;
  }

  (globalThis as unknown as Record<string, unknown>).navigator =
    navigatorLike as unknown as Navigator;
};

/** Build a fake WebGPU device whose `lost` promise never resolves. */
const createFakeDevice = (): GPUDevice => {
  return {
    lost: new Promise(() => {}),
  } as unknown as GPUDevice;
};

/** Build a fake GPU adapter that resolves to the supplied device. */
const createFakeAdapter = (device: GPUDevice): GPUAdapter => {
  return {
    requestDevice: jest.fn().mockResolvedValue(device),
    limits: {},
  } as unknown as GPUAdapter;
};

/** Build a fake WebGPU device whose `lost` promise is controlled by the test. */
const createFakeDeviceWithLost = (lost: Promise<unknown>): GPUDevice => {
  return {
    lost,
  } as unknown as GPUDevice;
};

/** Create a deferred `lost` promise for deterministic device-loss tests. */
const createDeferredLost = (): {
  promise: Promise<unknown>;
  resolve: () => void;
} => {
  let resolve!: () => void;
  const promise = new Promise<void>((res) => {
    resolve = res;
  }) as unknown as Promise<unknown>;
  return { promise, resolve };
};

beforeEach(async () => {
  jest.resetModules();
  jest.restoreAllMocks();
  const g = globalThis as unknown as Record<string, unknown>;
  delete g.navigator;
  delete g.telemetry;

  const mod = await import('./acceleration.gpu.device');
  requestGPUDevice = mod.requestGPUDevice;
  isDeviceReady = mod.isDeviceReady;
});

describe('acceleration.gpu.device', () => {
  describe('requestGPUDevice', () => {
    it('returns a Promise', async () => {
      setNavigator({
        gpu: {
          requestAdapter: jest.fn().mockResolvedValue(null),
        },
      });

      const result = requestGPUDevice();

      expect(result).toBeInstanceOf(Promise);
      await expect(result).rejects.toThrow(/adapter|webgpu|gpu/i);
    });

    it('resolves with a GPUDevice when adapter and device requests succeed', async () => {
      const device = createFakeDevice();
      setNavigator({
        gpu: {
          requestAdapter: jest
            .fn()
            .mockResolvedValue(createFakeAdapter(device)),
        },
      });

      const result = await requestGPUDevice();

      expect(result).toBe(device);
    });

    it('rejects when navigator is undefined', async () => {
      const g = globalThis as unknown as Record<string, unknown>;
      delete g.navigator;

      await expect(requestGPUDevice()).rejects.toThrow(
        /webgpu|gpu|unavailable/i,
      );
    });

    it('rejects when navigator.gpu is unavailable', async () => {
      setNavigator({});

      await expect(requestGPUDevice()).rejects.toThrow(
        /webgpu|gpu|unavailable/i,
      );
    });

    it('rejects when the adapter request returns null', async () => {
      setNavigator({
        gpu: {
          requestAdapter: jest.fn().mockResolvedValue(null),
        },
      });

      await expect(requestGPUDevice()).rejects.toThrow(/adapter|webgpu|gpu/i);
    });

    it('rejects when the device request fails', async () => {
      setNavigator({
        gpu: {
          requestAdapter: jest.fn().mockResolvedValue({
            requestDevice: jest
              .fn()
              .mockRejectedValue(new Error('device request failed')),
            limits: {},
          } as unknown as GPUAdapter),
        },
      });

      await expect(requestGPUDevice()).rejects.toThrow(/device|request|fail/i);
    });

    it('returns the cached device on subsequent calls', async () => {
      const device = createFakeDevice();
      const requestAdapter = jest
        .fn()
        .mockResolvedValue(createFakeAdapter(device));
      setNavigator({ gpu: { requestAdapter } });

      await requestGPUDevice();
      const second = await requestGPUDevice();

      expect(second).toBe(device);
    });

    it('re-requests when the cached device has been lost', async () => {
      const lostDevice = {
        ...createFakeDevice(),
        __lost: true,
      } as unknown as GPUDevice;
      const freshDevice = createFakeDevice();
      setNavigator({
        gpu: {
          requestAdapter: jest
            .fn()
            .mockResolvedValueOnce(createFakeAdapter(lostDevice))
            .mockResolvedValueOnce(createFakeAdapter(freshDevice)),
        },
      });

      await requestGPUDevice();
      const result = await requestGPUDevice();

      expect(result).toBe(freshDevice);
    });

    it('shares the in-flight request between concurrent calls', async () => {
      const device = createFakeDevice();
      const requestAdapter = jest
        .fn()
        .mockResolvedValue(createFakeAdapter(device));
      setNavigator({ gpu: { requestAdapter } });

      const first = requestGPUDevice();
      const second = requestGPUDevice();

      expect(await first).toBe(device);
      expect(await second).toBe(device);
      expect(requestAdapter).toHaveBeenCalledTimes(1);
    });

    it('passes adapter storage limits to requestDevice', async () => {
      const device = createFakeDevice();
      const requestDevice = jest.fn().mockResolvedValue(device);
      const requestAdapter = jest.fn().mockResolvedValue({
        requestDevice,
        limits: {
          maxStorageBufferBindingSize: 1024,
          maxBufferSize: 2048,
        },
      } as unknown as GPUAdapter);
      setNavigator({ gpu: { requestAdapter } });

      await requestGPUDevice();

      expect(requestDevice).toHaveBeenCalledWith({
        requiredLimits: {
          maxStorageBufferBindingSize: 1024,
          maxBufferSize: 2048,
        },
      });
    });
  });

  describe('isDeviceReady', () => {
    it('returns false before any device has been requested', () => {
      setNavigator({});

      expect(isDeviceReady()).toBe(false);
    });

    it('returns false while a device request is still pending', () => {
      setNavigator({
        gpu: {
          requestAdapter: jest
            .fn()
            .mockImplementation(() => new Promise(() => {})),
        },
      });

      requestGPUDevice();

      expect(isDeviceReady()).toBe(false);
    });

    it('returns true after requestGPUDevice resolves', async () => {
      const device = createFakeDevice();
      setNavigator({
        gpu: {
          requestAdapter: jest
            .fn()
            .mockResolvedValue(createFakeAdapter(device)),
        },
      });

      await requestGPUDevice();

      expect(isDeviceReady()).toBe(true);
    });

    it('returns false after requestGPUDevice rejects', async () => {
      setNavigator({});
      await requestGPUDevice().catch(() => {});

      expect(isDeviceReady()).toBe(false);
    });

    it('returns false after the device reports lost', async () => {
      const { promise, resolve } = createDeferredLost();
      const device = createFakeDeviceWithLost(promise);
      setNavigator({
        gpu: {
          requestAdapter: jest
            .fn()
            .mockResolvedValue(createFakeAdapter(device)),
        },
      });

      await requestGPUDevice();
      resolve();
      await promise.catch(() => {});

      expect(isDeviceReady(device)).toBe(false);
    });

    it('honours the synchronous __lost test marker', () => {
      const device = {
        __lost: true,
        lost: Promise.resolve(undefined),
      } as unknown as GPUDevice;

      expect(isDeviceReady(device)).toBe(false);
    });

    it('remains ready across repeated checks of the same device', async () => {
      const device = createFakeDevice();
      setNavigator({
        gpu: {
          requestAdapter: jest
            .fn()
            .mockResolvedValue(createFakeAdapter(device)),
        },
      });

      await requestGPUDevice();
      isDeviceReady(device);

      expect(isDeviceReady(device)).toBe(true);
    });
  });
});
