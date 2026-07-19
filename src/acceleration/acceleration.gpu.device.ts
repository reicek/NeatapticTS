/**
 * Generic WebGPU device bootstrap for the acceleration layer.
 *
 * `requestGPUDevice` asks the host for a high-performance WebGPU adapter and
 * device, rejecting cleanly when WebGPU is missing, no adapter is available,
 * or device creation fails. `isDeviceReady` reports whether a device (the
 * module's cached device, or a caller-supplied one) is still usable. Device
 * loss is tracked lazily through `device.lost` so the synchronous check stays
 * cheap.
 *
 * These helpers are intentionally environment-agnostic: they probe the global
 * `navigator.gpu` surface and throw descriptive errors instead of returning
 * opaque `null` values. That lets callers distinguish "WebGPU missing" from
 * "adapter denied" from "device creation rejected" while still falling back to
 * CPU when needed.
 *
 * Background reading:
 * - WebGPU is described in [WebGPU (Wikipedia)](https://en.wikipedia.org/wiki/WebGPU).
 * - The W3C WebGPU specification is the authoritative reference:
 *   [WebGPU API](https://www.w3.org/TR/webgpu/).
 */

/** Error message marker used when the WebGPU surface is not present. */
const GPU_UNAVAILABLE_MESSAGE = 'WebGPU is unavailable in this environment';

/** Error message marker used when no adapter can be obtained. */
const GPU_ADAPTER_MISSING_MESSAGE = 'No WebGPU adapter available';

/** Error message marker used when device creation fails. */
const GPU_DEVICE_REQUEST_FAILED_MESSAGE = 'GPU device request failed';

/**
 * Module-local tracking of which WebGPU devices have reported lost.
 *
 * WebGPU only exposes loss through an async `device.lost` promise, so the
 * runtime keeps a WeakMap that is flipped to `true` when that promise resolves.
 * This allows `isDeviceReady` to give a synchronous answer without polling the
 * GPU process.
 */
const lostDevices = new WeakMap<GPUDevice, boolean>();

/** The most recently resolved device, or `null` after a failed request. */
let cachedDevice: GPUDevice | null = null;

/** In-flight request promise, used to keep `isDeviceReady` honest mid-probe. */
let pendingRequest: Promise<GPUDevice> | null = null;

/**
 * Attach a one-shot listener to `device.lost` so the module can later answer
 * whether the device is still usable.
 *
 * @internal
 */
function trackDevice(device: GPUDevice): void {
  if (lostDevices.has(device)) {
    return;
  }

  lostDevices.set(device, false);
  void device.lost.then(() => {
    lostDevices.set(device, true);
  });
}

/**
 * Request a high-performance WebGPU device suitable for compute inference.
 *
 * Probes `navigator.gpu`, requests a `high-performance` adapter, then asks
 * the adapter for a device whose limits match the adapter's reported limits for
 * `maxStorageBufferBindingSize` and `maxBufferSize`. Rejects with a descriptive
 * error when WebGPU is unavailable, no adapter can be obtained, or device
 * creation fails.
 *
 * The resolved device is cached at module scope so subsequent calls can reuse
 * a ready device and `isDeviceReady` can report status without an argument.
 * Concurrent calls while a request is in flight return the same promise.
 *
 * @returns A ready-to-use `GPUDevice`.
 * @throws Error when WebGPU is unavailable, no adapter is found, or device
 *   creation is rejected.
 *
 * @example
 * ```ts
 * try {
 *   const device = await requestGPUDevice();
 *   network.gpuDevice = device;
 * } catch (error) {
 *   console.log('GPU unavailable:', error.message);
 * }
 * ```
 */
export async function requestGPUDevice(): Promise<GPUDevice> {
  if (cachedDevice !== null && isDeviceReady(cachedDevice)) {
    return cachedDevice;
  }

  if (pendingRequest !== null) {
    return pendingRequest;
  }

  if (typeof navigator === 'undefined' || !navigator.gpu) {
    cachedDevice = null;
    throw new Error(GPU_UNAVAILABLE_MESSAGE);
  }

  const gpuSurface = navigator.gpu;

  pendingRequest = (async (): Promise<GPUDevice> => {
    try {
      const adapter = await gpuSurface.requestAdapter({
        powerPreference: 'high-performance',
      });
      if (!adapter) {
        throw new Error(GPU_ADAPTER_MISSING_MESSAGE);
      }

      const requiredLimits: GPUSupportedLimits = {};
      if (typeof adapter.limits.maxStorageBufferBindingSize === 'number') {
        requiredLimits.maxStorageBufferBindingSize =
          adapter.limits.maxStorageBufferBindingSize;
      }
      if (typeof adapter.limits.maxBufferSize === 'number') {
        requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
      }

      const device = await adapter.requestDevice({ requiredLimits });
      trackDevice(device);
      cachedDevice = device;
      return device;
    } catch (error) {
      cachedDevice = null;
      throw new Error(GPU_DEVICE_REQUEST_FAILED_MESSAGE, { cause: error });
    }
  })();

  try {
    return await pendingRequest;
  } finally {
    pendingRequest = null;
  }
}

/**
 * Reports whether a WebGPU device is present and has not been reported lost.
 *
 * When called without an argument, this checks the module's cached device
 * (the one most recently returned by {@link requestGPUDevice}). When called
 * with a device, it checks that specific device. `null` and `undefined`
 * inputs are treated as not-ready so callers can safely chain GPU probing
 * with CPU fallback logic.
 *
 * @param device - Optional WebGPU device to check. When omitted, the
 *   module's cached device is used.
 * @returns `true` only when a non-null device is available and not lost.
 *
 * @example
 * ```ts
 * try {
 *   await requestGPUDevice();
 *   console.log(isDeviceReady()); // true
 * } catch {
 *   console.log(isDeviceReady()); // false
 * }
 * ```
 */
export function isDeviceReady(device?: GPUDevice | null | undefined): boolean {
  const target = device ?? cachedDevice;
  if (!target) {
    return false;
  }

  trackDevice(target);

  // `trackDevice` always registers the device, so the map is guaranteed to
  // contain a boolean value at this point.
  const trackedLost = lostDevices.get(target) === true;
  // Some test doubles expose a synchronous `__lost` marker so deterministic
  // device-loss simulation can avoid waiting for microtasks. Real WebGPU
  // devices never carry this field.
  const mockLost = (target as { __lost?: boolean }).__lost === true;

  return !trackedLost && !mockLost;
}
