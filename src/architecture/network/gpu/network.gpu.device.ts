/**
 * Module-local tracking of which WebGPU devices have reported lost.
 *
 * WebGPU only exposes loss through an async `device.lost` promise, so the
 * runtime keeps a WeakMap that is flipped to `true` when that promise resolves.
 * This allows `isDeviceReady` to give a synchronous answer without polling the
 * GPU process.
 */
const lostDevices = new WeakMap<GPUDevice, boolean>();

/**
 * Attach a one-shot listener to `device.lost` so the module can later answer
 * whether the device is still usable.
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
 * `maxStorageBufferBindingSize` and `maxBufferSize`. Returns `null` safely in
 * non-browser environments, when WebGPU is unavailable, when no adapter can be
 * obtained, or when device creation is rejected.
 *
 * @returns A ready-to-use `GPUDevice`, or `null` when WebGPU cannot be used.
 *
 * @example
 * ```ts
 * const device = await requestGPUDevice();
 * if (device) {
 *   network.gpuDevice = device;
 * }
 * ```
 */
export async function requestGPUDevice(): Promise<GPUDevice | null> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    return null;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) {
      return null;
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
    return device;
  } catch {
    return null;
  }
}

/**
 * Returns true when the supplied WebGPU device is present and has not been
 * reported lost.
 *
 * `null` and `undefined` inputs are treated as not-ready so callers can safely
 * chain GPU probing with CPU fallback logic. Device loss is tracked through
 * the module-local WeakMap populated by `requestGPUDevice` and by lazy
 * attachment on the first call to this function.
 *
 * @param device - WebGPU device to check, or a falsy value when no GPU exists.
 * @returns `true` only when a non-null device is available and not lost.
 */
export function isDeviceReady(device: GPUDevice | null | undefined): boolean {
  if (!device) {
    return false;
  }

  trackDevice(device);

  const trackedLost = lostDevices.get(device)!;
  // Some test doubles expose a synchronous `__lost` marker so deterministic
  // device-loss simulation can avoid waiting for microtasks. Real WebGPU
  // devices never carry this field.
  const mockLost = (device as { __lost?: boolean }).__lost === true;

  return !trackedLost && !mockLost;
}
