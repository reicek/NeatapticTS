
/**
 * Maximum allowed mean absolute difference between CPU and GPU inference
 * outputs for the smoke scenario to pass.
 */
export const MAX_MEAN_ABS_DIFF = 0.1;

/**
 * Maximum allowed absolute difference for any individual output element in
 * the WebGPU inference smoke scenario.
 */
export const MAX_ABS_DIFF = 0.5;

/**
 * Default number of activations used in the CPU and GPU timing loops.
 */
export const DEFAULT_ITERATION_COUNT = 10000;

/**
 * Input passed from the scenario page to `runWebGPUSmoke`.
 *
 * @property {number[]} cpuOutput - CPU reference activation output.
 * @property {number[]|Float32Array} gpuOutput - GPU activation output, if the
 *   device was bound.
 * @property {boolean} gpuDeviceBound - `true` when a WebGPU device was
 *   available and assigned to the network.
 * @property {number} [iterationCount=10000] - Number of timed activations.
 * @property {number} [cpuTotalMs=0] - Total CPU wall-clock time.
 * @property {number} [gpuTotalMs=0] - Total GPU wall-clock time.
 * @property {Record<string, unknown>|null} [gpuAdapterInfo=null] - WebGPU
 *   adapter information, if available.
 */
export const WebGPUSmokeInput = {};

/**
 * Result object emitted on `window.webgpuSmokeResult`.
 *
 * @property {boolean} success - `true` when the GPU path is available and all
 *   parity thresholds are satisfied.
 * @property {number[]} cpuOutput - CPU reference output.
 * @property {number[]} gpuOutput - GPU output, or an empty array when no GPU
 *   path was available.
 * @property {number} maxAbsDiff - Maximum absolute element-wise difference.
 * @property {number} meanAbsDiff - Mean absolute element-wise difference.
 * @property {boolean} gpuDeviceBound - Whether a WebGPU device was bound.
 * @property {number} iterationCount - Number of activations timed.
 * @property {number} cpuTotalMs - Total CPU time.
 * @property {number} gpuTotalMs - Total GPU time (0 if no GPU).
 * @property {number} cpuPerActivationMs - Average CPU time per activation.
 * @property {number} gpuPerActivationMs - Average GPU time per activation (0 if no GPU).
 * @property {number|null} speedUp - CPU total / GPU total, or `null` when no
 *   GPU path was available.
 * @property {Record<string, unknown>|null} gpuAdapterInfo - Adapter metadata.
 */
export const WebGPUSmokeResult = {};

/**
 * Compare CPU and GPU inference outputs and package timing metrics for the
 * 2-3-1 MLP WebGPU smoke test.
 *
 * The page under `docs/browser-tests/webgpu-inference-smoke.html` owns the
 * network construction, device binding, activation calls, and wall-clock
 * timing. It passes the captured outputs and timing data here for metric
 * computation and contract assembly.
 *
 * @param {WebGPUSmokeInput} input - CPU/GPU outputs, timing data, and device-bound flag.
 * @returns {Promise<WebGPUSmokeResult>} A deterministic result object.
 * @throws {Error} if `cpuOutput` is missing or has a different length than
 *   `gpuOutput` when a GPU path is available.
 *
 * @example
 * ```js
 * const result = await runWebGPUSmoke({
 *   cpuOutput: [0.42],
 *   gpuOutput: [0.41],
 *   gpuDeviceBound: true,
 *   iterationCount: 10000,
 *   cpuTotalMs: 12.3,
 *   gpuTotalMs: 4.5,
 * });
 * console.log(result.success); // true
 * console.log(result.speedUp); // ~2.7
 * ```
 */
export async function runWebGPUSmoke(input) {
  if (!Array.isArray(input?.cpuOutput)) {
    throw new Error('runWebGPUSmoke requires a cpuOutput array.');
  }

  const gpuDeviceBound = input.gpuDeviceBound === true;
  const cpuOutput = input.cpuOutput.slice();
  const gpuOutput = Array.from(input.gpuOutput ?? []);

  const iterationCount =
    typeof input.iterationCount === 'number' && input.iterationCount > 0
      ? Math.floor(input.iterationCount)
      : DEFAULT_ITERATION_COUNT;

  if (gpuDeviceBound && gpuOutput.length !== cpuOutput.length) {
    throw new Error(
      `CPU/GPU output length mismatch: ${cpuOutput.length} vs ${gpuOutput.length}`,
    );
  }

  let maxAbsDiff = 0;
  let totalAbsDiff = 0;
  const elementCount = cpuOutput.length;

  for (let index = 0; index < elementCount; index++) {
    const difference = Math.abs(cpuOutput[index] - (gpuOutput[index] ?? 0));
    maxAbsDiff = Math.max(maxAbsDiff, difference);
    totalAbsDiff += difference;
  }

  const meanAbsDiff = elementCount > 0 ? totalAbsDiff / elementCount : 0;
  const success =
    gpuDeviceBound &&
    maxAbsDiff <= MAX_ABS_DIFF &&
    meanAbsDiff <= MAX_MEAN_ABS_DIFF;

  const cpuTotalMs = typeof input.cpuTotalMs === 'number' ? input.cpuTotalMs : 0;
  const gpuTotalMs =
    gpuDeviceBound && typeof input.gpuTotalMs === 'number' ? input.gpuTotalMs : 0;

  const cpuPerActivationMs =
    iterationCount > 0 ? cpuTotalMs / iterationCount : 0;
  const gpuPerActivationMs =
    gpuDeviceBound && iterationCount > 0 ? gpuTotalMs / iterationCount : 0;

  let speedUp = null;
  if (gpuDeviceBound && gpuTotalMs > 0) {
    speedUp = cpuTotalMs / gpuTotalMs;
  } else if (gpuDeviceBound) {
    speedUp = Infinity;
  }

  return {
    success,
    cpuOutput,
    gpuOutput,
    maxAbsDiff,
    meanAbsDiff,
    gpuDeviceBound,
    iterationCount,
    cpuTotalMs,
    gpuTotalMs,
    cpuPerActivationMs,
    gpuPerActivationMs,
    speedUp,
    gpuAdapterInfo: input.gpuAdapterInfo ?? null,
  };
}
