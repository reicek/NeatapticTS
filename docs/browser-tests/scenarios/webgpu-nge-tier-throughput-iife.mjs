const Neataptic = window.Neataptic;
const Network = Neataptic.Network;
const methods = Neataptic.methods;
/**
 * Hidden-layer sizes for the single-window NGE throughput ladder.
 *
 * These sizes bracket the NGE-scale regime from small hidden layers (64) up
 * to the largest single-hidden-layer topology the current WebGPU kernel is
 * expected to support without splitting (32768). Total node counts are
 * `tier + INPUT_NODE_COUNT + OUTPUT_NODE_COUNT`.
 */
export const HIDDEN_NODE_TIERS = [64, 256, 1024, 4096, 8192, 16384, 32768];

/**
 * Number of input nodes for every tiered network.
 */
export const INPUT_NODE_COUNT = 10;

/**
 * Number of output nodes for every tiered network.
 */
export const OUTPUT_NODE_COUNT = 4;

/**
 * Number of untimed GPU activations to run before measuring each tier.
 *
 * Warmup amortizes one-off costs such as pipeline compilation and buffer
 * upload so the timed loop reflects steady-state throughput.
 */
export const DEFAULT_WARMUP_ITERATIONS = 3;

/**
 * Number of timed activations per tier for the throughput measurement.
 *
 * Kept small (10) so even the largest tier completes quickly in the browser,
 * while still yielding a stable per-second rate.
 */
export const DEFAULT_BENCHMARK_ITERATIONS = 10;

/**
 * Default artifact filename used by the browser download prompt.
 */
export const ARTIFACT_FILE_NAME = 'artifacts/webgpu-throughput-single.json';

/**
 * Reference hardware metadata embedded in every artifact.
 *
 * This is the canonical platform against which the NGE WebGPU throughput
 * experiments are reported. Probed GPU limits are recorded separately in
 * `environment.gpuProbedLimits`.
 */
export const REFERENCE_HARDWARE = {
  cpu: {
    model: 'Intel Core i7-10700',
    cores: 8,
    threads: 16,
  },
  ram: {
    totalGB: 32,
    type: 'DDR4-3200',
    availableVirtualMemoryGB: 81.5,
  },
  os: {
    name: 'Windows 11 Home',
    version: '10.0.26200',
  },
  device: {
    formFactor: 'Desktop PC',
  },
  gpu: {
    model: 'NVIDIA Lovelace RTX 4070',
  },
};

/**
 * Activation registry symbol used by the WebGPU kernel to resolve the
 * logistic squash across module/bundle boundaries.
 */
const ACTIVATION_KEY_SYMBOL = Symbol.for('neataptic.activation.key');

/**
 * Wrap the built-in logistic activation with a stable worker-registry key.
 *
 * Bundled browser builds can strip or rename function identities, which
 * breaks the strict-identity lookup used by the GPU activation resolver.
 * Assigning a known `Symbol.for('neataptic.activation.key')` and function name
 * guarantees the kernel can dispatch this activation regardless of how the
 * page was bundled.
 *
 * @returns {Function} A GPU-dispatchable logistic activation function.
 */
export function makeGPULogisticActivation() {
  const workerKey = 'logisticActivation2';

  /**
   * @param {number} value
   * @param {boolean} [derivate=false]
   * @returns {number}
   */
  function gpuLogisticActivation(value, derivate = false) {
    return Neataptic.methods.Activation.logistic(value, derivate);
  }

  gpuLogisticActivation[ACTIVATION_KEY_SYMBOL] = workerKey;

  try {
    Object.defineProperty(gpuLogisticActivation, 'name', {
      value: workerKey,
      configurable: true,
    });
  } catch (_) {
    // Older engines may not allow redefining function.name; the symbol key
    // is already sufficient for dispatch.
  }

  return gpuLogisticActivation;
}

/**
 * Build a deterministic, single-hidden-layer MLP for throughput measurement.
 *
 * `Network.createMLP` initialises weights and biases from random sources. This
 * helper overrides every connection weight and node bias with a deterministic
 * sequence so repeated runs produce the same network state and comparable
 * GPU timings.
 *
 * Every node that already has a squash function is switched to the wrapped
 * GPU-dispatchable logistic activation.
 *
 * @param {number} inputCount - Number of input nodes.
 * @param {number[]} hiddenCounts - Hidden-layer node counts.
 * @param {number} outputCount - Number of output nodes.
 * @returns {Network} A deterministic MLP eligible for the GPU fast path.
 */
export function buildDeterministicMLP(
  inputCount,
  hiddenCounts,
  outputCount,
) {
  const network = Network.createMLP(inputCount, hiddenCounts, outputCount);
  const gpuLogistic = makeGPULogisticActivation();

  for (let index = 0; index < network.nodes.length; index += 1) {
    const node = network.nodes[index];
    if (typeof node.squash === 'function') {
      node.squash = gpuLogistic;
    }
    node.bias = 0.001 * (index % 11);
  }

  for (let index = 0; index < network.connections.length; index += 1) {
    const connection = network.connections[index];
    connection.weight = 0.01 * ((index % 19) - 9);
  }

  return network;
}

/**
 * Produce a deterministic input vector of the requested length.
 *
 * The pattern is simple but not all zeros, so the forward pass exercises the
 * network with non-trivial signal values.
 *
 * @param {number} inputCount - Length of the input vector.
 * @returns {number[]} Deterministic input values.
 */
export function makeBenchmarkInput(inputCount) {
  const input = new Array(inputCount);
  for (let index = 0; index < inputCount; index += 1) {
    input[index] = 0.1 * ((index % 5) + 1);
  }
  return input;
}

/**
 * Convert a timed wall-clock interval into a per-second activation rate.
 *
 * @param {number} iterations - Number of activations completed.
 * @param {number} wallTimeMs - Total elapsed milliseconds.
 * @returns {number} Activations per second, or 0 when the interval is invalid.
 */
export function computeThroughput(iterations, wallTimeMs) {
  if (!Number.isFinite(wallTimeMs) || wallTimeMs <= 0) {
    return 0;
  }
  return (iterations / wallTimeMs) * 1000;
}

/**
 * Check whether the WebGPU device advertises `timestamp-query` support.
 *
 * @param {GPUDevice|null} device - WebGPU device, or null when unavailable.
 * @returns {boolean} True when `timestamp-query` is available.
 */
export function probeTimestampQuerySupport(device) {
  if (!device || typeof device.features?.has !== 'function') {
    return false;
  }
  try {
    return device.features.has('timestamp-query');
  } catch (_) {
    return false;
  }
}

/**
 * Run a minimal GPU timestamp-query probe and return the elapsed nanoseconds.
 *
 * This does not wrap the benchmark activation loop because `activateGPU`
 * owns its own command encoder. Instead it exercises the timestamp-query API
 * with two consecutive timestamps to confirm the feature works and capture a
 * tiny baseline delta.
 *
 * @param {GPUDevice|null} device - WebGPU device, or null when unavailable.
 * @returns {Promise<number|null>} Nanoseconds between two timestamps, or null
 *   when the feature is unavailable or the probe failed.
 */
export async function probeTimestampQueryNs(device) {
  if (!probeTimestampQuerySupport(device)) {
    return null;
  }

  const count = 2;
  const byteSize = count * 8;

  try {
    const querySet = device.createQuerySet({
      type: 'timestamp',
      count,
    });

    const resolveBuffer = device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });

    const readBuffer = device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = device.createCommandEncoder();
    encoder.writeTimestamp(querySet, 0);
    encoder.writeTimestamp(querySet, 1);
    device.queue.submit([encoder.finish()]);

    const resolveEncoder = device.createCommandEncoder();
    resolveEncoder.resolveQuerySet(querySet, 0, count, resolveBuffer, 0);
    resolveEncoder.copyBufferToBuffer(
      resolveBuffer,
      0,
      readBuffer,
      0,
      byteSize,
    );
    device.queue.submit([resolveEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const times = new BigInt64Array(readBuffer.getMappedRange());
    const deltaNs = Number(times[1] - times[0]);
    readBuffer.unmap();

    return Number.isFinite(deltaNs) && deltaNs >= 0 ? deltaNs : null;
  } catch (_) {
    return null;
  }
}

/**
 * Read a subset of probed WebGPU limits relevant to the compute kernel.
 *
 * @param {GPUDevice|null} device - WebGPU device, or null when unavailable.
 * @returns {Record<string, number>|null} Selected limit values, or null.
 */
export function probeGPULimits(device) {
  if (!device || typeof device.limits !== 'object' || device.limits === null) {
    return null;
  }

  const keys = [
    'maxStorageBufferBindingSize',
    'maxBufferSize',
    'maxComputeInvocationsPerWorkgroup',
    'maxComputeWorkgroupSizeX',
    'maxComputeWorkgroupSizeY',
    'maxComputeWorkgroupSizeZ',
    'maxComputeWorkgroupsPerDimension',
    'maxStorageBuffersPerShaderStage',
    'maxBindingsPerBindGroup',
  ];

  const limits = {};
  for (const key of keys) {
    const value = device.limits[key];
    if (typeof value === 'number') {
      limits[key] = value;
    }
  }

  return limits;
}

/**
 * Normalise WebGPU adapter info across the `requestAdapterInfo()` promise API
 * and the newer synchronous `adapter.info` property.
 *
 * @param {GPUAdapter|null} adapter - WebGPU adapter, or null when unavailable.
 * @returns {Promise<Record<string, unknown>|null>} Adapter metadata, or null.
 */
export async function formatAdapterInfo(adapter) {
  if (!adapter) {
    return null;
  }

  try {
    if (typeof adapter.requestAdapterInfo === 'function') {
      return await adapter.requestAdapterInfo();
    }

    if (adapter.info && typeof adapter.info === 'object') {
      return {
        vendor: adapter.info.vendor ?? null,
        architecture: adapter.info.architecture ?? null,
        device: adapter.info.device ?? null,
        description: adapter.info.description ?? null,
      };
    }
  } catch (error) {
    return { error: String(error) };
  }

  return null;
}

/**
 * Run the single-window NGE tiered throughput benchmark.
 *
 * Builds a deterministic MLP for each hidden-layer size, warms it up, then
 * times a fixed number of GPU activations with `performance.now()`. The result
 * object contains per-tier metrics, a summary, reference hardware metadata,
 * probed GPU limits, and an optional GPU timestamp-query probe.
 *
 * @param {object} [options={}] - Benchmark options.
 * @param {GPUDevice|null} [options.device=null] - WebGPU device to use.
 * @param {GPUAdapter|null} [options.adapter=null] - WebGPU adapter for info.
 * @param {number[]} [options.tiers=HIDDEN_NODE_TIERS] - Hidden-layer sizes.
 * @param {number} [options.inputCount=INPUT_NODE_COUNT] - Input nodes per tier.
 * @param {number} [options.outputCount=OUTPUT_NODE_COUNT] - Output nodes per tier.
 * @param {number} [options.warmupIterations=DEFAULT_WARMUP_ITERATIONS] - Warmup count.
 * @param {number} [options.benchmarkIterations=DEFAULT_BENCHMARK_ITERATIONS] - Timed count.
 * @returns {Promise<Record<string, unknown>>} Benchmark artifact.
 */
export async function runNGETierBenchmark(options = {}) {
  const device = options.device ?? null;
  const adapter = options.adapter ?? null;
  const tiers = options.tiers ?? HIDDEN_NODE_TIERS;
  const inputCount =
    typeof options.inputCount === 'number' && options.inputCount > 0
      ? options.inputCount
      : INPUT_NODE_COUNT;
  const outputCount =
    typeof options.outputCount === 'number' && options.outputCount > 0
      ? options.outputCount
      : OUTPUT_NODE_COUNT;
  const warmupIterations = Math.max(
    0,
    typeof options.warmupIterations === 'number'
      ? options.warmupIterations
      : DEFAULT_WARMUP_ITERATIONS,
  );
  const benchmarkIterations = Math.max(
    1,
    typeof options.benchmarkIterations === 'number'
      ? options.benchmarkIterations
      : DEFAULT_BENCHMARK_ITERATIONS,
  );

  const tierResults = [];
  let totalWallTimeMs = 0;

  for (const hiddenNodes of tiers) {
    const network = buildDeterministicMLP(
      inputCount,
      [hiddenNodes],
      outputCount,
    );
    if (device) {
      network.gpuDevice = device;
    }

    const input = makeBenchmarkInput(inputCount);

    for (let index = 0; index < warmupIterations; index += 1) {
      await network.activate(input, { useGPU: true });
    }

    const start = performance.now();
    for (let index = 0; index < benchmarkIterations; index += 1) {
      await network.activate(input, { useGPU: true });
    }
    const wallTimeMs = performance.now() - start;

    const throughput = computeThroughput(benchmarkIterations, wallTimeMs);
    totalWallTimeMs += wallTimeMs;

    tierResults.push({
      hiddenNodes,
      totalNodes: network.nodes.length,
      inputNodes: inputCount,
      outputNodes: outputCount,
      benchmarkIterations,
      warmupIterations,
      wallTimeMs,
      throughputActivationsPerSecond: throughput,
    });
  }

  const timestampQuerySupported = probeTimestampQuerySupport(device);
  const gpuTimestampQueryNs = await probeTimestampQueryNs(device);

  const averageThroughput =
    tierResults.reduce(
      (sum, result) => sum + result.throughputActivationsPerSecond,
      0,
    ) / (tierResults.length || 1);

  return {
    schemaVersion: '1.0.0',
    generatedAt: new Date().toISOString(),
    environment: {
      userAgent:
        typeof navigator !== 'undefined' ? navigator.userAgent : 'node',
      platform:
        typeof navigator !== 'undefined' ? navigator.platform : 'node',
      referenceHardware: REFERENCE_HARDWARE,
      gpuAdapterInfo: await formatAdapterInfo(adapter),
      gpuProbedLimits: probeGPULimits(device),
      timestampQuerySupported,
      gpuTimestampQueryNs,
    },
    configuration: {
      hiddenNodeTiers: tiers,
      inputNodeCount: inputCount,
      outputNodeCount: outputCount,
      warmupIterations,
      benchmarkIterations,
    },
    tiers: tierResults,
    summary: {
      totalWallTimeMs,
      tierCount: tierResults.length,
      minHiddenNodes: Math.min(...tiers),
      maxHiddenNodes: Math.max(...tiers),
      averageThroughputActivationsPerSecond: averageThroughput,
    },
    artifactFileName: ARTIFACT_FILE_NAME,
  };
}

