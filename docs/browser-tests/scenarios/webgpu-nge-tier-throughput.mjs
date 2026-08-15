import {
  Network,
  methods,
  batchActivate,
} from '../../../dist/neataptic.browser.esm.js';

export { batchActivate };

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
    return methods.Activation.logistic(value, derivate);
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

/**
 * Default warmup iterations for the parallel-agent scenario suite.
 */
export const DEFAULT_PARALLEL_WARMUP_ITERATIONS = 3;

/**
 * Default timed iterations for each scenario in the parallel suite.
 *
 * Kept modest so that very large configurations (e.g. 50×32k) still complete
 * in a browser-friendly amount of time, while giving enough samples for a
 * stable per-second rate.
 */
export const DEFAULT_PARALLEL_BENCHMARK_ITERATIONS = 10;

/**
 * Scenario ladder for the parallel-agent CPU-vs-GPU crossover study.
 *
 * Each entry fixes a hidden-layer size and an agent count. The suite measures
 * sequential-CPU, single-GPU, and batched-GPU throughput for the same total
 * forward-pass shape and reports the first configuration where batched GPU
 * exceeds single-thread CPU.
 */
export const PARALLEL_SCENARIO_CONFIGS = [
  { name: '1×8k', agentCount: 1, hiddenNodes: 8192 },
  { name: '6×8k', agentCount: 6, hiddenNodes: 8192 },
  { name: '12×8k', agentCount: 12, hiddenNodes: 8192 },
  { name: '6×16k', agentCount: 6, hiddenNodes: 16384 },
  { name: '6×32k', agentCount: 6, hiddenNodes: 32768 },
  { name: '10×32k', agentCount: 10, hiddenNodes: 32768 },
  { name: '20×32k', agentCount: 20, hiddenNodes: 32768 },
  { name: '50×32k', agentCount: 50, hiddenNodes: 32768 },
];

/**
 * Run a deterministic MLP cohort on the CPU sequentially.
 *
 * Builds `networkCount` independent networks and activates each one for
 * `iterations` passes in series. The total work is `networkCount * iterations`
 * forward passes measured with one wall-clock interval.
 *
 * @param {object} options - Sequential CPU benchmark options.
 * @param {number} options.hiddenNodes - Hidden-layer node count.
 * @param {number} options.networkCount - Number of networks to run in series.
 * @param {number} options.iterations - Forward passes per network.
 * @param {number} options.warmupIterations - Untimed warmup passes per network.
 * @returns {Promise<object>} CPU throughput metrics.
 */
export async function runSequentialCPU(options) {
  const { hiddenNodes, networkCount, iterations, warmupIterations } = options;

  const networks = [];
  for (let index = 0; index < networkCount; index += 1) {
    networks.push(
      buildDeterministicMLP(INPUT_NODE_COUNT, [hiddenNodes], OUTPUT_NODE_COUNT),
    );
  }

  const input = makeBenchmarkInput(INPUT_NODE_COUNT);

  for (const network of networks) {
    for (let index = 0; index < warmupIterations; index += 1) {
      await network.activate(input);
    }
  }

  const start = performance.now();
  for (const network of networks) {
    for (let index = 0; index < iterations; index += 1) {
      await network.activate(input);
    }
  }
  const wallTimeMs = performance.now() - start;
  const totalActivations = networkCount * iterations;

  return {
    hiddenNodes,
    networkCount,
    iterations,
    totalActivations,
    wallTimeMs,
    throughputActivationsPerSecond: computeThroughput(
      totalActivations,
      wallTimeMs,
    ),
  };
}

/**
 * Run a deterministic MLP on the GPU as a single-agent baseline.
 *
 * Routes the single-network measurement through {@link batchActivate} with
 * `iterations` and `skipUpload: true`. The naïve one-activation-per-call path
 * (`network.activate(..., { useGPU: true })`) pays for a full CPU-GPU round
 * trip on every forward pass and produced only ~54 activations/s for an 8k
 * hidden-node network. Fusing the work into one command buffer with a single
 * mapAsync readback removes that bottleneck and makes the single-GPU baseline
 * comparable to the parallel-GPU path.
 *
 * @param {object} options - Single-agent GPU benchmark options.
 * @param {GPUDevice} options.device - WebGPU device.
 * @param {number} options.hiddenNodes - Hidden-layer node count.
 * @param {number} options.iterations - Forward passes.
 * @param {number} options.warmupIterations - Untimed warmup passes.
 * @returns {Promise<object>} GPU throughput metrics.
 */
export async function runSingleGPU(options) {
  const { device, hiddenNodes, iterations, warmupIterations } = options;
  const network = buildDeterministicMLP(
    INPUT_NODE_COUNT,
    [hiddenNodes],
    OUTPUT_NODE_COUNT,
  );
  network.gpuDevice = device;

  const input = makeBenchmarkInput(INPUT_NODE_COUNT);
  const inputMatrix = new Float32Array(input);

  // Warm up the GPU caches and upload the static weights/biases once.
  await batchActivate(device, [network], inputMatrix);
  if (warmupIterations > 1) {
    await batchActivate(device, [network], inputMatrix, {
      skipUpload: true,
      iterations: warmupIterations - 1,
    });
  }

  // Record every repeated evaluation in a single command buffer so the
  // CPU-GPU round trip is paid once for the whole run.
  const start = performance.now();
  await batchActivate(device, [network], inputMatrix, {
    skipUpload: true,
    iterations,
  });
  const wallTimeMs = performance.now() - start;

  return {
    hiddenNodes,
    networkCount: 1,
    iterations,
    totalActivations: iterations,
    wallTimeMs,
    throughputActivationsPerSecond: computeThroughput(iterations, wallTimeMs),
  };
}

/**
 * Run several deterministic MLPs on the GPU in a single batched submission.
 *
 * Builds `agentCount` independent networks, uploads them once, then records
 * `iterations` forward passes inside one command buffer using
 * {@link batchActivate} with `skipUpload: true`. The aggregate throughput is
 * measured against the wall time for the fused batched call.
 *
 * @param {object} options - Parallel GPU benchmark options.
 * @param {GPUDevice} options.device - WebGPU device shared by agents.
 * @param {number} options.hiddenNodes - Hidden-layer node count.
 * @param {number} options.agentCount - Number of parallel agents.
 * @param {number} options.iterations - Forward passes per agent.
 * @param {number} options.warmupIterations - Untimed warmup passes per agent.
 * @returns {Promise<object>} Parallel GPU throughput metrics.
 */
export async function runParallelGPU(options) {
  const { device, hiddenNodes, agentCount, iterations, warmupIterations } =
    options;

  const networks = [];
  for (let index = 0; index < agentCount; index += 1) {
    const network = buildDeterministicMLP(
      INPUT_NODE_COUNT,
      [hiddenNodes],
      OUTPUT_NODE_COUNT,
    );
    network.gpuDevice = device;
    networks.push(network);
  }

  const input = makeBenchmarkInput(INPUT_NODE_COUNT);
  const inputMatrix = new Float32Array(agentCount * INPUT_NODE_COUNT);
  for (let agent = 0; agent < agentCount; agent += 1) {
    inputMatrix.set(input, agent * INPUT_NODE_COUNT);
  }

  // Warm up the GPU caches, upload the static weights/biases once, and
  // exercise the fused-iteration path so the timed loop measures
  // steady-state command-buffer amortization.
  await batchActivate(device, networks, inputMatrix);
  if (warmupIterations > 1) {
    await batchActivate(device, networks, inputMatrix, {
      skipUpload: true,
      iterations: warmupIterations - 1,
    });
  }

  // Record every repeated evaluation in a single command buffer so the
  // CPU-GPU round trip is paid once for the whole cohort.
  const start = performance.now();
  await batchActivate(device, networks, inputMatrix, {
    skipUpload: true,
    iterations,
  });
  const wallTimeMs = performance.now() - start;
  const totalActivations = agentCount * iterations;

  return {
    hiddenNodes,
    agentCount,
    iterations,
    totalActivations,
    wallTimeMs,
    throughputActivationsPerSecond: computeThroughput(
      totalActivations,
      wallTimeMs,
    ),
  };
}

/**
 * Measure the parallel-agent CPU-vs-GPU crossover across a ladder of
 * configurations.
 *
 * For each configured scenario this runs sequential CPU, single-GPU, and
 * batched-GPU measurements, then reports per-scenario throughputs and the
 * first scenario where batched GPU throughput exceeds single-thread CPU
 * throughput.
 *
 * @param {object} options - Crossover suite options.
 * @param {GPUDevice|null} options.device - WebGPU device.
 * @param {Array<{name:string,agentCount:number,hiddenNodes:number}>} [options.configs=PARALLEL_SCENARIO_CONFIGS] - Scenarios to run.
 * @param {number} [options.warmupIterations=DEFAULT_PARALLEL_WARMUP_ITERATIONS] - Warmup count.
 * @param {number} [options.benchmarkIterations=DEFAULT_PARALLEL_BENCHMARK_ITERATIONS] - Timed count.
 * @param {(name: string, phase: string) => void} [options.onScenarioStart] - Optional
 *   callback invoked before each scenario phase (for UI status updates).
 * @returns {Promise<object>} Crossover artifact.
 */
export async function runParallelScenarioSuite(options) {
  const device = options.device ?? null;
  const configs = options.configs ?? PARALLEL_SCENARIO_CONFIGS;
  const onScenarioStart = options.onScenarioStart;
  const warmupIterations =
    typeof options.warmupIterations === 'number' && options.warmupIterations >= 0
      ? options.warmupIterations
      : DEFAULT_PARALLEL_WARMUP_ITERATIONS;
  const benchmarkIterations =
    typeof options.benchmarkIterations === 'number' &&
    options.benchmarkIterations > 0
      ? options.benchmarkIterations
      : DEFAULT_PARALLEL_BENCHMARK_ITERATIONS;

  const scenarios = [];
  const errors = [];

  for (const config of configs) {
    const { name, agentCount, hiddenNodes } = config;
    let scenarioResult;

    try {
      onScenarioStart?.(name, 'sequential CPU');
      const sequentialCpu = await runSequentialCPU({
        hiddenNodes,
        networkCount: agentCount,
        iterations: benchmarkIterations,
        warmupIterations,
      });

      onScenarioStart?.(name, 'single GPU');
      const singleGpu = await runSingleGPU({
        device,
        hiddenNodes,
        iterations: benchmarkIterations,
        warmupIterations,
      });

      onScenarioStart?.(name, 'parallel GPU');
      const parallelGpu = await runParallelGPU({
        device,
        hiddenNodes,
        agentCount,
        iterations: benchmarkIterations,
        warmupIterations,
      });

      const crossoverMet =
        parallelGpu.throughputActivationsPerSecond >
        sequentialCpu.throughputActivationsPerSecond;
      const speedupRatio =
        sequentialCpu.throughputActivationsPerSecond > 0
          ? parallelGpu.throughputActivationsPerSecond /
            sequentialCpu.throughputActivationsPerSecond
          : 0;

      scenarioResult = {
        name,
        config,
        sequentialCpu,
        singleGpu,
        parallelGpu,
        crossover: {
          met: crossoverMet,
          speedupRatio,
          sequentialCpuThroughput:
            sequentialCpu.throughputActivationsPerSecond,
          parallelGpuThroughput: parallelGpu.throughputActivationsPerSecond,
        },
      };
    } catch (error) {
      scenarioResult = {
        name,
        config,
        error: String(error),
      };
      errors.push({ name, error: String(error) });
    }

    scenarios.push(scenarioResult);
  }

  const successfulScenarios = scenarios.filter((s) => !s.error);
  const crossoverConfigs = successfulScenarios
    .filter((s) => s.crossover?.met)
    .map((s) => ({
      name: s.name,
      agentCount: s.config.agentCount,
      hiddenNodes: s.config.hiddenNodes,
      speedupRatio: s.crossover.speedupRatio,
      sequentialCpuThroughput: s.crossover.sequentialCpuThroughput,
      parallelGpuThroughput: s.crossover.parallelGpuThroughput,
    }));

  const firstCrossover = crossoverConfigs[0] ?? null;
  const overallMet = firstCrossover !== null;

  return {
    success: errors.length === 0,
    errors: errors.length > 0 ? errors : undefined,
    configuration: {
      warmupIterations,
      benchmarkIterations,
      scenarioCount: configs.length,
    },
    scenarios,
    crossoverSummary: {
      overallMet,
      firstCrossover,
      crossoverConfigCount: crossoverConfigs.length,
      crossoverConfigs,
    },
  };
}

/**
 * Hidden-layer node count for the canonical 6×8k parallel crossover
 * measurement.
 */
export const CROSSOVER_HIDDEN_NODES = 8192;

/**
 * Timed activations for the canonical 6×8k parallel crossover measurement.
 *
 * A larger iteration count amortizes the fixed CPU-GPU mapAsync readback
 * cost over many forward passes. Real-device measurements showed that 10
 * iterations (60 total activations for the 6×8k cohort) kept the GPU just
 * below the CPU baseline; raising the fused batch to 360 activations lets
 * the GPU's compute throughput dominate the synchronization overhead.
 */
export const CROSSOVER_BENCHMARK_ITERATIONS = 60;

/**
 * Untimed warmup activations for the canonical 6×8k parallel crossover
 * measurement.
 */
export const CROSSOVER_WARMUP_ITERATIONS = 3;

/**
 * Measure the parallel-agent CPU-vs-GPU crossover at 8k hidden nodes.
 *
 * Compares 1×8k sequential CPU, 1×8k single GPU, 6×8k sequential CPU, and
 * 6×8k parallel GPU. The crossover verdict is true when the parallel GPU
 * throughput exceeds the single-thread CPU throughput.
 *
 * @param {object} options - Crossover options.
 * @param {GPUDevice|null} options.device - WebGPU device.
 * @param {(message: string) => void} [options.onStatus] - Optional callback
 *   invoked before each measurement phase (for UI status updates).
 * @returns {Promise<object>} Crossover artifact.
 */
export async function runParallelCrossover(options) {
  const device = options.device ?? null;
  const onStatus = options.onStatus;

  const hiddenNodes = CROSSOVER_HIDDEN_NODES;
  const iterations = CROSSOVER_BENCHMARK_ITERATIONS;
  const warmupIterations = CROSSOVER_WARMUP_ITERATIONS;

  onStatus?.('Measuring 1×8k sequential CPU...');
  const cpu1x = await runSequentialCPU({
    hiddenNodes,
    networkCount: 1,
    iterations,
    warmupIterations,
  });

  onStatus?.('Measuring 1×8k single GPU...');
  const gpu1x = await runSingleGPU({
    device,
    hiddenNodes,
    iterations,
    warmupIterations,
  });

  onStatus?.('Measuring 6×8k sequential CPU...');
  const cpu6x = await runSequentialCPU({
    hiddenNodes,
    networkCount: 6,
    iterations,
    warmupIterations,
  });

  onStatus?.('Measuring 6×8k parallel GPU...');
  const gpu6x = await runParallelGPU({
    device,
    hiddenNodes,
    agentCount: 6,
    iterations,
    warmupIterations,
  });

  const crossoverMet =
    gpu6x.throughputActivationsPerSecond >
    cpu1x.throughputActivationsPerSecond;
  const speedupRatio =
    cpu1x.throughputActivationsPerSecond > 0
      ? gpu6x.throughputActivationsPerSecond /
        cpu1x.throughputActivationsPerSecond
      : 0;

  return {
    success: true,
    hiddenNodes,
    configuration: {
      warmupIterations,
      benchmarkIterations: iterations,
    },
    scenarios: {
      cpu1x,
      gpu1x,
      cpu6x,
      gpu6x,
    },
    crossover: {
      met: crossoverMet,
      speedupRatio,
      sequentialCpuThroughput: cpu1x.throughputActivationsPerSecond,
      parallelGpuThroughput: gpu6x.throughputActivationsPerSecond,
    },
  };
}
