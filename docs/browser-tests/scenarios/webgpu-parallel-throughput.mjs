import {
  buildDeterministicMLP,
  makeBenchmarkInput,
  computeThroughput,
  probeGPULimits,
  formatAdapterInfo,
  HIDDEN_NODE_TIERS,
  INPUT_NODE_COUNT,
  OUTPUT_NODE_COUNT,
} from './webgpu-nge-tier-throughput-iife.mjs';

/**
 * Number of parallel agents emulated inside a single visible browser window.
 *
 * Each agent owns its own `Network.createMLP()` instance and its own GPU buffers,
 * so the benchmark isolates GPU contention on a single device from the number
 * of independent network objects in flight.
 */
export const PARALLEL_AGENT_COUNT = 6;

/**
 * Untimed GPU activations to run before measuring each agent.
 */
export const DEFAULT_PARALLEL_WARMUP_ITERATIONS = 3;

/**
 * Timed GPU activations per agent for the parallel measurement.
 */
export const DEFAULT_PARALLEL_BENCHMARK_ITERATIONS = 60;

/**
 * Timed activations for the single-agent baseline that feeds the contention
 * overhead calculation.
 */
export const DEFAULT_PARALLEL_BASELINE_ITERATIONS = 10;

/**
 * Minimum `maxStorageBuffersPerShaderStage` the current struct-packed kernel
 * needs. The kernel uses 6 bindings (5 storage buffers + 1 uniform). Probed
 * values below this threshold must fail cleanly instead of producing a cryptic
 * shader compilation error.
 */
export const REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE = 6;

/**
 * Default artifact filename used by the browser download prompt.
 */
export const ARTIFACT_FILE_NAME = 'artifacts/webgpu-throughput-parallel.json';

/**
 * Default scenario path for the single-window parallel benchmark page.
 */
export const DEFAULT_PARALLEL_SCENARIO_PATH =
  '/docs/browser-tests/webgpu-parallel-throughput.html';

/**
 * Single-window parallel benchmark visibility semantics.
 *
 * The benchmark window must be rendered, visible, and hold OS focus. Unlike
 * the previous multi-window design, only one window exists, so `document.hasFocus()`
 * is a valid foreground check here.
 */
const SINGLE_WINDOW_VISIBILITY_CHECK =
  'Single-window parallel benchmark: the benchmark window must be rendered, ' +
  'visible, and in the foreground (document.visibilityState === "visible", ' +
  'document.hidden === false, window.outerWidth > 0, window.outerHeight > 0, ' +
  'document.hasFocus() === true).';

/**
 * Check whether the current browser window is suitable for GPU measurements.
 *
 * Headless browsers, hidden tabs, minimized windows, and backgrounded windows
 * produce invalid GPU timing data and are rejected before any benchmark work
 * begins.
 *
 * @returns {{visible: boolean, browserVisibility: string, visibilityState: string, hasFocus: boolean, outerWidth: number, outerHeight: number, visibility_check: string, reason?: string}}
 *   Visibility verdict. `browserVisibility` is `visible-foreground` when the
 *   window is rendered, visible, and focused; otherwise a rejected reason is
 *   returned.
 */
export function checkBrowserVisibility() {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return {
      visible: false,
      browserVisibility: 'not-a-browser-window',
      visibilityState: 'unknown',
      hasFocus: false,
      outerWidth: 0,
      outerHeight: 0,
      visibility_check: SINGLE_WINDOW_VISIBILITY_CHECK,
      reason: 'No browser window context available.',
    };
  }

  const visibilityState = document.visibilityState ?? 'unknown';
  const hasFocus =
    typeof document.hasFocus === 'function' ? document.hasFocus() : false;
  const outerWidth = window.outerWidth ?? 0;
  const outerHeight = window.outerHeight ?? 0;
  const hidden = document.hidden ?? visibilityState !== 'visible';

  if (hidden || visibilityState !== 'visible') {
    return {
      visible: false,
      browserVisibility: visibilityState,
      visibilityState,
      hasFocus,
      outerWidth,
      outerHeight,
      visibility_check: SINGLE_WINDOW_VISIBILITY_CHECK,
      reason: `Document visibilityState is ${visibilityState} and document.hidden is ${hidden}.`,
    };
  }

  if (outerWidth <= 0 || outerHeight <= 0) {
    return {
      visible: false,
      browserVisibility: 'minimized-or-tiny',
      visibilityState,
      hasFocus,
      outerWidth,
      outerHeight,
      visibility_check: SINGLE_WINDOW_VISIBILITY_CHECK,
      reason: `Window has no rendered size (${outerWidth}x${outerHeight}).`,
    };
  }

  if (!hasFocus) {
    return {
      visible: false,
      browserVisibility: 'not-focused',
      visibilityState,
      hasFocus,
      outerWidth,
      outerHeight,
      visibility_check: SINGLE_WINDOW_VISIBILITY_CHECK,
      reason: 'Window is visible but does not have focus.',
    };
  }

  return {
    visible: true,
    browserVisibility: 'visible-foreground',
    visibilityState,
    hasFocus,
    outerWidth,
    outerHeight,
    visibility_check: SINGLE_WINDOW_VISIBILITY_CHECK,
  };
}

/**
 * Compute a latency distribution from an array of per-activation durations.
 *
 * @param {number[]} samples - Per-activation millisecond durations.
 * @returns {object} Distribution statistics including min, max, mean, median,
 *   p50, p95, p99, count, and standard deviation.
 */
export function computeLatencyDistribution(samples) {
  const valid = samples.filter((value) => Number.isFinite(value) && value >= 0);
  const count = valid.length;
  if (count === 0) {
    return {
      min: 0,
      max: 0,
      mean: 0,
      median: 0,
      p50: 0,
      p95: 0,
      p99: 0,
      count: 0,
      stdDev: 0,
    };
  }

  const sorted = valid.toSorted((a, b) => a - b);
  const min = sorted[0];
  const max = sorted.at(-1);
  const mean = valid.reduce((sum, value) => sum + value, 0) / count;

  const percentile = (p) => {
    const position = p * (count - 1);
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    if (lower === upper) {
      return sorted[lower];
    }
    return (
      sorted[lower] + (position - lower) * (sorted[upper] - sorted[lower])
    );
  };

  const median =
    count % 2 === 1
      ? sorted[Math.floor(count / 2)]
      : (sorted[count / 2 - 1] + sorted[count / 2]) / 2;

  const variance =
    valid.reduce((sum, value) => sum + (value - mean) ** 2, 0) / count;

  return {
    min,
    max,
    mean,
    median,
    p50: percentile(0.5),
    p95: percentile(0.95),
    p99: percentile(0.99),
    count,
    stdDev: Math.sqrt(variance),
  };
}

/**
 * Compute per-agent summary statistics from an array of per-activation
 * millisecond samples.
 *
 * @param {number[]} perActivationMs - Durations for one agent.
 * @returns {object} Per-agent mean/min/max latency and inferred FPS.
 */
function computePerAgentStats(perActivationMs) {
  const distribution = computeLatencyDistribution(perActivationMs);
  return {
    mean_latency_ms: distribution.mean,
    min_latency_ms: distribution.min,
    max_latency_ms: distribution.max,
    fps_per_agent:
      distribution.mean > 0 ? 1000 / distribution.mean : 0,
  };
}

/**
 * Compute aggregate throughput across all agents for one tier.
 *
 * @param {Array<{wallTimeMs: number, perActivationMs: number[]}>} agentResults -
 *   One entry per agent.
 * @param {number} benchmarkIterations - Iterations each agent completed.
 * @returns {object} Total forward passes, wall-clock time, and aggregate FPS.
 */
function computeAggregateStats(agentResults, benchmarkIterations) {
  const agentCount = agentResults.length;
  const totalForwardPasses = agentCount * benchmarkIterations;
  const wallClockMs = Math.max(
    ...agentResults.map((result) => result.wallTimeMs ?? 0),
  );
  return {
    total_forward_passes: totalForwardPasses,
    wall_clock_ms: wallClockMs,
    total_fps: computeThroughput(totalForwardPasses, wallClockMs),
  };
}

/**
 * Compute how much the parallel aggregate throughput lags behind the ideal
 * linear scaling of the single-agent baseline.
 *
 * @param {number} singleAgentFps - Baseline throughput of one agent on this tier.
 * @param {number} aggregateFps - Measured aggregate throughput with N agents.
 * @param {number} agentCount - Number of agents run concurrently.
 * @returns {number} Contention overhead in percent (0 = no overhead, 100 = all
 *   throughput lost).
 */
function computeContentionOverheadPct(
  singleAgentFps,
  aggregateFps,
  agentCount,
) {
  const idealFps = singleAgentFps * agentCount;
  if (!Number.isFinite(idealFps) || idealFps <= 0) {
    return 0;
  }
  return Math.max(0, ((idealFps - aggregateFps) / idealFps) * 100);
}

/**
 * Run a single-agent baseline for one tier.
 *
 * @param {GPUDevice} device - WebGPU device.
 * @param {number} hiddenNodes - Hidden-layer size.
 * @param {number} iterations - Timed activations.
 * @returns {Promise<{wallTimeMs: number, perActivationMs: number[], fps: number}>}
 *   Baseline metrics.
 */
async function runSingleAgentBaseline(device, hiddenNodes, iterations) {
  const network = buildDeterministicMLP(
    INPUT_NODE_COUNT,
    [hiddenNodes],
    OUTPUT_NODE_COUNT,
  );
  network.gpuDevice = device;

  const input = makeBenchmarkInput(INPUT_NODE_COUNT);

  // Minimal warmup so the baseline reflects steady-state dispatch.
  for (let index = 0; index < DEFAULT_PARALLEL_WARMUP_ITERATIONS; index += 1) {
    await network.activate(input, { useGPU: true });
  }

  const perActivationMs = [];
  const wallStart = performance.now();
  for (let index = 0; index < iterations; index += 1) {
    const iterStart = performance.now();
    await network.activate(input, { useGPU: true });
    perActivationMs.push(performance.now() - iterStart);
  }
  const wallTimeMs = performance.now() - wallStart;

  return {
    wallTimeMs,
    perActivationMs,
    fps: computeThroughput(iterations, wallTimeMs),
  };
}

/**
 * Run N agents concurrently for one tier and record per-agent timing.
 *
 * @param {GPUDevice} device - WebGPU device shared by all agents.
 * @param {number} hiddenNodes - Hidden-layer size.
 * @param {number} agentCount - Number of parallel agents.
 * @param {number} benchmarkIterations - Timed activations per agent.
 * @param {number} warmupIterations - Untimed activations per agent before timing.
 * @returns {Promise<Array<{wallTimeMs: number, perActivationMs: number[]}>>}
 *   Per-agent result rows.
 */
async function runParallelAgents(
  device,
  hiddenNodes,
  agentCount,
  benchmarkIterations,
  warmupIterations,
) {
  const agents = [];
  for (let index = 0; index < agentCount; index += 1) {
    const network = buildDeterministicMLP(
      INPUT_NODE_COUNT,
      [hiddenNodes],
      OUTPUT_NODE_COUNT,
    );
    network.gpuDevice = device;
    agents.push(network);
  }

  const input = makeBenchmarkInput(INPUT_NODE_COUNT);

  // Warmup every agent before any timing starts.
  for (const agent of agents) {
    for (let index = 0; index < warmupIterations; index += 1) {
      await agent.activate(input, { useGPU: true });
    }
  }

  const agentResults = agents.map(() => ({
    wallTimeMs: 0,
    perActivationMs: [],
  }));

  const batchStart = performance.now();
  for (let index = 0; index < benchmarkIterations; index += 1) {
    await Promise.all(
      agents.map(async (agent, agentIndex) => {
        const iterStart = performance.now();
        await agent.activate(input, { useGPU: true });
        const iterEnd = performance.now();
        agentResults[agentIndex].perActivationMs.push(iterEnd - iterStart);
      }),
    );
  }
  const batchEnd = performance.now();
  const batchWallTimeMs = batchEnd - batchStart;

  for (const result of agentResults) {
    result.wallTimeMs = batchWallTimeMs;
  }

  return agentResults;
}

/**
 * Build the `reference_hardware` object for the parallel artifact.
 *
 * GPU vendor and architecture are read from the live adapter, and
 * `maxStorageBuffersPerShaderStage` is probed from the device limits.
 *
 * @param {GPUAdapter|null} adapter - WebGPU adapter.
 * @param {GPUDevice|null} device - WebGPU device.
 * @returns {Promise<object>} Formatted reference hardware metadata.
 */
async function buildReferenceHardware(adapter, device) {
  const gpuLimits = probeGPULimits(device);
  const adapterInfo = (await formatAdapterInfo(adapter)) ?? {};

  return {
    processor: 'Intel i7-10700 @ 2.90GHz, 8 Cores/16 Logical',
    memory: '32GB DDR4 3200MHz',
    os: 'Windows 11 Home Build 26200',
    gpu_vendor: adapterInfo.vendor ?? 'unknown',
    gpu_architecture: adapterInfo.architecture ?? 'unknown',
    maxStorageBuffersPerShaderStage:
      gpuLimits?.maxStorageBuffersPerShaderStage ?? null,
  };
}

/**
 * Run the single-window parallel throughput benchmark.
 *
 * Creates `agentCount` independent `Network.createMLP()` instances, binds them
 * to the same WebGPU device, and runs concurrent forward passes for every tier
 * in the NGE ladder. A single-agent baseline per tier is measured first so the
 * aggregate result can report GPU contention overhead.
 *
 * @param {object} [options={}] - Benchmark options.
 * @param {GPUDevice|null} [options.device=null] - WebGPU device shared by all
 *   agents.
 * @param {GPUAdapter|null} [options.adapter=null] - WebGPU adapter for info.
 * @param {number[]} [options.tiers=HIDDEN_NODE_TIERS] - Hidden-layer sizes.
 * @param {number} [options.agentCount=PARALLEL_AGENT_COUNT] - Number of parallel
 *   agents to emulate.
 * @param {number} [options.warmupIterations=DEFAULT_PARALLEL_WARMUP_ITERATIONS]
 *   - Warmup activations per agent.
 * @param {number} [options.benchmarkIterations=DEFAULT_PARALLEL_BENCHMARK_ITERATIONS]
 *   - Timed activations per agent.
 * @param {number} [options.baselineIterations=DEFAULT_PARALLEL_BASELINE_ITERATIONS]
 *   - Timed activations for the single-agent baseline.
 * @returns {Promise<object>} Parallel benchmark artifact.
 */
export async function runParallelSingleWindowBenchmark(options = {}) {
  const visibility = checkBrowserVisibility();
  if (!visibility.visible) {
    return {
      benchmark_type: 'parallel-single-window',
      success: false,
      error: visibility.reason,
      browser_visibility: visibility.browserVisibility,
      artifactFileName: ARTIFACT_FILE_NAME,
    };
  }

  const {
    device = null,
    adapter = null,
    tiers = HIDDEN_NODE_TIERS,
    agentCount = PARALLEL_AGENT_COUNT,
    warmupIterations = DEFAULT_PARALLEL_WARMUP_ITERATIONS,
    benchmarkIterations = DEFAULT_PARALLEL_BENCHMARK_ITERATIONS,
    baselineIterations = DEFAULT_PARALLEL_BASELINE_ITERATIONS,
  } = options;

  const probedStorageBuffers = device?.limits?.maxStorageBuffersPerShaderStage ?? 0;
  if (probedStorageBuffers < REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE) {
    return {
      benchmark_type: 'parallel-single-window',
      success: false,
      error: `Adapter maxStorageBuffersPerShaderStage (${probedStorageBuffers}) is below the required ${REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE}.`,
      browser_visibility: visibility.browserVisibility,
      reference_hardware: await buildReferenceHardware(adapter, device),
      visibility_check: {
        visibilityState: visibility.visibilityState,
        outerWidth: visibility.outerWidth,
        outerHeight: visibility.outerHeight,
        hasFocus: visibility.hasFocus,
      },
      artifactFileName: ARTIFACT_FILE_NAME,
    };
  }

  const referenceHardware = await buildReferenceHardware(adapter, device);
  const tierResults = [];

  for (const hiddenNodes of tiers) {
    const baseline = await runSingleAgentBaseline(
      device,
      hiddenNodes,
      baselineIterations,
    );

    const agentResults = await runParallelAgents(
      device,
      hiddenNodes,
      agentCount,
      benchmarkIterations,
      warmupIterations,
    );

    const aggregate = computeAggregateStats(agentResults, benchmarkIterations);
    const perAgentStats = agentResults.map((result) =>
      computePerAgentStats(result.perActivationMs),
    );

    tierResults.push({
      tier: hiddenNodes,
      agent_count: agentCount,
      per_agent: perAgentStats,
      aggregate,
      contention_overhead_pct: computeContentionOverheadPct(
        baseline.fps,
        aggregate.total_fps,
        agentCount,
      ),
      single_agent_baseline_fps: baseline.fps,
    });
  }

  return {
    benchmark_type: 'parallel-single-window',
    success: true,
    generatedAt: new Date().toISOString(),
    reference_hardware: referenceHardware,
    browser_visibility: visibility.browserVisibility,
    environment: {
      userAgent:
        typeof navigator !== 'undefined' ? navigator.userAgent : 'node',
      platform:
        typeof navigator !== 'undefined' ? navigator.platform : 'node',
      gpuAdapterInfo: await formatAdapterInfo(adapter),
      gpuProbedLimits: probeGPULimits(device),
    },
    configuration: {
      agentCount,
      inputNodeCount: INPUT_NODE_COUNT,
      outputNodeCount: OUTPUT_NODE_COUNT,
      warmupIterations,
      benchmarkIterations,
      baselineIterations,
      tiers,
    },
    tier_results: tierResults,
    visibility_check: {
      visibilityState: visibility.visibilityState,
      outerWidth: visibility.outerWidth,
      outerHeight: visibility.outerHeight,
      hasFocus: visibility.hasFocus,
    },
    artifactFileName: ARTIFACT_FILE_NAME,
  };
}

