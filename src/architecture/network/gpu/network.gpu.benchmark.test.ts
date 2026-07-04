import Network from '../network';
import Activation from '../../../methods/activation/activation';
import {
  GpuProfilingTimer,
  profileGPUActivation,
  prepareActivationContext,
  identifyBottleneck,
  computeOverheadBreakdown,
  rankWeakPoints,
  buildOverheadArtifact,
  type ProfilingResult,
  type ProfilingPhaseTiming,
} from './network.gpu.profiling';
import { createMockGPUDevice } from './__mocks__/gpu.mock';

const ACTIVATION_KEY_SYMBOL = Symbol.for('neataptic.activation.key');

const HIDDEN_NODE_TIERS = [64, 256, 1024, 4096, 8192, 16384, 32768];
const INPUT_NODE_COUNT = 10;
const OUTPUT_NODE_COUNT = 4;
const DEFAULT_WARMUP_ITERATIONS = 3;
const DEFAULT_BENCHMARK_ITERATIONS = 10;

/**
 * Reference hardware metadata that the browser artifact embeds. These values
 * mirror the canonical platform used by the NGE WebGPU throughput benchmark.
 */
const REFERENCE_HARDWARE = {
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
 * Create a GPU-dispatchable wrapper around the built-in logistic activation.
 *
 * The wrapper carries the worker-registry symbol key and a stable name so the
 * GPU kernel can resolve it across bundle/module boundaries.
 */
function makeGPULogisticActivation() {
  const workerKey = 'logisticActivation2';

  function gpuLogisticActivation(
    value: number,
    derivate: boolean = false,
  ): number {
    return Activation.logistic(value, derivate);
  }

  (gpuLogisticActivation as unknown as Record<symbol, string | undefined>)[
    ACTIVATION_KEY_SYMBOL
  ] = workerKey;

  try {
    Object.defineProperty(gpuLogisticActivation, 'name', {
      value: workerKey,
      configurable: true,
    });
  } catch {
    // Older engines may not allow redefining function.name.
  }

  return gpuLogisticActivation;
}

/**
 * Build a deterministic MLP with fixed weights and biases.
 */
function buildDeterministicMLP(
  inputCount: number,
  hiddenCounts: number[],
  outputCount: number,
): Network {
  const network = Network.createMLP(inputCount, hiddenCounts, outputCount);
  const gpuLogistic = makeGPULogisticActivation();

  for (let index = 0; index < network.nodes.length; index += 1) {
    const node = network.nodes[index];
    if (typeof node.squash === 'function') {
      node.squash = gpuLogistic as typeof node.squash;
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
 * Produce the deterministic input pattern used by the benchmark loop.
 */
function makeBenchmarkInput(inputCount: number): number[] {
  const input = new Array<number>(inputCount);
  for (let index = 0; index < inputCount; index += 1) {
    input[index] = 0.1 * ((index % 5) + 1);
  }
  return input;
}

/**
 * Convert a wall-clock interval into a per-second activation rate.
 */
function computeThroughput(iterations: number, wallTimeMs: number): number {
  if (!Number.isFinite(wallTimeMs) || wallTimeMs <= 0) {
    return 0;
  }
  return (iterations / wallTimeMs) * 1000;
}

/**
 * Check whether a WebGPU device advertises `timestamp-query` support.
 */
function probeTimestampQuerySupport(device: GPUDevice | null): boolean {
  if (!device) {
    return false;
  }
  const features = (device as unknown as Record<string, unknown>).features;
  if (!features || typeof (features as Set<string>).has !== 'function') {
    return false;
  }
  try {
    return (features as Set<string>).has('timestamp-query');
  } catch {
    return false;
  }
}

/**
 * Read a subset of WebGPU limits relevant to the compute kernel.
 */
function probeGPULimits(
  device: GPUDevice | null,
): Record<string, number> | null {
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

  const limits: Record<string, number> = {};
  for (const key of keys) {
    const value = (device.limits as Record<string, unknown>)[key];
    if (typeof value === 'number') {
      limits[key] = value;
    }
  }

  return limits;
}

/**
 * Normalise adapter info from either the promise API or the sync property.
 */
async function formatAdapterInfo(
  adapter: GPUAdapter | null,
): Promise<Record<string, unknown> | null> {
  if (!adapter) {
    return null;
  }

  try {
    const adapterRecord = adapter as unknown as Record<string, unknown>;
    const requestAdapterInfo = adapterRecord.requestAdapterInfo;
    if (typeof requestAdapterInfo === 'function') {
      return (await (
        requestAdapterInfo as () => Promise<Record<string, unknown>>
      )()) as Record<string, unknown>;
    }

    const info = adapterRecord.info;
    if (info && typeof info === 'object') {
      const infoRecord = info as Record<string, unknown>;
      return {
        vendor: infoRecord.vendor ?? null,
        architecture: infoRecord.architecture ?? null,
        device: infoRecord.device ?? null,
        description: infoRecord.description ?? null,
      };
    }
  } catch (error) {
    return { error: String(error) };
  }

  return null;
}

/**
 * Assemble the benchmark artifact from tier results and options.
 */
function assembleArtifact(
  tierResults: Array<{
    hiddenNodes: number;
    totalNodes: number;
    throughputActivationsPerSecond: number;
    wallTimeMs: number;
  }>,
  options: {
    tiers?: number[];
    inputCount?: number;
    outputCount?: number;
    warmupIterations?: number;
    benchmarkIterations?: number;
    timestampQuerySupported?: boolean;
    gpuTimestampQueryNs?: number | null;
  } = {},
): Record<string, unknown> {
  const tiers = options.tiers ?? HIDDEN_NODE_TIERS;
  const inputCount =
    typeof options.inputCount === 'number' && options.inputCount > 0
      ? options.inputCount
      : INPUT_NODE_COUNT;
  const outputCount =
    typeof options.outputCount === 'number' && options.outputCount > 0
      ? options.outputCount
      : OUTPUT_NODE_COUNT;
  const warmupIterations =
    typeof options.warmupIterations === 'number'
      ? options.warmupIterations
      : DEFAULT_WARMUP_ITERATIONS;
  const benchmarkIterations =
    typeof options.benchmarkIterations === 'number'
      ? options.benchmarkIterations
      : DEFAULT_BENCHMARK_ITERATIONS;

  const totalWallTimeMs = tierResults.reduce(
    (sum, result) => sum + result.wallTimeMs,
    0,
  );
  const averageThroughput =
    tierResults.reduce(
      (sum, result) => sum + result.throughputActivationsPerSecond,
      0,
    ) / (tierResults.length || 1);

  return {
    schemaVersion: '1.0.0',
    generatedAt: new Date().toISOString(),
    environment: {
      userAgent: 'node',
      platform: 'node',
      referenceHardware: REFERENCE_HARDWARE,
      gpuAdapterInfo: null,
      gpuProbedLimits: null,
      timestampQuerySupported: options.timestampQuerySupported ?? false,
      gpuTimestampQueryNs: options.gpuTimestampQueryNs ?? null,
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
    artifactFileName: 'artifacts/webgpu-throughput-single.json',
  };
}

/**
 * Compute a latency distribution from an array of per-activation durations.
 *
 * Mirror of the browser-side helper so the Node test suite can exercise the
 * same math used by the single-window parallel benchmark.
 */
function computeLatencyDistribution(samples: number[]): Record<string, number> {
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
  const max = sorted.at(-1) ?? 0;
  const mean = valid.reduce((sum, value) => sum + value, 0) / count;

  const percentile = (p: number): number => {
    const position = p * (count - 1);
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    if (lower === upper) {
      return sorted[lower];
    }
    return sorted[lower] + (position - lower) * (sorted[upper] - sorted[lower]);
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
 * Compute per-agent summary statistics from per-activation millisecond samples.
 */
function computePerAgentStats(perActivationMs: number[]) {
  const distribution = computeLatencyDistribution(perActivationMs);
  return {
    mean_latency_ms: distribution.mean,
    min_latency_ms: distribution.min,
    max_latency_ms: distribution.max,
    fps_per_agent: distribution.mean > 0 ? 1000 / distribution.mean : 0,
  };
}

/**
 * Compute aggregate throughput across all agents for one tier.
 */
function computeAggregateStats(
  agentResults: Array<{ wallTimeMs: number; perActivationMs: number[] }>,
  benchmarkIterations: number,
) {
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
 * Compute how much parallel aggregate throughput lags behind ideal linear
 * scaling of the single-agent baseline.
 */
function computeContentionOverheadPct(
  singleAgentFps: number,
  aggregateFps: number,
  agentCount: number,
): number {
  const idealFps = singleAgentFps * agentCount;
  if (!Number.isFinite(idealFps) || idealFps <= 0) {
    return 0;
  }
  return Math.max(0, ((idealFps - aggregateFps) / idealFps) * 100);
}

/**
 * Reference hardware metadata for the single-window parallel artifact.
 */
const PARALLEL_REFERENCE_HARDWARE = {
  processor: 'Intel i7-10700 @ 2.90GHz, 8 Cores/16 Logical',
  memory: '32GB DDR4 3200MHz',
  os: 'Windows 11 Home Build 26200',
  gpu_vendor: 'nvidia',
  gpu_architecture: 'lovelace',
  maxStorageBuffersPerShaderStage: 8,
};

/**
 * Assemble the single-window parallel benchmark artifact.
 */
function assembleParallelArtifact(
  tierResults: Array<{
    tier: number;
    agent_count: number;
    per_agent: Array<{
      mean_latency_ms: number;
      min_latency_ms: number;
      max_latency_ms: number;
      fps_per_agent: number;
    }>;
    aggregate: {
      total_fps: number;
      total_forward_passes: number;
      wall_clock_ms: number;
    };
    contention_overhead_pct: number;
  }>,
  options: {
    agentCount?: number;
    tiers?: number[];
    warmupIterations?: number;
    benchmarkIterations?: number;
    baselineIterations?: number;
  } = {},
): Record<string, unknown> {
  const agentCount =
    typeof options.agentCount === 'number' && options.agentCount > 0
      ? options.agentCount
      : 6;
  const tiers = options.tiers ?? HIDDEN_NODE_TIERS;
  const warmupIterations =
    typeof options.warmupIterations === 'number'
      ? options.warmupIterations
      : DEFAULT_WARMUP_ITERATIONS;
  const benchmarkIterations =
    typeof options.benchmarkIterations === 'number'
      ? options.benchmarkIterations
      : DEFAULT_BENCHMARK_ITERATIONS;
  const baselineIterations =
    typeof options.baselineIterations === 'number'
      ? options.baselineIterations
      : 10;

  return {
    benchmark_type: 'parallel-single-window',
    success: true,
    generatedAt: new Date().toISOString(),
    reference_hardware: PARALLEL_REFERENCE_HARDWARE,
    browser_visibility: 'visible-foreground',
    environment: {
      userAgent: 'node',
      platform: 'node',
      gpuAdapterInfo: null,
      gpuProbedLimits: null,
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
      visibilityState: 'visible',
      outerWidth: 1920,
      outerHeight: 1080,
      hasFocus: true,
    },
    artifactFileName: 'artifacts/webgpu-throughput-parallel.json',
  };
}

describe('NGE tier benchmark helpers', () => {
  it('creates a GPU-dispatchable logistic activation wrapper', () => {
    const activation = makeGPULogisticActivation();

    expect(typeof activation).toBe('function');
    expect(
      (activation as unknown as Record<symbol, string | undefined>)[
        ACTIVATION_KEY_SYMBOL
      ],
    ).toBe('logisticActivation2');
    expect(activation.name).toBe('logisticActivation2');
  });

  it('builds a deterministic MLP with fixed weights and biases', () => {
    const network = buildDeterministicMLP(10, [64], 4);

    expect(network.nodes.length).toBe(78);
    expect(network.connections.length).toBe(896);

    expect(network.nodes[0].bias).toBe(0);
    expect(network.nodes[5].bias).toBe(0.005);
    expect(network.connections[0].weight).toBe(-0.09);
    expect(network.connections[10].weight).toBe(0.01);

    // All nodes with a squash function should use the GPU wrapper.
    const squashingNodes = network.nodes.filter(
      (node) => typeof node.squash === 'function',
    );
    for (const node of squashingNodes) {
      expect(
        (node.squash as unknown as Record<symbol, string | undefined>)[
          ACTIVATION_KEY_SYMBOL
        ],
      ).toBe('logisticActivation2');
    }
  });

  it('produces the deterministic benchmark input pattern', () => {
    const input = makeBenchmarkInput(INPUT_NODE_COUNT);

    expect(input).toEqual([
      0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.1, 0.2, 0.30000000000000004,
      0.4, 0.5,
    ]);
  });

  it('computes throughput from iterations and wall-clock time', () => {
    expect(computeThroughput(10, 100)).toBe(100);
    expect(computeThroughput(10, 0)).toBe(0);
    expect(computeThroughput(10, -5)).toBe(0);
    expect(computeThroughput(10, Number.NaN)).toBe(0);
  });

  it('detects timestamp-query support on a stub device', () => {
    expect(probeTimestampQuerySupport(null)).toBe(false);

    const withoutFeature = {
      features: {
        has: () => false,
      },
    } as unknown as GPUDevice;
    expect(probeTimestampQuerySupport(withoutFeature)).toBe(false);

    const withFeature = {
      features: {
        has: (name: string) => name === 'timestamp-query',
      },
    } as unknown as GPUDevice;
    expect(probeTimestampQuerySupport(withFeature)).toBe(true);
  });

  it('probes the expected GPU limit keys', () => {
    const device = {
      limits: {
        maxStorageBufferBindingSize: 1_073_741_824,
        maxBufferSize: 2_147_483_648,
        maxComputeInvocationsPerWorkgroup: 256,
        maxStorageBuffersPerShaderStage: 10,
        maxBindingsPerBindGroup: 16,
      },
    } as unknown as GPUDevice;

    const limits = probeGPULimits(device);
    expect(limits).not.toBeNull();
    expect(limits?.maxStorageBufferBindingSize).toBe(1_073_741_824);
    expect(limits?.maxStorageBuffersPerShaderStage).toBe(10);
    expect(limits?.maxComputeWorkgroupSizeX).toBeUndefined();
  });

  it('formats adapter info from the sync info property', async () => {
    const adapter = {
      info: {
        vendor: 'nvidia',
        architecture: 'ampere',
        device: 'RTX 4070',
        description: 'NVIDIA GeForce RTX 4070',
      },
    } as unknown as GPUAdapter;

    const info = await formatAdapterInfo(adapter);
    expect(info).toEqual({
      vendor: 'nvidia',
      architecture: 'ampere',
      device: 'RTX 4070',
      description: 'NVIDIA GeForce RTX 4070',
    });
  });

  it('assembles an artifact with reference hardware metadata', () => {
    const tierResults = [
      {
        hiddenNodes: 64,
        totalNodes: 78,
        throughputActivationsPerSecond: 1000,
        wallTimeMs: 10,
      },
      {
        hiddenNodes: 32768,
        totalNodes: 32782,
        throughputActivationsPerSecond: 50,
        wallTimeMs: 200,
      },
    ];

    const artifact = assembleArtifact(tierResults, {
      tiers: [64, 32768],
      timestampQuerySupported: true,
      gpuTimestampQueryNs: 128,
    });

    expect(artifact.schemaVersion).toBe('1.0.0');
    expect(artifact.environment).toEqual(
      expect.objectContaining({
        referenceHardware: REFERENCE_HARDWARE,
        timestampQuerySupported: true,
        gpuTimestampQueryNs: 128,
      }),
    );
    expect(artifact.configuration).toEqual(
      expect.objectContaining({
        hiddenNodeTiers: [64, 32768],
        inputNodeCount: INPUT_NODE_COUNT,
        outputNodeCount: OUTPUT_NODE_COUNT,
      }),
    );
    expect(artifact.summary).toEqual(
      expect.objectContaining({
        totalWallTimeMs: 210,
        tierCount: 2,
        minHiddenNodes: 64,
        maxHiddenNodes: 32768,
        averageThroughputActivationsPerSecond: 525,
      }),
    );
    expect(artifact.artifactFileName).toBe(
      'artifacts/webgpu-throughput-single.json',
    );
  });

  it('builds the full ladder of tier sizes', () => {
    for (const hiddenNodes of HIDDEN_NODE_TIERS) {
      const network = buildDeterministicMLP(
        INPUT_NODE_COUNT,
        [hiddenNodes],
        OUTPUT_NODE_COUNT,
      );
      expect(network.nodes.length).toBe(
        INPUT_NODE_COUNT + hiddenNodes + OUTPUT_NODE_COUNT,
      );
    }
  });

  it('computes latency distribution percentiles and stdDev', () => {
    const samples = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    const distribution = computeLatencyDistribution(samples);

    expect(distribution.min).toBe(10);
    expect(distribution.max).toBe(100);
    expect(distribution.mean).toBe(55);
    expect(distribution.median).toBe(55);
    expect(distribution.p50).toBe(55);
    expect(distribution.p95).toBeCloseTo(95.5, 10);
    expect(distribution.p99).toBeCloseTo(99.1, 10);
    expect(distribution.count).toBe(samples.length);
    expect(distribution.stdDev).toBeGreaterThan(0);
  });

  it('returns zeroed distribution for empty samples', () => {
    const distribution = computeLatencyDistribution([]);

    expect(distribution.min).toBe(0);
    expect(distribution.max).toBe(0);
    expect(distribution.mean).toBe(0);
    expect(distribution.median).toBe(0);
    expect(distribution.p50).toBe(0);
    expect(distribution.p95).toBe(0);
    expect(distribution.p99).toBe(0);
    expect(distribution.count).toBe(0);
    expect(distribution.stdDev).toBe(0);
  });

  it('ignores non-finite samples in latency distribution', () => {
    const distribution = computeLatencyDistribution([
      10,
      NaN,
      -5,
      20,
      Infinity,
    ]);

    expect(distribution.min).toBe(10);
    expect(distribution.max).toBe(20);
    expect(distribution.count).toBe(2);
    expect(distribution.mean).toBe(15);
  });

  it('computes per-agent stats from activation durations', () => {
    const perAgent = computePerAgentStats([10, 20, 30]);

    expect(perAgent.mean_latency_ms).toBe(20);
    expect(perAgent.min_latency_ms).toBe(10);
    expect(perAgent.max_latency_ms).toBe(30);
    expect(perAgent.fps_per_agent).toBe(50); // 1000 / 20
  });

  it('computes aggregate stats across multiple agents', () => {
    const agentResults = [
      { wallTimeMs: 100, perActivationMs: [10, 10, 10] },
      { wallTimeMs: 110, perActivationMs: [10, 10, 10] },
      { wallTimeMs: 105, perActivationMs: [10, 10, 10] },
    ];
    const aggregate = computeAggregateStats(agentResults, 3);

    expect(aggregate.total_forward_passes).toBe(9);
    expect(aggregate.wall_clock_ms).toBe(110);
    expect(aggregate.total_fps).toBeCloseTo(computeThroughput(9, 110), 10);
  });

  it('computes contention overhead against ideal linear scaling', () => {
    const singleAgentFps = 100;
    const aggregateFps = 280;
    const agentCount = 6;

    const overhead = computeContentionOverheadPct(
      singleAgentFps,
      aggregateFps,
      agentCount,
    );

    const idealFps = singleAgentFps * agentCount;
    const expected = ((idealFps - aggregateFps) / idealFps) * 100;
    expect(overhead).toBeCloseTo(expected, 10);
  });

  it('returns zero contention overhead for invalid inputs', () => {
    expect(computeContentionOverheadPct(0, 100, 6)).toBe(0);
    expect(computeContentionOverheadPct(100, 100, 0)).toBe(0);
    expect(computeContentionOverheadPct(NaN, 100, 6)).toBe(0);
  });

  it('caps negative contention overhead at zero', () => {
    // Aggregate faster than ideal should report 0 overhead.
    expect(computeContentionOverheadPct(100, 700, 6)).toBe(0);
  });

  it('assembles a single-window parallel benchmark artifact', () => {
    const perAgent = computePerAgentStats([10, 10, 10]);
    const agentResults = [
      { wallTimeMs: 30, perActivationMs: [10, 10, 10] },
      { wallTimeMs: 30, perActivationMs: [10, 10, 10] },
    ];
    const aggregate = computeAggregateStats(agentResults, 3);
    const contention = computeContentionOverheadPct(
      perAgent.fps_per_agent,
      aggregate.total_fps,
      2,
    );

    const artifact = assembleParallelArtifact(
      [
        {
          tier: 8,
          agent_count: 2,
          per_agent: [perAgent],
          aggregate,
          contention_overhead_pct: contention,
        },
      ],
      { agentCount: 2, tiers: [8], benchmarkIterations: 3 },
    );

    expect(artifact.benchmark_type).toBe('parallel-single-window');
    expect(artifact.success).toBe(true);
    expect(artifact.reference_hardware).toEqual(PARALLEL_REFERENCE_HARDWARE);
    expect(artifact.browser_visibility).toBe('visible-foreground');
    expect((artifact.configuration as Record<string, unknown>).agentCount).toBe(
      2,
    );
    expect(
      (artifact.configuration as Record<string, unknown>).benchmarkIterations,
    ).toBe(3);
    expect(
      (artifact.visibility_check as Record<string, unknown>).hasFocus,
    ).toBe(true);
    expect(artifact.artifactFileName).toBe(
      'artifacts/webgpu-throughput-parallel.json',
    );

    const tierResult = (
      artifact.tier_results as Array<Record<string, unknown>>
    )[0];
    expect(tierResult.tier).toBe(8);
    expect(tierResult.agent_count).toBe(2);
    expect(tierResult.contention_overhead_pct).toBeGreaterThanOrEqual(0);
  });

  describe('network.gpu.profiling', () => {
    function makeFakeProfilingResult(
      overrides: Partial<ProfilingResult> = {},
    ): ProfilingResult {
      const phases: ProfilingPhaseTiming[] = overrides.phases ?? [
        { name: 'cpuPreparation' as const, ms: 10, pct: 10 },
        { name: 'pipeline' as const, ms: 30, pct: 30 },
        { name: 'gpuCompletionWait' as const, ms: 60, pct: 60 },
      ];

      return {
        success: true,
        totalForwardPassMs: 100,
        cpuPreparationMs: 10,
        cpuTopoSortMs: 1,
        cpuCSRBuildMs: 1,
        cpuConnectionSlabBuildMs: 5,
        cpuTopologyHashMs: 1,
        bufferUploadMs: 5,
        pipelineMs: 30,
        bindGroupMs: 5,
        dynamicBufferUploadMs: 5,
        queueSubmissionMs: 5,
        gpuCompletionWaitMs: 60,
        outputReadbackMs: 5,
        phases,
        dominantBottleneck: 'gpuCompletionWait',
        overheadRatio: 0.4,
        output: new Float32Array([0.1, 0.2, 0.3, 0.4]),
        ...overrides,
      };
    }

    it('accumulates timer durations across multiple intervals', () => {
      const timer = new GpuProfilingTimer();

      timer.start('bufferUpload');
      timer.stop('bufferUpload');
      const first = timer.get('bufferUpload');

      timer.start('bufferUpload');
      timer.stop('bufferUpload');
      const second = timer.get('bufferUpload');

      expect(second).toBeGreaterThanOrEqual(first);
    });

    it('returns zero for unknown timer phases', () => {
      const timer = new GpuProfilingTimer();
      expect(timer.get('unknownPhase')).toBe(0);
    });

    it('stops an unknown phase without throwing', () => {
      const timer = new GpuProfilingTimer();
      expect(timer.stop('unknownPhase')).toBe(0);
    });

    it('computes overhead breakdown percentages sorted by impact', () => {
      const timings = {
        cpuPreparation: 30,
        gpuCompletionWait: 50,
        outputReadback: 20,
      };
      const breakdown = computeOverheadBreakdown(timings);

      expect(breakdown[0].name).toBe('gpuCompletionWait');
      expect(breakdown[0].pct).toBeCloseTo(50, 5);
      expect(breakdown[1].name).toBe('cpuPreparation');
      expect(breakdown[1].pct).toBeCloseTo(30, 5);
      expect(breakdown[2].name).toBe('outputReadback');
      expect(breakdown[2].pct).toBeCloseTo(20, 5);
    });

    it('ranks weak points by descending impact and attaches strategies', () => {
      const phases: ProfilingPhaseTiming[] = [
        { name: 'bufferUpload' as const, ms: 10, pct: 10 },
        { name: 'pipeline' as const, ms: 60, pct: 60 },
        { name: 'cpuPreparation' as const, ms: 30, pct: 30 },
      ];
      const ranked = rankWeakPoints(phases);

      expect(ranked[0].name).toBe('pipeline');
      expect(ranked[0].impactPct).toBe(60);
      expect(ranked[0].strategy).toContain('pipeline');
      expect(ranked[1].name).toBe('cpuPreparation');
      expect(ranked[2].name).toBe('bufferUpload');
    });

    it('builds an overhead artifact with all required sections', () => {
      const result = makeFakeProfilingResult();
      const artifact = buildOverheadArtifact([result], {
        tiers: [64],
        inputCount: 10,
        outputCount: 4,
      }) as Record<string, unknown>;
      const summary = artifact.summary as Record<string, unknown>;
      const configuration = artifact.configuration as Record<string, unknown>;

      expect(artifact.schemaVersion).toBe('1.0.0');
      expect(configuration).toEqual({
        hiddenNodeTiers: [64],
        inputNodeCount: 10,
        outputNodeCount: 4,
      });
      expect(summary.averageOverheadRatio).toBe(0.4);
      expect(summary.tierCount).toBe(1);
      expect(summary.successfulTierCount).toBe(1);
      expect(artifact.artifactFileName).toBe(
        'artifacts/webgpu-overhead-breakdown.json',
      );

      const tierResults = artifact.tier_results as Array<
        Record<string, unknown>
      >;
      expect(tierResults).toHaveLength(1);
      expect(tierResults[0].hiddenNodes).toBe(64);
      expect(tierResults[0].dominantBottleneck).toBe('gpuCompletionWait');
      expect(tierResults[0].overheadRatio).toBe(0.4);
      expect(Array.isArray(tierResults[0].weakPoints)).toBe(true);
      expect(artifact.ranked_weak_points).toBeDefined();
    });

    it('profiles a GPU activation on a mock device', async () => {
      const network = buildDeterministicMLP(2, [3], 1);
      const device = createMockGPUDevice();

      const result = await profileGPUActivation(device, network, [0.1, 0.2]);

      expect(result.success).toBe(true);
      expect(result.output).toBeDefined();
      expect(result.output).toHaveLength(1);
      expect(result.phases).toHaveLength(8);
      expect(result.phases.every((phase) => phase.ms >= 0)).toBe(true);
      expect(result.phases.every((phase) => phase.pct >= 0)).toBe(true);
      expect(result.dominantBottleneck).not.toBe('unknown');
      expect(result.overheadRatio).toBeGreaterThanOrEqual(0);
      expect(result.totalForwardPassMs).toBeGreaterThanOrEqual(0);
    });

    it('reports failure for an ineligible network', async () => {
      const network = buildDeterministicMLP(2, [3], 1);
      const result = await profileGPUActivation(
        null as unknown as GPUDevice,
        network,
        [0.1, 0.2],
      );

      expect(result.success).toBe(false);
      expect(result.error).toContain('not eligible');
    });

    it('resets accumulated timer durations', () => {
      const timer = new GpuProfilingTimer();
      timer.start('pipeline');
      timer.stop('pipeline');
      timer.reset();

      expect(timer.get('pipeline')).toBe(0);
    });

    it('identifies the dominant bottleneck from the highest phase share', () => {
      const result = makeFakeProfilingResult({
        phases: [
          { name: 'cpuPreparation' as const, ms: 30, pct: 30 },
          { name: 'pipeline' as const, ms: 50, pct: 50 },
          { name: 'gpuCompletionWait' as const, ms: 20, pct: 20 },
        ],
      });

      expect(identifyBottleneck(result)).toBe('pipeline');
    });

    it('returns unknown for a failed profiling result', () => {
      const result = makeFakeProfilingResult({ success: false });

      expect(identifyBottleneck(result)).toBe('unknown');
    });

    it('returns unknown when there are no phases', () => {
      const result = makeFakeProfilingResult({ phases: [] });

      expect(identifyBottleneck(result)).toBe('unknown');
    });

    it('treats non-finite timings as zero in the breakdown', () => {
      const breakdown = computeOverheadBreakdown({
        cpuPreparation: 10,
        pipeline: NaN,
        gpuCompletionWait: 30,
      });

      expect(
        breakdown.map(({ name, ms, pct }) => ({
          name,
          ms,
          pct: Math.round(pct),
        })),
      ).toEqual([
        { name: 'gpuCompletionWait', ms: 30, pct: 75 },
        { name: 'cpuPreparation', ms: 10, pct: 25 },
        { name: 'pipeline', ms: 0, pct: 0 },
      ]);
    });

    it('drops zero-duration phases from the weak-point ranking', () => {
      const ranked = rankWeakPoints([
        { name: 'pipeline' as const, ms: 10, pct: 50 },
        { name: 'bindGroup' as const, ms: 0, pct: 0 },
      ]);

      expect(ranked.map((point) => point.name)).toEqual(['pipeline']);
    });

    it('drops non-finite phase shares from the weak-point ranking', () => {
      const ranked = rankWeakPoints([
        { name: 'pipeline' as const, ms: 10, pct: NaN },
        { name: 'bufferUpload' as const, ms: 5, pct: 50 },
      ]);

      expect(ranked.map((point) => point.name)).toEqual(['bufferUpload']);
    });

    it('builds an artifact with no tier results', () => {
      const artifact = buildOverheadArtifact([], {});

      expect(
        (artifact.summary as Record<string, unknown>).successfulTierCount,
      ).toBe(0);
    });

    it('marks the successful-tier count as zero for failed results', () => {
      const result = makeFakeProfilingResult({ success: false });
      const artifact = buildOverheadArtifact([result], {
        tiers: [64],
      }) as Record<string, unknown>;

      expect(
        (artifact.summary as Record<string, unknown>).successfulTierCount,
      ).toBe(0);
    });

    it('includes ranked weak points, strategies, and ceiling in the artifact', () => {
      const artifact = buildOverheadArtifact([makeFakeProfilingResult()], {
        tiers: [64],
      }) as Record<string, unknown>;

      expect(
        Array.isArray(artifact.ranked_weak_points) &&
          Array.isArray(artifact.strategies) &&
          artifact.true_ceiling !== undefined,
      ).toBe(true);
    });

    it('builds an artifact with a custom visibility label', () => {
      const artifact = buildOverheadArtifact([], {
        browserVisibility: 'hidden-background',
      }) as Record<string, unknown>;

      expect(artifact.browser_visibility).toBe('hidden-background');
    });

    it('prepares a valid activation context for a GPU-eligible network', () => {
      const network = buildDeterministicMLP(2, [3], 1);
      const context = prepareActivationContext(network);
      context.restore();

      expect(context.index).toBe(0);
    });

    it('throws when the first node has no squash function', () => {
      const network = buildDeterministicMLP(2, [3], 1);
      network.nodes[0].squash = undefined as unknown as (
        x: number,
        derivate?: boolean,
      ) => number;

      expect(() => prepareActivationContext(network)).toThrow(
        'no squash function',
      );
    });

    it('throws when the first node uses an unsupported activation', () => {
      const network = buildDeterministicMLP(2, [3], 1);
      function unknownActivation(x: number): number {
        return x * 99999 + 0.12345;
      }
      network.nodes[0].squash = unknownActivation as (
        x: number,
        derivate?: boolean,
      ) => number;

      expect(() => prepareActivationContext(network)).toThrow(
        'not in the worker registry',
      );
    });

    it('reports failure when the input vector length mismatches', async () => {
      const network = buildDeterministicMLP(2, [3], 1);
      const result = await profileGPUActivation(
        createMockGPUDevice(),
        network,
        [0.1],
      );

      expect(result.success).toBe(false);
    });

    it('reports failure when the network has no nodes', async () => {
      const network = buildDeterministicMLP(2, [3], 1);
      network.nodes.length = 0;
      const result = await profileGPUActivation(
        createMockGPUDevice(),
        network,
        [0.1, 0.2],
      );

      expect(result.success).toBe(false);
    });

    it('accepts a Float32Array input vector', async () => {
      const network = buildDeterministicMLP(2, [3], 1);
      const result = await profileGPUActivation(
        createMockGPUDevice(),
        network,
        new Float32Array([0.1, 0.2]),
      );

      expect(result.success).toBe(true);
    });

    it('uses default options when no options are provided', () => {
      const artifact = buildOverheadArtifact([]) as Record<string, unknown>;

      expect(artifact.schemaVersion).toBe('1.0.0');
    });

    it('falls back to zero hidden nodes when tier results exceed the tier list', () => {
      const artifact = buildOverheadArtifact(
        [makeFakeProfilingResult(), makeFakeProfilingResult()],
        { tiers: [64] },
      ) as Record<string, unknown>;
      const tierResults = artifact.tier_results as Array<
        Record<string, unknown>
      >;

      expect(tierResults[1].hiddenNodes).toBe(0);
    });

    it('counts total nodes as zero when the result has no output', () => {
      const artifact = buildOverheadArtifact(
        [makeFakeProfilingResult({ output: undefined })],
        { tiers: [64] },
      ) as Record<string, unknown>;
      const tierResults = artifact.tier_results as Array<
        Record<string, unknown>
      >;

      expect(tierResults[0].totalNodes).toBe(0);
    });

    it('sorts ranked weak points by dominance frequency across tiers', () => {
      const resultA = makeFakeProfilingResult({
        dominantBottleneck: 'pipeline',
      });
      const resultB = makeFakeProfilingResult({
        dominantBottleneck: 'bufferUpload',
      });
      const artifact = buildOverheadArtifact([resultA, resultB], {
        tiers: [64, 256],
      }) as Record<string, unknown>;
      const ranked = artifact.ranked_weak_points as Array<
        Record<string, unknown>
      >;

      expect(ranked[0].name).toBe('pipeline');
    });

    it('labels the environment as node when navigator is unavailable', () => {
      const originalNavigator = (
        globalThis as unknown as Record<string, unknown>
      ).navigator;
      (globalThis as unknown as Record<string, unknown>).navigator = undefined;

      try {
        const artifact = buildOverheadArtifact([]) as Record<string, unknown>;
        const environment = artifact.environment as Record<string, unknown>;

        expect(environment).toEqual(
          expect.objectContaining({
            userAgent: 'node',
            platform: 'node',
          }),
        );
      } finally {
        (globalThis as unknown as Record<string, unknown>).navigator =
          originalNavigator;
      }
    });

    it('reports zero overhead ratio when total forward-pass time is zero', async () => {
      const network = buildDeterministicMLP(2, [3], 1);
      const nowSpy = jest.spyOn(performance, 'now').mockReturnValue(0);

      try {
        const result = await profileGPUActivation(
          createMockGPUDevice(),
          network,
          [0.1, 0.2],
        );

        expect(result.overheadRatio).toBe(0);
      } finally {
        nowSpy.mockRestore();
      }
    });
  });
});
