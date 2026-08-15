import {
  Network,
  methods,
  buildOverheadArtifact,
  profileGPUActivation,
  rankWeakPoints,
} from '../../../dist/neataptic.browser.esm.js';

import {
  HIDDEN_NODE_TIERS,
  INPUT_NODE_COUNT,
  OUTPUT_NODE_COUNT,
  makeBenchmarkInput,
  probeGPULimits,
  formatAdapterInfo,
} from './webgpu-nge-tier-throughput.mjs';

/**
 * Default artifact filename used by the browser overhead-breakdown download.
 */
export const OVERHEAD_ARTIFACT_FILE_NAME = 'artifacts/webgpu-overhead-breakdown.json';

/**
 * Reference hardware metadata embedded in the overhead-breakdown artifact.
 */
const OVERHEAD_REFERENCE_HARDWARE = {
  processor: 'Intel i7-10700 @ 2.90GHz, 8 Cores/16 Logical',
  memory: '32GB DDR4 3200MHz',
  os: 'Windows 11 Home Build 26200',
  gpu_vendor: 'nvidia',
  gpu_architecture: 'lovelace',
  maxStorageBuffersPerShaderStage: 8,
};

/**
 * Activation registry symbol used by the WebGPU kernel to resolve the
 * logistic squash across module/bundle boundaries.
 *
 * The bundled activation function must carry this symbol so the profiling
 * path can resolve it to the worker-registry index before compiling the
 * kernel.
 */
const ACTIVATION_KEY_SYMBOL = Symbol.for('neataptic.activation.key');

/**
 * Wrap the built-in logistic activation with a stable worker-registry key.
 *
 * This mirrors the helper in the tier throughput scenario so the two
 * benchmarks share the same GPU-dispatchable activation.
 *
 * @returns {Function} A GPU-dispatchable logistic activation function.
 */
function makeGPULogisticActivation() {
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
 * Build a deterministic, single-hidden-layer MLP for overhead profiling.
 *
 * Same contract as the tier throughput helper, but guarantees the wrapped
 * logistic activation is used for every node that already has a squash.
 *
 * @param {number} inputCount - Number of input nodes.
 * @param {number[]} hiddenCounts - Hidden-layer node counts.
 * @param {number} outputCount - Number of output nodes.
 * @returns {Network} A deterministic MLP eligible for the GPU fast path.
 */
export function buildDeterministicOverheadMLP(
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
 * Run the single-window GPU overhead breakdown across the canonical tier ladder.
 *
 * For each hidden-layer size, a fresh deterministic MLP is built and profiled
 * with `profileGPUActivation` using the cold-path instrumentation. Results are
 * aggregated into a downloadable artifact with per-tier timings, bottleneck
 * ranking, overhead ratios, and strategy recommendations.
 *
 * @param {object} [options={}] - Benchmark options.
 * @param {GPUDevice|null} [options.device=null] - WebGPU device to use.
 * @param {GPUAdapter|null} [options.adapter=null] - WebGPU adapter for info.
 * @param {number[]} [options.tiers=HIDDEN_NODE_TIERS] - Hidden-layer sizes.
 * @param {number} [options.inputCount=INPUT_NODE_COUNT] - Input nodes per tier.
 * @param {number} [options.outputCount=OUTPUT_NODE_COUNT] - Output nodes per tier.
 * @returns {Promise<Record<string, unknown>>} Overhead artifact.
 */
export async function runOverheadBreakdown(options = {}) {
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

  const tierResults = [];

  for (const hiddenNodes of tiers) {
    const network = buildDeterministicOverheadMLP(
      inputCount,
      [hiddenNodes],
      outputCount,
    );

    const input = makeBenchmarkInput(inputCount);

    const profile = await profileGPUActivation(device, network, input);

    const weakPoints = rankWeakPoints(profile.phases);
    const dominant = weakPoints[0] ?? { name: 'unknown', impactPct: 0 };

    tierResults.push({
      hiddenNodes,
      totalNodes: network.nodes.length,
      totalConnections: network.connections.length,
      inputNodes: inputCount,
      outputNodes: outputCount,
      success: profile.success,
      error: profile.error ?? null,
      totalForwardPassMs: profile.totalForwardPassMs,
      overheadRatio: profile.overheadRatio,
      dominantBottleneck: profile.dominantBottleneck,
      dominantImpactPct: dominant.impactPct,
      phases: profile.phases,
      weakPoints,
      output: profile.success && profile.output
        ? Array.from(profile.output)
        : null,
    });
  }

  const maxStorageBuffersPerShaderStage =
    device && typeof device.limits === 'object' && device.limits !== null
      ? (device.limits.maxStorageBuffersPerShaderStage ?? 8)
      : 8;

  const isVisible =
    typeof document !== 'undefined' &&
    typeof window !== 'undefined' &&
    document.visibilityState === 'visible' &&
    window.outerWidth > 0 &&
    window.outerHeight > 0;
  const browserVisibility = isVisible
    ? 'visible-foreground'
    : 'hidden-background';

  const artifact = buildOverheadArtifact(tierResults, {
    tiers,
    inputCount,
    outputCount,
    gpuAdapterInfo: formatAdapterInfo(adapter),
    gpuProbedLimits: probeGPULimits(device),
    referenceHardware: {
      ...OVERHEAD_REFERENCE_HARDWARE,
      maxStorageBuffersPerShaderStage,
    },
    browserVisibility,
    timestampQuerySupported: false,
  });

  return artifact;
}
