/**
 * WebGPU activation overhead profiling instrumentation.
 *
 * This module provides a self-contained, timer-instrumented version of the GPU
 * forward pass that measures where wall-clock time is spent without mutating the
 * production `activateGPU` path. It re-uses the same buffer-packing and kernel
 * helpers as the fast path so the numbers reflect real costs, but it creates
 * fresh GPU state per profile call to capture cold-path overheads such as buffer
 * allocation, pipeline compilation, and bind-group creation.
 *
 * The intended caller is the browser overhead-breakdown scenario
 * (`docs/browser-tests/scenarios/webgpu-overhead-breakdown.mjs`). The module
 * also exports pure artifact-assembly helpers so the Node test suite can verify
 * percentage math, bottleneck ranking, and strategy generation.
 *
 * @see [WebGPU Performance Guide](../../../docs/webgpu-performance-guide.md) for
 * the measured overhead breakdown and optimization narrative that uses these
 * profiling primitives.
 *
 * @module
 */

import type Network from '../network';
import { canUseGPU } from './network.gpu.capability';
import {
  buildConnectionsArray,
  buildIncomingCSR,
  buildNodesArray,
  buildTopoLevels,
  computeTopoLevelCount,
  createGPUBuffer,
  createGPUUniformBuffer,
  destroyGPUBufferSet,
  GPU_NODE_STRUCT_BYTES,
  uploadDynamicNetworkBuffers,
  writeInputValuesToNodeStruct,
  type GPUBufferSet,
} from './network.gpu.buffer';
import {
  compileActivationKernel,
  createBindGroupLayout,
  SUPPORTED_ACTIVATION_INDICES,
} from './network.gpu.kernel';
import { resolveActivationIndex } from './network.gpu.activate';
import { GPU_BUFFER_BINDING } from './network.gpu.types';

/**
 * Temporarily annotate the first node's squash with its worker-registry index
 * so `compileActivationKernel` can generate the correct WGSL switch, then restore
 * the original value.
 *
 * @param network - Network whose first node squash will be temporarily annotated.
 * @returns A context object with the resolved index and a `restore()` callback.
 * @throws Error when the first node has no squash or it is not a built-in worker
 *   activation.
 */
export function prepareActivationContext(network: Network): {
  index: number;
  restore: () => void;
} {
  const firstNode = network.nodes[0];
  const squash = firstNode.squash as
    ((value: number, derivate?: boolean) => number) | undefined;

  if (!squash) {
    throw new Error('profileGPUActivation: first node has no squash function');
  }

  const squashWithIndex = squash as {
    index?: number;
  } & ((value: number, derivate?: boolean) => number);
  const savedIndex = squashWithIndex.index;
  const index = resolveActivationIndex(squash);

  if (index === undefined) {
    throw new Error(
      'profileGPUActivation: first node uses an activation that is not in the worker registry',
    );
  }

  squashWithIndex.index = index;

  return {
    index,
    restore: () => {
      squashWithIndex.index = savedIndex;
    },
  };
}

/** Number of threads per compute workgroup for the activation kernel. */
const ACTIVATION_WORKGROUP_SIZE = 64;

/** WebGPU buffer usage flag for mappable readback buffers. */
const GPU_BUFFER_USAGE_MAP_READ = 0x0001;

/** WebGPU buffer usage flag for copy sources. */
const GPU_BUFFER_USAGE_COPY_SRC = 0x0004;

/** WebGPU buffer usage flag for copy destinations. */
const GPU_BUFFER_USAGE_COPY_DST = 0x0008;

/** WebGPU map mode for reading mapped buffers. */
const GPU_MAP_MODE_READ = 0x0001;

/** Byte size of the per-dispatch params uniform. */
const GPU_PARAMS_BYTES = 16;

/** Names of the canonical overhead phases measured by the profiler. */
export const PROFILING_PHASE_NAMES = [
  'cpuPreparation',
  'bufferUpload',
  'pipeline',
  'bindGroup',
  'dynamicBufferUpload',
  'queueSubmission',
  'gpuCompletionWait',
  'outputReadback',
] as const;

/** Canonical phase name used by the profiler. */
export type ProfilingPhaseName = (typeof PROFILING_PHASE_NAMES)[number];

/**
 * Timings for one overhead phase, including its share of the total forward pass.
 */
export interface ProfilingPhaseTiming {
  /** Canonical phase name. */
  name: ProfilingPhaseName;
  /** Wall-clock milliseconds spent in this phase. */
  ms: number;
  /** Percentage of the total forward pass time consumed by this phase. */
  pct: number;
}

/**
 * Result of a single instrumented GPU forward pass.
 *
 * The detailed CPU-prep sub-timers sum to `cpuPreparationMs`. The remaining
 * phases sum to `totalForwardPassMs`. `overheadRatio` reports the share of the
 * total time consumed by everything except the GPU compute wait, which is the
 * closest proxy for "useful" GPU work when timestamp queries are unavailable.
 */
export interface ProfilingResult {
  /** True when every phase completed and an output was read back. */
  success: boolean;
  /** Error message when `success` is false. */
  error?: string;
  /** Output vector produced by the forward pass, when successful. */
  output?: Float32Array;
  /** Total wall-clock milliseconds for the full instrumented activation. */
  totalForwardPassMs: number;
  /** Sum of all CPU-side preparation sub-timers. */
  cpuPreparationMs: number;
  /** Time to compute topological levels via Kahn sort. */
  cpuTopoSortMs: number;
  /** Time to build the incoming-CSR start-offset array. */
  cpuCSRBuildMs: number;
  /** Time to pack connection and node struct arrays. */
  cpuConnectionSlabBuildMs: number;
  /** Time to compute the deterministic topology hash. */
  cpuTopologyHashMs: number;
  /** Time to create the six GPU buffers and upload static topology data. */
  bufferUploadMs: number;
  /** Time to compile or retrieve the activation compute pipeline. */
  pipelineMs: number;
  /** Time to create the kernel bind group. */
  bindGroupMs: number;
  /** Time to re-upload mutable weights/biases and scatter input values. */
  dynamicBufferUploadMs: number;
  /** Time spent in `device.queue.submit()` calls across all levels. */
  queueSubmissionMs: number;
  /** Time spent awaiting `device.queue.onSubmittedWorkDone()` across all levels. */
  gpuCompletionWaitMs: number;
  /** Time to copy output nodes back to a staging buffer and read them. */
  outputReadbackMs: number;
  /** All measured phases with percentage breakdown. */
  phases: ProfilingPhaseTiming[];
  /** Name of the phase with the largest share of the total time. */
  dominantBottleneck: string;
  /** (total overhead) / (total forward pass time), where overhead excludes GPU wait. */
  overheadRatio: number;
}

/**
 * Simple high-resolution timer for named overhead phases.
 *
 * Uses `performance.now()` so the same instrumentation works in the browser
 * and in Node test environments. A phase may be started and stopped multiple
 * times; reported durations are accumulated.
 *
 * @example
 * ```ts
 * const timer = new GpuProfilingTimer();
 * timer.start('bufferUpload');
 * // ... GPU upload work ...
 * const ms = timer.stop('bufferUpload');
 * ```
 */
export class GpuProfilingTimer {
  /** In-flight start marks keyed by phase name. */
  private marks = new Map<string, number>();

  /** Accumulated durations keyed by phase name. */
  private durations = new Map<string, number>();

  /**
   * Record the start time for a named phase.
   *
   * @param name - Phase identifier.
   */
  start(name: string): void {
    this.marks.set(name, performance.now());
  }

  /**
   * Stop a phase and return the elapsed milliseconds.
   *
   * If the phase was never started, returns `0` and records nothing.
   *
   * @param name - Phase identifier that was previously passed to `start`.
   * @returns Accumulated milliseconds for the phase, including this interval.
   */
  stop(name: string): number {
    const start = this.marks.get(name);
    if (start === undefined) {
      return 0;
    }
    const delta = performance.now() - start;
    this.durations.set(name, (this.durations.get(name) ?? 0) + delta);
    this.marks.delete(name);
    return this.durations.get(name) as number;
  }

  /**
   * Return the accumulated milliseconds for a phase.
   *
   * @param name - Phase identifier.
   * @returns Accumulated milliseconds, or `0` when the phase was never timed.
   */
  get(name: string): number {
    return this.durations.get(name) ?? 0;
  }

  /** Clear all marks and accumulated durations. */
  reset(): void {
    this.marks.clear();
    this.durations.clear();
  }
}

/**
 * Compute a deterministic topology hash for a network.
 *
 * Mirrors the hash used by the production GPU cache so profiles and normal
 * activations agree on whether two networks share a topology.
 *
 * @param network - Network whose topology will be hashed.
 * @returns Stable hash string.
 */
function computeTopologyHash(network: Network): string {
  const nodeCount = network.nodes.length;
  const connections = network.connections;

  let hash = nodeCount;
  for (let index = 0; index < connections.length; index++) {
    const connection = connections[index];
    hash = ((hash << 5) - hash + connection.from.index!) | 0;
    hash = ((hash << 5) - hash + connection.to.index!) | 0;
  }

  return `${connections.length}:${hash}`;
}

/**
 * Build the bind group that wires the six struct-packed kernel buffers into the
 * pipeline layout.
 *
 * This is a local mirror of the production bind-group creation so the profiler
 * can time it independently.
 *
 * @param device - WebGPU device that will own the bind group.
 * @param layout - Bind-group layout created by `createBindGroupLayout`.
 * @param bufferSet - Uploaded slab buffers.
 * @returns A fresh bind group for the activation kernel.
 */
function createActivationBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  bufferSet: GPUBufferSet,
): GPUBindGroup {
  return device.createBindGroup({
    layout,
    entries: [
      {
        binding: GPU_BUFFER_BINDING.connections,
        resource: { buffer: bufferSet.connections },
      },
      {
        binding: GPU_BUFFER_BINDING.nodes,
        resource: { buffer: bufferSet.nodes },
      },
      {
        binding: GPU_BUFFER_BINDING.outputs,
        resource: { buffer: bufferSet.outputs },
      },
      {
        binding: GPU_BUFFER_BINDING.params,
        resource: { buffer: bufferSet.params },
      },
      {
        binding: GPU_BUFFER_BINDING.topoLevels,
        resource: { buffer: bufferSet.topoLevels },
      },
      {
        binding: GPU_BUFFER_BINDING.inStart,
        resource: { buffer: bufferSet.inStart },
      },
    ],
  });
}

/**
 * Local extension of the ambient GPU command encoder so we can copy a storage
 * buffer to a mappable staging buffer.
 */
interface GPUCommandEncoderCopy extends GPUCommandEncoder {
  copyBufferToBuffer(
    sourceBuffer: GPUBuffer,
    sourceOffset: number,
    destinationBuffer: GPUBuffer,
    destinationOffset: number,
    size: number,
  ): void;
}

/**
 * Copy the output-node slice of the GPU output buffer to a mappable staging
 * buffer, await the mapping, and return a detached Float32Array copy.
 *
 * @param device - WebGPU device that owns the buffers.
 * @param network - Network whose output nodes will be read.
 * @param bufferSet - Uploaded slab buffers.
 * @returns Detached Float32Array of output-node values.
 */
async function readOutputValues(
  device: GPUDevice,
  network: Network,
  bufferSet: GPUBufferSet,
): Promise<Float32Array> {
  const outputNodeCount = network.output;
  const outputByteLength = outputNodeCount * Float32Array.BYTES_PER_ELEMENT;
  const outputStartOffset =
    (bufferSet.nodeCount - outputNodeCount) * Float32Array.BYTES_PER_ELEMENT;

  const stagingBuffer = device.createBuffer({
    label: 'network_outputs_staging',
    size: outputByteLength,
    usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
  });

  const copyEncoder = device.createCommandEncoder({
    label: 'network_outputs_copy',
  }) as unknown as GPUCommandEncoderCopy;
  copyEncoder.copyBufferToBuffer(
    bufferSet.outputs,
    outputStartOffset,
    stagingBuffer,
    0,
    outputByteLength,
  );
  device.queue.submit([copyEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  await stagingBuffer.mapAsync(GPU_MAP_MODE_READ);
  const mappedRange = stagingBuffer.getMappedRange();
  const output = new Float32Array(mappedRange.slice(0));
  stagingBuffer.unmap();
  stagingBuffer.destroy();

  return output;
}

/**
 * Dispatch the activation kernel once per topological level while timing
 * `device.queue.submit()` and `device.queue.onSubmittedWorkDone()` separately.
 *
 * @param device - WebGPU device used to dispatch.
 * @param bufferSet - Uploaded slab buffers.
 * @param pipeline - Compiled activation compute pipeline.
 * @param bindGroup - Bind group wiring the kernel buffers.
 * @param outputNodeCount - Number of output nodes in the network.
 * @param timer - Profiler timer updated with submission and wait durations.
 */
async function dispatchActivationKernel(
  device: GPUDevice,
  bufferSet: GPUBufferSet,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  outputNodeCount: number,
  timer: GpuProfilingTimer,
): Promise<void> {
  const levelCount = bufferSet.topoLevelCount;
  const workgroupCount = Math.ceil(
    bufferSet.nodeCount / ACTIVATION_WORKGROUP_SIZE,
  );
  const params = new Uint32Array(4);
  params[1] = bufferSet.nodeCount;
  params[2] = bufferSet.connectionCount;
  params[3] = bufferSet.nodeCount - outputNodeCount;

  for (let level = 1; level < levelCount; level++) {
    params[0] = level;
    device.queue.writeBuffer(bufferSet.params, 0, params);

    const commandEncoder = device.createCommandEncoder({
      label: `network_activation_level_${level}`,
    });
    const pass = commandEncoder.beginComputePass({
      label: `network_activation_pass_level_${level}`,
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();

    timer.start('queueSubmission');
    device.queue.submit([commandEncoder.finish()]);
    timer.stop('queueSubmission');

    timer.start('gpuCompletionWait');
    await device.queue.onSubmittedWorkDone();
    timer.stop('gpuCompletionWait');
  }
}

/**
 * Create the six struct-packed GPU buffers for a network and upload static
 * topology data.
 *
 * @param device - WebGPU device that will own the buffers.
 * @param network - Network whose slab arrays will be uploaded.
 * @param connectionsArray - Packed connection struct array.
 * @param nodesArray - Packed node struct array.
 * @param inStart - Incoming-CSR start offsets.
 * @param topoLevels - Per-node topological levels.
 * @returns A fully uploaded `GPUBufferSet`.
 */
function createAndUploadBuffers(
  device: GPUDevice,
  network: Network,
  connectionsArray: ArrayBuffer,
  nodesArray: ArrayBuffer,
  inStart: Uint32Array,
  topoLevels: Uint32Array,
): GPUBufferSet {
  const nodeCount = network.nodes.length;
  const connectionCount = network.connections.length;

  const connectionsBuffer = createGPUBuffer(
    device,
    connectionsArray.byteLength,
    'network_connections',
  );
  const nodesBuffer = createGPUBuffer(
    device,
    nodesArray.byteLength,
    'network_nodes',
  );
  const outputsBuffer = createGPUBuffer(
    device,
    nodeCount * Float32Array.BYTES_PER_ELEMENT,
    'network_outputs',
    GPU_BUFFER_USAGE_COPY_SRC,
  );
  const paramsBuffer = createGPUUniformBuffer(
    device,
    GPU_PARAMS_BYTES,
    'network_params',
  );
  const topoLevelsBuffer = createGPUBuffer(
    device,
    topoLevels.byteLength,
    'network_topo_levels',
  );
  const inStartBuffer = createGPUBuffer(
    device,
    inStart.byteLength,
    'network_in_start',
  );

  device.queue.writeBuffer(connectionsBuffer, 0, connectionsArray);
  device.queue.writeBuffer(nodesBuffer, 0, nodesArray);
  device.queue.writeBuffer(outputsBuffer, 0, new Float32Array(nodeCount));
  device.queue.writeBuffer(
    paramsBuffer,
    0,
    new Uint32Array([
      0,
      nodeCount,
      connectionCount,
      nodeCount - network.output,
    ]),
  );
  device.queue.writeBuffer(topoLevelsBuffer, 0, topoLevels);
  device.queue.writeBuffer(inStartBuffer, 0, inStart);

  return {
    connections: connectionsBuffer,
    nodes: nodesBuffer,
    outputs: outputsBuffer,
    params: paramsBuffer,
    topoLevels: topoLevelsBuffer,
    inStart: inStartBuffer,
    nodeCount,
    connectionCount,
    topoLevelsArray: topoLevels,
    topoLevelCount: computeTopoLevelCount(topoLevels),
  };
}

/**
 * Profile a single GPU forward pass, measuring every major cold-path overhead.
 *
 * This function deliberately bypasses the production `activateGPU` caches so it
 * can time buffer allocation, pipeline compilation, and bind-group creation.
 * It creates and destroys a fresh `GPUBufferSet` per call. The returned result
 * includes a per-phase percentage breakdown, the dominant bottleneck, and an
 * overhead ratio.
 *
 * @param device - WebGPU device used to run the forward pass.
 * @param network - Network whose topology will be uploaded.
 * @param inputs - Input vector of length `network.input`.
 * @returns A `ProfilingResult` with timing breakdowns and the output vector.
 * @throws Error when the network is ineligible for GPU inference.
 *
 * @example
 * ```ts
 * const result = await profileGPUActivation(device, network, [0.5, -0.2]);
 * console.log(result.dominantBottleneck, result.overheadRatio);
 * ```
 */
export async function profileGPUActivation(
  device: GPUDevice,
  network: Network,
  inputs: Float32Array | number[],
): Promise<ProfilingResult> {
  const supportedActivations = new Set<number>(
    SUPPORTED_ACTIVATION_INDICES as readonly number[],
  );

  const result: ProfilingResult = {
    success: false,
    totalForwardPassMs: 0,
    cpuPreparationMs: 0,
    cpuTopoSortMs: 0,
    cpuCSRBuildMs: 0,
    cpuConnectionSlabBuildMs: 0,
    cpuTopologyHashMs: 0,
    bufferUploadMs: 0,
    pipelineMs: 0,
    bindGroupMs: 0,
    dynamicBufferUploadMs: 0,
    queueSubmissionMs: 0,
    gpuCompletionWaitMs: 0,
    outputReadbackMs: 0,
    phases: [],
    dominantBottleneck: 'unknown',
    overheadRatio: 0,
  };

  const totalStart = performance.now();

  try {
    if (!canUseGPU(network, device, supportedActivations)) {
      throw new Error(
        'profileGPUActivation: network is not eligible for GPU inference',
      );
    }

    if (network.nodes.length === 0) {
      throw new Error('profileGPUActivation: network has no nodes');
    }

    const typedInputs =
      inputs instanceof Float32Array ? inputs : new Float32Array(inputs);
    if (typedInputs.length !== network.input) {
      throw new Error(
        `profileGPUActivation: expected ${network.input} inputs, received ${typedInputs.length}`,
      );
    }

    const nodeCount = network.nodes.length;
    const connectionCount = network.connections.length;

    // Step 1: CPU-side preparation.
    const cpuTimer = new GpuProfilingTimer();
    cpuTimer.start('cpuTopoSort');
    const slab = network.getConnectionSlab() as {
      from: Uint32Array;
      to: Uint32Array;
      weights: Float32Array | Float64Array;
      flags: Uint8Array;
    };
    const topoLevels = buildTopoLevels(slab, nodeCount, connectionCount);
    cpuTimer.stop('cpuTopoSort');

    cpuTimer.start('cpuCSRBuild');
    const { inStart } = buildIncomingCSR(slab, nodeCount, connectionCount);
    cpuTimer.stop('cpuCSRBuild');

    cpuTimer.start('cpuConnectionSlabBuild');
    const connectionsArray = buildConnectionsArray(network, connectionCount);
    const nodesArray = buildNodesArray(network);
    cpuTimer.stop('cpuConnectionSlabBuild');

    cpuTimer.start('cpuTopologyHash');
    computeTopologyHash(network);
    cpuTimer.stop('cpuTopologyHash');

    result.cpuTopoSortMs = cpuTimer.get('cpuTopoSort');
    result.cpuCSRBuildMs = cpuTimer.get('cpuCSRBuild');
    result.cpuConnectionSlabBuildMs = cpuTimer.get('cpuConnectionSlabBuild');
    result.cpuTopologyHashMs = cpuTimer.get('cpuTopologyHash');
    result.cpuPreparationMs =
      result.cpuTopoSortMs +
      result.cpuCSRBuildMs +
      result.cpuConnectionSlabBuildMs +
      result.cpuTopologyHashMs;

    // Step 2: Create and upload GPU buffers.
    const gpuTimer = new GpuProfilingTimer();
    gpuTimer.start('bufferUpload');
    const bufferSet = createAndUploadBuffers(
      device,
      network,
      connectionsArray,
      nodesArray,
      inStart,
      topoLevels,
    );
    gpuTimer.stop('bufferUpload');
    result.bufferUploadMs = gpuTimer.get('bufferUpload');

    // Step 3: Compile (or retrieve) the activation pipeline.
    let pipeline: GPUComputePipeline | undefined;
    const activationContext = prepareActivationContext(network);
    try {
      gpuTimer.start('pipeline');
      pipeline = compileActivationKernel(device, network);
      gpuTimer.stop('pipeline');
      result.pipelineMs = gpuTimer.get('pipeline');
    } finally {
      activationContext.restore();
    }

    // Step 4: Create the kernel bind group.
    gpuTimer.start('bindGroup');
    const bindGroupLayout = createBindGroupLayout(device);
    const bindGroup = createActivationBindGroup(
      device,
      bindGroupLayout,
      bufferSet,
    );
    gpuTimer.stop('bindGroup');
    result.bindGroupMs = gpuTimer.get('bindGroup');

    // Step 5: Re-upload mutable weights/biases and scatter input values.
    gpuTimer.start('dynamicBufferUpload');
    uploadDynamicNetworkBuffers(device, bufferSet, network);
    writeInputValuesToNodeStruct(device, bufferSet.nodes, typedInputs);
    gpuTimer.stop('dynamicBufferUpload');
    result.dynamicBufferUploadMs = gpuTimer.get('dynamicBufferUpload');

    // Step 6: Dispatch the kernel and read back outputs.
    await dispatchActivationKernel(
      device,
      bufferSet,
      pipeline!,
      bindGroup,
      network.output,
      gpuTimer,
    );
    result.queueSubmissionMs = gpuTimer.get('queueSubmission');
    result.gpuCompletionWaitMs = gpuTimer.get('gpuCompletionWait');

    gpuTimer.start('outputReadback');
    const output = await readOutputValues(device, network, bufferSet);
    gpuTimer.stop('outputReadback');
    result.outputReadbackMs = gpuTimer.get('outputReadback');

    destroyGPUBufferSet(device, bufferSet);

    result.output = output;
    result.success = true;
  } catch (error) {
    result.error = String(error);
  }

  result.totalForwardPassMs = performance.now() - totalStart;

  // Compute percentage breakdown from the measured phases.
  const phaseMap = new Map<string, number>([
    ['cpuPreparation', result.cpuPreparationMs],
    ['bufferUpload', result.bufferUploadMs],
    ['pipeline', result.pipelineMs],
    ['bindGroup', result.bindGroupMs],
    ['dynamicBufferUpload', result.dynamicBufferUploadMs],
    ['queueSubmission', result.queueSubmissionMs],
    ['gpuCompletionWait', result.gpuCompletionWaitMs],
    ['outputReadback', result.outputReadbackMs],
  ]);

  const total = Math.max(result.totalForwardPassMs, Number.EPSILON);
  const phases: ProfilingPhaseTiming[] = PROFILING_PHASE_NAMES.map((name) => {
    const ms = phaseMap.get(name) as number;
    return {
      name,
      ms,
      pct: (ms / total) * 100,
    };
  });
  result.phases = phases;

  // The dominant bottleneck is the phase with the largest percentage share.
  const dominant = phases.toSorted(
    (a, b) => b.ms - a.ms,
  )[0] as ProfilingPhaseTiming;
  result.dominantBottleneck = dominant.name;

  // Overhead ratio: everything except the GPU compute wait.
  // Guard against the original measured total, not the clamped divisor, so
  // that a zero-millisecond forward pass reports zero overhead rather than
  // the EPSILON-driven value of 1.
  const gpuWaitMs = result.gpuCompletionWaitMs;
  result.overheadRatio =
    result.totalForwardPassMs > 0
      ? Math.max(0, (total - gpuWaitMs) / total)
      : 0;

  return result;
}

/**
 * Rank measured overhead phases by impact and attach a strategy to each.
 *
 * @param phases - Phase timings from one or more profile runs.
 * @returns Weak points sorted by descending percentage share.
 */
export function rankWeakPoints(
  phases: ProfilingPhaseTiming[],
): Array<{ name: ProfilingPhaseName; impactPct: number; strategy: string }> {
  const strategies: Record<ProfilingPhaseName, string> = {
    cpuPreparation:
      'Cache topological sorts and CSR arrays across activations; only rebuild when topology mutates. Consider pre-computing source-node ranks at construction time.',
    bufferUpload:
      'Use persistent mapped buffers or staging-ring uploads; move static topology buffers (connections, topoLevels, inStart) to device-local memory and only rewrite mutable weights/biases each frame.',
    pipeline:
      'Pre-warm pipelines for known topology/activation combinations at application startup or in a background compile queue. Share one pipeline across networks with identical topology.',
    bindGroup:
      'Create bind groups once per uploaded buffer set and reuse them across activations; avoid re-creating bind groups on the hot path.',
    dynamicBufferUpload:
      'Batch weight/bias updates into fewer `queue.writeBuffer` calls, coalesce input scattering, and consider double-buffering node arrays to overlap upload with GPU compute.',
    queueSubmission:
      'Batch multiple level dispatches into a single command encoder when topology allows, reduce per-level `writeBuffer` calls, and amortize submit overhead across several forward passes.',
    gpuCompletionWait:
      'Hide latency by overlapping CPU work from the next frame with the current GPU dispatch; use triple buffering or pipeline readback techniques if timestamp queries are unavailable.',
    outputReadback:
      'Read back only output nodes (already done); for repeated inference, keep outputs in GPU memory and avoid staging-buffer round-trips, or use async readback with mapped buffer rings.',
  };

  return phases
    .filter((phase) => phase.ms > 0 && Number.isFinite(phase.pct))
    .toSorted((a, b) => b.pct - a.pct)
    .map((phase) => ({
      name: phase.name,
      impactPct: phase.pct,
      strategy: strategies[phase.name],
    }));
}

/**
 * Identify the single dominant bottleneck from a profiling result.
 *
 * @param result - Profiling result produced by `profileGPUActivation`.
 * @returns Human-readable bottleneck label, or `'unknown'` when no phases were timed.
 */
export function identifyBottleneck(result: ProfilingResult): string {
  if (!result.success || result.phases.length === 0) {
    return 'unknown';
  }
  const sorted = result.phases.toSorted((a, b) => b.pct - a.pct);
  return sorted[0].name;
}

/**
 * Compute the percentage share of each overhead phase relative to the total.
 *
 * @param timings - Milliseconds per phase.
 * @returns Phase timings with percentages, sorted by descending share.
 */
export function computeOverheadBreakdown(
  timings: Record<string, number>,
): ProfilingPhaseTiming[] {
  const total = Object.values(timings).reduce(
    (sum, value) => sum + (Number.isFinite(value) ? value : 0),
    0,
  );
  const safeTotal = Math.max(total, Number.EPSILON);

  return Object.entries(timings)
    .map(([name, ms]) => ({
      name: name as ProfilingPhaseName,
      ms: Number.isFinite(ms) ? ms : 0,
      pct: ((Number.isFinite(ms) ? ms : 0) / safeTotal) * 100,
    }))
    .toSorted((a, b) => b.pct - a.pct);
}

/**
 * Reference hardware metadata embedded in the overhead-breakdown artifact.
 */
export const OVERHEAD_REFERENCE_HARDWARE = {
  processor: 'Intel i7-10700 @ 2.90GHz, 8 Cores/16 Logical',
  memory: '32GB DDR4 3200MHz',
  os: 'Windows 11 Home Build 26200',
  gpu_vendor: 'nvidia',
  gpu_architecture: 'lovelace',
  maxStorageBuffersPerShaderStage: 8,
};

/**
 * Build the overhead-breakdown artifact consumed by the browser scenario.
 *
 * @param tierResults - Per-tier profiling results.
 * @param options - Benchmark options and probed environment values.
 * @param options.browserVisibility - Visibility label, e.g. 'visible-foreground'.
 * @returns JSON-serializable artifact object.
 */
export function buildOverheadArtifact(
  tierResults: ProfilingResult[],
  options: {
    tiers?: number[];
    inputCount?: number;
    outputCount?: number;
    timestampQuerySupported?: boolean;
    gpuTimestampQueryNs?: number | null;
    gpuAdapterInfo?: Record<string, unknown> | null;
    gpuProbedLimits?: Record<string, number> | null;
    referenceHardware?: typeof OVERHEAD_REFERENCE_HARDWARE;
    browserVisibility?: string;
  } = {},
): Record<string, unknown> {
  const tiers = options.tiers ?? [64, 256, 1024, 4096, 8192, 16384, 32768];
  const inputCount = options.inputCount ?? 10;
  const outputCount = options.outputCount ?? 4;
  const referenceHardware =
    options.referenceHardware ?? OVERHEAD_REFERENCE_HARDWARE;

  const perTier = tierResults.map((result, index) => ({
    hiddenNodes: tiers[index] ?? 0,
    totalNodes:
      typeof result.output?.length === 'number'
        ? result.output.length + inputCount
        : 0,
    success: result.success,
    error: result.error,
    timings: {
      totalForwardPassMs: result.totalForwardPassMs,
      cpuPreparationMs: result.cpuPreparationMs,
      cpuTopoSortMs: result.cpuTopoSortMs,
      cpuCSRBuildMs: result.cpuCSRBuildMs,
      cpuConnectionSlabBuildMs: result.cpuConnectionSlabBuildMs,
      cpuTopologyHashMs: result.cpuTopologyHashMs,
      bufferUploadMs: result.bufferUploadMs,
      pipelineMs: result.pipelineMs,
      bindGroupMs: result.bindGroupMs,
      dynamicBufferUploadMs: result.dynamicBufferUploadMs,
      queueSubmissionMs: result.queueSubmissionMs,
      gpuCompletionWaitMs: result.gpuCompletionWaitMs,
      outputReadbackMs: result.outputReadbackMs,
    },
    phases: result.phases,
    dominantBottleneck: result.dominantBottleneck,
    overheadRatio: result.overheadRatio,
    weakPoints: rankWeakPoints(result.phases),
  }));

  // Aggregate weak points across tiers: count how often each phase dominates.
  const dominanceCounts = new Map<string, number>();
  for (const tier of perTier) {
    const name = tier.dominantBottleneck as string;
    dominanceCounts.set(name, (dominanceCounts.get(name) ?? 0) + 1);
  }
  const rankedWeakPoints = Array.from(dominanceCounts.entries())
    .toSorted((a, b) => b[1] - a[1])
    .map(([name, count]) => {
      const strategy =
        rankWeakPoints([
          {
            name: name as ProfilingPhaseName,
            ms: 0,
            pct: 0,
          },
        ])[0]?.strategy ?? 'No strategy available.';
      return { name, tierCount: count, strategy };
    });

  const successfulTierCount = perTier.filter((tier) => tier.success).length;
  const averageOverheadRatio =
    perTier.length === 0
      ? 0
      : perTier.reduce((sum, tier) => sum + (tier.overheadRatio as number), 0) /
        perTier.length;

  // True ceiling: rough estimate based on memory bandwidth and compute units.
  // These numbers are documentation-only estimates for the reference RTX 4070.
  const memoryBandwidthGBps = 504;
  const computeUnits = 46;
  const trueCeiling = {
    description:
      'Practical upper bound for single-dispatch logistic-activation forward passes on the reference RTX 4070, assuming the kernel is memory-bandwidth bound.',
    memoryBandwidthGBps,
    computeUnits,
    estimatedMaxActivationsPerSecondFor64NodeTier: Math.floor(
      (memoryBandwidthGBps * 1_000_000_000) / (64 * GPU_NODE_STRUCT_BYTES),
    ),
    notes: [
      'Ceiling assumes every byte of node/connection data is read once per activation and the workload is entirely memory-bound.',
      'Real throughput is lower due to CPU prep, buffer uploads, pipeline compile, and per-level dispatch overhead.',
      'Timestamp-query-free timing with performance.now() adds several microseconds of jitter; use GPU timestamps for tighter bounds when available.',
    ],
  };

  return {
    schemaVersion: '1.0.0',
    benchmark_type: 'overhead-breakdown',
    success: true,
    generatedAt: new Date().toISOString(),
    reference_hardware: referenceHardware,
    browser_visibility: options.browserVisibility ?? 'visible-foreground',
    environment: {
      userAgent:
        typeof navigator !== 'undefined' ? navigator.userAgent : 'node',
      platform: typeof navigator !== 'undefined' ? navigator.platform : 'node',
      gpuAdapterInfo: options.gpuAdapterInfo ?? null,
      gpuProbedLimits: options.gpuProbedLimits ?? null,
      timestampQuerySupported: options.timestampQuerySupported ?? false,
      gpuTimestampQueryNs: options.gpuTimestampQueryNs ?? null,
    },
    configuration: {
      hiddenNodeTiers: tiers,
      inputNodeCount: inputCount,
      outputNodeCount: outputCount,
    },
    summary: {
      averageOverheadRatio,
      tierCount: perTier.length,
      successfulTierCount,
    },
    tier_results: perTier,
    weak_points: rankedWeakPoints,
    ranked_weak_points: rankedWeakPoints,
    strategies: rankedWeakPoints,
    true_ceiling: trueCeiling,
    artifactFileName: 'artifacts/webgpu-overhead-breakdown.json',
  };
}
