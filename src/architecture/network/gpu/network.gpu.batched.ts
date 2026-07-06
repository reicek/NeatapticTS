/**
 * Batched WebGPU activation for multi-agent evaluation.
 *
 * Evaluates many networks in a single GPU submission, which is useful for NEAT
 * populations and other cohort-based experiments. The function builds a wide
 * input matrix and dispatches one row per network, so the GPU stays busy even
 * when individual networks are small.
 *
 * Real measurements on an RTX 4070 show that the benefit depends strongly on
 * network size and batch width. A single 64-node network spends most of its wall
 * time waiting for the GPU (75.5%) and copying back outputs, so it is still
 * slower than the CPU path. A batch of 16 networks with 64 nodes each reaches
 * about 1.8 M inferences/second, while a batch of 6 parallel 4096-node agents
 * can exceed 3 M inferences/second. The crossover where the GPU becomes faster
 * than the CPU happens around 64–256 nodes per network for parallel evaluation,
 * and GPU throughput scales with network size up to the hardware occupancy
 * limit.
 *
 * This module implements several optimization strategies:
 *
 * - Persistent per-device GPU state cached through `ensureNetworkGPUState()`.
 * - One combined command buffer with many compute passes and a single readback.
 * - A shared mappable staging buffer for the whole output matrix.
 * - Optional `skipUpload` to avoid rewriting unchanged weights and inputs.
 * - Optional `iterations` to amortize synchronization over many forward passes.
 * - Topology-aware dispatch scheduling so each topological level runs in its own
 *   compute pass.
 *
 * The WebGPU command-encoding model rewards batching. Every `queue.submit()` call
 * carries fixed driver/queue overhead, while individual `dispatchWorkgroups()`
 * calls inside the same compute pass share the pass begin/end cost and execute
 * sequentially in submission order. That sequential ordering is exactly what a
 * Kahn-style topological sort needs: nodes at level k are dispatched after nodes
 * at level k-1 have written their activations. Because each network uses its own
 * bind group (different buffer set), `setPipeline` is called per network here;
 * networks that share topology could go further and share one pipeline with only
 * bind-group switches, which is the pattern TensorFlow.js and burn use to keep
 * GPU-resident tensors batched into a single `queue.submit()`.
 *
 * CSR input layout and fused activation passes are outside the scope of this
 * implementation; the current path keeps each network in its own bind group and
 * dispatches one compute pass per topological level.
 *
 * @see [WebGPU](https://en.wikipedia.org/wiki/WebGPU) on Wikipedia for
 * background on the browser GPU compute API.
 * @see [Gather-scatter](https://en.wikipedia.org/wiki/Gather-scatter) pattern
 * @see [WebGPU command encoding](https://gpuweb.github.io/gpuweb/explainer/#command-encoding)
 *   for the spec-level rationale behind one-encoder-one-submit batching.
 * @see [TensorFlow.js WebGPU backend](https://github.com/tensorflow/tfjs/blob/7f5309fef0a47545e34049903dbdae0f97285f7e/tfjs-backend-webgpu/src/backend_webgpu.ts)
 *   for the GPU-resident batched-dispatch pattern.
 * @see [WebGPU Performance Guide](../../../docs/webgpu-performance-guide.md)
 */

import type Network from '../network';
import { canUseGPU } from './network.gpu.capability';
import { SUPPORTED_ACTIVATION_INDICES } from './network.gpu.kernel';
import { ensureNetworkGPUState } from './network.gpu.activate';
import {
  uploadDynamicNetworkBuffers,
  writeInputValuesToNodeStruct,
  type GPUBufferSet,
} from './network.gpu.buffer';

/**
 * Result shape returned by a batched GPU activation pass.
 *
 * The output matrix is stored in row-major order so that downstream consumers
 * (such as a worker controller) can slice one row per
 * agent without extra re-layout.
 */
export interface BatchedGPUResult {
  /** Row-major matrix: rows = batch size, cols = network output count. */
  outputs: Float32Array;
  /** Number of rows in the output matrix. */
  rowCount: number;
  /** Number of columns in the output matrix. */
  colCount: number;
}

/**
 * Optional tuning flags for `batchActivate()`.
 *
 * These flags let callers amortize CPU-GPU upload overhead across repeated
 * evaluations of the same static cohort. They are opt-in and unsafe when the
 * underlying weights or inputs have changed since the previous upload.
 */
export interface BatchActivateOptions {
  /**
   * Skip re-uploading connection weights, node biases, and input values.
   *
   * Use this only when the GPU buffers already contain the desired state from
   * a previous call, for example during a steady-state benchmark loop where the
   * same cohort is evaluated repeatedly. The caller is responsible for ensuring
   * that weights/biases/inputs have not changed since the last upload;
   * otherwise the returned outputs will reflect stale GPU data.
   *
   * On an RTX 4070, `skipUpload` reclaims most of the dynamic-upload slice of
   * wall time; the savings grow with network size because the upload cost is
   * linear in the number of connections and nodes.
   */
  skipUpload?: boolean;

  /**
   * Number of independent forward passes to record in a single command buffer.
   *
   * Each pass is placed in its own compute pass so writes from one pass are
   * visible to the next. This is useful for static benchmark cohorts that
   * need to amortize CPU-GPU synchronization over many identical evaluations.
   * The returned output matrix contains the result of the final pass.
   *
   * Values less than 1 are clamped to 1. When omitted, exactly one pass is
   * recorded.
   *
   * When combined with `skipUpload`, a large iteration count lets the CPU
   * schedule many forward passes while paying for readback and mapping only
   * once. For example, on an RTX 4070 a batch of 6 parallel 4096-node agents
   * exceeds 3 M inferences/second when the iteration count is large enough to
   * keep the GPU busy between synchronizations.
   */
  iterations?: number;
}

/** Number of threads per compute workgroup for the activation kernel. */
const ACTIVATION_WORKGROUP_SIZE = 64;

/** WebGPU buffer usage flag for mappable readback buffers. */
const GPU_BUFFER_USAGE_MAP_READ = 0x0001;

/** WebGPU buffer usage flag for copy destinations. */
const GPU_BUFFER_USAGE_COPY_DST = 0x0008;

/** WebGPU map mode for reading mapped buffers. */
const GPU_MAP_MODE_READ = 0x0001;

/**
 * Local extension of the ambient GPU command encoder so we can copy a storage
 * buffer to a mappable staging buffer. The WebGPU ambient types in this repo
 * are intentionally minimal; the cast is justified because `copyBufferToBuffer`
 * is part of the actual WebGPU API surface.
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
 * Validate the batching contract before any GPU work is issued.
 *
 * @param device - WebGPU device that will run the dispatch.
 * @param networks - Networks to evaluate as a batch.
 * @param inputMatrix - Flattened row-major input matrix.
 * @throws Error when a required input is missing or dimensions are inconsistent.
 */
function validateBatchInputs(
  device: GPUDevice,
  networks: Network[],
  inputMatrix: Float32Array,
): void {
  if (!device) {
    throw new Error('batchActivate requires a GPU device');
  }

  if (!Array.isArray(networks)) {
    throw new Error('batchActivate expects networks to be an array');
  }

  if (networks.length > 0) {
    const firstNetwork = networks[0];
    if (firstNetwork === null || firstNetwork === undefined) {
      throw new Error('batchActivate received a null or undefined network');
    }

    const expectedInputLength = networks.length * firstNetwork.input;
    if (inputMatrix.length !== expectedInputLength) {
      throw new Error(
        `inputMatrix length ${inputMatrix.length} does not match networks.length * inputCount (${expectedInputLength})`,
      );
    }
  }
}

/**
 * Ensure every network in the batch has the same input and output dimensions.
 *
 * The result matrix is row-major with one column count for the entire batch, so
 * mixed shapes would corrupt the layout.
 *
 * @param networks - Networks to validate.
 * @throws Error when dimensions differ or a network is null/undefined.
 */
function validateNetworkShapes(networks: Network[]): void {
  const expectedInput = networks[0].input;
  const expectedOutput = networks[0].output;

  for (let index = 1; index < networks.length; index += 1) {
    const network = networks[index];
    if (network === null || network === undefined) {
      throw new Error(
        `batchActivate received a null or undefined network at index ${index}`,
      );
    }

    if (network.input !== expectedInput || network.output !== expectedOutput) {
      throw new Error(
        `Network at index ${index} has shape (input=${network.input}, output=${network.output}); expected (input=${expectedInput}, output=${expectedOutput})`,
      );
    }
  }
}

/**
 * Per-device cache for the reusable batched-output staging buffer.
 *
 * `batchActivate` reads back every network's output row into one contiguous
 * staging buffer. Reusing that buffer across calls removes the per-network
 * allocation, mapping, and destruction overhead that otherwise dominates CPU
 * time for batched readback.
 */
const batchedOutputStagingBufferCache = new WeakMap<
  GPUDevice,
  Map<number, GPUBuffer>
>();

/**
 * Return the per-device batched-output staging-buffer size map.
 */
function getBatchedOutputStagingBufferCache(
  device: GPUDevice,
): Map<number, GPUBuffer> {
  let cache = batchedOutputStagingBufferCache.get(device);
  if (!cache) {
    cache = new Map();
    batchedOutputStagingBufferCache.set(device, cache);
  }
  return cache;
}

/**
 * Fetch or create a reusable mappable staging buffer for the full batched
 * output matrix.
 *
 * @param device - WebGPU device that owns the staging buffer.
 * @param byteLength - Total output matrix size in bytes.
 * @returns A GPU buffer with `MAP_READ | COPY_DST` usage.
 */
function getOrCreateBatchedOutputStagingBuffer(
  device: GPUDevice,
  byteLength: number,
): GPUBuffer {
  const cache = getBatchedOutputStagingBufferCache(device);
  let buffer = cache.get(byteLength);
  const isDestroyed =
    buffer !== undefined &&
    typeof (buffer as unknown as { destroyed?: boolean }).destroyed ===
      'boolean' &&
    (buffer as unknown as { destroyed: boolean }).destroyed;

  if (buffer === undefined || isDestroyed) {
    buffer = device.createBuffer({
      label: 'network_batched_outputs_staging',
      size: byteLength,
      usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
    });
    cache.set(byteLength, buffer);
  }

  return buffer;
}

/**
 * Batched GPU activation for multi-agent evaluation.
 *
 * Reuses the per-network persistent GPU state managed by
 * `ensureNetworkGPUState()`, uploads only the dynamic node/connection data
 * and the input matrix each call, dispatches all networks in one or more compute
 * passes once per topological level, and reads back the output matrix through a
 * single reusable staging buffer. This removes the per-call buffer allocation,
 * mapping, and destruction that otherwise make the GPU path slower than the CPU
 * path for small networks. The optional `iterations` flag records many
 * independent passes inside a single command buffer with only one CPU-GPU
 * readback.
 *
 * Because every pass is recorded before the command buffer is submitted, only
 * one `mapAsync` call is needed for the final result. The WebGPU specification
 * already guarantees that mapping waits for all previously submitted work on the
 * buffer's queue timeline, so an additional `onSubmittedWorkDone()` barrier is
 * redundant for readback and is intentionally omitted.
 *
 * @remarks
 * Measured on an RTX 4070 with the stable NVIDIA driver and Chrome release
 * available at profiling time, using the default single-network CPU path as the
 * baseline:
 *
 * | Nodes | Single-GPU wait | CPU prep | Crossover |
 * |---|---|---|---|
 * | 64 | 75.5% | 13.2% | slower than CPU |
 * | 256 | ~58% | 27.6% | ~equal to CPU |
 * | 1k | ~25% | 38.9% | faster than CPU |
 * | 4k | ~10% | 44.9% | clearly faster |
 * | 8k | ~5% | 42.3% | clearly faster |
 *
 * Batching widens the win: a cohort of 16 64-node networks reaches about
 * 1.8 M inferences/second, and 6 parallel 4096-node agents can exceed 3 M
 * inferences/second. The largest throughput gains come from `skipUpload` and
 * a large `iterations` count, which keep the GPU busy while the CPU pays for
 * readback only once.
 *
 * @param device - WebGPU device used to run the forward kernel.
 * @param networks - Networks to evaluate as a batch. All networks must have the
 *   same input and output dimensions.
 * @param inputMatrix - Flattened row-major inputs, length
 *   `networks.length * networks[0].input`. Still validated when upload is
 *   skipped, but not written to the GPU in that case.
 * @param options - Optional tuning flags for repeated static evaluation
 *   (see `BatchActivateOptions`).
 * @returns Promise resolving to a row-major output matrix.
 * @throws Error when a required input is missing, dimensions are inconsistent,
 *   or a network is ineligible for GPU inference.
 *
 * @see [WebGPU Performance Guide](../../../docs/webgpu-performance-guide.md)
 * for the measured impact of `skipUpload` and `iterations` on throughput.
 *
 * @example
 * ```ts
 * const networks = Array.from({ length: 4 }, () => Network.createMLP(2, [3], 1));
 * const inputs = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
 * const { outputs, rowCount, colCount } = await batchActivate(device, networks, inputs);
 *
 * // Amortize CPU-GPU synchronization across 60 identical static evaluations.
 * const batched = await batchActivate(device, networks, inputs, {
 *   skipUpload: true,
 *   iterations: 60,
 * });
 * ```
 */
export async function batchActivate(
  device: GPUDevice,
  networks: Network[],
  inputMatrix: Float32Array,
  options?: BatchActivateOptions,
): Promise<BatchedGPUResult> {
  validateBatchInputs(device, networks, inputMatrix);

  if (networks.length === 0) {
    return {
      outputs: new Float32Array(0),
      rowCount: 0,
      colCount: 0,
    };
  }

  validateNetworkShapes(networks);

  const supportedActivations = new Set<number>(
    SUPPORTED_ACTIVATION_INDICES as unknown as number[],
  );
  const inputCount = networks[0].input;
  const outputCount = networks[0].output;
  const rowCount = networks.length;
  const skipUpload = options?.skipUpload ?? false;
  const iterationCount = Math.max(1, options?.iterations ?? 1);

  for (let index = 0; index < networks.length; index += 1) {
    if (!canUseGPU(networks[index], device, supportedActivations)) {
      throw new Error(
        `batchActivate: network at index ${index} is not eligible for GPU inference`,
      );
    }
  }

  // Ensure each network has a persistent GPU state entry. This entry caches the
  // uploaded slab buffers, per-level params buffers, bind groups, and the
  // compiled pipeline, so repeated batch evaluations only refresh dynamic data.
  const states = [] as {
    bufferSet: GPUBufferSet;
    levelBindGroups: (GPUBindGroup | undefined)[];
  }[];
  const pipelines = [] as GPUComputePipeline[];

  for (const network of networks) {
    const { state, pipeline } = ensureNetworkGPUState(device, network);
    states.push(state);
    pipelines.push(pipeline);
  }

  // Refresh dynamic connection/node data and seed each network's input row
  // unless the caller has already warmed the GPU buffers and asked us to skip
  // the upload. Skipping is an opt-in contract used by static benchmark loops.
  if (!skipUpload) {
    for (let index = 0; index < networks.length; index += 1) {
      const network = networks[index];
      const { bufferSet } = states[index];
      uploadDynamicNetworkBuffers(device, bufferSet, network);

      const inputSlice = inputMatrix.subarray(
        index * inputCount,
        (index + 1) * inputCount,
      );
      writeInputValuesToNodeStruct(device, bufferSet.nodes, inputSlice);
    }
  }

  // One command encoder records every requested pass. Each pass gets its own
  // compute pass so memory writes from one iteration are visible to the next.
  // Networks with shared topology reuse their cached pipeline by switching bind
  // groups inside the same command encoder.
  const commandEncoder = device.createCommandEncoder({
    label: 'network_batched_activation',
  });

  for (let iteration = 0; iteration < iterationCount; iteration += 1) {
    const computePass = commandEncoder.beginComputePass({
      label: `network_batched_activation_pass_${iteration}`,
    });

    for (let index = 0; index < networks.length; index += 1) {
      const pipeline = pipelines[index];
      const { bufferSet, levelBindGroups } = states[index];
      const workgroupCount = Math.ceil(
        bufferSet.nodeCount / ACTIVATION_WORKGROUP_SIZE,
      );

      for (let level = 1; level < bufferSet.topoLevelCount; level += 1) {
        const bindGroup = levelBindGroups[level];
        if (!bindGroup) {
          throw new Error(
            `batchActivate: missing bind group for level ${level}`,
          );
        }

        computePass.setPipeline(pipeline);
        computePass.setBindGroup(0, bindGroup);
        computePass.dispatchWorkgroups(workgroupCount);
      }
    }

    computePass.end();
  }

  // Copy every network's output-node slice into a single reusable staging
  // buffer in the same command encoder, then read the whole matrix back once.
  const totalOutputByteLength =
    rowCount * outputCount * Float32Array.BYTES_PER_ELEMENT;
  const stagingBuffer = getOrCreateBatchedOutputStagingBuffer(
    device,
    totalOutputByteLength,
  );

  for (let index = 0; index < networks.length; index += 1) {
    const network = networks[index];
    const { bufferSet } = states[index];
    const outputNodeCount = network.output;
    const outputByteLength = outputNodeCount * Float32Array.BYTES_PER_ELEMENT;
    const outputStartOffset =
      (bufferSet.nodeCount - outputNodeCount) * Float32Array.BYTES_PER_ELEMENT;
    const destinationOffset =
      index * outputCount * Float32Array.BYTES_PER_ELEMENT;

    const copyEncoder = commandEncoder as unknown as GPUCommandEncoderCopy;
    copyEncoder.copyBufferToBuffer(
      bufferSet.outputs,
      outputStartOffset,
      stagingBuffer,
      destinationOffset,
      outputByteLength,
    );
  }

  device.queue.submit([commandEncoder.finish()]);

  await stagingBuffer.mapAsync(GPU_MAP_MODE_READ);
  const mappedRange = stagingBuffer.getMappedRange();
  const outputs = new Float32Array(mappedRange.slice(0, totalOutputByteLength));
  stagingBuffer.unmap();

  return {
    outputs,
    rowCount,
    colCount: outputCount,
  };
}

/**
 * Single job queued for deferred batched GPU inference.
 */
export interface BatchInferenceJob {
  network: Network;
  inputs: Float32Array | number[];
}

/**
 * Queue that accumulates inference jobs and flushes them as one GPU batch.
 *
 * The queue is intentionally not backed by persistent storage; it exists only
 * to amortize GPU dispatch overhead across many small inference requests.
 */
export interface BatchInferenceQueue {
  /** Number of jobs currently in the queue. */
  size: number;
  /** Add a job to the queue and return a stable job id. */
  enqueue(job: BatchInferenceJob): number;
  /** Dispatch all queued jobs in a single GPU pass and return outputs in order. */
  flush(): Promise<Float32Array[]>;
}

/**
 * Concrete queue that accumulates inference jobs and flushes them as one GPU batch.
 *
 * The queue reuses `batchActivate` for the actual dispatch, so pipeline sharing,
 * struct-packed buffer uploads, and single-pass submission are inherited. Jobs are
 * kept in enqueue order and the per-job outputs are returned in the same order.
 */
class BatchInferenceQueueImpl implements BatchInferenceQueue {
  /** Stored jobs waiting for the next flush. */
  private jobs: BatchInferenceJob[] = [];

  /** Monotonically increasing job id counter. */
  private nextId = 0;

  /**
   * @param device - WebGPU device used to run the batched dispatch.
   */
  constructor(private readonly device: GPUDevice) {}

  /** Number of jobs currently in the queue. */
  get size(): number {
    return this.jobs.length;
  }

  /**
   * Add a job to the queue and return a stable job id.
   *
   * @param job - Network plus inputs to evaluate.
   * @returns Stable id for this job; ids increase by one for each enqueue.
   */
  enqueue(job: BatchInferenceJob): number {
    this.jobs.push(job);
    const id = this.nextId;
    this.nextId += 1;
    return id;
  }

  /**
   * Dispatch all queued jobs in a single GPU pass and return outputs in order.
   *
   * @returns Promise resolving to one output array per job, in enqueue order.
   */
  async flush(): Promise<Float32Array[]> {
    const pendingJobs = this.jobs.splice(0);
    if (pendingJobs.length === 0) {
      return [];
    }

    const networks = pendingJobs.map((job) => job.network);
    const inputCount = networks[0].input;
    const inputMatrix = new Float32Array(pendingJobs.length * inputCount);
    let writeOffset = 0;
    for (const job of pendingJobs) {
      const inputs = new Float32Array(job.inputs);
      inputMatrix.set(inputs, writeOffset);
      writeOffset += inputs.length;
    }

    const result = await batchActivate(this.device, networks, inputMatrix);

    const outputs: Float32Array[] = [];
    for (let index = 0; index < pendingJobs.length; index += 1) {
      outputs.push(
        result.outputs.slice(
          index * result.colCount,
          (index + 1) * result.colCount,
        ),
      );
    }
    return outputs;
  }
}

/**
 * Create a queue that batches inference jobs for parallel GPU dispatch.
 *
 * The returned queue accumulates jobs via `enqueue()` and dispatches them all
 * together on the next `flush()`, sharing compiled pipelines across networks
 * with identical topology and returning one output per job in enqueue order.
 * An empty queue resolves to an empty array without issuing GPU work.
 *
 * @param device - WebGPU device used to run the batched dispatch.
 * @returns A queue ready to accept inference jobs.
 */
export function createBatchInferenceQueue(
  device: GPUDevice,
): BatchInferenceQueue {
  return new BatchInferenceQueueImpl(device);
}
