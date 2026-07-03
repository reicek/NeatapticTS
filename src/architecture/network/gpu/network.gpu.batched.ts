/**
 * Batched WebGPU activation for multi-agent evaluation.
 *
 * This module evaluates many networks in a single GPU dispatch, which is useful
 * when the racing-curriculum worker or another demo needs to score a whole
 * generation at once. Networks with the same topology share compiled pipelines,
 * and the output is returned as a row-major matrix with one row per network.
 *
 * The seam remains opt-in: callers must supply a usable `GPUDevice` and every
 * network must pass the same structural eligibility checks used by the
 * single-network GPU path. Ineligible networks or missing hardware fall back
 * to per-network CPU activation through `evaluateRacingGeneration` or a
 * caller-local fallback.
 *
 * @see [WebGPU](https://en.wikipedia.org/wiki/WebGPU) on Wikipedia for
 * background on the browser GPU compute API.
 */

import type Network from '../network';
import { ACTIVATION_FUNCTIONS } from '../../../multithreading/multi.utils';
import { canUseGPU } from './network.gpu.capability';
import {
  compileActivationKernel,
  SUPPORTED_ACTIVATION_INDICES,
} from './network.gpu.kernel';
import {
  destroyGPUBufferSet,
  uploadNetworkToGPU,
  type GPUBufferSet,
} from './network.gpu.buffer';
import { GPU_BUFFER_BINDING } from './network.gpu.types';

/**
 * Result shape returned by a batched GPU activation pass.
 *
 * The output matrix is stored in row-major order so that downstream consumers
 * (such as the racing-curriculum worker controller) can slice one row per
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
 * Stable cross-module activation key used by the runtime registry.
 *
 * `Symbol.for` keeps the key identical even when `methods/activation` and the
 * worker registry are loaded from different bundles or mocked contexts, which
 * is exactly the boundary that breaks strict function identity.
 */
const ACTIVATION_KEY_SYMBOL = Symbol.for('neataptic.activation.key');

/**
 * Map from activation base name to worker-registry index.
 *
 * The worker registry function names end in "Activation" (e.g.
 * `logisticActivation`); the runtime registry and the WGSL naming table use the
 * shorter base form (e.g. `logistic`). Stripping the suffix gives a single stable
 * lookup key that works for both.
 */
const WORKER_ACTIVATION_INDEX_BY_NAME = new Map<string, number>(
  ACTIVATION_FUNCTIONS.map((activation, index) => {
    const baseName = activation.name.replace(/Activation$/, '');
    return [baseName, index];
  }),
);

/**
 * Names that the runtime registry uses but the worker registry names
 * differently.
 *
 * The runtime registry exposes `sigmoid` as an alias for the logistic function,
 * while the worker registry only stores the canonical `logisticActivation`.
 */
const ACTIVATION_NAME_ALIASES = new Map<string, string>([
  ['sigmoid', 'logistic'],
]);

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
 * Look up the worker-registry activation index for a built-in squash function.
 *
 * The lookup is intentionally robust across module-loading boundaries and mock
 * environments where the same activation may be imported from different source
 * files and therefore fails a strict `===` comparison. It first tries strict
 * identity, then the runtime-registry symbol key, then falls back to the
 * function name.
 *
 * @param squash - Activation function attached to a node.
 * @returns The corresponding worker index, or `undefined` when the function is
 *   not part of the canonical registry.
 */
function resolveActivationIndex(
  squash: (value: number, derivate?: boolean) => number,
): number | undefined {
  const identityIndex = ACTIVATION_FUNCTIONS.indexOf(squash);
  if (identityIndex >= 0) {
    return identityIndex;
  }

  const keyedSquash = squash as typeof squash & {
    [ACTIVATION_KEY_SYMBOL]?: string;
  };
  const symbolKey = keyedSquash[ACTIVATION_KEY_SYMBOL];
  if (typeof symbolKey === 'string') {
    const aliasedName = ACTIVATION_NAME_ALIASES.get(symbolKey) ?? symbolKey;
    const index = WORKER_ACTIVATION_INDEX_BY_NAME.get(aliasedName);
    if (index !== undefined) {
      return index;
    }
  }

  const name = squash.name;
  if (name) {
    const baseName = name.replace(/Activation$/, '');
    const aliasedName = ACTIVATION_NAME_ALIASES.get(baseName) ?? baseName;
    const index = WORKER_ACTIVATION_INDEX_BY_NAME.get(aliasedName);
    if (index !== undefined) {
      return index;
    }
  }

  return undefined;
}

/**
 * Temporarily annotate the first node's squash with its worker-registry index
 * so `compileActivationKernel` can generate the correct WGSL switch, then
 * restore the original value.
 *
 * @param network - Network whose first node squash will be temporarily annotated.
 * @returns A context object with a `restore()` callback.
 * @throws Error when the first node has no squash or it is not a built-in worker
 *   activation.
 */
function prepareActivationContext(network: Network): { restore: () => void } {
  const firstNode = network.nodes[0];
  const squash = firstNode.squash as
    | (((value: number, derivate?: boolean) => number) & { index?: number })
    | undefined;

  if (!squash) {
    throw new Error('batchActivate: first node has no squash function');
  }

  const savedIndex = squash.index;
  const index = resolveActivationIndex(squash);

  if (index === undefined) {
    throw new Error(
      'batchActivate: first node uses an activation that is not in the worker registry',
    );
  }

  squash.index = index;

  return {
    restore: () => {
      squash.index = savedIndex;
    },
  };
}

/**
 * Create the bind group for the supplied compiled pipeline and uploaded buffer
 * set.
 *
 * @param device - WebGPU device used to create the bind group.
 * @param pipeline - Compiled activation pipeline.
 * @param bufferSet - Uploaded network slab buffers.
 * @returns A bind group wired to the four struct-packed storage-buffer and
 *   uniform bindings.
 */
function createBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bufferSet: GPUBufferSet,
): GPUBindGroup {
  const bindGroupLayout = pipeline.getBindGroupLayout(0);

  return device.createBindGroup({
    layout: bindGroupLayout,
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
    ],
  });
}
/**
 * Batched GPU activation for multi-agent evaluation.
 *
 * Uploads the input matrix and every network's fast-slab topology to the GPU,
 * reuses compiled pipelines for networks that share topology, dispatches all
 * networks in a single compute pass, and reads back one output row per network
 * into a row-major result matrix. The CPU path remains the default; this seam
 * is opt-in and gated by `canUseGPU`.
 *
 * @param device - WebGPU device used to run the forward kernel.
 * @param networks - Networks to evaluate as a batch. All networks must have the
 *   same input and output dimensions.
 * @param inputMatrix - Flattened row-major inputs, length
 *   `networks.length * networks[0].input`.
 * @returns Promise resolving to a row-major output matrix.
 * @throws Error when a required input is missing, dimensions are inconsistent,
 *   or a network is ineligible for GPU inference.
 *
 * @example
 * ```ts
 * const networks = Array.from({ length: 4 }, () => Network.createMLP(2, [3], 1));
 * const inputs = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
 * const { outputs, rowCount, colCount } = await batchActivate(device, networks, inputs);
 * ```
 */
export async function batchActivate(
  device: GPUDevice,
  networks: Network[],
  inputMatrix: Float32Array,
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

  for (let index = 0; index < networks.length; index += 1) {
    if (!canUseGPU(networks[index], device, supportedActivations)) {
      throw new Error(
        `batchActivate: network at index ${index} is not eligible for GPU inference`,
      );
    }
  }

  const bufferSets: GPUBufferSet[] = [];
  const pipelines: GPUComputePipeline[] = [];

  try {
    // Compile or reuse a pipeline for each network. The pipeline cache is keyed
    // by topology, so identical networks share one compiled shader.
    for (const network of networks) {
      const context = prepareActivationContext(network);
      try {
        const pipeline = compileActivationKernel(device, network);
        pipelines.push(pipeline);
      } finally {
        context.restore();
      }
    }

    // Upload every network and write the corresponding input row into its
    // node buffer. The kernel reads from the same node buffer it writes to.
    for (let index = 0; index < networks.length; index += 1) {
      const network = networks[index];
      const bufferSet = uploadNetworkToGPU(device, network);
      bufferSets.push(bufferSet);

      const inputSlice = inputMatrix.subarray(
        index * inputCount,
        (index + 1) * inputCount,
      );
      device.queue.writeBuffer(
        bufferSet.nodes,
        0,
        inputSlice,
        0,
        inputSlice.length,
      );
    }

    // One command encoder and one compute pass dispatch every network in the
    // batch. Networks with shared topology reuse their cached pipeline inside
    // the same pass by switching bind groups.
    const commandEncoder = device.createCommandEncoder({
      label: 'network_batched_activation',
    });
    const computePass = commandEncoder.beginComputePass({
      label: 'network_batched_activation_pass',
    });

    for (let index = 0; index < networks.length; index += 1) {
      const pipeline = pipelines[index];
      const bufferSet = bufferSets[index];
      const bindGroup = createBindGroup(device, pipeline, bufferSet);
      const workgroupCount = Math.ceil(
        bufferSet.nodeCount / ACTIVATION_WORKGROUP_SIZE,
      );

      computePass.setPipeline(pipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(workgroupCount);
    }

    computePass.end();

    // Copy the output-node slice of every network's output buffer to a
    // dedicated staging buffer in the same command encoder.
    const stagingBuffers: GPUBuffer[] = [];
    for (let index = 0; index < networks.length; index += 1) {
      const network = networks[index];
      const bufferSet = bufferSets[index];
      const outputNodeCount = network.output;
      const outputByteLength = outputNodeCount * Float32Array.BYTES_PER_ELEMENT;
      const outputStartOffset =
        (bufferSet.nodeCount - outputNodeCount) *
        Float32Array.BYTES_PER_ELEMENT;

      const stagingBuffer = device.createBuffer({
        label: `network_batched_outputs_staging_${index}`,
        size: outputByteLength,
        usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
      });
      stagingBuffers.push(stagingBuffer);

      const copyEncoder = commandEncoder as unknown as GPUCommandEncoderCopy;
      copyEncoder.copyBufferToBuffer(
        bufferSet.outputs,
        outputStartOffset,
        stagingBuffer,
        0,
        outputByteLength,
      );
    }

    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Read back each staging buffer and assemble the row-major output matrix.
    const outputs = new Float32Array(rowCount * outputCount);
    for (let index = 0; index < networks.length; index += 1) {
      const stagingBuffer = stagingBuffers[index];
      await stagingBuffer.mapAsync(GPU_MAP_MODE_READ);
      const mappedRange = stagingBuffer.getMappedRange();
      const row = new Float32Array(mappedRange.slice(0));
      outputs.set(row, index * outputCount);
      stagingBuffer.unmap();
      stagingBuffer.destroy();
    }

    return {
      outputs,
      rowCount,
      colCount: outputCount,
    };
  } finally {
    for (const bufferSet of bufferSets) {
      destroyGPUBufferSet(device, bufferSet);
    }
  }
}
