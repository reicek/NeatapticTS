import type Network from '../network';
import { canUseGPU } from './network.gpu.capability';

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

/**
 * Worker-compatible activation indices that the future WGSL kernel will support.
 *
 * The numeric order matches the ordered registry in
 * `src/multithreading/multi.utils.ts` so that GPU and CPU paths decode the same
 * activation index consistently.
 */
const SUPPORTED_ACTIVATION_INDICES = new Set<number>([
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
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
 * Count how many of the supplied networks are structurally eligible for GPU
 * inference. This count is exposed to the contract tests even though the
 * current placeholder does not issue real dispatches.
 *
 * @param device - WebGPU device used for eligibility checks.
 * @param networks - Candidate networks.
 * @returns Number of eligible networks.
 */
function countEligibleNetworks(device: GPUDevice, networks: Network[]): number {
  return networks.filter((network) =>
    canUseGPU(network, device, SUPPORTED_ACTIVATION_INDICES),
  ).length;
}

/**
 * Placeholder batched GPU activation for the red-test seam.
 *
 * This function exists so the owner-local test suite can exercise the batching
 * contract before real WebGPU dispatch is implemented. It validates inputs,
 * filters eligible networks, and returns a zero-filled result matrix of the
 * expected shape. It intentionally does not create command encoders, dispatch
 * workgroups, or map read-back buffers; those real-contract assertions remain
 * failing in the red-test phase.
 *
 * @param device - WebGPU device used for eligibility checks.
 * @param networks - Networks to evaluate as a batch.
 * @param inputMatrix - Flattened row-major inputs, length
 *   `networks.length * inputCount`.
 * @returns Promise resolving to a zero-filled result matrix of the correct
 *   shape.
 * @throws Error when device is missing or input dimensions are inconsistent.
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

  const eligibleCount = countEligibleNetworks(device, networks);
  void eligibleCount;

  const outputCount = networks[0].output;
  const rowCount = networks.length;
  const outputs = new Float32Array(rowCount * outputCount).fill(0);

  return {
    outputs,
    rowCount,
    colCount: outputCount,
  };
}
