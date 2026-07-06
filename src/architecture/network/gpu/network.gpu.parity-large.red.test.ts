import Network from '../network';
import { createMockGPUDevice } from './__mocks__/gpu.mock';
import { ACTIVATION_FUNCTIONS } from '../../../multithreading/multi.utils';

const LARGE_INPUTS = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0];

// The GPU connection buffer is now packed in the same source-topological order
// the CPU fast-slab path uses, so both paths sum identical f32 terms in the
// same order. These tight tolerances assert the ordering fix eliminated the
// cross-path rounding drift for this 10-64-4 feed-forward network.
const MAX_ABS_TOLERANCE = 1e-3;
const MEAN_ABS_TOLERANCE = 1e-4;

describe('network.gpu.parity-large.red', () => {
  describe('large feed-forward network (10-64-4)', () => {
    let network: Network;
    let device: ReturnType<typeof createMockGPUDevice>;
    let cpuOutput: number[];
    let gpuOutput: Float32Array;
    const originalSquashes: Array<(x: number, derivate?: boolean) => number> =
      [];

    beforeEach(async () => {
      network = Network.createMLP(10, [64], 4);

      for (const node of network.nodes) {
        originalSquashes.push(node.squash);
        node.squash = ACTIVATION_FUNCTIONS[0];
      }

      cpuOutput = network.activate(LARGE_INPUTS);

      const logisticActivation = ACTIVATION_FUNCTIONS[0];
      device = createMockGPUDevice({
        generateOutput: ({ inputs }) => {
          const generated = new Float32Array(inputs.length);
          for (let index = 0; index < generated.length; index += 1) {
            generated[index] = logisticActivation(inputs[index]);
          }
          return generated;
        },
      });
      network.gpuDevice = device;

      gpuOutput = await network.activate(LARGE_INPUTS, { useGPU: true });
    });

    afterEach(() => {
      for (
        let nodeIndex = 0;
        nodeIndex < network.nodes.length;
        nodeIndex += 1
      ) {
        network.nodes[nodeIndex].squash = originalSquashes[nodeIndex];
      }
      network.gpuDevice = undefined;
      originalSquashes.length = 0;
    });

    it('keeps per-element absolute difference below 1e-3', () => {
      const maxAbsoluteDifference = Math.max(
        ...cpuOutput.map((value, index) => Math.abs(value - gpuOutput[index])),
      );

      expect(maxAbsoluteDifference).toBeLessThan(MAX_ABS_TOLERANCE);
    });

    it('keeps mean absolute difference below 1e-4', () => {
      const meanAbsoluteDifference =
        cpuOutput.reduce(
          (sum, value, index) => sum + Math.abs(value - gpuOutput[index]),
          0,
        ) / cpuOutput.length;

      expect(meanAbsoluteDifference).toBeLessThan(MEAN_ABS_TOLERANCE);
    });
  });
});
