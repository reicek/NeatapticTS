import Network from '../network';
import { ACTIVATION_FUNCTIONS } from '../../../multithreading/multi.utils';
import {
  batchActivate,
  createBatchInferenceQueue,
} from './network.gpu.batched';
import * as gpuActivate from './network.gpu.activate';
import * as gpuBuffer from './network.gpu.buffer';
import { GPU_NODE_STRUCT_BYTES } from './network.gpu.buffer';
import { createMockGPUDevice } from './__mocks__/gpu.mock';

describe('network.gpu.batched', () => {
  describe('batchActivate', () => {
    it('is a defined async function that can be imported and called', async () => {
      const device = createMockGPUDevice();

      const result = await batchActivate(device, [], new Float32Array(0));

      expect(result).toBeDefined();
    });

    it('produces an outputs array of length batchSize * outputCount', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const batchSize = 4;
      const inputMatrix = new Float32Array(batchSize * network.input);

      const result = await batchActivate(
        device,
        Array.from({ length: batchSize }, () => network),
        inputMatrix,
      );

      expect(result.outputs.length).toBe(batchSize * network.output);
    });

    it('returns an empty result for an empty batch', async () => {
      const device = createMockGPUDevice();

      const result = await batchActivate(device, [], new Float32Array(0));

      expect(result).toEqual({
        outputs: new Float32Array(0),
        rowCount: 0,
        colCount: 0,
      });
    });

    it('issues no GPU submissions for an empty batch', async () => {
      const device = createMockGPUDevice();

      await batchActivate(device, [], new Float32Array(0));

      expect(device.recorded.submissions.length).toBe(0);
    });

    it('uploads one input row per network to the GPU', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const batchSize = 3;
      const inputMatrix = new Float32Array(batchSize * network.input);

      await batchActivate(
        device,
        Array.from({ length: batchSize }, () => network.clone()),
        inputMatrix,
      );

      const inputByteLength = Float32Array.BYTES_PER_ELEMENT;
      const inputWrites = device.recorded.writeBuffers.filter(
        (record) =>
          (record.buffer as unknown as { label?: string }).label ===
            'network_nodes' && record.byteLength === inputByteLength,
      );
      const totalBytes = inputWrites.reduce(
        (sum, record) => sum + record.byteLength,
        0,
      );

      const allOffsetsAligned = inputWrites.every(
        (record) => record.offset % GPU_NODE_STRUCT_BYTES === 0,
      );

      expect({
        writeCount: inputWrites.length,
        totalBytes,
        allOffsetsAligned,
      }).toEqual({
        writeCount: batchSize * network.input,
        totalBytes: batchSize * network.input * inputByteLength,
        allOffsetsAligned: true,
      });
    });

    it('skips dynamic uploads when skipUpload is true', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const batchSize = 2;
      const inputMatrix = new Float32Array(batchSize * network.input).fill(0.5);
      const networks = Array.from({ length: batchSize }, () => network.clone());

      const uploadSpy = jest
        .spyOn(gpuBuffer, 'uploadDynamicNetworkBuffers')
        .mockImplementation(() => undefined);
      const writeSpy = jest
        .spyOn(gpuBuffer, 'writeInputValuesToNodeStruct')
        .mockImplementation(() => undefined);

      await batchActivate(device, networks, inputMatrix, { skipUpload: true });

      expect({
        uploadCalls: uploadSpy.mock.calls.length,
        writeCalls: writeSpy.mock.calls.length,
      }).toEqual({
        uploadCalls: 0,
        writeCalls: 0,
      });

      uploadSpy.mockRestore();
      writeSpy.mockRestore();
    });

    it('reuses one compiled pipeline for networks with shared topology', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const batchSize = 4;

      await batchActivate(
        device,
        Array.from({ length: batchSize }, () => network.clone()),
        new Float32Array(batchSize * network.input),
      );

      expect(device.recorded.pipelines.length).toBe(1);
    });

    it('schedules a single mapAsync readback for the whole batch', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const batchSize = 3;

      await batchActivate(
        device,
        Array.from({ length: batchSize }, () => network.clone()),
        new Float32Array(batchSize * network.input),
      );

      expect(device.recorded.mapAsyncCalls.length).toBe(1);
    });

    it('matches CPU reference output for each input row', async () => {
      const referenceNetwork = Network.createMLP(2, [3], 1);
      const batchSize = 3;
      const inputMatrix = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
      const device = createMockGPUDevice({
        emulateNetwork: referenceNetwork,
      });
      const networks = Array.from({ length: batchSize }, () =>
        referenceNetwork.clone(),
      );

      const result = await batchActivate(device, networks, inputMatrix);

      let maxDifference = 0;
      for (let index = 0; index < batchSize; index += 1) {
        const inputRow = inputMatrix.subarray(
          index * referenceNetwork.input,
          (index + 1) * referenceNetwork.input,
        );
        const expected = referenceNetwork.activate(Array.from(inputRow));
        const actual = result.outputs.subarray(
          index * referenceNetwork.output,
          (index + 1) * referenceNetwork.output,
        );

        for (
          let outputIndex = 0;
          outputIndex < expected.length;
          outputIndex += 1
        ) {
          maxDifference = Math.max(
            maxDifference,
            Math.abs(actual[outputIndex] - expected[outputIndex]),
          );
        }
      }

      expect(maxDifference).toBeLessThan(1e-4);
    });

    it('throws when device is missing', async () => {
      const network = Network.createMLP(2, [3], 1);

      await expect(
        batchActivate(
          null as unknown as GPUDevice,
          [network],
          new Float32Array(network.input),
        ),
      ).rejects.toThrow('batchActivate requires a GPU device');
    });

    it('throws when networks is not an array', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);

      await expect(
        batchActivate(
          device,
          network as unknown as Network[],
          new Float32Array(network.input),
        ),
      ).rejects.toThrow('batchActivate expects networks to be an array');
    });

    it('throws when the first network is null or undefined', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);

      await expect(
        batchActivate(
          device,
          [null as unknown as Network, network],
          new Float32Array(network.input * 2),
        ),
      ).rejects.toThrow('batchActivate received a null or undefined network');
    });

    it('throws when a non-first network is null or undefined', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);

      await expect(
        batchActivate(
          device,
          [network, null as unknown as Network],
          new Float32Array(network.input * 2),
        ),
      ).rejects.toThrow(
        'batchActivate received a null or undefined network at index 1',
      );
    });

    it('throws when networks have mismatched input or output sizes', async () => {
      const device = createMockGPUDevice();
      const networkA = Network.createMLP(2, [3], 1);
      const networkB = Network.createMLP(3, [3], 1);

      await expect(
        batchActivate(
          device,
          [networkA, networkB],
          new Float32Array(networkA.input * 2),
        ),
      ).rejects.toThrow(
        'Network at index 1 has shape (input=3, output=1); expected (input=2, output=1)',
      );
    });

    it('throws when a network is not eligible for GPU inference', async () => {
      const device = createMockGPUDevice({
        limits: { maxStorageBufferBindingSize: 1, maxBufferSize: 1 },
      });
      const network = Network.createMLP(2, [3], 1);

      await expect(
        batchActivate(device, [network], new Float32Array(network.input)),
      ).rejects.toThrow(
        'batchActivate: network at index 0 is not eligible for GPU inference',
      );
    });

    it('throws when input matrix length does not match the batch', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const batchSize = 3;

      await expect(
        batchActivate(
          device,
          Array.from({ length: batchSize }, () => network),
          new Float32Array(batchSize * network.input - 1),
        ),
      ).rejects.toThrow(
        'inputMatrix length 5 does not match networks.length * inputCount (6)',
      );
    });

    it('resolves activations by strict identity when possible', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      network.nodes[0].squash = ACTIVATION_FUNCTIONS[0];

      const result = await batchActivate(
        device,
        [network],
        new Float32Array(network.input),
      );

      expect(result.outputs.length).toBe(network.output);
    });

    it('resolves activations by function name as a fallback', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const namedLogistic = function logisticActivation(
        inputValue: number,
      ): number {
        return 1 / (1 + Math.exp(-inputValue));
      };
      network.nodes[0].squash = namedLogistic;

      const result = await batchActivate(
        device,
        [network],
        new Float32Array(network.input),
      );

      expect(result.outputs.length).toBe(network.output);
    });

    it('throws when the first node has no squash function', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      network.nodes[0].squash = undefined as unknown as (
        inputValue: number,
        shouldComputeDerivative?: boolean,
      ) => number;

      await expect(
        batchActivate(device, [network], new Float32Array(network.input)),
      ).rejects.toThrow('activateGPU: first node has no squash function');
    });

    it('throws when the first node uses an unsupported activation', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      function unknownActivation(inputValue: number): number {
        return inputValue * 2 + 1;
      }
      network.nodes[0].squash = unknownActivation;

      await expect(
        batchActivate(device, [network], new Float32Array(network.input)),
      ).rejects.toThrow(
        'activateGPU: first node uses an activation that is not in the worker registry',
      );
    });

    it('falls back through a symbol-key miss to a matching function name', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const activationKeySymbol = Symbol.for('neataptic.activation.key');
      const namedFallback = function logisticActivation(
        inputValue: number,
      ): number {
        return 1 / (1 + Math.exp(-inputValue));
      };
      (namedFallback as unknown as Record<symbol, string>)[
        activationKeySymbol
      ] = 'nonexistent';
      network.nodes[0].squash = namedFallback;

      const result = await batchActivate(
        device,
        [network],
        new Float32Array(network.input),
      );

      expect(result.outputs.length).toBe(network.output);
    });

    it('throws for an activation with no symbol key and no function name', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      network.nodes[0].squash = ((inputValue: number) =>
        inputValue * 2 + 1) as (
        inputValue: number,
        shouldComputeDerivative?: boolean,
      ) => number;

      await expect(
        batchActivate(device, [network], new Float32Array(network.input)),
      ).rejects.toThrow(
        'activateGPU: first node uses an activation that is not in the worker registry',
      );
    });
  });

  describe('batchActivate iterations option', () => {
    it('records the default single pass when iterations is omitted', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const batchSize = 2;

      await batchActivate(
        device,
        Array.from({ length: batchSize }, () => network.clone()),
        new Float32Array(batchSize * network.input),
      );

      expect(device.recorded.mapAsyncCalls.length).toBe(1);
    });

    it('records multiple compute passes when iterations > 1', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const batchSize = 2;
      const iterations = 3;

      await batchActivate(
        device,
        Array.from({ length: batchSize }, () => network.clone()),
        new Float32Array(batchSize * network.input),
        { iterations },
      );

      const singlePassDispatches =
        batchSize * (network.nodes.length > 0 ? 2 : 0);
      expect({
        submissions: device.recorded.submissions.length,
        mapAsyncCalls: device.recorded.mapAsyncCalls.length,
      }).toEqual({
        submissions: singlePassDispatches * iterations,
        mapAsyncCalls: 1,
      });
    });

    it('clamps iterations <= 0 to a single pass', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const batchSize = 1;

      await batchActivate(
        device,
        Array.from({ length: batchSize }, () => network.clone()),
        new Float32Array(batchSize * network.input),
        { iterations: 0 },
      );

      const singlePassDispatches = batchSize * 2;
      expect({
        submissions: device.recorded.submissions.length,
        mapAsyncCalls: device.recorded.mapAsyncCalls.length,
      }).toEqual({
        submissions: singlePassDispatches,
        mapAsyncCalls: 1,
      });
    });

    it('returns an output row for each network when iterations > 1', async () => {
      const referenceNetwork = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice({
        emulateNetwork: referenceNetwork,
      });
      const inputMatrix = new Float32Array([0.1, 0.2]);

      const result = await batchActivate(
        device,
        [referenceNetwork.clone()],
        inputMatrix,
        { iterations: 4 },
      );

      expect(result.outputs.length).toBe(referenceNetwork.output);
    });

    it('matches CPU reference output when iterations > 1', async () => {
      const referenceNetwork = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice({
        emulateNetwork: referenceNetwork,
      });
      const inputMatrix = new Float32Array([0.1, 0.2]);

      const result = await batchActivate(
        device,
        [referenceNetwork.clone()],
        inputMatrix,
        { iterations: 4 },
      );

      const expected = referenceNetwork.activate([0.1, 0.2]);
      const maxDifference = Math.max(
        ...expected.map((value, index) =>
          Math.abs(value - (result.outputs[index] ?? Number.POSITIVE_INFINITY)),
        ),
      );
      expect(maxDifference).toBeLessThan(1e-4);
    });

    it('combines skipUpload and iterations', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const uploadSpy = jest
        .spyOn(gpuBuffer, 'uploadDynamicNetworkBuffers')
        .mockImplementation(() => undefined);
      const writeSpy = jest
        .spyOn(gpuBuffer, 'writeInputValuesToNodeStruct')
        .mockImplementation(() => undefined);

      const result = await batchActivate(
        device,
        [network],
        new Float32Array(network.input),
        { skipUpload: true, iterations: 5 },
      );

      expect({
        uploadCalls: uploadSpy.mock.calls.length,
        writeCalls: writeSpy.mock.calls.length,
        mapAsyncCalls: device.recorded.mapAsyncCalls.length,
        outputLength: result.outputs.length,
      }).toEqual({
        uploadCalls: 0,
        writeCalls: 0,
        mapAsyncCalls: 1,
        outputLength: network.output,
      });

      uploadSpy.mockRestore();
      writeSpy.mockRestore();
    });
  });

  describe('batchActivate persistent staging buffer', () => {
    it('reuses the batched output staging buffer across calls', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const batchSize = 2;
      const inputMatrix = new Float32Array(batchSize * network.input);

      await batchActivate(
        device,
        Array.from({ length: batchSize }, () => network.clone()),
        inputMatrix,
      );
      await batchActivate(
        device,
        Array.from({ length: batchSize }, () => network.clone()),
        inputMatrix,
      );

      expect(
        device.recorded.buffers.filter(
          (buffer) => buffer.label === 'network_batched_outputs_staging',
        ).length,
      ).toBe(1);
    });

    it('recreates the batched output staging buffer when it was destroyed', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const batchSize = 2;
      const inputMatrix = new Float32Array(batchSize * network.input);

      await batchActivate(
        device,
        Array.from({ length: batchSize }, () => network.clone()),
        inputMatrix,
      );
      const stagingBuffer = device.recorded.buffers.find(
        (buffer) => buffer.label === 'network_batched_outputs_staging',
      );
      (stagingBuffer as unknown as { destroyed: boolean }).destroyed = true;

      await batchActivate(
        device,
        Array.from({ length: batchSize }, () => network.clone()),
        inputMatrix,
      );

      expect(
        device.recorded.buffers.filter(
          (buffer) => buffer.label === 'network_batched_outputs_staging',
        ).length,
      ).toBe(2);
    });
  });

  describe('batchActivate bind group validation', () => {
    it('throws when a level bind group is missing', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const originalEnsure = gpuActivate.ensureNetworkGPUState;
      const spy = jest
        .spyOn(gpuActivate, 'ensureNetworkGPUState')
        .mockImplementation((dev, net) => {
          const result = originalEnsure(dev, net);
          result.state.levelBindGroups[1] = undefined;
          return result;
        });

      await expect(
        batchActivate(device, [network], new Float32Array(network.input)),
      ).rejects.toThrow('batchActivate: missing bind group for level 1');

      spy.mockRestore();
    });
  });

  describe('createBatchInferenceQueue', () => {
    it('starts with queue size 0', () => {
      const device = createMockGPUDevice();

      const queue = createBatchInferenceQueue(device);

      expect(queue.size).toBe(0);
    });

    it('returns incrementing job ids and increments queue size', () => {
      const device = createMockGPUDevice();
      const queue = createBatchInferenceQueue(device);
      const network = Network.createMLP(2, [3], 1);

      const firstJobId = queue.enqueue({ network, inputs: [0.1, 0.2] });
      const secondJobId = queue.enqueue({ network, inputs: [0.3, 0.4] });

      expect({
        firstJobId,
        secondJobId,
        size: queue.size,
      }).toEqual({
        firstJobId: 0,
        secondJobId: 1,
        size: 2,
      });
    });

    it('submits multiple enqueued jobs in a single GPU command pass', async () => {
      const device = createMockGPUDevice();
      const queue = createBatchInferenceQueue(device);
      const network = Network.createMLP(2, [3], 1);
      queue.enqueue({ network, inputs: [0.1, 0.2] });
      queue.enqueue({ network, inputs: [0.3, 0.4] });

      await queue.flush();

      expect(device.queue.submit).toHaveBeenCalledTimes(1);
    });

    it('returns outputs in the same order as enqueue', async () => {
      const referenceNetwork = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice({ emulateNetwork: referenceNetwork });
      const queue = createBatchInferenceQueue(device);
      const firstInputs = [0.1, 0.2];
      const secondInputs = [0.3, 0.4];
      queue.enqueue({ network: referenceNetwork.clone(), inputs: firstInputs });
      queue.enqueue({
        network: referenceNetwork.clone(),
        inputs: secondInputs,
      });

      const outputs = await queue.flush();

      expect(outputs).toEqual([
        new Float32Array(referenceNetwork.activate(firstInputs)),
        new Float32Array(referenceNetwork.activate(secondInputs)),
      ]);
    });

    it('shares one compiled pipeline for networks with identical topology', async () => {
      const network = Network.createMLP(2, [3], 1);
      const device = createMockGPUDevice();
      const queue = createBatchInferenceQueue(device);
      queue.enqueue({ network: network.clone(), inputs: [0.1, 0.2] });
      queue.enqueue({ network: network.clone(), inputs: [0.3, 0.4] });

      await queue.flush();

      expect(device.recorded.pipelines.length).toBe(1);
    });

    it('resolves an empty array when the queue is empty', async () => {
      const device = createMockGPUDevice();
      const queue = createBatchInferenceQueue(device);

      const outputs = await queue.flush();

      expect(outputs).toEqual([]);
    });
  });
});
