import Network from '../network';
import { batchActivate } from './network.gpu.batched';
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

    it('records one GPU dispatch submission per batch item', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const batchSize = 3;
      const inputMatrix = new Float32Array(batchSize * network.input);

      await batchActivate(
        device,
        Array.from({ length: batchSize }, () => network),
        inputMatrix,
      );

      expect(device.recorded.submissions.length).toBe(batchSize);
    });

    it('creates and maps a result buffer for read-back', async () => {
      const device = createMockGPUDevice();
      const network = Network.createMLP(2, [3], 1);
      const batchSize = 2;
      const inputMatrix = new Float32Array(batchSize * network.input);

      await batchActivate(
        device,
        Array.from({ length: batchSize }, () => network),
        inputMatrix,
      );

      expect(device.recorded.mapAsyncCalls.length).toBeGreaterThan(0);
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
  });
});
