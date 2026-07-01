import { Activation } from '../../../methods/activation/activation';
import type Network from '../network';
import { canUseGPU } from './network.gpu.capability';
import { createMockGPUDevice } from './__mocks__/gpu.mock';

const RELU_INDEX = 4;
const SUPPORTED_ACTIVATIONS = new Set<number>([RELU_INDEX]);

interface MockGPUCapableNetwork {
  _canUseFastSlab: (training: boolean) => boolean;
  gates: unknown[];
  selfconns: unknown[];
  dropout: number;
  _enforceAcyclic: boolean;
  _topoDirty: boolean;
  _weightNoiseStd: number;
  _weightNoisePerHidden?: unknown[];
  _stochasticDepth?: unknown[];
  nodes: unknown[];
  connections: unknown[];
  input: number;
  output: number;
}

function reluWithIndex(): ((x: number, derivate?: boolean) => number) & {
  index: number;
} {
  return Object.assign(
    (x: number, derivate?: boolean) => Activation.relu(x, derivate),
    { index: RELU_INDEX },
  );
}

function createSlabEligibleNetwork(
  overrides: Partial<MockGPUCapableNetwork> = {},
): MockGPUCapableNetwork {
  const squash = reluWithIndex();
  const base: MockGPUCapableNetwork = {
    _canUseFastSlab: () => true,
    gates: [],
    selfconns: [],
    dropout: 0,
    _enforceAcyclic: true,
    _topoDirty: false,
    _weightNoiseStd: 0,
    _weightNoisePerHidden: undefined,
    _stochasticDepth: undefined,
    nodes: [{ squash }, { squash }, { squash }],
    connections: [],
    input: 2,
    output: 1,
  };
  return Object.assign(base, overrides);
}

describe('network.gpu.capability', () => {
  describe('canUseGPU', () => {
    it('returns false when no device is supplied', () => {
      const network = createSlabEligibleNetwork();
      expect(
        canUseGPU(network as unknown as Network, null, SUPPORTED_ACTIVATIONS),
      ).toBe(false);
    });

    it('returns false for a gated network', () => {
      const network = createSlabEligibleNetwork({
        gates: [{}],
        _canUseFastSlab: () => false,
      });
      const device = createMockGPUDevice();
      expect(
        canUseGPU(network as unknown as Network, device, SUPPORTED_ACTIVATIONS),
      ).toBe(false);
    });

    it('returns false for a self-connected network', () => {
      const network = createSlabEligibleNetwork({
        selfconns: [{}],
      });
      const device = createMockGPUDevice();
      expect(
        canUseGPU(network as unknown as Network, device, SUPPORTED_ACTIVATIONS),
      ).toBe(false);
    });

    it('returns false when a connection is self-referential', () => {
      const network = createSlabEligibleNetwork({
        connections: [{ from: 0, to: 0 }],
      });
      const device = createMockGPUDevice();
      expect(
        canUseGPU(network as unknown as Network, device, SUPPORTED_ACTIVATIONS),
      ).toBe(false);
    });

    it('returns false when a node uses an unsupported activation', () => {
      const unsupportedSquash = Object.assign(
        (x: number, derivate?: boolean) => Activation.logistic(x, derivate),
        { index: 99 },
      );
      const network = createSlabEligibleNetwork({
        nodes: [{ squash: unsupportedSquash }],
      });
      const device = createMockGPUDevice();
      expect(
        canUseGPU(network as unknown as Network, device, SUPPORTED_ACTIVATIONS),
      ).toBe(false);
    });

    it('returns false when estimated buffer size exceeds device limits', () => {
      const network = createSlabEligibleNetwork({
        nodes: Array.from({ length: 1000 }, () => ({
          squash: reluWithIndex(),
        })),
      });
      const device = createMockGPUDevice({ maxStorageBufferBindingSize: 1024 });
      expect(
        canUseGPU(network as unknown as Network, device, SUPPORTED_ACTIVATIONS),
      ).toBe(false);
    });

    it('returns true only for a fully eligible network', () => {
      const network = createSlabEligibleNetwork();
      const device = createMockGPUDevice();
      expect(
        canUseGPU(network as unknown as Network, device, SUPPORTED_ACTIVATIONS),
      ).toBe(true);
    });
  });
});
