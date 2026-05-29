import type { ActivationSchedule } from '../network.types';
import Connection from '../../connection';
import Node from '../../node';
import type { ActivationFunction } from '../../../methods/activation/activation.utils';
import { resolveActivationKey } from '../serialize/network.serialize.activation.utils';
import { Architect, methods } from '../../../neataptic';
import {
  createInferencePredictor,
  detectInferenceWorkerCapabilities,
  openSharedInferenceWorker,
  openInferenceChannel,
  exportTransferableInferencePayload,
  exportPortableInferencePayload,
  extractNetworkInferenceIR,
  getTransferList,
  INFERENCE_ACTIVATION_TABLE,
  resolveBrowserWorkerAssetUrl,
  resolveAutoInferenceTransport,
} from './network.worker-payload';
import { INFERENCE_CHANNEL_HOST_INTERNALS } from './network.worker-payload.channel';
import {
  INFERENCE_CHANNEL_WORKER_INTERNALS,
  registerInferenceChannelWorkerRuntime,
} from './network.worker-payload.channel.worker';
import { SHARED_INFERENCE_HOST_INTERNALS } from './network.worker-payload.shared';
import {
  SHARED_INFERENCE_WORKER_INTERNALS,
  registerSharedInferenceWorkerRuntime,
} from './network.worker-payload.shared.worker';
import type {
  PortableInferencePayload,
  TransferableInferencePayload,
} from './network.worker-payload';
import { SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION } from './network.worker-payload';

type PortablePayload = PortableInferencePayload;
type TransferablePayload = TransferableInferencePayload;

const MANUAL_PORTABLE_ACTIVATION_TABLE = [
  'logistic',
  'tanh',
  'identity',
  'step',
  'relu',
  'softsign',
  'sinusoid',
  'gaussian',
  'bentIdentity',
  'bipolar',
  'bipolarSigmoid',
  'hardTanh',
  'absolute',
  'inverse',
  'selu',
  'softplus',
  'swish',
  'gelu',
  'mish',
];

function roundValue(value: number) {
  return Number(value.toFixed(12));
}

function roundVector(values: readonly number[]) {
  return values.map((value) => roundValue(value));
}

function createPortableRoundtripOutputs(inputValues: readonly number[]) {
  const network = createWorkerPayloadNetwork();
  disableFastSlab(network);
  const payload = exportPortableInferencePayload(network);
  const predictor = createInferencePredictor(payload);

  return {
    payload,
    predictedOutput: predictor.predict(inputValues),
    runtimeOutput: network.noTraceActivate([...inputValues]),
  };
}

function createTransferableRoundtripOutputs(
  inputValues: readonly number[],
  options?: Parameters<typeof exportTransferableInferencePayload>[1],
) {
  const network = createWorkerPayloadNetwork();
  disableFastSlab(network);
  const portablePayload = exportPortableInferencePayload(network);
  const transferablePayload = exportTransferableInferencePayload(
    network,
    options,
  );
  const portablePredictor = createInferencePredictor(portablePayload);
  const transferablePredictor = createInferencePredictor(transferablePayload);

  return {
    portableOutput: portablePredictor.predict(inputValues),
    runtimeOutput: network.noTraceActivate([...inputValues]),
    transferableOutput: transferablePredictor.predict(inputValues),
  };
}

function resolveOutputNodeIndexes(network: { nodes: readonly Node[] }) {
  return network.nodes.flatMap((node) => {
    if (node.type !== 'output') {
      return [];
    }

    return [node.index];
  });
}

function disableFastSlab(network: object) {
  Reflect.set(network, '_canUseFastSlab', () => false);
}

function createWorkerPayloadNetwork() {
  const network = Architect.perceptron(2, 2, 1);
  const hiddenNodes = network.nodes.filter(
    (candidateNode) => candidateNode.type === 'hidden',
  );
  const outputNode = network.nodes.find(
    (candidateNode) => candidateNode.type === 'output',
  );

  if (!hiddenNodes[0] || !hiddenNodes[1] || !outputNode) {
    throw new Error(
      'Expected a perceptron with two hidden nodes and one output node.',
    );
  }

  hiddenNodes[0].squash = methods.Activation.relu;
  hiddenNodes[1].squash = methods.Activation.gaussian;
  hiddenNodes[1].response = 0.5;
  outputNode.mask = 0.25;
  outputNode.squash = methods.Activation.identity;

  return network;
}

function createWorkerPayloadNetworkParts() {
  const network = createWorkerPayloadNetwork();
  const hiddenNodes = network.nodes.filter(
    (candidateNode) => candidateNode.type === 'hidden',
  );
  const outputNode = network.nodes.find(
    (candidateNode) => candidateNode.type === 'output',
  );

  if (!hiddenNodes[0] || !hiddenNodes[1] || !outputNode) {
    throw new Error(
      'Expected a perceptron with two hidden nodes and one output node.',
    );
  }

  return {
    hiddenNodes,
    network,
    outputNode,
  };
}

function createSyntheticTransferablePayload(
  inputCount: number,
  outputCount: number,
): TransferablePayload {
  return {
    activationStepsData: new Int32Array(),
    activationStepsIndex: new Int32Array([0]),
    activationTableLength: INFERENCE_ACTIVATION_TABLE.length,
    edgeFrom: new Int32Array(),
    edgeGaterIndices: new Int32Array(),
    edgeTo: new Int32Array(),
    edgeWeights: new Float64Array(),
    inputCount,
    nodeActivationIds: new Int32Array(),
    nodeBiases: new Float64Array(),
    nodeIds: new Int32Array(),
    nodeMasks: new Float64Array(),
    nodeResponses: new Float64Array(),
    nodeSelfGaterIndices: new Int32Array(),
    nodeSelfWeights: new Float64Array(),
    outputCount,
    outputNodeIndices: new Int32Array(outputCount),
    strategy: 'transferable',
    version: 1,
  };
}

function withProcessOverride<T>(
  nextProcess: NodeJS.Process | undefined,
  callback: () => T,
): T {
  const originalProcess = globalThis.process;

  Reflect.set(globalThis, 'process', nextProcess);

  try {
    return callback();
  } finally {
    Reflect.set(globalThis, 'process', originalProcess);
  }
}

function withCrossOriginIsolationOverride<T>(
  nextCrossOriginIsolated: boolean | undefined,
  callback: () => T,
): T {
  const descriptor = Object.getOwnPropertyDescriptor(
    globalThis,
    'crossOriginIsolated',
  );

  Object.defineProperty(globalThis, 'crossOriginIsolated', {
    configurable: true,
    value: nextCrossOriginIsolated,
  });

  try {
    return callback();
  } finally {
    if (descriptor) {
      Object.defineProperty(globalThis, 'crossOriginIsolated', descriptor);
    } else {
      Reflect.deleteProperty(globalThis, 'crossOriginIsolated');
    }
  }
}

function withBrowserWorkerGlobals<T>(
  callback: (state: {
    createdChannels: Array<{
      port1: {
        emit: (type: string, event: unknown) => void;
        isClosed: boolean;
        postedMessages: Array<{
          message: unknown;
          transferList: Transferable[];
        }>;
        startCallCount: number;
      };
      port2: {
        emit: (type: string, event: unknown) => void;
        isClosed: boolean;
        postedMessages: Array<{
          message: unknown;
          transferList: Transferable[];
        }>;
        startCallCount: number;
      };
    }>;
    createdWorkers: Array<{
      emit: (type: string, event: unknown) => void;
      options: { type: 'module' };
      scriptUrl: string;
    }>;
  }) => T,
): T {
  const originalWorker = globalThis.Worker;
  const originalMessageChannel = globalThis.MessageChannel;
  const createdChannels: Array<{
    port1: {
      emit: (type: string, event: unknown) => void;
      isClosed: boolean;
      postedMessages: Array<{ message: unknown; transferList: Transferable[] }>;
      startCallCount: number;
    };
    port2: {
      emit: (type: string, event: unknown) => void;
      isClosed: boolean;
      postedMessages: Array<{ message: unknown; transferList: Transferable[] }>;
      startCallCount: number;
    };
  }> = [];
  const createdWorkers: Array<{
    emit: (type: string, event: unknown) => void;
    options: { type: 'module' };
    postedMessages: Array<{
      message: unknown;
      transferList: Transferable[];
    }>;
    scriptUrl: string;
  }> = [];

  class FakeBrowserMessagePort {
    public isClosed = false;
    public postedMessages: Array<{
      message: unknown;
      transferList: Transferable[];
    }> = [];
    public readonly listeners = new Map<string, (event: unknown) => void>();
    public startCallCount = 0;

    addEventListener(type: string, listener: (event: unknown) => void) {
      this.listeners.set(type, listener);
    }

    close() {
      this.isClosed = true;
    }

    emit(type: string, event: unknown) {
      this.listeners.get(type)?.(event);
    }

    postMessage(message: unknown, transferList: Transferable[] = []) {
      this.postedMessages.push({ message, transferList });
    }

    start() {
      this.startCallCount += 1;
    }
  }

  class FakeBrowserMessageChannel {
    public readonly port1 = new FakeBrowserMessagePort();
    public readonly port2 = new FakeBrowserMessagePort();

    constructor() {
      createdChannels.push(
        this as unknown as {
          port1: {
            emit: (type: string, event: unknown) => void;
            isClosed: boolean;
            postedMessages: Array<{
              message: unknown;
              transferList: Transferable[];
            }>;
            startCallCount: number;
          };
          port2: {
            emit: (type: string, event: unknown) => void;
            isClosed: boolean;
            postedMessages: Array<{
              message: unknown;
              transferList: Transferable[];
            }>;
            startCallCount: number;
          };
        },
      );
    }
  }

  class FakeBrowserWorker {
    public postedMessages: Array<{
      message: unknown;
      transferList: Transferable[];
    }> = [];
    public readonly listeners = new Map<string, (event: unknown) => void>();

    constructor(
      public readonly scriptUrl: string,
      public readonly options: { type: 'module' },
    ) {
      createdWorkers.push(
        this as unknown as {
          emit: (type: string, event: unknown) => void;
          options: { type: 'module' };
          postedMessages: Array<{
            message: unknown;
            transferList: Transferable[];
          }>;
          scriptUrl: string;
        },
      );
    }

    addEventListener(type: string, listener: (event: unknown) => void) {
      this.listeners.set(type, listener);
    }

    emit(type: string, event: unknown) {
      this.listeners.get(type)?.(event);
    }

    postMessage(message: unknown, transferList: Transferable[] = []) {
      this.postedMessages.push({ message, transferList });
    }

    terminate() {
      return 0;
    }
  }

  Reflect.set(globalThis, 'MessageChannel', FakeBrowserMessageChannel);
  Reflect.set(globalThis, 'Worker', FakeBrowserWorker);

  try {
    return callback({ createdChannels, createdWorkers });
  } finally {
    Reflect.set(globalThis, 'MessageChannel', originalMessageChannel);
    Reflect.set(globalThis, 'Worker', originalWorker);
  }
}

function createEventTargetPortHarness() {
  const listeners = new Map<string, (event: unknown) => void>();
  const postedMessages: Array<{
    message: unknown;
    transferList: Transferable[];
  }> = [];
  let isClosed = false;
  let startCallCount = 0;

  return {
    emit(type: string, event: unknown) {
      listeners.get(type)?.(event);
    },
    getSnapshot() {
      return {
        isClosed,
        postedMessages,
        startCallCount,
      };
    },
    port: {
      addEventListener(type: string, listener: (event: unknown) => void) {
        listeners.set(type, listener);
      },
      close() {
        isClosed = true;
      },
      postMessage(message: unknown, transferList: Transferable[] = []) {
        postedMessages.push({ message, transferList });
      },
      start() {
        startCallCount += 1;
      },
    },
  };
}

function createEmitterPortHarness() {
  const listeners = new Map<string, (...args: unknown[]) => void>();
  const postedMessages: Array<{
    message: unknown;
    transferList: Transferable[];
  }> = [];
  let isClosed = false;

  return {
    emit(type: string, event: unknown) {
      listeners.get(type)?.(event);
    },
    getSnapshot() {
      return {
        isClosed,
        postedMessages,
      };
    },
    port: {
      close() {
        isClosed = true;
      },
      on(type: string, listener: (...args: unknown[]) => void) {
        listeners.set(type, listener);
      },
      postMessage(message: unknown, transferList: Transferable[] = []) {
        postedMessages.push({ message, transferList });
      },
    },
  };
}

function createEventTargetWorkerHarness() {
  const listeners = new Map<string, (event: unknown) => void>();

  return {
    emit(type: string, event: unknown) {
      listeners.get(type)?.(event);
    },
    worker: {
      addEventListener(type: string, listener: (event: unknown) => void) {
        listeners.set(type, listener);
      },
      postMessage() {
        return undefined;
      },
      terminate() {
        return Promise.resolve(0);
      },
    },
  };
}

function createEmitterWorkerHarness() {
  const listeners = new Map<string, (...args: unknown[]) => void>();

  return {
    emit(type: string, event: unknown) {
      listeners.get(type)?.(event);
    },
    worker: {
      on(type: string, listener: (...args: unknown[]) => void) {
        listeners.set(type, listener);
      },
      postMessage() {
        return undefined;
      },
      terminate() {
        return Promise.resolve(0);
      },
    },
  };
}

function resolveExpectedActivationSteps() {
  const network = createWorkerPayloadNetwork();
  extractNetworkInferenceIR(network);

  const activationSchedule = Reflect.get(
    network,
    '_activationSchedule',
  ) as ActivationSchedule;
  const nodeIndexesByGeneId = new Map(
    network.nodes.map((node) => [node.geneId, node.index] as const),
  );

  return activationSchedule.steps.flatMap((activationStep) => {
    const stepIndexes = activationStep.nodeIds
      .map((nodeGeneId) => {
        const matchingNode = network.nodes.find(
          (node) => node.geneId === nodeGeneId,
        );

        if (matchingNode?.type === 'input') {
          return null;
        }

        const nodeIndex = nodeIndexesByGeneId.get(nodeGeneId);

        if (typeof nodeIndex !== 'number') {
          throw new Error(`Expected node index for gene id ${nodeGeneId}.`);
        }

        return nodeIndex;
      })
      .filter(
        (nodeIndex): nodeIndex is number => typeof nodeIndex === 'number',
      );

    if (stepIndexes.length === 0) {
      return [];
    }

    const iterationCount =
      activationStep.kind === 'recurrent-component'
        ? (activationStep.iterations ?? 1)
        : 1;

    return Array.from({ length: iterationCount }, () => [...stepIndexes]);
  });
}

function createManualPortablePayload(
  overrides: Partial<PortablePayload> = {},
): PortablePayload {
  return {
    version: 1,
    strategy: 'portable',
    inputCount: 1,
    outputCount: 1,
    activationSteps: [[1]],
    nodes: [
      {
        id: 0,
        bias: 0,
        response: 1,
        mask: 1,
        activation: 'identity',
        selfWeight: 0,
        selfGaterIndex: -1,
      },
      {
        id: 1,
        bias: 0,
        response: 1,
        mask: 1,
        activation: 'identity',
        selfWeight: 0,
        selfGaterIndex: -1,
      },
    ],
    edges: [
      {
        from: 0,
        to: 1,
        weight: 1,
        gaterIndex: -1,
      },
    ],
    outputNodeIndices: [1],
    activationTable: [...MANUAL_PORTABLE_ACTIVATION_TABLE],
    ...overrides,
  };
}

function createManualTransferablePayload(
  overrides: Partial<TransferablePayload> = {},
): TransferablePayload {
  return {
    version: 1,
    strategy: 'transferable',
    inputCount: 1,
    outputCount: 1,
    activationStepsIndex: new Int32Array([0]),
    activationStepsData: new Int32Array([1]),
    nodeIds: new Int32Array([0, 1]),
    nodeBiases: new Float64Array([0, 0]),
    nodeResponses: new Float64Array([1, 1]),
    nodeMasks: new Float64Array([1, 1]),
    nodeActivationIds: new Int32Array([2, 2]),
    nodeSelfWeights: new Float64Array([0, 0]),
    nodeSelfGaterIndices: new Int32Array([-1, -1]),
    edgeFrom: new Int32Array([0]),
    edgeTo: new Int32Array([1]),
    edgeWeights: new Float64Array([1]),
    edgeGaterIndices: new Int32Array([-1]),
    outputNodeIndices: new Int32Array([1]),
    activationTableLength: INFERENCE_ACTIVATION_TABLE.length,
    ...overrides,
  };
}

function flattenActivationSteps(
  activationSteps: ReadonlyArray<ReadonlyArray<number>>,
) {
  const activationStepsIndex: number[] = [];
  const activationStepsData: number[] = [];
  let nextActivationStepOffset = 0;

  for (const activationStep of activationSteps) {
    activationStepsIndex.push(nextActivationStepOffset);
    activationStepsData.push(...activationStep);
    nextActivationStepOffset += activationStep.length;
  }

  return {
    activationStepsData,
    activationStepsIndex,
  };
}

function createSharedInferenceBufferPair(
  inputCount: number,
  outputCount: number,
) {
  const layout =
    SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout(
      inputCount,
      outputCount,
    );

  return {
    controlBuffer: new SharedArrayBuffer(
      Int32Array.BYTES_PER_ELEMENT * layout.controlElementCount,
    ),
    dataBuffer: new SharedArrayBuffer(
      Float64Array.BYTES_PER_ELEMENT * layout.dataElementCount,
    ),
    layout,
  };
}

describe('network worker payload chapter', () => {
  describe('transport capability detection', () => {
    it('explains browser shared-memory gating and falls back to channel transport when isolation is missing', async () => {
      // Act
      const capabilities = await detectInferenceWorkerCapabilities({
        crossOriginIsolated: false,
        hasChannelWorker: true,
        hasSharedWorker: true,
        runtime: 'browser',
        workerConstructorAvailable: true,
      });
      const result = {
        capabilities,
        selectedTransport: resolveAutoInferenceTransport(capabilities),
      };

      // Assert
      expect(result).toEqual({
        capabilities: {
          inferenceChannel: true,
          reasons: [
            'Shared-memory inference requires crossOriginIsolated=true in browser hosts.',
          ],
          runtime: 'browser',
          sharedMemory: false,
          transferable: true,
        },
        selectedTransport: 'channel',
      });
    });

    it('falls back to transferable transport when browser worker delivery is unavailable', async () => {
      // Act
      const capabilities = await detectInferenceWorkerCapabilities({
        crossOriginIsolated: true,
        hasChannelWorker: false,
        hasSharedWorker: false,
        runtime: 'browser',
        workerConstructorAvailable: true,
      });
      const result = {
        capabilities,
        selectedTransport: resolveAutoInferenceTransport(capabilities),
      };

      // Assert
      expect(result).toEqual({
        capabilities: {
          inferenceChannel: false,
          reasons: [
            'InferenceChannel browser transport needs one worker delivery path.',
            'Shared-memory inference needs one shared worker delivery path in this host.',
          ],
          runtime: 'browser',
          sharedMemory: false,
          transferable: true,
        },
        selectedTransport: 'transferable',
      });
    });

    it('explains browser fallback when the host runtime does not expose Worker support', async () => {
      // Act
      const capabilities = detectInferenceWorkerCapabilities({
        crossOriginIsolated: true,
        hasChannelWorker: true,
        hasSharedWorker: true,
        runtime: 'browser',
        sharedArrayBufferAvailable: true,
        workerConstructorAvailable: false,
      });
      const result = {
        capabilities,
        selectedTransport: resolveAutoInferenceTransport(capabilities),
      };

      // Assert
      expect(result).toEqual({
        capabilities: {
          inferenceChannel: false,
          reasons: [
            'InferenceChannel browser transport requires Worker support in the host runtime.',
            'Shared-memory inference requires Worker support in the host runtime.',
          ],
          runtime: 'browser',
          sharedMemory: false,
          transferable: true,
        },
        selectedTransport: 'transferable',
      });
    });

    it('keeps channel transport available when browser shared memory lacks SharedArrayBuffer support', async () => {
      // Act
      const capabilities = detectInferenceWorkerCapabilities({
        crossOriginIsolated: true,
        hasChannelWorker: true,
        hasSharedWorker: true,
        runtime: 'browser',
        sharedArrayBufferAvailable: false,
        workerConstructorAvailable: true,
      });
      const result = {
        capabilities,
        selectedTransport: resolveAutoInferenceTransport(capabilities),
      };

      // Assert
      expect(result).toEqual({
        capabilities: {
          inferenceChannel: true,
          reasons: [
            'Shared-memory inference requires SharedArrayBuffer support in the host runtime.',
          ],
          runtime: 'browser',
          sharedMemory: false,
          transferable: true,
        },
        selectedTransport: 'channel',
      });
    });

    it('prefers shared-memory transport in node hosts when shared buffers are available', async () => {
      // Act
      const capabilities = await detectInferenceWorkerCapabilities({
        runtime: 'node',
        sharedArrayBufferAvailable: true,
      });
      const result = {
        capabilities,
        selectedTransport: resolveAutoInferenceTransport(capabilities),
      };

      // Assert
      expect(result).toEqual({
        capabilities: {
          inferenceChannel: true,
          reasons: [],
          runtime: 'node',
          sharedMemory: true,
          transferable: true,
        },
        selectedTransport: 'shared-memory',
      });
    });

    it('uses shared-memory transport when browser capability facts come from the runtime globals', () => {
      // Act
      const capabilities = withProcessOverride(undefined, () =>
        withBrowserWorkerGlobals(() =>
          detectInferenceWorkerCapabilities({
            crossOriginIsolated: true,
            hasChannelWorker: true,
            hasSharedWorker: true,
          }),
        ),
      );
      const result = {
        capabilities,
        selectedTransport: resolveAutoInferenceTransport(capabilities),
      };

      // Assert
      expect(result).toEqual({
        capabilities: {
          inferenceChannel: true,
          reasons: [],
          runtime: 'browser',
          sharedMemory: true,
          transferable: true,
        },
        selectedTransport: 'shared-memory',
      });
    });

    it('reads browser isolation from the host global when the option is omitted', () => {
      // Act
      const capabilities = withProcessOverride(undefined, () =>
        withBrowserWorkerGlobals(() =>
          withCrossOriginIsolationOverride(false, () =>
            detectInferenceWorkerCapabilities({
              hasChannelWorker: true,
              hasSharedWorker: true,
            }),
          ),
        ),
      );
      const result = {
        capabilities,
        selectedTransport: resolveAutoInferenceTransport(capabilities),
      };

      // Assert
      expect(result).toEqual({
        capabilities: {
          inferenceChannel: true,
          reasons: [
            'Shared-memory inference requires crossOriginIsolated=true in browser hosts.',
          ],
          runtime: 'browser',
          sharedMemory: false,
          transferable: true,
        },
        selectedTransport: 'channel',
      });
    });

    it('defaults missing browser delivery facts to transferable fallback', () => {
      // Act
      const capabilities = withProcessOverride(undefined, () =>
        withBrowserWorkerGlobals(() =>
          detectInferenceWorkerCapabilities({
            crossOriginIsolated: true,
          }),
        ),
      );
      const result = {
        capabilities,
        selectedTransport: resolveAutoInferenceTransport(capabilities),
      };

      // Assert
      expect(result).toEqual({
        capabilities: {
          inferenceChannel: false,
          reasons: [
            'InferenceChannel browser transport needs one worker delivery path.',
            'Shared-memory inference needs one shared worker delivery path in this host.',
          ],
          runtime: 'browser',
          sharedMemory: false,
          transferable: true,
        },
        selectedTransport: 'transferable',
      });
    });

    it('falls back to channel transport in node hosts when shared buffers are unavailable', async () => {
      // Act
      const capabilities = detectInferenceWorkerCapabilities({
        runtime: 'node',
        sharedArrayBufferAvailable: false,
      });
      const result = {
        capabilities,
        selectedTransport: resolveAutoInferenceTransport(capabilities),
      };

      // Assert
      expect(result).toEqual({
        capabilities: {
          inferenceChannel: true,
          reasons: [
            'Shared-memory inference requires SharedArrayBuffer support in the host runtime.',
          ],
          runtime: 'node',
          sharedMemory: false,
          transferable: true,
        },
        selectedTransport: 'channel',
      });
    });

    it('detects node runtime automatically when process exposes cwd', () => {
      // Act
      const capabilities = detectInferenceWorkerCapabilities();
      const result = {
        capabilities,
        selectedTransport: resolveAutoInferenceTransport(capabilities),
      };

      // Assert
      expect(result).toEqual({
        capabilities: {
          inferenceChannel: true,
          reasons:
            typeof globalThis.SharedArrayBuffer === 'function'
              ? []
              : [
                  'Shared-memory inference requires SharedArrayBuffer support in the host runtime.',
                ],
          runtime: 'node',
          sharedMemory: typeof globalThis.SharedArrayBuffer === 'function',
          transferable: true,
        },
        selectedTransport:
          typeof globalThis.SharedArrayBuffer === 'function'
            ? 'shared-memory'
            : 'channel',
      });
    });

    it('detects browser runtime automatically when process is unavailable', () => {
      // Act
      const capabilities = withProcessOverride(undefined, () =>
        detectInferenceWorkerCapabilities({
          crossOriginIsolated: true,
          hasChannelWorker: true,
          runtime: 'auto',
          workerConstructorAvailable: true,
        }),
      );
      const result = {
        capabilities,
        selectedTransport: resolveAutoInferenceTransport(capabilities),
      };

      // Assert
      expect(result).toEqual({
        capabilities: {
          inferenceChannel: true,
          reasons: [
            'Shared-memory inference needs one shared worker delivery path in this host.',
          ],
          runtime: 'browser',
          sharedMemory: false,
          transferable: true,
        },
        selectedTransport: 'channel',
      });
    });

    it('treats process records without cwd as browser hosts for automatic runtime detection', () => {
      // Act
      const capabilities = withProcessOverride({} as NodeJS.Process, () =>
        detectInferenceWorkerCapabilities({
          crossOriginIsolated: true,
          hasChannelWorker: true,
          runtime: 'auto',
          workerConstructorAvailable: true,
        }),
      );
      const result = {
        capabilities,
        selectedTransport: resolveAutoInferenceTransport(capabilities),
      };

      // Assert
      expect(result).toEqual({
        capabilities: {
          inferenceChannel: true,
          reasons: [
            'Shared-memory inference needs one shared worker delivery path in this host.',
          ],
          runtime: 'browser',
          sharedMemory: false,
          transferable: true,
        },
        selectedTransport: 'channel',
      });
    });
  });

  describe('inference channels', () => {
    describe('when one transferable payload bootstraps a persistent channel worker', () => {
      it('predicts the same output as the runtime network for the same input vector', async () => {
        // Arrange
        const inputValues = [0.25, 0.75];
        const network = createWorkerPayloadNetwork();
        disableFastSlab(network);
        const payload = exportTransferableInferencePayload(network);
        const channel = openInferenceChannel(payload);

        try {
          // Act
          const outputValues = await channel.predict(inputValues);

          // Assert
          expect(roundVector(Array.from(outputValues))).toEqual(
            roundVector(network.noTraceActivate([...inputValues])),
          );
        } finally {
          await channel.close();
        }
      });
    });

    describe('when the host closes the channel', () => {
      it('rejects subsequent predict calls with a descriptive closed-channel error', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );
        const channel = openInferenceChannel(payload);
        await channel.close();

        // Assert
        await expect(channel.predict([0.25, 0.75])).rejects.toThrow(
          'InferenceChannel is closed.',
        );
      });
    });

    describe('when concurrent predictions exceed the configured in-flight limit', () => {
      it('resolves queued predictions in input order with the same outputs as the runtime network', async () => {
        // Arrange
        const inputVectors = [
          [0.1, 0.9],
          [0.25, 0.75],
          [0.5, 0.5],
          [0.75, 0.25],
        ];
        const network = createWorkerPayloadNetwork();
        disableFastSlab(network);
        const channel = openInferenceChannel(
          exportTransferableInferencePayload(network),
          {
            maxConcurrentRequests: 2,
          },
        );

        try {
          // Act
          const channelOutputs = await Promise.all(
            inputVectors.map((inputVector) => channel.predict(inputVector)),
          );

          // Assert
          expect(
            channelOutputs.map((outputValues) =>
              roundVector(Array.from(outputValues)),
            ),
          ).toEqual(
            inputVectors.map((inputVector) =>
              roundVector(network.noTraceActivate([...inputVector])),
            ),
          );
        } finally {
          await channel.close();
        }
      });
    });

    describe('when one channel resets a recurrent predictor after warm predictions', () => {
      it('matches a cleared runtime baseline after the reset call', async () => {
        // Arrange
        const inputValues = [0.5, 0.25];
        const network = createWorkerPayloadNetwork();
        const recurrentOutputNode = network.nodes.find(
          (candidateNode) => candidateNode.type === 'output',
        );

        if (!recurrentOutputNode) {
          throw new Error(
            'Expected an output node for the channel reset test.',
          );
        }

        const selfConnection = new Connection(
          recurrentOutputNode,
          recurrentOutputNode,
          0.5,
        );
        recurrentOutputNode.connections.self = [selfConnection];
        network.selfconns = [selfConnection];
        disableFastSlab(network);

        const channel = openInferenceChannel(
          exportTransferableInferencePayload(network),
        );

        try {
          // Act
          const firstChannelOutput = roundVector(
            Array.from(await channel.predict(inputValues)),
          );
          const secondChannelOutput = roundVector(
            Array.from(await channel.predict(inputValues)),
          );
          await channel.reset();
          const resetChannelOutput = roundVector(
            Array.from(await channel.predict(inputValues)),
          );

          const firstRuntimeOutput = roundVector(
            network.noTraceActivate([...inputValues]),
          );
          const secondRuntimeOutput = roundVector(
            network.noTraceActivate([...inputValues]),
          );
          network.clear();
          const resetRuntimeOutput = roundVector(
            network.noTraceActivate([...inputValues]),
          );

          // Assert
          expect({
            channel: [
              firstChannelOutput,
              secondChannelOutput,
              resetChannelOutput,
            ],
            runtime: [
              firstRuntimeOutput,
              secondRuntimeOutput,
              resetRuntimeOutput,
            ],
          }).toEqual({
            channel: [
              firstRuntimeOutput,
              secondRuntimeOutput,
              resetRuntimeOutput,
            ],
            runtime: [
              firstRuntimeOutput,
              secondRuntimeOutput,
              resetRuntimeOutput,
            ],
          });
        } finally {
          await channel.close();
        }
      });
    });

    describe('host internal helpers', () => {
      it('normalizes host-side error helpers and event payload helpers', () => {
        // Arrange
        const primitiveError = {
          [Symbol.toPrimitive]() {
            throw new Error('cannot stringify');
          },
        };

        // Act
        const result = {
          closedChannelError:
            INFERENCE_CHANNEL_HOST_INTERNALS.createClosedChannelError().message,
          customError: (() => {
            const normalizedError = INFERENCE_CHANNEL_HOST_INTERNALS.asError({
              message: 'custom failure',
              name: 'CustomError',
              stack: 'custom-stack',
            });

            return {
              message: normalizedError.message,
              name: normalizedError.name,
              stack: normalizedError.stack,
            };
          })(),
          dataMessage: INFERENCE_CHANNEL_HOST_INTERNALS.resolveMessageEventData(
            {
              data: 'payload',
            },
          ),
          directError: INFERENCE_CHANNEL_HOST_INTERNALS.asError(
            new Error('boom'),
          ).message,
          primitiveStringError:
            INFERENCE_CHANNEL_HOST_INTERNALS.asError('string boom').message,
          fallbackWorkerError:
            INFERENCE_CHANNEL_HOST_INTERNALS.resolveWorkerError({
              message: 'worker failed',
            }).message,
          unknownError:
            INFERENCE_CHANNEL_HOST_INTERNALS.asError(primitiveError).message,
          workerError: INFERENCE_CHANNEL_HOST_INTERNALS.resolveWorkerError({
            error: new Error('worker exploded'),
          }).message,
        };

        // Assert
        expect(result).toEqual({
          closedChannelError: 'InferenceChannel is closed.',
          customError: {
            message: 'custom failure',
            name: 'CustomError',
            stack: 'custom-stack',
          },
          dataMessage: 'payload',
          directError: 'boom',
          primitiveStringError: 'string boom',
          fallbackWorkerError: 'worker failed',
          unknownError: 'InferenceChannel worker failed with an unknown error.',
          workerError: 'worker exploded',
        });
      });

      it('resolves browser and node bootstrap contracts for the host transport helpers', () => {
        // Arrange
        const originalJestWorkerId = process.env.JEST_WORKER_ID;
        process.env.JEST_WORKER_ID = '1';

        // Act
        const result = withBrowserWorkerGlobals(() => {
          const browserWorker =
            INFERENCE_CHANNEL_HOST_INTERNALS.createBrowserInferenceChannelWorker(
              'worker-entry.js',
              'worker-entry.js',
            );
          const nodeSpecifier =
            INFERENCE_CHANNEL_HOST_INTERNALS.resolveInferenceChannelWorkerSpecifier(
              undefined,
            );
          const sourceWorkerPath =
            INFERENCE_CHANNEL_HOST_INTERNALS.resolveNodeDefaultWorkerPath();
          delete process.env.JEST_WORKER_ID;
          const distWorkerPath =
            INFERENCE_CHANNEL_HOST_INTERNALS.resolveNodeDefaultWorkerPath();

          return {
            browserWorkerScript: (
              browserWorker.worker as unknown as { scriptUrl: string }
            ).scriptUrl,
            browserWorkerType: (
              browserWorker.worker as unknown as {
                options: { type: string };
              }
            ).options.type,
            directMessage:
              INFERENCE_CHANNEL_HOST_INTERNALS.resolveMessageEventData(
                'direct-message',
              ),
            nodeOptionsForJs:
              INFERENCE_CHANNEL_HOST_INTERNALS.resolveNodeInferenceChannelWorkerOptions(
                'worker-entry.js',
              ),
            nodeOptionsForTs:
              INFERENCE_CHANNEL_HOST_INTERNALS.resolveNodeInferenceChannelWorkerOptions(
                'worker-entry.ts',
              ),
            plainObjectError: INFERENCE_CHANNEL_HOST_INTERNALS.asError({})
              .message,
            runtimeWorkerError:
              INFERENCE_CHANNEL_HOST_INTERNALS.resolveWorkerError({}).message,
            distWorkerPath,
            nodeSpecifier,
            nodeWorkerPath: sourceWorkerPath,
            overrideSpecifier:
              INFERENCE_CHANNEL_HOST_INTERNALS.resolveInferenceChannelWorkerSpecifier(
                'custom-worker.js',
              ),
          };
        });

        process.env.JEST_WORKER_ID = originalJestWorkerId;

        // Assert
        expect(result).toEqual({
          browserWorkerScript: 'worker-entry.js',
          browserWorkerType: 'module',
          directMessage: 'direct-message',
          nodeOptionsForJs: { type: 'module' },
          nodeOptionsForTs: {
            execArgv: [
              '--loader',
              'ts-node/esm',
              '--experimental-specifier-resolution=node',
            ],
            type: 'module',
          },
          plainObjectError: '[object Object]',
          runtimeWorkerError: 'InferenceChannel worker failed.',
          distWorkerPath: expect.stringContaining(
            '/dist/architecture/network/worker-payload/network.worker-payload.channel.worker.js',
          ),
          nodeSpecifier: expect.stringContaining(
            '/src/architecture/network/worker-payload/network.worker-payload.channel.worker.ts',
          ),
          nodeWorkerPath: expect.stringContaining(
            '/src/architecture/network/worker-payload/network.worker-payload.channel.worker.ts',
          ),
          overrideSpecifier: 'custom-worker.js',
        });
      });

      it('loads node worker_threads through the builtin resolver or opaque import for the host helper', async () => {
        // Act
        const requireLoaded =
          await INFERENCE_CHANNEL_HOST_INTERNALS.loadNodeWorkerThreadsModule(
            () => ({
              MessageChannel: class MockRequireMessageChannel {
                public readonly port1 = {};
                public readonly port2 = {};
              },
              Worker: class MockRequireWorker {},
            }),
            async () => {
              throw new Error(
                'host helper should prefer the builtin resolver.',
              );
            },
          );
        const importLoaded =
          await INFERENCE_CHANNEL_HOST_INTERNALS.loadNodeWorkerThreadsModule(
            () => undefined,
            async (moduleSpecifier) => ({
              MessageChannel: class MockImportMessageChannel {
                public readonly port1 = {};
                public readonly port2 = {};
              },
              Worker: class MockImportWorker {},
              moduleSpecifier,
            }),
          );

        // Assert
        expect({
          importMessageChannelType: typeof importLoaded.MessageChannel,
          importSpecifier: (
            importLoaded as unknown as { moduleSpecifier: string }
          ).moduleSpecifier,
          importWorkerType: typeof importLoaded.Worker,
          requireMessageChannelType: typeof requireLoaded.MessageChannel,
          requireWorkerType: typeof requireLoaded.Worker,
        }).toEqual({
          importMessageChannelType: 'function',
          importSpecifier: 'worker_threads',
          importWorkerType: 'function',
          requireMessageChannelType: 'function',
          requireWorkerType: 'function',
        });
      });

      it('rejects browser helper bootstrap when no workerUrl is available', () => {
        // Assert
        expect({
          browserCreate: (() => {
            try {
              INFERENCE_CHANNEL_HOST_INTERNALS.createBrowserInferenceChannelWorker(
                'worker-entry.js',
                undefined,
              );
              return 'no-error';
            } catch (error) {
              return (error as Error).message;
            }
          })(),
          noProcessResolve: (() => {
            try {
              return withProcessOverride(undefined, () =>
                INFERENCE_CHANNEL_HOST_INTERNALS.resolveInferenceChannelWorkerSpecifier(
                  undefined,
                ),
              );
            } catch (error) {
              return (error as Error).message;
            }
          })(),
        }).toEqual({
          browserCreate:
            'InferenceChannel browser bootstrap requires workerUrl until inline delivery is configured.',
          noProcessResolve:
            'InferenceChannel browser bootstrap requires workerUrl until inline delivery is configured.',
        });
      });

      it('computes the fatal close transition for open and closed channel states', async () => {
        // Arrange
        const openBootstrapRejections: string[] = [];
        const closedBootstrapRejections: string[] = [];
        const openShutdownErrors: string[] = [];
        const closedShutdownErrors: string[] = [];
        let openClearCallCount = 0;
        let closedClearCallCount = 0;

        // Act
        const openClosePromise =
          INFERENCE_CHANNEL_HOST_INTERNALS.resolveFatalChannelClosePromise(
            {
              bootstrapReject: (reason) => {
                openBootstrapRejections.push((reason as Error).message);
              },
              clearBootstrapState: () => {
                openClearCallCount += 1;
              },
              isOpen: true,
              shutdownChannel: async (shutdownError) => {
                openShutdownErrors.push(shutdownError.message);
              },
            },
            new Error('open failure'),
          );
        const closedClosePromise =
          INFERENCE_CHANNEL_HOST_INTERNALS.resolveFatalChannelClosePromise(
            {
              bootstrapReject: (reason) => {
                closedBootstrapRejections.push((reason as Error).message);
              },
              clearBootstrapState: () => {
                closedClearCallCount += 1;
              },
              isOpen: false,
              shutdownChannel: async (shutdownError) => {
                closedShutdownErrors.push(shutdownError.message);
              },
            },
            new Error('closed failure'),
          );

        await openClosePromise;

        // Assert
        expect({
          closed: {
            bootstrapRejections: closedBootstrapRejections,
            clearCallCount: closedClearCallCount,
            hasClosePromise: Boolean(closedClosePromise),
            shutdownErrors: closedShutdownErrors,
          },
          open: {
            bootstrapRejections: openBootstrapRejections,
            clearCallCount: openClearCallCount,
            hasClosePromise: Boolean(openClosePromise),
            shutdownErrors: openShutdownErrors,
          },
        }).toEqual({
          closed: {
            bootstrapRejections: [],
            clearCallCount: 0,
            hasClosePromise: false,
            shutdownErrors: [],
          },
          open: {
            bootstrapRejections: ['open failure'],
            clearCallCount: 1,
            hasClosePromise: true,
            shutdownErrors: ['open failure'],
          },
        });
      });

      it('handles emitter teardown races and missing worker handles through the host teardown helpers', async () => {
        // Arrange
        const listenerTransitions: string[] = [];
        const emitterPort = {
          close() {
            throw new Error('emitter close failed');
          },
          off(type: string, listener: (...args: unknown[]) => void) {
            listenerTransitions.push(`off:${type}:${typeof listener}`);
          },
          on(type: string, listener: (...args: unknown[]) => void) {
            listenerTransitions.push(`on:${type}:${typeof listener}`);
          },
          postMessage() {
            return undefined;
          },
        };

        // Act
        const result = {
          disposeResult:
            await INFERENCE_CHANNEL_HOST_INTERNALS.disposeWorkerHandle(
              undefined,
            ),
          listenerTransitions: await (async () => {
            await INFERENCE_CHANNEL_HOST_INTERNALS.closeChannelPort(
              emitterPort,
            );
            return [...listenerTransitions];
          })(),
        };

        // Assert
        expect(result).toEqual({
          disposeResult: undefined,
          listenerTransitions: ['on:close:function', 'off:close:function'],
        });
      });

      it('supports browser-backed queued shutdown, idempotent close, and closed reset rejection', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(async () => {
          const channel = openInferenceChannel(payload, {
            workerUrl: 'worker-entry.js',
          });
          const queuedPredictionRejection = channel
            .predict([0.25, 0.75])
            .catch((error) => (error as Error).message);
          const isOpenBeforeClose = channel.isOpen;

          await channel.close();
          await channel.close();

          return {
            isOpenAfterClose: channel.isOpen,
            isOpenBeforeClose,
            queuedRejection: await queuedPredictionRejection,
            resetRejection: await channel
              .reset()
              .catch((error) => (error as Error).message),
          };
        });

        // Assert
        expect(result).toEqual({
          isOpenAfterClose: false,
          isOpenBeforeClose: true,
          queuedRejection: 'InferenceChannel is closed.',
          resetRejection: 'InferenceChannel is closed.',
        });
      });

      it('throws synchronously when browser bootstrap has no workerUrl', () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Assert
        expect(() =>
          withProcessOverride(undefined, () =>
            withBrowserWorkerGlobals(() => {
              openInferenceChannel(payload);
            }),
          ),
        ).toThrow(
          'InferenceChannel browser bootstrap requires workerUrl until inline delivery is configured.',
        );
      });

      it('closes one browser-backed ready channel even when the port close throws', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdChannels }) => {
            const channel = openInferenceChannel(payload, {
              workerUrl: 'worker-entry.js',
            });
            const throwingClosePort = createdChannels[0]?.port1 as unknown as
              | {
                  close: () => void;
                  emit: (type: string, event: unknown) => void;
                  postedMessages: Array<{
                    message: unknown;
                    transferList: Transferable[];
                  }>;
                }
              | undefined;

            throwingClosePort?.emit('message', { type: 'ready' });
            if (throwingClosePort) {
              throwingClosePort.close = () => {
                throw new Error('port close failed');
              };
            }
            await channel.close();

            return {
              isOpen: channel.isOpen,
              postedRequestCount: throwingClosePort?.postedMessages.length ?? 0,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          isOpen: false,
          postedRequestCount: 1,
        });
      });

      it('rejects one browser-backed pending prediction when the worker returns a request error', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdChannels }) => {
            const channel = openInferenceChannel(payload, {
              workerUrl: 'worker-entry.js',
            });
            createdChannels[0]?.port1.emit('message', { type: 'ready' });
            const pendingPrediction = channel.predict([0.25, 0.75]);
            await Promise.resolve();
            createdChannels[0]?.port1.emit('message', {
              id: 1,
              message: 'predict failed',
              type: 'request-error',
            });
            const rejectionMessage = await pendingPrediction.catch(
              (error) => (error as Error).message,
            );
            await channel.close();

            return {
              isOpen: channel.isOpen,
              rejectionMessage,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          isOpen: false,
          rejectionMessage: 'predict failed',
        });
      });

      it('uses the default browser request error message when the worker omits one', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdChannels }) => {
            const channel = openInferenceChannel(payload, {
              workerUrl: 'worker-entry.js',
            });
            createdChannels[0]?.port1.emit('message', { type: 'ready' });
            const pendingPrediction = channel.predict([0.25, 0.75]);
            await Promise.resolve();
            const predictionRequestId =
              (
                createdChannels[0]?.port1.postedMessages[0]?.message as
                  | { id?: number }
                  | undefined
              )?.id ?? -1;
            createdChannels[0]?.port1.emit('message', {
              id: predictionRequestId,
              type: 'request-error',
            });
            const rejectionMessage = await pendingPrediction.catch(
              (error) => (error as Error).message,
            );
            await channel.close();

            return {
              isOpen: channel.isOpen,
              rejectionMessage,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          isOpen: false,
          rejectionMessage: 'InferenceChannel worker request failed.',
        });
      });

      it('returns one empty output vector when the worker omits predict-result output', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdChannels }) => {
            const channel = openInferenceChannel(payload, {
              workerUrl: 'worker-entry.js',
            });
            createdChannels[0]?.port1.emit('message', { type: 'ready' });
            const pendingPrediction = channel.predict([0.25, 0.75]);
            await Promise.resolve();
            const predictionRequestId =
              (
                createdChannels[0]?.port1.postedMessages[0]?.message as
                  | { id?: number }
                  | undefined
              )?.id ?? -1;
            createdChannels[0]?.port1.emit('message', {
              id: predictionRequestId,
              type: 'predict-result',
            });
            const outputValues = Array.from(await pendingPrediction);
            await channel.close();

            return {
              isOpen: channel.isOpen,
              outputValues,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          isOpen: false,
          outputValues: [],
        });
      });

      it('rejects one pending browser prediction when the host closes before the worker responds', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdChannels }) => {
            const channel = openInferenceChannel(payload, {
              workerUrl: 'worker-entry.js',
            });
            createdChannels[0]?.port1.emit('message', { type: 'ready' });
            const pendingPredictionRejection = channel
              .predict([0.25, 0.75])
              .catch((error) => (error as Error).message);
            await Promise.resolve();
            await Promise.resolve();
            await channel.close();

            return {
              pendingRejection: await pendingPredictionRejection,
              postedRequestCount:
                createdChannels[0]?.port1.postedMessages.length ?? 0,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          pendingRejection: 'InferenceChannel is closed.',
          postedRequestCount: 2,
        });
      });

      it('rejects one browser-backed prediction when posting to the port throws', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdChannels }) => {
            const channel = openInferenceChannel(payload, {
              workerUrl: 'worker-entry.js',
            });
            const throwingPort = createdChannels[0]?.port1 as unknown as
              | {
                  emit: (type: string, event: unknown) => void;
                  postMessage: (
                    message: unknown,
                    transferList?: Transferable[],
                  ) => void;
                }
              | undefined;

            throwingPort?.emit('message', { type: 'ready' });
            if (throwingPort) {
              throwingPort.postMessage = () => {
                throw new Error('post failed');
              };
            }

            const rejectionMessage = await channel
              .predict([0.25, 0.75])
              .catch((error) => (error as Error).message);
            await channel.close();

            return {
              isOpen: channel.isOpen,
              rejectionMessage,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          isOpen: false,
          rejectionMessage: 'post failed',
        });
      });

      it('closes one browser-backed ready channel even when worker termination rejects', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdChannels, createdWorkers }) => {
            const channel = openInferenceChannel(payload, {
              workerUrl: 'worker-entry.js',
            });
            createdChannels[0]?.port1.emit('message', { type: 'ready' });
            const rejectingWorker = createdWorkers[0] as unknown as {
              terminate: () => Promise<number>;
            };

            rejectingWorker.terminate = async () => {
              throw new Error('terminate failed');
            };
            await channel.close();

            return {
              isOpen: channel.isOpen,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          isOpen: false,
        });
      });

      it('closes one node-backed channel when worker construction fails before bootstrap assigns the port', async () => {
        // Arrange
        const originalBuiltinModuleResolver = Reflect.get(
          process,
          'getBuiltinModule',
        ) as ((moduleSpecifier: string) => unknown) | undefined;
        const originalWorker = globalThis.Worker;
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );
        let result: { isOpen: boolean } | undefined;

        Reflect.set(globalThis, 'Worker', undefined);
        Reflect.set(process, 'getBuiltinModule', (moduleSpecifier: string) => {
          if (moduleSpecifier !== 'worker_threads') {
            return undefined;
          }

          return {
            MessageChannel: class MockMessageChannel {
              public readonly port1 = {};
              public readonly port2 = {};
            },
            Worker: class MockWorker {
              constructor() {
                throw new Error('node worker construction failed');
              }
            },
          };
        });

        try {
          // Act
          const channel = openInferenceChannel(payload);
          await channel.close();
          result = { isOpen: channel.isOpen };
        } finally {
          Reflect.set(
            process,
            'getBuiltinModule',
            originalBuiltinModuleResolver,
          );
          Reflect.set(globalThis, 'Worker', originalWorker);
        }

        // Assert
        expect(result).toEqual({
          isOpen: false,
        });
      });

      it('ignores unknown browser response ids without closing the channel', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdChannels }) => {
            const channel = openInferenceChannel(payload, {
              workerUrl: 'worker-entry.js',
            });
            createdChannels[0]?.port1.emit('message', { type: 'ready' });
            createdChannels[0]?.port1.emit('message', {
              id: 99,
              message: 'late request error',
              type: 'request-error',
            });
            createdChannels[0]?.port1.emit('message', {
              id: 98,
              output: new Float64Array([1]),
              type: 'predict-result',
            });
            createdChannels[0]?.port1.emit('message', {
              id: 97,
              type: 'reset-result',
            });
            createdChannels[0]?.port1.emit('message', {
              id: 96,
              type: 'unknown-result',
            });
            const isOpen = channel.isOpen;
            await channel.close();

            return {
              isOpen,
              postedRequestCount:
                createdChannels[0]?.port1.postedMessages.length ?? 0,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          isOpen: true,
          postedRequestCount: 1,
        });
      });

      it('closes the browser channel on bootstrap request errors with and without messages plus port message errors', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdChannels }) => {
            const bootstrapErrorChannel = openInferenceChannel(payload, {
              workerUrl: 'worker-entry.js',
            });
            createdChannels[0]?.port1.emit('message', {
              message: 'bootstrap failed',
              type: 'request-error',
            });

            const defaultBootstrapErrorChannel = openInferenceChannel(payload, {
              workerUrl: 'worker-entry.js',
            });
            createdChannels[1]?.port1.emit('message', {
              type: 'request-error',
            });

            const portErrorChannel = openInferenceChannel(payload, {
              workerUrl: 'worker-entry.js',
            });
            createdChannels[2]?.port1.emit('messageerror', undefined);

            const channelStates = {
              bootstrapErrorOpen: bootstrapErrorChannel.isOpen,
              defaultBootstrapErrorOpen: defaultBootstrapErrorChannel.isOpen,
              portErrorOpen: portErrorChannel.isOpen,
            };

            await bootstrapErrorChannel.close();
            await defaultBootstrapErrorChannel.close();
            await portErrorChannel.close();

            return channelStates;
          },
        );

        // Assert
        expect(result).toEqual({
          bootstrapErrorOpen: false,
          defaultBootstrapErrorOpen: false,
          portErrorOpen: false,
        });
      });

      it('keeps browser channels open when worker responses use the wrong result type', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdChannels }) => {
            const predictChannel = openInferenceChannel(payload, {
              workerUrl: 'worker-entry.js',
            });
            createdChannels[0]?.port1.emit('message', { type: 'ready' });
            const pendingPrediction = predictChannel.predict([0.25, 0.75]);
            const resetChannel = openInferenceChannel(payload, {
              workerUrl: 'worker-entry.js',
            });
            createdChannels[1]?.port1.emit('message', { type: 'ready' });
            const pendingReset = resetChannel.reset();
            await Promise.resolve();
            const predictRequestId =
              (
                createdChannels[0]?.port1.postedMessages[0]?.message as
                  | { id?: number }
                  | undefined
              )?.id ?? -1;
            createdChannels[0]?.port1.emit('message', {
              id: predictRequestId,
              type: 'reset-result',
            });
            const predictState = await Promise.race([
              pendingPrediction.then(
                () => 'resolved',
                () => 'rejected',
              ),
              new Promise<'pending'>((resolve) => {
                queueMicrotask(() => {
                  resolve('pending');
                });
              }),
            ]);
            const predictOpenBeforeClose = predictChannel.isOpen;
            const resetRequestId =
              (
                createdChannels[1]?.port1.postedMessages[0]?.message as
                  | { id?: number }
                  | undefined
              )?.id ?? -1;
            createdChannels[1]?.port1.emit('message', {
              id: resetRequestId,
              output: new Float64Array([1]),
              type: 'predict-result',
            });
            const resetState = await Promise.race([
              pendingReset.then(
                () => 'resolved',
                () => 'rejected',
              ),
              new Promise<'pending'>((resolve) => {
                queueMicrotask(() => {
                  resolve('pending');
                });
              }),
            ]);
            const resetOpenBeforeClose = resetChannel.isOpen;
            await predictChannel.close();
            await resetChannel.close();

            return {
              predictOpenBeforeClose,
              predictState,
              resetOpenBeforeClose,
              resetState,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          predictOpenBeforeClose: true,
          predictState: 'pending',
          resetOpenBeforeClose: true,
          resetState: 'pending',
        });
      });

      it('keeps browser channels open when worker responses omit result ids', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdChannels }) => {
            const predictChannel = openInferenceChannel(payload, {
              workerUrl: 'worker-entry.js',
            });
            createdChannels[0]?.port1.emit('message', { type: 'ready' });
            const pendingPrediction = predictChannel.predict([0.25, 0.75]);
            const resetChannel = openInferenceChannel(payload, {
              workerUrl: 'worker-entry.js',
            });
            createdChannels[1]?.port1.emit('message', { type: 'ready' });
            const pendingReset = resetChannel.reset();
            await Promise.resolve();
            createdChannels[0]?.port1.emit('message', {
              output: new Float64Array([1]),
              type: 'predict-result',
            });
            const predictState = await Promise.race([
              pendingPrediction.then(
                () => 'resolved',
                () => 'rejected',
              ),
              new Promise<'pending'>((resolve) => {
                queueMicrotask(() => {
                  resolve('pending');
                });
              }),
            ]);
            const predictOpenBeforeClose = predictChannel.isOpen;
            createdChannels[1]?.port1.emit('message', {
              type: 'reset-result',
            });
            const resetState = await Promise.race([
              pendingReset.then(
                () => 'resolved',
                () => 'rejected',
              ),
              new Promise<'pending'>((resolve) => {
                queueMicrotask(() => {
                  resolve('pending');
                });
              }),
            ]);
            const resetOpenBeforeClose = resetChannel.isOpen;
            await predictChannel.close();
            await resetChannel.close();

            return {
              predictOpenBeforeClose,
              predictState,
              resetOpenBeforeClose,
              resetState,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          predictOpenBeforeClose: true,
          predictState: 'pending',
          resetOpenBeforeClose: true,
          resetState: 'pending',
        });
      });

      it('wires event-target ports and workers for message and failure delivery', () => {
        // Arrange
        const portHarness = createEventTargetPortHarness();
        const workerHarness = createEventTargetWorkerHarness();
        const seenMessages: unknown[] = [];
        const seenErrors: string[] = [];

        // Act
        INFERENCE_CHANNEL_HOST_INTERNALS.attachPortMessageListener(
          portHarness.port,
          (message) => {
            seenMessages.push(message);
          },
          (error) => {
            seenErrors.push(error.message);
          },
        );
        INFERENCE_CHANNEL_HOST_INTERNALS.attachWorkerLifecycleListeners(
          workerHarness.worker,
          (error) => {
            seenErrors.push(error.message);
          },
        );
        portHarness.emit('message', { data: 'from-port' });
        portHarness.emit('messageerror', undefined);
        workerHarness.emit('error', { error: new Error('worker-event-error') });
        workerHarness.emit('messageerror', undefined);

        // Assert
        expect({
          errors: seenErrors,
          messages: seenMessages,
          port: portHarness.getSnapshot(),
        }).toEqual({
          errors: [
            'InferenceChannel message port rejected one message.',
            'worker-event-error',
            'InferenceChannel worker rejected one message.',
          ],
          messages: ['from-port'],
          port: {
            isClosed: false,
            postedMessages: [],
            startCallCount: 1,
          },
        });
      });

      it('wires emitter-style ports and workers for message and failure delivery', () => {
        // Arrange
        const portHarness = createEmitterPortHarness();
        const workerHarness = createEmitterWorkerHarness();
        const seenMessages: unknown[] = [];
        const seenErrors: string[] = [];

        // Act
        INFERENCE_CHANNEL_HOST_INTERNALS.attachPortMessageListener(
          portHarness.port,
          (message) => {
            seenMessages.push(message);
          },
          (error) => {
            seenErrors.push(error.message);
          },
        );
        INFERENCE_CHANNEL_HOST_INTERNALS.attachWorkerLifecycleListeners(
          workerHarness.worker,
          (error) => {
            seenErrors.push(error.message);
          },
        );
        portHarness.emit('message', 'from-emitter-port');
        portHarness.emit('messageerror', undefined);
        workerHarness.emit('error', { message: 'worker-emitter-error' });
        workerHarness.emit('messageerror', undefined);
        workerHarness.emit('exit', 17);

        // Assert
        expect({
          errors: seenErrors,
          messages: seenMessages,
          port: portHarness.getSnapshot(),
        }).toEqual({
          errors: [
            'InferenceChannel message port rejected one message.',
            'worker-emitter-error',
            'InferenceChannel worker rejected one message.',
            'InferenceChannel worker exited with code 17.',
          ],
          messages: ['from-emitter-port'],
          port: {
            isClosed: false,
            postedMessages: [],
          },
        });
      });
    });

    describe('worker internal helpers', () => {
      it('registers one browser message listener when process is unavailable', () => {
        // Arrange
        const originalAddEventListener = globalThis.addEventListener;
        const capturedRegistrations: Array<{
          type: string;
          once: boolean | undefined;
        }> = [];

        Reflect.set(
          globalThis,
          'addEventListener',
          (
            type: string,
            _listener: (event: unknown) => void,
            options?: { once?: boolean },
          ) => {
            capturedRegistrations.push({ once: options?.once, type });
          },
        );

        // Act
        withProcessOverride(undefined, () => {
          registerInferenceChannelWorkerRuntime();
        });

        Reflect.set(globalThis, 'addEventListener', originalAddEventListener);

        // Assert
        expect(capturedRegistrations).toEqual([
          { once: true, type: 'message' },
        ]);
      });

      it('registers one node parent-port listener through the public worker runtime wrapper', async () => {
        // Arrange
        const originalBuiltinModuleResolver = Reflect.get(
          process,
          'getBuiltinModule',
        ) as ((moduleSpecifier: string) => unknown) | undefined;
        const registrationTypes: string[] = [];

        Reflect.set(process, 'getBuiltinModule', (moduleSpecifier: string) => {
          if (moduleSpecifier !== 'worker_threads') {
            return undefined;
          }

          return {
            parentPort: {
              once(type: string) {
                registrationTypes.push(type);
              },
            },
          };
        });

        try {
          // Act
          registerInferenceChannelWorkerRuntime();
          await Promise.resolve();
          await Promise.resolve();
        } finally {
          Reflect.set(
            process,
            'getBuiltinModule',
            originalBuiltinModuleResolver,
          );
        }

        // Assert
        expect(registrationTypes).toEqual(['message']);
      });

      it('auto-registers one browser runtime when the channel worker module sees a worker-like global scope', () => {
        // Arrange
        const originalAddEventListener = globalThis.addEventListener;
        const originalDocument = Reflect.get(globalThis, 'document');
        const registrations: Array<{
          once: boolean | undefined;
          type: string;
        }> = [];
        let result:
          | {
              registrations: Array<{ once: boolean | undefined; type: string }>;
              shouldAutoRegister: boolean;
            }
          | undefined;

        Reflect.set(
          globalThis,
          'addEventListener',
          (
            type: string,
            _listener: (event: unknown) => void,
            options?: { once?: boolean },
          ) => {
            registrations.push({ once: options?.once, type });
          },
        );
        Reflect.set(globalThis, 'document', undefined);

        try {
          // Act
          result = withProcessOverride(undefined, () => {
            INFERENCE_CHANNEL_WORKER_INTERNALS.autoRegisterInferenceChannelWorkerRuntime();

            return {
              registrations: [...registrations],
              shouldAutoRegister:
                INFERENCE_CHANNEL_WORKER_INTERNALS.shouldAutoRegisterInferenceChannelWorkerRuntime(),
            };
          });
        } finally {
          Reflect.set(globalThis, 'addEventListener', originalAddEventListener);
          Reflect.set(globalThis, 'document', originalDocument);
        }

        // Assert
        expect(result).toEqual({
          registrations: [{ once: true, type: 'message' }],
          shouldAutoRegister: true,
        });
      });

      it('registers one node parent-port listener through the worker runtime helper', () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );
        const portHarness = createEmitterPortHarness();
        let capturedBootstrapListener: ((message: unknown) => void) | undefined;
        const parentPort = {
          close() {
            return undefined;
          },
          once(_type: string, listener: (message: unknown) => void) {
            capturedBootstrapListener = listener;
          },
        };
        let exitCallCount = 0;

        // Act
        INFERENCE_CHANNEL_WORKER_INTERNALS.registerNodeInferenceChannelWorkerRuntime(
          parentPort,
          () => {
            exitCallCount += 1;
          },
        );
        capturedBootstrapListener?.({
          payload,
          port: portHarness.port,
          type: 'bootstrap',
        });
        portHarness.emit('message', { type: 'close' });

        // Assert
        expect({
          exitCallCount,
          port: portHarness.getSnapshot(),
        }).toEqual({
          exitCallCount: 1,
          port: {
            isClosed: true,
            postedMessages: [{ message: { type: 'ready' }, transferList: [] }],
          },
        });
      });

      it('loads node worker_threads through the builtin resolver or opaque import for the worker helper', async () => {
        // Act
        const requireLoaded =
          await INFERENCE_CHANNEL_WORKER_INTERNALS.loadNodeWorkerThreadsModule(
            () => ({
              parentPort: {
                close() {
                  return undefined;
                },
              },
            }),
            async () => {
              throw new Error(
                'worker helper should prefer the builtin resolver.',
              );
            },
          );
        const importLoaded =
          await INFERENCE_CHANNEL_WORKER_INTERNALS.loadNodeWorkerThreadsModule(
            () => undefined,
            async (moduleSpecifier) => ({
              moduleSpecifier,
              parentPort: {
                once() {
                  return undefined;
                },
              },
            }),
          );

        // Assert
        expect({
          importHasParentPortListener:
            typeof importLoaded.parentPort?.once === 'function',
          importSpecifier: (
            importLoaded as unknown as { moduleSpecifier: string }
          ).moduleSpecifier,
          requireHasParentPortClose:
            typeof requireLoaded.parentPort?.close === 'function',
        }).toEqual({
          importHasParentPortListener: true,
          importSpecifier: 'worker_threads',
          requireHasParentPortClose: true,
        });
      });

      it('bootstraps one worker port and posts the ready message', () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );
        const portHarness = createEventTargetPortHarness();

        // Act
        INFERENCE_CHANNEL_WORKER_INTERNALS.handleBootstrapMessage(
          {
            payload,
            port: portHarness.port,
            type: 'bootstrap',
          },
          () => undefined,
        );

        // Assert
        expect(portHarness.getSnapshot()).toEqual({
          isClosed: false,
          postedMessages: [{ message: { type: 'ready' }, transferList: [] }],
          startCallCount: 1,
        });
      });

      it('ignores non-bootstrap worker bootstrap messages', () => {
        // Arrange
        const portHarness = createEventTargetPortHarness();

        // Act
        INFERENCE_CHANNEL_WORKER_INTERNALS.handleBootstrapMessage(
          { type: 'not-bootstrap' },
          () => undefined,
        );

        // Assert
        expect(portHarness.getSnapshot()).toEqual({
          isClosed: false,
          postedMessages: [],
          startCallCount: 0,
        });
      });

      it('rejects malformed bootstrap messages before predictor construction', () => {
        // Assert
        expect(() => {
          INFERENCE_CHANNEL_WORKER_INTERNALS.handleBootstrapMessage(
            { type: 'bootstrap' },
            () => undefined,
          );
        }).toThrow(
          'InferenceChannel worker expected bootstrap payload and port.',
        );
      });

      it('handles predict, reset, close, and request-error worker port messages', () => {
        // Arrange
        const portHarness = createEmitterPortHarness();
        const closeEvents: string[] = [];
        const predictor = {
          predict(inputValues: number[]) {
            if (inputValues[0] === 99) {
              throw 'predict failed';
            }

            return [inputValues.reduce((sum, value) => sum + value, 0)];
          },
          reset() {
            closeEvents.push('reset');
          },
        } as ReturnType<typeof createInferencePredictor>;

        // Act
        INFERENCE_CHANNEL_WORKER_INTERNALS.handleChannelPortRequest(
          portHarness.port,
          { id: 1, input: [0.25, 0.75], type: 'predict' },
          predictor,
          () => {
            closeEvents.push('closeWorker');
          },
        );
        INFERENCE_CHANNEL_WORKER_INTERNALS.handleChannelPortRequest(
          portHarness.port,
          { id: 2, type: 'reset' },
          predictor,
          () => {
            closeEvents.push('closeWorker');
          },
        );
        INFERENCE_CHANNEL_WORKER_INTERNALS.handleChannelPortRequest(
          portHarness.port,
          { type: 'close' },
          predictor,
          () => {
            closeEvents.push('closeWorker');
          },
        );
        INFERENCE_CHANNEL_WORKER_INTERNALS.handleChannelPortRequest(
          portHarness.port,
          { id: 3, input: [99], type: 'predict' },
          predictor,
          () => {
            closeEvents.push('closeWorker');
          },
        );

        // Assert
        expect({
          closeEvents,
          port: portHarness.getSnapshot(),
        }).toEqual({
          closeEvents: ['reset', 'closeWorker'],
          port: {
            isClosed: true,
            postedMessages: [
              {
                message: {
                  id: 1,
                  output: new Float64Array([1]),
                  type: 'predict-result',
                },
                transferList: [expect.anything()],
              },
              {
                message: {
                  id: 2,
                  type: 'reset-result',
                },
                transferList: [],
              },
              {
                message: {
                  id: 3,
                  message: 'predict failed',
                  type: 'request-error',
                },
                transferList: [],
              },
            ],
          },
        });
      });

      it('uses worker fallback ids, default predict input, and ignores unknown request types', () => {
        // Arrange
        const portHarness = createEmitterPortHarness();
        const closeEvents: string[] = [];
        const predictor = {
          predict(inputValues: number[]) {
            return [inputValues.length];
          },
          reset() {
            closeEvents.push('reset');
          },
        } as ReturnType<typeof createInferencePredictor>;

        // Act
        INFERENCE_CHANNEL_WORKER_INTERNALS.handleChannelPortRequest(
          portHarness.port,
          { type: 'predict' },
          predictor,
          () => {
            closeEvents.push('closeWorker');
          },
        );
        INFERENCE_CHANNEL_WORKER_INTERNALS.handleChannelPortRequest(
          portHarness.port,
          { type: 'reset' },
          predictor,
          () => {
            closeEvents.push('closeWorker');
          },
        );
        INFERENCE_CHANNEL_WORKER_INTERNALS.handleChannelPortRequest(
          portHarness.port,
          { type: 'unknown' },
          predictor,
          () => {
            closeEvents.push('closeWorker');
          },
        );

        // Assert
        expect({
          closeEvents,
          port: portHarness.getSnapshot(),
        }).toEqual({
          closeEvents: ['reset'],
          port: {
            isClosed: false,
            postedMessages: [
              {
                message: {
                  id: -1,
                  output: new Float64Array([0]),
                  type: 'predict-result',
                },
                transferList: [expect.anything()],
              },
              {
                message: {
                  id: -1,
                  type: 'reset-result',
                },
                transferList: [],
              },
            ],
          },
        });
      });

      it('wires worker-port listener helpers and message normalization helpers', () => {
        // Arrange
        const eventTargetHarness = createEventTargetPortHarness();
        const emitterHarness = createEmitterPortHarness();
        const seenMessages: unknown[] = [];

        // Act
        INFERENCE_CHANNEL_WORKER_INTERNALS.attachPortListener(
          eventTargetHarness.port,
          (message) => {
            seenMessages.push(message);
          },
        );
        INFERENCE_CHANNEL_WORKER_INTERNALS.attachPortListener(
          emitterHarness.port,
          (message) => {
            seenMessages.push(message);
          },
        );
        eventTargetHarness.emit('message', { data: 'event-target-message' });
        emitterHarness.emit('message', 'emitter-message');
        INFERENCE_CHANNEL_WORKER_INTERNALS.postPortMessage(
          emitterHarness.port,
          { id: 5, type: 'reset-result' },
        );

        // Assert
        expect({
          errorMessage: INFERENCE_CHANNEL_WORKER_INTERNALS.asError(
            'worker-string-error',
          ).message,
          eventTargetSnapshot: eventTargetHarness.getSnapshot(),
          normalizedMessages: [
            INFERENCE_CHANNEL_WORKER_INTERNALS.resolveMessageEventData({
              data: 'message-data',
            }),
            INFERENCE_CHANNEL_WORKER_INTERNALS.resolveMessageEventData(
              'raw-message',
            ),
          ],
          seenMessages,
          emitterSnapshot: emitterHarness.getSnapshot(),
        }).toEqual({
          errorMessage: 'worker-string-error',
          emitterSnapshot: {
            isClosed: false,
            postedMessages: [
              {
                message: { id: 5, type: 'reset-result' },
                transferList: [],
              },
            ],
          },
          eventTargetSnapshot: {
            isClosed: false,
            postedMessages: [],
            startCallCount: 1,
          },
          normalizedMessages: ['message-data', 'raw-message'],
          seenMessages: ['event-target-message', 'emitter-message'],
        });
      });

      it('delegates the browser and node worker wrapper exit helpers to the environment', () => {
        // Arrange
        const originalClose = globalThis.close;
        const originalProcessExit = process.exit;
        let browserCloseCallCount = 0;
        let nodeExitCallCount = 0;

        Reflect.set(globalThis, 'close', () => {
          browserCloseCallCount += 1;
        });
        process.exit = (() => {
          nodeExitCallCount += 1;
          return undefined as never;
        }) as typeof process.exit;

        // Act
        INFERENCE_CHANNEL_WORKER_INTERNALS.closeCurrentBrowserInferenceChannelWorkerRuntime();
        INFERENCE_CHANNEL_WORKER_INTERNALS.exitCurrentNodeInferenceChannelWorkerRuntime();

        Reflect.set(globalThis, 'close', originalClose);
        process.exit = originalProcessExit;

        // Assert
        expect({
          browserCloseCallCount,
          nodeExitCallCount,
        }).toEqual({
          browserCloseCallCount: 1,
          nodeExitCallCount: 1,
        });
      });

      it('uses the browser registration wrapper to close after one close message', () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );
        const portHarness = createEventTargetPortHarness();
        const capturedListeners: Array<(event: unknown) => void> = [];
        let closeWorkerCallCount = 0;

        // Act
        INFERENCE_CHANNEL_WORKER_INTERNALS.registerBrowserInferenceChannelWorkerRuntime(
          (listener) => {
            capturedListeners.push(listener);
          },
          () => {
            closeWorkerCallCount += 1;
          },
        );
        capturedListeners[0]?.({
          data: {
            payload,
            port: portHarness.port,
            type: 'bootstrap',
          },
        });
        portHarness.emit('message', { type: 'close' });

        // Assert
        expect({
          closeWorkerCallCount,
          directErrorMessage: INFERENCE_CHANNEL_WORKER_INTERNALS.asError(
            new Error('worker-error'),
          ).message,
          port: portHarness.getSnapshot(),
        }).toEqual({
          closeWorkerCallCount: 1,
          directErrorMessage: 'worker-error',
          port: {
            isClosed: true,
            postedMessages: [{ message: { type: 'ready' }, transferList: [] }],
            startCallCount: 1,
          },
        });
      });
    });
  });

  describe('shared inference workers', () => {
    describe('when one transferable payload bootstraps a persistent shared-memory worker', () => {
      it('matches the runtime network outputs before and after one reset cycle', async () => {
        // Arrange
        const inputValues = [0.5, 0.25];
        const network = createWorkerPayloadNetwork();
        const recurrentOutputNode = network.nodes.find(
          (candidateNode) => candidateNode.type === 'output',
        );

        if (!recurrentOutputNode) {
          throw new Error(
            'Expected an output node for the shared worker reset test.',
          );
        }

        const selfConnection = new Connection(
          recurrentOutputNode,
          recurrentOutputNode,
          0.5,
        );
        recurrentOutputNode.connections.self = [selfConnection];
        network.selfconns = [selfConnection];
        disableFastSlab(network);

        const sharedWorker = openSharedInferenceWorker(
          exportTransferableInferencePayload(network),
        );

        try {
          // Act
          const firstSharedOutput = roundVector(
            Array.from(await sharedWorker.infer(inputValues)),
          );
          const secondSharedOutput = roundVector(
            Array.from(await sharedWorker.infer(inputValues)),
          );
          await sharedWorker.reset();
          const resetSharedOutput = roundVector(
            Array.from(await sharedWorker.infer(inputValues)),
          );

          const firstRuntimeOutput = roundVector(
            network.noTraceActivate([...inputValues]),
          );
          const secondRuntimeOutput = roundVector(
            network.noTraceActivate([...inputValues]),
          );
          network.clear();
          const resetRuntimeOutput = roundVector(
            network.noTraceActivate([...inputValues]),
          );

          // Assert
          expect({
            runtime: [
              firstRuntimeOutput,
              secondRuntimeOutput,
              resetRuntimeOutput,
            ],
            sharedWorker: [
              firstSharedOutput,
              secondSharedOutput,
              resetSharedOutput,
            ],
          }).toEqual({
            runtime: [
              firstRuntimeOutput,
              secondRuntimeOutput,
              resetRuntimeOutput,
            ],
            sharedWorker: [
              firstRuntimeOutput,
              secondRuntimeOutput,
              resetRuntimeOutput,
            ],
          });
        } finally {
          await sharedWorker.release();
        }
      });

      it('matches the runtime baseline across repeated sequential inference calls', async () => {
        // Arrange
        const inputValues = [0.5, 0.25];
        const sequentialInferenceCount = 4;
        const network = createWorkerPayloadNetwork();
        const recurrentOutputNode = network.nodes.find(
          (candidateNode) => candidateNode.type === 'output',
        );

        if (!recurrentOutputNode) {
          throw new Error(
            'Expected an output node for the shared worker stress test.',
          );
        }

        const selfConnection = new Connection(
          recurrentOutputNode,
          recurrentOutputNode,
          0.5,
        );
        recurrentOutputNode.connections.self = [selfConnection];
        network.selfconns = [selfConnection];
        disableFastSlab(network);

        const payload = exportTransferableInferencePayload(network);

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdWorkers }) => {
            const sharedWorker = openSharedInferenceWorker(payload, {
              workerUrl: 'browser-shared-worker.js',
            });
            const sharedPredictor = createInferencePredictor(payload);
            const browserWorker = createdWorkers[0] as unknown as {
              emit: (type: string, event: unknown) => void;
              postedMessages: Array<{
                message: {
                  controlBuffer: SharedArrayBuffer;
                  dataBuffer: SharedArrayBuffer;
                  payload: { outputCount: number };
                  type: 'bootstrap';
                };
                transferList: Transferable[];
              }>;
            };

            browserWorker.emit('message', { data: { type: 'ready' } });

            const bootstrapMessage = browserWorker.postedMessages[0]?.message;

            if (!bootstrapMessage) {
              throw new Error(
                'Expected a shared worker bootstrap message for the parity test.',
              );
            }

            const controlView = new Int32Array(bootstrapMessage.controlBuffer);
            const dataView = new Float64Array(bootstrapMessage.dataBuffer);
            const sharedLayout =
              SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout(
                payload.inputCount,
                bootstrapMessage.payload.outputCount,
              );
            const runtimeOutputs: number[][] = [];
            const sharedWorkerOutputs: number[][] = [];

            try {
              for (
                let iterationIndex = 0;
                iterationIndex < sequentialInferenceCount;
                iterationIndex += 1
              ) {
                const sharedOutputPromise = sharedWorker.infer(inputValues);

                await SHARED_INFERENCE_HOST_INTERNALS.waitForSharedStatus(
                  controlView,
                  sharedLayout.statusIndexes.status,
                  sharedLayout.statusValues.inputReady,
                  () => undefined,
                );

                const predictorOutput = sharedPredictor.predict(inputValues);

                for (
                  let outputIndex = 0;
                  outputIndex < predictorOutput.length;
                  outputIndex += 1
                ) {
                  dataView[sharedLayout.outputOffset + outputIndex] =
                    predictorOutput[outputIndex] ?? 0;
                }

                Atomics.store(
                  controlView,
                  sharedLayout.statusIndexes.status,
                  sharedLayout.statusValues.outputReady,
                );
                Atomics.notify(controlView, sharedLayout.statusIndexes.status);

                sharedWorkerOutputs.push(
                  roundVector(Array.from(await sharedOutputPromise)),
                );
                runtimeOutputs.push(
                  roundVector(network.noTraceActivate([...inputValues])),
                );
              }

              return {
                runtimeOutputs,
                sharedWorkerOutputs,
              };
            } finally {
              await sharedWorker.release();
            }
          },
        );

        // Assert
        expect(result).toEqual({
          runtimeOutputs: result.runtimeOutputs,
          sharedWorkerOutputs: result.runtimeOutputs,
        });
      });
    });

    describe('when the public shared worker API hits lifecycle edge cases', () => {
      it('guards browser-backed shared worker calls before readiness and without one in-flight request', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdWorkers }) => {
            const sharedWorker = openSharedInferenceWorker(payload, {
              workerUrl: 'browser-shared-worker.js',
            });
            const browserWorker = createdWorkers[0] as unknown as {
              emit: (type: string, event: unknown) => void;
            };
            let submitBeforeReadyError = '';
            let awaitWithoutInputError = '';
            let wrongLengthError = '';

            try {
              sharedWorker.submitInput([0.25, 0.75]);
            } catch (error) {
              submitBeforeReadyError = (error as Error).message;
            }

            browserWorker.emit('message', { data: { type: 'ready' } });

            try {
              await sharedWorker.awaitOutput();
            } catch (error) {
              awaitWithoutInputError = (error as Error).message;
            }

            try {
              sharedWorker.submitInput([0.25]);
            } catch (error) {
              wrongLengthError = (error as Error).message;
            }

            await sharedWorker.release();

            return {
              awaitWithoutInputError,
              submitBeforeReadyError,
              wrongLengthError,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          awaitWithoutInputError:
            'SharedInferenceWorker has no in-flight request.',
          submitBeforeReadyError: 'SharedInferenceWorker is not ready.',
          wrongLengthError:
            'SharedInferenceWorker expected 2 inputs but received 1.',
        });
      });

      it('rejects busy submit, reset, and release calls while one shared inference request is in flight', async () => {
        // Arrange
        const inputValues = [0.5, 0.25];

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdWorkers }) => {
            const sharedWorker = openSharedInferenceWorker(
              exportTransferableInferencePayload(createWorkerPayloadNetwork()),
              {
                workerUrl: 'browser-shared-worker.js',
              },
            );
            const browserWorker = createdWorkers[0] as unknown as {
              emit: (type: string, event: unknown) => void;
              postedMessages: Array<{
                message: {
                  controlBuffer: SharedArrayBuffer;
                  dataBuffer: SharedArrayBuffer;
                  payload: { outputCount: number };
                  type: 'bootstrap';
                };
                transferList: Transferable[];
              }>;
            };

            browserWorker.emit('message', { data: { type: 'ready' } });
            sharedWorker.submitInput(inputValues);

            const busySubmitError = (() => {
              try {
                sharedWorker.submitInput(inputValues);
                return 'no-error';
              } catch (error) {
                return (error as Error).message;
              }
            })();
            const busyResetError = await sharedWorker
              .reset()
              .catch((error) => (error as Error).message);
            const busyReleaseError = await sharedWorker
              .release()
              .catch((error) => (error as Error).message);

            const bootstrapMessage = browserWorker.postedMessages[0]?.message;
            const controlView = new Int32Array(bootstrapMessage.controlBuffer);
            const dataView = new Float64Array(bootstrapMessage.dataBuffer);
            const sharedLayout =
              SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout(
                inputValues.length,
                bootstrapMessage.payload.outputCount,
              );
            dataView[sharedLayout.outputOffset] = 0.75;
            Atomics.store(
              controlView,
              sharedLayout.statusIndexes.status,
              sharedLayout.statusValues.outputReady,
            );
            Atomics.notify(controlView, sharedLayout.statusIndexes.status);

            const completedOutput = roundVector(
              Array.from(await sharedWorker.awaitOutput()),
            );
            await sharedWorker.release();

            return {
              busyReleaseError,
              busyResetError,
              busySubmitError,
              completedOutputLength: completedOutput.length,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          busyReleaseError:
            'SharedInferenceWorker cannot release while one request is in flight.',
          busyResetError:
            'SharedInferenceWorker cannot reset while one request is in flight.',
          busySubmitError: 'SharedInferenceWorker is busy.',
          completedOutputLength: 1,
        });
      });

      it('detaches large shared output shelves across timer turns without changing the returned values', async () => {
        // Arrange
        const payload = createSyntheticTransferablePayload(1, 65_537);
        let timerTurns = 0;

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdWorkers }) => {
            const sharedWorker = openSharedInferenceWorker(payload, {
              workerUrl: 'browser-shared-worker.js',
            });
            const browserWorker = createdWorkers[0] as unknown as {
              emit: (type: string, event: unknown) => void;
              postedMessages: Array<{
                message: {
                  controlBuffer: SharedArrayBuffer;
                  dataBuffer: SharedArrayBuffer;
                  payload: { outputCount: number };
                  type: 'bootstrap';
                };
                transferList: Transferable[];
              }>;
            };
            const originalSetTimeout = globalThis.setTimeout;

            globalThis.setTimeout = ((
              handler: TimerHandler,
              _timeout?: number,
              ...arguments_: unknown[]
            ) => {
              timerTurns += 1;

              if (typeof handler === 'function') {
                handler(...arguments_);
              }

              return 0 as unknown as ReturnType<typeof setTimeout>;
            }) as unknown as typeof setTimeout;

            try {
              browserWorker.emit('message', { data: { type: 'ready' } });
              sharedWorker.submitInput([0.25]);

              const bootstrapMessage = browserWorker.postedMessages[0]?.message;
              const controlView = new Int32Array(
                bootstrapMessage.controlBuffer,
              );
              const dataView = new Float64Array(bootstrapMessage.dataBuffer);
              const sharedLayout =
                SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout(
                  payload.inputCount,
                  payload.outputCount,
                );

              await SHARED_INFERENCE_HOST_INTERNALS.waitForSharedStatus(
                controlView,
                sharedLayout.statusIndexes.status,
                sharedLayout.statusValues.inputReady,
                () => undefined,
              );

              for (
                let outputIndex = 0;
                outputIndex < payload.outputCount;
                outputIndex += 1
              ) {
                dataView[sharedLayout.outputOffset + outputIndex] =
                  outputIndex + 0.5;
              }

              Atomics.store(
                controlView,
                sharedLayout.statusIndexes.status,
                sharedLayout.statusValues.outputReady,
              );
              Atomics.notify(controlView, sharedLayout.statusIndexes.status);

              const detachedOutput = await sharedWorker.awaitOutput();
              await sharedWorker.release();

              return {
                firstOutputValue: detachedOutput[0],
                lastOutputValue: detachedOutput.at(-1),
                timerTurns,
              };
            } finally {
              globalThis.setTimeout = originalSetTimeout;
            }
          },
        );

        // Assert
        expect(result).toEqual({
          firstOutputValue: 0.5,
          lastOutputValue: payload.outputCount - 1 + 0.5,
          timerTurns: expect.any(Number),
        });
      });

      it('rejects inference after release and reports the closed-ready state', async () => {
        // Arrange
        const inputValues = [0.5, 0.25];
        const sharedWorker = openSharedInferenceWorker(
          exportTransferableInferencePayload(createWorkerPayloadNetwork()),
        );

        // Act
        await sharedWorker.infer(inputValues);
        await sharedWorker.release();
        const inferError = await sharedWorker
          .infer(inputValues)
          .catch((error) => (error as Error).message);

        // Assert
        expect({
          inferError,
          isReady: sharedWorker.isReady,
        }).toEqual({
          inferError: 'SharedInferenceWorker is closed.',
          isReady: false,
        });
      });

      it('propagates browser-backed shared worker request-error bootstrap failures and reuses the close promise on release', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdWorkers }) => {
            const sharedWorker = openSharedInferenceWorker(payload, {
              workerUrl: 'browser-shared-worker.js',
            });
            const browserWorker = createdWorkers[0] as unknown as {
              emit: (type: string, event: unknown) => void;
            };

            browserWorker.emit('message', {
              data: {
                message: 'shared bootstrap failed',
                type: 'request-error',
              },
            });
            const inferError = await sharedWorker
              .infer([0.25, 0.75])
              .catch((error) => (error as Error).message);
            await sharedWorker.release();

            return {
              inferError,
              isReady: sharedWorker.isReady,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          inferError: 'shared bootstrap failed',
          isReady: false,
        });
      });

      it('uses the default shared bootstrap failure message when the worker omits one', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdWorkers }) => {
            const sharedWorker = openSharedInferenceWorker(payload, {
              workerUrl: 'browser-shared-worker.js',
            });
            const browserWorker = createdWorkers[0] as unknown as {
              emit: (type: string, event: unknown) => void;
            };

            browserWorker.emit('message', {
              data: {
                type: 'request-error',
              },
            });
            const inferError = await sharedWorker
              .infer([0.25, 0.75])
              .catch((error) => (error as Error).message);
            await sharedWorker.release();

            return {
              inferError,
              isReady: sharedWorker.isReady,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          inferError: 'SharedInferenceWorker bootstrap failed.',
          isReady: false,
        });
      });

      it('ignores unknown shared bootstrap messages until one ready response arrives', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdWorkers }) => {
            const sharedWorker = openSharedInferenceWorker(payload, {
              workerUrl: 'browser-shared-worker.js',
            });
            const browserWorker = createdWorkers[0] as unknown as {
              emit: (type: string, event: unknown) => void;
            };

            browserWorker.emit('message', {
              data: {
                type: 'unknown-bootstrap-message',
              },
            });
            const readyBeforeReadyMessage = sharedWorker.isReady;
            browserWorker.emit('message', { data: { type: 'ready' } });
            const readyAfterReadyMessage = sharedWorker.isReady;
            await sharedWorker.release();

            return {
              readyAfterReadyMessage,
              readyBeforeReadyMessage,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          readyAfterReadyMessage: true,
          readyBeforeReadyMessage: false,
        });
      });

      it('propagates browser-backed shared worker messageerror bootstrap failures', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdWorkers }) => {
            const sharedWorker = openSharedInferenceWorker(payload, {
              workerUrl: 'browser-shared-worker.js',
            });
            const browserWorker = createdWorkers[0] as unknown as {
              emit: (type: string, event: unknown) => void;
            };

            browserWorker.emit('messageerror', undefined);
            const inferError = await sharedWorker
              .infer([0.25, 0.75])
              .catch((error) => (error as Error).message);
            await sharedWorker.release();

            return {
              inferError,
              isReady: sharedWorker.isReady,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          inferError: 'SharedInferenceWorker rejected one message.',
          isReady: false,
        });
      });
    });

    describe('when the host sizes one shared-memory inference slot', () => {
      it('describes stable control indexes, status flags, and data offsets for the slot layout', () => {
        // Act
        const result = {
          ...SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout(
            3,
            2,
          ),
          requiresCrossOriginIsolation:
            SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION,
        };

        // Assert
        expect(result).toEqual({
          controlElementCount: 2,
          dataElementCount: 5,
          inputCount: 3,
          inputOffset: 0,
          outputCount: 2,
          outputOffset: 3,
          requiresCrossOriginIsolation: true,
          statusIndexes: {
            inputCount: 1,
            status: 0,
          },
          statusValues: {
            closed: 4,
            idle: 0,
            inputReady: 1,
            outputReady: 2,
            resetRequested: 3,
          },
        });
      });
    });

    describe('host internal helpers', () => {
      it('resolves shared host-side error helpers and browser-default bootstrap contracts', () => {
        // Arrange
        const originalDocument = globalThis.document;
        const browserDocument = (originalDocument ?? {}) as Document & {
          currentScript?: { src?: string } | null;
        };
        const shouldRestoreDocument = typeof originalDocument === 'undefined';
        if (shouldRestoreDocument) {
          Reflect.set(globalThis, 'document', browserDocument);
        }
        const currentScriptDescriptor = Object.getOwnPropertyDescriptor(
          browserDocument,
          'currentScript',
        );
        const originalJestWorkerId = process.env.JEST_WORKER_ID;
        const originalSharedArrayBuffer = globalThis.SharedArrayBuffer;
        process.env.JEST_WORKER_ID = '1';
        Object.defineProperty(browserDocument, 'currentScript', {
          configurable: true,
          value: {
            src: 'https://example.test/dist/neataptic.js',
          },
        });

        // Act
        const result = withBrowserWorkerGlobals(() => {
          const browserWorker =
            SHARED_INFERENCE_HOST_INTERNALS.createBrowserSharedInferenceWorker(
              'worker-entry.js',
            );
          const browserDefaultSpecifier = withProcessOverride(undefined, () =>
            SHARED_INFERENCE_HOST_INTERNALS.resolveInferenceSharedWorkerSpecifier(
              undefined,
            ),
          );
          const browserDefaultWorker =
            SHARED_INFERENCE_HOST_INTERNALS.createBrowserSharedInferenceWorker(
              browserDefaultSpecifier,
            );
          const nodeSpecifier =
            SHARED_INFERENCE_HOST_INTERNALS.resolveInferenceSharedWorkerSpecifier(
              undefined,
            );
          const sourceWorkerPath =
            SHARED_INFERENCE_HOST_INTERNALS.resolveNodeDefaultWorkerPath();
          delete process.env.JEST_WORKER_ID;
          const distWorkerPath =
            SHARED_INFERENCE_HOST_INTERNALS.resolveNodeDefaultWorkerPath();

          let closedError = '';
          let plainClosedError = '';
          let missingSharedArrayBufferError = '';

          try {
            SHARED_INFERENCE_HOST_INTERNALS.ensureSharedWorkerOpen(
              new Error('closed-shared-worker'),
              false,
            );
          } catch (error) {
            closedError = (error as Error).message;
          }

          try {
            SHARED_INFERENCE_HOST_INTERNALS.ensureSharedWorkerOpen(
              undefined,
              false,
            );
          } catch (error) {
            plainClosedError = (error as Error).message;
          }

          Reflect.set(globalThis, 'SharedArrayBuffer', undefined);

          try {
            SHARED_INFERENCE_HOST_INTERNALS.ensureSharedArrayBufferSupport();
          } catch (error) {
            missingSharedArrayBufferError = (error as Error).message;
          }

          Reflect.set(
            globalThis,
            'SharedArrayBuffer',
            originalSharedArrayBuffer,
          );
          SHARED_INFERENCE_HOST_INTERNALS.ensureSharedWorkerOpen(
            undefined,
            true,
          );
          SHARED_INFERENCE_HOST_INTERNALS.ensureSharedArrayBufferSupport();

          return {
            browserDefaultSpecifier,
            browserDefaultWorkerScript: (
              browserDefaultWorker.worker as unknown as { scriptUrl: string }
            ).scriptUrl,
            browserWorkerAssetUrl: resolveBrowserWorkerAssetUrl(
              'worker-entry.js',
              {
                baseUrl: 'https://example.test/runtime.bundle.js',
              },
            ),
            browserWorkerScript: (
              browserWorker.worker as unknown as { scriptUrl: string }
            ).scriptUrl,
            browserWorkerType: (
              browserWorker.worker as unknown as {
                options: { type: string };
              }
            ).options.type,
            closedError,
            customError: (() => {
              const normalizedError = SHARED_INFERENCE_HOST_INTERNALS.asError({
                message: 'custom failure',
                name: 'CustomError',
                stack: 'custom-stack',
              });

              return {
                message: normalizedError.message,
                name: normalizedError.name,
                stack: normalizedError.stack,
              };
            })(),
            dataMessage:
              SHARED_INFERENCE_HOST_INTERNALS.resolveMessageEventData({
                data: 'payload',
              }),
            directMessage:
              SHARED_INFERENCE_HOST_INTERNALS.resolveMessageEventData(
                'raw-message',
              ),
            directError: SHARED_INFERENCE_HOST_INTERNALS.asError(
              new Error('boom'),
            ).message,
            distWorkerPath,
            missingSharedArrayBufferError,
            nodeOptionsForJs:
              SHARED_INFERENCE_HOST_INTERNALS.resolveNodeSharedWorkerOptions(
                'worker-entry.js',
              ),
            nodeOptionsForTs:
              SHARED_INFERENCE_HOST_INTERNALS.resolveNodeSharedWorkerOptions(
                'worker-entry.ts',
              ),
            nodeSpecifier,
            nodeWorkerPath: sourceWorkerPath,
            overrideSpecifier:
              SHARED_INFERENCE_HOST_INTERNALS.resolveInferenceSharedWorkerSpecifier(
                'custom-worker.js',
              ),
            plainObjectError: SHARED_INFERENCE_HOST_INTERNALS.asError({})
              .message,
            plainClosedError,
            runtimeWorkerError:
              SHARED_INFERENCE_HOST_INTERNALS.resolveWorkerError({}).message,
            stringError: SHARED_INFERENCE_HOST_INTERNALS.asError(
              'shared-host-string-error',
            ).message,
            workerError: SHARED_INFERENCE_HOST_INTERNALS.resolveWorkerError({
              error: new Error('worker exploded'),
            }).message,
          };
        });

        process.env.JEST_WORKER_ID = originalJestWorkerId;
        Reflect.set(globalThis, 'SharedArrayBuffer', originalSharedArrayBuffer);
        if (currentScriptDescriptor) {
          Object.defineProperty(
            browserDocument,
            'currentScript',
            currentScriptDescriptor,
          );
        } else {
          Reflect.deleteProperty(browserDocument, 'currentScript');
        }
        if (shouldRestoreDocument) {
          Reflect.deleteProperty(globalThis, 'document');
        }

        // Assert
        expect(result).toEqual({
          browserDefaultSpecifier: expect.stringContaining(
            '/dist/architecture/network/worker-payload/network.worker-payload.shared.worker.js',
          ),
          browserDefaultWorkerScript: expect.stringContaining(
            '/dist/architecture/network/worker-payload/network.worker-payload.shared.worker.js',
          ),
          browserWorkerAssetUrl: 'https://example.test/worker-entry.js',
          browserWorkerScript: 'worker-entry.js',
          browserWorkerType: 'module',
          closedError: 'closed-shared-worker',
          customError: {
            message: 'custom failure',
            name: 'CustomError',
            stack: 'custom-stack',
          },
          dataMessage: 'payload',
          directMessage: 'raw-message',
          directError: 'boom',
          distWorkerPath: expect.stringContaining(
            '/dist/architecture/network/worker-payload/network.worker-payload.shared.worker.js',
          ),
          missingSharedArrayBufferError:
            'SharedInferenceWorker requires SharedArrayBuffer support.',
          nodeOptionsForJs: { type: 'module' },
          nodeOptionsForTs: {
            execArgv: [
              '--loader',
              'ts-node/esm',
              '--experimental-specifier-resolution=node',
            ],
            type: 'module',
          },
          nodeSpecifier: expect.stringContaining(
            '/src/architecture/network/worker-payload/network.worker-payload.shared.worker.ts',
          ),
          nodeWorkerPath: expect.stringContaining(
            '/src/architecture/network/worker-payload/network.worker-payload.shared.worker.ts',
          ),
          overrideSpecifier: 'custom-worker.js',
          plainClosedError: 'SharedInferenceWorker is closed.',
          plainObjectError: '[object Object]',
          runtimeWorkerError: 'SharedInferenceWorker failed.',
          stringError: 'shared-host-string-error',
          workerError: 'worker exploded',
        });
      });

      it('falls back to one relative browser worker path when no script or location exists', () => {
        // Arrange
        const originalDocument = globalThis.document;
        const originalLocation = globalThis.location;
        const shouldRestoreDocument = typeof originalDocument === 'undefined';

        Reflect.set(globalThis, 'document', undefined);
        Reflect.set(globalThis, 'location', undefined);

        // Act
        const result = withProcessOverride(undefined, () =>
          SHARED_INFERENCE_HOST_INTERNALS.resolveInferenceSharedWorkerSpecifier(
            undefined,
          ),
        );

        if (shouldRestoreDocument) {
          Reflect.deleteProperty(globalThis, 'document');
        } else {
          Reflect.set(globalThis, 'document', originalDocument);
        }
        Reflect.set(globalThis, 'location', originalLocation);

        // Assert
        expect(result).toBe(
          'architecture/network/worker-payload/network.worker-payload.shared.worker.js',
        );
      });

      it('loads node worker_threads through the builtin resolver or opaque import for the shared host helper', async () => {
        // Act
        const requireLoaded =
          await SHARED_INFERENCE_HOST_INTERNALS.loadNodeWorkerThreadsModule(
            () => ({
              Worker: class MockRequireWorker {},
            }),
            async () => {
              throw new Error(
                'shared host helper should prefer the builtin resolver.',
              );
            },
          );
        const importLoaded =
          await SHARED_INFERENCE_HOST_INTERNALS.loadNodeWorkerThreadsModule(
            () => undefined,
            async (moduleSpecifier) => ({
              Worker: class MockImportWorker {},
              moduleSpecifier,
            }),
          );

        // Assert
        expect({
          importHasWorkerConstructor: typeof importLoaded.Worker === 'function',
          importSpecifier: (
            importLoaded as unknown as { moduleSpecifier: string }
          ).moduleSpecifier,
          requireHasWorkerConstructor:
            typeof requireLoaded.Worker === 'function',
        }).toEqual({
          importHasWorkerConstructor: true,
          importSpecifier: 'worker_threads',
          requireHasWorkerConstructor: true,
        });
      });

      it('wires shared worker event-target and emitter listeners and waits for one status change', async () => {
        // Arrange
        const eventTargetHarness = createEventTargetWorkerHarness();
        const emitterHarness = createEmitterWorkerHarness();
        const seenErrors: string[] = [];
        const seenMessages: unknown[] = [];
        const controlView = new Int32Array(
          new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT),
        );

        Atomics.store(controlView, 0, 0);

        // Act
        SHARED_INFERENCE_HOST_INTERNALS.attachSharedWorkerMessageListener(
          eventTargetHarness.worker,
          (message) => {
            seenMessages.push(message);
          },
          (error) => {
            seenErrors.push(error.message);
          },
        );
        SHARED_INFERENCE_HOST_INTERNALS.attachSharedWorkerLifecycleListeners(
          eventTargetHarness.worker,
          (error) => {
            seenErrors.push(error.message);
          },
        );
        SHARED_INFERENCE_HOST_INTERNALS.attachSharedWorkerMessageListener(
          emitterHarness.worker,
          (message) => {
            seenMessages.push(message);
          },
          (error) => {
            seenErrors.push(error.message);
          },
        );
        SHARED_INFERENCE_HOST_INTERNALS.attachSharedWorkerLifecycleListeners(
          emitterHarness.worker,
          (error) => {
            seenErrors.push(error.message);
          },
        );
        eventTargetHarness.emit('message', { data: 'event-target-message' });
        eventTargetHarness.emit('messageerror', undefined);
        eventTargetHarness.emit('error', {
          error: new Error('event-target-worker-error'),
        });
        emitterHarness.emit('message', 'emitter-message');
        emitterHarness.emit('messageerror', undefined);
        emitterHarness.emit('error', { message: 'emitter-worker-error' });
        emitterHarness.emit('exit', 17);
        setTimeout(() => {
          Atomics.store(controlView, 0, 2);
          Atomics.notify(controlView, 0);
        }, 0);
        await SHARED_INFERENCE_HOST_INTERNALS.waitForSharedStatus(
          controlView,
          0,
          2,
          () => undefined,
        );

        // Assert
        expect({
          errors: seenErrors,
          messages: seenMessages,
          status: Atomics.load(controlView, 0),
        }).toEqual({
          errors: [
            'SharedInferenceWorker rejected one message.',
            'event-target-worker-error',
            'SharedInferenceWorker rejected one message.',
            'emitter-worker-error',
            'SharedInferenceWorker exited with code 17.',
          ],
          messages: ['event-target-message', 'emitter-message'],
          status: 2,
        });
      });

      it('uses shared host wait helpers for the async, not-equal, ok, and timeout fallback branches', async () => {
        // Arrange
        const firstControlView = new Int32Array(
          new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT),
        );
        const secondControlView = new Int32Array(
          new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT),
        );
        const thirdControlView = new Int32Array(
          new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT),
        );
        const fourthControlView = new Int32Array(
          new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT),
        );
        const waitAsyncDescriptor = Object.getOwnPropertyDescriptor(
          Atomics,
          'waitAsync',
        );

        Object.defineProperty(Atomics, 'waitAsync', {
          configurable: true,
          value: (typedArray: Int32Array, index: number) => {
            if (typedArray === thirdControlView) {
              return {
                async: true,
                value: Promise.resolve('ok').then((waitState) => {
                  Atomics.store(typedArray, index, 1);
                  return waitState;
                }),
              };
            }

            if (typedArray === fourthControlView) {
              setTimeout(() => {
                Atomics.store(typedArray, index, 1);
              }, 0);

              return {
                async: false,
                value: 'ok',
              };
            }

            Atomics.store(typedArray, index, 1);

            return {
              async: false,
              value: 'not-equal',
            };
          },
        });

        // Act
        await SHARED_INFERENCE_HOST_INTERNALS.waitForSharedStatus(
          firstControlView,
          0,
          1,
          () => undefined,
        );
        await SHARED_INFERENCE_HOST_INTERNALS.waitForSharedStatus(
          thirdControlView,
          0,
          1,
          () => undefined,
        );
        await SHARED_INFERENCE_HOST_INTERNALS.waitForSharedStatus(
          fourthControlView,
          0,
          1,
          () => undefined,
        );
        Object.defineProperty(Atomics, 'waitAsync', {
          configurable: true,
          value: undefined,
        });
        const timeoutWaitPromise =
          SHARED_INFERENCE_HOST_INTERNALS.waitForSharedStatus(
            secondControlView,
            0,
            1,
            () => undefined,
          ).then(() => 'resolved');
        setTimeout(() => {
          Atomics.store(secondControlView, 0, 1);
        }, 0);
        const timeoutWaitState = await timeoutWaitPromise;

        if (waitAsyncDescriptor) {
          Object.defineProperty(Atomics, 'waitAsync', waitAsyncDescriptor);
        }

        // Assert
        expect({
          asyncStatus: Atomics.load(thirdControlView, 0),
          firstStatus: Atomics.load(firstControlView, 0),
          okStatus: Atomics.load(fourthControlView, 0),
          timeoutWaitState,
          secondStatus: Atomics.load(secondControlView, 0),
        }).toEqual({
          asyncStatus: 1,
          firstStatus: 1,
          okStatus: 1,
          timeoutWaitState: 'resolved',
          secondStatus: 1,
        });
      });

      it('wakes blocked shared host waits when shutdown posts the closed status', async () => {
        // Arrange
        const layout =
          SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout(
            1,
            1,
          );
        const controlView = new Int32Array(
          new SharedArrayBuffer(
            Int32Array.BYTES_PER_ELEMENT * layout.controlElementCount,
          ),
        );
        let isOpen = true;

        Atomics.store(
          controlView,
          layout.statusIndexes.status,
          layout.statusValues.idle,
        );

        // Act
        const closedWaitPromise =
          SHARED_INFERENCE_HOST_INTERNALS.waitForSharedStatus(
            controlView,
            layout.statusIndexes.status,
            layout.statusValues.outputReady,
            () => {
              if (!isOpen) {
                throw new Error('shared host closed');
              }
            },
          ).catch((error) => (error as Error).message);
        await Promise.resolve();
        isOpen = false;
        SHARED_INFERENCE_HOST_INTERNALS.signalSharedInferenceWorkerClosed(
          controlView,
          layout,
        );
        const closedError = await closedWaitPromise;

        // Assert
        expect({
          closedError,
          status: Atomics.load(controlView, layout.statusIndexes.status),
        }).toEqual({
          closedError: 'shared host closed',
          status: layout.statusValues.closed,
        });
      });

      it('chunks large shared numeric conversions across timer turns without changing copied values', async () => {
        // Arrange
        const inputCount = 65_537;
        const outputCount = 65_537;
        const sharedLayout =
          SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout(
            inputCount,
            outputCount,
          );
        const dataView = new Float64Array(
          new SharedArrayBuffer(
            Float64Array.BYTES_PER_ELEMENT * sharedLayout.dataElementCount,
          ),
        );
        const inputValues = Array.from(
          { length: inputCount },
          (_, inputIndex) => inputIndex + 0.25,
        );
        let timerTurns = 0;

        for (let outputIndex = 0; outputIndex < outputCount; outputIndex += 1) {
          dataView[sharedLayout.outputOffset + outputIndex] = outputIndex + 0.5;
        }

        // Act
        const result = await withBrowserWorkerGlobals(async () => {
          const originalSetTimeout = globalThis.setTimeout;

          globalThis.setTimeout = ((
            handler: TimerHandler,
            _timeout?: number,
            ...arguments_: unknown[]
          ) => {
            timerTurns += 1;

            if (typeof handler === 'function') {
              handler(...arguments_);
            }

            return 0 as unknown as ReturnType<typeof setTimeout>;
          }) as unknown as typeof setTimeout;

          try {
            await SHARED_INFERENCE_HOST_INTERNALS.copyInputValuesIntoSharedBuffer(
              dataView,
              inputValues,
              sharedLayout.inputOffset,
              true,
            );
            timerTurns = 0;
            const detachedOutput =
              await SHARED_INFERENCE_HOST_INTERNALS.copySharedOutputValues(
                dataView,
                sharedLayout.outputOffset,
                outputCount,
                true,
              );

            return {
              firstInputValue: dataView[sharedLayout.inputOffset],
              firstOutputValue: detachedOutput[0],
              lastInputValue:
                dataView[sharedLayout.inputOffset + inputCount - 1],
              lastOutputValue: detachedOutput.at(-1),
              outputTimerTurns: timerTurns,
            };
          } finally {
            globalThis.setTimeout = originalSetTimeout;
          }
        });

        // Assert
        expect(
          result.outputTimerTurns > 1 &&
            result.firstInputValue === 0.25 &&
            result.lastInputValue === inputCount - 1 + 0.25 &&
            result.firstOutputValue === 0.5 &&
            result.lastOutputValue === outputCount - 1 + 0.5,
        ).toBe(true);
      });

      it('copies small shared numeric conversions without yielding timer turns', async () => {
        // Arrange
        const inputCount = 4;
        const outputCount = 3;
        const sharedLayout =
          SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout(
            inputCount,
            outputCount,
          );
        const dataView = new Float64Array(
          new SharedArrayBuffer(
            Float64Array.BYTES_PER_ELEMENT * sharedLayout.dataElementCount,
          ),
        );
        const inputValues = [1.25, 2.25, 3.25, 4.25];
        let timerTurns = 0;

        for (let outputIndex = 0; outputIndex < outputCount; outputIndex += 1) {
          dataView[sharedLayout.outputOffset + outputIndex] =
            outputIndex + 10.5;
        }

        // Act
        const result = await withBrowserWorkerGlobals(async () => {
          const originalSetTimeout = globalThis.setTimeout;

          globalThis.setTimeout = ((
            handler: TimerHandler,
            _timeout?: number,
            ...arguments_: unknown[]
          ) => {
            timerTurns += 1;

            if (typeof handler === 'function') {
              handler(...arguments_);
            }

            return 0 as unknown as ReturnType<typeof setTimeout>;
          }) as unknown as typeof setTimeout;

          try {
            await SHARED_INFERENCE_HOST_INTERNALS.copyInputValuesIntoSharedBuffer(
              dataView,
              inputValues,
              sharedLayout.inputOffset,
              false,
            );
            const detachedOutput =
              await SHARED_INFERENCE_HOST_INTERNALS.copySharedOutputValues(
                dataView,
                sharedLayout.outputOffset,
                outputCount,
                false,
              );

            return {
              detachedOutputValues: Array.from(detachedOutput),
              inputValues: Array.from(
                dataView.subarray(
                  sharedLayout.inputOffset,
                  sharedLayout.outputOffset,
                ),
              ),
              timerTurns,
            };
          } finally {
            globalThis.setTimeout = originalSetTimeout;
          }
        });

        // Assert
        expect(result).toEqual({
          detachedOutputValues: [10.5, 11.5, 12.5],
          inputValues,
          timerTurns: 0,
        });
      });

      it('closes one shared worker when a chunked browser input copy fails before readiness changes', async () => {
        // Arrange
        const payload = createSyntheticTransferablePayload(65_537, 1);

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdWorkers }) => {
            const sharedWorker = openSharedInferenceWorker(payload, {
              workerUrl: 'browser-shared-worker.js',
            });
            const browserWorker = createdWorkers[0] as unknown as {
              emit: (type: string, event: unknown) => void;
            };
            const originalSetTimeout = globalThis.setTimeout;

            globalThis.setTimeout = (() => {
              throw new Error('shared chunk timer failed');
            }) as unknown as typeof setTimeout;

            try {
              browserWorker.emit('message', { data: { type: 'ready' } });
              sharedWorker.submitInput(
                Array.from(
                  { length: payload.inputCount },
                  (_, inputIndex) => inputIndex + 0.25,
                ),
              );
              const awaitOutputError = await sharedWorker
                .awaitOutput()
                .catch((error) => (error as Error).message);
              await sharedWorker.release();

              return {
                awaitOutputError,
                isReady: sharedWorker.isReady,
              };
            } finally {
              globalThis.setTimeout = originalSetTimeout;
            }
          },
        );

        // Assert
        expect(result).toEqual({
          awaitOutputError: 'shared chunk timer failed',
          isReady: false,
        });
      });

      it('releases one browser-backed shared worker even when async termination rejects after bootstrap', async () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );

        // Act
        const result = await withBrowserWorkerGlobals(
          async ({ createdWorkers }) => {
            const sharedWorker = openSharedInferenceWorker(payload, {
              workerUrl: 'browser-shared-worker.js',
            });
            const browserWorker = createdWorkers[0] as unknown as {
              emit: (type: string, event: unknown) => void;
              options: { type: 'module' };
              scriptUrl: string;
              terminate: () => Promise<number>;
            };

            browserWorker.emit('message', { data: { type: 'ready' } });
            browserWorker.terminate = () => {
              return Promise.reject(new Error('shared-browser-terminate-race'));
            };
            const readyBeforeRelease = sharedWorker.isReady;
            await sharedWorker.release();

            return {
              readyAfterRelease: sharedWorker.isReady,
              readyBeforeRelease,
              workerCount: createdWorkers.length,
              workerScript: browserWorker.scriptUrl,
              workerType: browserWorker.options.type,
            };
          },
        );

        // Assert
        expect(result).toEqual({
          readyAfterRelease: false,
          readyBeforeRelease: true,
          workerCount: 1,
          workerScript: 'browser-shared-worker.js',
          workerType: 'module',
        });
      });
    });

    describe('worker internal helpers', () => {
      it('registers one browser message listener when process is unavailable', () => {
        // Arrange
        const originalAddEventListener = globalThis.addEventListener;
        const capturedRegistrations: Array<{
          once: boolean | undefined;
          type: string;
        }> = [];

        Reflect.set(
          globalThis,
          'addEventListener',
          (
            type: string,
            _listener: (event: unknown) => void,
            options?: { once?: boolean },
          ) => {
            capturedRegistrations.push({ once: options?.once, type });
          },
        );

        // Act
        withProcessOverride(undefined, () => {
          registerSharedInferenceWorkerRuntime();
        });

        Reflect.set(globalThis, 'addEventListener', originalAddEventListener);

        // Assert
        expect(capturedRegistrations).toEqual([
          { once: true, type: 'message' },
        ]);
      });

      it('registers one node parent-port listener through the public shared worker runtime wrapper', async () => {
        // Arrange
        const originalBuiltinModuleResolver = Reflect.get(
          process,
          'getBuiltinModule',
        ) as ((moduleSpecifier: string) => unknown) | undefined;
        const registrationTypes: string[] = [];

        Reflect.set(process, 'getBuiltinModule', (moduleSpecifier: string) => {
          if (moduleSpecifier !== 'worker_threads') {
            return undefined;
          }

          return {
            parentPort: {
              once(type: string) {
                registrationTypes.push(type);
              },
            },
          };
        });

        // Act
        registerSharedInferenceWorkerRuntime();
        await Promise.resolve();
        await Promise.resolve();

        Reflect.set(process, 'getBuiltinModule', originalBuiltinModuleResolver);

        // Assert
        expect(registrationTypes).toEqual(['message']);
      });

      it('auto-registers one browser runtime when the shared worker module loads in a worker-like global scope', async () => {
        // Arrange
        const originalAddEventListener = globalThis.addEventListener;
        const originalClose = globalThis.close;
        const originalDocument = Reflect.get(globalThis, 'document');
        const originalPostMessage = Reflect.get(globalThis, 'postMessage') as
          | ((message: unknown) => void)
          | undefined;
        let capturedMessageListener: ((event: unknown) => void) | undefined;
        let closeCallCount = 0;
        const postedMessages: unknown[] = [];
        const registrations: Array<{
          once: boolean | undefined;
          type: string;
        }> = [];

        Reflect.set(
          globalThis,
          'addEventListener',
          (
            type: string,
            listener: (event: unknown) => void,
            options?: { once?: boolean },
          ) => {
            capturedMessageListener = listener;
            registrations.push({ once: options?.once, type });
          },
        );
        Reflect.set(globalThis, 'close', () => {
          closeCallCount += 1;
        });
        Reflect.set(globalThis, 'document', undefined);
        Reflect.set(globalThis, 'postMessage', (message: unknown) => {
          postedMessages.push(message);
        });

        // Act
        const result = await withProcessOverride(undefined, async () => {
          SHARED_INFERENCE_WORKER_INTERNALS.autoRegisterSharedInferenceWorkerRuntime();
          capturedMessageListener?.({ data: { type: 'bootstrap' } });

          return {
            closeCallCount,
            postedMessages,
            registrations,
          };
        });

        Reflect.set(globalThis, 'addEventListener', originalAddEventListener);
        Reflect.set(globalThis, 'close', originalClose);
        Reflect.set(globalThis, 'document', originalDocument);
        Reflect.set(globalThis, 'postMessage', originalPostMessage);

        // Assert
        expect(result).toEqual({
          closeCallCount: 1,
          postedMessages: [
            {
              message:
                'SharedInferenceWorker expected bootstrap payload plus shared buffers.',
              type: 'request-error',
            },
          ],
          registrations: [{ once: true, type: 'message' }],
        });
      });

      it('registers one node parent-port listener and schedules the shared loop after bootstrap', () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );
        const { controlBuffer, dataBuffer, layout } =
          createSharedInferenceBufferPair(
            payload.inputCount,
            payload.outputCount,
          );
        let capturedBootstrapListener: ((message: unknown) => void) | undefined;
        let exitCallCount = 0;
        let runLoopCallCount = 0;
        let scheduledLoopCallCount = 0;
        const postedMessages: unknown[] = [];
        const parentPort = {
          close() {
            return undefined;
          },
          once(_type: string, listener: (message: unknown) => void) {
            capturedBootstrapListener = listener;
          },
          postMessage(message: unknown) {
            postedMessages.push(message);
          },
        };

        // Act
        SHARED_INFERENCE_WORKER_INTERNALS.registerNodeSharedInferenceWorkerRuntime(
          parentPort,
          () => {
            exitCallCount += 1;
          },
          (callback) => {
            scheduledLoopCallCount += 1;
            callback();
          },
          () => {
            runLoopCallCount += 1;
            return undefined as never;
          },
        );
        capturedBootstrapListener?.({
          controlBuffer,
          dataBuffer,
          payload,
          type: 'bootstrap',
        });

        // Assert
        expect({
          exitCallCount,
          inputCount: Atomics.load(
            new Int32Array(controlBuffer),
            layout.statusIndexes.inputCount,
          ),
          postedMessages,
          runLoopCallCount,
          scheduledLoopCallCount,
          status: Atomics.load(
            new Int32Array(controlBuffer),
            layout.statusIndexes.status,
          ),
        }).toEqual({
          exitCallCount: 0,
          inputCount: payload.inputCount,
          postedMessages: [{ type: 'ready' }],
          runLoopCallCount: 1,
          scheduledLoopCallCount: 1,
          status: 0,
        });
      });

      it('ignores one non-bootstrap node message and exits after one malformed bootstrap', () => {
        // Arrange
        let capturedBootstrapListener: ((message: unknown) => void) | undefined;
        let closeCallCount = 0;
        let exitCallCount = 0;
        const postedMessages: unknown[] = [];
        const parentPort = {
          close() {
            closeCallCount += 1;
          },
          once(_type: string, listener: (message: unknown) => void) {
            capturedBootstrapListener = listener;
          },
          postMessage(message: unknown) {
            postedMessages.push(message);
          },
        };

        // Act
        SHARED_INFERENCE_WORKER_INTERNALS.registerNodeSharedInferenceWorkerRuntime(
          parentPort,
          () => {
            exitCallCount += 1;
          },
        );
        capturedBootstrapListener?.({ type: 'not-bootstrap' });
        capturedBootstrapListener?.({ type: 'bootstrap' });

        // Assert
        expect({
          closeCallCount,
          exitCallCount,
          postedMessages,
        }).toEqual({
          closeCallCount: 1,
          exitCallCount: 1,
          postedMessages: [
            {
              message:
                'SharedInferenceWorker expected bootstrap payload plus shared buffers.',
              type: 'request-error',
            },
          ],
        });
      });

      it('loads node worker_threads through the builtin resolver or opaque import for the shared worker helper', async () => {
        // Act
        const requireLoaded =
          await SHARED_INFERENCE_WORKER_INTERNALS.loadNodeWorkerThreadsModule(
            () => ({
              parentPort: {
                close() {
                  return undefined;
                },
              },
            }),
            async () => {
              throw new Error(
                'shared worker helper should prefer the builtin resolver.',
              );
            },
          );
        const importLoaded =
          await SHARED_INFERENCE_WORKER_INTERNALS.loadNodeWorkerThreadsModule(
            () => undefined,
            async (moduleSpecifier) => ({
              moduleSpecifier,
              parentPort: {
                once() {
                  return undefined;
                },
              },
            }),
          );

        // Assert
        expect({
          importHasParentPortListener:
            typeof importLoaded.parentPort?.once === 'function',
          importSpecifier: (
            importLoaded as unknown as { moduleSpecifier: string }
          ).moduleSpecifier,
          requireHasParentPortClose:
            typeof requireLoaded.parentPort?.close === 'function',
        }).toEqual({
          importHasParentPortListener: true,
          importSpecifier: 'worker_threads',
          requireHasParentPortClose: true,
        });
      });

      it('bootstraps shared buffers and rejects malformed bootstrap payloads', () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );
        const { controlBuffer, dataBuffer, layout } =
          createSharedInferenceBufferPair(
            payload.inputCount,
            payload.outputCount,
          );
        let malformedError = '';

        // Act
        const bootstrapState =
          SHARED_INFERENCE_WORKER_INTERNALS.handleBootstrapMessage({
            controlBuffer,
            dataBuffer,
            payload,
            type: 'bootstrap',
          });
        const ignoredBootstrapState =
          SHARED_INFERENCE_WORKER_INTERNALS.handleBootstrapMessage({
            type: 'not-bootstrap',
          });

        try {
          SHARED_INFERENCE_WORKER_INTERNALS.handleBootstrapMessage({
            type: 'bootstrap',
          });
        } catch (error) {
          malformedError = (error as Error).message;
        }

        // Assert
        expect({
          ignoredBootstrapState,
          inputCount: Atomics.load(
            bootstrapState?.controlView ?? new Int32Array(controlBuffer),
            layout.statusIndexes.inputCount,
          ),
          malformedError,
          outputLength: bootstrapState?.dataView.length,
          status: Atomics.load(
            bootstrapState?.controlView ?? new Int32Array(controlBuffer),
            layout.statusIndexes.status,
          ),
        }).toEqual({
          ignoredBootstrapState: undefined,
          inputCount: payload.inputCount,
          malformedError:
            'SharedInferenceWorker expected bootstrap payload plus shared buffers.',
          outputLength: layout.dataElementCount,
          status: 0,
        });
      });

      it('handles predict, reset, closed, and waiting shared loop steps', () => {
        // Arrange
        const layout =
          SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout(
            2,
            1,
          );
        const controlView = new Int32Array(
          new SharedArrayBuffer(
            Int32Array.BYTES_PER_ELEMENT * layout.controlElementCount,
          ),
        );
        const dataView = new Float64Array(
          new SharedArrayBuffer(
            Float64Array.BYTES_PER_ELEMENT * layout.dataElementCount,
          ),
        );
        const loopEvents: string[] = [];
        const predictor = {
          predict(inputValues: number[]) {
            return [inputValues.reduce((sum, value) => sum + value, 0)];
          },
          reset() {
            loopEvents.push('reset');
          },
          strategy: 'transferable',
        } as ReturnType<typeof createInferencePredictor>;

        dataView.set([0.25, 0.75], layout.inputOffset);
        Atomics.store(
          controlView,
          layout.statusIndexes.status,
          layout.statusValues.inputReady,
        );

        // Act
        const predictState =
          SHARED_INFERENCE_WORKER_INTERNALS.handleSharedInferenceLoopStep(
            controlView,
            dataView,
            predictor,
            layout,
          );
        const outputValue = dataView[layout.outputOffset];
        Atomics.store(
          controlView,
          layout.statusIndexes.status,
          layout.statusValues.resetRequested,
        );
        const resetState =
          SHARED_INFERENCE_WORKER_INTERNALS.handleSharedInferenceLoopStep(
            controlView,
            dataView,
            predictor,
            layout,
          );
        const postResetStatus = Atomics.load(
          controlView,
          layout.statusIndexes.status,
        );
        Atomics.store(
          controlView,
          layout.statusIndexes.status,
          layout.statusValues.closed,
        );
        const closedState =
          SHARED_INFERENCE_WORKER_INTERNALS.handleSharedInferenceLoopStep(
            controlView,
            dataView,
            predictor,
            layout,
          );
        Atomics.store(
          controlView,
          layout.statusIndexes.status,
          layout.statusValues.idle,
        );
        const waitingState =
          SHARED_INFERENCE_WORKER_INTERNALS.handleSharedInferenceLoopStep(
            controlView,
            dataView,
            predictor,
            layout,
          );

        // Assert
        expect({
          closedState,
          loopEvents,
          outputValue,
          postResetStatus,
          predictState,
          resetState,
          waitingState,
        }).toEqual({
          closedState: 'closed',
          loopEvents: ['reset'],
          outputValue: 1,
          postResetStatus: 0,
          predictState: 'predicted',
          resetState: 'reset',
          waitingState: 'waiting',
        });
      });

      it('returns from the shared worker loop when the host posts the closed status', () => {
        // Arrange
        const layout =
          SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout(
            1,
            1,
          );
        const controlView = new Int32Array(
          new SharedArrayBuffer(
            Int32Array.BYTES_PER_ELEMENT * layout.controlElementCount,
          ),
        );
        const dataView = new Float64Array(
          new SharedArrayBuffer(
            Float64Array.BYTES_PER_ELEMENT * layout.dataElementCount,
          ),
        );
        const predictor = {
          predict() {
            return [1];
          },
          reset() {
            return undefined;
          },
          strategy: 'transferable',
        } as ReturnType<typeof createInferencePredictor>;
        let waitCallCount = 0;

        Atomics.store(
          controlView,
          layout.statusIndexes.status,
          layout.statusValues.closed,
        );

        // Act
        SHARED_INFERENCE_WORKER_INTERNALS.runSharedInferenceLoop(
          controlView,
          dataView,
          predictor,
          layout,
          () => {
            waitCallCount += 1;
          },
        );

        // Assert
        expect(waitCallCount).toBe(0);
      });

      it('uses shared worker message wrappers, browser registration, and exit helpers', () => {
        // Arrange
        const payload = exportTransferableInferencePayload(
          createWorkerPayloadNetwork(),
        );
        const { controlBuffer, dataBuffer } = createSharedInferenceBufferPair(
          payload.inputCount,
          payload.outputCount,
        );
        const postedMessages: unknown[] = [];
        const capturedListeners: Array<(event: unknown) => void> = [];
        const directPostMessages: unknown[] = [];
        const originalClose = globalThis.close;
        const originalProcessExit = process.exit;
        let browserCloseCallCount = 0;
        let nodeExitCallCount = 0;
        let runLoopCallCount = 0;
        let scheduledLoopCallCount = 0;

        Reflect.set(globalThis, 'close', () => {
          browserCloseCallCount += 1;
        });
        process.exit = (() => {
          nodeExitCallCount += 1;
          return undefined as never;
        }) as typeof process.exit;

        // Act
        SHARED_INFERENCE_WORKER_INTERNALS.registerBrowserSharedInferenceWorkerRuntime(
          (listener) => {
            capturedListeners.push(listener);
          },
          (message) => {
            postedMessages.push(message);
          },
          () => {
            browserCloseCallCount += 1;
          },
          (callback) => {
            scheduledLoopCallCount += 1;
            callback();
          },
          () => {
            runLoopCallCount += 1;
            return undefined as never;
          },
        );
        capturedListeners[0]?.({
          data: {
            controlBuffer,
            dataBuffer,
            payload,
            type: 'bootstrap',
          },
        });
        capturedListeners[0]?.({
          data: {
            type: 'not-bootstrap',
          },
        });
        capturedListeners[0]?.({
          data: {
            type: 'bootstrap',
          },
        });
        SHARED_INFERENCE_WORKER_INTERNALS.postWorkerMessage(
          {
            postMessage(message: unknown) {
              directPostMessages.push(message);
            },
          },
          { type: 'ready' },
        );
        SHARED_INFERENCE_WORKER_INTERNALS.closeCurrentBrowserSharedInferenceWorkerRuntime();
        SHARED_INFERENCE_WORKER_INTERNALS.exitCurrentNodeSharedInferenceWorkerRuntime();

        Reflect.set(globalThis, 'close', originalClose);
        process.exit = originalProcessExit;

        // Assert
        expect({
          asErrorMessage: SHARED_INFERENCE_WORKER_INTERNALS.asError(
            'worker-string-error',
          ).message,
          browserCloseCallCount,
          directPostMessages,
          nodeExitCallCount,
          normalizedMessages: [
            SHARED_INFERENCE_WORKER_INTERNALS.resolveMessageEventData({
              data: 'message-data',
            }),
            SHARED_INFERENCE_WORKER_INTERNALS.resolveMessageEventData(
              'raw-message',
            ),
          ],
          postedMessages,
          runLoopCallCount,
          scheduledLoopCallCount,
        }).toEqual({
          asErrorMessage: 'worker-string-error',
          browserCloseCallCount: 2,
          directPostMessages: [{ type: 'ready' }],
          nodeExitCallCount: 1,
          normalizedMessages: ['message-data', 'raw-message'],
          postedMessages: [
            { type: 'ready' },
            {
              message:
                'SharedInferenceWorker expected bootstrap payload plus shared buffers.',
              type: 'request-error',
            },
          ],
          runLoopCallCount: 1,
          scheduledLoopCallCount: 1,
        });
      });

      it('uses the shared worker wait wrapper inside the loop when the status stays idle', () => {
        // Arrange
        const layout =
          SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout(
            1,
            1,
          );
        const controlView = new Int32Array(
          new SharedArrayBuffer(
            Int32Array.BYTES_PER_ELEMENT * layout.controlElementCount,
          ),
        );
        const dataView = new Float64Array(
          new SharedArrayBuffer(
            Float64Array.BYTES_PER_ELEMENT * layout.dataElementCount,
          ),
        );
        const predictor = {
          predict() {
            return [1];
          },
          reset() {
            return undefined;
          },
          strategy: 'transferable',
        } as ReturnType<typeof createInferencePredictor>;
        let waitState = '';

        Atomics.store(
          controlView,
          layout.statusIndexes.status,
          layout.statusValues.inputReady,
        );

        // Act
        try {
          SHARED_INFERENCE_WORKER_INTERNALS.runSharedInferenceLoop(
            controlView,
            dataView,
            predictor,
            layout,
            (_nextControlView, _statusIndex, currentStatus) => {
              waitState = `wait:${String(currentStatus)}`;
              throw new Error('stop-shared-loop');
            },
          );
        } catch {
          // Swallow the sentinel stop error from the injected wait function.
        }

        // Assert
        expect(waitState).toBe('wait:2');
      });

      it('continues the shared worker loop after one prediction before waiting again', () => {
        // Arrange
        const layout =
          SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout(
            1,
            1,
          );
        const controlView = new Int32Array(
          new SharedArrayBuffer(
            Int32Array.BYTES_PER_ELEMENT * layout.controlElementCount,
          ),
        );
        const dataView = new Float64Array(
          new SharedArrayBuffer(
            Float64Array.BYTES_PER_ELEMENT * layout.dataElementCount,
          ),
        );
        const predictor = {
          predict(inputValues: number[]) {
            return [inputValues[0] * 2];
          },
          reset() {
            return undefined;
          },
          strategy: 'transferable',
        } as ReturnType<typeof createInferencePredictor>;
        let thrownError = '';
        let waitCallCount = 0;

        dataView.set([0.5], layout.inputOffset);
        Atomics.store(
          controlView,
          layout.statusIndexes.status,
          layout.statusValues.inputReady,
        );

        // Act
        try {
          SHARED_INFERENCE_WORKER_INTERNALS.runSharedInferenceLoop(
            controlView,
            dataView,
            predictor,
            layout,
            () => {
              waitCallCount += 1;
              throw new Error('stop-shared-loop');
            },
          );
        } catch (error) {
          thrownError = (error as Error).message;
        }

        // Assert
        expect({
          outputValue: dataView[layout.outputOffset],
          status: Atomics.load(controlView, layout.statusIndexes.status),
          thrownError,
          waitCallCount,
        }).toEqual({
          outputValue: 1,
          status: layout.statusValues.outputReady,
          thrownError: 'stop-shared-loop',
          waitCallCount: 1,
        });
      });

      it('uses the default shared worker wait helper when the loop waits without an override', () => {
        // Arrange
        const layout =
          SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout(
            1,
            1,
          );
        const controlView = new Int32Array(
          new SharedArrayBuffer(
            Int32Array.BYTES_PER_ELEMENT * layout.controlElementCount,
          ),
        );
        const dataView = new Float64Array(
          new SharedArrayBuffer(
            Float64Array.BYTES_PER_ELEMENT * layout.dataElementCount,
          ),
        );
        const predictor = {
          predict() {
            return [1];
          },
          reset() {
            return undefined;
          },
          strategy: 'transferable',
        } as ReturnType<typeof createInferencePredictor>;
        const originalAtomicsWait = Atomics.wait;
        let thrownError = '';
        let waitCall = '';

        Reflect.set(
          Atomics,
          'wait',
          (typedArray: Int32Array, index: number, value: number) => {
            waitCall = `${String(index)}:${String(value)}:${String(typedArray.length)}`;
            throw new Error('stop-default-shared-loop');
          },
        );

        // Act
        try {
          SHARED_INFERENCE_WORKER_INTERNALS.runSharedInferenceLoop(
            controlView,
            dataView,
            predictor,
            layout,
          );
        } catch (error) {
          thrownError = (error as Error).message;
        }

        Reflect.set(Atomics, 'wait', originalAtomicsWait);

        // Assert
        expect({
          thrownError,
          waitCall,
        }).toEqual({
          thrownError: 'stop-default-shared-loop',
          waitCall: '0:0:2',
        });
      });

      it('delegates the default shared worker wait helper to Atomics.wait', () => {
        // Arrange
        const originalAtomicsWait = Atomics.wait;
        const controlView = new Int32Array(
          new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT),
        );
        let waitCall = '';

        Reflect.set(
          Atomics,
          'wait',
          (typedArray: Int32Array, index: number, value: number) => {
            waitCall = `${String(index)}:${String(value)}:${String(typedArray.length)}`;

            return 'ok';
          },
        );

        // Act
        SHARED_INFERENCE_WORKER_INTERNALS.defaultWaitForSharedStatusChange(
          controlView,
          0,
          0,
        );

        Reflect.set(Atomics, 'wait', originalAtomicsWait);

        // Assert
        expect(waitCall).toBe('0:0:1');
      });
    });
  });

  describe('transferable inference payloads', () => {
    describe('when a runtime network is exported for typed-array transport', () => {
      it('packs activation steps and runtime-significant node fields into aligned typed arrays', () => {
        // Arrange
        const network = createWorkerPayloadNetwork();
        const inferenceIr = extractNetworkInferenceIR(network);
        const flattenedActivationSteps = flattenActivationSteps(
          inferenceIr.activationSteps,
        );

        // Act
        const payload = exportTransferableInferencePayload(network);

        // Assert
        expect({
          activationStepsData: Array.from(payload.activationStepsData),
          activationStepsIndex: Array.from(payload.activationStepsIndex),
          activationTableLength: payload.activationTableLength,
          edgeFrom: Array.from(payload.edgeFrom),
          edgeGaterIndices: Array.from(payload.edgeGaterIndices),
          edgeTo: Array.from(payload.edgeTo),
          edgeWeights: roundVector(Array.from(payload.edgeWeights)),
          inputCount: payload.inputCount,
          nodeActivationIds: Array.from(payload.nodeActivationIds),
          nodeBiases: roundVector(Array.from(payload.nodeBiases)),
          nodeIds: Array.from(payload.nodeIds),
          nodeMasks: roundVector(Array.from(payload.nodeMasks)),
          nodeResponses: roundVector(Array.from(payload.nodeResponses)),
          nodeSelfGaterIndices: Array.from(payload.nodeSelfGaterIndices),
          nodeSelfWeights: roundVector(Array.from(payload.nodeSelfWeights)),
          outputCount: payload.outputCount,
          outputNodeIndices: Array.from(payload.outputNodeIndices),
          strategy: payload.strategy,
          version: payload.version,
        }).toEqual({
          activationStepsData: flattenedActivationSteps.activationStepsData,
          activationStepsIndex: flattenedActivationSteps.activationStepsIndex,
          activationTableLength: INFERENCE_ACTIVATION_TABLE.length,
          edgeFrom: inferenceIr.edges.map(
            (inferenceEdge) => inferenceEdge.from,
          ),
          edgeGaterIndices: inferenceIr.edges.map(
            (inferenceEdge) => inferenceEdge.gaterIndex,
          ),
          edgeTo: inferenceIr.edges.map((inferenceEdge) => inferenceEdge.to),
          edgeWeights: roundVector(
            inferenceIr.edges.map((inferenceEdge) => inferenceEdge.weight),
          ),
          inputCount: inferenceIr.inputCount,
          nodeActivationIds: inferenceIr.nodes.map(
            (inferenceNode) => inferenceNode.activationId,
          ),
          nodeBiases: roundVector(
            inferenceIr.nodes.map((inferenceNode) => inferenceNode.bias),
          ),
          nodeIds: inferenceIr.nodes.map(
            (inferenceNode) => inferenceNode.index,
          ),
          nodeMasks: roundVector(
            inferenceIr.nodes.map((inferenceNode) => inferenceNode.mask),
          ),
          nodeResponses: roundVector(
            inferenceIr.nodes.map((inferenceNode) => inferenceNode.response),
          ),
          nodeSelfGaterIndices: inferenceIr.nodes.map(
            (inferenceNode) => inferenceNode.selfGaterIndex,
          ),
          nodeSelfWeights: roundVector(
            inferenceIr.nodes.map((inferenceNode) => inferenceNode.selfWeight),
          ),
          outputCount: inferenceIr.outputCount,
          outputNodeIndices: [...inferenceIr.outputNodeIndices],
          strategy: 'transferable',
          version: 1,
        });
      });

      it('returns every typed-array buffer in transfer-ready payload order', () => {
        // Arrange
        const network = createWorkerPayloadNetwork();
        const payload = exportTransferableInferencePayload(network);

        // Act
        const transferList = getTransferList(payload);

        // Assert
        expect(transferList).toEqual([
          payload.activationStepsIndex.buffer,
          payload.activationStepsData.buffer,
          payload.nodeIds.buffer,
          payload.nodeBiases.buffer,
          payload.nodeResponses.buffer,
          payload.nodeMasks.buffer,
          payload.nodeActivationIds.buffer,
          payload.nodeSelfWeights.buffer,
          payload.nodeSelfGaterIndices.buffer,
          payload.edgeFrom.buffer,
          payload.edgeTo.buffer,
          payload.edgeWeights.buffer,
          payload.edgeGaterIndices.buffer,
          payload.outputNodeIndices.buffer,
        ]);
      });

      it('replays the same inference outputs as the portable predictor and runtime network', () => {
        // Arrange
        const inputValues = [0.25, 0.75];

        // Act
        const { portableOutput, runtimeOutput, transferableOutput } =
          createTransferableRoundtripOutputs(inputValues);

        // Assert
        expect({
          portableOutput: roundVector(portableOutput),
          runtimeOutput: roundVector(runtimeOutput),
          transferableOutput: roundVector(transferableOutput),
        }).toEqual({
          portableOutput: roundVector(runtimeOutput),
          runtimeOutput: roundVector(runtimeOutput),
          transferableOutput: roundVector(runtimeOutput),
        });
      });
    });

    describe('when transferable export requests reduced numeric precision', () => {
      it('emits Float32Array numeric buffers for the precision-sensitive fields', () => {
        // Arrange
        const network = createWorkerPayloadNetwork();

        // Act
        const payload = exportTransferableInferencePayload(network, {
          numericPrecision: 'f32',
        });

        // Assert
        expect({
          edgeWeights: payload.edgeWeights instanceof Float32Array,
          nodeBiases: payload.nodeBiases instanceof Float32Array,
          nodeMasks: payload.nodeMasks instanceof Float32Array,
          nodeResponses: payload.nodeResponses instanceof Float32Array,
          nodeSelfWeights: payload.nodeSelfWeights instanceof Float32Array,
        }).toEqual({
          edgeWeights: true,
          nodeBiases: true,
          nodeMasks: true,
          nodeResponses: true,
          nodeSelfWeights: true,
        });
      });

      it('keeps prediction output within the reduced-precision tolerance budget', () => {
        // Arrange
        const inputValues = [0.25, 0.75];

        // Act
        const { portableOutput, transferableOutput } =
          createTransferableRoundtripOutputs(inputValues, {
            numericPrecision: 'f32',
          });

        // Assert
        expect(
          transferableOutput.map(
            (outputValue, outputIndex) =>
              Math.abs(outputValue - (portableOutput[outputIndex] ?? 0)) <=
              1e-4,
          ),
        ).toEqual([true]);
      });
    });

    describe('when one transferable predictor reuses gated recurrent state', () => {
      it('updates gated gains across warm predictions and returns to baseline after reset', () => {
        // Arrange
        const predictor = createInferencePredictor(
          createManualTransferablePayload({
            inputCount: 2,
            activationStepsIndex: new Int32Array([0, 1]),
            activationStepsData: new Int32Array([2, 3]),
            nodeIds: new Int32Array([0, 1, 2, 3]),
            nodeBiases: new Float64Array([0, 0, 0, 0]),
            nodeResponses: new Float64Array([1, 1, 1, 1]),
            nodeMasks: new Float64Array([1, 1, 1, 1]),
            nodeActivationIds: new Int32Array([2, 2, 2, 2]),
            nodeSelfWeights: new Float64Array([0, 0, 0, 1]),
            nodeSelfGaterIndices: new Int32Array([-1, -1, -1, 2]),
            edgeFrom: new Int32Array([0, 0]),
            edgeTo: new Int32Array([2, 3]),
            edgeWeights: new Float64Array([1, 1]),
            edgeGaterIndices: new Int32Array([-1, 2]),
            outputNodeIndices: new Int32Array([3]),
          }),
        );

        // Act
        const firstOutput = predictor.predict([
          0.5,
          undefined as unknown as number,
        ]);
        const secondOutput = predictor.predict([
          0.5,
          undefined as unknown as number,
        ]);
        predictor.reset();
        const resetOutput = predictor.predict([
          0.5,
          undefined as unknown as number,
        ]);

        // Assert
        expect([
          roundValue(firstOutput[0]),
          roundValue(secondOutput[0]),
          roundValue(resetOutput[0]),
        ]).toEqual([0.25, 0.375, 0.25]);
      });
    });

    describe('when a transferable activation step is truncated', () => {
      it('throws a descriptive missing transferable-step error at prediction time', () => {
        // Arrange
        const predictor = createInferencePredictor(
          createManualTransferablePayload({
            activationStepsIndex: new Int32Array([0, 1]),
            activationStepsData: new Int32Array([]),
          }),
        );

        // Assert
        expect(() => predictor.predict([1])).toThrow(
          'Expected transferable activation step node 0.',
        );
      });
    });

    describe('when a transferable activation step references a missing node entry', () => {
      it('throws a descriptive missing transferable node error at prediction time', () => {
        // Arrange
        const predictor = createInferencePredictor(
          createManualTransferablePayload({
            activationStepsData: new Int32Array([99]),
          }),
        );

        // Assert
        expect(() => predictor.predict([1])).toThrow(
          'Expected transferable node 99.',
        );
      });
    });

    describe('when a transferable output index is outside the activation buffer', () => {
      it('returns zero for the missing output slot instead of leaking undefined', () => {
        // Arrange
        const predictor = createInferencePredictor(
          createManualTransferablePayload({
            outputNodeIndices: new Int32Array([99]),
          }),
        );

        // Act
        const outputValues = predictor.predict([1]);

        // Assert
        expect(outputValues).toEqual([0]);
      });
    });

    describe('when the transferable activation shelf length drifts from the runtime registry', () => {
      it('rejects the payload before predictor construction begins', () => {
        // Arrange
        const malformedPayload = createManualTransferablePayload({
          activationTableLength: INFERENCE_ACTIVATION_TABLE.length - 1,
        });

        // Assert
        expect(() => createInferencePredictor(malformedPayload)).toThrow(
          `Expected activation table length ${INFERENCE_ACTIVATION_TABLE.length}, received ${INFERENCE_ACTIVATION_TABLE.length - 1}.`,
        );
      });
    });

    describe('when transferable node ids are not contiguous from zero', () => {
      it('rejects the malformed transferable payload before prediction begins', () => {
        // Arrange
        const malformedPayload = createManualTransferablePayload({
          nodeIds: new Int32Array([0, 2]),
        });

        // Assert
        expect(() => createInferencePredictor(malformedPayload)).toThrow(
          'Expected transferable node id 1, received 2.',
        );
      });
    });

    describe('when a transferable payload shortens one node shelf', () => {
      it('rejects the malformed node field length during predictor construction', () => {
        // Arrange
        const malformedPayload = createManualTransferablePayload({
          nodeResponses: new Float64Array([1]),
        });

        // Assert
        expect(() => createInferencePredictor(malformedPayload)).toThrow(
          "Expected transferable field 'nodeResponses' to have length 2, received 1.",
        );
      });
    });

    describe('when a transferable payload shortens one edge shelf', () => {
      it('rejects the malformed edge field length during predictor construction', () => {
        // Arrange
        const malformedPayload = createManualTransferablePayload({
          edgeWeights: new Float64Array([]),
        });

        // Assert
        expect(() => createInferencePredictor(malformedPayload)).toThrow(
          "Expected transferable field 'edgeWeights' to have length 1, received 0.",
        );
      });
    });

    describe('when a transferable payload names an unsupported activation id', () => {
      it('rejects the unsupported transferable activation during predictor construction', () => {
        // Arrange
        const malformedPayload = createManualTransferablePayload({
          nodeActivationIds: new Int32Array([2, 999]),
        });

        // Assert
        expect(() => createInferencePredictor(malformedPayload)).toThrow(
          "Unsupported transferable activation id '999' on node 1.",
        );
      });
    });
  });

  describe('portable inference payloads', () => {
    describe('when a runtime network is exported for structured-clone transport', () => {
      it('preserves the runtime-significant node and output metadata needed for prediction', () => {
        // Arrange
        const network = createWorkerPayloadNetwork();

        // Act
        const payload = exportPortableInferencePayload(network);

        // Assert
        expect({
          activationTableLength: payload.activationTable.length,
          outputNodeIndices: payload.outputNodeIndices,
          strategy: payload.strategy,
          version: payload.version,
          visibleNodeFields: payload.nodes.map(
            (portableNode: {
              activation: string;
              id: number;
              mask: number;
              response: number;
              selfGaterIndex: number;
              selfWeight: number;
            }) => ({
              activation: portableNode.activation,
              id: portableNode.id,
              mask: portableNode.mask,
              response: portableNode.response,
              selfGaterIndex: portableNode.selfGaterIndex,
              selfWeight: portableNode.selfWeight,
            }),
          ),
        }).toEqual({
          activationTableLength: INFERENCE_ACTIVATION_TABLE.length,
          outputNodeIndices: resolveOutputNodeIndexes(network),
          strategy: 'portable',
          version: 1,
          visibleNodeFields: network.nodes.map((node) => ({
            activation: resolveActivationKey(node.squash as ActivationFunction),
            id: node.index,
            mask: node.mask,
            response: node.response,
            selfGaterIndex: -1,
            selfWeight: 0,
          })),
        });
      });
    });

    describe('when a portable predictor replays a feed-forward inference step', () => {
      it('matches the runtime network output for the same input vector', () => {
        // Arrange
        const inputValues = [0.25, 0.75];

        // Act
        const { predictedOutput, runtimeOutput } =
          createPortableRoundtripOutputs(inputValues);

        // Assert
        expect(roundVector(predictedOutput)).toEqual(
          roundVector(runtimeOutput),
        );
      });
    });

    describe('when a runtime network uses a feed-forward mutation activation', () => {
      it('exports and replays the supported smooth activation without transport drift', () => {
        // Arrange
        const inputValues = [0.25, 0.75];
        const { network, outputNode } = createWorkerPayloadNetworkParts();
        outputNode.squash = methods.Activation.swish;
        disableFastSlab(network);

        // Act
        const payload = exportPortableInferencePayload(network);
        const predictor = createInferencePredictor(payload);
        const predictedOutput = predictor.predict(inputValues);
        const runtimeOutput = network.noTraceActivate([...inputValues]);

        // Assert
        expect(roundVector(predictedOutput)).toEqual(
          roundVector(runtimeOutput),
        );
      });
    });

    describe('when predictor state is reset between recurrent calls', () => {
      it('returns to the same output as a fresh runtime network baseline', () => {
        // Arrange
        const network = createWorkerPayloadNetwork();
        const recurrentOutputNode = network.nodes.find(
          (candidateNode) => candidateNode.type === 'output',
        );

        if (!recurrentOutputNode) {
          throw new Error(
            'Expected an output node for the recurrent reset test.',
          );
        }

        const selfConnection = new Connection(
          recurrentOutputNode,
          recurrentOutputNode,
          0.5,
        );
        recurrentOutputNode.connections.self = [selfConnection];
        network.selfconns = [selfConnection];
        disableFastSlab(network);

        const payload = exportPortableInferencePayload(network);
        const predictor = createInferencePredictor(payload);

        predictor.predict([0.5, 0.25]);
        network.noTraceActivate([0.5, 0.25]);

        // Act
        predictor.reset();
        network.clear();
        const predictedOutput = predictor.predict([0.5, 0.25]);
        const runtimeOutput = network.noTraceActivate([0.5, 0.25]);

        // Assert
        expect(roundVector(predictedOutput)).toEqual(
          roundVector(runtimeOutput),
        );
      });
    });

    describe('when gated edges and self-connections reuse one warm predictor', () => {
      it('updates gate gains and nullish input slots consistently across repeated predictions', () => {
        // Arrange
        const predictor = createInferencePredictor(
          createManualPortablePayload({
            inputCount: 2,
            activationSteps: [[2], [3]],
            nodes: [
              {
                id: 0,
                bias: 0,
                response: 1,
                mask: 1,
                activation: 'identity',
                selfWeight: 0,
                selfGaterIndex: -1,
              },
              {
                id: 1,
                bias: 0,
                response: 1,
                mask: 1,
                activation: 'identity',
                selfWeight: 0,
                selfGaterIndex: -1,
              },
              {
                id: 2,
                bias: 0,
                response: 1,
                mask: 1,
                activation: 'identity',
                selfWeight: 0,
                selfGaterIndex: -1,
              },
              {
                id: 3,
                bias: 0,
                response: 1,
                mask: 1,
                activation: 'identity',
                selfWeight: 1,
                selfGaterIndex: 2,
              },
            ],
            edges: [
              {
                from: 0,
                to: 2,
                weight: 1,
                gaterIndex: -1,
              },
              {
                from: 0,
                to: 3,
                weight: 1,
                gaterIndex: 2,
              },
            ],
            outputNodeIndices: [3],
          }),
        );

        // Act
        const firstOutput = predictor.predict([
          0.5,
          undefined as unknown as number,
        ]);
        const secondOutput = predictor.predict([
          0.5,
          undefined as unknown as number,
        ]);

        // Assert
        expect([
          roundValue(firstOutput[0]),
          roundValue(secondOutput[0]),
        ]).toEqual([0.25, 0.375]);
      });
    });

    describe('when portable node ids are not contiguous from zero', () => {
      it('rejects the malformed payload before prediction begins', () => {
        // Arrange
        const malformedPayload = createManualPortablePayload({
          inputCount: 0,
          outputCount: 0,
          activationSteps: [],
          nodes: [
            {
              id: 1,
              bias: 0,
              response: 1,
              mask: 1,
              activation: 'identity',
              selfWeight: 0,
              selfGaterIndex: -1,
            },
          ],
          edges: [],
          outputNodeIndices: [],
        });

        // Assert
        expect(() => createInferencePredictor(malformedPayload)).toThrow(
          'Expected portable node id 0, received 1.',
        );
      });
    });

    describe('when a portable payload names an activation outside its table', () => {
      it('rejects the unsupported portable activation during predictor construction', () => {
        // Arrange
        const malformedPayload = createManualPortablePayload({
          inputCount: 0,
          outputCount: 0,
          activationSteps: [],
          nodes: [
            {
              id: 0,
              bias: 0,
              response: 1,
              mask: 1,
              activation: 'unknown',
              selfWeight: 0,
              selfGaterIndex: -1,
            },
          ],
          edges: [],
          outputNodeIndices: [],
        });

        // Assert
        expect(() => createInferencePredictor(malformedPayload)).toThrow(
          "Unsupported portable activation 'unknown' on node 0.",
        );
      });
    });

    describe('when prediction input width does not match the payload contract', () => {
      it('throws a descriptive input-length error', () => {
        // Arrange
        const predictor = createInferencePredictor(
          createManualPortablePayload(),
        );

        // Assert
        expect(() => predictor.predict([])).toThrow(
          'Expected 1 input values, received 0.',
        );
      });
    });

    describe('when an output index is outside the activation buffer', () => {
      it('returns zero for the missing output slot instead of leaking undefined', () => {
        // Arrange
        const predictor = createInferencePredictor(
          createManualPortablePayload({
            outputNodeIndices: [99],
          }),
        );

        // Act
        const outputValues = predictor.predict([1]);

        // Assert
        expect(outputValues).toEqual([0]);
      });
    });

    describe('when activationSteps reference a missing node entry', () => {
      it('throws a descriptive missing portable node error at prediction time', () => {
        // Arrange
        const predictor = createInferencePredictor(
          createManualPortablePayload({
            inputCount: 0,
            outputCount: 0,
            activationSteps: [[1]],
            nodes: [
              {
                id: 0,
                bias: 0,
                response: 1,
                mask: 1,
                activation: 'identity',
                selfWeight: 0,
                selfGaterIndex: -1,
              },
            ],
            edges: [],
            outputNodeIndices: [],
          }),
        );

        // Assert
        expect(() => predictor.predict([])).toThrow(
          'Expected portable node 1.',
        );
      });
    });

    describe('when payload edges change after predictor construction', () => {
      it('throws a descriptive missing portable edge error during prediction', () => {
        // Arrange
        const payload = createManualPortablePayload();
        const predictor = createInferencePredictor(payload);
        payload.edges[0] =
          undefined as unknown as (typeof payload.edges)[number];

        // Assert
        expect(() => predictor.predict([1])).toThrow(
          'Expected portable edge 0.',
        );
      });
    });
  });

  describe('extractNetworkInferenceIR', () => {
    describe('when the same runtime network is snapshotted twice', () => {
      it('returns the same deterministic IR payload', () => {
        // Arrange
        const network = createWorkerPayloadNetwork();

        // Act
        const firstInferenceIr = extractNetworkInferenceIR(network);
        const secondInferenceIr = extractNetworkInferenceIR(network);

        // Assert
        expect(firstInferenceIr).toEqual(secondInferenceIr);
      });
    });

    describe('when runtime nodes use supported activation functions and scalar modifiers', () => {
      it('records stable activation ids plus response and mask values', () => {
        // Arrange
        const network = createWorkerPayloadNetwork();

        // Act
        const inferenceIr = extractNetworkInferenceIR(network);

        // Assert
        expect(
          inferenceIr.nodes.map((inferenceNode) => ({
            activationName:
              INFERENCE_ACTIVATION_TABLE[inferenceNode.activationId]?.name,
            index: inferenceNode.index,
            mask: inferenceNode.mask,
            response: inferenceNode.response,
          })),
        ).toEqual(
          network.nodes.map((node) => ({
            activationName: node.squash.name,
            index: node.index,
            mask: node.mask,
            response: node.response,
          })),
        );
      });
    });

    describe('when the runtime network has a compiled activation schedule', () => {
      it('preserves grouped runtime step boundaries in activationSteps', () => {
        // Arrange
        const expectedActivationSteps = resolveExpectedActivationSteps();
        const network = createWorkerPayloadNetwork();

        // Act
        const inferenceIr = extractNetworkInferenceIR(network);

        // Assert
        expect(inferenceIr.activationSteps).toEqual(expectedActivationSteps);
      });
    });

    describe('when the runtime network is feed-forward', () => {
      it('covers every non-input node exactly once in activation order', () => {
        // Arrange
        const network = createWorkerPayloadNetwork();
        const expectedActivationIndexes = network.nodes.flatMap(
          (node, nodeIndex) => {
            if (node.type === 'input') {
              return [];
            }

            return [nodeIndex];
          },
        );

        // Act
        const inferenceIr = extractNetworkInferenceIR(network);

        // Assert
        expect(inferenceIr.activationSteps.flat()).toEqual(
          expectedActivationIndexes,
        );
      });
    });

    describe('when the compiled schedule is unavailable but topology order exists', () => {
      it('falls back to singleton activation groups from the runtime traversal order', () => {
        // Arrange
        const network = createWorkerPayloadNetwork();
        Reflect.set(network, '_activationSchedule', {
          mode: 'acyclic',
          outputNodeIds: network.outputNodeIds,
          steps: [],
        } satisfies ActivationSchedule);
        Reflect.set(network, '_topoDirty', false);
        Reflect.set(network, '_topoOrder', [...network.nodes]);

        // Act
        const inferenceIr = extractNetworkInferenceIR(network);

        // Assert
        expect(inferenceIr.activationSteps).toEqual(
          network.nodes.flatMap((node) => {
            if (node.type === 'input') {
              return [];
            }

            return [[node.index]];
          }),
        );
      });
    });

    describe('when one forward edge is masked out and another edge is gated', () => {
      it('filters the masked edge and records the active gater index', () => {
        // Arrange
        const { hiddenNodes, network } = createWorkerPayloadNetworkParts();

        network.connections[0].dcMask = 0;
        network.connections.at(-1)!.gater = hiddenNodes[0];

        // Act
        const inferenceIr = extractNetworkInferenceIR(network);

        // Assert
        expect({
          edgeCount: inferenceIr.edges.length,
          gatedEdgeGaterIndex: inferenceIr.edges.find(
            (inferenceEdge) => inferenceEdge.gaterIndex !== -1,
          )?.gaterIndex,
        }).toEqual({
          edgeCount: network.connections.length - 1,
          gatedEdgeGaterIndex: hiddenNodes[0].index,
        });
      });
    });

    describe('when one forward edge is disabled', () => {
      it('excludes the disabled edge from the IR edge list', () => {
        // Arrange
        const network = createWorkerPayloadNetwork();
        network.connections[0].enabled = false;

        // Act
        const inferenceIr = extractNetworkInferenceIR(network);

        // Assert
        expect(inferenceIr.edges.length).toBe(network.connections.length - 1);
      });
    });

    describe('when a recurrent schedule omits its explicit iteration count', () => {
      it('defaults the recurrent step to one grouped activation pass', () => {
        // Arrange
        const { network, outputNode } = createWorkerPayloadNetworkParts();
        Reflect.set(network, '_activationSchedule', {
          mode: 'recurrent',
          outputNodeIds: [outputNode.geneId],
          steps: [
            {
              kind: 'recurrent-component',
              nodeIds: [outputNode.geneId],
            },
          ],
        } satisfies ActivationSchedule);
        Reflect.set(network, '_topoDirty', false);
        Reflect.set(network, '_topoOrder', [...network.nodes]);

        // Act
        const inferenceIr = extractNetworkInferenceIR(network);

        // Assert
        expect(inferenceIr.activationSteps).toEqual([[outputNode.index]]);
      });
    });

    describe('when a node has an active gated self-connection', () => {
      it('records the self weight and self gater index in the node snapshot', () => {
        // Arrange
        const { hiddenNodes, network, outputNode } =
          createWorkerPayloadNetworkParts();
        const selfConnection = new Connection(outputNode, outputNode, 0.75);

        selfConnection.gater = hiddenNodes[1];
        outputNode.connections.self = [selfConnection];
        network.selfconns = [selfConnection];

        // Act
        const inferenceIr = extractNetworkInferenceIR(network);

        // Assert
        expect(
          inferenceIr.nodes.find(
            (inferenceNode) => inferenceNode.index === outputNode.index,
          ),
        ).toMatchObject({
          selfGaterIndex: hiddenNodes[1].index,
          selfWeight: 0.75,
        });
      });
    });

    describe('when a connection gater is outside the seeded node set', () => {
      it('throws a missing seeded index error', () => {
        // Arrange
        const network = createWorkerPayloadNetwork();
        network.connections[0].gater = { geneId: 999_999 } as unknown as Node;

        // Assert
        expect(() => extractNetworkInferenceIR(network)).toThrow(
          'Expected a seeded node index',
        );
      });
    });

    describe('when a node uses an unsupported activation function', () => {
      it('throws a descriptive unsupported activation error', () => {
        // Arrange
        const { outputNode, network } = createWorkerPayloadNetworkParts();
        outputNode.squash = function customActivation(inputValue: number) {
          return inputValue;
        } as typeof outputNode.squash;

        // Assert
        expect(() => extractNetworkInferenceIR(network)).toThrow(
          "Unsupported worker inference activation 'customActivation'",
        );
      });
    });
  });
});
