import { createInferencePredictor } from './network.worker-payload.utils';
import type { TransferableInferencePayload } from './network.worker-payload.types';

const CHANNEL_READY_MESSAGE_TYPE = 'ready';
const CHANNEL_PREDICT_MESSAGE_TYPE = 'predict';
const CHANNEL_PREDICT_RESULT_MESSAGE_TYPE = 'predict-result';
const CHANNEL_REQUEST_ERROR_MESSAGE_TYPE = 'request-error';
const CHANNEL_RESET_MESSAGE_TYPE = 'reset';
const CHANNEL_RESET_RESULT_MESSAGE_TYPE = 'reset-result';

type ChannelPortLike = {
  addEventListener?: (type: string, listener: (event: unknown) => void) => void;
  close(): void;
  on?: (type: string, listener: (...args: unknown[]) => void) => void;
  postMessage(message: unknown, transferList?: Transferable[]): void;
  start?: () => void;
};

type BootstrapMessage = {
  payload: TransferableInferencePayload;
  port: unknown;
  type: 'bootstrap';
};

type NodeParentPortLike = {
  close?: () => void;
  once?: (type: string, listener: (message: unknown) => void) => void;
};

type NodeWorkerThreadsModule = {
  parentPort?: NodeParentPortLike;
};

type ChannelPortRequest =
  | {
      id: number;
      input: Float64Array | ArrayLike<number>;
      type: typeof CHANNEL_PREDICT_MESSAGE_TYPE;
    }
  | { id: number; type: typeof CHANNEL_RESET_MESSAGE_TYPE }
  | { type: 'close' };

type ChannelPortResponse =
  | { type: typeof CHANNEL_READY_MESSAGE_TYPE }
  | {
      id: number;
      output: Float64Array;
      type: typeof CHANNEL_PREDICT_RESULT_MESSAGE_TYPE;
    }
  | { id: number; type: typeof CHANNEL_RESET_RESULT_MESSAGE_TYPE }
  | {
      id?: number;
      message: string;
      type: typeof CHANNEL_REQUEST_ERROR_MESSAGE_TYPE;
    };

/**
 * Register the worker-side dedicated inference-channel runtime.
 *
 * The worker receives one transferable predictor payload during bootstrap,
 * rebuilds the local predictor once, then services predict and reset requests
 * over the transferred `MessagePort` instead of the default worker channel.
 *
 * @returns Nothing.
 */
export function registerInferenceChannelWorkerRuntime(): void {
  if (
    typeof process !== 'undefined' &&
    typeof process.versions?.node === 'string'
  ) {
    void loadNodeWorkerThreadsModule().then((workerThreadsModule) => {
      registerNodeInferenceChannelWorkerRuntime(
        workerThreadsModule.parentPort as NodeParentPortLike | undefined,
        exitCurrentNodeInferenceChannelWorkerRuntime,
      );
    });
    return;
  }

  registerBrowserInferenceChannelWorkerRuntime((listener) => {
    globalThis.addEventListener('message', listener, { once: true });
  }, closeCurrentBrowserInferenceChannelWorkerRuntime);
}

function autoRegisterInferenceChannelWorkerRuntime(): void {
  if (shouldAutoRegisterInferenceChannelWorkerRuntime()) {
    registerInferenceChannelWorkerRuntime();
  }
}

autoRegisterInferenceChannelWorkerRuntime();

function shouldAutoRegisterInferenceChannelWorkerRuntime(): boolean {
  if (
    typeof process !== 'undefined' &&
    typeof process.versions?.node === 'string'
  ) {
    return Boolean(resolveNodeWorkerThreadsModule()?.parentPort);
  }

  const browserGlobalScope = globalThis as typeof globalThis & {
    document?: unknown;
  };

  return (
    typeof globalThis.addEventListener === 'function' &&
    typeof browserGlobalScope.document === 'undefined'
  );
}

async function loadNodeWorkerThreadsModule(
  builtinModuleResolver: () =>
    NodeWorkerThreadsModule | undefined = resolveNodeWorkerThreadsModule,
  importFunction: (moduleSpecifier: string) => Promise<unknown> = Function(
    'moduleSpecifier',
    'return import(moduleSpecifier);',
  ) as (moduleSpecifier: string) => Promise<unknown>,
): Promise<NodeWorkerThreadsModule> {
  const builtinModule = builtinModuleResolver();

  if (builtinModule) {
    return builtinModule;
  }

  return (await importFunction('worker_threads')) as NodeWorkerThreadsModule;
}

function resolveNodeWorkerThreadsModule(): NodeWorkerThreadsModule | undefined {
  const nodeProcess = process as NodeJS.Process & {
    getBuiltinModule?: (moduleSpecifier: string) => unknown;
  };

  return nodeProcess.getBuiltinModule?.('worker_threads') as
    NodeWorkerThreadsModule | undefined;
}

function handleBootstrapMessage(
  message: unknown,
  closeWorker: () => void,
): void {
  const bootstrapMessage = message as Partial<BootstrapMessage>;

  if (bootstrapMessage.type !== 'bootstrap') {
    return;
  }

  const channelPort = bootstrapMessage.port as ChannelPortLike | undefined;

  if (!channelPort || !bootstrapMessage.payload) {
    throw new Error(
      'InferenceChannel worker expected bootstrap payload and port.',
    );
  }

  const predictor = createInferencePredictor(bootstrapMessage.payload);

  attachPortListener(channelPort, (requestMessage) => {
    handleChannelPortRequest(
      channelPort,
      requestMessage,
      predictor,
      closeWorker,
    );
  });
  postPortMessage(channelPort, { type: CHANNEL_READY_MESSAGE_TYPE });
}

function registerNodeInferenceChannelWorkerRuntime(
  parentPort: NodeParentPortLike | undefined,
  exitProcess: () => void,
): void {
  parentPort?.once?.('message', (message: unknown) => {
    handleBootstrapMessage(message, () => {
      parentPort.close?.();
      exitProcess();
    });
  });
}

function registerBrowserInferenceChannelWorkerRuntime(
  addMessageListener: (listener: (event: unknown) => void) => void,
  closeWorker: () => void,
): void {
  addMessageListener((event) => {
    handleBootstrapMessage(resolveMessageEventData(event), () => {
      closeWorker();
    });
  });
}

function exitCurrentNodeInferenceChannelWorkerRuntime(): void {
  process.exit(0);
}

function closeCurrentBrowserInferenceChannelWorkerRuntime(): void {
  globalThis.close();
}

function handleChannelPortRequest(
  channelPort: ChannelPortLike,
  requestMessage: unknown,
  predictor: ReturnType<typeof createInferencePredictor>,
  closeWorker: () => void,
): void {
  const typedRequest = requestMessage as Partial<ChannelPortRequest>;
  const requestId =
    'id' in typedRequest && typeof typedRequest.id === 'number'
      ? typedRequest.id
      : undefined;

  try {
    if (typedRequest.type === CHANNEL_PREDICT_MESSAGE_TYPE) {
      const outputValues = Float64Array.from(
        predictor.predict(Array.from(typedRequest.input ?? [])),
      );
      postPortMessage(
        channelPort,
        {
          id: requestId ?? -1,
          output: outputValues,
          type: CHANNEL_PREDICT_RESULT_MESSAGE_TYPE,
        },
        [outputValues.buffer],
      );
      return;
    }

    if (typedRequest.type === CHANNEL_RESET_MESSAGE_TYPE) {
      predictor.reset();
      postPortMessage(channelPort, {
        id: requestId ?? -1,
        type: CHANNEL_RESET_RESULT_MESSAGE_TYPE,
      });
      return;
    }

    if (typedRequest.type === 'close') {
      channelPort.close();
      closeWorker();
    }
  } catch (error) {
    postPortMessage(channelPort, {
      id: requestId,
      message: asError(error).message,
      type: CHANNEL_REQUEST_ERROR_MESSAGE_TYPE,
    });
  }
}

function attachPortListener(
  channelPort: ChannelPortLike,
  onMessage: (message: unknown) => void,
): void {
  if (typeof channelPort.addEventListener === 'function') {
    channelPort.addEventListener('message', (event) => {
      onMessage(resolveMessageEventData(event));
    });
    channelPort.start?.();
    return;
  }

  channelPort.on?.('message', onMessage);
}

function postPortMessage(
  channelPort: ChannelPortLike,
  message: ChannelPortResponse,
  transferList: Transferable[] = [],
): void {
  channelPort.postMessage(message, transferList);
}

function resolveMessageEventData(event: unknown): unknown {
  const dataCarrier = event as { data?: unknown };

  if (typeof event === 'object' && event !== null && 'data' in dataCarrier) {
    return dataCarrier.data;
  }

  return event;
}

function asError(error: unknown): Error {
  if (error instanceof Error) {
    return error;
  }

  return new Error(String(error));
}

/**
 * Test-only internal helper surface exposing private functions for owner-local coverage
 * of the dedicated inference-channel worker runtime.
 *
 * Not part of the public API. Exported only so unit tests can exercise runtime branches
 * unreachable through the public {@link registerInferenceChannelWorkerRuntime} entry point.
 *
 * @internal
 */
export const INFERENCE_CHANNEL_WORKER_INTERNALS = {
  autoRegisterInferenceChannelWorkerRuntime,
  asError,
  attachPortListener,
  closeCurrentBrowserInferenceChannelWorkerRuntime,
  exitCurrentNodeInferenceChannelWorkerRuntime,
  handleBootstrapMessage,
  handleChannelPortRequest,
  loadNodeWorkerThreadsModule,
  postPortMessage,
  registerBrowserInferenceChannelWorkerRuntime,
  registerNodeInferenceChannelWorkerRuntime,
  resolveMessageEventData,
  shouldAutoRegisterInferenceChannelWorkerRuntime,
};
