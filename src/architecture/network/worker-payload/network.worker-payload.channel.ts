import { getTransferList } from './network.worker-payload.utils';
import type {
  InferenceChannel,
  InferenceChannelOptions,
  TransferableInferencePayload,
} from './network.worker-payload.types';

const CHANNEL_STRATEGY = 'channel';
const DEFAULT_MAX_CONCURRENT_REQUESTS = 8;
const CHANNEL_READY_MESSAGE_TYPE = 'ready';
const CHANNEL_PREDICT_MESSAGE_TYPE = 'predict';
const CHANNEL_PREDICT_RESULT_MESSAGE_TYPE = 'predict-result';
const CHANNEL_REQUEST_ERROR_MESSAGE_TYPE = 'request-error';
const CHANNEL_RESET_MESSAGE_TYPE = 'reset';
const CHANNEL_RESET_RESULT_MESSAGE_TYPE = 'reset-result';
const CHANNEL_CLOSE_MESSAGE_TYPE = 'close';

type ChannelPortLike = {
  addEventListener?: (type: string, listener: (event: unknown) => void) => void;
  close(): void;
  off?: (type: string, listener: (...args: unknown[]) => void) => void;
  on?: (type: string, listener: (...args: unknown[]) => void) => void;
  postMessage(message: unknown, transferList?: Transferable[]): void;
  removeEventListener?: (
    type: string,
    listener: (event: unknown) => void,
  ) => void;
  start?: () => void;
  unref?: () => void;
};

type WorkerLike = {
  addEventListener?: (type: string, listener: (event: unknown) => void) => void;
  off?: (type: string, listener: (...args: unknown[]) => void) => void;
  on?: (type: string, listener: (...args: unknown[]) => void) => void;
  postMessage(message: unknown, transferList?: Transferable[]): void;
  removeEventListener?: (
    type: string,
    listener: (event: unknown) => void,
  ) => void;
  terminate(): void | Promise<number>;
  unref?: () => void;
};

type InitializedInferenceChannelWorker = {
  localPort: ChannelPortLike;
  remotePort: Transferable;
  revokeWorkerUrl?: () => void;
  worker: WorkerLike;
};

type NodeWorkerThreadsModule = {
  MessageChannel: new () => {
    port1: unknown;
    port2: unknown;
  };
  Worker: new (
    nextWorkerSpecifier: string,
    options: {
      execArgv?: string[];
      type: 'module';
    },
  ) => unknown;
};

type PredictRequestMessage = {
  id: number;
  input: Float64Array;
  type: typeof CHANNEL_PREDICT_MESSAGE_TYPE;
};

type ResetRequestMessage = {
  id: number;
  type: typeof CHANNEL_RESET_MESSAGE_TYPE;
};

type CloseRequestMessage = {
  type: typeof CHANNEL_CLOSE_MESSAGE_TYPE;
};

type BootstrapMessage = {
  payload: TransferableInferencePayload;
  port: Transferable;
  type: 'bootstrap';
};

type ChannelResponseMessage =
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

type QueuedPredictRequest = {
  id: number;
  message: PredictRequestMessage;
  reject: (reason: Error) => void;
  resolve: (value: Float64Array) => void;
  transferList: Transferable[];
  type: 'predict';
};

type QueuedResetRequest = {
  id: number;
  message: ResetRequestMessage;
  reject: (reason: Error) => void;
  resolve: () => void;
  transferList: Transferable[];
  type: 'reset';
};

type QueuedChannelRequest = QueuedPredictRequest | QueuedResetRequest;

type PendingPredictRequest = {
  reject: (reason: Error) => void;
  resolve: (value: Float64Array) => void;
  type: 'predict';
};

type PendingResetRequest = {
  reject: (reason: Error) => void;
  resolve: () => void;
  type: 'reset';
};

type PendingChannelRequest = PendingPredictRequest | PendingResetRequest;

type FatalChannelTransitionOptions = {
  bootstrapReject?: (reason?: unknown) => void;
  clearBootstrapState: () => void;
  isOpen: boolean;
  shutdownChannel: (shutdownError: Error) => Promise<void>;
};

/**
 * Open one persistent inference worker channel from a transferable payload.
 *
 * The channel transfers the predictor payload exactly once during bootstrap,
 * then reuses the worker-side predictor state for every later `predict()` or
 * `reset()` request sent over one dedicated `MessageChannel` port pair.
 *
 * @param payload - Transferable inference payload used to bootstrap the worker predictor.
 * @param options - Channel concurrency and worker delivery options.
 * @returns Persistent inference channel.
 * @example
 * ```ts
 * const payload = exportTransferableInferencePayload(network);
 * const channel = openInferenceChannel(payload);
 * const outputValues = await channel.predict([0.25, 0.75]);
 * channel.close();
 * ```
 */
export function openInferenceChannel(
  payload: TransferableInferencePayload,
  options: InferenceChannelOptions = {},
): InferenceChannel {
  const maxConcurrentRequests = Math.max(
    1,
    Math.trunc(
      options.maxConcurrentRequests ?? DEFAULT_MAX_CONCURRENT_REQUESTS,
    ),
  );
  const pendingRequests = new Map<number, PendingChannelRequest>();
  const queuedRequests: QueuedChannelRequest[] = [];
  let activeRequestCount = 0;
  let bootstrapReject: ((reason?: unknown) => void) | undefined;
  let bootstrapResolve: (() => void) | undefined;
  let channelPort: ChannelPortLike | undefined;
  let closePromise: Promise<void> | undefined;
  let dispatchLoopActive = false;
  let isOpen = true;
  let nextRequestId = 1;
  let revokeWorkerUrl: (() => void) | undefined;
  let workerHandle: WorkerLike | undefined;

  const bootstrapReadyPromise = new Promise<void>((resolve, reject) => {
    bootstrapResolve = resolve;
    bootstrapReject = reject;
  });

  const bootstrapPromise = createBootstrapPromise();
  void bootstrapPromise.catch((error) => {
    handleFatalChannelError(asError(error));
  });

  return {
    async close(): Promise<void> {
      if (closePromise) {
        await closePromise;
        return;
      }

      closePromise = shutdownChannel(createClosedChannelError());
      await closePromise;
    },
    predict(
      input: Float64Array | ReadonlyArray<number>,
    ): Promise<Float64Array> {
      if (!isOpen) {
        return Promise.reject(createClosedChannelError());
      }

      const requestId = nextRequestId;
      nextRequestId += 1;

      return new Promise<Float64Array>((resolve, reject) => {
        const transferredInput = Float64Array.from(input);
        queuedRequests.push({
          id: requestId,
          message: {
            id: requestId,
            input: transferredInput,
            type: CHANNEL_PREDICT_MESSAGE_TYPE,
          },
          reject,
          resolve,
          transferList: [transferredInput.buffer],
          type: 'predict',
        });
        void drainQueuedRequests();
      });
    },
    reset(): Promise<void> {
      if (!isOpen) {
        return Promise.reject(createClosedChannelError());
      }

      const requestId = nextRequestId;
      nextRequestId += 1;

      return new Promise<void>((resolve, reject) => {
        queuedRequests.push({
          id: requestId,
          message: {
            id: requestId,
            type: CHANNEL_RESET_MESSAGE_TYPE,
          },
          reject,
          resolve,
          transferList: [],
          type: 'reset',
        });
        void drainQueuedRequests();
      });
    },
    strategy: CHANNEL_STRATEGY,
    get isOpen(): boolean {
      return isOpen;
    },
  };

  function handleChannelResponse(message: unknown): void {
    const responseMessage = message as Partial<ChannelResponseMessage>;

    if (responseMessage.type === CHANNEL_READY_MESSAGE_TYPE) {
      bootstrapResolve?.();
      bootstrapResolve = undefined;
      bootstrapReject = undefined;
      return;
    }

    if (
      responseMessage.type === CHANNEL_REQUEST_ERROR_MESSAGE_TYPE &&
      typeof responseMessage.id === 'number'
    ) {
      const pendingRequest = pendingRequests.get(responseMessage.id);

      if (!pendingRequest) {
        return;
      }

      pendingRequests.delete(responseMessage.id);
      activeRequestCount = Math.max(0, activeRequestCount - 1);
      pendingRequest.reject(
        new Error(
          responseMessage.message ?? 'InferenceChannel worker request failed.',
        ),
      );
      void drainQueuedRequests();
      return;
    }

    if (
      responseMessage.type === CHANNEL_REQUEST_ERROR_MESSAGE_TYPE &&
      typeof responseMessage.id !== 'number'
    ) {
      handleFatalChannelError(
        new Error(
          responseMessage.message ?? 'InferenceChannel bootstrap failed.',
        ),
      );
      return;
    }

    if (
      responseMessage.type === CHANNEL_PREDICT_RESULT_MESSAGE_TYPE &&
      typeof responseMessage.id === 'number'
    ) {
      const pendingRequest = pendingRequests.get(responseMessage.id);

      if (!pendingRequest) {
        return;
      }

      pendingRequests.delete(responseMessage.id);
      activeRequestCount = Math.max(0, activeRequestCount - 1);
      if (pendingRequest.type === 'predict') {
        pendingRequest.resolve(responseMessage.output ?? new Float64Array());
      }
      void drainQueuedRequests();
      return;
    }

    if (
      responseMessage.type === CHANNEL_RESET_RESULT_MESSAGE_TYPE &&
      typeof responseMessage.id === 'number'
    ) {
      const pendingRequest = pendingRequests.get(responseMessage.id);

      if (!pendingRequest) {
        return;
      }

      pendingRequests.delete(responseMessage.id);
      activeRequestCount = Math.max(0, activeRequestCount - 1);
      if (pendingRequest.type === 'reset') {
        pendingRequest.resolve();
      }
      void drainQueuedRequests();
    }
  }

  function createBootstrapPromise(): Promise<void> {
    const initializedWorker = initializeInferenceChannelWorker(options);

    if (isPromiseLike(initializedWorker)) {
      return initializedWorker.then((resolvedWorker) => {
        return startWorkerBootstrap(resolvedWorker);
      });
    }

    return startWorkerBootstrap(initializedWorker);
  }

  function startWorkerBootstrap(
    initializedWorker: InitializedInferenceChannelWorker,
  ): Promise<void> {
    workerHandle = initializedWorker.worker;
    channelPort = initializedWorker.localPort;
    revokeWorkerUrl = initializedWorker.revokeWorkerUrl;

    // Allow Node-owned worker resources to stop keeping the Jest host alive once
    // the channel has no remaining work.
    workerHandle.unref?.();
    channelPort.unref?.();
    (initializedWorker.remotePort as { unref?: () => void }).unref?.();

    attachPortMessageListener(
      channelPort,
      (message) => {
        handleChannelResponse(message);
      },
      (error) => {
        handleFatalChannelError(error);
      },
    );
    attachWorkerLifecycleListeners(initializedWorker.worker, (error) => {
      handleFatalChannelError(error);
    });

    postMessageToWorker(
      initializedWorker.worker,
      {
        payload,
        port: initializedWorker.remotePort,
        type: 'bootstrap',
      } satisfies BootstrapMessage,
      [initializedWorker.remotePort, ...getTransferList(payload)],
    );

    return bootstrapReadyPromise;
  }

  function handleFatalChannelError(error: Error): void {
    const nextClosePromise = resolveFatalChannelClosePromise(
      {
        bootstrapReject,
        clearBootstrapState: () => {
          bootstrapResolve = undefined;
          bootstrapReject = undefined;
        },
        isOpen,
        shutdownChannel,
      },
      error,
    );

    if (!nextClosePromise) {
      return;
    }

    closePromise = nextClosePromise;
    void closePromise;
  }

  async function drainQueuedRequests(): Promise<void> {
    if (dispatchLoopActive) {
      return;
    }

    dispatchLoopActive = true;

    try {
      await bootstrapPromise;

      while (
        isOpen &&
        channelPort &&
        activeRequestCount < maxConcurrentRequests &&
        queuedRequests.length > 0
      ) {
        const queuedRequest = queuedRequests.shift()!;

        pendingRequests.set(queuedRequest.id, queuedRequest);
        activeRequestCount += 1;
        postMessageToPort(
          channelPort,
          queuedRequest.message,
          queuedRequest.transferList,
        );
      }
    } catch (error) {
      handleFatalChannelError(asError(error));
    } finally {
      dispatchLoopActive = false;
    }
  }

  async function shutdownChannel(shutdownError: Error): Promise<void> {
    isOpen = false;
    rejectQueuedRequests(shutdownError);
    rejectPendingRequests(shutdownError);

    try {
      if (channelPort) {
        postMessageToPort(
          channelPort,
          {
            type: CHANNEL_CLOSE_MESSAGE_TYPE,
          },
          [],
        );
      }

      await new Promise<void>((resolve) => {
        setImmediate(resolve);
      });
    } catch {
      // Ignore shutdown post races because local teardown continues below.
    }

    if (channelPort) {
      await closeChannelPort(channelPort);
    }

    await disposeWorkerHandle(workerHandle);

    revokeWorkerUrl?.();
    channelPort = undefined;
    workerHandle = undefined;
    revokeWorkerUrl = undefined;
  }

  function rejectQueuedRequests(error: Error): void {
    while (queuedRequests.length > 0) {
      const queuedRequest = queuedRequests.shift();

      queuedRequest?.reject(error);
    }
  }

  function rejectPendingRequests(error: Error): void {
    pendingRequests.forEach((pendingRequest) => {
      pendingRequest.reject(error);
    });
    pendingRequests.clear();
    activeRequestCount = 0;
  }
}

function initializeInferenceChannelWorker(
  options: InferenceChannelOptions,
):
  | InitializedInferenceChannelWorker
  | Promise<InitializedInferenceChannelWorker> {
  const workerSpecifier = resolveInferenceChannelWorkerSpecifier(
    options.workerUrl,
  );

  if (typeof globalThis.Worker === 'function') {
    return createBrowserInferenceChannelWorker(
      workerSpecifier,
      options.workerUrl,
    );
  }

  return createNodeInferenceChannelWorker(workerSpecifier);
}

function resolveInferenceChannelWorkerSpecifier(
  workerUrl: string | undefined,
): string {
  if (workerUrl) {
    return workerUrl;
  }

  if (typeof process !== 'undefined' && typeof process.cwd === 'function') {
    return resolveNodeDefaultWorkerPath();
  }

  throw new Error(
    'InferenceChannel browser bootstrap requires workerUrl until inline delivery is configured.',
  );
}

async function closeChannelPort(channelPort: ChannelPortLike): Promise<void> {
  if (typeof channelPort.on === 'function') {
    await new Promise<void>((resolve) => {
      const handleClose = () => {
        channelPort.off?.('close', handleClose);
        resolve();
      };

      channelPort.on?.('close', handleClose);

      try {
        channelPort.close();
      } catch {
        channelPort.off?.('close', handleClose);
        resolve();
      }
    });

    return;
  }

  try {
    channelPort.close();
  } catch {
    // Ignore repeated or environment-specific close failures during teardown.
  }
}

async function disposeWorkerHandle(
  workerHandle: WorkerLike | undefined,
): Promise<void> {
  if (!workerHandle) {
    return;
  }

  const asyncDispose = (
    workerHandle as WorkerLike & {
      [Symbol.asyncDispose]?: () => Promise<void>;
    }
  )[Symbol.asyncDispose];

  if (typeof asyncDispose === 'function') {
    try {
      await asyncDispose.call(workerHandle);
      return;
    } catch {
      // Ignore disposal races during close because the channel is already closed.
    }
  }

  const workerTermination = workerHandle.terminate();
  if (workerTermination instanceof Promise) {
    try {
      await workerTermination;
    } catch {
      // Ignore termination races during close because the channel is already closed.
    }
  }
}

function createBrowserInferenceChannelWorker(
  workerSpecifier: string,
  workerUrl: string | undefined,
): InitializedInferenceChannelWorker {
  const messageChannel = new MessageChannel();

  if (!workerUrl) {
    throw new Error(
      'InferenceChannel browser bootstrap requires workerUrl until inline delivery is configured.',
    );
  }

  return {
    localPort: messageChannel.port1 as unknown as ChannelPortLike,
    remotePort: messageChannel.port2,
    worker: new Worker(workerSpecifier, {
      type: 'module',
    }) as unknown as WorkerLike,
  };
}

async function createNodeInferenceChannelWorker(
  workerSpecifier: string,
): Promise<InitializedInferenceChannelWorker> {
  const workerThreadsModule = await loadNodeWorkerThreadsModule();
  const messageChannel = new workerThreadsModule.MessageChannel();
  const worker = new workerThreadsModule.Worker(
    workerSpecifier,
    resolveNodeInferenceChannelWorkerOptions(workerSpecifier),
  );

  return {
    localPort: messageChannel.port1 as unknown as ChannelPortLike,
    remotePort: messageChannel.port2 as unknown as Transferable,
    worker: worker as unknown as WorkerLike,
  };
}

async function loadNodeWorkerThreadsModule(
  builtinModuleResolver: () =>
    | NodeWorkerThreadsModule
    | undefined = resolveNodeWorkerThreadsModule,
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
    | NodeWorkerThreadsModule
    | undefined;
}

function resolveNodeInferenceChannelWorkerOptions(workerSpecifier: string): {
  execArgv?: string[];
  type: 'module';
} {
  if (workerSpecifier.endsWith('.ts')) {
    return {
      execArgv: [
        '--loader',
        'ts-node/esm',
        '--experimental-specifier-resolution=node',
      ],
      type: 'module',
    };
  }

  return {
    type: 'module',
  };
}

function attachPortMessageListener(
  channelPort: ChannelPortLike,
  onMessage: (message: unknown) => void,
  onMessageError: (error: Error) => void,
): void {
  if (typeof channelPort.addEventListener === 'function') {
    channelPort.addEventListener('message', (event) => {
      onMessage(resolveMessageEventData(event));
    });
    channelPort.addEventListener('messageerror', () => {
      onMessageError(
        new Error('InferenceChannel message port rejected one message.'),
      );
    });
    channelPort.start?.();
    return;
  }

  channelPort.on?.('message', onMessage);
  channelPort.on?.('messageerror', () => {
    onMessageError(
      new Error('InferenceChannel message port rejected one message.'),
    );
  });
}

function attachWorkerLifecycleListeners(
  worker: WorkerLike,
  onFailure: (error: Error) => void,
): void {
  if (typeof worker.addEventListener === 'function') {
    worker.addEventListener('error', (event) => {
      onFailure(resolveWorkerError(event));
    });
    worker.addEventListener('messageerror', () => {
      onFailure(new Error('InferenceChannel worker rejected one message.'));
    });
    return;
  }

  worker.on?.('error', (error) => {
    onFailure(asError(error));
  });
  worker.on?.('messageerror', () => {
    onFailure(new Error('InferenceChannel worker rejected one message.'));
  });
  worker.on?.('exit', (exitCode) => {
    onFailure(
      new Error(
        `InferenceChannel worker exited with code ${String(exitCode)}.`,
      ),
    );
  });
}

function postMessageToWorker(
  worker: WorkerLike,
  message: BootstrapMessage,
  transferList: Transferable[],
): void {
  worker.postMessage(message, transferList);
}

function postMessageToPort(
  channelPort: ChannelPortLike,
  message: PredictRequestMessage | ResetRequestMessage | CloseRequestMessage,
  transferList: Transferable[],
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

function resolveWorkerError(event: unknown): Error {
  const workerErrorEvent = event as { error?: unknown; message?: string };

  if (workerErrorEvent.error instanceof Error) {
    return workerErrorEvent.error;
  }

  return new Error(
    workerErrorEvent.message ?? 'InferenceChannel worker failed.',
  );
}

function createClosedChannelError(): Error {
  return new Error('InferenceChannel is closed.');
}

function asError(error: unknown): Error {
  if (error instanceof Error) {
    return error;
  }

  if (typeof error === 'object' && error !== null) {
    const errorRecord = error as {
      message?: unknown;
      name?: unknown;
      stack?: unknown;
    };

    if (typeof errorRecord.message === 'string') {
      const normalizedError = new Error(errorRecord.message);

      if (typeof errorRecord.name === 'string') {
        normalizedError.name = errorRecord.name;
      }

      if (typeof errorRecord.stack === 'string') {
        normalizedError.stack = errorRecord.stack;
      }

      return normalizedError;
    }
  }

  try {
    return new Error(String(error));
  } catch {
    return new Error('InferenceChannel worker failed with an unknown error.');
  }
}

function resolveNodeDefaultWorkerPath(): string {
  const normalizedWorkingDirectory = process.cwd().replace(/\\/g, '/');

  if (typeof process.env.JEST_WORKER_ID === 'string') {
    return `${normalizedWorkingDirectory}/src/architecture/network/worker-payload/network.worker-payload.channel.worker.ts`;
  }

  return `${normalizedWorkingDirectory}/dist/architecture/network/worker-payload/network.worker-payload.channel.worker.js`;
}

function resolveFatalChannelClosePromise(
  options: FatalChannelTransitionOptions,
  error: Error,
): Promise<void> | undefined {
  if (!options.isOpen) {
    return undefined;
  }

  options.bootstrapReject?.(error);
  options.clearBootstrapState();
  return options.shutdownChannel(error);
}

function isPromiseLike<T>(value: T | Promise<T>): value is Promise<T> {
  return value instanceof Promise;
}

/** @internal Test-only helper surface for owner-local channel coverage. */
export const INFERENCE_CHANNEL_HOST_INTERNALS = {
  asError,
  attachPortMessageListener,
  attachWorkerLifecycleListeners,
  closeChannelPort,
  createBrowserInferenceChannelWorker,
  createClosedChannelError,
  disposeWorkerHandle,
  loadNodeWorkerThreadsModule,
  resolveFatalChannelClosePromise,
  resolveInferenceChannelWorkerSpecifier,
  resolveMessageEventData,
  resolveNodeDefaultWorkerPath,
  resolveNodeInferenceChannelWorkerOptions,
  resolveWorkerError,
};
