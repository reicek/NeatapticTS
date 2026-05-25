import { resolveBrowserWorkerAssetUrl } from './network.worker-payload.browser-url';

const SHARED_INFERENCE_STATUS_INDEX = 0;
const SHARED_INFERENCE_INPUT_COUNT_INDEX = 1;
const SHARED_INFERENCE_CONTROL_ELEMENT_COUNT = 2;
const SHARED_INFERENCE_IDLE_STATUS = 0;
const SHARED_INFERENCE_INPUT_READY_STATUS = 1;
const SHARED_INFERENCE_OUTPUT_READY_STATUS = 2;
const SHARED_INFERENCE_RESET_REQUESTED_STATUS = 3;
const SHARED_INFERENCE_READY_MESSAGE_TYPE = 'ready';
const SHARED_INFERENCE_REQUEST_ERROR_MESSAGE_TYPE = 'request-error';
const SHARED_INFERENCE_ASYNC_CONVERSION_THRESHOLD = 32_768;
const SHARED_INFERENCE_ASYNC_CONVERSION_CHUNK_SIZE = 16_384;

import { getTransferList } from './network.worker-payload.utils';
import type {
  SharedInferenceWorker,
  SharedInferenceWorkerOptions,
  TransferableInferencePayload,
} from './network.worker-payload.types';

type WorkerLike = {
  addEventListener?: (type: string, listener: (event: unknown) => void) => void;
  on?: (type: string, listener: (...args: unknown[]) => void) => void;
  postMessage(message: unknown, transferList?: Transferable[]): void;
  terminate(): void | Promise<number>;
  unref?: () => void;
};

type InitializedSharedInferenceWorker = {
  revokeWorkerUrl?: () => void;
  worker: WorkerLike;
};

type NodeWorkerThreadsModule = {
  Worker: new (
    nextWorkerSpecifier: string,
    options: {
      execArgv?: string[];
      type: 'module';
    },
  ) => unknown;
};

type SharedWorkerBootstrapMessage = {
  controlBuffer: SharedArrayBuffer;
  dataBuffer: SharedArrayBuffer;
  payload: TransferableInferencePayload;
  type: 'bootstrap';
};

type SharedWorkerResponseMessage =
  | { type: typeof SHARED_INFERENCE_READY_MESSAGE_TYPE }
  | {
      message: string;
      type: typeof SHARED_INFERENCE_REQUEST_ERROR_MESSAGE_TYPE;
    };

type SharedInferenceBufferLayout = ReturnType<
  typeof resolveSharedInferenceBufferLayout
>;

type SharedInferenceWorkerState = {
  bootstrapReject?: (reason?: unknown) => void;
  bootstrapResolve?: () => void;
  closePromise?: Promise<void>;
  fatalError?: Error;
  isBusy: boolean;
  isOpen: boolean;
  isReady: boolean;
  pendingInputCopyPromise?: Promise<void>;
  revokeWorkerUrl?: () => void;
  workerHandle?: WorkerLike;
};

type SharedInferenceWorkerRuntime = {
  controlBuffer: SharedArrayBuffer;
  controlView: Int32Array;
  dataBuffer: SharedArrayBuffer;
  dataView: Float64Array;
  layout: SharedInferenceBufferLayout;
  payload: TransferableInferencePayload;
  shouldUseChunkedConversions: boolean;
  state: SharedInferenceWorkerState;
};

/**
 * Browser `SharedArrayBuffer` inference requires cross-origin isolation.
 *
 * Node.js workers can use the shared-memory path without `COOP` or `COEP`, but
 * browsers only expose `SharedArrayBuffer` reliably when the host page is
 * cross-origin isolated.
 *
 * @example
 * ```ts
 * if (SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION) {
 *   console.log('Configure COOP/COEP before enabling shared-memory inference.');
 * }
 * ```
 */
export const SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION = true as const;

/**
 * Open one persistent shared-memory inference worker from a transferable payload.
 *
 * Shared-memory workers keep one predictor alive inside a dedicated worker and
 * exchange inputs and outputs through `SharedArrayBuffer` shelves rather than
 * cloning request payloads for every inference call.
 *
 * Browser hosts default to a module-relative worker script emitted alongside
 * the library files. Bundled or CSP-constrained hosts can override that entry
 * with `workerUrl`.
 *
 * @param payload - Transferable inference payload used to bootstrap the shared worker predictor.
 * @param options - Shared worker delivery options.
 * @returns Persistent shared-memory inference worker.
 * @example
 * ```ts
 * const payload = exportTransferableInferencePayload(network);
 * const sharedWorker = openSharedInferenceWorker(payload);
 * const outputValues = await sharedWorker.infer([0.25, 0.75]);
 * await sharedWorker.release();
 * ```
 */
export function openSharedInferenceWorker(
  payload: TransferableInferencePayload,
  options: SharedInferenceWorkerOptions = {},
): SharedInferenceWorker {
  ensureSharedArrayBufferSupport();
  const runtime = createSharedInferenceWorkerRuntime(payload);
  const bootstrapPromise = createSharedInferenceBootstrapPromise(
    runtime,
    options,
  );
  void bootstrapPromise.catch((error) => {
    handleFatalSharedWorkerError(runtime, asError(error));
  });

  return {
    awaitOutput: () => awaitSharedInferenceOutput(runtime, bootstrapPromise),
    infer: (input) =>
      inferWithSharedInferenceWorker(runtime, bootstrapPromise, input),
    get isReady(): boolean {
      return runtime.state.isOpen && runtime.state.isReady;
    },
    release: () => releaseSharedInferenceWorker(runtime),
    reset: () => resetSharedInferenceWorker(runtime, bootstrapPromise),
    strategy: 'shared-memory',
    submitInput: (input) => submitSharedInferenceInput(runtime, input),
  };
}

function createSharedInferenceWorkerRuntime(
  payload: TransferableInferencePayload,
): SharedInferenceWorkerRuntime {
  const layout = resolveSharedInferenceBufferLayout(
    payload.inputCount,
    payload.outputCount,
  );
  const controlBuffer = new SharedArrayBuffer(
    Int32Array.BYTES_PER_ELEMENT * layout.controlElementCount,
  );
  const dataBuffer = new SharedArrayBuffer(
    Float64Array.BYTES_PER_ELEMENT * layout.dataElementCount,
  );
  const controlView = new Int32Array(controlBuffer);
  const state: SharedInferenceWorkerState = {
    isBusy: false,
    isOpen: true,
    isReady: false,
  };

  const runtime = {
    controlBuffer,
    controlView,
    dataBuffer,
    dataView: new Float64Array(dataBuffer),
    layout,
    payload,
    shouldUseChunkedConversions: typeof globalThis.Worker === 'function',
    state,
  };

  Atomics.store(
    runtime.controlView,
    runtime.layout.statusIndexes.inputCount,
    runtime.layout.inputCount,
  );
  Atomics.store(
    runtime.controlView,
    runtime.layout.statusIndexes.status,
    runtime.layout.statusValues.idle,
  );

  return runtime;
}

function createSharedInferenceBootstrapPromise(
  runtime: SharedInferenceWorkerRuntime,
  options: SharedInferenceWorkerOptions,
): Promise<void> {
  const initializedWorker = initializeSharedInferenceWorker(options);
  const bootstrapReadyPromise = new Promise<void>((resolve, reject) => {
    runtime.state.bootstrapResolve = resolve;
    runtime.state.bootstrapReject = reject;
  });

  if (isPromiseLike(initializedWorker)) {
    return initializedWorker.then((resolvedWorker) => {
      return startSharedInferenceWorkerBootstrap(
        runtime,
        resolvedWorker,
        bootstrapReadyPromise,
      );
    });
  }

  return startSharedInferenceWorkerBootstrap(
    runtime,
    initializedWorker,
    bootstrapReadyPromise,
  );
}

function startSharedInferenceWorkerBootstrap(
  runtime: SharedInferenceWorkerRuntime,
  initializedWorker: InitializedSharedInferenceWorker,
  bootstrapReadyPromise: Promise<void>,
): Promise<void> {
  runtime.state.workerHandle = initializedWorker.worker;
  runtime.state.revokeWorkerUrl = initializedWorker.revokeWorkerUrl;

  runtime.state.workerHandle.unref?.();

  attachSharedWorkerMessageListener(
    initializedWorker.worker,
    (message) => {
      handleSharedWorkerResponse(runtime, message);
    },
    (error) => {
      handleFatalSharedWorkerError(runtime, error);
    },
  );
  attachSharedWorkerLifecycleListeners(initializedWorker.worker, (error) => {
    handleFatalSharedWorkerError(runtime, error);
  });

  postMessageToSharedWorker(
    initializedWorker.worker,
    {
      controlBuffer: runtime.controlBuffer,
      dataBuffer: runtime.dataBuffer,
      payload: runtime.payload,
      type: 'bootstrap',
    } satisfies SharedWorkerBootstrapMessage,
    getTransferList(runtime.payload),
  );

  return bootstrapReadyPromise;
}

function handleSharedWorkerResponse(
  runtime: SharedInferenceWorkerRuntime,
  message: unknown,
): void {
  const responseMessage = message as Partial<SharedWorkerResponseMessage>;

  if (responseMessage.type === SHARED_INFERENCE_READY_MESSAGE_TYPE) {
    runtime.state.isReady = true;
    runtime.state.bootstrapResolve?.();
    clearSharedInferenceBootstrapState(runtime.state);
    return;
  }

  if (responseMessage.type === SHARED_INFERENCE_REQUEST_ERROR_MESSAGE_TYPE) {
    handleFatalSharedWorkerError(
      runtime,
      new Error(
        responseMessage.message ?? 'SharedInferenceWorker bootstrap failed.',
      ),
    );
  }
}

function clearSharedInferenceBootstrapState(
  state: SharedInferenceWorkerState,
): void {
  state.bootstrapResolve = undefined;
  state.bootstrapReject = undefined;
}

function handleFatalSharedWorkerError(
  runtime: SharedInferenceWorkerRuntime,
  error: Error,
): void {
  runtime.state.fatalError = error;

  if (!runtime.state.isOpen) {
    return;
  }

  runtime.state.bootstrapReject?.(error);
  clearSharedInferenceBootstrapState(runtime.state);
  runtime.state.closePromise ??= shutdownSharedInferenceWorker(runtime);
  void runtime.state.closePromise;
}

async function awaitSharedInferenceOutput(
  runtime: SharedInferenceWorkerRuntime,
  bootstrapPromise: Promise<void>,
): Promise<Float64Array> {
  await bootstrapPromise;
  ensureSharedWorkerOpen(runtime.state.fatalError, runtime.state.isOpen);

  if (!runtime.state.isBusy) {
    throw new Error('SharedInferenceWorker has no in-flight request.');
  }

  await runtime.state.pendingInputCopyPromise;

  await waitForSharedStatus(
    runtime.controlView,
    runtime.layout.statusIndexes.status,
    runtime.layout.statusValues.outputReady,
    () =>
      ensureSharedWorkerOpen(runtime.state.fatalError, runtime.state.isOpen),
  );

  const outputValues = await copySharedOutputValues(
    runtime.dataView,
    runtime.layout.outputOffset,
    runtime.layout.outputCount,
    runtime.shouldUseChunkedConversions,
  );

  Atomics.store(
    runtime.controlView,
    runtime.layout.statusIndexes.status,
    runtime.layout.statusValues.idle,
  );
  Atomics.notify(runtime.controlView, runtime.layout.statusIndexes.status);
  runtime.state.isBusy = false;
  return outputValues;
}

async function inferWithSharedInferenceWorker(
  runtime: SharedInferenceWorkerRuntime,
  bootstrapPromise: Promise<void>,
  input: ReadonlyArray<number>,
): Promise<Float64Array> {
  await bootstrapPromise;
  submitSharedInferenceInput(runtime, input);
  return await awaitSharedInferenceOutput(runtime, bootstrapPromise);
}

async function releaseSharedInferenceWorker(
  runtime: SharedInferenceWorkerRuntime,
): Promise<void> {
  if (runtime.state.closePromise) {
    await runtime.state.closePromise;
    return;
  }

  if (runtime.state.isBusy) {
    throw new Error(
      'SharedInferenceWorker cannot release while one request is in flight.',
    );
  }

  runtime.state.closePromise = shutdownSharedInferenceWorker(runtime);
  await runtime.state.closePromise;
}

async function resetSharedInferenceWorker(
  runtime: SharedInferenceWorkerRuntime,
  bootstrapPromise: Promise<void>,
): Promise<void> {
  await bootstrapPromise;
  ensureSharedWorkerOpen(runtime.state.fatalError, runtime.state.isOpen);

  if (runtime.state.isBusy) {
    throw new Error(
      'SharedInferenceWorker cannot reset while one request is in flight.',
    );
  }

  Atomics.store(
    runtime.controlView,
    runtime.layout.statusIndexes.status,
    runtime.layout.statusValues.resetRequested,
  );
  Atomics.notify(runtime.controlView, runtime.layout.statusIndexes.status);

  await waitForSharedStatus(
    runtime.controlView,
    runtime.layout.statusIndexes.status,
    runtime.layout.statusValues.idle,
    () =>
      ensureSharedWorkerOpen(runtime.state.fatalError, runtime.state.isOpen),
  );
}

function submitSharedInferenceInput(
  runtime: SharedInferenceWorkerRuntime,
  input: ReadonlyArray<number>,
): void {
  ensureSharedWorkerOpen(runtime.state.fatalError, runtime.state.isOpen);

  if (!runtime.state.isReady) {
    throw new Error('SharedInferenceWorker is not ready.');
  }

  if (runtime.state.isBusy) {
    throw new Error('SharedInferenceWorker is busy.');
  }

  if (input.length !== runtime.layout.inputCount) {
    throw new Error(
      `SharedInferenceWorker expected ${String(runtime.layout.inputCount)} inputs but received ${String(input.length)}.`,
    );
  }

  runtime.state.isBusy = true;
  const nextInputCopyPromise = copyInputValuesIntoSharedBuffer(
    runtime.dataView,
    input,
    runtime.layout.inputOffset,
    runtime.shouldUseChunkedConversions,
  )
    .then(() => {
      Atomics.store(
        runtime.controlView,
        runtime.layout.statusIndexes.status,
        runtime.layout.statusValues.inputReady,
      );
      Atomics.notify(runtime.controlView, runtime.layout.statusIndexes.status);
    })
    .catch((error) => {
      const normalizedError = asError(error);

      handleFatalSharedWorkerError(runtime, normalizedError);
      throw normalizedError;
    })
    .finally(() => {
      runtime.state.pendingInputCopyPromise = undefined;
    });

  runtime.state.pendingInputCopyPromise = nextInputCopyPromise;
}

async function shutdownSharedInferenceWorker(
  runtime: SharedInferenceWorkerRuntime,
): Promise<void> {
  runtime.state.isOpen = false;
  runtime.state.isReady = false;
  runtime.state.isBusy = false;

  const workerTermination = runtime.state.workerHandle?.terminate();

  if (workerTermination instanceof Promise) {
    try {
      await workerTermination;
    } catch {
      // Ignore worker termination races because local teardown already won.
    }
  }

  runtime.state.revokeWorkerUrl?.();
  runtime.state.workerHandle = undefined;
  runtime.state.revokeWorkerUrl = undefined;
}

function resolveSharedInferenceBufferLayout(
  inputCount: number,
  outputCount: number,
) {
  const normalizedInputCount = Math.max(0, Math.trunc(inputCount));
  const normalizedOutputCount = Math.max(0, Math.trunc(outputCount));

  return {
    controlElementCount: SHARED_INFERENCE_CONTROL_ELEMENT_COUNT,
    dataElementCount: normalizedInputCount + normalizedOutputCount,
    inputCount: normalizedInputCount,
    inputOffset: 0,
    outputCount: normalizedOutputCount,
    outputOffset: normalizedInputCount,
    statusIndexes: {
      inputCount: SHARED_INFERENCE_INPUT_COUNT_INDEX,
      status: SHARED_INFERENCE_STATUS_INDEX,
    },
    statusValues: {
      idle: SHARED_INFERENCE_IDLE_STATUS,
      inputReady: SHARED_INFERENCE_INPUT_READY_STATUS,
      outputReady: SHARED_INFERENCE_OUTPUT_READY_STATUS,
      resetRequested: SHARED_INFERENCE_RESET_REQUESTED_STATUS,
    },
  };
}

function initializeSharedInferenceWorker(
  options: SharedInferenceWorkerOptions,
):
  | InitializedSharedInferenceWorker
  | Promise<InitializedSharedInferenceWorker> {
  const workerSpecifier = resolveInferenceSharedWorkerSpecifier(
    options.workerUrl,
  );

  if (typeof globalThis.Worker === 'function') {
    return createBrowserSharedInferenceWorker(workerSpecifier);
  }

  return createNodeSharedInferenceWorker(workerSpecifier);
}

function resolveInferenceSharedWorkerSpecifier(
  workerUrl: string | undefined,
): string {
  if (workerUrl) {
    return workerUrl;
  }

  if (typeof process !== 'undefined' && typeof process.cwd === 'function') {
    return resolveNodeDefaultWorkerPath();
  }

  return resolveBrowserDefaultWorkerPath();
}

function createBrowserSharedInferenceWorker(
  workerSpecifier: string,
): InitializedSharedInferenceWorker {
  return {
    worker: new Worker(workerSpecifier, {
      type: 'module',
    }) as unknown as WorkerLike,
  };
}

function resolveBrowserDefaultWorkerPath(
  browserDocument:
    | { currentScript?: { src?: string | null } | null }
    | undefined = globalThis.document as
    | { currentScript?: { src?: string | null } | null }
    | undefined,
  currentLocationHref: string | undefined = globalThis.location?.href,
): string {
  const currentScriptUrl = browserDocument?.currentScript?.src ?? undefined;

  return (
    resolveBrowserWorkerAssetUrl(
      'architecture/network/worker-payload/network.worker-payload.shared.worker.js',
      {
        baseUrl: currentScriptUrl ?? currentLocationHref,
      },
    ) ??
    'architecture/network/worker-payload/network.worker-payload.shared.worker.js'
  );
}

async function createNodeSharedInferenceWorker(
  workerSpecifier: string,
): Promise<InitializedSharedInferenceWorker> {
  const workerThreadsModule = await loadNodeWorkerThreadsModule();
  const worker = new workerThreadsModule.Worker(
    workerSpecifier,
    resolveNodeSharedWorkerOptions(workerSpecifier),
  );

  return {
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

function resolveNodeSharedWorkerOptions(workerSpecifier: string): {
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

function attachSharedWorkerMessageListener(
  worker: WorkerLike,
  onMessage: (message: unknown) => void,
  onMessageError: (error: Error) => void,
): void {
  if (typeof worker.addEventListener === 'function') {
    worker.addEventListener('message', (event) => {
      onMessage(resolveMessageEventData(event));
    });
    worker.addEventListener('messageerror', () => {
      onMessageError(new Error('SharedInferenceWorker rejected one message.'));
    });
    return;
  }

  worker.on?.('message', onMessage);
  worker.on?.('messageerror', () => {
    onMessageError(new Error('SharedInferenceWorker rejected one message.'));
  });
}

function attachSharedWorkerLifecycleListeners(
  worker: WorkerLike,
  onFailure: (error: Error) => void,
): void {
  if (typeof worker.addEventListener === 'function') {
    worker.addEventListener('error', (event) => {
      onFailure(resolveWorkerError(event));
    });
    return;
  }

  worker.on?.('error', (error) => {
    onFailure(asError(error));
  });
  worker.on?.('exit', (exitCode) => {
    onFailure(
      new Error(`SharedInferenceWorker exited with code ${String(exitCode)}.`),
    );
  });
}

function postMessageToSharedWorker(
  worker: WorkerLike,
  message: SharedWorkerBootstrapMessage,
  transferList: Transferable[],
): void {
  worker.postMessage(message, transferList);
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

  return new Error(workerErrorEvent.message ?? 'SharedInferenceWorker failed.');
}

function resolveNodeDefaultWorkerPath(): string {
  const normalizedWorkingDirectory = process.cwd().replace(/\\/g, '/');

  if (typeof process.env.JEST_WORKER_ID === 'string') {
    return `${normalizedWorkingDirectory}/src/architecture/network/worker-payload/network.worker-payload.shared.worker.ts`;
  }

  return `${normalizedWorkingDirectory}/dist/architecture/network/worker-payload/network.worker-payload.shared.worker.js`;
}

function ensureSharedArrayBufferSupport(): void {
  if (typeof SharedArrayBuffer !== 'function') {
    throw new Error(
      'SharedInferenceWorker requires SharedArrayBuffer support.',
    );
  }
}

/**
 * Copy one submitted input vector into the shared numeric shelf.
 *
 * Browser-backed hosts can chunk very large numeric copies because the shared
 * worker API already exposes an async boundary. Node keeps the native bulk copy
 * path because there is no UI thread to protect.
 *
 * @param dataView - Shared numeric shelf covering inputs and outputs.
 * @param inputValues - Caller-provided numeric input vector.
 * @param inputOffset - Start index of the shared input shelf.
 * @param useChunkedConversion - True when the caller wants browser-style chunking for large copies.
 * @returns Promise resolved after the shared input shelf mirrors the caller input.
 */
async function copyInputValuesIntoSharedBuffer(
  dataView: Float64Array,
  inputValues: ReadonlyArray<number>,
  inputOffset: number,
  useChunkedConversion: boolean,
): Promise<void> {
  if (
    !shouldChunkSharedNumericConversion(
      inputValues.length,
      useChunkedConversion,
    )
  ) {
    dataView.set(inputValues, inputOffset);
    return;
  }

  await copySharedNumericValuesInChunks(
    inputValues.length,
    (startIndex, endIndex) => {
      for (
        let inputIndex = startIndex;
        inputIndex < endIndex;
        inputIndex += 1
      ) {
        dataView[inputOffset + inputIndex] = inputValues[inputIndex] as number;
      }
    },
  );
}

/**
 * Detach one shared-memory output shelf into a standalone typed array.
 *
 * @param dataView - Shared numeric shelf covering inputs and outputs.
 * @param outputOffset - Start index of the output shelf.
 * @param outputCount - Number of output values to copy.
 * @param useChunkedConversion - True when the caller wants browser-style chunking for large copies.
 * @returns Detached output values safe for the caller to retain.
 */
async function copySharedOutputValues(
  dataView: Float64Array,
  outputOffset: number,
  outputCount: number,
  useChunkedConversion: boolean,
): Promise<Float64Array> {
  const sharedOutputView = dataView.subarray(
    outputOffset,
    outputOffset + outputCount,
  );

  if (!shouldChunkSharedNumericConversion(outputCount, useChunkedConversion)) {
    return sharedOutputView.slice();
  }

  const detachedOutputValues = new Float64Array(outputCount);

  await copySharedNumericValuesInChunks(outputCount, (startIndex, endIndex) => {
    for (
      let outputIndex = startIndex;
      outputIndex < endIndex;
      outputIndex += 1
    ) {
      detachedOutputValues[outputIndex] = sharedOutputView[
        outputIndex
      ] as number;
    }
  });

  return detachedOutputValues;
}

/**
 * Decide whether one shared numeric copy should yield between browser-sized chunks.
 *
 * @param valueCount - Number of numeric slots involved in the conversion.
 * @param useChunkedConversion - True when the current call path wants cooperative chunking.
 * @returns True when the current host is browser-backed and the conversion is large.
 */
function shouldChunkSharedNumericConversion(
  valueCount: number,
  useChunkedConversion: boolean,
): boolean {
  return (
    useChunkedConversion &&
    valueCount > SHARED_INFERENCE_ASYNC_CONVERSION_THRESHOLD
  );
}

/**
 * Copy one large numeric conversion in browser-friendly chunks.
 *
 * @param valueCount - Total numeric slots to copy.
 * @param copyChunk - Callback that copies one contiguous chunk range.
 * @returns Promise resolved after every chunk has been copied.
 */
async function copySharedNumericValuesInChunks(
  valueCount: number,
  copyChunk: (startIndex: number, endIndex: number) => void,
): Promise<void> {
  for (
    let startIndex = 0;
    startIndex < valueCount;
    startIndex += SHARED_INFERENCE_ASYNC_CONVERSION_CHUNK_SIZE
  ) {
    const endIndex = Math.min(
      valueCount,
      startIndex + SHARED_INFERENCE_ASYNC_CONVERSION_CHUNK_SIZE,
    );

    copyChunk(startIndex, endIndex);

    if (endIndex < valueCount) {
      await yieldSharedConversionMacrotask();
    }
  }
}

/**
 * Yield one timer turn between browser conversion chunks.
 *
 * @returns Promise resolved on the next macrotask turn.
 */
async function yieldSharedConversionMacrotask(): Promise<void> {
  await new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
  });
}

function ensureSharedWorkerOpen(
  fatalError: Error | undefined,
  isOpen: boolean,
): void {
  if (isOpen) {
    return;
  }

  throw fatalError ?? new Error('SharedInferenceWorker is closed.');
}

async function waitForSharedStatus(
  controlView: Int32Array,
  statusIndex: number,
  expectedStatus: number,
  assertOpen: () => void,
): Promise<void> {
  while (Atomics.load(controlView, statusIndex) !== expectedStatus) {
    assertOpen();

    const currentStatus = Atomics.load(controlView, statusIndex);
    const waitAsync = Reflect.get(Atomics, 'waitAsync') as
      | undefined
      | ((
          typedArray: Int32Array,
          index: number,
          value: number,
        ) =>
          | { async: false; value: 'not-equal' | 'ok' | 'timed-out' }
          | {
              async: true;
              value: Promise<'not-equal' | 'ok' | 'timed-out'>;
            });

    if (typeof waitAsync === 'function') {
      const waitResult = waitAsync(controlView, statusIndex, currentStatus);

      if (waitResult.async) {
        await waitResult.value;
        continue;
      }

      if (waitResult.value === 'not-equal') {
        continue;
      }
    }

    await new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });
  }
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

  return new Error(String(error));
}

function isPromiseLike<T>(value: T | Promise<T>): value is Promise<T> {
  return value instanceof Promise;
}

/**
 * Test-only helper surface exposing private shared-memory helpers for owner-local coverage.
 *
 * @internal
 */
export const SHARED_INFERENCE_HOST_INTERNALS = {
  asError,
  attachSharedWorkerLifecycleListeners,
  attachSharedWorkerMessageListener,
  copyInputValuesIntoSharedBuffer,
  copySharedOutputValues,
  createBrowserSharedInferenceWorker,
  ensureSharedArrayBufferSupport,
  ensureSharedWorkerOpen,
  loadNodeWorkerThreadsModule,
  resolveBrowserDefaultWorkerPath,
  resolveInferenceSharedWorkerSpecifier,
  resolveMessageEventData,
  resolveNodeDefaultWorkerPath,
  resolveNodeSharedWorkerOptions,
  resolveSharedInferenceBufferLayout,
  resolveWorkerError,
  waitForSharedStatus,
};
