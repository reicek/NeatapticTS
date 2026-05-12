import { SHARED_INFERENCE_HOST_INTERNALS } from './network.worker-payload.shared';
import { createInferencePredictor } from './network.worker-payload.utils';
import type { TransferableInferencePayload } from './network.worker-payload.types';

const SHARED_INFERENCE_READY_MESSAGE_TYPE = 'ready';
const SHARED_INFERENCE_REQUEST_ERROR_MESSAGE_TYPE = 'request-error';

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

type NodeParentPortLike = {
  close?: () => void;
  once?: (type: string, listener: (message: unknown) => void) => void;
  postMessage?: (message: SharedWorkerResponseMessage) => void;
};

type NodeWorkerThreadsModule = {
  parentPort?: NodeParentPortLike;
};

/**
 * Register the worker-side shared-memory inference runtime.
 *
 * The worker rebuilds one predictor from the transferred payload during
 * bootstrap, then services inference requests by reading and writing the
 * shared control and data shelves coordinated through `Atomics`.
 *
 * @returns Nothing.
 */
export function registerSharedInferenceWorkerRuntime(): void {
  if (
    typeof process !== 'undefined' &&
    typeof process.versions?.node === 'string'
  ) {
    void loadNodeWorkerThreadsModule().then((workerThreadsModule) => {
      registerNodeSharedInferenceWorkerRuntime(
        workerThreadsModule.parentPort as NodeParentPortLike | undefined,
        exitCurrentNodeSharedInferenceWorkerRuntime,
      );
    });
    return;
  }

  registerBrowserSharedInferenceWorkerRuntime(
    (listener) => {
      globalThis.addEventListener('message', listener, { once: true });
    },
    (message) => {
      globalThis.postMessage(message);
    },
    closeCurrentBrowserSharedInferenceWorkerRuntime,
  );
}

function autoRegisterSharedInferenceWorkerRuntime(): void {
  if (shouldAutoRegisterSharedInferenceWorkerRuntime()) {
    registerSharedInferenceWorkerRuntime();
  }
}

autoRegisterSharedInferenceWorkerRuntime();

function shouldAutoRegisterSharedInferenceWorkerRuntime(): boolean {
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

function handleBootstrapMessage(message: unknown):
  | undefined
  | {
      controlView: Int32Array;
      dataView: Float64Array;
      layout: ReturnType<
        typeof SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout
      >;
      predictor: ReturnType<typeof createInferencePredictor>;
    } {
  const bootstrapMessage = message as Partial<SharedWorkerBootstrapMessage>;

  if (bootstrapMessage.type !== 'bootstrap') {
    return undefined;
  }

  if (
    !(bootstrapMessage.controlBuffer instanceof SharedArrayBuffer) ||
    !(bootstrapMessage.dataBuffer instanceof SharedArrayBuffer) ||
    !bootstrapMessage.payload
  ) {
    throw new Error(
      'SharedInferenceWorker expected bootstrap payload plus shared buffers.',
    );
  }

  const layout =
    SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout(
      bootstrapMessage.payload.inputCount,
      bootstrapMessage.payload.outputCount,
    );
  const controlView = new Int32Array(bootstrapMessage.controlBuffer);
  const dataView = new Float64Array(bootstrapMessage.dataBuffer);
  const predictor = createInferencePredictor(bootstrapMessage.payload);

  Atomics.store(
    controlView,
    layout.statusIndexes.inputCount,
    layout.inputCount,
  );
  Atomics.store(
    controlView,
    layout.statusIndexes.status,
    layout.statusValues.idle,
  );
  Atomics.notify(controlView, layout.statusIndexes.status);

  return {
    controlView,
    dataView,
    layout,
    predictor,
  };
}

function registerNodeSharedInferenceWorkerRuntime(
  parentPort: NodeParentPortLike | undefined,
  exitProcess: () => void,
  scheduleLoop: (callback: () => void) => void = queueMicrotask,
  runLoop: typeof runSharedInferenceLoop = runSharedInferenceLoop,
): void {
  parentPort?.once?.('message', (message: unknown) => {
    try {
      const bootstrapState = handleBootstrapMessage(message);

      if (!bootstrapState) {
        return;
      }

      postWorkerMessage(parentPort, {
        type: SHARED_INFERENCE_READY_MESSAGE_TYPE,
      });
      scheduleLoop(() => {
        runLoop(
          bootstrapState.controlView,
          bootstrapState.dataView,
          bootstrapState.predictor,
          bootstrapState.layout,
        );
      });
    } catch (error) {
      postWorkerMessage(parentPort, {
        message: asError(error).message,
        type: SHARED_INFERENCE_REQUEST_ERROR_MESSAGE_TYPE,
      });
      parentPort.close?.();
      exitProcess();
    }
  });
}

function registerBrowserSharedInferenceWorkerRuntime(
  addMessageListener: (listener: (event: unknown) => void) => void,
  postMessage: (message: SharedWorkerResponseMessage) => void,
  closeWorker: () => void,
  scheduleLoop: (callback: () => void) => void = queueMicrotask,
  runLoop: typeof runSharedInferenceLoop = runSharedInferenceLoop,
): void {
  addMessageListener((event) => {
    try {
      const bootstrapState = handleBootstrapMessage(
        resolveMessageEventData(event),
      );

      if (!bootstrapState) {
        return;
      }

      postMessage({ type: SHARED_INFERENCE_READY_MESSAGE_TYPE });
      scheduleLoop(() => {
        runLoop(
          bootstrapState.controlView,
          bootstrapState.dataView,
          bootstrapState.predictor,
          bootstrapState.layout,
        );
      });
    } catch (error) {
      postMessage({
        message: asError(error).message,
        type: SHARED_INFERENCE_REQUEST_ERROR_MESSAGE_TYPE,
      });
      closeWorker();
    }
  });
}

function runSharedInferenceLoop(
  controlView: Int32Array,
  dataView: Float64Array,
  predictor: ReturnType<typeof createInferencePredictor>,
  layout: ReturnType<
    typeof SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout
  >,
  waitForStatusChange: (
    nextControlView: Int32Array,
    statusIndex: number,
    currentStatus: number,
  ) => void = defaultWaitForSharedStatusChange,
): never {
  while (true) {
    const loopState = handleSharedInferenceLoopStep(
      controlView,
      dataView,
      predictor,
      layout,
    );

    if (loopState !== 'waiting') {
      continue;
    }

    const currentStatus = Atomics.load(
      controlView,
      layout.statusIndexes.status,
    );
    waitForStatusChange(
      controlView,
      layout.statusIndexes.status,
      currentStatus,
    );
  }
}

function handleSharedInferenceLoopStep(
  controlView: Int32Array,
  dataView: Float64Array,
  predictor: ReturnType<typeof createInferencePredictor>,
  layout: ReturnType<
    typeof SHARED_INFERENCE_HOST_INTERNALS.resolveSharedInferenceBufferLayout
  >,
): 'predicted' | 'reset' | 'waiting' {
  const currentStatus = Atomics.load(controlView, layout.statusIndexes.status);

  if (currentStatus === layout.statusValues.inputReady) {
    const inputValues = Array.from(
      dataView.subarray(
        layout.inputOffset,
        layout.inputOffset + layout.inputCount,
      ),
    );
    const outputValues = predictor.predict(inputValues);

    dataView.fill(
      0,
      layout.outputOffset,
      layout.outputOffset + layout.outputCount,
    );
    outputValues.slice(0, layout.outputCount).forEach((outputValue, index) => {
      dataView[layout.outputOffset + index] = outputValue;
    });
    Atomics.store(
      controlView,
      layout.statusIndexes.status,
      layout.statusValues.outputReady,
    );
    Atomics.notify(controlView, layout.statusIndexes.status);
    return 'predicted';
  }

  if (currentStatus === layout.statusValues.resetRequested) {
    predictor.reset();
    Atomics.store(
      controlView,
      layout.statusIndexes.status,
      layout.statusValues.idle,
    );
    Atomics.notify(controlView, layout.statusIndexes.status);
    return 'reset';
  }

  return 'waiting';
}

function postWorkerMessage(
  postTarget:
    | NodeParentPortLike
    | { postMessage: (message: SharedWorkerResponseMessage) => void }
    | undefined,
  message: SharedWorkerResponseMessage,
): void {
  postTarget?.postMessage?.(message);
}

function defaultWaitForSharedStatusChange(
  controlView: Int32Array,
  statusIndex: number,
  currentStatus: number,
): void {
  Atomics.wait(controlView, statusIndex, currentStatus);
}

function resolveMessageEventData(event: unknown): unknown {
  const dataCarrier = event as { data?: unknown };

  if (typeof event === 'object' && event !== null && 'data' in dataCarrier) {
    return dataCarrier.data;
  }

  return event;
}

function exitCurrentNodeSharedInferenceWorkerRuntime(): void {
  process.exit(0);
}

function closeCurrentBrowserSharedInferenceWorkerRuntime(): void {
  globalThis.close();
}

function asError(error: unknown): Error {
  if (error instanceof Error) {
    return error;
  }

  return new Error(String(error));
}

/** @internal Test-only helper surface for owner-local shared-memory coverage. */
export const SHARED_INFERENCE_WORKER_INTERNALS = {
  autoRegisterSharedInferenceWorkerRuntime,
  asError,
  closeCurrentBrowserSharedInferenceWorkerRuntime,
  defaultWaitForSharedStatusChange,
  exitCurrentNodeSharedInferenceWorkerRuntime,
  handleBootstrapMessage,
  handleSharedInferenceLoopStep,
  loadNodeWorkerThreadsModule,
  postWorkerMessage,
  registerBrowserSharedInferenceWorkerRuntime,
  registerNodeSharedInferenceWorkerRuntime,
  resolveMessageEventData,
  runSharedInferenceLoop,
  shouldAutoRegisterSharedInferenceWorkerRuntime,
};
