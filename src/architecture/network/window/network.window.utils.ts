import type Network from '../network';
import type {
  NetworkForwardWindowAsyncOptions,
  NetworkForwardWindowChunk,
  NetworkForwardWindowOptions,
} from '../network.types';
import { defaultMemoryManager } from '../../../memory/manager';
import type { MemoryManagerEnvironment } from '../../../memory/config';
import {
  BATCH_INPUTS_COLLECTION_ERROR_MESSAGE,
  UNDEFINED_INPUT_LENGTH_TEXT,
} from '../activate/network.activate.utils.types';
import {
  NetworkActivateBatchInputsCollectionError,
  NetworkActivateInputSizeMismatchError,
} from '../activate/network.activate.errors';

/** Smallest legal activation-window size. */
const MIN_FORWARD_WINDOW_SIZE = 1;

/** Default browser-side window size for bounded sequence activation. */
const DEFAULT_BROWSER_FORWARD_WINDOW_SIZE = 32;

/** Default Node-side window size for bounded sequence activation. */
const DEFAULT_NODE_FORWARD_WINDOW_SIZE = 128;

/** Default browser yield cadence measured in completed windows. */
const DEFAULT_BROWSER_WINDOWS_PER_YIELD = 1;

type ForwardWindowYieldControl = () => Promise<void>;

type ForwardWindowContext = {
  collectOutputs: boolean;
  expectedInputSize: number;
  isTraining: boolean;
  network: Network;
  outputRows: number[][];
  sequenceInputs: number[][];
  windowBuffer: Array<number[] | undefined>;
  windowSize: number;
};

/**
 * Advance one input sequence in bounded windows while preserving carried recurrent state.
 *
 * This is the first common-path Phase 9 surface: it keeps the same activation
 * semantics as repeated `activate()` calls, but it advances the sequence in
 * explicit windows so later browser and low-memory follow-up work has one
 * stable orchestration boundary.
 *
 * @param this - Bound network instance.
 * @param inputs - Ordered sequence of input vectors.
 * @param options - Optional activation-window configuration.
 * @returns Output vectors aligned to the input order.
 */
export function forwardWindowed(
  this: Network,
  inputs: number[][],
  options: NetworkForwardWindowOptions = {},
): number[][] {
  const environment = resolveForwardWindowEnvironment();
  const forwardWindowContext = createForwardWindowContext(
    this,
    inputs,
    options,
    environment,
  );

  // Step 1: Advance the sequence in bounded windows while keeping carried state intact.
  return collectForwardWindowOutputs(forwardWindowContext, options.onWindow);
}

/**
 * Advance one input sequence in bounded windows while yielding between browser-sized slices.
 *
 * This keeps the same recurrent semantics as `forwardWindowed()`, but it can
 * cooperatively yield after a configurable number of completed windows so long
 * browser sequences do not monopolize the main thread.
 *
 * @param this - Bound network instance.
 * @param inputs - Ordered sequence of input vectors.
 * @param options - Optional async activation-window configuration.
 * @returns Output vectors aligned to the input order.
 */
export async function forwardWindowedAsync(
  this: Network,
  inputs: number[][],
  options: NetworkForwardWindowAsyncOptions = {},
): Promise<number[][]> {
  const environment = resolveForwardWindowEnvironment();
  const forwardWindowContext = createForwardWindowContext(
    this,
    inputs,
    options,
    environment,
  );
  const yieldAfterWindows = normalizeYieldAfterWindows(
    options.yieldAfterWindows,
    environment,
  );
  const yieldControl =
    options.yieldControl ??
    resolveDefaultForwardWindowYieldControl(environment);

  // Step 1: Advance the sequence in bounded windows and yield between browser slices.
  return collectForwardWindowOutputsAsync(
    forwardWindowContext,
    options.onWindow,
    yieldAfterWindows,
    yieldControl,
  );
}

/**
 * Validate that the windowed input collection is an array of input vectors.
 *
 * @param inputs - Candidate input sequence.
 * @returns Nothing.
 */
function assertForwardWindowInputsCollection(inputs: number[][]): void {
  if (!Array.isArray(inputs)) {
    throw new NetworkActivateBatchInputsCollectionError(
      BATCH_INPUTS_COLLECTION_ERROR_MESSAGE,
    );
  }
}

/**
 * Advance one input sequence in explicit windows and collect ordered outputs.
 *
 * @param network - Bound network instance.
 * @param inputs - Ordered sequence of input vectors.
 * @param windowSize - Number of rows processed per window.
 * @param isTraining - Whether training-time activation semantics are enabled.
 * @returns Output vectors aligned to the input order.
 */
function collectForwardWindowOutputs(
  forwardWindowContext: ForwardWindowContext,
  onWindow: NetworkForwardWindowOptions['onWindow'],
): number[][] {
  let windowIndex = 0;
  let windowRowCount = 0;
  let windowStartIndex = 0;

  for (
    let inputIndex = 0;
    inputIndex < forwardWindowContext.sequenceInputs.length;
    inputIndex += 1
  ) {
    const inputVector = forwardWindowContext.sequenceInputs[inputIndex]!;
    const isLastInput =
      inputIndex === forwardWindowContext.sequenceInputs.length - 1;

    assertForwardWindowInputSize(
      inputVector,
      forwardWindowContext.expectedInputSize,
      inputIndex,
    );
    forwardWindowContext.windowBuffer[windowRowCount] =
      forwardWindowContext.network.activate(
        inputVector,
        forwardWindowContext.isTraining,
      );
    windowRowCount += 1;

    if (windowRowCount === forwardWindowContext.windowSize || isLastInput) {
      const windowChunk = createForwardWindowChunk(
        forwardWindowContext.windowBuffer,
        windowRowCount,
        windowIndex,
        windowStartIndex,
        inputIndex + 1,
        isLastInput,
      );

      onWindow?.(windowChunk);
      appendForwardWindowOutputs(forwardWindowContext, windowChunk.outputs);
      clearForwardWindowBuffer(
        forwardWindowContext.windowBuffer,
        windowRowCount,
      );
      windowIndex += 1;
      windowRowCount = 0;
      windowStartIndex = inputIndex + 1;
    }
  }

  return forwardWindowContext.outputRows;
}

/**
 * Advance one input sequence in explicit windows while yielding between browser-sized slices.
 *
 * @param forwardWindowContext - Shared windowed activation state.
 * @param onWindow - Optional async window callback.
 * @param yieldAfterWindows - Yield cadence measured in completed windows.
 * @param yieldControl - Optional async yield hook.
 * @returns Output vectors aligned to the input order.
 */
async function collectForwardWindowOutputsAsync(
  forwardWindowContext: ForwardWindowContext,
  onWindow: NetworkForwardWindowAsyncOptions['onWindow'],
  yieldAfterWindows: number,
  yieldControl: ForwardWindowYieldControl | undefined,
): Promise<number[][]> {
  let windowIndex = 0;
  let windowRowCount = 0;
  let windowStartIndex = 0;

  for (
    let inputIndex = 0;
    inputIndex < forwardWindowContext.sequenceInputs.length;
    inputIndex += 1
  ) {
    const inputVector = forwardWindowContext.sequenceInputs[inputIndex]!;
    const isLastInput =
      inputIndex === forwardWindowContext.sequenceInputs.length - 1;

    assertForwardWindowInputSize(
      inputVector,
      forwardWindowContext.expectedInputSize,
      inputIndex,
    );
    forwardWindowContext.windowBuffer[windowRowCount] =
      forwardWindowContext.network.activate(
        inputVector,
        forwardWindowContext.isTraining,
      );
    windowRowCount += 1;

    if (windowRowCount !== forwardWindowContext.windowSize && !isLastInput) {
      continue;
    }

    const windowChunk = createForwardWindowChunk(
      forwardWindowContext.windowBuffer,
      windowRowCount,
      windowIndex,
      windowStartIndex,
      inputIndex + 1,
      isLastInput,
    );

    if (onWindow) {
      await onWindow(windowChunk);
    }

    appendForwardWindowOutputs(forwardWindowContext, windowChunk.outputs);
    clearForwardWindowBuffer(forwardWindowContext.windowBuffer, windowRowCount);
    windowIndex += 1;
    windowRowCount = 0;
    windowStartIndex = inputIndex + 1;

    const activeYieldControl = yieldControl;

    if (
      activeYieldControl &&
      shouldYieldAfterCompletedWindow(
        windowIndex,
        isLastInput,
        yieldAfterWindows,
        activeYieldControl,
      )
    ) {
      await activeYieldControl();
    }
  }

  return forwardWindowContext.outputRows;
}

/**
 * Validate one input vector width for windowed activation.
 *
 * @param inputVector - Candidate input vector.
 * @param expectedInputSize - Required input width.
 * @param inputIndex - Sequence index used in the error message.
 * @returns Nothing.
 */
function assertForwardWindowInputSize(
  inputVector: number[],
  expectedInputSize: number,
  inputIndex: number,
): void {
  if (Array.isArray(inputVector) && inputVector.length === expectedInputSize) {
    return;
  }

  throw new NetworkActivateInputSizeMismatchError(
    `Input[${inputIndex}] size mismatch: expected ${expectedInputSize}, got ${formatInputLengthForMessage(inputVector)}`,
  );
}

/**
 * Normalize the requested forward-window size to a positive integer.
 *
 * @param windowSize - Optional requested window size.
 * @returns Positive integer window size.
 */
function normalizeForwardWindowSize(
  windowSize: number | undefined,
  environment: MemoryManagerEnvironment,
): number {
  if (typeof windowSize !== 'number' || !Number.isFinite(windowSize)) {
    return resolveDefaultForwardWindowSize(environment);
  }

  return Math.max(MIN_FORWARD_WINDOW_SIZE, Math.floor(windowSize));
}

/**
 * Normalize the requested async yield cadence to a positive integer or infinity.
 *
 * @param yieldAfterWindows - Optional requested yield cadence.
 * @param environment - Active runtime environment.
 * @returns Positive integer yield cadence or infinity when yielding is disabled.
 */
function normalizeYieldAfterWindows(
  yieldAfterWindows: number | undefined,
  environment: MemoryManagerEnvironment,
): number {
  if (
    typeof yieldAfterWindows !== 'number' ||
    !Number.isFinite(yieldAfterWindows)
  ) {
    return environment === 'browser'
      ? DEFAULT_BROWSER_WINDOWS_PER_YIELD
      : Number.POSITIVE_INFINITY;
  }

  return Math.max(MIN_FORWARD_WINDOW_SIZE, Math.floor(yieldAfterWindows));
}

/**
 * Resolve the current runtime environment through the shared memory owner.
 *
 * @returns Active runtime environment label.
 */
function resolveForwardWindowEnvironment(): MemoryManagerEnvironment {
  return defaultMemoryManager.getConfig().environment;
}

/**
 * Resolve the default forward-window size for the active runtime.
 *
 * @param environment - Active runtime environment.
 * @returns Default bounded window size.
 */
function resolveDefaultForwardWindowSize(
  environment: MemoryManagerEnvironment,
): number {
  return environment === 'browser'
    ? DEFAULT_BROWSER_FORWARD_WINDOW_SIZE
    : DEFAULT_NODE_FORWARD_WINDOW_SIZE;
}

/**
 * Resolve the default async yield hook for the active runtime.
 *
 * @param environment - Active runtime environment.
 * @returns Promise-based yield hook when the runtime exposes one.
 */
function resolveDefaultForwardWindowYieldControl(
  environment: MemoryManagerEnvironment,
): ForwardWindowYieldControl | undefined {
  if (environment !== 'browser') {
    return undefined;
  }

  const requestAnimationFrameReference = Reflect.get(
    globalThis,
    'requestAnimationFrame',
  );

  if (typeof requestAnimationFrameReference === 'function') {
    return () => {
      return new Promise<void>((resolve) => {
        requestAnimationFrameReference(() => {
          resolve();
        });
      });
    };
  }

  if (typeof setTimeout === 'function') {
    return () => {
      return new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
    };
  }

  return undefined;
}

/**
 * Create the shared state used by sync and async windowed activation.
 *
 * @param network - Bound network instance.
 * @param inputs - Ordered sequence of input vectors.
 * @param options - Shared windowed activation options.
 * @param environment - Active runtime environment.
 * @returns Shared forward-window activation context.
 */
function createForwardWindowContext(
  network: Network,
  inputs: number[][],
  options: NetworkForwardWindowOptions,
  environment: MemoryManagerEnvironment,
): ForwardWindowContext {
  assertForwardWindowInputsCollection(inputs);

  const windowSize = normalizeForwardWindowSize(
    options.windowSize,
    environment,
  );

  return {
    collectOutputs: options.collectOutputs !== false,
    expectedInputSize: network.input,
    isTraining: options.training ?? false,
    network,
    outputRows: [],
    sequenceInputs: inputs,
    windowBuffer: new Array<number[] | undefined>(windowSize),
    windowSize,
  };
}

/**
 * Create one emitted window chunk from the reusable buffer.
 *
 * @param windowBuffer - Reusable bounded buffer of row outputs.
 * @param windowRowCount - Number of valid rows currently buffered.
 * @param windowIndex - Zero-based emitted window index.
 * @param startIndex - Inclusive sequence start index.
 * @param endIndexExclusive - Exclusive sequence end index.
 * @param done - Whether this is the final emitted window.
 * @returns One stable window chunk for callbacks and collectors.
 */
function createForwardWindowChunk(
  windowBuffer: Array<number[] | undefined>,
  windowRowCount: number,
  windowIndex: number,
  startIndex: number,
  endIndexExclusive: number,
  done: boolean,
): NetworkForwardWindowChunk {
  return {
    done,
    endIndexExclusive,
    outputs: windowBuffer
      .slice(0, windowRowCount)
      .map((outputRow) => outputRow!),
    startIndex,
    windowIndex,
  };
}

/**
 * Append one emitted window to the collected result matrix when collection is enabled.
 *
 * @param forwardWindowContext - Shared windowed activation state.
 * @param windowOutputs - Stable emitted window outputs.
 * @returns Nothing.
 */
function appendForwardWindowOutputs(
  forwardWindowContext: ForwardWindowContext,
  windowOutputs: number[][],
): void {
  if (forwardWindowContext.collectOutputs) {
    forwardWindowContext.outputRows.push(...windowOutputs);
  }
}

/**
 * Release bounded buffer references once one emitted window has been handled.
 *
 * @param windowBuffer - Reusable bounded buffer of row outputs.
 * @param windowRowCount - Number of valid rows currently buffered.
 * @returns Nothing.
 */
function clearForwardWindowBuffer(
  windowBuffer: Array<number[] | undefined>,
  windowRowCount: number,
): void {
  for (let rowIndex = 0; rowIndex < windowRowCount; rowIndex += 1) {
    windowBuffer[rowIndex] = undefined;
  }
}

/**
 * Determine whether the async path should yield after one completed window.
 *
 * @param completedWindowCount - Total number of emitted windows so far.
 * @param isLastInput - Whether the just-emitted window closed the sequence.
 * @param yieldAfterWindows - Yield cadence measured in completed windows.
 * @param yieldControl - Optional async yield hook.
 * @returns Whether the async path should yield now.
 */
function shouldYieldAfterCompletedWindow(
  completedWindowCount: number,
  isLastInput: boolean,
  yieldAfterWindows: number,
  yieldControl: ForwardWindowYieldControl | undefined,
): boolean {
  if (isLastInput || !yieldControl || !Number.isFinite(yieldAfterWindows)) {
    return false;
  }

  return completedWindowCount % yieldAfterWindows === 0;
}

/**
 * Convert one input length into a display-safe string for error messages.
 *
 * @param inputVector - Candidate input vector.
 * @returns Numeric length as text or the shared undefined marker.
 */
function formatInputLengthForMessage(inputVector: number[]): string {
  if (inputVector) {
    return `${inputVector.length}`;
  }

  return UNDEFINED_INPUT_LENGTH_TEXT;
}
