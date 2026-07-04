# architecture/network/window

## architecture/network/window/network.window.utils.ts

### appendForwardWindowOutputs

```ts
appendForwardWindowOutputs(
  forwardWindowContext: ForwardWindowContext,
  windowOutputs: number[][],
): void
```

Append one emitted window to the collected result matrix when collection is enabled.

Parameters:

- `forwardWindowContext` - Shared windowed activation state.
- `windowOutputs` - Stable emitted window outputs.

Returns: Nothing.

### assertForwardWindowInputsCollection

```ts
assertForwardWindowInputsCollection(
  inputs: number[][],
): void
```

Validate that the windowed input collection is an array of input vectors.

Parameters:

- `inputs` - Candidate input sequence.

Returns: Nothing.

### assertForwardWindowInputSize

```ts
assertForwardWindowInputSize(
  inputVector: number[],
  expectedInputSize: number,
  inputIndex: number,
): void
```

Validate one input vector width for windowed activation.

Parameters:

- `inputVector` - Candidate input vector.
- `expectedInputSize` - Required input width.
- `inputIndex` - Sequence index used in the error message.

Returns: Nothing.

### clearForwardWindowBuffer

```ts
clearForwardWindowBuffer(
  windowBuffer: (number[] | undefined)[],
  windowRowCount: number,
): void
```

Release bounded buffer references once one emitted window has been handled.

Parameters:

- `windowBuffer` - Reusable bounded buffer of row outputs.
- `windowRowCount` - Number of valid rows currently buffered.

Returns: Nothing.

### collectForwardWindowOutputs

```ts
collectForwardWindowOutputs(
  forwardWindowContext: ForwardWindowContext,
  onWindow: ((chunk: NetworkForwardWindowChunk) => void) | undefined,
): number[][]
```

Advance one input sequence in explicit windows and collect ordered outputs.

Parameters:

- `network` - Bound network instance.
- `inputs` - Ordered sequence of input vectors.
- `windowSize` - Number of rows processed per window.
- `isTraining` - Whether training-time activation semantics are enabled.

Returns: Output vectors aligned to the input order.

### collectForwardWindowOutputsAsync

```ts
collectForwardWindowOutputsAsync(
  forwardWindowContext: ForwardWindowContext,
  onWindow: ((chunk: NetworkForwardWindowChunk) => void | Promise<void>) | undefined,
  yieldAfterWindows: number,
  yieldControl: ForwardWindowYieldControl | undefined,
): Promise<number[][]>
```

Advance one input sequence in explicit windows while yielding between browser-sized slices.

Parameters:

- `forwardWindowContext` - Shared windowed activation state.
- `onWindow` - Optional async window callback.
- `yieldAfterWindows` - Yield cadence measured in completed windows.
- `yieldControl` - Optional async yield hook.

Returns: Output vectors aligned to the input order.

### createForwardWindowChunk

```ts
createForwardWindowChunk(
  windowBuffer: (number[] | undefined)[],
  windowRowCount: number,
  windowIndex: number,
  startIndex: number,
  endIndexExclusive: number,
  done: boolean,
): NetworkForwardWindowChunk
```

Create one emitted window chunk from the reusable buffer.

Parameters:

- `windowBuffer` - Reusable bounded buffer of row outputs.
- `windowRowCount` - Number of valid rows currently buffered.
- `windowIndex` - Zero-based emitted window index.
- `startIndex` - Inclusive sequence start index.
- `endIndexExclusive` - Exclusive sequence end index.
- `done` - Whether this is the final emitted window.

Returns: One stable window chunk for callbacks and collectors.

### createForwardWindowContext

```ts
createForwardWindowContext(
  network: default,
  inputs: number[][],
  options: NetworkForwardWindowOptions,
  environment: MemoryManagerEnvironment,
): ForwardWindowContext
```

Create the shared state used by sync and async windowed activation.

Parameters:

- `network` - Bound network instance.
- `inputs` - Ordered sequence of input vectors.
- `options` - Shared windowed activation options.
- `environment` - Active runtime environment.

Returns: Shared forward-window activation context.

### formatInputLengthForMessage

```ts
formatInputLengthForMessage(
  inputVector: number[],
): string
```

Convert one input length into a display-safe string for error messages.

Parameters:

- `inputVector` - Candidate input vector.

Returns: Numeric length as text or the shared undefined marker.

### forwardWindowed

```ts
forwardWindowed(
  inputs: number[][],
  options: NetworkForwardWindowOptions,
): number[][]
```

Advance one input sequence in bounded windows while preserving carried recurrent state.

This is the first common-path Phase 9 surface: it keeps the same activation
semantics as repeated `activate()` calls, but it advances the sequence in
explicit windows so later browser and low-memory follow-up work has one
stable orchestration boundary.

Parameters:

- `this` - Bound network instance.
- `inputs` - Ordered sequence of input vectors.
- `options` - Optional activation-window configuration.

Returns: Output vectors aligned to the input order.

### forwardWindowedAsync

```ts
forwardWindowedAsync(
  inputs: number[][],
  options: NetworkForwardWindowAsyncOptions,
): Promise<number[][]>
```

Advance one input sequence in bounded windows while yielding between browser-sized slices.

This keeps the same recurrent semantics as `forwardWindowed()`, but it can
cooperatively yield after a configurable number of completed windows so long
browser sequences do not monopolize the main thread.

Parameters:

- `this` - Bound network instance.
- `inputs` - Ordered sequence of input vectors.
- `options` - Optional async activation-window configuration.

Returns: Output vectors aligned to the input order.

### normalizeForwardWindowSize

```ts
normalizeForwardWindowSize(
  windowSize: number | undefined,
  environment: MemoryManagerEnvironment,
): number
```

Normalize the requested forward-window size to a positive integer.

Parameters:

- `windowSize` - Optional requested window size.

Returns: Positive integer window size.

### normalizeYieldAfterWindows

```ts
normalizeYieldAfterWindows(
  yieldAfterWindows: number | undefined,
  environment: MemoryManagerEnvironment,
): number
```

Normalize the requested async yield cadence to a positive integer or infinity.

Parameters:

- `yieldAfterWindows` - Optional requested yield cadence.
- `environment` - Active runtime environment.

Returns: Positive integer yield cadence or infinity when yielding is disabled.

### resolveDefaultForwardWindowSize

```ts
resolveDefaultForwardWindowSize(
  environment: MemoryManagerEnvironment,
): number
```

Resolve the default forward-window size for the active runtime.

Parameters:

- `environment` - Active runtime environment.

Returns: Default bounded window size.

### resolveDefaultForwardWindowYieldControl

```ts
resolveDefaultForwardWindowYieldControl(
  environment: MemoryManagerEnvironment,
): ForwardWindowYieldControl | undefined
```

Resolve the default async yield hook for the active runtime.

Parameters:

- `environment` - Active runtime environment.

Returns: Promise-based yield hook when the runtime exposes one.

### resolveForwardWindowEnvironment

```ts
resolveForwardWindowEnvironment(): MemoryManagerEnvironment
```

Resolve the current runtime environment through the shared memory owner.

Returns: Active runtime environment label.

### shouldYieldAfterCompletedWindow

```ts
shouldYieldAfterCompletedWindow(
  completedWindowCount: number,
  isLastInput: boolean,
  yieldAfterWindows: number,
  yieldControl: ForwardWindowYieldControl | undefined,
): boolean
```

Determine whether the async path should yield after one completed window.

Parameters:

- `completedWindowCount` - Total number of emitted windows so far.
- `isLastInput` - Whether the just-emitted window closed the sequence.
- `yieldAfterWindows` - Yield cadence measured in completed windows.
- `yieldControl` - Optional async yield hook.

Returns: Whether the async path should yield now.
