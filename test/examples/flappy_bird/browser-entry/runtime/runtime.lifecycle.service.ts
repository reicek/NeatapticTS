import { disconnectRuntimeTelemetry } from './runtime.telemetry.service';
import type {
  RuntimeMutableLifecycleState,
  RuntimeRunHandle,
  RuntimeStartContext,
} from './runtime.types';

/**
 * Creates mutable lifecycle state for stop semantics and completion signaling.
 *
 * @returns Mutable lifecycle state used by the run handle.
 */
export function createRuntimeLifecycleState(): RuntimeMutableLifecycleState {
  let resolveDone: (() => void) | undefined;
  const done = new Promise<void>((resolve) => {
    resolveDone = resolve;
  });

  // Step 1: Return idempotent lifecycle state with an externally resolvable promise.
  return {
    stopped: false,
    resolveDone,
    done,
  };
}

/**
 * Builds the public run handle and binds it to runtime teardown behavior.
 *
 * @param runtimeStartContext - Shared runtime start context.
 * @param runtimeLifecycleState - Mutable lifecycle state for stop semantics.
 * @returns Public run handle exposed to callers.
 */
export function createRuntimeRunHandle(
  runtimeStartContext: RuntimeStartContext,
  runtimeLifecycleState: RuntimeMutableLifecycleState,
): RuntimeRunHandle {
  /**
   * Stops the running demo and performs idempotent resource teardown.
   *
   * @returns Nothing.
   */
  const stop = () => {
    // Step 1: Keep stop idempotent for repeated calls from multiple paths.
    if (runtimeLifecycleState.stopped) {
      return;
    }

    // Step 2: Flip the loop guard before releasing runtime resources.
    runtimeLifecycleState.stopped = true;

    // Step 3: Disconnect telemetry, notify the worker, and terminate it.
    disconnectRuntimeTelemetry(runtimeStartContext.runtimeTelemetryState);
    runtimeStartContext.evolutionWorker.postMessage({ type: 'stop' });
    runtimeStartContext.evolutionWorker.terminate();

    // Step 4: Resolve the externally visible completion promise.
    runtimeLifecycleState.resolveDone?.();
  };

  /**
   * Reports whether the runtime loop has been stopped.
   *
   * @returns `true` when the runtime has been stopped.
   */
  const isStopped = () => runtimeLifecycleState.stopped;

  /**
   * Reports whether the runtime loop is currently active.
   *
   * @returns `true` while the runtime remains active.
   */
  const isRunning = () => !isStopped();

  // Step 1: Return the public lifecycle handle bound to the mutable state.
  return {
    stop,
    isRunning,
    done: runtimeLifecycleState.done,
  };
}
