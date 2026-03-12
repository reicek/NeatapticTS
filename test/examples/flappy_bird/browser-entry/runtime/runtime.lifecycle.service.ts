import { disconnectRuntimeTelemetry } from './runtime.telemetry.service';
import type {
  RuntimeMutableLifecycleState,
  RuntimeRunHandle,
  RuntimeStartContext,
} from './runtime.types';

/**
 * Lifecycle and teardown helpers for the browser runtime.
 *
 * The runtime behaves like a small application process. These helpers create the
 * mutable state and public handle needed to stop it cleanly, terminate the
 * worker, and resolve the completion promise exactly once.
 */

/**
 * Creates mutable lifecycle state for stop semantics and completion signaling.
 *
 * The returned object is shared across runtime orchestration paths so both
 * expected shutdown and failure-driven shutdown follow the same completion flow.
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
 * The handle is the user-facing control surface for the demo. Internally it is
 * just a thin closure layer over the mutable lifecycle state and startup
 * context.
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
