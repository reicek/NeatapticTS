import { DEFAULT_CONTAINER_ID } from '../../constants/constants';
import { installRuntimeBrowserGlobals } from './runtime.browser-globals.service';
import { launchRuntimeEvolution } from './runtime.evolution-launch.service';
import {
  createRuntimeLifecycleState,
  createRuntimeRunHandle,
} from './runtime.lifecycle.service';
import {
  createRuntimeStartContext,
  initializeRuntimeHud,
} from './runtime.startup.service';
import type { RuntimeContainerTarget, RuntimeRunHandle } from './runtime.types';

export type { RuntimeRunHandle as FlappyBirdRunHandle } from './runtime.types';

/**
 * Starts the Flappy Bird NeatapticTS browser demo and returns lifecycle controls.
 *
 * This function is intentionally orchestration-focused:
 * 1) resolve runtime dependencies (DOM host, worker, host UI),
 * 2) initialize worker and telemetry plumbing,
 * 3) run the evolve -> playback -> HUD fold loop until stopped,
 * 4) expose a small stop/isRunning/done handle for callers.
 *
 * @param container - Element id or HTMLElement to host the demo.
 * @returns Run handle for stop/state control.
 * @example
 * ```ts
 * const runHandle = await start('flappy-bird-output');
 * // later
 * runHandle.stop();
 * await runHandle.done;
 * ```
 */
export async function start(
  container: RuntimeContainerTarget = DEFAULT_CONTAINER_ID,
): Promise<RuntimeRunHandle> {
  // Step 1: Resolve the runtime view, worker, telemetry, and static config.
  const runtimeStartContext = createRuntimeStartContext(container);

  // Step 2: Paint the initial HUD state before evolution bootstrapping starts.
  initializeRuntimeHud(runtimeStartContext);

  // Step 3: Create lifecycle state and externally exposed run-handle methods.
  const runtimeLifecycleState = createRuntimeLifecycleState();
  const runtimeRunHandle = createRuntimeRunHandle(
    runtimeStartContext,
    runtimeLifecycleState,
  );

  // Step 4: Launch the runtime evolution loop with error-to-HUD routing.
  launchRuntimeEvolution(
    runtimeStartContext,
    runtimeLifecycleState,
    runtimeRunHandle.stop,
  );

  // Step 5: Return the public lifecycle controls to the caller.
  return runtimeRunHandle;
}

installRuntimeBrowserGlobals(start);
