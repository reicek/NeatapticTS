import { DEFAULT_CONTAINER_ID } from '../../constants/constants';
import { FLAPPY_HUD_ZERO_TEXT } from '../../constants/constants';
import { resolveExampleArchitectureProfile } from '../../../architectureProfiles';
import { updateStatsTableValues } from '../host/host';
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
import type {
  RuntimeContainerTarget,
  RuntimeMutableLifecycleState,
  RuntimeRunHandle,
  RuntimeStartContext,
  RuntimeStartOptions,
} from './runtime.types';

export type { RuntimeRunHandle as FlappyBirdRunHandle } from './runtime.types';

/**
 * Top-level browser runtime facade for the Flappy Bird example.
 *
 * This is the narrowest public browser entry with real behavior behind it.
 * Calling `start` resolves the DOM host, installs the worker-backed runtime,
 * initializes the HUD, and launches the evolve-to-playback loop that powers the
 * demo. The surrounding services keep those responsibilities split, but this
 * file is the place where they are folded back into one lifecycle.
 *
 * The boundary matters because the browser demo is more than a canvas render:
 * it is host setup, worker orchestration, telemetry, HUD updates, and shutdown
 * control presented as one small API.
 */

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
  return startRuntimeSession(container, {});
}

/**
 * Starts one Flappy browser runtime session with optional internal profile selection state.
 *
 * The public `start(...)` API always enters through this helper. Internal
 * browser-UI restarts reuse it so architecture button clicks can launch a fresh
 * worker-backed run without widening the public API surface.
 *
 * @param container - Element id or HTMLElement to host the demo.
 * @param runtimeStartOptions - Internal per-session runtime inputs.
 * @returns Run handle for the started browser session.
 */
async function startRuntimeSession(
  container: RuntimeContainerTarget,
  runtimeStartOptions: RuntimeStartOptions,
): Promise<RuntimeRunHandle> {
  const runtimeLifecycleState: RuntimeMutableLifecycleState =
    createRuntimeLifecycleState();
  let queuedRestartProfileId: RuntimeStartOptions['architectureProfileId'];

  // Step 1: Resolve the runtime view, worker, telemetry, and static config.
  const runtimeStartContext: RuntimeStartContext = createRuntimeStartContext(
    container,
    {
    ...runtimeStartOptions,
    onSelectArchitectureProfile: (profileId): void => {
      if (
        queuedRestartProfileId ||
        runtimeLifecycleState?.stopped ||
        !runtimeRunHandle
      ) {
        return;
      }

      queuedRestartProfileId = profileId;
      runtimeStartContext.viewContext.architectureSelectorController.setDisabled(
        true,
      );

      const nextArchitectureProfile = resolveExampleArchitectureProfile(
        'flappy-bird',
        profileId,
      );
      updateStatsTableValues(runtimeStartContext.viewContext.statsValueByKey, {
        status: 'restarting',
        currentArchitecture: nextArchitectureProfile.label,
        summaryArchitecture: nextArchitectureProfile.label,
        birds: `${FLAPPY_HUD_ZERO_TEXT}/${runtimeStartContext.config.populationSize}`,
      });

      runtimeRunHandle.stop();
      void runtimeRunHandle.done.then(() => {
        void startRuntimeSession(runtimeStartContext.hostElement, {
          architectureProfileId: profileId,
        }).catch(() => undefined);
      });
    },
    },
  );

  // Step 2: Paint the initial HUD state before evolution bootstrapping starts.
  initializeRuntimeHud(runtimeStartContext);

  // Step 3: Create lifecycle state and externally exposed run-handle methods.
  const runtimeRunHandle: RuntimeRunHandle = createRuntimeRunHandle(
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
