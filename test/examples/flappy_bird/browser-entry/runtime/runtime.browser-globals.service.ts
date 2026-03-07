import type {
  RuntimeContainerTarget,
  RuntimeGlobalWindow,
  RuntimeRunHandle,
} from './runtime.types';

/**
 * Runtime start function signature used by browser-global wiring.
 */
export type RuntimeStartFunction = (
  container?: RuntimeContainerTarget,
) => Promise<RuntimeRunHandle>;

/**
 * Publishes browser globals for demo auto-start and host-driven control.
 *
 * This keeps parity with the asciiMaze entry style:
 * - `window.flappyBird.start(...)` for explicit invocation,
 * - `window.flappyBirdStart(...)` for compatibility,
 * - one guarded auto-start for standalone HTML usage.
 *
 * @param startRuntime - Runtime entry function.
 * @returns Nothing.
 */
export function installRuntimeBrowserGlobals(
  startRuntime: RuntimeStartFunction,
): void {
  try {
    // Step 1: Resolve runtime window shape used by the browser demo host.
    const runtimeWindow = window as unknown as RuntimeGlobalWindow;

    // Step 2: Publish canonical entry points.
    runtimeWindow.flappyBird = runtimeWindow.flappyBird ?? {};
    runtimeWindow.flappyBird.start = startRuntime;
    runtimeWindow.flappyBirdStart = (containerElement?: unknown) =>
      startRuntime(containerElement as RuntimeContainerTarget);

    // Step 3: Auto-start exactly once when loaded in standalone pages.
    if (!runtimeWindow.flappyBird._autoStarted) {
      runtimeWindow.flappyBird._autoStarted = true;
      setTimeout(() => {
        try {
          startRuntime().catch(() => undefined);
        } catch {
          // ignore
        }
      }, 20);
    }
  } catch {
    // ignore global wiring failures (non-browser environments).
  }
}
