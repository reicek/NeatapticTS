import { createCanvasHost } from '../host/host';
import type {
  ExampleArchitectureProfile,
  ExampleArchitectureProfileId,
} from '../../../architectureProfiles';
import type {
  FlappyBirdRunHandle,
  RuntimeWindow,
} from '../browser-entry.types';
import type { RuntimeTelemetryState } from './runtime.telemetry.service';
import type { RuntimeArchitectureHistoryByProfileId } from './runtime.architecture-profile.service';

/**
 * Core runtime contracts for the Flappy Bird browser demo.
 *
 * The runtime boundary is where the browser-side application comes together:
 * DOM host setup, worker creation, telemetry wiring, lifecycle control, and the
 * configuration passed into the evolution loop.
 */

/**
 * Public run handle returned by the browser runtime entrypoint.
 *
 * The handle is intentionally minimal so callers can treat the demo like a
 * long-running process with start/stop/status semantics.
 */
export type RuntimeRunHandle = FlappyBirdRunHandle;

/**
 * Browser `window` extension shape used for global runtime wiring.
 *
 * This keeps the auto-start and compatibility globals typed without coupling
 * the runtime modules directly to ad hoc window-property access.
 */
export type RuntimeGlobalWindow = RuntimeWindow;

/**
 * Container argument accepted by the browser runtime start function.
 *
 * Callers can either pass a host element directly or provide an element id for
 * late resolution inside the runtime startup path.
 */
export type RuntimeContainerTarget = string | HTMLElement;

/**
 * Static runtime configuration resolved before the browser loop starts.
 *
 * These values define the high-level NEAT and network shape used for the whole
 * browser session.
 */
export interface RuntimeStartConfig {
  architectureHistoryByProfileId: RuntimeArchitectureHistoryByProfileId;
  availableArchitectureProfiles: ExampleArchitectureProfile[];
  inputSize: number;
  outputSize: number;
  populationSize: number;
  elitismCount: number;
  selectedArchitectureProfile: ExampleArchitectureProfile;
}

/** Optional runtime session inputs used by internal browser startup flows. */
export interface RuntimeStartOptions {
  architectureProfileId?: ExampleArchitectureProfileId;
  onSelectArchitectureProfile?: (
    profileId: ExampleArchitectureProfileId,
  ) => void;
  onResetScores?: () => void;
}

/**
 * Browser host view handles used by the runtime entry orchestration.
 *
 * This is the typed bundle of canvas, HUD, and visualization handles returned
 * by the host layer once the browser DOM has been prepared.
 */
export type RuntimeHostViewContext = ReturnType<typeof createCanvasHost>;

/**
 * Mutable lifecycle state used to coordinate stop semantics and completion.
 *
 * The runtime loop is asynchronous and long-lived, so the browser keeps a small
 * shared lifecycle object for idempotent shutdown and completion signaling.
 */
export interface RuntimeMutableLifecycleState {
  stopped: boolean;
  resolveDone?: () => void;
  done: Promise<void>;
}

/**
 * Shared runtime startup dependencies created before evolution begins.
 *
 * Once this context exists, the browser has everything it needs to launch the
 * actual evolution/playback loop.
 */
export interface RuntimeStartContext {
  config: RuntimeStartConfig;
  hostElement: HTMLElement;
  viewContext: RuntimeHostViewContext;
  runtimeTelemetryState: RuntimeTelemetryState;
  evolutionWorker: Worker;
}

/** Internal legend-preview handle used while a pre-playback canvas card is active. */
export interface RuntimeStartupPreviewHandle {
  complete: () => Promise<void>;
  stop: () => void;
}
