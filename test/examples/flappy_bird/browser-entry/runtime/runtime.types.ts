import { createCanvasHost } from '../host/host';
import type {
  FlappyBirdRunHandle,
  RuntimeWindow,
} from '../browser-entry.types';
import type { RuntimeTelemetryState } from './runtime.telemetry.service';

/**
 * Public run handle returned by the browser runtime entrypoint.
 */
export type RuntimeRunHandle = FlappyBirdRunHandle;

/**
 * Browser `window` extension shape used for global runtime wiring.
 */
export type RuntimeGlobalWindow = RuntimeWindow;

/**
 * Container argument accepted by the browser runtime start function.
 */
export type RuntimeContainerTarget = string | HTMLElement;

/**
 * Static runtime configuration resolved before the browser loop starts.
 */
export interface RuntimeStartConfig {
  inputSize: number;
  outputSize: number;
  populationSize: number;
  elitismCount: number;
}

/**
 * Browser host view handles used by the runtime entry orchestration.
 */
export type RuntimeHostViewContext = ReturnType<typeof createCanvasHost>;

/**
 * Mutable lifecycle state used to coordinate stop semantics and completion.
 */
export interface RuntimeMutableLifecycleState {
  stopped: boolean;
  resolveDone?: () => void;
  done: Promise<void>;
}

/**
 * Shared runtime startup dependencies created before evolution begins.
 */
export interface RuntimeStartContext {
  config: RuntimeStartConfig;
  viewContext: RuntimeHostViewContext;
  runtimeTelemetryState: RuntimeTelemetryState;
  evolutionWorker: Worker;
}
