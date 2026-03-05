import type {
  FlappyBirdRunHandle,
  RuntimeWindow,
} from '../browser-entry.types';

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
