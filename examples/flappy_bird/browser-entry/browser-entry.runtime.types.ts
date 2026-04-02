/**
 * Public lifecycle contracts for the Flappy Bird browser runtime.
 *
 * These types describe how the demo is started, stopped, and exposed on the
 * browser `window` object. They are intentionally small because callers should
 * control the demo at a high level without depending on private implementation
 * details.
 */

/** Handle returned by `start` for controlling demo execution lifecycle. */
export interface FlappyBirdRunHandle {
  stop: () => void;
  isRunning: () => boolean;
  done: Promise<void>;
}

/**
 * Runtime window contract for the Flappy Bird browser demo.
 *
 * The demo exposes a small debug-friendly surface on `window` so manual browser
 * experiments and docs examples can start the simulation without importing the
 * bundle as a module.
 */
export interface RuntimeWindow extends Window {
  flappyBird?: {
    start?: (container?: string | HTMLElement) => Promise<FlappyBirdRunHandle>;
    _autoStarted?: boolean;
    [key: string]: unknown;
  };
  flappyBirdStart?: (containerElement?: unknown) => unknown;
}
