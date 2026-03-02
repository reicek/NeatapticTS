/** Handle returned by `start` for controlling demo execution lifecycle. */
export interface FlappyBirdRunHandle {
  stop: () => void;
  isRunning: () => boolean;
  done: Promise<void>;
}

/** Runtime window contract for the Flappy Bird browser demo. */
export interface RuntimeWindow extends Window {
  flappyBird?: {
    start?: (container?: string | HTMLElement) => Promise<FlappyBirdRunHandle>;
    _autoStarted?: boolean;
    [key: string]: unknown;
  };
  flappyBirdStart?: (containerElement?: unknown) => unknown;
}
