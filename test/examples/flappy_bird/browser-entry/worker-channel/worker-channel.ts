/**
 * Public worker-channel facade for the Flappy Bird browser runtime.
 *
 * This boundary wraps the lower-level message protocol in a smaller browser API:
 * create the worker, request a generation result, or request the next playback
 * step. The point is to keep the rest of the UI code thinking in terms of
 * intent rather than raw `postMessage` plumbing.
 */
export {
  createEvolutionWorker,
  requestWorkerGeneration,
  requestWorkerPlaybackStep,
} from '../browser-entry.worker-channel.utils';
