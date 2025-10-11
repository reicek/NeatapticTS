// Lightweight polling helper prototype (Phase 5 -> Phase 6 preparation)
// Resolves when predicate returns a truthy value (or explicitly true).
// Rejects on timeout or abort signal.

export interface PollUntilOptions {
  intervalMs?: number;
  timeoutMs?: number;
  signal?: AbortSignal;
}

export class PollUntilAbortError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AbortError';
  }
}

export const pollUntil = async <T>(
  predicate: () => T | Promise<T>,
  { intervalMs = 25, timeoutMs = 2000, signal }: PollUntilOptions = {}
): Promise<T> => {
  if (signal?.aborted) {
    throw new PollUntilAbortError('Operation aborted before start');
  }

  const start = performance.now();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let abortListener: (() => void) | undefined;

  return new Promise<T>((resolve, reject) => {
    const cleanup = () => {
      if (timer) clearTimeout(timer);
      if (signal && abortListener)
        signal.removeEventListener('abort', abortListener);
    };

    const tick = async (): Promise<void> => {
      try {
        const result = await predicate();
        if (result) {
          cleanup();
          return resolve(result);
        }
      } catch (error: unknown) {
        cleanup();
        return reject(error);
      }

      if (performance.now() - start >= timeoutMs) {
        cleanup();
        return reject(new PollUntilAbortError('pollUntil timeout exceeded'));
      }

      timer = setTimeout(tick, intervalMs);
    };

    if (signal) {
      abortListener = () => {
        cleanup();
        reject(new PollUntilAbortError('pollUntil aborted'));
      };
      signal.addEventListener('abort', abortListener);
    }

    timer = setTimeout(tick, 0);
  });
};
