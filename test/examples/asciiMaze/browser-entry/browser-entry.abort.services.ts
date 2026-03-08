import type {
  RuntimeAbortSignal,
  RuntimeAbortSignalConstructor,
} from './browser-entry.types';

/**
 * Browser abort-composition service boundary for the ASCII Maze browser entry.
 *
 * This leaf module isolates runtime-safe signal composition so browser-entry
 * orchestration can stay focused on lifecycle flow instead of platform quirks.
 */

/**
 * Compose an internal and external abort signal into one cooperative signal.
 *
 * @param internalController - Internal controller owned by the browser run handle.
 * @param externalSignal - Optional caller-provided signal.
 * @returns A signal that aborts when either source aborts.
 */
export const composeBrowserEntryAbortSignal = (
  internalController: AbortController,
  externalSignal?: AbortSignal,
): AbortSignal => {
  if (!externalSignal) {
    return internalController.signal;
  }

  const runtimeExternalSignal = externalSignal as RuntimeAbortSignal;
  const internalSignal = internalController.signal;

  switch (true) {
    case !!runtimeExternalSignal.aborted:
      return runtimeExternalSignal as unknown as AbortSignal;

    case typeof (AbortSignal as unknown as RuntimeAbortSignalConstructor)
      .any === 'function': {
      try {
        return (AbortSignal as unknown as RuntimeAbortSignalConstructor).any!([
          runtimeExternalSignal as unknown as AbortSignal,
          internalSignal,
        ]);
      } catch {
        // Fall through to manual wiring when native composition is unavailable at runtime.
      }
      break;
    }

    default:
      break;
  }

  try {
    runtimeExternalSignal.addEventListener(
      'abort',
      () => {
        try {
          internalController.abort();
        } catch {
          // Ignore duplicate or unsupported aborts.
        }
      },
      { once: true },
    );
  } catch {
    // Ignore event wiring failures in minimal DOM environments.
  }

  try {
    runtimeExternalSignal.onabort = () => {
      try {
        internalController.abort();
      } catch {
        // Ignore duplicate or unsupported aborts.
      }
    };
  } catch {
    // Ignore environments that do not allow onabort assignment.
  }

  try {
    queueMicrotask(() => {
      if (runtimeExternalSignal.aborted) {
        try {
          internalController.abort();
        } catch {
          // Ignore duplicate or unsupported aborts.
        }
      }
    });
  } catch {
    // Ignore environments that do not expose queueMicrotask.
  }

  return internalSignal;
};
