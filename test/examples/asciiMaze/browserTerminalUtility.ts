/**
 * BrowserTerminalUtility
 *
 * Provides minimal DOM-backed helpers mirroring a subset of the Node-oriented
 * terminal utility used by the ASCII maze evolutionary demo. It keeps the
 * educational surface identical across environments while avoiding heavy
 * dependencies in the browser bundle.
 *
 * @remarks Methods are intentionally small and synchronous (except for the async
 * evolution loop) to remain readable in tutorials. No internal pooling is
 * required given the tiny allocation footprint. Complexity of
 * {@link evolveUntilSolved} is O(A) with A = number of attempts.
 */

import { IEvolutionFunctionResult, IEvolutionStepResult } from './interfaces';

export class BrowserTerminalUtility {
  /** Default minimum progress percentage that counts as sufficiently solved. */
  static #DefaultMinProgressToPass = 60;
  /** Default maximum number of evolutionary attempts before aborting. */
  static #DefaultMaxAttemptCount = 10;
  /** Backwards compatibility alias for attempt count (deprecated). */
  static readonly deprecatedTriesKey = 'tries' as const;

  /** Resolve (or locate) the host element used for output. */
  static #resolveHostElement(container?: HTMLElement): HTMLElement | null {
    return (
      container ??
      (typeof document !== 'undefined'
        ? document.getElementById('ascii-maze-output')
        : null)
    );
  }
  /**
   * Create a clearer that clears a DOM container's contents.
   * If no container is provided it will try to use an element with id "ascii-maze-output".
   */
  static createTerminalClearer(container?: HTMLElement): () => void {
    const hostElement = this.#resolveHostElement(container);
    return () => {
      if (hostElement) hostElement.innerHTML = '';
    };
  }

  /**
   * Same semantics as the Node version: repeatedly call evolveFn until success or threshold reached.
   */
  static async evolveUntilSolved(
    evolveFn: () => Promise<IEvolutionFunctionResult>,
    minProgressToPass: number = BrowserTerminalUtility
      .#DefaultMinProgressToPass,
    maxAttemptCount: number = BrowserTerminalUtility.#DefaultMaxAttemptCount
  ): Promise<{
    finalResult: IEvolutionStepResult;
    attemptCount: number;
    /** @deprecated Use attemptCount instead. */
    tries: number;
  }> {
    // Step 1: Initialize attempt counter & placeholder result.
    let attemptCount = 0;
    let lastResult: IEvolutionStepResult = {
      success: false,
      progress: 0,
    } as IEvolutionStepResult;

    // Step 2: Evolution loop (bounded by maxAttemptCount).
    while (attemptCount < maxAttemptCount) {
      attemptCount++;
      const { finalResult } = await evolveFn();
      lastResult = finalResult;
      // Step 3: Success or sufficient progress short‑circuit.
      if (finalResult.success || finalResult.progress >= minProgressToPass) {
        return { finalResult, attemptCount, tries: attemptCount };
      }
    }

    // Step 4: Exhausted attempts – return last observed result.
    return { finalResult: lastResult, attemptCount, tries: attemptCount };
  }
}
