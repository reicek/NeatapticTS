/**
 * BrowserTerminalUtility
 *
 * Minimal drop-in replacement for `TerminalUtility` that works in the browser.
 * It provides a DOM-based "clear screen" and keeps the same evolveUntilSolved API.
 */
import { IEvolutionFunctionResult, IEvolutionStepResult } from './interfaces';

export class BrowserTerminalUtility {
  /**
   * Create a clearer that clears a DOM container's contents.
   * If no container is provided it will try to use an element with id "ascii-maze-output".
   */
  static createTerminalClearer(container?: HTMLElement): () => void {
    const el =
      container ??
      (typeof document !== 'undefined'
        ? document.getElementById('ascii-maze-output')
        : null);
    return () => {
      if (el) el.innerHTML = '';
    };
  }

  /**
   * Same semantics as the Node version: repeatedly call evolveFn until success or threshold reached.
   */
  static async evolveUntilSolved(
    evolveFn: () => Promise<IEvolutionFunctionResult>,
    minProgressToPass: number = 60,
    maxTries: number = 10
  ): Promise<{ finalResult: IEvolutionStepResult; tries: number }> {
    let tries = 0;
    let lastResult: IEvolutionStepResult = {
      success: false,
      progress: 0,
    } as any;
    while (tries < maxTries) {
      tries++;
      const { finalResult } = await evolveFn();
      lastResult = finalResult;
      if (finalResult.success || finalResult.progress >= minProgressToPass) {
        return { finalResult, tries };
      }
    }
    return { finalResult: lastResult, tries };
  }
}
