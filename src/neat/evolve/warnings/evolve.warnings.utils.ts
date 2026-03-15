/** Warning emitted when evolution finishes without a best genome. */
export const EVOLVE_NO_BEST_GENOME_WARNING =
  'Evolution completed without finding a valid best genome (no fitness improvements recorded).';

/**
 * Emit the standard warning for runs that end without a valid best genome.
 */
export function warnIfNoBestGenome(): void {
  try {
    console.warn(EVOLVE_NO_BEST_GENOME_WARNING);
  } catch {
    // Silent failure: console may be unavailable in restricted environments.
  }
}
