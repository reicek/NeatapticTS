/**
 * Raised when crowding helpers cannot resolve a genome back to its source index.
 */
export class MultiobjectiveCrowdingGenomeIndexResolutionError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'MultiobjectiveCrowdingGenomeIndexResolutionError';
  }
}
