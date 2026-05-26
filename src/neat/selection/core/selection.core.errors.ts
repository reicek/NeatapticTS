/**
 * Raised when tournament selection is asked to sample more entries than exist.
 */
export class SelectionTournamentOverflowError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'SelectionTournamentOverflowError';
  }
}
