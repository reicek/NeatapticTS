/**
 * Invalidate per-genome caches used across compatibility and forward-pass logic.
 *
 * @param genomeCandidate - Genome object whose caches should be cleared.
 */
export function invalidateGenomeCaches(genomeCandidate: unknown): void {
  if (!genomeCandidate || typeof genomeCandidate !== 'object') return;
  const genomeObject = genomeCandidate as Record<string, unknown>;
  delete genomeObject._compatCache;
  delete genomeObject._outputCache;
  delete genomeObject._traceCache;
}
