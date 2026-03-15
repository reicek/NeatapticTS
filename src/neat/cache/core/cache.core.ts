import { GENOME_CACHE_FIELD_KEYS } from './cache.constants';

/**
 * Invalidate the derived caches attached to a genome candidate.
 *
 * Mutation and crossover helpers attach memoized compatibility, activation,
 * and trace data directly onto genome objects for speed. Once the genome
 * changes, those values become stale. This helper centralizes that cleanup so
 * every mutation path clears the same cache fields.
 *
 * @param genomeCandidate - Genome-shaped value whose attached caches should be cleared.
 * @returns Nothing. The helper mutates the candidate in place when it is an object.
 *
 * @example
 * ```ts
 * const genome = { _compatCache: {}, _outputCache: [1, 0] };
 * invalidateGenomeCaches(genome);
 * console.log('_compatCache' in genome, '_outputCache' in genome);
 * ```
 */
export function invalidateGenomeCaches(genomeCandidate: unknown): void {
  if (!genomeCandidate || typeof genomeCandidate !== 'object') {
    return;
  }

  const genomeObject = genomeCandidate as Record<string, unknown>;

  for (const cacheFieldKey of GENOME_CACHE_FIELD_KEYS) {
    delete genomeObject[cacheFieldKey];
  }
}
