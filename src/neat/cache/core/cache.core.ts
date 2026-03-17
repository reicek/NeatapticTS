import { GENOME_CACHE_FIELD_KEYS } from './cache.constants';

/**
 * Cache invalidation mechanics for genome-local derived state.
 *
 * This file is the small execution layer beneath the cache-core chapter. The
 * surrounding constant defines which fields are disposable caches; this helper
 * applies that contract uniformly whenever a genome has just been edited.
 *
 * The intent is deliberately modest: do not infer meaning from any cache field,
 * do not try to repair cached values in place, and do not make individual write
 * paths remember their own bespoke cleanup list. Once a genome changes, this
 * helper erases the derived fields so later reads must rebuild from the new
 * canonical structure.
 */

/**
 * Invalidate the derived caches attached to a genome candidate.
 *
 * Mutation, crossover, repair, and other genome-editing helpers attach or rely
 * on memoized compatibility, activation, and trace data directly on genome
 * objects for speed. That optimization only works when every write path also
 * respects the invalidation boundary. Once the genome changes, those memoized
 * values are stale and must be removed before any later read assumes they still
 * describe the current structure.
 *
 * Centralizing the cleanup here avoids a fragile situation where each edit path
 * remembers a slightly different subset of cache keys. One helper and one key
 * list keeps invalidation deterministic across the controller. Read it as the
 * "final broom" after a write: the structural edit owns the real behavior
 * change, while this helper only removes the stale evidence that no longer
 * matches the updated genome.
 *
 * The cleanup path stays intentionally simple:
 *
 * 1. ignore non-object inputs,
 * 2. treat the remaining value as a genome-shaped record,
 * 3. delete every field named by `GENOME_CACHE_FIELD_KEYS`.
 *
 * That simplicity is part of the design. The helper should be safe to call from
 * many write paths, even when some genomes do not currently carry every cached
 * field.
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
