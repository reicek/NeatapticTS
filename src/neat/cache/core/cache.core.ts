import { GENOME_CACHE_FIELD_KEYS } from './cache.constants';

/**
 * Cache invalidation mechanics for genome-local derived state.
 *
 * Read this file as the execution half of the cache-core chapter. The constant
 * beside it names the genome-owned fields that count as disposable derived
 * state; this helper is the single write-side cleanup step that enforces that
 * contract after mutation, crossover, repair, or direct genome edits.
 *
 * The boundary is intentionally conservative. Cached compatibility summaries,
 * activation outputs, and trace artifacts belong to read-side acceleration, not
 * to the genome's canonical structure. Once a write changes nodes, connections,
 * or weights, those cached values stop being trustworthy evidence about the
 * genome's current state.
 *
 * That is why the helper stays narrow instead of trying to be clever: it does
 * not inspect cache contents, selectively preserve "probably still valid"
 * entries, or make each editing path remember its own cleanup list. It simply
 * erases the known stale fields so the next read must rebuild them from the
 * updated genome.
 *
 * Practical reading order:
 *
 * 1. Start with `GENOME_CACHE_FIELD_KEYS` to see which genome-owned fields are
 *    treated as disposable caches.
 * 2. Read `invalidateGenomeCaches()` as the shared broom every write path can
 *    call after changing the genome.
 * 3. Move back to the parent `cache/` chapter when you want the broader
 *    controller-facing explanation for why centralized invalidation is safer
 *    than bespoke cleanup scattered across mutation helpers.
 */

/**
 * Invalidate the derived caches attached to a genome candidate.
 *
 * Mutation, crossover, repair, and other genome-editing helpers attach or rely
 * on memoized compatibility, activation, and trace data directly on genome
 * objects for speed. That optimization only works when every write path also
 * respects the invalidation boundary. Once the genome changes, those memoized
 * values are stale and must be removed before a later read assumes they still
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
 * field. In practice, that means mutation flows, crossover assembly, repair
 * passes, and manual graph-edit utilities can all reuse the same final cleanup
 * contract instead of maintaining subtly different invalidation rules.
 *
 * @param genomeCandidate - Genome-shaped value whose attached caches should be cleared.
 * @returns Nothing. The helper mutates the candidate in place when it is an object.
 *
 * @example
 * ```ts
 * const genome = {
 *   _compatCache: { neighbor: 0.42 },
 *   _outputCache: [1, 0],
 *   connections: [{ from: 0, to: 1, weight: 0.9 }],
 * };
 *
 * genome.connections[0].weight = 1.1;
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
