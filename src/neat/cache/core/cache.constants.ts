/**
 * Genome-owned cache fields that should be cleared when a mutation changes
 * structure or outputs.
 *
 * These cache keys are grouped here so the cache chapter documents the
 * invalidation surface in one place.
 */
export const GENOME_CACHE_FIELD_KEYS = [
  '_compatCache',
  '_outputCache',
  '_traceCache',
] as const;
