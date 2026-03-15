/**
 * Cache maintenance helpers for NEAT genomes.
 *
 * The root cache chapter stays intentionally small: it points readers to the
 * focused core cache mechanics without leaving cache invalidation logic mixed
 * into the wider `src/neat` monolith.
 *
 * - `core/` explains which per-genome caches exist and how to clear them
 *   safely after structural or weight changes.
 */
export * from './core/cache.constants';
export * from './core/cache.core';
