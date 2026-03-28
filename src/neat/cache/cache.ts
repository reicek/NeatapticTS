/**
 * Cache maintenance helpers for NEAT genomes.
 *
 * This chapter explains one of the less glamorous but more important runtime
 * boundaries in NEAT: derived genome caches are useful only while the genome
 * they describe is still unchanged. Activation outputs, compatibility views,
 * and tracing data can all be memoized for speed, but once mutation,
 * crossover, repair, or another structural edit touches the genome, those
 * memoized values stop being evidence and start being stale state.
 *
 * The root cache surface exists to answer three controller-facing questions:
 *
 * 1. which genome-owned fields count as derived caches,
 * 2. when those fields must be invalidated,
 * 3. why one centralized invalidation helper is safer than letting each
 *    mutation path remember its own cleanup list.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   genome[Genome with derived caches]:::base --> edit[Mutation or structural edit]:::accent
 *   edit --> stale[Compatibility, output, and trace caches become stale]:::base
 *   stale --> invalidate[Centralized cache invalidation]:::base
 *   invalidate --> rebuild[Later reads rebuild fresh derived state]:::base
 * ```
 *
 * The deeper teaching point is ownership. Mutation, crossover, repair, and
 * manual graph-edit helpers own structural change, but they should not each own
 * a private opinion about which genome-attached caches are now stale. That is
 * how subtle documentation drift and behavioral drift start: one write path
 * remembers to clear compatibility state, another clears output caches, and a
 * third silently forgets trace artifacts.
 *
 * This root chapter exists to keep that responsibility legible. The write path
 * owns the edit. The cache root owns the invalidation contract that follows the
 * edit. Separating those responsibilities keeps the controller easier to audit,
 * easier to extend, and easier to teach.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   writer[Write-side helper]:::accent --> edit[Change nodes weights or connections]:::base
 *   edit --> owner{Who owns cache cleanup?}:::accent
 *   owner -->|Scattered cleanup| drift[Different paths clear different fields]:::base
 *   owner -->|Shared contract| cacheRoot[cache root invalidation rule]:::base
 *   cacheRoot --> consistent[Every edit path clears the same stale fields]:::base
 * ```
 *
 * The root cache chapter stays intentionally compact because the actual cleanup
 * mechanics are simple. What matters educationally is understanding why the
 * invalidation boundary exists and why every structural edit path should reuse
 * the same cleanup contract. In older evolutionary codebases this boundary is often
 * implicit, because caches accumulate as performance patches rather than as a
 * planned subsystem. Making the boundary explicit here helps readers see that
 * cache invalidation is not incidental housekeeping. It is part of keeping the
 * evolutionary record truthful after a write.
 *
 * - `core/` explains which per-genome caches exist and how to clear them
 *   safely after structural or weight changes.
 *
 * Practical reading order:
 *
 * 1. Start with `GENOME_CACHE_FIELD_KEYS` to see the exact invalidation surface.
 * 2. Read `invalidateGenomeCaches()` to understand the centralized cleanup rule.
 * 3. Continue into `core/` when you want the lower-level stale-field contract
 *    and the concrete deletion mechanics.
 *
 * @example
 * ```ts
 * mutateAddConnReuse(neat, genome);
 * invalidateGenomeCaches(genome);
 * ```
 */
export * from './core/cache.constants';
export * from './core/cache.core';
