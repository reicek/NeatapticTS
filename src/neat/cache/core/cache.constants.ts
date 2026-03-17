/**
 * Lower-level invalidation mechanics for genome-owned NEAT caches.
 *
 * The root cache chapter explains why centralized invalidation matters. This
 * core chapter explains the narrower mechanics contract beneath that story:
 * which derived genome fields are treated as disposable caches, why that list
 * stays explicit, and how one shared cleanup helper keeps every structural edit
 * path aligned.
 *
 * Read this layer when you want the operational answer to "what exactly becomes
 * stale after a write?" Compatibility views, activation outputs, and trace
 * artifacts may all be cached directly on genomes for speed, but none of those
 * fields are authoritative after mutation, crossover, repair, or manual graph
 * edits. The core surface keeps that rule small and reviewable.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   write[Structural or weight write]:::accent --> stale[Genome-owned caches become stale]:::base
 *   stale --> keys[Explicit invalidation key list]:::base
 *   keys --> clear[Shared deletion helper]:::base
 *   clear --> rebuild[Later reads rebuild fresh derived state]:::base
 * ```
 *
 * Practical reading order:
 *
 * 1. Start with `GENOME_CACHE_FIELD_KEYS` to see the exact stale-field surface.
 * 2. Continue into `cache.core.ts` to see how the cleanup helper applies that
 *    contract safely.
 * 3. Move back to the root `cache/` chapter when you want the broader
 *    controller-facing explanation for why every edit path should reuse this
 *    same invalidation rule.
 */

// Cache-core symbol docs begin below.

/**
 * Genome-owned cache fields that should be cleared when a mutation changes
 * structure or outputs.
 *
 * Treat this list as the mechanical invalidation contract for genome objects.
 * Each key names a field that may be cheap to rebuild but dangerous to trust
 * after a write:
 *
 * - `_compatCache` stores derived compatibility-comparison views,
 * - `_outputCache` stores memoized activation outputs,
 * - `_traceCache` stores debugging or tracing artifacts.
 *
 * Keeping the list explicit makes review easier. When a new genome-owned cache
 * is introduced, adding it here makes the invalidation surface visible instead
 * of relying on scattered ad hoc cleanup.
 */
export const GENOME_CACHE_FIELD_KEYS = [
  '_compatCache',
  '_outputCache',
  '_traceCache',
] as const;
