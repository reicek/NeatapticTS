/**
 * The evolve-time offspring boundary owns the narrow mechanics of turning
 * selected parents into one structurally valid child.
 *
 * The surrounding `population/` chapter explains how the next generation is
 * assembled through elitism, provenance, and offspring allocation. This smaller
 * chapter answers the next question underneath that orchestration: once a pair
 * of parents has been requested, how does evolve recover from selection failure,
 * annotate lineage metadata, and preserve minimum structural invariants on the
 * resulting child?
 *
 * Read this chapter when you want to understand:
 *
 * - why offspring creation has its own fallback and metadata layer,
 * - how lineage depth defaults behave when parent metadata is incomplete,
 * - why structural cleanup runs immediately after crossover.
 */

/* Module introduction boundary for generated README output. */

/**
 * Index used when falling back to the first genome in the population.
 *
 * The fallback stays explicit so parent selection failure still produces a
 * deterministic recovery path before the helper tries more permissive random
 * rescue.
 */
export const OFFSPRING_FALLBACK_INDEX = 0;

/**
 * Depth increment applied when deriving a child from its parents.
 *
 * Offspring depth is always one generation deeper than the deepest available
 * parent depth so shallow lineage summaries remain monotonic.
 */
export const LINEAGE_DEPTH_INCREMENT = 1;

/**
 * Baseline lineage depth when parent depth metadata is missing.
 *
 * This keeps lineage annotation tolerant of older or narrower runtime surfaces
 * that do not carry full parent-depth metadata.
 */
export const LINEAGE_BASE_DEPTH = 0;
