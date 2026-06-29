/**
 * @module routing-table
 * @description Per-class routing table for query classification and retrieval strategy.
 *
 * Provides the canonical routing configuration that maps each query class to
 * its default alpha, family filter, expansion strategy, and post-processing
 * strategy. Also exports `classifyAndRoute` which combines classification
 * with routing lookup for full-featured use cases (e.g. `search_advanced`).
 *
 * The routing table is the single source of truth for per-class defaults.
 * `classify-query.mjs` embeds a copy of the alpha and family values for
 * its synchronous `classifyForSearchCorpus` function; tests verify the two
 * copies stay in sync.
 *
 * @example
 * ```js
 * import { classifyAndRoute, DEFAULTS, ROUTING } from './routing-table.mjs';
 *
 * const result = classifyAndRoute('how does NEAT evolve use speciation');
 * // { query_class: 'cross_boundary', confidence: 0.7, alpha: 0.5,
 * //   strategy: { family: null, expansion: 'multi_family', post_processing: 'cross_family_dedup' } }
 *
 * const simple = classifyAndRoute('Network.activate');
 * // { query_class: 'simple_lookup', confidence: 0.9, alpha: 0.75,
 * //   strategy: { family: null, expansion: 'none', post_processing: 'default' } }
 * ```
 */

import { classifyQuery } from './classify-query.mjs';

// ---------------------------------------------------------------------------
// Per-class alpha defaults (canonical source of truth)
// ---------------------------------------------------------------------------

/**
 * Per-class BM25/dense blend weight defaults.
 *
 * Higher alpha = more BM25 weight (keyword-heavy).
 * Lower alpha = more dense weight (semantic-heavy).
 *
 * | Class | Alpha | Rationale |
 * |---|---|---|
 * | `simple_lookup` | 0.75 | BM25-heavy for exact symbol matches |
 * | `cross_boundary` | 0.50 | Balanced hybrid for cross-family queries |
 * | `multi_hop` | 0.35 | Dense-heavy for semantic starting points |
 * | `exploratory` | 0.30 | Dense-heavy for conceptual overview |
 * | `code_specific` | 0.70 | BM25-heavy with source-family filter |
 * | `plan_specific` | 0.70 | BM25-heavy with plan-family filter |
 */
export const DEFAULTS = Object.freeze({
  simple_lookup: 0.75,
  cross_boundary: 0.5,
  multi_hop: 0.35,
  exploratory: 0.3,
  code_specific: 0.7,
  plan_specific: 0.7,
});

// ---------------------------------------------------------------------------
// Per-class routing strategies (canonical source of truth)
// ---------------------------------------------------------------------------

/**
 * Per-class routing strategies controlling family filter, result expansion,
 * and post-processing behavior.
 *
 * | Class | Family | Expansion | Post-processing |
 * |---|---|---|---|
 * | `simple_lookup` | null (all) | none | default top-K |
 * | `cross_boundary` | null (multi-family) | multi_family | cross_family_dedup |
 * | `multi_hop` | null (all) | entity_graph | hop_decay |
 * | `exploratory` | null (all) | context_assembly | budget_assembly |
 * | `code_specific` | `ts-source` | none | default top-K |
 * | `plan_specific` | `plan,completed-plan` | none | default top-K |
 */
export const ROUTING = Object.freeze({
  simple_lookup: Object.freeze({
    family: null,
    expansion: 'none',
    post_processing: 'default',
  }),
  cross_boundary: Object.freeze({
    family: null,
    expansion: 'multi_family',
    post_processing: 'cross_family_dedup',
  }),
  multi_hop: Object.freeze({
    family: null,
    expansion: 'entity_graph',
    post_processing: 'hop_decay',
  }),
  exploratory: Object.freeze({
    family: null,
    expansion: 'context_assembly',
    post_processing: 'budget_assembly',
  }),
  code_specific: Object.freeze({
    family: 'ts-source',
    expansion: 'none',
    post_processing: 'default',
  }),
  plan_specific: Object.freeze({
    family: 'plan,completed-plan',
    expansion: 'none',
    post_processing: 'default',
  }),
});

// ---------------------------------------------------------------------------
// Full classification + routing
// ---------------------------------------------------------------------------

/**
 * Classify a query and compute the optimal retrieval strategy.
 *
 * Combines `classifyQuery` with the routing table lookup and applies
 * optional caller overrides for alpha and family. When the caller
 * provides explicit `classification_hints.alpha` or
 * `classification_hints.family`, those values take precedence over the
 * per-class defaults.
 *
 * @param {string} query - The raw query string to classify and route.
 * @param {{ alpha?: number, family?: string }} [classification_hints] - Optional caller overrides.
 * @returns {{ query_class: string, confidence: number, alpha: number, strategy: { family: string | null, expansion: string, post_processing: string } }}
 *   Full classification and routing result.
 *
 * @example
 * ```js
 * classifyAndRoute('how does NEAT evolve use speciation');
 * // { query_class: 'cross_boundary', confidence: 0.7, alpha: 0.5,
 * //   strategy: { family: null, expansion: 'multi_family', post_processing: 'cross_family_dedup' } }
 *
 * classifyAndRoute('Network.activate', { alpha: 0.9 });
 * // { query_class: 'simple_lookup', confidence: 0.9, alpha: 0.9,
 * //   strategy: { family: null, expansion: 'none', post_processing: 'default' } }
 *
 * classifyAndRoute('crossover code', { family: 'ts-source' });
 * // { query_class: 'simple_lookup', confidence: 0.5, alpha: 0.75,
 * //   strategy: { family: 'ts-source', expansion: 'none', post_processing: 'default' } }
 * ```
 */
export function classifyAndRoute(query, classification_hints) {
  const classification = classifyQuery(query);

  // Compute alpha: per-class default, overridden by caller if provided
  const alpha =
    classification_hints?.alpha ?? DEFAULTS[classification.query_class] ?? 0.5;

  // Build strategy from routing table, overridden by caller if provided
  const baseStrategy = ROUTING[classification.query_class] ?? {
    family: null,
    expansion: 'none',
    post_processing: 'default',
  };
  const strategy = {
    ...baseStrategy,
    ...(classification_hints?.family !== undefined
      ? { family: classification_hints.family }
      : {}),
  };

  return {
    query_class: classification.query_class,
    confidence: classification.confidence,
    alpha,
    strategy,
  };
}
