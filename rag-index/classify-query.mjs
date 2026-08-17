/**
 * @module classify-query
 * @description Rule-based query classifier for Repo Cortex search routing.
 *
 * Classifies free-text queries into six intent classes and computes
 * per-class retrieval parameters (alpha, family filter) so that
 * `search_corpus` can use classification-aware hybrid ranking without
 * requiring the caller to know about classification internals.
 *
 * The classifier is a **pure function**: same query always produces the
 * same classification, no side effects, no database reads.
 *
 * ### Classification Priority
 *
 * ```mermaid
 * flowchart TD
 *   Q[Incoming query] --> L{Length &lt; 5 tokens<br/>and no family hints<br/>and no code identifiers?}
 *   L -- yes --> SL[simple_lookup]
 *   L -- no --> P{Contains plan/doc<br/>hint terms?}
 *   P -- yes --> PS[plan_specific]
 *   P -- no --> I{Contains code<br/>identifier?}
 *   I -- yes --> CS[code_specific]
 *   I -- no --> C{Contains code<br/>hint terms?}
 *   C -- yes --> CS[code_specific]
 *   C -- no --> M{Contains multi-hop<br/>indicators?}
 *   M -- yes --> MH[multi_hop]
 *   M -- no --> X{Contains cross-family<br/>or exploratory hints?}
 *   X -- cross-family --> CB[cross_boundary]
 *   X -- exploratory --> EX[exploratory]
 *   X -- neither --> SL
 * ```
 *
 * @example
 * ```js
 * import { classifyQuery } from './classify-query.mjs';
 *
 * const result = classifyQuery('network.activate');
 * // { query_class: 'code_specific', confidence: 0.85, hints: { family_filter: 'ts-source' } }
 *
 * const result2 = classifyQuery('how does the training pipeline work');
 * // { query_class: 'exploratory', confidence: 0.65, hints: { broad_retrieval: true } }
 * ```
 */

// ---------------------------------------------------------------------------
// Named constants — confidence scores per classification trigger
// ---------------------------------------------------------------------------

/** Minimum token count that triggers the short-query heuristic. */
const SHORT_QUERY_TOKEN_THRESHOLD = 5;

/** Confidence for length-based simple_lookup classification. */
const CONFIDENCE_SHORT_QUERY = 0.9;

/** Confidence for plan-specific keyword detection. */
const CONFIDENCE_PLAN_HINTS = 0.85;

/** Confidence for code-specific keyword detection. */
const CONFIDENCE_CODE_HINTS = 0.8;

/** Confidence for code-identifier detection (dotted, camelCase, snake_case). */
const CONFIDENCE_CODE_IDENTIFIERS = 0.85;

/** Confidence for multi-hop structural indicator detection. */
const CONFIDENCE_MULTI_HOP = 0.75;

/** Confidence for cross-family indicator detection. */
const CONFIDENCE_CROSS_FAMILY = 0.7;

/** Confidence for exploratory keyword detection. */
const CONFIDENCE_EXPLORATORY = 0.65;

/** Confidence for fallback (no pattern matched). */
const CONFIDENCE_FALLBACK = 0.5;

/** Threshold below which the classifier degrades to simple_lookup with α=0.50. */
const CONFIDENCE_DEGRADATION_THRESHOLD = 0.5;

// ---------------------------------------------------------------------------
// Keyword patterns (frozen for determinism)
// ---------------------------------------------------------------------------

/** Plan/design document keywords that signal plan_specific intent. */
const PLAN_KEYWORDS = Object.freeze([
  'plan',
  'design',
  'architecture',
  'roadmap',
  'decision',
  'specification',
]);

/** Code/implementation keywords that signal code_specific intent. */
const CODE_KEYWORDS = Object.freeze([
  'implementation',
  'code for',
  'source of',
  'typescript',
  'implement',
  'function body',
]);

/** Multi-hop connective patterns that signal multi_hop intent. */
const MULTI_HOP_PATTERNS = Object.freeze([
  'that.*also',
  'which also',
  'and then',
  'call.*that',
  'where.*also',
]);

/** Cross-family bridge keywords that signal cross_boundary intent. */
const CROSS_FAMILY_KEYWORDS = Object.freeze([
  'relationship',
  'connection',
  'between',
]);

/** Exploratory keywords that signal exploratory intent. */
const EXPLORATORY_KEYWORDS = Object.freeze([
  'how does',
  'explain',
  'overview',
  'describe',
  'what is',
  'tell me about',
  'how do',
]);

/**
 * Pattern matching code-like identifiers in a query.
 *
 * Captures:
 * - dotted identifiers (`network.activate`)
 * - snake_case identifiers (`snake_case_function`)
 * - file-extension hints (`network.ts`, `code.ts`)
 * - camelCase identifiers (`findTheNEATSelectionCode`)
 *
 * Uses Unicode property escapes so non-ASCII identifiers are supported.
 */
const CODE_IDENTIFIER_PATTERN = new RegExp(
  String.raw`[\p{L}_][\p{L}\p{N}_]*(?:[._][\p{L}_][\p{L}\p{N}_]*)+|` +
    String.raw`[\p{Ll}][\p{Ll}\p{N}]*[\p{Lu}][\p{L}\p{N}]*`,
  'u',
);

// ---------------------------------------------------------------------------
// Embedded alpha and family defaults for lightweight sync classification
// ---------------------------------------------------------------------------
// These values are intentionally identical to the canonical routing table
// in routing-table.mjs. Tests verify they stay in sync.
// ---------------------------------------------------------------------------

/** Per-class alpha defaults embedded for synchronous lookup. */
export const EMBEDDED_ALPHA_DEFAULTS = Object.freeze({
  simple_lookup: 0.75,
  cross_boundary: 0.5,
  multi_hop: 0.35,
  exploratory: 0.3,
  code_specific: 0.7,
  plan_specific: 0.7,
});

/** Per-class family defaults embedded for synchronous lookup. */
export const EMBEDDED_FAMILY_DEFAULTS = Object.freeze({
  simple_lookup: null,
  cross_boundary: null,
  multi_hop: null,
  exploratory: null,
  code_specific: 'ts-source',
  plan_specific: 'plan,completed-plan',
});

// ---------------------------------------------------------------------------
// Pattern detection helpers
// ---------------------------------------------------------------------------

/**
 * Detect plan/design document keywords in a normalized query.
 *
 * Matches any of: plan, design, architecture, roadmap, decision, specification.
 *
 * @param {string} normalizedQuery - Lowercased query string.
 * @returns {boolean} `true` when the query contains a plan-family keyword.
 *
 * @example
 * ```js
 * hasPlanHints('what is the checkpointing design'); // true
 * hasPlanHints('how does crossover work');            // false
 * ```
 */
export function hasPlanHints(normalizedQuery) {
  return PLAN_KEYWORDS.some((keyword) => normalizedQuery.includes(keyword));
}

/**
 * Detect code/implementation keywords in a normalized query.
 *
 * Matches any of: implementation, code for, source of, typescript, implement,
 * function body.
 *
 * @param {string} normalizedQuery - Lowercased query string.
 * @returns {boolean} `true` when the query contains a code-family keyword.
 *
 * @example
 * ```js
 * hasCodeHints('implementation of crossover in NEAT'); // true
 * hasCodeHints('how does crossover work');               // false
 * ```
 */
export function hasCodeHints(normalizedQuery) {
  return CODE_KEYWORDS.some((keyword) => normalizedQuery.includes(keyword));
}

/**
 * Detect code-like identifiers in a raw query.
 *
 * Captures dotted, snake_case, file-extension, and camelCase identifiers
 * such as `network.activate`, `snake_case_function`, `code.ts`, and
 * `findTheNEATSelectionCode`. Short plain words or all-caps acronyms alone
 * are not enough to trigger this detector.
 *
 * @param {string} query - Raw query string.
 * @returns {boolean} `true` when the query contains a code identifier.
 *
 * @example
 * ```js
 * hasCodeIdentifiers('network.activate');        // true
 * hasCodeIdentifiers('snake_case_function');     // true
 * hasCodeIdentifiers('findTheNEATSelectionCode'); // true
 * hasCodeIdentifiers('how does NEAT work');      // false
 * ```
 */
export function hasCodeIdentifiers(query) {
  return CODE_IDENTIFIER_PATTERN.test(query);
}

/**
 * Detect multi-hop structural indicators in a normalized query.
 *
 * Matches connective patterns like "that also", "which also", "and then",
 * or regex patterns like "call.*that" and "where.*also".
 *
 * @param {string} normalizedQuery - Lowercased query string.
 * @returns {boolean} `true` when the query contains a multi-hop indicator.
 *
 * @example
 * ```js
 * hasMultiHopIndicators('functions that call activate and also use slab'); // true
 * hasMultiHopIndicators('what is crossover');                              // false
 * ```
 */
export function hasMultiHopIndicators(normalizedQuery) {
  return MULTI_HOP_PATTERNS.some((pattern) =>
    new RegExp(pattern).test(normalizedQuery),
  );
}

/**
 * Detect cross-family bridge keywords in a normalized query.
 *
 * Matches: relationship, connection, between.
 *
 * @param {string} normalizedQuery - Lowercased query string.
 * @returns {boolean} `true` when the query contains a cross-family indicator.
 *
 * @example
 * ```js
 * hasCrossFamilyIndicators('relationship between crossover and mutation'); // true
 * hasCrossFamilyIndicators('how does crossover work');                       // false
 * ```
 */
export function hasCrossFamilyIndicators(normalizedQuery) {
  return CROSS_FAMILY_KEYWORDS.some((keyword) =>
    normalizedQuery.includes(keyword),
  );
}

/**
 * Detect exploratory keywords in a normalized query.
 *
 * Matches: how does, explain, overview, describe, what is, tell me about, how do.
 *
 * @param {string} normalizedQuery - Lowercased query string.
 * @returns {boolean} `true` when the query contains an exploratory keyword.
 *
 * @example
 * ```js
 * hasExploratoryHints('how does the training pipeline work'); // true
 * hasExploratoryHints('Network.activate');                     // false
 * ```
 */
export function hasExploratoryHints(normalizedQuery) {
  return EXPLORATORY_KEYWORDS.some((keyword) =>
    normalizedQuery.includes(keyword),
  );
}

/**
 * Detect any family-specific pattern hints in a normalized query.
 *
 * Returns `true` when any of the family-specific detection functions
 * (plan, code, multi-hop, cross-family, exploratory) match, which
 * prevents short queries with family hints from falling into
 * `simple_lookup`.
 *
 * @param {string} normalizedQuery - Lowercased query string.
 * @returns {boolean} `true` when any family hint is present.
 *
 * @example
 * ```js
 * hasFamilyHints('what is the checkpointing design'); // true (plan)
 * hasFamilyHints('short api call');                     // false
 * ```
 */
export function hasFamilyHints(normalizedQuery) {
  return (
    hasPlanHints(normalizedQuery) ||
    hasCodeHints(normalizedQuery) ||
    hasMultiHopIndicators(normalizedQuery) ||
    hasCrossFamilyIndicators(normalizedQuery) ||
    hasExploratoryHints(normalizedQuery)
  );
}

// ---------------------------------------------------------------------------
// Main classifier
// ---------------------------------------------------------------------------

/**
 * Classify a query into one of six intent classes with confidence and hints.
 *
 * The classifier is deterministic: the same query always produces the same
 * result. It follows a strict priority order:
 *
 * 1. **Length check** — short queries (< 5 tokens, no family hints, no code
 *    identifiers) → `simple_lookup`
 * 2. **Plan-specific** — plan/design keywords → `plan_specific`
 * 3. **Code-specific** — code identifiers or code/implementation keywords →
 *    `code_specific`
 * 4. **Multi-hop** — connective patterns → `multi_hop`
 * 5. **Cross-boundary** — relationship/connection keywords → `cross_boundary`
 * 6. **Exploratory** — "how does"/"explain" patterns → `exploratory`
 * 7. **Fallback** — no match → `simple_lookup` with confidence 0.50
 *
 * @param {string} query - The raw query string to classify.
 * @returns {{ query_class: string, confidence: number, hints: object }}
 *   Classification result with:
 *   - `query_class`: one of `simple_lookup`, `cross_boundary`, `multi_hop`,
 *     `exploratory`, `code_specific`, `plan_specific`.
 *   - `confidence`: 0–1, how certain the classifier is.
 *   - `hints`: classification hints (family_filter, short_query, multi_hop,
 *     multi_family, broad_retrieval, fallback).
 *
 * @example
 * ```js
 * classifyQuery('network.activate');
 * // { query_class: 'code_specific', confidence: 0.85, hints: { family_filter: 'ts-source' } }
 *
 * classifyQuery('findTheNEATSelectionCode');
 * // { query_class: 'code_specific', confidence: 0.85, hints: { family_filter: 'ts-source' } }
 *
 * classifyQuery('how does the training pipeline work');
 * // { query_class: 'exploratory', confidence: 0.65, hints: { broad_retrieval: true } }
 *
 * classifyQuery('implementation of crossover in NEAT');
 * // { query_class: 'code_specific', confidence: 0.8, hints: { family_filter: 'ts-source' } }
 *
 * classifyQuery('what is the checkpointing design');
 * // { query_class: 'plan_specific', confidence: 0.85, hints: { family_filter: 'plan,completed-plan' } }
 * ```
 */
export function classifyQuery(query) {
  const tokens = query.trim().split(/\s+/);
  const normalizedQuery = query.toLowerCase();

  // Step 1: Length heuristic — short queries with no family hints and no
  // code identifiers are simple lookups
  if (
    tokens.length < SHORT_QUERY_TOKEN_THRESHOLD &&
    !hasFamilyHints(normalizedQuery) &&
    !hasCodeIdentifiers(query)
  ) {
    return {
      query_class: 'simple_lookup',
      confidence: CONFIDENCE_SHORT_QUERY,
      hints: { short_query: true },
    };
  }

  // Step 2: Plan-specific detection
  if (hasPlanHints(normalizedQuery)) {
    return {
      query_class: 'plan_specific',
      confidence: CONFIDENCE_PLAN_HINTS,
      hints: { family_filter: 'plan,completed-plan' },
    };
  }

  // Step 3: Code-specific detection (identifiers or keywords)
  if (hasCodeIdentifiers(query)) {
    return {
      query_class: 'code_specific',
      confidence: CONFIDENCE_CODE_IDENTIFIERS,
      hints: { family_filter: 'ts-source' },
    };
  }

  if (hasCodeHints(normalizedQuery)) {
    return {
      query_class: 'code_specific',
      confidence: CONFIDENCE_CODE_HINTS,
      hints: { family_filter: 'ts-source' },
    };
  }

  // Step 4: Multi-hop detection
  if (hasMultiHopIndicators(normalizedQuery)) {
    return {
      query_class: 'multi_hop',
      confidence: CONFIDENCE_MULTI_HOP,
      hints: { multi_hop: true },
    };
  }

  // Step 5: Cross-boundary detection
  if (hasCrossFamilyIndicators(normalizedQuery)) {
    return {
      query_class: 'cross_boundary',
      confidence: CONFIDENCE_CROSS_FAMILY,
      hints: { multi_family: true },
    };
  }

  // Step 6: Exploratory detection
  if (hasExploratoryHints(normalizedQuery)) {
    return {
      query_class: 'exploratory',
      confidence: CONFIDENCE_EXPLORATORY,
      hints: { broad_retrieval: true },
    };
  }

  // Step 7: Fallback to simple_lookup
  return {
    query_class: 'simple_lookup',
    confidence: CONFIDENCE_FALLBACK,
    hints: { fallback: true },
  };
}

// ---------------------------------------------------------------------------
// Lightweight classification for search_corpus
// ---------------------------------------------------------------------------

/**
 * Compute lightweight classification results for `search_corpus` integration.
 *
 * Returns the per-class alpha and family filter needed by the search pipeline,
 * without requiring the full routing table. Uses embedded defaults that are
 * verified to match the canonical `routing-table.mjs` by tests.
 *
 * When confidence falls below the degradation threshold (0.50), the function
 * degrades to `simple_lookup` with α=0.50 and no family filter — identical
 * to the current fixed-alpha baseline.
 *
 * @param {string} query - The raw query string to classify.
 * @returns {{ alpha: number, family: string | null, query_class: string, confidence: number, classification_fallback: boolean }}
 *   Lightweight classification result with alpha, family filter, and
 *   degradation metadata.
 *
 * @example
 * ```js
 * classifyForSearchCorpus('Network.activate');
 * // { alpha: 0.75, family: null, query_class: 'simple_lookup', confidence: 0.9, classification_fallback: false }
 *
 * classifyForSearchCorpus('how does the training pipeline work');
 * // { alpha: 0.30, family: null, query_class: 'exploratory', confidence: 0.65, classification_fallback: false }
 * ```
 */
export function classifyForSearchCorpus(query) {
  const classification = classifyQuery(query);

  // Graceful degradation: low-confidence queries fall back to balanced hybrid
  /* istanbul ignore next -- defensive: no current confidence value is < 0.5 */
  if (classification.confidence < CONFIDENCE_DEGRADATION_THRESHOLD) {
    return {
      alpha: 0.5,
      family: null,
      query_class: 'simple_lookup',
      confidence: classification.confidence,
      classification_fallback: true,
    };
  }

  /* istanbul ignore next -- defensive: all 6 classes are in the defaults maps */
  const alpha = EMBEDDED_ALPHA_DEFAULTS[classification.query_class] ?? 0.5;
  /* istanbul ignore next -- defensive: all 6 classes are in the defaults maps */
  const family = EMBEDDED_FAMILY_DEFAULTS[classification.query_class] ?? null;

  return {
    alpha,
    family,
    query_class: classification.query_class,
    confidence: classification.confidence,
    classification_fallback: false,
  };
}
