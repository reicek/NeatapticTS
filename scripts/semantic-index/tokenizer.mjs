/**
 * @module tokenizer
 * @description Query-time tokenization for Repo Cortex BM25 search.
 *
 * Converts raw agent queries into FTS5 MATCH strings that preserve code
 * identifiers (dotted, camelCase, snake_case, file-extension hints) as
 * searchable quoted phrases while applying prefix wildcards to plain
 * English words.
 *
 * FTS5 treats `.`, `-`, `:`, `"` and other punctuation as query operators
 * or syntax markers, so identifiers such as `network.activate` must be
 * emitted as quoted phrases (`"network.activate"`) to avoid syntax errors
 * and over-splitting.
 *
 * @example
 * ```js
 * import { sanitizeFtsQuery } from './tokenizer.mjs';
 *
 * sanitizeFtsQuery('network.activate');
 * // '"network.activate"'
 *
 * sanitizeFtsQuery('NEAT selection code');
 * // 'NEAT selection* code*'
 * ```
 */

// ---------------------------------------------------------------------------
// Identifier detection patterns
// ---------------------------------------------------------------------------

/**
 * Token pattern that captures code-like identifiers and plain word tokens.
 *
 * Alternatives are ordered from most specific to least specific:
 * 1. dotted/snake_case identifiers (`network.activate`, `snake_case_function`)
 * 2. camelCase identifiers starting with a lowercase letter
 * 3. all-caps acronyms (`NEAT`, `API`)
 * 4. plain word tokens
 *
 * Uses Unicode property escapes so non-ASCII letters and digits are treated
 * the same as ASCII word characters, matching the previous cleaning behavior.
 */
const WORD_TOKEN_PATTERN = new RegExp(
  String.raw`[\p{L}_][\p{L}\p{N}_]*(?:[._][\p{L}_][\p{L}\p{N}_]*)+|` +
    String.raw`[\p{Ll}][\p{Ll}\p{N}]*[\p{Lu}][\p{L}\p{N}]*|` +
    String.raw`[\p{Lu}][\p{Lu}\p{N}]+|` +
    String.raw`[\p{L}\p{N}_]+`,
  'gu',
);

/** Matches dotted or underscore-separated identifiers. */
const DOTTED_OR_SNAKE_PATTERN = new RegExp(
  String.raw`^[\p{L}_][\p{L}\p{N}_]*(?:[._][\p{L}_][\p{L}\p{N}_]*)+$`,
  'u',
);

/** Matches camelCase identifiers that start with a lowercase letter. */
const CAMEL_CASE_PATTERN = new RegExp(
  String.raw`^[\p{Ll}][\p{Ll}\p{N}]*[\p{Lu}]`,
  'u',
);

/** Matches all-caps acronyms such as `NEAT` or `API2`. */
const ACRONYM_PATTERN = new RegExp(String.raw`^[\p{Lu}][\p{Lu}\p{N}]+$`, 'u');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Determine whether a token is a code-like identifier.
 *
 * Returns `true` for dotted identifiers (`network.activate`),
 * snake_case identifiers (`snake_case_function`), and camelCase
 * identifiers (`findTheNEATSelectionCode`).
 *
 * @param {string} token - A single whitespace-delimited token.
 * @returns {boolean} `true` when the token should be quoted as a phrase.
 */
function isCodeIdentifier(token) {
  return DOTTED_OR_SNAKE_PATTERN.test(token) || CAMEL_CASE_PATTERN.test(token);
}

/**
 * Determine whether a token is an all-caps acronym.
 *
 * Acronyms are kept as exact terms (no wildcard suffix) because prefix
 * expansion rarely improves acronym search quality.
 *
 * @param {string} token - A single whitespace-delimited token.
 * @returns {boolean} `true` when the token is an acronym.
 */
function isAcronym(token) {
  return ACRONYM_PATTERN.test(token);
}

/**
 * Format a token for use in an FTS5 MATCH expression.
 *
 * - Code identifiers become quoted phrases (`"network.activate"`).
 * - Acronyms are emitted as exact terms (`NEAT`).
 * - All other tokens get a prefix wildcard (`selection*`).
 *
 * @param {string} token - A single whitespace-delimited token.
 * @returns {string} FTS5-safe token string.
 */
function formatToken(token) {
  if (isCodeIdentifier(token)) return `"${token}"`;
  if (isAcronym(token)) return token;
  return `${token}*`;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Sanitize a raw query into an FTS5 MATCH expression.
 *
 * Code identifiers are preserved as quoted phrases so the BM25 index can
 * match them exactly. Plain words receive a trailing `*` for prefix search.
 * All punctuation and symbols that are not part of a code identifier are
 * treated as separators.
 *
 * @param {string} raw - Raw user or agent query.
 * @returns {string} FTS5-safe query string, or an empty string when no tokens remain.
 *
 * @example
 * ```js
 * sanitizeFtsQuery('network.activate');
 * // '"network.activate"'
 *
 * sanitizeFtsQuery('findTheNEATSelectionCode');
 * // '"findTheNEATSelectionCode"'
 *
 * sanitizeFtsQuery('snake_case_function');
 * // '"snake_case_function"'
 *
 * sanitizeFtsQuery('NEAT selection code');
 * // 'NEAT selection* code*'
 * ```
 */
export function sanitizeFtsQuery(raw) {
  const input = String(raw ?? '').trim();
  if (input.length === 0) {
    return '';
  }

  const tokens = input.match(WORD_TOKEN_PATTERN) ?? [];
  if (tokens.length === 0) {
    return '';
  }

  return tokens.map(formatToken).join(' ');
}
