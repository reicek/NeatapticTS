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
 * The emitted expression is THRESHOLD-GATED. At or below
 * {@link IMPLICIT_AND_TOKEN_LIMIT} tokens the tokens are space-joined, which
 * FTS5 reads as an implicit AND over barewords. Above that threshold the
 * tokens are joined with the uppercase FTS5 keyword `OR`, so a long
 * natural-language query no longer demands that every token co-occur in a
 * single chunk - the reach defect this module exists to repair.
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
 *
 * sanitizeFtsQuery('design specification for network activation runtime');
 * // 'design* OR specification* OR for* OR network* OR activation* OR runtime*'
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
// Join strategy
// ---------------------------------------------------------------------------

/**
 * Maximum number of tokens that keeps the implicit-AND space join.
 *
 * FTS5 ANDs barewords implicitly, so a short query still demands that every
 * token co-occurs in a single chunk. Above this threshold the tokens are
 * joined with the uppercase FTS5 keyword `OR` instead, which repairs the
 * reach for long natural-language queries while leaving every short query
 * byte-identical to the space join.
 */
const IMPLICIT_AND_TOKEN_LIMIT = 3;

/** Separator used between tokens inside the implicit-AND band. */
const IMPLICIT_AND_SEPARATOR = ' ';

/** Separator used between tokens above the threshold: space, `OR`, space. */
const OR_JOIN_SEPARATOR = ' OR ';

/**
 * FTS5 reserved keywords that must never be emitted bare.
 *
 * A user token that is exactly one of these is emitted as a quoted phrase so
 * that it keeps its exact-term search semantics and can never be read as a
 * query operator. The comparison is exact and case-sensitive, so `or` stays
 * a plain word and `ANDROID` stays an acronym.
 */
const RESERVED_FTS5_KEYWORDS = new Set(['AND', 'OR', 'NOT']);

/**
 * A formatted FTS5 term and the metadata the joiner needs.
 *
 * @typedef {object} FormattedTerm
 * @property {string} text - The FTS5-safe text of the term.
 * @property {boolean} isReserved - `true` when the term is a quoted reserved keyword.
 */

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
 * Determine whether a token is exactly an FTS5 reserved keyword.
 *
 * The match is exact and case-sensitive, so a lowercase `or` stays a plain
 * word and an acronym such as `ANDROID` is never quoted.
 *
 * @param {string} token - A single whitespace-delimited token.
 * @returns {boolean} `true` when the token is a reserved FTS5 keyword.
 */
function isReservedKeyword(token) {
  return RESERVED_FTS5_KEYWORDS.has(token);
}

/**
 * Format a token for use in an FTS5 MATCH expression.
 *
 * - Reserved FTS5 keywords become quoted phrases (`"OR"`).
 * - Code identifiers become quoted phrases (`"network.activate"`).
 * - Acronyms are emitted as exact terms (`NEAT`).
 * - All other tokens get a prefix wildcard (`selection*`).
 *
 * @param {string} token - A single whitespace-delimited token.
 * @returns {FormattedTerm} The FTS5-safe term and its reserved-keyword flag.
 */
function formatToken(token) {
  if (isReservedKeyword(token)) {
    return { text: `"${token}"`, isReserved: true };
  }
  if (isCodeIdentifier(token)) {
    return { text: `"${token}"`, isReserved: false };
  }
  if (isAcronym(token)) {
    return { text: token, isReserved: false };
  }
  return { text: `${token}*`, isReserved: false };
}

/**
 * Choose the separator that follows a term.
 *
 * A quoted reserved keyword is never separated from its neighbour by the
 * `OR` join operator, so the join operator can never be read as a user
 * token and the keyword keeps its exact-term search semantics.
 *
 * @param {FormattedTerm} term - The term the separator follows.
 * @param {boolean} isLast - `true` when the term is the last one.
 * @param {string} bandSeparator - The separator of the current join band.
 * @returns {string} The separator to emit after the term.
 */
function separatorAfter(term, isLast, bandSeparator) {
  if (isLast) {
    return '';
  }
  return term.isReserved ? IMPLICIT_AND_SEPARATOR : bandSeparator;
}

/**
 * Join the formatted terms into a single FTS5 MATCH expression.
 *
 * At or below {@link IMPLICIT_AND_TOKEN_LIMIT} terms the terms are
 * space-joined, which FTS5 reads as an implicit AND over barewords. Above
 * the threshold the terms are joined with the uppercase FTS5 keyword `OR`,
 * so no single chunk has to carry every token of a long query.
 *
 * @param {FormattedTerm[]} terms - The formatted terms, in query order.
 * @returns {string} The FTS5 MATCH expression.
 */
function joinTerms(terms) {
  const bandSeparator =
    terms.length <= IMPLICIT_AND_TOKEN_LIMIT ? IMPLICIT_AND_SEPARATOR : OR_JOIN_SEPARATOR;
  const lastIndex = terms.length - 1;

  return terms
    .map((term, index) => `${term.text}${separatorAfter(term, index === lastIndex, bandSeparator)}`)
    .join('');
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
 * The join is threshold-gated. At or below {@link IMPLICIT_AND_TOKEN_LIMIT}
 * tokens the expression is space-joined and FTS5 ANDs the barewords
 * implicitly. Above the threshold the tokens are joined with the uppercase
 * FTS5 keyword `OR`, so a long natural-language query no longer requires
 * that every token co-occur in a single chunk.
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
 *
 * sanitizeFtsQuery('alpha beta gamma delta');
 * // 'alpha* OR beta* OR gamma* OR delta*'
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

  return joinTerms(tokens.map(formatToken));
}
