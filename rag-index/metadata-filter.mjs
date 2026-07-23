/**
 * @module metadata-filter
 * @description Structured metadata filter grammar for the Repo Cortex search pipeline.
 *
 * Validates, compiles, and applies filter predicates against chunk metadata columns.
 * The filter grammar supports 14 predicate types with boolean composition (AND, OR, NOT),
 * depth-limited nesting (max 10), predicate count limits (max 50), and full parameterized
 * SQL compilation to prevent injection.
 *
 * ### Predicate Types
 *
 * | Op | Fields | Description |
 * |---|---|---|
 * | `eq` | all | Equality: field = value |
 * | `neq` | all | Inequality: field ≠ value |
 * | `in` | all | Set membership: field IN [values] |
 * | `not_in` | all | Set exclusion: field NOT IN [values] |
 * | `gt` | numeric | Greater than: field > value |
 * | `gte` | numeric | Greater than or equal: field ≥ value |
 * | `lt` | numeric | Less than: field < value |
 * | `lte` | numeric | Less than or equal: field ≤ value |
 * | `like` | all | SQL LIKE: field LIKE pattern |
 * | `is_null` | all | Null check: field IS NULL |
 * | `is_not_null` | all | Not-null check: field IS NOT NULL |
 * | `and` | — | Conjunction: AND [predicates] |
 * | `or` | — | Disjunction: OR [predicates] |
 * | `not` | — | Negation: NOT predicate |
 *
 * @example
 * // Validate and compile a filter
 * import { validateFilter, compileFilterToSql } from './metadata-filter.mjs';
 *
 * const filter = { op: 'and', predicates: [
 *   { op: 'eq', field: 'family', value: 'ts-source' },
 *   { op: 'eq', field: 'arch_layer', value: 'network' },
 * ]};
 * validateFilter(filter);
 * const { sql, params } = compileFilterToSql(filter);
 * // sql = '(chunks.doc_id IN (SELECT doc_id FROM documents WHERE doc_family = ?) AND chunks.arch_layer = ?)'
 * // params = ['ts-source', 'network']
 */

/** Maximum nesting depth for filter predicates. */
const MAX_FILTER_DEPTH = 10;

/** Maximum number of predicates in a single filter tree. */
const MAX_PREDICATE_COUNT = 50;

/** Maximum number of values in an `in` or `not_in` predicate. */
const MAX_IN_VALUES = 100;

/**
 * All filterable field names.
 *
 * String fields can use `eq`, `neq`, `in`, `not_in`, `like`, `is_null`, `is_not_null`.
 * Numeric fields can additionally use `gt`, `gte`, `lt`, `lte`.
 *
 * @type {ReadonlySet<string>}
 */
const VALID_FIELDS = new Set([
  'family',
  'export_type',
  'arch_layer',
  'jsdoc_quality',
  'test_coverage',
  'module_path',
  'source_path_pattern',
  'symbol_name',
  'file_path',
  'jsdoc_word_count',
  'cyclomatic_complexity',
  'depth',
  'char_start',
  'char_end',
  'slice_id',
  'step_number',
]);

/**
 * Fields that require numeric values for comparison operators.
 *
 * Numeric fields support `gt`, `gte`, `lt`, `lte` in addition to the
 * string-compatible operators. String fields only support `eq`, `neq`,
 * `in`, `not_in`, `like`, `is_null`, `is_not_null`.
 *
 * @type {ReadonlySet<string>}
 */
const NUMERIC_FIELDS = new Set([
  'jsdoc_word_count',
  'cyclomatic_complexity',
  'depth',
  'char_start',
  'char_end',
  'step_number',
]);

/**
 * Enum-validated string fields and their allowed values.
 *
 * When a field appears in this map, its value must be one of the listed
 * strings (or `null` for `eq`/`neq`). Fields not in this map accept any
 * string value.
 *
 * @type {ReadonlyMap<string, string[]>}
 */
const VALID_STRING_VALUES = new Map([
  ['jsdoc_quality', ['none', 'weak', 'adequate', 'good']],
  ['test_coverage', ['full', 'partial', 'none', 'unknown']],
  [
    'export_type',
    ['function', 'class', 'interface', 'type', 'variable', 'reexport'],
  ],
]);

/**
 * Mapping from filter field names to the SQL column they reference.
 *
 * Most fields map directly to a column on the `chunks` table, but `family`
 * and `file_path` require a JOIN through `documents`. The SQL compiler
 * uses this mapping to determine whether a predicate targets a chunk column
 * or a document column.
 *
 * @type {ReadonlyMap<string, { table: 'chunks' | 'documents', column: string }>}
 */
const FIELD_TO_SQL = new Map([
  ['family', { table: 'documents', column: 'doc_family' }],
  ['file_path', { table: 'documents', column: 'file_path' }],
  ['export_type', { table: 'chunks', column: 'export_type' }],
  ['arch_layer', { table: 'chunks', column: 'arch_layer' }],
  ['jsdoc_quality', { table: 'chunks', column: 'jsdoc_quality' }],
  ['jsdoc_word_count', { table: 'chunks', column: 'jsdoc_word_count' }],
  [
    'cyclomatic_complexity',
    { table: 'chunks', column: 'cyclomatic_complexity' },
  ],
  ['test_coverage', { table: 'chunks', column: 'test_coverage' }],
  ['module_path', { table: 'chunks', column: 'module_path' }],
  ['source_path_pattern', { table: 'chunks', column: 'source_path_pattern' }],
  ['symbol_name', { table: 'chunks', column: 'symbol_name' }],
  ['depth', { table: 'chunks', column: 'depth' }],
  ['char_start', { table: 'chunks', column: 'char_start' }],
  ['char_end', { table: 'chunks', column: 'char_end' }],
  ['slice_id', { table: 'chunks', column: 'slice_id' }],
  ['step_number', { table: 'chunks', column: 'step_number' }],
]);

/**
 * Error class for filter validation and compilation failures.
 *
 * Thrown by {@link validateFilter} when a predicate tree violates the grammar
 * rules (invalid field, wrong value type, excessive nesting, unknown operator).
 *
 * @example
 * try {
 *   validateFilter({ op: 'eq', field: 'invalid', value: 'x' });
 * } catch (error) {
 *   console.error(error.message); // "Invalid field: invalid"
 * }
 */
export class FilterError extends Error {
  /**
   * @param {string} message - Human-readable description of the validation failure.
   */
  constructor(message) {
    super(message);
    this.name = 'FilterError';
  }
}

/**
 * Validate a filter predicate tree against the grammar rules.
 *
 * Checks that every field name is in the allowed set, every operator is
 * recognized, numeric fields receive numeric values, enum-validated fields
 * receive allowed values, `LIKE` patterns contain only safe characters, and
 * the tree does not exceed depth or predicate count limits.
 *
 * @param {object} predicate - The filter predicate tree root.
 * @param {number} [depth=0] - Current nesting depth (internal use for recursion).
 * @param {{ count: number }} [counter] - Predicate counter (internal use for limit enforcement).
 * @throws {FilterError} When the predicate violates any grammar rule.
 *
 * @example
 * validateFilter({ op: 'eq', field: 'arch_layer', value: 'network' }); // passes
 * validateFilter({ op: 'gt', field: 'jsdoc_word_count', value: 10 });  // passes
 * validateFilter({ op: 'eq', field: 'invalid', value: 'x' });          // throws FilterError
 */
export function validateFilter(predicate, depth = 0, counter = { count: 0 }) {
  if (depth > MAX_FILTER_DEPTH) {
    throw new FilterError(
      `Filter nesting depth exceeds maximum (${MAX_FILTER_DEPTH})`,
    );
  }

  counter.count += 1;
  if (counter.count > MAX_PREDICATE_COUNT) {
    throw new FilterError(
      `Filter exceeds maximum predicate count (${MAX_PREDICATE_COUNT})`,
    );
  }

  if (!predicate || typeof predicate !== 'object') {
    throw new FilterError('Predicate must be a non-null object');
  }

  switch (predicate.op) {
    case 'eq':
    case 'neq':
      validateFieldPredicate(predicate, depth, counter);
      break;
    case 'in':
    case 'not_in':
      validateInPredicate(predicate, depth, counter);
      break;
    case 'gt':
    case 'gte':
    case 'lt':
    case 'lte':
      validateRangePredicate(predicate);
      break;
    case 'like':
      validateLikePredicate(predicate);
      break;
    case 'is_null':
    case 'is_not_null':
      validateNullPredicate(predicate);
      break;
    case 'and':
    case 'or':
      validateBooleanPredicate(predicate, depth, counter);
      break;
    case 'not':
      validateNotPredicate(predicate, depth, counter);
      break;
    default:
      throw new FilterError(`Unknown operator: ${predicate.op}`);
  }
}

/**
 * Validate an equality/inequality predicate (`eq` or `neq`).
 *
 * @param {object} predicate - The predicate to validate.
 * @throws {FilterError} When the field is invalid or the value type mismatches.
 */
function validateFieldPredicate(predicate) {
  if (!VALID_FIELDS.has(predicate.field)) {
    throw new FilterError(`Invalid field: ${predicate.field}`);
  }
  if (
    NUMERIC_FIELDS.has(predicate.field) &&
    predicate.value !== null &&
    typeof predicate.value !== 'number'
  ) {
    throw new FilterError(`Field ${predicate.field} requires a numeric value`);
  }
  validateEnumValue(predicate.field, predicate.value);
}

/**
 * Validate an `in` or `not_in` predicate.
 *
 * @param {object} predicate - The predicate to validate.
 * @throws {FilterError} When the field is invalid, values array is too long, or enum values are wrong.
 */
function validateInPredicate(predicate) {
  if (!VALID_FIELDS.has(predicate.field)) {
    throw new FilterError(`Invalid field: ${predicate.field}`);
  }
  if (!Array.isArray(predicate.values)) {
    throw new FilterError(`${predicate.op} requires a values array`);
  }
  if (predicate.values.length > MAX_IN_VALUES) {
    throw new FilterError(
      `Too many values in ${predicate.op} (max ${MAX_IN_VALUES})`,
    );
  }
  if (predicate.values.length === 0) {
    throw new FilterError(`${predicate.op} requires at least one value`);
  }
  for (const value of predicate.values) {
    if (NUMERIC_FIELDS.has(predicate.field) && typeof value !== 'number') {
      throw new FilterError(`Field ${predicate.field} requires numeric values`);
    }
    validateEnumValue(predicate.field, value);
  }
}

/**
 * Validate a range predicate (`gt`, `gte`, `lt`, `lte`).
 *
 * @param {object} predicate - The predicate to validate.
 * @throws {FilterError} When the field does not support range operators or value is not numeric.
 */
function validateRangePredicate(predicate) {
  if (!NUMERIC_FIELDS.has(predicate.field)) {
    throw new FilterError(
      `Field ${predicate.field} does not support range operators`,
    );
  }
  if (typeof predicate.value !== 'number') {
    throw new FilterError('Range value must be a number');
  }
}

/**
 * Validate a `like` predicate.
 *
 * LIKE patterns may only contain `%`, `_`, and alphanumeric characters.
 * This prevents regex-like wildcards and injection through LIKE.
 *
 * @param {object} predicate - The predicate to validate.
 * @throws {FilterError} When the field is invalid or the pattern contains disallowed characters.
 */
function validateLikePredicate(predicate) {
  if (!VALID_FIELDS.has(predicate.field)) {
    throw new FilterError(`Invalid field: ${predicate.field}`);
  }
  if (typeof predicate.value !== 'string') {
    throw new FilterError('LIKE value must be a string');
  }
  // Allow only alphanumeric, %, _, /, ., - in LIKE patterns
  if (/[^\w%/_.\-]/u.test(predicate.value)) {
    throw new FilterError(
      'LIKE pattern contains invalid characters (only alphanumeric, %, _, /, ., - are allowed)',
    );
  }
}

/**
 * Validate an `is_null` or `is_not_null` predicate.
 *
 * @param {object} predicate - The predicate to validate.
 * @throws {FilterError} When the field is invalid.
 */
function validateNullPredicate(predicate) {
  if (!VALID_FIELDS.has(predicate.field)) {
    throw new FilterError(`Invalid field: ${predicate.field}`);
  }
}

/**
 * Validate an `and` or `or` boolean predicate.
 *
 * @param {object} predicate - The predicate to validate.
 * @param {number} depth - Current nesting depth.
 * @param {{ count: number }} counter - Predicate counter.
 * @throws {FilterError} When the predicates array is too short or too long.
 */
function validateBooleanPredicate(predicate, depth, counter) {
  if (!Array.isArray(predicate.predicates) || predicate.predicates.length < 2) {
    throw new FilterError(`${predicate.op} requires at least 2 predicates`);
  }
  if (predicate.predicates.length > MAX_PREDICATE_COUNT) {
    throw new FilterError(
      `${predicate.op} exceeds maximum ${MAX_PREDICATE_COUNT} predicates`,
    );
  }
  for (const child of predicate.predicates) {
    validateFilter(child, depth + 1, counter);
  }
}

/**
 * Validate a `not` predicate.
 *
 * @param {object} predicate - The predicate to validate.
 * @param {number} depth - Current nesting depth.
 * @param {{ count: number }} counter - Predicate counter.
 * @throws {FilterError} When the inner predicate is invalid.
 */
function validateNotPredicate(predicate, depth, counter) {
  if (!predicate.predicate || typeof predicate.predicate !== 'object') {
    throw new FilterError('not predicate requires an inner predicate');
  }
  validateFilter(predicate.predicate, depth + 1, counter);
}

/**
 * Validate that a value is an allowed enum value for an enum-validated field.
 *
 * @param {string} field - The field name.
 * @param {string | number | null} value - The value to check.
 * @throws {FilterError} When the value is not in the field's allowed set.
 */
function validateEnumValue(field, value) {
  const allowedValues = VALID_STRING_VALUES.get(field);
  if (!allowedValues || value === null || value === undefined) return;
  if (!allowedValues.includes(value)) {
    throw new FilterError(
      `Field ${field} value must be one of: ${allowedValues.join(', ')}`,
    );
  }
}

/**
 * Compile a validated filter predicate tree to a parameterized SQL WHERE clause.
 *
 * Returns a `{ sql, params }` object where `sql` is a WHERE clause fragment
 * (without the `WHERE` keyword) and `params` is an array of parameter values
 * for `?` placeholders. All values are parameterized to prevent SQL injection.
 *
 * For predicates targeting document-level columns (`family`, `file_path`),
 * the compiler generates subquery expressions that join through the `documents`
 * table. For chunk-level columns, it references `chunks.<column>` directly.
 *
 * @param {object} predicate - A validated filter predicate tree.
 * @returns {{ sql: string, params: Array<string | number | null> }} Compiled SQL and parameters.
 * @throws {FilterError} When the predicate contains an unknown operator.
 *
 * @example
 * const result = compileFilterToSql({ op: 'eq', field: 'arch_layer', value: 'network' });
 * // result.sql = 'chunks.arch_layer = ?'
 * // result.params = ['network']
 */
export function compileFilterToSql(predicate) {
  const params = [];
  const sql = compilePredicate(predicate, params);
  return { sql, params };
}

/**
 * Recursively compile a single predicate into a SQL expression.
 *
 * @param {object} predicate - The predicate to compile.
 * @param {Array<string | number | null>} params - Accumulator for parameter values.
 * @returns {string} SQL expression fragment.
 * @throws {FilterError} When the operator is unknown.
 */
function compilePredicate(predicate, params) {
  switch (predicate.op) {
    case 'eq':
      return compileComparison(predicate, '=', params);
    case 'neq':
      return compileComparison(predicate, '!=', params);
    case 'in':
      return compileIn(predicate, params);
    case 'not_in':
      return compileNotIn(predicate, params);
    case 'gt':
      return compileComparison(predicate, '>', params);
    case 'gte':
      return compileComparison(predicate, '>=', params);
    case 'lt':
      return compileComparison(predicate, '<', params);
    case 'lte':
      return compileComparison(predicate, '<=', params);
    case 'like':
      return compileLike(predicate, params);
    case 'is_null':
      return `${fieldRef(predicate.field)} IS NULL`;
    case 'is_not_null':
      return `${fieldRef(predicate.field)} IS NOT NULL`;
    case 'and':
      return `(${predicate.predicates.map((p) => compilePredicate(p, params)).join(' AND ')})`;
    case 'or':
      return `(${predicate.predicates.map((p) => compilePredicate(p, params)).join(' OR ')})`;
    case 'not':
      return `NOT (${compilePredicate(predicate.predicate, params)})`;
    default:
      throw new FilterError(`Unknown operator: ${predicate.op}`);
  }
}

/**
 * Compile an equality/inequality or range comparison predicate.
 *
 * Handles both chunk-level and document-level field references.
 * Document-level fields (`family`, `file_path`) use a subquery against
 * the `documents` table; all other fields reference `chunks.<column>`.
 *
 * @param {object} predicate - The comparison predicate.
 * @param {string} operator - SQL comparison operator (`=`, `!=`, `>`, `>=`, `<`, `<=`).
 * @param {Array} params - Accumulator for parameter values.
 * @returns {string} SQL comparison expression.
 */
function compileComparison(predicate, operator, params) {
  const ref = fieldRef(predicate.field);
  if (predicate.value === null) {
    return operator === '=' ? `${ref} IS NULL` : `${ref} IS NOT NULL`;
  }
  params.push(predicate.value);
  return `${ref} ${operator} ?`;
}

/**
 * Compile an `in` predicate to a parameterized `IN (?, ?, …)` expression.
 *
 * @param {object} predicate - The `in` predicate.
 * @param {Array} params - Accumulator for parameter values.
 * @returns {string} SQL `IN` expression.
 */
function compileIn(predicate, params) {
  const ref = fieldRef(predicate.field);
  for (const value of predicate.values) {
    params.push(value);
  }
  const placeholders = predicate.values.map(() => '?').join(', ');
  return `${ref} IN (${placeholders})`;
}

/**
 * Compile a `not_in` predicate to a parameterized `NOT IN (?, ?, …)` expression.
 *
 * @param {object} predicate - The `not_in` predicate.
 * @param {Array} params - Accumulator for parameter values.
 * @returns {string} SQL `NOT IN` expression.
 */
function compileNotIn(predicate, params) {
  const ref = fieldRef(predicate.field);
  for (const value of predicate.values) {
    params.push(value);
  }
  const placeholders = predicate.values.map(() => '?').join(', ');
  return `${ref} NOT IN (${placeholders})`;
}

/**
 * Compile a `like` predicate to a parameterized `LIKE ?` expression.
 *
 * @param {object} predicate - The `like` predicate.
 * @param {Array} params - Accumulator for parameter values.
 * @returns {string} SQL `LIKE` expression.
 */
function compileLike(predicate, params) {
  const ref = fieldRef(predicate.field);
  params.push(predicate.value);
  return `${ref} LIKE ?`;
}

/**
 * Resolve a filter field name to its SQL column reference.
 *
 * For document-level fields (`family`, `file_path`), returns a qualified
 * column reference like `documents.doc_family` or `documents.file_path`.
 * This requires the caller to have JOINed the `documents` table (as `d`
 * or unaliased). For chunk-level fields, references `chunks.<column>` directly.
 *
 * @param {string} field - The filter grammar field name.
 * @returns {string} SQL column reference expression.
 */
function fieldRef(field) {
  const mapping = FIELD_TO_SQL.get(field);
  if (!mapping) return `chunks.${field}`;
  if (mapping.table === 'documents') {
    return `documents.${mapping.column}`;
  }
  return `${mapping.table}.${mapping.column}`;
}

/**
 * Apply a filter predicate tree to an in-memory candidate pool.
 *
 * Used for dense-search post-retrieval filtering where SQL-level filtering
 * is not available. Evaluates each candidate against the predicate tree and
 * returns only matching candidates.
 *
 * Candidate objects must have properties matching the filter field names
 * (e.g. `family`, `arch_layer`, `jsdoc_quality`, etc.). The `family` field
 * is mapped from the `doc_family` property of the candidate if present.
 *
 * @param {Array<Record<string, unknown>>} candidates - Candidate result objects.
 * @param {object} predicate - A validated filter predicate tree.
 * @returns {Array<Record<string, unknown>>} Filtered candidates matching the predicate.
 *
 * @example
 * const filtered = applyPostRetrievalFilter(candidates, {
 *   op: 'and',
 *   predicates: [
 *     { op: 'eq', field: 'arch_layer', value: 'network' },
 *     { op: 'eq', field: 'export_type', value: 'class' },
 *   ],
 * });
 */
export function applyPostRetrievalFilter(candidates, predicate) {
  return candidates.filter((candidate) =>
    evaluatePredicate(candidate, predicate),
  );
}

/**
 * Evaluate a single predicate against a candidate object.
 *
 * Maps candidate property names to filter field names, handling the
 * `family` → `doc_family` mapping for document-level fields.
 *
 * @param {Record<string, unknown>} candidate - A single result candidate.
 * @param {object} predicate - The predicate to evaluate.
 * @returns {boolean} Whether the candidate matches the predicate.
 */
function evaluatePredicate(candidate, predicate) {
  switch (predicate.op) {
    case 'eq':
      return evaluateEq(candidate, predicate);
    case 'neq':
      return !evaluateEq(candidate, predicate);
    case 'in':
      return predicate.values.includes(
        getFieldValue(candidate, predicate.field),
      );
    case 'not_in':
      return !predicate.values.includes(
        getFieldValue(candidate, predicate.field),
      );
    case 'gt':
      return (
        Number(getFieldValue(candidate, predicate.field)) > predicate.value
      );
    case 'gte':
      return (
        Number(getFieldValue(candidate, predicate.field)) >= predicate.value
      );
    case 'lt':
      return (
        Number(getFieldValue(candidate, predicate.field)) < predicate.value
      );
    case 'lte':
      return (
        Number(getFieldValue(candidate, predicate.field)) <= predicate.value
      );
    case 'like':
      return evaluateLike(candidate, predicate);
    case 'is_null':
      return (
        getFieldValue(candidate, predicate.field) === null ||
        getFieldValue(candidate, predicate.field) === undefined
      );
    case 'is_not_null':
      return (
        getFieldValue(candidate, predicate.field) !== null &&
        getFieldValue(candidate, predicate.field) !== undefined
      );
    case 'and':
      return predicate.predicates.every((p) => evaluatePredicate(candidate, p));
    case 'or':
      return predicate.predicates.some((p) => evaluatePredicate(candidate, p));
    case 'not':
      return !evaluatePredicate(candidate, predicate.predicate);
    default:
      return false;
  }
}

/**
 * Evaluate an equality predicate against a candidate.
 *
 * @param {Record<string, unknown>} candidate - A result candidate.
 * @param {object} predicate - The `eq` predicate.
 * @returns {boolean} Whether the candidate's field value equals the predicate value.
 */
function evaluateEq(candidate, predicate) {
  const fieldValue = getFieldValue(candidate, predicate.field);
  if (predicate.value === null)
    return fieldValue === null || fieldValue === undefined;
  return fieldValue === predicate.value;
}

/**
 * Evaluate a LIKE predicate against a candidate using SQL-style pattern matching.
 *
 * Converts the SQL LIKE pattern (`%` = any sequence, `_` = single character) to
 * a JavaScript RegExp for in-memory evaluation.
 *
 * @param {Record<string, unknown>} candidate - A result candidate.
 * @param {object} predicate - The `like` predicate.
 * @returns {boolean} Whether the candidate's field value matches the pattern.
 */
function evaluateLike(candidate, predicate) {
  const fieldValue = String(getFieldValue(candidate, predicate.field) ?? '');
  // Convert SQL LIKE pattern to RegExp: % → .*, _ → .
  const regexPattern = predicate.value
    .replace(/[.+^${}()|[\]\\]/g, '\\$&')
    .replace(/%/g, '.*')
    .replace(/_/g, '.');
  return new RegExp(`^${regexPattern}$`).test(fieldValue);
}

/**
 * Get a field value from a candidate object, mapping filter field names to
 * candidate property names.
 *
 * The `family` filter field maps to the `family` candidate property (which
 * comes from `doc_family` in the database row). Other fields map directly.
 *
 * @param {Record<string, unknown>} candidate - A result candidate.
 * @param {string} field - Filter grammar field name.
 * @returns {unknown} The field value from the candidate.
 */
function getFieldValue(candidate, field) {
  // The `family` filter field maps to the `family` property in the result
  // (which was populated from `doc_family` in readChunkRow)
  if (field === 'family') {
    return candidate.family ?? candidate.doc_family ?? null;
  }
  if (field === 'file_path') {
    return candidate.file_path ?? null;
  }
  // Metadata fields may come from the `metadata` sub-object or direct properties
  const metadata = candidate.metadata ?? candidate;
  return metadata[field] ?? candidate[field] ?? null;
}

/**
 * Compile a filter predicate to a SQL WHERE clause fragment for BM25 queries.
 *
 * This is the primary entry point for BM25 search integration. It generates
 * a WHERE clause that can be appended to the FTS5 query's existing conditions.
 * Document-level predicates (`family`, `file_path`) target the `documents` table
 * directly (since the BM25 query already JOINs documents). Chunk-level predicates
 * target the `chunks` table.
 *
 * The returned `sql` fragment does **not** include the `WHERE` keyword — it is
 * meant to be appended to existing WHERE conditions with `AND`.
 *
 * @param {object} predicate - A validated filter predicate tree.
 * @returns {{ sql: string, params: Array<string | number | null> }} SQL fragment and parameters.
 *
 * @example
 * // BM25 query with family + arch_layer filter
 * const { sql, params } = compileFilterToSql({
 *   op: 'and',
 *   predicates: [
 *     { op: 'eq', field: 'family', value: 'ts-source' },
 *     { op: 'eq', field: 'arch_layer', value: 'network' },
 *   ],
 * });
 * // sql = '(documents.doc_family = ? AND chunks.arch_layer = ?)'
 * // params = ['ts-source', 'network']
 */
// compileFilterToSql is already exported as a named export at line 374.
// No re-export needed — the function is available for both direct import and BM25 integration.

/**
 * Compile a filter predicate for BM25 queries where the `documents` table
 * is already JOINed as `d` and the `chunks` table as `c`.
 *
 * Uses short table aliases (`d` for documents, `c` for chunks) to match
 * the BM25 query pattern in `runBm25Search`.
 *
 * @param {object} predicate - A validated filter predicate tree.
 * @returns {{ sql: string, params: Array<string | number | null> }} SQL fragment and parameters.
 */
export function compileFilterToSqlAliased(predicate) {
  const params = [];
  const sql = compilePredicateAliased(predicate, params);
  return { sql, params };
}

/**
 * Recursively compile a predicate using short table aliases.
 *
 * @param {object} predicate - The predicate to compile.
 * @param {Array} params - Accumulator for parameter values.
 * @returns {string} SQL expression with `d.` and `c.` aliases.
 */
function compilePredicateAliased(predicate, params) {
  switch (predicate.op) {
    case 'eq':
      return compileComparisonAliased(predicate, '=', params);
    case 'neq':
      return compileComparisonAliased(predicate, '!=', params);
    case 'in':
      return compileInAliased(predicate, params);
    case 'not_in':
      return compileNotInAliased(predicate, params);
    case 'gt':
      return compileComparisonAliased(predicate, '>', params);
    case 'gte':
      return compileComparisonAliased(predicate, '>=', params);
    case 'lt':
      return compileComparisonAliased(predicate, '<', params);
    case 'lte':
      return compileComparisonAliased(predicate, '<=', params);
    case 'like':
      return compileLikeAliased(predicate, params);
    case 'is_null':
      return `${fieldRefAliased(predicate.field)} IS NULL`;
    case 'is_not_null':
      return `${fieldRefAliased(predicate.field)} IS NOT NULL`;
    case 'and':
      return `(${predicate.predicates.map((p) => compilePredicateAliased(p, params)).join(' AND ')})`;
    case 'or':
      return `(${predicate.predicates.map((p) => compilePredicateAliased(p, params)).join(' OR ')})`;
    case 'not':
      return `NOT (${compilePredicateAliased(predicate.predicate, params)})`;
    default:
      throw new FilterError(`Unknown operator: ${predicate.op}`);
  }
}

/**
 * Compile a comparison predicate with short table aliases.
 *
 * @param {object} predicate - The comparison predicate.
 * @param {string} operator - SQL comparison operator.
 * @param {Array} params - Accumulator for parameter values.
 * @returns {string} SQL comparison expression with aliases.
 */
function compileComparisonAliased(predicate, operator, params) {
  const ref = fieldRefAliased(predicate.field);
  if (predicate.value === null) {
    return operator === '=' ? `${ref} IS NULL` : `${ref} IS NOT NULL`;
  }
  params.push(predicate.value);
  return `${ref} ${operator} ?`;
}

/**
 * Compile an `in` predicate with short table aliases.
 *
 * @param {object} predicate - The `in` predicate.
 * @param {Array} params - Accumulator for parameter values.
 * @returns {string} SQL `IN` expression with aliases.
 */
function compileInAliased(predicate, params) {
  const ref = fieldRefAliased(predicate.field);
  for (const value of predicate.values) {
    params.push(value);
  }
  const placeholders = predicate.values.map(() => '?').join(', ');
  return `${ref} IN (${placeholders})`;
}

/**
 * Compile a `not_in` predicate with short table aliases.
 *
 * @param {object} predicate - The `not_in` predicate.
 * @param {Array} params - Accumulator for parameter values.
 * @returns {string} SQL `NOT IN` expression with aliases.
 */
function compileNotInAliased(predicate, params) {
  const ref = fieldRefAliased(predicate.field);
  for (const value of predicate.values) {
    params.push(value);
  }
  const placeholders = predicate.values.map(() => '?').join(', ');
  return `${ref} NOT IN (${placeholders})`;
}

/**
 * Compile a `like` predicate with short table aliases.
 *
 * @param {object} predicate - The `like` predicate.
 * @param {Array} params - Accumulator for parameter values.
 * @returns {string} SQL `LIKE` expression with aliases.
 */
function compileLikeAliased(predicate, params) {
  const ref = fieldRefAliased(predicate.field);
  params.push(predicate.value);
  return `${ref} LIKE ?`;
}

/**
 * Resolve a filter field name to a SQL column reference using short aliases.
 *
 * `d` for `documents`, `c` for `chunks`.
 *
 * @param {string} field - The filter grammar field name.
 * @returns {string} SQL column reference with alias.
 */
function fieldRefAliased(field) {
  const mapping = FIELD_TO_SQL.get(field);
  if (!mapping) return `c.${field}`;
  if (mapping.table === 'documents') return `d.${mapping.column}`;
  return `c.${mapping.column}`;
}
