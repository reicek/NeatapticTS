/**
 * @module metadata-enrichment
 * @description Metadata enrichment functions for the Repo Cortex semantic index.
 *
 * Computes architectural layer, JSDoc quality, cyclomatic complexity, test coverage,
 * and source path pattern from chunk metadata and repository data at index time.
 * These enrichment functions are called during `build-index` to populate the
 * metadata columns added by the v3 schema migration.
 *
 * ### Enrichment Pipeline
 *
 * ```mermaid
 * flowchart TD
 *   A[Chunk with v2 metadata] --> B[resolveArchLayer]
 *   A --> C[classifyJsdocQuality]
 *   A --> D[countJsdocWords]
 *   A --> E[computeCyclomaticComplexity]
 *   A --> F[resolveSourcePathPattern]
 *   G[Coverage report] --> H[classifyTestCoverage]
 *   B --> I[Enriched chunk]
 *   C --> I
 *   D --> I
 *   E --> I
 *   F --> I
 *   H --> I
 * ```
 *
 * @example
 * import { enrichChunkMetadata } from './metadata-enrichment.mjs';
 *
 * const enriched = enrichChunkMetadata({
 *   jsdoc_text: 'Activate the network forward pass. @param input - The input values. @returns The output values.',
 *   export_type: 'function',
 *   module_path: 'src/architecture/network/activate',
 *   file_path: 'src/architecture/network/activate/network.activate.ts',
 *   family: 'ts-source',
 * }, null);
 * // enriched.arch_layer === 'network'
 * // enriched.jsdoc_quality === 'good'
 * // enriched.jsdoc_word_count === 11
 * // enriched.test_coverage === 'unknown'
 */

import path from 'node:path';

/**
 * Architectural layer mapping from module path or file path prefixes.
 *
 * Each entry maps a path prefix to a fixed architectural layer name.
 * The resolution function checks `module_path` first (for ts-source chunks),
 * then falls back to `file_path` for non-ts-source families.
 *
 * @type {ReadonlyArray<{ prefix: string, layer: string }>}
 */
const ARCH_LAYER_RULES = [
  { prefix: 'src/architecture/network', layer: 'network' },
  { prefix: 'src/neat', layer: 'neat' },
  { prefix: 'src/methods', layer: 'methods' },
  { prefix: 'src/multithreading', layer: 'multithreading' },
  { prefix: 'src/config', layer: 'config' },
];

/**
 * Fallback architectural layer mapping for non-src file paths.
 *
 * Checked when a file path does not start with `src/` and no module path
 * matched the primary rules above.
 *
 * @type {ReadonlyArray<{ prefix: string, layer: string }>}
 */
const ARCH_LAYER_FALLBACK_RULES = [
  { prefix: '.github/skills/', layer: 'skill' },
  { prefix: '.github/agents/', layer: 'agent' },
  { prefix: 'plans/', layer: 'plan' },
];

/**
 * Resolve the architectural layer from a module path and/or file path.
 *
 * Checks `module_path` first (for ts-source chunks), then falls back to
 * `file_path` for non-ts-source families. Returns `'utils'` for any `src/`
 * path that does not match a known layer, and `'doc'` for all other paths.
 *
 * @param {string | null} modulePath - The chunk's module path (e.g. `src/architecture/network`).
 * @param {string | null} filePath - The chunk's file path (e.g. `src/architecture/network/activate/network.activate.ts`).
 * @returns {string} Architectural layer name: `network`, `neat`, `methods`, `multithreading`, `config`, `utils`, `skill`, `agent`, `plan`, or `doc`.
 *
 * @example
 * resolveArchLayer('src/architecture/network', null); // 'network'
 * resolveArchLayer(null, 'src/neat/neat.ts');         // 'neat'
 * resolveArchLayer(null, '.github/skills/test/SKILL.md'); // 'skill'
 * resolveArchLayer('src/utilities', 'src/utilities/format.ts'); // 'utils'
 * resolveArchLayer(null, 'README.md');                // 'doc'
 */
export function resolveArchLayer(modulePath, filePath) {
  // Check module_path first (for ts-source chunks)
  if (modulePath) {
    for (const rule of ARCH_LAYER_RULES) {
      if (modulePath.startsWith(rule.prefix)) {
        return rule.layer;
      }
    }
    if (modulePath.startsWith('src/')) {
      return 'utils';
    }
  }

  // Fallback to file_path for non-ts-source families
  if (filePath) {
    for (const rule of ARCH_LAYER_FALLBACK_RULES) {
      if (filePath.startsWith(rule.prefix)) {
        return rule.layer;
      }
    }
    if (filePath.startsWith('src/')) {
      return 'utils';
    }
  }

  return 'doc';
}

/**
 * Count the number of words in a JSDoc text string.
 *
 * Words are sequences of non-whitespace characters. Strips leading `*`
 * characters from multi-line JSDoc comment lines before counting.
 *
 * @param {string | null} jsdocText - Raw JSDoc text from the chunk metadata.
 * @returns {number} Word count, or 0 when the input is null or empty.
 *
 * @example
 * countJsdocWords('Activate the network');           // 3
 * countJsdocWords('');                               // 0
 * countJsdocWords(null);                             // 0
 * countJsdocWords('@param input - The input values'); // 5
 */
export function countJsdocWords(jsdocText) {
  if (!jsdocText || typeof jsdocText !== 'string') return 0;
  // Strip leading * from JSDoc comment lines
  const cleaned = jsdocText.replace(/^\s*\*\s?/gm, ' ').trim();
  const words = cleaned.split(/\s+/).filter((word) => word.length > 0);
  return words.length;
}

/**
 * Classify JSDoc quality from the raw JSDoc text.
 *
 * Quality levels:
 * - `none`: no JSDoc text or empty string
 * - `weak`: fewer than 10 words in the summary
 * - `adequate`: 10 or more words, but missing `@param` or `@returns`
 * - `good`: 10 or more words, has both `@param` and `@returns`
 *
 * @param {string | null} jsdocText - Raw JSDoc text from the chunk metadata.
 * @returns {'none' | 'weak' | 'adequate' | 'good'} JSDoc quality classification.
 *
 * @example
 * classifyJsdocQuality(null);                                             // 'none'
 * classifyJsdocQuality('');                                               // 'none'
 * classifyJsdocQuality('Activate');                                        // 'weak'
 * classifyJsdocQuality('Activate the network forward pass with inputs');  // 'adequate'
 * classifyJsdocQuality('Activate. @param input - values. @returns out');  // 'good'
 */
export function classifyJsdocQuality(jsdocText) {
  if (!jsdocText || jsdocText.trim().length === 0) return 'none';

  const wordCount = countJsdocWords(jsdocText);
  if (wordCount < 10) return 'weak';

  const hasParam = jsdocText.includes('@param');
  const hasReturns =
    jsdocText.includes('@returns') || jsdocText.includes('@return');

  if (hasParam && hasReturns) return 'good';
  return 'adequate';
}

/**
 * Classify test coverage for a file based on a coverage report.
 *
 * Coverage levels:
 * - `full`: 100% statement coverage
 * - `partial`: > 0% and < 100% statement coverage
 * - `none`: 0% statement coverage
 * - `unknown`: file not in coverage report, or report is missing
 *
 * @param {string} filePath - Repo-relative file path.
 * @param {Record<string, { statements: { pct: number } }> | null} coverageReport - Coverage summary object or null.
 * @returns {'full' | 'partial' | 'none' | 'unknown'} Test coverage classification.
 *
 * @example
 * classifyTestCoverage('src/neat/neat.ts', { 'src/neat/neat.ts': { statements: { pct: 100 } } }); // 'full'
 * classifyTestCoverage('src/neat/neat.ts', null);  // 'unknown'
 */
export function classifyTestCoverage(filePath, coverageReport) {
  if (!coverageReport) return 'unknown';
  if (!filePath) return 'unknown';

  const fileCoverage = coverageReport[filePath];
  if (!fileCoverage) return 'unknown';

  const statementPct = fileCoverage.statements?.pct ?? 0;
  if (statementPct === 100) return 'full';
  if (statementPct > 0) return 'partial';
  return 'none';
}

/**
 * Compute a glob-style source path pattern from a file path.
 *
 * For `src/` files, the pattern includes the directory path with `/**` suffix.
 * For non-`src/` files, the pattern uses the top-level directory with `/**`.
 *
 * @param {string} filePath - Repo-relative file path.
 * @returns {string} Glob-style path pattern (e.g. `src/neat/mutation/**`).
 *
 * @example
 * resolveSourcePathPattern('src/architecture/network/activate/network.activate.ts'); // 'src/architecture/network/activate/**'
 * resolveSourcePathPattern('.github/skills/coverage-guard/SKILL.md');               // '.github/skills/**'
 * resolveSourcePathPattern('plans/test-repair.plans.md');                            // 'plans/**'
 */
export function resolveSourcePathPattern(filePath) {
  if (!filePath) return '';
  const dir = path.posix.dirname(filePath);
  if (filePath.startsWith('src/')) {
    return `${dir}/**`;
  }
  const segments = dir.split('/');
  return `${segments.slice(0, Math.min(2, segments.length)).join('/')}/**`;
}

/**
 * Compute cyclomatic complexity for a TypeScript declaration.
 *
 * Cyclomatic complexity starts at 1 (the function itself) and increments
 * for each decision point: `if`, `else if`, `case`, `for`, `while`, `do`,
 * `catch`, `&&`, `||`, `??`, and ternary (`? :`).
 *
 * This function performs a simple text-based count rather than a full AST
 * traversal, which is sufficient for index-time metadata. For non-ts-source
 * chunks, the complexity should be set to `null`.
 *
 * @param {string | null} sourceText - The source text of the declaration, or null.
 * @returns {number | null} Cyclomatic complexity, or null if source text is unavailable.
 *
 * @example
 * computeCyclomaticComplexity('function add(a, b) { return a + b; }');     // 1
 * computeCyclomaticComplexity('function max(a, b) { if (a > b) return a; return b; }'); // 2
 * computeCyclomaticComplexity(null);                                        // null
 */
export function computeCyclomaticComplexity(sourceText) {
  if (!sourceText || typeof sourceText !== 'string') return null;

  let complexity = 1;

  // Count decision points with word boundaries where needed
  // if, else if, for, while, case, catch, &&, ||, ??, ternary
  const patterns = [
    /\bif\b/g,
    /\belse\s+if\b/g,
    /\bfor\b/g,
    /\bwhile\b/g,
    /\bcase\b/g,
    /\bcatch\b/g,
    /&&/g,
    /\|\|/g,
    /\?\?/g,
    /\?[^?.]/g, // ternary ? (but not ?. or ??)
  ];

  for (const pattern of patterns) {
    const matches = sourceText.match(pattern);
    if (matches) {
      complexity += matches.length;
    }
  }

  // Subtract double-counted else-if (counted by both 'if' and 'else if')
  const elseIfMatches = sourceText.match(/\belse\s+if\b/g);
  if (elseIfMatches) {
    complexity -= elseIfMatches.length;
  }

  return complexity;
}

/**
 * Load the coverage summary report from disk.
 *
 * Reads `coverage/coverage-summary.json` from the repository root. Returns
 * `null` when the file is missing or invalid, which causes all test coverage
 * classifications to fall back to `'unknown'`.
 *
 * @param {string} repoRootPath - Absolute path to the repository root.
 * @returns {Promise<Record<string, { statements: { pct: number } }> | null>} Parsed coverage report, or null.
 */
export async function loadCoverageReport(repoRootPath) {
  const coveragePath = path.join(
    repoRootPath,
    'coverage',
    'coverage-summary.json',
  );
  try {
    const { readFile } = await import('node:fs/promises');
    const data = await readFile(coveragePath, 'utf8');
    return JSON.parse(data);
  } catch {
    // Coverage report not available — all files get 'unknown'
    return null;
  }
}

/**
 * Enrich a single chunk's metadata with computed columns.
 *
 * Computes `arch_layer`, `jsdoc_quality`, `jsdoc_word_count`,
 * `cyclomatic_complexity`, `test_coverage`, and `source_path_pattern` from
 * the chunk's existing v2 metadata and the optional coverage report.
 *
 * @param {object} chunk - A chunk object with v2 metadata fields.
 * @param {string | null} chunk.jsdoc_text - Raw JSDoc text.
 * @param {string | null} chunk.export_type - Export type.
 * @param {string | null} chunk.module_path - Module path.
 * @param {string} chunk.file_path - Repo-relative file path.
 * @param {string} chunk.family - Document family.
 * @param {string | null} chunk.body_text - Chunk source text (for cyclomatic complexity).
 * @param {Record<string, { statements: { pct: number } }> | null} coverageReport - Coverage summary or null.
 * @returns {{ arch_layer: string, jsdoc_quality: string, jsdoc_word_count: number, cyclomatic_complexity: number | null, test_coverage: string, source_path_pattern: string }} Enriched metadata fields.
 *
 * @example
 * const metadata = enrichChunkMetadata({
 *   jsdoc_text: 'Activate. @param x - input. @returns result',
 *   export_type: 'function',
 *   module_path: 'src/architecture/network/activate',
 *   file_path: 'src/architecture/network/activate/network.activate.ts',
 *   family: 'ts-source',
 *   body_text: 'function activate(x) { if (x > 0) return x; return 0; }',
 * }, null);
 * // metadata.arch_layer === 'network'
 * // metadata.jsdoc_quality === 'good'
 * // metadata.test_coverage === 'unknown'
 */
export function enrichChunkMetadata(chunk, coverageReport) {
  const modulePath = chunk.module_path ?? null;
  const filePath = chunk.file_path ?? '';

  const archLayer = resolveArchLayer(modulePath, filePath);
  const jsdocQuality = classifyJsdocQuality(chunk.jsdoc_text);
  const jsdocWordCount = countJsdocWords(chunk.jsdoc_text);

  // Cyclomatic complexity is only meaningful for ts-source function/method chunks
  const cyclomaticComplexity =
    chunk.family === 'ts-source' && chunk.body_text
      ? computeCyclomaticComplexity(chunk.body_text)
      : null;

  // Test coverage is resolved from the coverage report for ts-source files
  const testCoverage =
    chunk.family === 'ts-source'
      ? classifyTestCoverage(filePath, coverageReport)
      : 'unknown';

  const sourcePathPattern = resolveSourcePathPattern(filePath);

  return {
    arch_layer: archLayer,
    jsdoc_quality: jsdocQuality,
    jsdoc_word_count: jsdocWordCount,
    cyclomatic_complexity: cyclomaticComplexity,
    test_coverage: testCoverage,
    source_path_pattern: sourcePathPattern,
  };
}

/**
 * Enrich a document-level metadata with computed columns.
 *
 * Computes `arch_layer`, `test_coverage`, and `source_path_pattern` for a
 * document row. These are document-level fields that all chunks in the
 * document inherit.
 *
 * @param {object} document - A document object with `file_path` and `doc_family`.
 * @param {Record<string, { statements: { pct: number } }> | null} coverageReport - Coverage summary or null.
 * @returns {{ arch_layer: string, test_coverage: string, source_path_pattern: string }} Enriched document metadata.
 */
export function enrichDocumentMetadata(document, coverageReport) {
  const filePath = document.file_path ?? '';
  const archLayer = resolveArchLayer(null, filePath);
  const testCoverage =
    document.family === 'ts-source'
      ? classifyTestCoverage(filePath, coverageReport)
      : 'unknown';
  const sourcePathPattern = resolveSourcePathPattern(filePath);

  return {
    arch_layer: archLayer,
    test_coverage: testCoverage,
    source_path_pattern: sourcePathPattern,
  };
}
