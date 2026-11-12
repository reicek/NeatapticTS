/**
 * @module customization-utils
 *
 * Shared utilities for NeatapticTS agent-customization validation scripts.
 *
 * Provides helpers for: CLI argument parsing, human/JSON report output, recursive
 * Markdown file discovery, YAML frontmatter parsing, and issue-aggregation
 * primitives consumed by `tier-inventory.mjs`, `validate-agent-graph.mjs`,
 * `validate-plan-sync.mjs`, and related scripts under `scripts/agent-customization/`.
 */
import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';

/** Absolute path to the repository root (the current working directory at startup). */
export const repoRoot = process.cwd();

/**
 * Recognized tool identifiers that may appear in an agent's `tools:` frontmatter list.
 * Used by validators to flag unknown or unsupported tool names.
 */
export const knownAgentTools = new Set([
  'agent',
  'bash',
  'edit',
  'execute',
  'read',
  'search',
  'todo',
  'web',
  // Lightweight lazy-load MCP facades (router tools that spawn the real server on demand)
  'cortex/cortex',
  'devtools/devtools',
]);

/**
 * Parses CLI argument strings into a structured options object.
 *
 * Recognized flags: `--json`, `--strict`, `--help` / `-h`,
 * `--contract=<value>`, `--input=<value>`, `--plan=<value>`.
 *
 * @param argv - Argument strings from `process.argv.slice(2)`.
 * @returns Parsed options with typed fields and sane defaults.
 */
export function parseArgs(argv) {
  const options = {
    json: false,
    strict: false,
    help: false,
    all: false,
    dryRun: false,
    contract: 'tier0',
    plan: 'plans/completed/Agentic_Workflow_Architecture.plans.md',
  };

  for (let argIndex = 0; argIndex < argv.length; argIndex++) {
    const rawArg = argv[argIndex];
    if (rawArg === '--json') options.json = true;
    else if (rawArg === '--strict') options.strict = true;
    else if (rawArg === '--all') options.all = true;
    else if (rawArg === '--dry-run' || rawArg === '--dry_run')
      options.dryRun = true;
    else if (rawArg === '--help' || rawArg === '-h') options.help = true;
    else if (rawArg.startsWith('--contract='))
      options.contract = rawArg.slice('--contract='.length);
    else if (rawArg === '--contract' && argv[argIndex + 1] !== undefined)
      options.contract = argv[++argIndex];
    else if (rawArg.startsWith('--input='))
      options.input = rawArg.slice('--input='.length);
    else if (rawArg === '--input' && argv[argIndex + 1] !== undefined)
      options.input = argv[++argIndex];
    else if (rawArg.startsWith('--plan='))
      options.plan = rawArg.slice('--plan='.length);
    else if (rawArg === '--plan' && argv[argIndex + 1] !== undefined)
      options.plan = argv[++argIndex];
  }

  return options;
}

/**
 * Prints a human-readable usage block to stdout.
 *
 * @param options - Descriptor object.
 * @param options.title - Script name / headline printed as the first line.
 * @param options.usage - Example invocation string shown under `Usage:`.
 * @param options.options - Extra `[flag, description]` pairs appended after the standard flags.
 */
export function printUsage({ title, usage, options = [] }) {
  const optionLines = [
    ['--json', 'Write machine-readable JSON to stdout.'],
    [
      '--strict',
      'Enforce the final target architecture rather than auditing the current state.',
    ],
    ['--help, -h', 'Show this help text.'],
    ...options,
  ];

  console.log(`${title}\n\nUsage:\n  ${usage}\n\nOptions:`);
  for (const [flag, description] of optionLines) {
    console.log(`  ${flag.padEnd(18)} ${description}`);
  }
}

/**
 * Writes a validation report to stdout in either JSON or human-readable format.
 *
 * When `json` is `true` the full report object is serialized with two-space
 * indentation. Otherwise the `summaryText` (or a default `PASS/FAIL <name>`
 * line) is printed followed by one line per issue.
 *
 * @param report - Validation report with at least `ok`, `name`, and optional `issues` / `summaryText`.
 * @param opts - Output options.
 * @param opts.json - When `true`, serializes the full report as JSON.
 */
export function writeReport(report, { json }) {
  // Write UTF-8 encoded output to stdout so non-ASCII characters (e.g. ≤, —)
  // are preserved regardless of the host console code page. Using
  // process.stdout.write with explicit 'utf8' encoding avoids the implicit
  // encoding translation that console.log may undergo on some platforms.
  if (json) {
    process.stdout.write(JSON.stringify(report, null, 2) + '\n', 'utf8');
    return;
  }

  process.stdout.write(
    (report.summaryText ?? `${report.ok ? 'PASS' : 'FAIL'} ${report.name}`) +
      '\n',
    'utf8',
  );
  for (const issue of report.issues ?? []) {
    process.stdout.write(
      `- ${issue.severity.toUpperCase()}: ${issue.path}: ${issue.message}\n`,
      'utf8',
    );
  }
}

/**
 * Recursively discovers Markdown files under a repo-relative directory path.
 *
 * Silently skips directories that cannot be read. The returned array is sorted
 * lexicographically by repo-relative path.
 *
 * @param rootRelativePath - Directory path relative to {@link repoRoot} to scan.
 * @param predicate - Receives each repo-relative file path; return `true` to include it.
 * @returns Sorted array of repo-relative paths that satisfy the predicate.
 */
export async function listMarkdownFiles(rootRelativePath, predicate) {
  const root = path.join(repoRoot, rootRelativePath);
  const discovered = [];

  async function visit(directory) {
    let entries;
    try {
      entries = await readdir(directory, { withFileTypes: true });
    } catch {
      return;
    }

    for (const entry of entries) {
      const absolutePath = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(absolutePath);
        continue;
      }

      const relativePath = normalizePath(path.relative(repoRoot, absolutePath));
      if (entry.isFile() && predicate(relativePath))
        discovered.push(relativePath);
    }
  }

  await visit(root);
  return discovered.toSorted();
}

/**
 * Reads a workspace file as a UTF-8 string.
 *
 * @param relativePath - Path relative to {@link repoRoot}.
 * @returns File contents as a string.
 */
export async function readWorkspaceFile(relativePath) {
  return readFile(path.join(repoRoot, relativePath), 'utf8');
}

/**
 * Reads a workspace file as raw bytes.
 *
 * Used for encoding checks before text decoding.
 *
 * @param relativePath - Path relative to {@link repoRoot}.
 * @returns File contents as a Uint8Array.
 */
export async function readWorkspaceFileBytes(relativePath) {
  const buffer = await readFile(path.join(repoRoot, relativePath));
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
}

/**
 * Checks whether a byte sequence is valid UTF-8.
 *
 * @param bytes - Raw file bytes.
 * @returns `true` when the bytes decode cleanly as UTF-8.
 */
export function isValidUtf8(bytes) {
  try {
    new TextDecoder('utf-8', { fatal: true }).decode(bytes);
    return true;
  } catch {
    return false;
  }
}

/**
 * Checks whether a regular file exists at the given repo-relative path.
 *
 * @param relativePath - Path relative to {@link repoRoot}.
 * @returns `true` when the path resolves to an existing file (not a directory).
 */
export async function fileExists(relativePath) {
  try {
    return (await stat(path.join(repoRoot, relativePath))).isFile();
  } catch {
    return false;
  }
}

/**
 * Parses YAML frontmatter from a Markdown document.
 *
 * Recognizes the `---` fence convention used by `.agent.md`, `.prompt.md`,
 * and `SKILL.md` files. Scalar, boolean, and inline-array values are converted
 * to their native JavaScript types; multi-line YAML blocks are not supported.
 *
 * @param text - Raw file contents (BOM-stripped automatically).
 * @param relativePath - Repo-relative file path, used to annotate issue records.
 * @returns Object with `data` map, `body` (post-frontmatter text), `raw` frontmatter
 *   string, and a `issues` array of any parse warnings or errors.
 */
export function parseFrontmatter(text, relativePath) {
  const normalized = text.replace(/^\uFEFF/, '');
  const lines = normalized.split(/\r?\n/);
  if (lines.at(0) !== '---') {
    return {
      data: {},
      body: normalized,
      raw: '',
      issues: [
        issue('error', relativePath, 'Missing opening YAML frontmatter fence.'),
      ],
    };
  }

  const endIndex = lines.findIndex(
    (line, index) => index > 0 && line === '---',
  );
  if (endIndex === -1) {
    return {
      data: {},
      body: normalized,
      raw: lines.slice(1).join('\n'),
      issues: [
        issue('error', relativePath, 'Missing closing YAML frontmatter fence.'),
      ],
    };
  }

  const rawLines = lines.slice(1, endIndex);
  const data = {};
  const issues = [];

  for (let i = 0; i < rawLines.length; i++) {
    const line = rawLines[i];
    if (!line.trim() || line.trim().startsWith('#')) continue;

    const match = /^(?<key>[A-Za-z0-9_-]+):(?<value>.*)$/.exec(line);
    if (!match?.groups) {
      // Skip indented structural lines (handoffs, nested maps, list items)
      if (/^\s+/.test(line)) continue;
      issues.push(
        issue(
          'warning',
          relativePath,
          `Could not parse frontmatter line: ${line}`,
        ),
      );
      continue;
    }

    const key = match.groups.key;
    let value = match.groups.value.trim();

    // Support multi-line bracketed inline arrays where the opening '[' is on
    // a following indented line or the array spans multiple lines before ']'.
    if (value === '') {
      // Peek ahead for a bracketed array block starting on subsequent lines.
      let j = i + 1;
      while (j < rawLines.length && !rawLines[j].trim()) j++;
      if (j < rawLines.length && rawLines[j].trim().startsWith('[')) {
        const collected = [];
        while (j < rawLines.length) {
          const next = rawLines[j].trim();
          collected.push(next);
          if (next.includes(']')) break;
          j++;
        }
        value = collected.join(' ');
        i = j; // advance outer loop to skip consumed lines
      }
    } else if (/^\[.*$/.test(value) && !/\]$/.test(value)) {
      // Handle case where '[' starts on the same line but the ']' appears later.
      let j = i + 1;
      while (j < rawLines.length) {
        value += ' ' + rawLines[j].trim();
        if (rawLines[j].includes(']')) {
          i = j;
          break;
        }
        j++;
      }
    }

    data[key] = parseFrontmatterValue(value);
  }

  return {
    data,
    body: lines.slice(endIndex + 1).join('\n'),
    raw: rawLines.join('\n'),
    issues,
  };
}

/**
 * Converts a raw frontmatter scalar string to its native JavaScript value.
 *
 * Conversion rules:
 * - Empty string → `true` (bare key with no value)
 * - `"true"` → `true`, `"false"` → `false`
 * - Quoted string → stripped string
 * - Inline array `[...]` → `string[]` via {@link parseInlineArray}
 * - Anything else → trimmed string with trailing inline `# comment` removed
 *
 * @param value - Raw value substring from the frontmatter line (after the colon).
 * @returns Typed JavaScript value.
 */
export function parseFrontmatterValue(value) {
  if (value === '') return true;
  if (value === 'true') return true;
  if (value === 'false') return false;
  if (/^['"].*['"]$/.test(value)) return value.slice(1, -1);
  if (/^\[.*\]$/.test(value)) return parseInlineArray(value);
  return value.replace(/\s+#.*$/, '').trim();
}

/**
 * Parses a YAML-style inline array literal such as `[foo, "bar", 'baz']`.
 *
 * @param value - Raw value string including the surrounding `[` and `]` brackets.
 * @returns Array of unquoted, trimmed, non-empty string elements.
 */
export function parseInlineArray(value) {
  const inner = value.slice(1, -1).trim();
  if (!inner) return [];

  return inner
    .split(',')
    .map((item) => item.trim().replace(/^['"]|['"]$/g, ''))
    .filter(Boolean);
}

/**
 * Converts a file-system path to forward-slash form for cross-platform stability.
 *
 * On Windows `path.sep` is `\`; this helper replaces all separators so that
 * repo-relative path strings are consistent between Windows and Linux runners.
 *
 * @param value - Path that may use the platform-native `path.sep`.
 * @returns Path with all separators replaced by `'/'`.
 */
export function normalizePath(value) {
  return value.replaceAll(path.sep, '/');
}

/**
 * Creates a typed validation issue record.
 *
 * @param severity - `'error'` for blocking violations; `'warning'` for advisory findings.
 * @param relativePath - Repo-relative source file path where the issue was detected.
 * @param message - Human-readable description of the violation.
 * @returns Issue object consumed by {@link summarizeIssues} and report writers.
 */
export function issue(severity, relativePath, message) {
  return { severity, path: relativePath, message };
}

/**
 * Aggregates a list of issues into a structured validation report.
 *
 * The `ok` field is `true` only when there are zero errors (warnings are allowed).
 *
 * @param name - Short label for the validation surface (e.g. `'agent graph'`).
 * @param issues - Issue records produced by one or more validators.
 * @returns Report with `ok`, `counts` (`errors`/`warnings`), `summaryText`, and `issues`.
 */
export function summarizeIssues(name, issues) {
  const errors = issues.filter(
    (currentIssue) => currentIssue.severity === 'error',
  ).length;
  const warnings = issues.filter(
    (currentIssue) => currentIssue.severity === 'warning',
  ).length;
  return {
    name,
    ok: errors === 0,
    issues,
    counts: { errors, warnings },
    summaryText: `${errors === 0 ? 'PASS' : 'FAIL'} ${name}: ${errors} errors, ${warnings} warnings`,
  };
}

/**
 * Structured error raised when a plan YAML block cannot be parsed without
 * silently dropping content.
 *
 * The class exists because of learning-log event
 * `learn-20261109-f9-plan-parser-silent-skip-overindented-id-items`: a `- id:`
 * list item indented deeper than the object that owns it used to be skipped
 * without a trace, so a corrupted step packet still parsed "cleanly" and every
 * shape-only gate passed. A malformed packet must fail at the gate instead of
 * degrading silently, and the failure must name the source file, the line
 * number and the observed indentation so it can be repaired.
 */
export class PlanYamlParseError extends Error {
  /**
   * @param {string} message - Human-readable failure description.
   * @param {object} [details] - Structured failure details.
   * @param {string} [details.code] - Stable machine-readable finding code.
   * @param {string|null} [details.file] - Source file path, when known.
   * @param {number|null} [details.line] - 1-based line number of the offending line IN THE SOURCE FILE.
   * @param {number|null} [details.blockLine] - 1-based line number of the offending line INSIDE THE YAML BLOCK.
   * @param {number|null} [details.indent] - Observed indentation of the offending line.
   * @param {number|null} [details.baseIndent] - Enclosing base indentation.
   * @param {string|null} [details.snippet] - Trimmed text of the offending line.
   */
  constructor(message, details = {}) {
    super(message);
    this.name = 'PlanYamlParseError';
    this.code = details.code ?? 'PLAN_YAML_PARSE_ERROR';
    this.file = details.file ?? null;
    this.line = details.line ?? null;
    this.blockLine = details.blockLine ?? null;
    this.indent = details.indent ?? null;
    this.baseIndent = details.baseIndent ?? null;
    this.snippet = details.snippet ?? null;
  }
}

/**
 * Builds a {@link PlanYamlParseError} for a malformed plan YAML line.
 *
 * @param {object} context - Parse context carrying the source file path.
 * @param {object} details - Failure details (`code`, `message`, `line`,
 *   `indent`, `baseIndent`, `snippet`).
 * @returns {PlanYamlParseError} Error describing the malformed line.
 */
function planYamlError(context, details) {
  const blockLine = details.line ?? null;
  // `details.line` is relative to the YAML block; shift it into the source file
  // so the message names a location a human can actually open.
  const sourceLine =
    typeof blockLine === 'number'
      ? (context.startLine ?? 1) + Math.max(0, blockLine - 1)
      : null;
  const location = context.file
    ? `${context.file}:${sourceLine ?? '?'}`
    : `line ${sourceLine ?? '?'}`;
  const indentText =
    typeof details.indent === 'number'
      ? ` (indent ${details.indent}, enclosing base indent ${details.baseIndent})`
      : '';
  const snippet = details.snippet ? ` "${details.snippet}"` : '';
  return new PlanYamlParseError(
    `Plan YAML parse error at ${location}${indentText}: ${details.message}${snippet}`,
    {
      ...details,
      file: context.file ?? null,
      line: sourceLine,
      blockLine,
    },
  );
}

/**
 * Strict YAML field pattern: the identifier-like keys used by plan packets.
 */
const YAML_FIELD_PATTERN = /^(?<key>[A-Za-z0-9_]+):(?<rest>.*)$/u;

/**
 * Wide YAML field pattern: an identifier key that may carry hyphens.
 *
 * Plan packets use hyphenated keys such as `RED-1:` or `P1-S1-A:`. The
 * strict pattern rejects them, and a rejected key used to be stepped over
 * silently, which dropped the value that belonged to it.
 */
const YAML_WIDE_FIELD_PATTERN =
  /^(?<key>[A-Za-z0-9_][A-Za-z0-9_-]*):(?<rest>.*)$/u;

/**
 * Loose YAML field pattern: a key that carries extra text (for example a key
 * with a parenthetical suffix) and declares an empty value.
 *
 * The pattern deliberately refuses a quoted scalar, so a plain list item such
 * as `- 'a: b'` is never mistaken for a key, and it refuses a value, so a
 * scalar such as `npm run foo: bar` keeps its scalar meaning.
 *
 * The key must contain whitespace. A key without whitespace that the strict
 * pattern rejects (for example `some-key:`) keeps its legacy handling, where
 * the item is read as "the object starts on the next line".
 */
const YAML_LOOSE_FIELD_PATTERN = /^(?<key>[^:'"\s][^:]*\s[^:]*):(?<rest>)$/u;

/**
 * Matches a YAML field line and returns its key and value.
 *
 * The strict pattern is tried first; the wide pattern adds hyphenated keys;
 * the loose pattern is the fallback for a key whose text is not an identifier.
 * A key that every pattern rejects must NOT be dropped: dropping the key
 * silently drops the list or object that belongs to it, which is the failure
 * mode recorded in `learn-20261109-f9-plan-parser-silent-skip-overindented-id-items`.
 *
 * @param {string} line - Trimmed line text.
 * @param {object} [options] - Match options.
 * @param {boolean} [options.allowHyphen] - When true, a hyphenated key is
 *   accepted (used for object fields, where a hyphenated key is a key).
 * @returns {RegExpExecArray|null} Match result, or `null` when the line is not a field.
 */
function matchYamlField(line, options = {}) {
  const strict = YAML_FIELD_PATTERN.exec(line);
  if (strict) return strict;
  if (options.allowHyphen) {
    const wide = YAML_WIDE_FIELD_PATTERN.exec(line);
    if (wide) return wide;
  }
  return YAML_LOOSE_FIELD_PATTERN.exec(line);
}

/**
 * Determines whether a list item carries STRUCTURED content.
 *
 * A list item is structured when its payload would be parsed as an object or
 * a nested list, that is when the payload is a YAML field (`id: ...`,
 * `command: ...`) or a key whose value starts on the next line. A structured
 * item that the parser cannot place is CONTENT: dropping it loses a
 * criterion, a command or a field, which is the F9 defect class.
 *
 * A PLAIN SCALAR item (`- 'a free-form note'`) carries no structure. The
 * repo's plan evidence blocks carry such items at a deeper indent than their
 * siblings on purpose, and dropping a scalar does not drop a criterion, so
 * the parser tolerates it exactly as it did before the F9 fix. The
 * indentation lint is the detector for the structured case.
 *
 * @param {string} remainder - List item payload with the `- ` marker removed.
 * @returns {boolean} `true` when the payload is structured.
 */
function isStructuredListItem(remainder) {
  const payload = String(remainder ?? '').trim();
  if (payload === '') return true;
  if (matchYamlField(payload, { allowHyphen: true })) return true;
  // A key whose value starts on the next line (`- acceptance_criteria:`).
  if (/^.*:\s*$/u.test(payload)) return true;
  return false;
}

/**
 * Parses a plan YAML metadata block into a JavaScript object.
 *
 * Supports the subset used by plan step/phase packets: scalar key–value pairs,
 * lists of scalars, lists of objects (e.g. `slices`), and nested objects
 * (e.g. `pre_execute_hook`). Empty values on a top-level key are interpreted as
 * the start of a list. Nested object fields with no children parse as an empty
 * object `{}` so that object-valued fields such as `args:` remain objects.
 * Multi-line object lists use the standard YAML `- key: value` indentation
 * pattern.
 *
 * The parser FAILS LOUDLY: a list item whose indentation exceeds the base
 * indentation of the list or object that owns it raises a
 * {@link PlanYamlParseError} instead of being skipped, because a silently
 * skipped item corrupts the packet without any visible symptom.
 *
 * @param yamlText - Raw YAML string (without the surrounding fence).
 * @param {object} [options] - Parse options.
 * @param {string|null} [options.file] - Source file path used in error messages.
 * @param {number} [options.startLine] - 1-based line number in the source file of the
 *   first content line of this block; used to report file-absolute line numbers.
 * @returns Object with parsed values; lists are arrays, nested object lists are
 *   arrays of objects, and nested objects are plain objects.
 * @throws {PlanYamlParseError} When the block contains an over-indented list item.
 */
export function parsePlanYamlBlock(yamlText, options = {}) {
  const context = {
    file: options.file ?? null,
    // 1-based line number in the SOURCE FILE of the first content line of this
    // YAML block. Defaults to 1 so block-relative and file-absolute numbering
    // coincide for standalone snippets.
    startLine:
      Number.isInteger(options.startLine) && options.startLine > 0
        ? options.startLine
        : 1,
  };
  const lines = yamlText.split(/\r?\n/u);
  const result = {};
  let index = 0;

  while (index < lines.length) {
    const rawLine = lines[index];
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) {
      index++;
      continue;
    }

    const topMatch = matchYamlField(line);
    if (!topMatch?.groups) {
      index++;
      continue;
    }

    const key = topMatch.groups.key;
    const rest = topMatch.groups.rest.trim();

    if (isBlockScalarIndicator(rest)) {
      // Block scalar: the value continues on the following, more-indented lines.
      // The established contract keeps a bare block-scalar indicator as an
      // empty list, but the continuation lines MUST be consumed so a bullet
      // inside the scalar is never mistaken for a (mis-indented) YAML list.
      result[key] = [];
      index = skipBlockScalarLines(lines, index + 1, 0);
      continue;
    }

    if (rest === '') {
      const next = peekNextNonEmptyLine(lines, index);
      if (next && /^\s*-\s/.test(next.line)) {
        const parsed = parseListBlock(lines, index + 1, next.indent, context);
        result[key] = parsed.list;
        index = parsed.nextIndex;
        continue;
      }
      if (
        next &&
        next.indent > 0 &&
        /^\s*[A-Za-z0-9_][^:]*:/u.test(next.line)
      ) {
        const parsed = parseObjectFields(lines, index, 0, true, true, context);
        result[key] = parsed.object;
        index = parsed.nextIndex;
        continue;
      }
      result[key] = [];
      index++;
      continue;
    }

    result[key] = normalizeYamlScalar(rest);
    index++;
  }

  return result;
}

/**
 * Normalize an acceptance_criteria list into test-contract objects.
 *
 * Only string `validation` values are preserved; non-string values are dropped
 * so the wire format stays compact and deterministic.
 *
 * @param {unknown} criteria - Raw acceptance criteria from plan metadata.
 * @returns {Array<{ id: string, text: string, validation?: string }>} Test contracts.
 */
export function normalizeTestContracts(criteria) {
  if (!Array.isArray(criteria)) {
    return [];
  }

  return criteria.map((criterion) => {
    const text = String(criterion.text ?? '');
    const explicitId = String(criterion.id ?? '').trim();
    const extractedId = explicitId || (text.match(/AC-\d+/)?.[0] ?? '');
    return {
      id: extractedId,
      text,
      validation:
        typeof criterion.validation === 'string'
          ? criterion.validation
          : undefined,
    };
  });
}

/**
 * Returns the next non-empty line and its indentation (number of leading spaces).
 *
 * @param lines - Line array.
 * @param startIndex - Index to search after.
 * @returns Object with `line`, `indent`, and `index`, or `null` when exhausted.
 */
function peekNextNonEmptyLine(lines, startIndex) {
  for (let i = startIndex + 1; i < lines.length; i++) {
    const line = lines[i];
    if (line === undefined) continue;
    if (line.trim() === '') continue;
    const indent = line.match(/^\s*/u)?.[0].length ?? 0;
    return { line, indent, index: i };
  }
  return null;
}

/**
 * Determines whether a scalar value is a YAML block scalar indicator.
 *
 * Block scalars (`|`, `>`, `|-`, `>-`, `|+`, `>+`) carry their value on the
 * following, more-indented lines. Plan packets use them for prose, so the
 * parser must consume those continuation lines instead of trying to read
 * them as fields or list items.
 *
 * @param {string} value - Raw scalar text (already trimmed).
 * @returns {boolean} `true` when the value opens a block scalar.
 */
function isBlockScalarIndicator(value) {
  return /^[|>][+-]?$/u.test(value);
}

/**
 * Skips the continuation lines of a block scalar.
 *
 * @param {string[]} lines - Full line array.
 * @param {number} startIndex - Index of the first continuation line.
 * @param {number} baseIndent - Indentation of the key that owns the scalar.
 * @returns {number} Index of the first line that is not part of the scalar.
 */
function skipBlockScalarLines(lines, startIndex, baseIndent) {
  let index = startIndex;
  while (index < lines.length) {
    const rawLine = lines[index];
    const line = rawLine.trim();
    if (line === '') {
      index++;
      continue;
    }
    const indent = rawLine.match(/^\s*/u)?.[0].length ?? 0;
    if (indent <= baseIndent) {
      break;
    }
    index++;
  }
  return index;
}

/**
 * Parses a YAML list block starting at a line that begins with `- `.
 *
 * Handles both scalar lists and object lists. Nested object fields may
 * themselves contain scalar lists.
 *
 * A list item whose indentation exceeds the base indentation of the list is a
 * malformed packet: it raises a {@link PlanYamlParseError} instead of being
 * folded into the surrounding object (mirrored guard for
 * `learn-20261109-f9-plan-parser-silent-skip-overindented-id-items`).
 *
 * @param lines - Full line array.
 * @param startIndex - Index of the first list item line.
 * @param baseIndent - Indentation (spaces) of the `- ` marker.
 * @param {object} [context] - Parse context carrying the source file path.
 * @returns Object with `list` and the `nextIndex` after the list.
 * @throws {PlanYamlParseError} When a list item is indented deeper than `baseIndent`.
 */
function parseListBlock(lines, startIndex, baseIndent, context = {}) {
  const list = [];
  let index = startIndex;

  while (index < lines.length) {
    const rawLine = lines[index];
    const line = rawLine.trim();
    if (line === '' || line.startsWith('#')) {
      index++;
      continue;
    }

    const currentIndent = rawLine.match(/^\s*/u)?.[0].length ?? 0;
    if (currentIndent < baseIndent || !/^-\s/.test(line)) {
      break;
    }

    const remainder = line.slice(line.indexOf('-') + 1).trim();

    if (currentIndent > baseIndent) {
      // An over-indented item that carries STRUCTURED content would be
      // dropped by the parser, so it fails loudly instead of degrading
      // silently (learning event
      // `learn-20261109-f9-plan-parser-silent-skip-overindented-id-items`).
      // An over-indented PLAIN SCALAR is tolerated: the repo's evidence
      // blocks carry scalar items at a deeper indent than their siblings,
      // and losing a scalar does not lose a criterion.
      if (isStructuredListItem(remainder)) {
        throw planYamlError(context, {
          code: 'PLAN_YAML_OVER_INDENTED_LIST_ITEM',
          message: 'List item is indented deeper than the list it belongs to',
          line: index + 1,
          indent: currentIndent,
          baseIndent,
          snippet: line,
        });
      }
      index++;
      continue;
    }

    if (isBlockScalarIndicator(remainder)) {
      // A list item whose value is a block scalar: the scalar content follows
      // on the more-indented lines. Those lines belong to the scalar and must
      // be consumed here, never read as (mis-indented) list items.
      list.push([]);
      index = skipBlockScalarLines(lines, index + 1, baseIndent);
      continue;
    }

    if (matchYamlField(remainder)) {
      // Object list item; first key is inline, remaining fields follow.
      const objectResult = parseObjectFields(
        lines,
        index,
        baseIndent,
        false,
        false,
        context,
      );
      list.push(objectResult.object);
      index = objectResult.nextIndex;
      continue;
    }

    if (/^.*:\s*$/.test(remainder)) {
      // Object list item starting on next line.
      const objectResult = parseObjectFields(
        lines,
        index,
        baseIndent,
        true,
        false,
        context,
      );
      list.push(objectResult.object);
      index = objectResult.nextIndex;
      continue;
    }

    // Scalar list item.
    list.push(normalizeYamlScalar(remainder));
    index++;
  }

  return { list, nextIndex: index };
}

/**
 * Parses the fields of one object inside a YAML list block.
 *
 * @param lines - Full line array.
 * @param startIndex - Index of the `- ` line that starts the object.
 * @param baseIndent - Indentation of the list item marker.
 * @param firstFieldOnNextLine - Whether the first field appears on the next line.
 * @param emptyDefaultToObject - When true, a bare key with no children parses
 *   as an empty object `{}`; otherwise it parses as an empty array `[]`. This
 *   distinguishes nested object contexts from list contexts.
 * @param {object} [context] - Parse context carrying the source file path.
 * @returns Object with `object` and `nextIndex` after the object.
 * @throws {PlanYamlParseError} When a list item is indented deeper than `baseIndent`.
 */
function parseObjectFields(
  lines,
  startIndex,
  baseIndent,
  firstFieldOnNextLine = false,
  emptyDefaultToObject = false,
  context = {},
) {
  const object = {};
  let index = firstFieldOnNextLine ? startIndex + 1 : startIndex;

  // Parse optional inline first key on the `- ` line.
  if (!firstFieldOnNextLine) {
    const rawLine = lines[startIndex];
    const remainder = rawLine
      .trim()
      .slice(rawLine.trim().indexOf('-') + 1)
      .trim();
    const firstMatch = matchYamlField(remainder);
    if (firstMatch?.groups) {
      const key = firstMatch.groups.key.trim();
      const rest = firstMatch.groups.rest.trim();
      if (rest === '') {
        const next = peekNextNonEmptyLine(lines, startIndex);
        if (next && /^\s*-\s/.test(next.line) && next.indent > baseIndent) {
          const parsed = parseListBlock(
            lines,
            next.index,
            next.indent,
            context,
          );
          object[key] = parsed.list;
          index = parsed.nextIndex;
        } else {
          object[key] = [];
          index = startIndex + 1;
        }
      } else if (isBlockScalarIndicator(rest)) {
        // Block scalar: consume the indented continuation lines so their
        // content is never mistaken for a stray or mis-indented list item.
        object[key] = [];
        index = skipBlockScalarLines(lines, startIndex + 1, baseIndent);
      } else {
        object[key] = normalizeYamlScalar(rest);
        index = startIndex + 1;
      }
    }
  }

  while (index < lines.length) {
    const rawLine = lines[index];
    const line = rawLine.trim();
    if (line === '' || line.startsWith('#')) {
      index++;
      continue;
    }

    const currentIndent = rawLine.match(/^\s*/u)?.[0].length ?? 0;
    if (currentIndent <= baseIndent) {
      // Back to list level or lower means end of this object.
      break;
    }

    const fieldMatch = matchYamlField(line, { allowHyphen: true });
    if (!fieldMatch?.groups) {
      // A list item that reaches this field loop is indented deeper than the
      // object that owns it. Skipping it silently (the pre-F9 behaviour) made
      // the packet corrupt without any visible symptom, so a STRUCTURED item
      // fails loudly. A plain scalar is tolerated (see
      // `isStructuredListItem`).
      if (/^-\s/.test(line)) {
        const payload = line.slice(line.indexOf('-') + 1).trim();
        if (isStructuredListItem(payload)) {
          throw planYamlError(context, {
            code: 'PLAN_YAML_OVER_INDENTED_LIST_ITEM',
            message:
              'List item is indented deeper than the object that owns it',
            line: index + 1,
            indent: currentIndent,
            baseIndent,
            snippet: line,
          });
        }
      }
      index++;
      continue;
    }

    const key = fieldMatch.groups.key;
    const rest = fieldMatch.groups.rest.trim();
    if (rest === '') {
      const next = peekNextNonEmptyLine(lines, index);
      // A YAML sequence may be indented at the SAME level as the mapping key
      // that owns it (`options:` followed by `- item` at equal indentation),
      // so the list is recognized at `>=` and not only at `>`. The stricter
      // comparison used to drop such lists silently.
      if (next && /^\s*-\s/.test(next.line) && next.indent >= currentIndent) {
        const parsed = parseListBlock(lines, next.index, next.indent, context);
        object[key] = parsed.list;
        index = parsed.nextIndex;
        continue;
      }
      if (
        next &&
        next.indent > currentIndent &&
        /^\s*[A-Za-z0-9_][^:]*:/u.test(next.line)
      ) {
        const parsed = parseObjectFields(
          lines,
          index,
          currentIndent,
          true,
          true,
          context,
        );
        object[key] = parsed.object;
        index = parsed.nextIndex;
        continue;
      }
      object[key] = emptyDefaultToObject ? {} : [];
      index++;
      continue;
    }

    if (isBlockScalarIndicator(rest)) {
      // Block scalar: consume the indented continuation lines so their
      // content is never mistaken for a stray or mis-indented list item.
      object[key] = normalizeYamlScalar(rest);
      index = skipBlockScalarLines(lines, index + 1, currentIndent);
      continue;
    }

    object[key] = normalizeYamlScalar(rest);
    index++;
  }

  return { object, nextIndex: index };
}

/**
 * Normalizes a YAML scalar string: strips matching quotes and collapses
 * bracketed empty arrays to `[]`.
 *
 * @param value - Raw scalar string.
 * @returns Normalized scalar value.
 */
export function normalizeYamlScalar(value) {
  const trimmed = value.trim();
  if (trimmed === '[]') return [];
  if (trimmed === 'true') return true;
  if (trimmed === 'false') return false;
  if (/^['"].*['"]$/u.test(trimmed)) return trimmed.slice(1, -1);
  if (/^-?\d+(?:\.\d+)?$/u.test(trimmed)) {
    const numeric = Number(trimmed);
    if (Number.isFinite(numeric)) return numeric;
  }
  return trimmed;
}

/**
 * Extracts all local Markdown link targets that start with `./` from a body string.
 *
 * Matches the pattern `[label](./relative/path)`. Absolute URLs and anchor-only
 * links are not included.
 *
 * @param body - Markdown document body (frontmatter should be excluded before calling).
 * @returns Array of relative link target strings starting with `'./'`.
 */
export function extractMarkdownLinks(body) {
  return [...body.matchAll(/\[[^\]]+]\((\.\/[^)]+)\)/g)].map(
    (match) => match[1],
  );
}

/**
 * Extracts the `[DONE|WIP|PLANNED]` status marker from a plan body.
 *
 * Matches the pattern `**Status:** [DONE]` (or `WIP` / `PLANNED`) used in
 * NeatapticTS tracker files.
 *
 * @param text - Plan document body text to scan.
 * @returns `'DONE'`, `'WIP'`, `'PLANNED'`, or `null` when the pattern is absent.
 */
export function extractStatus(text) {
  return (
    /\*\*Status:\*\* \[(?<status>DONE|WIP|PLANNED)\]/.exec(text)?.groups
      ?.status ?? null
  );
}

/**
 * Extract all repo-relative plan links from a block of Markdown text.
 *
 * Matches both `.plans.md` tracker plans and `.md` summary plans, keeping the
 * repo-relative `plans/` prefix so downstream filters can exclude index files,
 * archived completed plans, and the active plan itself.
 *
 * @param text - Markdown text to scan.
 * @returns Array of repo-relative paths like `plans/...`.
 */
function extractPlanLinks(text) {
  return [
    ...text.matchAll(/\bplans\/[A-Za-z0-9_\-\/]+(?:\.plans)?\.md\b/g),
  ].map((match) => normalizePath(match[0]));
}

/**
 * Normalize a plan link found in a Roadmap.md section to a repo-relative
 * `plans/...` path.
 *
 * Roadmap links appear in several forms: bare filenames like
 * `NEAT_Genesis_EvoDevo_AntHive_Demo.md`, completed-archive links like
 * `completed/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`, or fully-qualified
 * paths like `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`.
 *
 * @param link - Raw plan link text from the roadmap.
 * @returns Repo-relative path under `plans/`.
 */
function normalizePlanLink(link) {
  const normalized = normalizePath(link).replace(/^\.\//, '');
  if (normalized.startsWith('plans/')) return normalized;
  if (normalized.startsWith('completed/')) return `plans/${normalized}`;
  return `plans/${normalized}`;
}

/**
 * Extract plan links from the Roadmap.md phase section that contains the active
 * plan.
 *
 * Roadmap phase sections group related active and completed tracker plans. When
 * a plan body is compressed and no longer lists every downstream tracker inline
 * (as is normal for archived plans), the Roadmap section remains the durable
 * source of truth for which tracker plans belong to the same initiative.
 *
 * @param activePlanPath - Repo-relative path of the plan being analyzed.
 * @returns Array of repo-relative plan paths from the same Roadmap section.
 */
async function extractPlanLinksFromRoadmap(activePlanPath) {
  let roadmapText;
  try {
    roadmapText = await readWorkspaceFile('plans/Roadmap.md');
  } catch {
    return [];
  }

  const activeBasename = path.basename(activePlanPath);

  // Split on `## ` headings to isolate phase sections, then find the section
  // that references the active plan by basename.
  const phaseSections = roadmapText.split(/^## /m);
  for (const section of phaseSections) {
    if (!section.includes(activeBasename)) continue;

    // Markdown link targets: `[text](target)`.
    const markdownTargets = [
      ...section.matchAll(/\[([^\]]+)\]\(([^)]+)\)/g),
    ].map((match) => match[2]);
    // Stand-alone plan-looking paths that may be bare or already prefixed.
    const planPaths = [
      ...section.matchAll(/\b(?:plans\/)?[A-Za-z0-9_\-\/]+(?:\.plans)?\.md\b/g),
    ].map((match) => match[0]);

    return [
      ...new Set([...markdownTargets, ...planPaths].map(normalizePlanLink)),
    ];
  }

  return [];
}

/**
 * Extracts downstream tracker plan paths referenced by an active plan.
 *
 * Downstream trackers come from two sources:
 * 1. Repo-relative plan links inside the plan body itself.
 * 2. Plan links in the same Roadmap.md phase section, which is the durable
 *    source of truth when a plan body has been compressed.
 *
 * Results are filtered to exclude the active plan itself, index files,
 * completed archive files, and any paths that do not exist on disk.
 *
 * @param text - Plan document body text to scan.
 * @param activePlanPath - Repo-relative path of the plan being analyzed.
 * @returns Sorted array of downstream tracker paths.
 */
export async function extractDownstreamTrackers(text, activePlanPath) {
  const normalizedActive = normalizePath(activePlanPath).replace(/^\.\//, '');
  const bodyLinks = extractPlanLinks(text);
  const roadmapLinks = await extractPlanLinksFromRoadmap(normalizedActive);

  const candidates = [
    ...new Set(
      [...bodyLinks, ...roadmapLinks].filter(
        (link) => link !== normalizedActive,
      ),
    ),
  ];

  const downstream = [];
  for (const candidate of candidates) {
    if (candidate === 'plans/README.md' || candidate === 'plans/Roadmap.md') {
      continue;
    }
    if (candidate.includes('plans/completed/')) {
      continue;
    }
    if (await fileExists(candidate)) {
      downstream.push(candidate);
    }
  }

  return downstream.toSorted();
}
