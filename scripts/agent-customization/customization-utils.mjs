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
    contract: 'tier0',
    plan: 'plans/completed/Agentic_Workflow_Architecture.plans.md',
  };

  for (const rawArg of argv) {
    if (rawArg === '--json') options.json = true;
    else if (rawArg === '--strict') options.strict = true;
    else if (rawArg === '--help' || rawArg === '-h') options.help = true;
    else if (rawArg.startsWith('--contract=')) options.contract = rawArg.slice('--contract='.length);
    else if (rawArg.startsWith('--input=')) options.input = rawArg.slice('--input='.length);
    else if (rawArg.startsWith('--plan=')) options.plan = rawArg.slice('--plan='.length);
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
    ['--strict', 'Enforce the final target architecture rather than auditing the current state.'],
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
  if (json) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }

  console.log(report.summaryText ?? `${report.ok ? 'PASS' : 'FAIL'} ${report.name}`);
  for (const issue of report.issues ?? []) {
    console.log(`- ${issue.severity.toUpperCase()}: ${issue.path}: ${issue.message}`);
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
      if (entry.isFile() && predicate(relativePath)) discovered.push(relativePath);
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
      issues: [issue('error', relativePath, 'Missing opening YAML frontmatter fence.')],
    };
  }

  const endIndex = lines.findIndex((line, index) => index > 0 && line === '---');
  if (endIndex === -1) {
    return {
      data: {},
      body: normalized,
      raw: lines.slice(1).join('\n'),
      issues: [issue('error', relativePath, 'Missing closing YAML frontmatter fence.')],
    };
  }

  const rawLines = lines.slice(1, endIndex);
  const data = {};
  const issues = [];

  for (const line of rawLines) {
    if (!line.trim() || line.trim().startsWith('#')) continue;
    if (/^\s+-\s/.test(line)) continue;
    if (/^\s+/.test(line)) continue;

    const match = /^(?<key>[A-Za-z0-9_-]+):(?<value>.*)$/.exec(line);
    if (!match?.groups) {
      issues.push(issue('warning', relativePath, `Could not parse frontmatter line: ${line}`));
      continue;
    }

    const key = match.groups.key;
    const value = match.groups.value.trim();
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
  const errors = issues.filter((currentIssue) => currentIssue.severity === 'error').length;
  const warnings = issues.filter((currentIssue) => currentIssue.severity === 'warning').length;
  return {
    name,
    ok: errors === 0,
    issues,
    counts: { errors, warnings },
    summaryText: `${errors === 0 ? 'PASS' : 'FAIL'} ${name}: ${errors} errors, ${warnings} warnings`,
  };
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
  return [...body.matchAll(/\[[^\]]+]\((\.\/[^)]+)\)/g)].map((match) => match[1]);
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
  return /\*\*Status:\*\* \[(?<status>DONE|WIP|PLANNED)\]/.exec(text)?.groups?.status ?? null;
}