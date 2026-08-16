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
 * @param yamlText - Raw YAML string (without the surrounding fence).
 * @returns Object with parsed values; lists are arrays, nested object lists are
 *   arrays of objects, and nested objects are plain objects.
 */
export function parsePlanYamlBlock(yamlText) {
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

    const topMatch = /^(?<key>[A-Za-z0-9_]+):(?<rest>.*)$/u.exec(line);
    if (!topMatch?.groups) {
      index++;
      continue;
    }

    const key = topMatch.groups.key;
    const rest = topMatch.groups.rest.trim();

    if (rest === '' || rest === '>' || rest === '|') {
      const next = peekNextNonEmptyLine(lines, index);
      if (next && /^\s*-\s/.test(next.line)) {
        const parsed = parseListBlock(lines, index + 1, next.indent);
        result[key] = parsed.list;
        index = parsed.nextIndex;
        continue;
      }
      if (next && next.indent > 0 && /^\s*[A-Za-z0-9_]+:/u.test(next.line)) {
        const parsed = parseObjectFields(lines, index, 0, true, true);
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
 * Parses a YAML list block starting at a line that begins with `- `.
 *
 * Handles both scalar lists and object lists. Nested object fields may
 * themselves contain scalar lists.
 *
 * @param lines - Full line array.
 * @param startIndex - Index of the first list item line.
 * @param baseIndent - Indentation (spaces) of the `- ` marker.
 * @returns Object with `list` and the `nextIndex` after the list.
 */
function parseListBlock(lines, startIndex, baseIndent) {
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
    if (/^(?<key>[A-Za-z0-9_]+):(?<rest>.*)$/u.test(remainder)) {
      // Object list item; first key is inline, remaining fields follow.
      const objectResult = parseObjectFields(lines, index, baseIndent);
      list.push(objectResult.object);
      index = objectResult.nextIndex;
      continue;
    }

    if (/^.*:\s*$/.test(remainder)) {
      // Object list item starting on next line.
      const objectResult = parseObjectFields(lines, index, baseIndent, true);
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
 * @returns Object with `object` and `nextIndex` after the object.
 */
function parseObjectFields(
  lines,
  startIndex,
  baseIndent,
  firstFieldOnNextLine = false,
  emptyDefaultToObject = false,
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
    const firstMatch = /^(?<key>[A-Za-z0-9_]+):(?<rest>.*)$/u.exec(remainder);
    if (firstMatch?.groups) {
      const key = firstMatch.groups.key;
      const rest = firstMatch.groups.rest.trim();
      if (rest === '') {
        const next = peekNextNonEmptyLine(lines, startIndex);
        if (next && /^\s*-\s/.test(next.line) && next.indent > baseIndent) {
          const parsed = parseListBlock(lines, next.index, next.indent);
          object[key] = parsed.list;
          index = parsed.nextIndex;
        } else {
          object[key] = emptyDefaultToObject ? {} : [];
          index = startIndex + 1;
        }
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

    const fieldMatch = /^(?<key>[A-Za-z0-9_]+):(?<rest>.*)$/u.exec(line);
    if (!fieldMatch?.groups) {
      index++;
      continue;
    }

    const key = fieldMatch.groups.key;
    const rest = fieldMatch.groups.rest.trim();
    if (rest === '') {
      const next = peekNextNonEmptyLine(lines, index);
      if (next && /^\s*-\s/.test(next.line) && next.indent > currentIndent) {
        const parsed = parseListBlock(lines, next.index, next.indent);
        object[key] = parsed.list;
        index = parsed.nextIndex;
        continue;
      }
      if (
        next &&
        next.indent > currentIndent &&
        /^\s*[A-Za-z0-9_]+:/u.test(next.line)
      ) {
        const parsed = parseObjectFields(
          lines,
          index,
          currentIndent,
          true,
          true,
        );
        object[key] = parsed.object;
        index = parsed.nextIndex;
        continue;
      }
      object[key] = emptyDefaultToObject ? {} : [];
      index++;
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
function normalizeYamlScalar(value) {
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
