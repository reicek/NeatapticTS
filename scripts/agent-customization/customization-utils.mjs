import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';

export const repoRoot = process.cwd();

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

export function parseArgs(argv) {
  const options = {
    json: false,
    strict: false,
    help: false,
    plan: 'plans/Agentic_Workflow_Architecture.plans.md',
  };

  for (const rawArg of argv) {
    if (rawArg === '--json') options.json = true;
    else if (rawArg === '--strict') options.strict = true;
    else if (rawArg === '--help' || rawArg === '-h') options.help = true;
    else if (rawArg.startsWith('--input=')) options.input = rawArg.slice('--input='.length);
    else if (rawArg.startsWith('--plan=')) options.plan = rawArg.slice('--plan='.length);
  }

  return options;
}

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

export async function readWorkspaceFile(relativePath) {
  return readFile(path.join(repoRoot, relativePath), 'utf8');
}

export async function fileExists(relativePath) {
  try {
    return (await stat(path.join(repoRoot, relativePath))).isFile();
  } catch {
    return false;
  }
}

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

export function parseFrontmatterValue(value) {
  if (value === '') return true;
  if (value === 'true') return true;
  if (value === 'false') return false;
  if (/^['"].*['"]$/.test(value)) return value.slice(1, -1);
  if (/^\[.*\]$/.test(value)) return parseInlineArray(value);
  return value.replace(/\s+#.*$/, '').trim();
}

export function parseInlineArray(value) {
  const inner = value.slice(1, -1).trim();
  if (!inner) return [];

  return inner
    .split(',')
    .map((item) => item.trim().replace(/^['"]|['"]$/g, ''))
    .filter(Boolean);
}

export function normalizePath(value) {
  return value.replaceAll(path.sep, '/');
}

export function issue(severity, relativePath, message) {
  return { severity, path: relativePath, message };
}

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

export function extractMarkdownLinks(body) {
  return [...body.matchAll(/\[[^\]]+]\((\.\/[^)]+)\)/g)].map((match) => match[1]);
}

export function extractStatus(text) {
  return /\*\*Status:\*\* \[(?<status>DONE|WIP|PLANNED)\]/.exec(text)?.groups?.status ?? null;
}