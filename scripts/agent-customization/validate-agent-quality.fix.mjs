#!/usr/bin/env node
import path from 'node:path';
import { writeFile } from 'node:fs/promises';
import {
  listMarkdownFiles,
  readWorkspaceFile,
  issue,
  summarizeIssues,
  parseFrontmatter,
} from './customization-utils.mjs';

// Match only the exact canonical footer at file end (case-sensitive header)
const FOOTER_REGEX =
  /## Output format\s*\r?\n```structured-v1[\s\S]*?```\s*$/mu;
// Find any Output format heading (case-insensitive) for prefix extraction
const OUTPUT_HEADING_RE = /##\s+Output format/iu;

// Minimal per-tier required fields mapping (kept in sync with the v2 validator)
const TIER_REQUIRED_FIELDS = {
  1: [
    'OUTPUT_CONTRACT',
    'TASK_STATUS',
    'TIER',
    'ROLE',
    'TASK_RECEIVED',
    'FILES_READ',
    'FILES_CHANGED',
    'KEY_FINDINGS',
    'ACTIONS_TAKEN',
    'VALIDATION_EVIDENCE',
    'BLOCKERS',
    'RISKS_OR_GAPS',
    'LEARNING_EVENT_NEEDED',
    'SUGGESTED_NEXT_AGENT',
    'PHASE_COMPLETE',
    'SUB_ORCHESTRATORS_USED',
    'SUMMARY',
  ],
  2: [
    'OUTPUT_CONTRACT',
    'TASK_STATUS',
    'TIER',
    'ROLE',
    'TASK_RECEIVED',
    'FILES_READ',
    'FILES_CHANGED',
    'KEY_FINDINGS',
    'ACTIONS_TAKEN',
    'VALIDATION_EVIDENCE',
    'SPECIALISTS_USED',
    'HANDOFF',
    'BLOCKERS',
    'RISKS_OR_GAPS',
    'LEARNING_EVENT_NEEDED',
    'SUGGESTED_NEXT_AGENT',
    'SUMMARY',
  ],
  3: [
    'OUTPUT_CONTRACT',
    'TASK_STATUS',
    'TIER',
    'ROLE',
    'TASK_RECEIVED',
    'FILES_READ',
    'FILES_CHANGED',
    'KEY_FINDINGS',
    'ACTIONS_TAKEN',
    'VALIDATION_EVIDENCE',
    'HANDOFF',
    'BLOCKERS',
    'RISKS_OR_GAPS',
    'LEARNING_EVENT_NEEDED',
    'SUGGESTED_NEXT_AGENT',
    'SUMMARY',
  ],
  4: [
    'OUTPUT_CONTRACT',
    'TASK_STATUS',
    'TIER',
    'ROLE',
    'TASK_RECEIVED',
    'FILES_READ',
    'FILES_CHANGED',
    'KEY_FINDINGS',
    'ACTIONS_TAKEN',
    'BLOCKERS',
    'RISKS_OR_GAPS',
    'LEARNING_EVENT_NEEDED',
    'SUGGESTED_NEXT_AGENT',
    'SUMMARY',
  ],
};

function makeCanonicalFooter(tier, role) {
  const t = String(tier ?? '').trim() || '';
  const r = String(role ?? '').trim() || '';
  const fields = TIER_REQUIRED_FIELDS[t] ?? TIER_REQUIRED_FIELDS['1'];

  const fieldRenderer = {
    OUTPUT_CONTRACT: () => 'OUTPUT_CONTRACT: structured-v1',
    TASK_STATUS: () => 'TASK_STATUS: SUCCESS | PARTIAL | FAILED',
    TIER: () => `TIER: ${t}`,
    ROLE: () => `ROLE: ${r}`,
    TASK_RECEIVED: () => 'TASK_RECEIVED: <brief restatement>',
    FILES_READ: () => 'FILES_READ:\n- <path or NONE>',
    FILES_CHANGED: () => 'FILES_CHANGED:\n- <path or NONE>',
    KEY_FINDINGS: () => 'KEY_FINDINGS:\n- <finding or NONE>',
    ACTIONS_TAKEN: () => 'ACTIONS_TAKEN:\n- <action or NONE>',
    VALIDATION_EVIDENCE: () =>
      'VALIDATION_EVIDENCE:\n- <command/result or NOT RUN>',
    BLOCKERS: () => 'BLOCKERS:\n- <blocker or NONE>',
    RISKS_OR_GAPS: () => 'RISKS_OR_GAPS:\n- <risk or NONE>',
    LEARNING_EVENT_NEEDED: () => 'LEARNING_EVENT_NEEDED: true | false',
    SUGGESTED_NEXT_AGENT: () => 'SUGGESTED_NEXT_AGENT: <agent name or NONE>',
    PHASE_COMPLETE: () => 'PHASE_COMPLETE: true | false',
    SUB_ORCHESTRATORS_USED: () => 'SUB_ORCHESTRATORS_USED:\n- <agent or NONE>',
    SUMMARY: () => 'SUMMARY: <brief truthful summary>',
    SPECIALISTS_USED: () => 'SPECIALISTS_USED:\n- <agent or NONE>',
    HANDOFF: () => 'HANDOFF: <next step, reroute, or NONE>',
  };

  const lines = ['## Output format', '', '```structured-v1'];
  for (const f of fields) {
    const renderer = fieldRenderer[f];
    lines.push(renderer ? renderer() : `${f}: <value>`);
  }
  lines.push('```', '');
  return lines.join('\n');
}

export async function runFix({ json = false } = {}) {
  const agentPaths = await listMarkdownFiles('.github/agents', (relativePath) =>
    relativePath.endsWith('.agent.md'),
  );

  const report = {
    name: 'agent-quality-footer-fix',
    ok: true,
    issues: [],
    fixed: [],
    unchanged: [],
    counts: { fixed: 0, unchanged: 0, checked: agentPaths.length },
  };

  for (const relativePath of agentPaths) {
    const text = await readWorkspaceFile(relativePath);
    const normalized = text.replace(/\uFEFF/, '');
    const hasFrontmatter = normalized.trimStart().startsWith('---');
    // Always attempt to produce a canonical footer when frontmatter is present.
    if (!hasFrontmatter) {
      report.issues.push(
        issue(
          'error',
          relativePath,
          'Missing YAML frontmatter; skipping auto-fix',
        ),
      );
      continue;
    }

    // Gather metadata for a better canonical footer
    let parsed = { data: {} };
    try {
      parsed = parseFrontmatter(normalized, relativePath);
    } catch (err) {
      // ignore parse errors; we will still attempt a safe fix using filename defaults
    }
    const role =
      parsed.data?.name ??
      relativePath.split('/').at(-1)?.replace('.agent.md', '') ??
      '';
    const tier = parsed.data?.tier ?? '';

    // Attempt fix: preserve content up to any existing '## Output format' heading
    // Use case-insensitive search for existing heading variants and replace with canonical
    const match = OUTPUT_HEADING_RE.exec(normalized);
    const prefix = match ? normalized.slice(0, match.index) : normalized;
    const newFooter = makeCanonicalFooter(String(tier), String(role));
    const newContent = prefix.replace(/\s+$/u, '') + '\n\n' + newFooter;

    try {
      await writeFile(
        path.join(process.cwd(), relativePath),
        newContent,
        'utf8',
      );
      report.fixed.push(relativePath);
    } catch (err) {
      report.ok = false;
      report.issues.push(
        issue(
          'error',
          relativePath,
          `Failed to write fixed file: ${String(err)}`,
        ),
      );
    }
  }

  report.counts.fixed = report.fixed.length;
  report.counts.unchanged = report.unchanged.length;
  report.ok = report.issues.filter((i) => i.severity === 'error').length === 0;

  return report;
}
