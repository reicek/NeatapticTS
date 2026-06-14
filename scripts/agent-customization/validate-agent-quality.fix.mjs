#!/usr/bin/env node
import path from 'node:path';
import { writeFile } from 'node:fs/promises';
import {
  listMarkdownFiles,
  readWorkspaceFile,
  issue,
  summarizeIssues,
} from './customization-utils.mjs';

const FOOTER_REGEX = /## Output format\s*\r?\n```structured-v1[\s\S]*?```\s*$/im;
const OUTPUT_HEADING_RE = /##\s+Output format/iu;

const CANONICAL_FOOTER = `## Output format

```structured-v1
OUTPUT_CONTRACT: structured-v1
TASK_STATUS: SUCCESS | PARTIAL | FAILED
TIER: 1
ROLE: 00-helping
TASK_RECEIVED: <brief restatement>
FILES_READ:
- <path or NONE>
FILES_CHANGED:
- <path or NONE>
KEY_FINDINGS:
- <finding or NONE>
ACTIONS_TAKEN:
- <action or NONE>
VALIDATION_EVIDENCE:
- <command/result or NOT RUN>
BLOCKERS:
- <blocker or NONE>
RISKS_OR_GAPS:
- <risk or NONE>
LEARNING_EVENT_NEEDED: true | false
SUGGESTED_NEXT_AGENT: <agent name or NONE>
PHASE_COMPLETE: true | false
SUB_ORCHESTRATORS_USED:
- <agent or NONE>
SUMMARY: <brief truthful summary>
```
`;

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
    const hasFooter = FOOTER_REGEX.test(normalized);

    if (hasFrontmatter && hasFooter) {
      report.unchanged.push(relativePath);
      continue;
    }

    // Record issue
    const missing = [];
    if (!hasFrontmatter) missing.push('missing-frontmatter');
    if (!hasFooter) missing.push('missing-footer');
    report.issues.push(
      issue(
        'error',
        relativePath,
        `Agent file missing required parts: ${missing.join(', ')}`,
      ),
    );

    // Attempt fix: preserve content up to any existing '## Output format' heading
    const match = OUTPUT_HEADING_RE.exec(normalized);
    const prefix = match ? normalized.slice(0, match.index) : normalized;
    const newContent = prefix.replace(/\s+$/u, '') + '\n\n' + CANONICAL_FOOTER;

    try {
      await writeFile(path.join(process.cwd(), relativePath), newContent, 'utf8');
      report.fixed.push(relativePath);
    } catch (err) {
      report.ok = false;
      report.issues.push(
        issue('error', relativePath, `Failed to write fixed file: ${String(err)}`),
      );
    }
  }

  report.counts.fixed = report.fixed.length;
  report.counts.unchanged = report.unchanged.length;
  report.ok = report.issues.filter((i) => i.severity === 'error').length === 0;

  return report;
}
