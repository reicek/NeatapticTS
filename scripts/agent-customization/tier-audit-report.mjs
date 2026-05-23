#!/usr/bin/env node
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { parseArgs, printUsage } from './customization-utils.mjs';
import { collectTierInventory, runValidateAgentGraph } from './validate-agent-graph.mjs';

const options = parseCliOptions(process.argv.slice(2));

if (options.help) {
  printUsage({
    title: 'Generate a markdown or JSON audit report for agent delegation tiers.',
    usage: 'node scripts/agent-customization/tier-audit-report.mjs [--json] [--markdown]',
    options: [['--markdown', 'Write the audit report as Markdown (default when --json is omitted).']],
  });
  process.exit(0);
}

export async function runTierAuditReport({ workspaceRoot = process.cwd() } = {}) {
  const inventory = await collectTierInventory({ workspaceRoot });
  const validation = await runValidateAgentGraph({ workspaceRoot });

  return {
    name: 'tier audit report',
    ok: validation.ok,
    generated_at: inventory.generated_at,
    summary: inventory.summary,
    agents: inventory.agents,
    issues: validation.issues,
    notes: [
      'Frontmatter `tier` is the delegation-tier policy field audited here.',
      'Uppercase `TIER` inside structured-v1 output contracts is a prompt-contract field and is not treated as frontmatter.',
    ],
  };
}

async function main() {
  const report = await runTierAuditReport();

  if (options.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(renderMarkdownReport(report));
  }

  process.exitCode = report.ok ? 0 : 1;
}

function parseCliOptions(argv) {
  const base = parseArgs(argv);
  return {
    ...base,
    markdown: argv.includes('--markdown') || !base.json,
  };
}

function renderMarkdownReport(report) {
  const markdownLines = [
    '# Tier Audit Report',
    '',
    `- Generated at: ${report.generated_at}`,
    `- Total agents: ${report.summary.total}`,
    `- Tier counts: Tier 1=${report.summary.by_tier[1]}, Tier 2=${report.summary.by_tier[2]}, Tier 3=${report.summary.by_tier[3]}, Tier 4=${report.summary.by_tier[4]}`,
    `- User-invocable agents: ${report.summary.user_invocable_total}`,
    `- Violations: ${report.summary.violation_count}`,
    '',
    '## Notes',
    '',
    ...report.notes.map((note) => `- ${note}`),
    '',
    '## Agents',
    '',
    '| Agent | Tier | User-invocable | Delegates to | Violations |',
    '| --- | --- | --- | --- | --- |',
    ...report.agents.map((agent) => {
      const delegationText = agent.delegates_to.length === 0 ? '—' : agent.delegates_to.join(', ');
      const violationText = agent.violations.length === 0
        ? '—'
        : agent.violations.map((currentIssue) => currentIssue.message).join('<br>');

      return `| ${agent.name} | ${agent.tier} (${agent.tier_label}) | ${agent.user_invocable ? 'true' : 'false'} | ${delegationText} | ${violationText} |`;
    }),
    '',
    '## Violations',
    '',
    ...(report.issues.length === 0
      ? ['- None']
      : report.issues.map((currentIssue) => `- ${currentIssue.path}: ${currentIssue.message}`)),
  ];

  return markdownLines.join('\n');
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await main();
}