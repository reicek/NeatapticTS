#!/usr/bin/env node
import {
  extractStatus,
  issue,
  parseArgs,
  printUsage,
  readWorkspaceFile,
  summarizeIssues,
  writeReport,
} from './customization-utils.mjs';

const options = parseArgs(process.argv.slice(2));

if (options.help) {
  printUsage({
    title: 'Validate agentic workflow plan registration.',
    usage: 'node scripts/agent-customization/validate-plan-sync.mjs [--json] [--plan=plans/Agentic_Workflow_Architecture.plans.md]',
    options: [['--plan=<path>', 'Plan file to compare against the index and roadmap.']],
  });
  process.exit(0);
}

const planPath = options.plan;
const planText = await readWorkspaceFile(planPath);
const readmeText = await readWorkspaceFile('plans/README.md');
const roadmapText = await readWorkspaceFile('plans/Roadmap.md');
const planStatus = extractStatus(planText);
const issues = [];

if (!planStatus) {
  issues.push(issue('error', planPath, 'Plan is missing a top-level status line.'));
}

for (const [path, text] of [
  ['plans/README.md', readmeText],
  ['plans/Roadmap.md', roadmapText],
]) {
  if (!text.includes(planPath.replace('plans/', ''))) {
    issues.push(issue('error', path, `Missing reference to ${planPath}.`));
  }
  if (planStatus && !text.includes(`[${planStatus}]`)) {
    issues.push(issue('error', path, `Missing status [${planStatus}] for ${planPath}.`));
  }
}

if (!readmeText.includes('agent architecture, custom agents')) {
  issues.push(issue('warning', 'plans/README.md', 'Missing agent-customization trigger phrase entry.'));
}

if (!roadmapText.includes('Standalone Meta-Workflow Lane')) {
  issues.push(issue('error', 'plans/Roadmap.md', 'Missing standalone meta-workflow lane section.'));
}

const report = {
  ...summarizeIssues('plan sync', issues),
  plan: { path: planPath, status: planStatus },
};

writeReport(report, options);
process.exitCode = report.ok ? 0 : 1;