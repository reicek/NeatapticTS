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
    usage:
      'node scripts/agent-customization/validate-plan-sync.mjs --plan=<path> [--json]',
    options: [
      [
        '--plan=<path>',
        '(Required) Active plan file to compare against the index and roadmap.',
      ],
    ],
  });
  process.exit(0);
}

// Require --plan to be supplied explicitly so the validator never silently
// targets the completed archive plan instead of the active plan.
const planExplicit = process.argv
  .slice(2)
  .some((arg) => arg.startsWith('--plan='));
if (!planExplicit) {
  console.error(
    'ERROR: --plan=<path> is required. Pass the active plan path explicitly, e.g. --plan=plans/NEATchat.plans.md',
  );
  process.exitCode = 1;
  process.exit(1);
}

// Normalize the plan path so Windows-style backslash invocations are handled
// the same as POSIX-style forward-slash invocations.
const planPath = options.plan.replace(/\\/g, '/');
const planText = await readWorkspaceFile(planPath);
const readmeText = await readWorkspaceFile('plans/README.md');
const roadmapText = await readWorkspaceFile('plans/Roadmap.md');
const planStatus = extractStatus(planText);
const issues = [];

if (!planStatus) {
  issues.push(
    issue('error', planPath, 'Plan is missing a top-level status line.'),
  );
}

for (const [path, text] of [
  ['plans/README.md', readmeText],
  ['plans/Roadmap.md', roadmapText],
]) {
  if (!text.includes(planPath.replace('plans/', ''))) {
    issues.push(issue('error', path, `Missing reference to ${planPath}.`));
  }
  if (planStatus && !text.includes(`[${planStatus}]`)) {
    issues.push(
      issue('error', path, `Missing status [${planStatus}] for ${planPath}.`),
    );
  }
}

if (!readmeText.includes('agent architecture, custom agents')) {
  issues.push(
    issue(
      'warning',
      'plans/README.md',
      'Missing agent-customization trigger phrase entry.',
    ),
  );
}

if (!roadmapText.includes('Standalone Meta-Workflow Lane')) {
  issues.push(
    issue(
      'error',
      'plans/Roadmap.md',
      'Missing standalone meta-workflow lane section.',
    ),
  );
}

const baseReport = summarizeIssues('plan sync', issues);
const report = {
  ...baseReport,
  // Surface the validated plan path so callers can confirm the right plan was checked.
  summaryText: `${baseReport.summaryText} (plan: ${planPath})`,
  plan: { path: planPath, status: planStatus },
};

writeReport(report, options);
process.exitCode = report.ok ? 0 : 1;
