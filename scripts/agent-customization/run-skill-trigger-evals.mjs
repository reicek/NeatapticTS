#!/usr/bin/env node
import {
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
    title: 'Grade NeatapticTS skill and agent trigger eval observations.',
    usage: 'node scripts/agent-customization/run-skill-trigger-evals.mjs [--json] [--input=scripts/agent-customization/evals/skill-trigger-evals.json]',
    options: [['--input=<path>', 'Trigger eval fixture with expected and optional observed trigger results.']],
  });
  process.exit(0);
}

const inputPath = options.input ?? 'scripts/agent-customization/evals/skill-trigger-evals.json';
const fixture = JSON.parse(await readWorkspaceFile(inputPath));
const evals = Array.isArray(fixture) ? fixture : fixture.evals ?? [];
const issues = [];
const results = evals.map(gradeEval);

if (evals.length === 0) {
  issues.push(issue('error', inputPath, 'Trigger eval fixture has no evals.'));
}

const failures = results.filter((result) => result.passed === false).length;
const pending = results.filter((result) => result.passed === null).length;
if (failures > 0) issues.push(issue('error', inputPath, `${failures} trigger evals failed.`));
if (pending > 0) issues.push(issue('warning', inputPath, `${pending} trigger evals are pending observed results.`));

const report = {
  ...summarizeIssues('skill trigger evals', issues),
  input: inputPath,
  summary: {
    total: results.length,
    passed: results.filter((result) => result.passed === true).length,
    failed: failures,
    pending,
  },
  results,
};

writeReport(report, options);
process.exitCode = report.counts.errors === 0 ? 0 : 1;

function gradeEval(evalCase) {
  const observed = evalCase.observedTriggered;
  const expected = Boolean(evalCase.shouldTrigger);
  const passed = typeof observed === 'boolean' ? observed === expected : null;
  return {
    id: evalCase.id,
    target: evalCase.target,
    query: evalCase.query,
    shouldTrigger: expected,
    observedTriggered: observed ?? null,
    passed,
  };
}