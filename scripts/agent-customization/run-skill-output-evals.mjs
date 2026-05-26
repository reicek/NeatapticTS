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
    title: 'Grade NeatapticTS skill output eval assertions.',
    usage: 'node scripts/agent-customization/run-skill-output-evals.mjs [--json] [--input=scripts/agent-customization/evals/skill-output-evals.json]',
    options: [['--input=<path>', 'Output eval fixture with assertion results and evidence.']],
  });
  process.exit(0);
}

const inputPath = options.input ?? 'scripts/agent-customization/evals/skill-output-evals.json';
const fixture = JSON.parse(await readWorkspaceFile(inputPath));
const evals = Array.isArray(fixture) ? fixture : fixture.evals ?? [];
const issues = [];
const results = evals.map(gradeEval);

if (evals.length === 0) {
  issues.push(issue('error', inputPath, 'Output eval fixture has no evals.'));
}

const failedAssertions = results.reduce((total, result) => total + result.failed, 0);
const pendingAssertions = results.reduce((total, result) => total + result.pending, 0);
if (failedAssertions > 0) issues.push(issue('error', inputPath, `${failedAssertions} output assertions failed.`));
if (pendingAssertions > 0) issues.push(issue('warning', inputPath, `${pendingAssertions} output assertions are pending evidence.`));

const report = {
  ...summarizeIssues('skill output evals', issues),
  input: inputPath,
  summary: {
    evals: results.length,
    assertions: results.reduce((total, result) => total + result.total, 0),
    passed: results.reduce((total, result) => total + result.passed, 0),
    failed: failedAssertions,
    pending: pendingAssertions,
  },
  results,
};

writeReport(report, options);
process.exitCode = report.counts.errors === 0 ? 0 : 1;

function gradeEval(evalCase) {
  const assertions = evalCase.assertions ?? [];
  const graded = assertions.map((assertion) => ({
    text: assertion.text,
    passed: typeof assertion.passed === 'boolean' ? assertion.passed : null,
    evidence: assertion.evidence ?? '',
  }));
  return {
    id: evalCase.id,
    target: evalCase.target,
    prompt: evalCase.prompt,
    total: graded.length,
    passed: graded.filter((assertion) => assertion.passed === true).length,
    failed: graded.filter((assertion) => assertion.passed === false).length,
    pending: graded.filter((assertion) => assertion.passed === null).length,
    assertions: graded,
  };
}