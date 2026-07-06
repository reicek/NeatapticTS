#!/usr/bin/env node
/**
 * RED-phase check for Spec-Kit Assimilation Slice 11.
 *
 * Asserts AC-040..AC-046:
 *  - AC-040..AC-043: named flow YAML files include the new constitutional and
 *    spec-checklist gate / learning-event references.
 *  - AC-044..AC-046: customization validators and routing-table regeneration
 *    remain green.
 *
 * This script must FAIL before implementation and PASS after.
 */

import { readFileSync, existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const checks = [];

function check(id, condition, message) {
  if (!condition) {
    checks.push({ id, pass: false, message });
    console.error(`FAIL ${id}: ${message}`);
  } else {
    checks.push({ id, pass: true, message: 'ok' });
    console.log(`PASS ${id}`);
  }
}

function fileText(path) {
  return existsSync(path) ? readFileSync(path, 'utf-8') : '';
}

function runCommand(label, command, args = []) {
  const result = spawnSync(command, args, {
    shell: process.platform === 'win32',
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  const stdout = result.stdout?.toString('utf-8') ?? '';
  const stderr = result.stderr?.toString('utf-8') ?? '';
  const exitCode = result.status ?? 1;
  return { label, exitCode, stdout, stderr };
}

const flowFiles = {
  phaseKickoff: '.github/flows/01.phase-kickoff.flow.yml',
  stepExpansion: '.github/flows/01.step-expansion.flow.yml',
  stepPacketRevision: '.github/flows/01.step-packet-revision.flow.yml',
  coverageGuard: '.github/flows/05.coverage-guard.flow.yml',
  ciGreenConfirmation: '.github/flows/05.ci-green-confirmation.flow.yml',
  trackerClosure: '.github/flows/07.tracker-closure.flow.yml',
};

const texts = Object.fromEntries(
  Object.entries(flowFiles).map(([key, path]) => [key, fileText(path)]),
);

// AC-040: phase-kickoff references the constitutional exit gate.
check(
  'AC-040',
  texts.phaseKickoff.includes('constitution_check'),
  `${flowFiles.phaseKickoff} must reference 'constitution_check'`,
);

// AC-041: step-expansion and step-packet-revision reference spec-checklist.
check(
  'AC-041-a',
  texts.stepExpansion.includes('spec-checklist'),
  `${flowFiles.stepExpansion} must reference 'spec-checklist'`,
);
check(
  'AC-041-b',
  texts.stepPacketRevision.includes('spec-checklist'),
  `${flowFiles.stepPacketRevision} must reference 'spec-checklist'`,
);

// AC-042: coverage-guard and ci-green-confirmation reference both gates.
check(
  'AC-042-a-coverage',
  texts.coverageGuard.includes('constitution_check') &&
    texts.coverageGuard.includes('spec-checklist'),
  `${flowFiles.coverageGuard} must reference both 'constitution_check' and 'spec-checklist'`,
);
check(
  'AC-042-b-ci',
  texts.ciGreenConfirmation.includes('constitution_check') &&
    texts.ciGreenConfirmation.includes('spec-checklist'),
  `${flowFiles.ciGreenConfirmation} must reference both 'constitution_check' and 'spec-checklist'`,
);

// AC-043: tracker-closure records the three required learning-event strings.
check(
  'AC-043',
  texts.trackerClosure.includes('constitution-update') &&
    texts.trackerClosure.includes('spec-checklist') &&
    texts.trackerClosure.includes('gate-run'),
  `${flowFiles.trackerClosure} must record 'constitution-update', 'spec-checklist', and 'gate-run' learning events`,
);

// AC-044: agent frontmatter validator passes in strict mode.
const ac044 = runCommand('AC-044', 'node', [
  'scripts/agent-customization/validate-agent-frontmatter.mjs',
  '--json',
  '--strict',
]);
check(
  'AC-044',
  ac044.exitCode === 0,
  `validate-agent-frontmatter --strict exited ${ac044.exitCode}: ${ac044.stderr || ac044.stdout}`,
);

// AC-045: agent graph validator passes.
const ac045 = runCommand('AC-045', 'node', [
  'scripts/agent-customization/validate-agent-graph.mjs',
  '--json',
]);
check(
  'AC-045',
  ac045.exitCode === 0,
  `validate-agent-graph exited ${ac045.exitCode}: ${ac045.stderr || ac045.stdout}`,
);

// AC-046: routing table regenerates with no new diff and its gate passes.
// "No diff" means the generator does not alter the existing routing table file
// (changed=false). The table may carry uncommitted updates from earlier slices;
// what matters for this slice is that regeneration is stable.
const ac046Regen = runCommand('AC-046-regen', 'npm', [
  'run',
  'agents:routing-table',
]);
const ac046Gate = runCommand('AC-046-gate', 'npm', [
  'run',
  'agents:routing-table:gate',
]);
const routingTableStable =
  ac046Regen.exitCode === 0 && /changed=false\b/.test(ac046Regen.stdout);
check(
  'AC-046-a',
  routingTableStable,
  `routing-table must regenerate without changing the file (regen exit=${ac046Regen.exitCode}, stdout=${JSON.stringify(ac046Regen.stdout)})`,
);
const gateOutput = ac046Gate.stdout || '{}';
const gatePass = ac046Gate.exitCode === 0 && /"pass":\s*true/.test(gateOutput);
check(
  'AC-046-b',
  gatePass,
  `agents:routing-table:gate must pass (exit=${ac046Gate.exitCode}): ${ac046Gate.stderr || ac046Gate.stdout}`,
);

const failed = checks.filter((c) => !c.pass);
if (failed.length > 0) {
  console.error(
    `\nRED check failed: ${failed.length}/${checks.length} AC(s) not satisfied.`,
  );
  process.exit(1);
}

console.log(
  `\nGREEN check passed: ${checks.length}/${checks.length} AC(s) satisfied.`,
);
process.exit(0);
