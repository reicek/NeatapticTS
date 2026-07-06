#!/usr/bin/env node
/**
 * RED-phase check for Spec-Kit Assimilation Slice 5.
 *
 * Asserts AC-018..AC-022 for the spec-checklist "unit tests for English" skill.
 * This script must FAIL before implementation and PASS after.
 */

import { readFileSync, existsSync } from 'node:fs';

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

const skillPath = '.github/skills/spec-checklist/SKILL.md';
const planningAgentPath = '.github/agents/01-planning.agent.md';
const greenAgentPath = '.github/agents/05-green-testing.agent.md';
const phaseHandoffPath = '.github/skills/phase-handoff-workflow/SKILL.md';

const skillExists = existsSync(skillPath);
let skillText = '';
if (skillExists) {
  skillText = readFileSync(skillPath, 'utf-8');
}

// AC-018: skill file exists and calls itself "unit tests for English".
check(
  'AC-018',
  skillExists && /unit tests for English/i.test(skillText),
  skillExists
    ? `expected skill to describe itself as "unit tests for English"`
    : `${skillPath} does not exist`,
);

// AC-019: the four gap types are defined with one-line definitions.
const gapTypes = ['missing', 'partial', 'contradicts', 'unrequested'];
let gapTypesOk = true;
for (const gap of gapTypes) {
  const hasType = new RegExp(`\\b${gap}\\b`, 'i').test(skillText);
  if (!hasType) {
    gapTypesOk = false;
    console.error(`  gap type "${gap}" not found in skill text`);
  }
}
check(
  'AC-019',
  skillExists && gapTypesOk,
  `expected definitions for gap types: ${gapTypes.join(', ')}`,
);

// AC-020: requires >= 80% traceability coverage before implementation dispatch.
check(
  'AC-020',
  skillExists &&
    /(?:>=|≥)\s*80%|80\s*%|eighty percent/i.test(skillText) &&
    /traceab/i.test(skillText),
  `expected skill to require >= 80% traceability coverage before implementation`,
);

// AC-021: both 01-planning and 05-green-testing list spec-checklist in skills.
const planningText = readFileSync(planningAgentPath, 'utf-8');
const greenText = readFileSync(greenAgentPath, 'utf-8');
const planningSkillsMatch = planningText.match(/skills:\s*\[([\s\S]*?)\]/);
const greenSkillsMatch = greenText.match(/skills:\s*\[([\s\S]*?)\]/);
const planningHasSpecChecklist =
  planningSkillsMatch && /['"]spec-checklist['"]/.test(planningSkillsMatch[1]);
const greenHasSpecChecklist =
  greenSkillsMatch && /['"]spec-checklist['"]/.test(greenSkillsMatch[1]);
check(
  'AC-021',
  planningHasSpecChecklist && greenHasSpecChecklist,
  `01-planning has spec-checklist=${planningHasSpecChecklist}, 05-green-testing has spec-checklist=${greenHasSpecChecklist}`,
);

// AC-022: phase-handoff-workflow lists spec-checklist as a pre-implementation gate.
const handoffText = readFileSync(phaseHandoffPath, 'utf-8');
const handoffHasSpecChecklist = /spec-checklist/i.test(handoffText);
const handoffPreImplementation =
  /pre-implementation|before implementation dispatch/i.test(handoffText);
check(
  'AC-022',
  handoffHasSpecChecklist && handoffPreImplementation,
  `phase-handoff mentions spec-checklist=${handoffHasSpecChecklist}, pre-implementation language=${handoffPreImplementation}`,
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
