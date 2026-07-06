#!/usr/bin/env node
/**
 * RED-phase check for Spec-Kit Assimilation Slice 1.
 *
 * Asserts AC-001..AC-005 for the constitution foundation slice.
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

const constitutionPath = 'plans/constitution.md';
const planningAgentPath = '.github/agents/01-planning.agent.md';
const phaseHandoffPath = '.github/skills/phase-handoff-workflow/SKILL.md';
const learningEventPath = '.github/skills/capturing-learning-event/SKILL.md';

// AC-001: plans/constitution.md exists with five Core Principles and three supporting sections.
const constitutionExists = existsSync(constitutionPath);
let constitutionText = '';
if (constitutionExists) {
  constitutionText = readFileSync(constitutionPath, 'utf-8');
}

const corePrincipleHeaders = (
  constitutionText.match(
    /^###?\s+\d+\.\s+.*Core Principle|Core Principles.*$/gim,
  ) || []
).length;
// Count numbered principle subsections: "### 1. ..." or "## Core Principles" style.
const principlePattern =
  /^#{1,3}\s+(?:(?:\d+\.\s+)|(?:.*Core Principle)).*$/gim;
const principleMatches = (constitutionText.match(principlePattern) || [])
  .length;
const hasSecuritySection =
  /#{1,3}\s*Security\s*(?:&|and)\s*Cross-Platform\s*Constraints/i.test(
    constitutionText,
  );
const hasWorkflowSection =
  /#{1,3}\s*Development\s*Workflow\s*(?:&|and)\s*Quality\s*Gates/i.test(
    constitutionText,
  );
const hasGovernanceSection = /#{1,3}\s*Governance/i.test(constitutionText);

// We expect at least 5 principle headings and the 3 supporting sections.
const hasFivePrinciples = principleMatches >= 5 || corePrincipleHeaders >= 5;
const hasThreeSupporting =
  hasSecuritySection && hasWorkflowSection && hasGovernanceSection;

check(
  'AC-001',
  constitutionExists && hasFivePrinciples && hasThreeSupporting,
  constitutionExists
    ? `expected ≥5 principle sections (found ${principleMatches}) plus Security, Development Workflow, and Governance sections (found security=${hasSecuritySection}, workflow=${hasWorkflowSection}, governance=${hasGovernanceSection})`
    : `${constitutionPath} does not exist`,
);

// AC-002: constitution authority phrase and SemVer ratification block.
const hasConstitutionAuthority = constitutionText.includes(
  'constitution authority',
);
// Markdown bold may wrap the colon, so allow * and whitespace between Version, the separator, and the SemVer.
const hasVersionRatification =
  /\bVersion\b[*\s:]*\d+\.\d+\.\d+[\s\S]*?Ratified\b/i.test(constitutionText) ||
  /\*\*Version\*\*[*\s:]*\d+\.\d+\.\d+[\s\S]*?\*\*Ratified\*\*/i.test(
    constitutionText,
  );
check(
  'AC-002',
  hasConstitutionAuthority && hasVersionRatification,
  `constitution authority=${hasConstitutionAuthority}, version+ratification block=${hasVersionRatification}`,
);

// AC-003: 01-planning.agent.md Plan Block Schema documents constitution_check.
const planningAgentText = readFileSync(planningAgentPath, 'utf-8');
const planningSchemaSection = planningAgentText.indexOf(
  '## Plan Block Schemas',
);
const planningHasConstitutionCheck =
  planningSchemaSection !== -1 &&
  planningAgentText.indexOf('constitution_check', planningSchemaSection) !== -1;
check(
  'AC-003',
  planningHasConstitutionCheck,
  `constitution_check documented in Plan Block Schemas=${planningHasConstitutionCheck}`,
);

// AC-004: phase-handoff-workflow/SKILL.md Step Packet Shape documents constitution_check.
const phaseHandoffText = readFileSync(phaseHandoffPath, 'utf-8');
const stepPacketShapeSection = phaseHandoffText.indexOf('## Step Packet Shape');
const handoffHasConstitutionCheck =
  stepPacketShapeSection !== -1 &&
  phaseHandoffText.indexOf('constitution_check', stepPacketShapeSection) !== -1;
check(
  'AC-004',
  handoffHasConstitutionCheck,
  `constitution_check documented in Step Packet Shape=${handoffHasConstitutionCheck}`,
);

// AC-005: capturing-learning-event/SKILL.md accepts constitution-update eventType.
const learningEventText = readFileSync(learningEventPath, 'utf-8');
const learningHasConstitutionUpdate = /constitution-update/.test(
  learningEventText,
);
check(
  'AC-005',
  learningHasConstitutionUpdate,
  `constitution-update accepted in capturing-learning-event=${learningHasConstitutionUpdate}`,
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
