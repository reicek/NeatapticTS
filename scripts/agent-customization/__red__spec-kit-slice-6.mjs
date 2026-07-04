#!/usr/bin/env node
/**
 * RED-phase check for Spec-Kit Assimilation Slice 6.
 *
 * Asserts AC-023..AC-025 for the research artifact requirement.
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

const researchAgentPath = '.github/agents/02-researching.agent.md';
const researchReadmePath = 'docs/research/README.md';

// AC-023: 02-researching.agent.md requires writing docs/research/<feature>.md.
const researchAgentText = existsSync(researchAgentPath)
  ? readFileSync(researchAgentPath, 'utf-8')
  : '';
const requiresDocsResearch = /docs\/research\//i.test(researchAgentText);
check('AC-023', requiresDocsResearch, researchAgentPath);

// AC-024: 02-researching.agent.md requires the produced step packet to link to the research artifact.
const requiresStepPacketLink =
  /research_artifact/i.test(researchAgentText) ||
  /step packet.*link.*research/i.test(researchAgentText) ||
  /research artifact.*step packet/i.test(researchAgentText) ||
  /link.*research artifact/i.test(researchAgentText);
check(
  'AC-024',
  requiresStepPacketLink,
  'step packet link to research artifact',
);

// AC-025: docs/research/README.md exists with naming convention and required sections.
const readmeExists = existsSync(researchReadmePath);
let readmeText = '';
if (readmeExists) {
  readmeText = readFileSync(researchReadmePath, 'utf-8');
}
const hasNamingConvention =
  /<feature-slug>\.md/i.test(readmeText) || /<feature>\.md/i.test(readmeText);
const hasQuestion = /#{1,3}\s*Question/i.test(readmeText);
const hasEvidence = /#{1,3}\s*Evidence/i.test(readmeText);
const hasDecision = /#{1,3}\s*Decision/i.test(readmeText);
const hasRisks = /#{1,3}\s*Risks/i.test(readmeText);
const hasAllSections = hasQuestion && hasEvidence && hasDecision && hasRisks;

check(
  'AC-025',
  readmeExists && hasNamingConvention && hasAllSections,
  readmeExists
    ? `naming convention=${hasNamingConvention}, Question=${hasQuestion}, Evidence=${hasEvidence}, Decision=${hasDecision}, Risks=${hasRisks}`
    : `${researchReadmePath} does not exist`,
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
