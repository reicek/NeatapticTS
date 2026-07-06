#!/usr/bin/env node
/**
 * RED-phase check for Spec-Kit Assimilation Slice 7.
 *
 * Asserts AC-026..AC-029 for the bug-triage extension layout and skill set.
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

const bugsReadmePath = '.github/bugs/README.md';
const bugTriageSkillPath = '.github/skills/bug-triage/SKILL.md';

// AC-026: .github/bugs/README.md exists and documents the assess → fix → test workflow.
const bugsReadmeExists = existsSync(bugsReadmePath);
let bugsReadmeText = '';
if (bugsReadmeExists) {
  bugsReadmeText = readFileSync(bugsReadmePath, 'utf-8');
}
const hasAssessPhase = /bug-assess|assess\s*→|assess phase/i.test(
  bugsReadmeText,
);
const hasFixPhase = /bug-fix|fix\s*→|fix phase/i.test(bugsReadmeText);
const hasTestPhase = /bug-test|test phase/i.test(bugsReadmeText);
const hasWorkflowArrow = /assess\s*[-→]\s*fix\s*[-→]\s*test/i.test(
  bugsReadmeText,
);
check(
  'AC-026',
  bugsReadmeExists &&
    hasAssessPhase &&
    hasFixPhase &&
    hasTestPhase &&
    hasWorkflowArrow,
  bugsReadmeExists
    ? `expected assess→fix→test workflow (assess=${hasAssessPhase}, fix=${hasFixPhase}, test=${hasTestPhase}, arrow=${hasWorkflowArrow})`
    : `${bugsReadmePath} does not exist`,
);

// AC-027: .github/bugs/README.md specifies the artifact layout and URL-trust/evidence rules.
const hasSlugLayout = /\.github\/bugs\/<slug>\//i.test(bugsReadmeText);
const hasAssessArtifact = /assess\.md/i.test(bugsReadmeText);
const hasFixArtifact = /fix\.md/i.test(bugsReadmeText);
const hasTestArtifact = /test\.md/i.test(bugsReadmeText);
const hasEvidenceDir = /evidence\//i.test(bugsReadmeText);
const hasUrlTrust =
  /URL[- ]?trust|trusted URL|untrusted URL|safe list|allowlisted/i.test(
    bugsReadmeText,
  );
const hasEvidenceRule = /evidence/i.test(bugsReadmeText);
check(
  'AC-027',
  hasSlugLayout &&
    hasAssessArtifact &&
    hasFixArtifact &&
    hasTestArtifact &&
    hasEvidenceDir &&
    hasUrlTrust &&
    hasEvidenceRule,
  `slug layout=${hasSlugLayout}, assess.md=${hasAssessArtifact}, fix.md=${hasFixArtifact}, test.md=${hasTestArtifact}, evidence/=${hasEvidenceDir}, URL-trust=${hasUrlTrust}, evidence rules=${hasEvidenceRule}`,
);

// AC-028: .github/skills/bug-triage/SKILL.md exists with valid frontmatter and defines phases.
const bugTriageSkillExists = existsSync(bugTriageSkillPath);
let bugTriageSkillText = '';
if (bugTriageSkillExists) {
  bugTriageSkillText = readFileSync(bugTriageSkillPath, 'utf-8');
}
const hasFrontmatter =
  bugTriageSkillText.startsWith('---') &&
  /^name:\s*\S+/im.test(bugTriageSkillText) &&
  /^description:/im.test(bugTriageSkillText);
const skillHasAssess = /bug-assess|assess phase/i.test(bugTriageSkillText);
const skillHasFix = /bug-fix|fix phase/i.test(bugTriageSkillText);
const skillHasTest = /bug-test|test phase/i.test(bugTriageSkillText);
check(
  'AC-028',
  bugTriageSkillExists &&
    hasFrontmatter &&
    skillHasAssess &&
    skillHasFix &&
    skillHasTest,
  bugTriageSkillExists
    ? `valid frontmatter=${hasFrontmatter}, bug-assess=${skillHasAssess}, bug-fix=${skillHasFix}, bug-test=${skillHasTest}`
    : `${bugTriageSkillPath} does not exist`,
);

// AC-029: bug-triage/SKILL.md references the four gap types and requires a regression test before closing.
const hasMissing = /\bmissing\b/i.test(bugTriageSkillText);
const hasPartial = /\bpartial\b/i.test(bugTriageSkillText);
const hasContradicts = /\bcontradicts\b/i.test(bugTriageSkillText);
const hasUnrequested = /\bunrequested\b/i.test(bugTriageSkillText);
const hasRegressionTest = /regression test|regression-test/i.test(
  bugTriageSkillText,
);
const hasBeforeClose = /before closing|close the bug|bug.*closed/i.test(
  bugTriageSkillText,
);
check(
  'AC-029',
  hasMissing &&
    hasPartial &&
    hasContradicts &&
    hasUnrequested &&
    hasRegressionTest &&
    hasBeforeClose,
  `missing=${hasMissing}, partial=${hasPartial}, contradicts=${hasContradicts}, unrequested=${hasUnrequested}, regression test=${hasRegressionTest}, before closing=${hasBeforeClose}`,
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
