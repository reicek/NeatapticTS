#!/usr/bin/env node
/**
 * RED-phase check for Spec-Kit Assimilation Slice 4.
 *
 * Asserts AC-014..AC-017 for the user-story-driven tasks template.
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

const tasksTemplatePath = '.github/templates/tasks-template.md';

const tasksTemplateExists = existsSync(tasksTemplatePath);
let tasksTemplateText = '';
if (tasksTemplateExists) {
  tasksTemplateText = readFileSync(tasksTemplatePath, 'utf-8');
}

// AC-014: tasks-template.md exists and includes the four phase headings.
const hasSetupHeading = /^##\s+Setup\s*$/m.test(tasksTemplateText);
const hasFoundationalHeading = /^##\s+Foundational\s*$/m.test(
  tasksTemplateText,
);
const hasStoryHeading = /^##\s+Story\s*$/m.test(tasksTemplateText);
const hasPolishHeading = /^##\s+Polish\s*$/m.test(tasksTemplateText);
const hasAllPhaseHeadings =
  hasSetupHeading &&
  hasFoundationalHeading &&
  hasStoryHeading &&
  hasPolishHeading;

check(
  'AC-014',
  tasksTemplateExists && hasAllPhaseHeadings,
  tasksTemplateExists
    ? `expected phase headings Setup=${hasSetupHeading}, Foundational=${hasFoundationalHeading}, Story=${hasStoryHeading}, Polish=${hasPolishHeading}`
    : `${tasksTemplatePath} does not exist`,
);

// AC-015: at least one parallelizable task marked with [P] in the template.
const hasParallelMarker = /\[P\]/.test(tasksTemplateText);

check(
  'AC-015',
  hasParallelMarker,
  `expected at least one [P] parallel marker in ${tasksTemplatePath}`,
);

// AC-016: traceability table placeholder mapping T### -> AC-### and Constitution Check section.
const hasTraceabilitySection = /^##\s+Traceability\s*$/m.test(
  tasksTemplateText,
);
const hasTraceabilityTable =
  /Task ID.*Phase.*Description.*Acceptance Criterion.*Constitution Principle.*Parallel/.test(
    tasksTemplateText,
  );
const hasTaskToAcceptanceMapping = /T\d+.*AC-\d+/.test(tasksTemplateText);
const hasConstitutionCheckSection = /^##\s+Constitution Check\s*$/m.test(
  tasksTemplateText,
);
const referencesConstitution = /plans\/constitution\.md/.test(
  tasksTemplateText,
);

check(
  'AC-016',
  hasTraceabilitySection &&
    hasTraceabilityTable &&
    hasTaskToAcceptanceMapping &&
    hasConstitutionCheckSection &&
    referencesConstitution,
  `traceability section=${hasTraceabilitySection}, table columns=${hasTraceabilityTable}, T### -> AC-### mapping=${hasTaskToAcceptanceMapping}, constitution check section=${hasConstitutionCheckSection}, references plans/constitution.md=${referencesConstitution}`,
);

// AC-017: uses Spec-Kit-style task IDs (T001, T002, etc.) and references AC-### acceptance criteria.
const hasTaskIds = /T\d{3,}/.test(tasksTemplateText);
const hasAcceptanceCriteria = /AC-\d{3,}/.test(tasksTemplateText);

check(
  'AC-017',
  hasTaskIds && hasAcceptanceCriteria,
  `expected T### task IDs=${hasTaskIds} and AC-### references=${hasAcceptanceCriteria}`,
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
