/**
 * @module validate-plan-phase-packets.parse-error.test
 * @description Covers the __parseError branches in parseMetadataBlock by
 *   mocking parsePlanYamlBlock to throw on YAML containing a THROW_MARKER.
 *   Separate file so the mock does not affect the main coverage tests.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

const utils = await import('./customization-utils.mjs');

jest.unstable_mockModule('./customization-utils.mjs', () => ({
  ...utils,
  parsePlanYamlBlock: (yamlText) => {
    if (typeof yamlText === 'string' && yamlText.includes('THROW_MARKER'))
      throw new Error('mocked parse failure');
    return utils.parsePlanYamlBlock(yamlText);
  },
}));

const { validatePlanText } = await import('./validate-plan-phase-packets.mjs');

const PLAN_PATH = 'plans/test.plans.md';

function makePhaseSections() {
  return [
    '**Phase objective:** Test objective.',
    '**Stop conditions:** Conditions met.',
    '**Required validation:** Tests pass.',
  ].join('\n\n');
}

function makeStepSections() {
  return [
    '**User instruction:** Do the thing.',
    '**Step objective:** Implement feature.',
    '**Stop conditions:** Done.',
    '**Required validation:** Tests pass.',
  ].join('\n\n');
}

function makeValidPhaseYaml() {
  return [
    '```yaml',
    'phase: A',
    'title: Test Phase',
    'status: [WIP]',
    'goal: planning',
    'expansion: steps',
    'auto_expand: false',
    'mode: fresh-session',
    `source_of_truth: ${PLAN_PATH}`,
    'copy_paste: true',
    'next_phase: B',
    'skills:',
    '  - plan-alignment',
    'validation:',
    '  - scripts/test.test.ts',
    'acceptance_criteria:',
    '  - AC-001: Criterion',
    'placeholder_steps:',
    '  - Step 01: Placeholder',
    '```',
  ].join('\n');
}

function makeValidStepYaml() {
  return [
    '```yaml',
    'phase: A',
    'step: 1',
    'title: Step One',
    'status: [WIP]',
    'goal: implementing',
    'mode: fresh-session',
    `source_of_truth: ${PLAN_PATH}`,
    'copy_paste: true',
    'next_step: 2',
    'skills:',
    '  - implementation-standards',
    'validation:',
    '  - scripts/test.test.ts',
    'acceptance_criteria:',
    '  - AC-001: Criterion',
    'expansion: none',
    '```',
  ].join('\n');
}

describe('validate-plan-phase-packets parse-error coverage', () => {
  it('reports Could not parse phase YAML when parsePlanYamlBlock throws', async () => {
    const planText = [
      '## Implementation phases',
      '',
      '### Phase A — Test Phase [WIP]',
      '',
      makePhaseSections(),
      '',
      '```yaml',
      'THROW_MARKER: yes',
      'phase: A',
      '```',
      '',
      '## Validation gates',
    ].join('\n');
    const report = await validatePlanText(planText, PLAN_PATH);
    const parseErrors = report.issues.filter((i) =>
      i.message.includes('Could not parse phase YAML'),
    );
    assert.ok(parseErrors.length > 0);
    assert.ok(parseErrors[0].message.includes('mocked parse failure'));
  });

  it('reports Could not parse step YAML when parsePlanYamlBlock throws', async () => {
    const planText = [
      '## Implementation phases',
      '',
      '### Phase A — Test Phase [WIP]',
      '',
      makePhaseSections(),
      '',
      makeValidPhaseYaml(),
      '',
      '#### Step 01 — Step One [WIP]',
      '',
      makeStepSections(),
      '',
      '```yaml',
      'THROW_MARKER: yes',
      'step: 1',
      '```',
      '',
      '## Validation gates',
    ].join('\n');
    const report = await validatePlanText(planText, PLAN_PATH);
    const parseErrors = report.issues.filter((i) =>
      i.message.includes('Could not parse step YAML'),
    );
    assert.ok(parseErrors.length > 0);
    assert.ok(parseErrors[0].message.includes('mocked parse failure'));
  });
});