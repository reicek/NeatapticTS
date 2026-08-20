/**
 * @module validate-plan-phase-packets.edge-mock.test
 * @description Covers defensive-code branches that are only reachable when
 *   parsePlanYamlBlock returns unusual values (null array entries, non-object
 *   metadata). Mocks parsePlanYamlBlock to inject these edge cases.
 *   Separate file so the mock does not affect the main coverage tests.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

const utils = await import('./customization-utils.mjs');

jest.unstable_mockModule('./customization-utils.mjs', () => ({
  ...utils,
  parsePlanYamlBlock: (yamlText) => {
    if (
      typeof yamlText === 'string' &&
      yamlText.includes('NULL_SLICE_MARKER')
    ) {
      return {
        phase: 'A',
        step: 1,
        title: 'Step One',
        status: '[WIP]',
        goal: 'implementing',
        mode: 'fresh-session',
        source_of_truth: 'plans/test.plans.md',
        copy_paste: true,
        next_step: 2,
        skills: ['implementation-standards'],
        validation: ['scripts/test.test.ts'],
        acceptance_criteria: ['AC-001: Criterion'],
        expansion: 'slices',
        tdd_sequence: 'red-green',
        slices: [null],
      };
    }
    if (
      typeof yamlText === 'string' &&
      yamlText.includes('NON_OBJECT_MARKER')
    ) {
      return 'not-an-object';
    }
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

describe('validate-plan-phase-packets edge-mock coverage', () => {
  it('covers slice ?? {} when parsePlanYamlBlock returns null slice entry', async () => {
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
      'NULL_SLICE_MARKER: yes',
      '```',
      '',
      '## Validation gates',
    ].join('\n');
    const report = await validatePlanText(planText, PLAN_PATH);
    // null slice entry → slice ?? {} produces {} → all required keys missing
    const missingKeyErrors = report.issues.filter((i) =>
      i.message.includes('Missing slice key'),
    );
    assert.ok(missingKeyErrors.length > 0);
  });

  it('covers isLegacyBlock non-object metadata (line 1032)', async () => {
    const planText = [
      '## Implementation phases',
      '',
      '### Phase A — Test Phase [WIP]',
      '',
      makePhaseSections(),
      '',
      '```yaml',
      'NON_OBJECT_MARKER: yes',
      '```',
      '',
      '## Validation gates',
    ].join('\n');
    const report = await validatePlanText(planText, PLAN_PATH);
    // Non-object metadata → isLegacyBlock returns false at line 1032.
    // Code continues to isStepPacket check, which accesses metadata.step
    // on a string — this is undefined, so isStepPacket is false.
    // Then validatePhasePacket runs, which will push errors for missing keys.
    // The test just needs to not crash and produce some issues.
    assert.ok(report.issues.length > 0);
  });
});
