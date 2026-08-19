/**
 * @module validate-plan-phase-packets.test
 * @description Coverage tests for validate-plan-phase-packets.mjs.
 *   Exercises validatePlanText with crafted plan text to cover all validation paths.
 */
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  validatePlanText,
  looksLikeFilePath,
  validateValidationList,
} from './validate-plan-phase-packets.mjs';
import { issue } from './customization-utils.mjs';

const PLAN_PATH = 'plans/test.plans.md';

function makeValidPhaseYaml(overrides = {}) {
  const defaults = {
    phase: 'A',
    title: 'Test Phase',
    status: '[WIP]',
    goal: 'planning',
    expansion: 'steps',
    auto_expand: false,
    mode: 'fresh-session',
    source_of_truth: PLAN_PATH,
    copy_paste: true,
    next_phase: 'B',
    skills: ['plan-alignment'],
    validation: ['scripts/test.test.ts'],
    acceptance_criteria: ['AC-001: Criterion'],
    placeholder_steps: ['Step 01: Placeholder'],
  };
  return { ...defaults, ...overrides };
}

function yamlBlock(obj) {
  const lines = ['```yaml'];
  serializeYamlObject(obj, 0, lines);
  lines.push('```');
  return lines.join('\n');
}

function serializeYamlValue(key, value, indent, lines, prefix) {
  const pad = ' '.repeat(indent);
  const lead = prefix ?? `${pad}${key}:`;
  if (Array.isArray(value)) {
    if (value.length === 0) {
      lines.push(`${lead} []`);
    } else if (typeof value[0] === 'object' && value[0] !== null) {
      lines.push(lead);
      for (const item of value) {
        serializeYamlListItem(item, indent + 2, lines);
      }
    } else {
      lines.push(lead);
      for (const item of value) {
        lines.push(`${' '.repeat(indent + 2)}- ${item}`);
      }
    }
  } else if (value === null) {
    lines.push(`${lead} null`);
  } else if (typeof value === 'boolean') {
    lines.push(`${lead} ${value}`);
  } else {
    lines.push(`${lead} ${value}`);
  }
}

function serializeYamlObject(obj, indent, lines) {
  for (const [key, value] of Object.entries(obj)) {
    serializeYamlValue(key, value, indent, lines);
  }
}

function serializeYamlListItem(obj, indent, lines) {
  const pad = ' '.repeat(indent);
  if (obj === null || typeof obj !== 'object') {
    lines.push(`${pad}- ${obj}`);
    return;
  }
  const entries = Object.entries(obj);
  if (entries.length === 0) {
    lines.push(`${pad}- {}`);
    return;
  }
  const [firstKey, firstValue] = entries[0];
  serializeYamlValue(
    firstKey,
    firstValue,
    indent,
    lines,
    `${pad}- ${firstKey}:`,
  );
  for (let i = 1; i < entries.length; i++) {
    const [key, value] = entries[i];
    serializeYamlValue(key, value, indent + 2, lines);
  }
}

function makeValidStepYaml(overrides = {}) {
  const defaults = {
    phase: 'A',
    step: 1,
    title: 'Step One',
    status: '[WIP]',
    goal: 'implementing',
    mode: 'fresh-session',
    source_of_truth: PLAN_PATH,
    copy_paste: true,
    next_step: 2,
    skills: ['implementation-standards'],
    validation: ['scripts/test.test.ts'],
    acceptance_criteria: ['AC-001: Criterion'],
    expansion: 'none',
  };
  return { ...defaults, ...overrides };
}

function makePhaseSections(status) {
  if (status === 'DONE') return '';
  return [
    '**Phase objective:** Test objective.',
    '**Stop conditions:** Conditions met.',
    '**Required validation:** Tests pass.',
  ].join('\n\n');
}

function makeStepSections(status) {
  if (status === 'DONE') return '';
  return [
    '**User instruction:** Do the thing.',
    '**Step objective:** Implement feature.',
    '**Stop conditions:** Done.',
    '**Required validation:** Tests pass.',
  ].join('\n\n');
}

function buildPlan(phases) {
  const phaseTexts = phases.map((p) => {
    const lines = [
      `### Phase ${p.headingPhase} — ${p.headingTitle} [${p.headingStatus}]`,
      '',
    ];
    if (p.phaseSections) lines.push(p.phaseSections, '');
    if (p.yaml) lines.push(yamlBlock(p.yaml), '');
    if (p.steps) {
      for (const step of p.steps) {
        lines.push(
          `#### Step ${String(step.headingStep).padStart(2, '0')} — ${step.headingTitle} [${step.headingStatus}]`,
          '',
        );
        if (step.stepSections) lines.push(step.stepSections, '');
        if (step.yaml) lines.push(yamlBlock(step.yaml), '');
      }
    }
    return lines.join('\n');
  });

  return [
    '## Implementation phases',
    '',
    ...phaseTexts,
    '',
    '## Validation gates',
    '',
    '- All tests pass.',
  ].join('\n');
}

function validPhase(status = 'WIP') {
  return {
    headingPhase: 'A',
    headingTitle: 'Test Phase',
    headingStatus: status,
    phaseSections: makePhaseSections(status),
    yaml: makeValidPhaseYaml({ status: `[${status}]` }),
    steps:
      status === 'PLANNED'
        ? []
        : [
            {
              headingStep: 1,
              headingTitle: 'Step One',
              headingStatus: status === 'DONE' ? 'DONE' : 'WIP',
              stepSections: makeStepSections(
                status === 'DONE' ? 'DONE' : 'WIP',
              ),
              yaml: makeValidStepYaml({
                status: `[${status === 'DONE' ? 'DONE' : 'WIP'}]`,
              }),
            },
          ],
  };
}

function findIssues(report, substring) {
  return report.issues.filter((i) => i.message.includes(substring));
}

describe('validate-plan-phase-packets coverage', () => {
  describe('looksLikeFilePath', () => {
    it('returns false for non-strings', () => {
      assert.equal(looksLikeFilePath(123), false);
      assert.equal(looksLikeFilePath(null), false);
      assert.equal(looksLikeFilePath(undefined), false);
    });

    it('returns false for multiline strings', () => {
      assert.equal(looksLikeFilePath('foo\nbar'), false);
    });

    it('returns false for CLI flags', () => {
      assert.equal(looksLikeFilePath('--testPathPattern'), false);
      assert.equal(looksLikeFilePath('-flag'), false);
    });

    it('returns true for paths with separators', () => {
      assert.equal(looksLikeFilePath('src/foo.ts'), true);
      assert.equal(looksLikeFilePath('scripts\\bar.mjs'), true);
    });

    it('returns true for paths with extensions', () => {
      assert.equal(looksLikeFilePath('foo.ts'), true);
      assert.equal(looksLikeFilePath('bar.test.mjs'), true);
    });

    it('returns true for dot and dot-dot', () => {
      assert.equal(looksLikeFilePath('.'), true);
      assert.equal(looksLikeFilePath('..'), true);
    });

    it('returns false for plain words without extensions', () => {
      assert.equal(looksLikeFilePath('justaword'), false);
    });
  });

  describe('validateValidationList', () => {
    it('reports non-string entries as errors', () => {
      const issuesList = [];
      validateValidationList([123, 'valid.ts'], 'ctx', issuesList);
      assert.ok(issuesList.some((i) => i.message.includes('must be a string')));
    });

    it('reports stale --testPathPattern flags as errors', () => {
      const issuesList = [];
      validateValidationList(
        ['--testPathPattern=foo', 'valid.ts'],
        'ctx',
        issuesList,
      );
      assert.ok(
        issuesList.some((i) => i.message.includes('stale --testPathPattern')),
      );
    });

    it('warns for entries that do not look like file paths', () => {
      const issuesList = [];
      validateValidationList(['justaword', 'valid.ts'], 'ctx', issuesList);
      assert.ok(
        issuesList.some(
          (i) =>
            i.severity === 'warning' &&
            i.message.includes('does not look like a file path'),
        ),
      );
    });

    it('passes for valid file paths', () => {
      const issuesList = [];
      validateValidationList(
        ['src/foo.test.ts', 'scripts/bar.mjs'],
        'ctx',
        issuesList,
      );
      assert.equal(issuesList.length, 0);
    });

    it('uses default issues list when not provided', () => {
      // When called without issuesList, it uses module-level issues.
      // This exercises the default parameter.
      const result = validateValidationList(['valid.ts'], 'ctx');
      assert.ok(Array.isArray(result));
    });
  });

  describe('validatePlanText', () => {
    it('reports no phases for empty plan', async () => {
      const report = await validatePlanText(
        '## Implementation phases\n\n## Validation gates\n',
        PLAN_PATH,
      );
      assert.ok(
        findIssues(report, 'No implementation phase packets').length > 0,
      );
    });

    it('validates a correct WIP phase with one WIP step', async () => {
      const plan = buildPlan([validPhase('WIP')]);
      const report = await validatePlanText(plan, PLAN_PATH);
      const errors = report.issues.filter((i) => i.severity === 'error');
      assert.equal(
        errors.length,
        0,
        `Unexpected errors: ${JSON.stringify(errors, null, 2)}`,
      );
    });

    it('validates a DONE phase without requiring sections', async () => {
      const plan = buildPlan([validPhase('DONE')]);
      const report = await validatePlanText(plan, PLAN_PATH);
      // DONE phase with DONE step should not require sections
      const sectionIssues = findIssues(report, 'Missing required section');
      assert.equal(sectionIssues.length, 0);
    });

    it('warns when non-archived plan has 0 WIP phases', async () => {
      const plan = buildPlan([validPhase('DONE')]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'Expected exactly one [WIP] phase').length > 0,
      );
    });

    it('warns when non-archived plan has 2 WIP phases', async () => {
      const phaseA = validPhase('WIP');
      const phaseB = {
        ...validPhase('WIP'),
        headingPhase: 'B',
        yaml: makeValidPhaseYaml({
          phase: 'B',
          status: '[WIP]',
          next_phase: 'C',
        }),
      };
      const plan = buildPlan([phaseA, phaseB]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'Expected exactly one [WIP] phase').length > 0,
      );
    });

    it('errors when archived DONE plan has WIP phases', async () => {
      const plan = '**Status:** [DONE]\n\n' + buildPlan([validPhase('WIP')]);
      const report = await validatePlanText(
        plan,
        'plans/completed/test.plans.md',
      );
      const wipIssues = findIssues(
        report,
        'Archived [DONE] plans must not contain [WIP]',
      );
      assert.ok(wipIssues.length > 0);
    });

    it('does not error for archived DONE plan with no phases', async () => {
      const report = await validatePlanText(
        '**Status:** [DONE]\n\n## Implementation phases\n\n## Validation gates\n',
        'plans/completed/test.plans.md',
      );
      // Archived closed plan with no phases should not produce "No phases" error
      assert.equal(
        findIssues(report, 'No implementation phase packets').length,
        0,
      );
    });

    it('reports phase label sequence mismatch', async () => {
      const phaseA = validPhase('WIP');
      const phaseC = {
        ...validPhase('WIP'),
        headingPhase: 'C',
        yaml: makeValidPhaseYaml({ phase: 'C', status: '[WIP]' }),
        steps: [
          {
            ...validPhase('WIP').steps[0],
            yaml: makeValidStepYaml({ phase: 'C', status: '[WIP]' }),
          },
        ],
      };
      const plan = buildPlan([phaseA, phaseC]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'Expected phase B, found phase C').length > 0,
      );
    });

    it('reports missing YAML metadata in non-DONE phase', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'WIP',
        phaseSections: makePhaseSections('WIP'),
        yaml: null,
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'must contain a YAML metadata block').length > 0,
      );
    });

    it('does not require YAML in DONE phase', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'DONE',
        phaseSections: null,
        yaml: null,
        steps: [],
      };
      // Need another WIP phase to avoid the wipCount warning
      const wipPhase = validPhase('WIP');
      const plan = buildPlan([phase, wipPhase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      const metadataIssues = findIssues(
        report,
        'must contain a YAML metadata block',
      );
      // The DONE phase should not produce this error
      const donePhaseIssues = metadataIssues.filter((i) =>
        i.path.includes('phase-A'),
      );
      assert.equal(donePhaseIssues.length, 0);
    });

    it('reports legacy block in phase metadata', async () => {
      // Legacy: has agent or agent_file but no expansion
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'WIP',
        phaseSections: makePhaseSections('WIP'),
        yaml: null,
        rawYaml:
          '```yaml\nphase: A\ntitle: Test Phase\nstatus: [WIP]\nagent: some-agent\ngoal: planning\n```',
        steps: [],
      };
      const planText = [
        '## Implementation phases',
        '',
        `### Phase A — Test Phase [WIP]`,
        '',
        makePhaseSections('WIP'),
        '',
        '```yaml',
        'phase: A',
        'title: Test Phase',
        'status: [WIP]',
        'agent: some-agent',
        'goal: planning',
        '```',
        '',
        '## Validation gates',
      ].join('\n');
      const report = await validatePlanText(planText, PLAN_PATH);
      assert.ok(findIssues(report, 'Legacy plan block detected').length > 0);
    });

    it('detects step-packet at phase level (metadata.step !== undefined)', async () => {
      // Phase with metadata.step is treated as step packet
      const planText = [
        '## Implementation phases',
        '',
        `### Phase A — Test Phase [WIP]`,
        '',
        makePhaseSections('WIP'),
        '',
        yamlBlock(makeValidStepYaml({ status: '[WIP]' })),
        '',
        '## Validation gates',
      ].join('\n');
      const report = await validatePlanText(planText, PLAN_PATH);
      // Should validate as step packet, finding issues for step-specific keys
      const report2 = report;
      // It should produce a 'step' schema in phases summary
      assert.equal(report2.phases[0].schema, 'step');
    });

    it('reports missing phase metadata keys', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'WIP',
        phaseSections: makePhaseSections('WIP'),
        yaml: {
          phase: 'A',
          title: 'Test Phase',
          status: '[WIP]',
          expansion: 'steps',
        },
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'Missing phase metadata key').length > 0);
    });

    it('reports title mismatch', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: makeValidPhaseYaml({ title: 'Wrong Title', status: '[WIP]' }),
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'does not match heading title').length > 0);
    });

    it('reports phase mismatch in metadata', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: makeValidPhaseYaml({ phase: 'B', status: '[WIP]' }),
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'does not match heading').length > 0);
    });

    it('reports status mismatch in phase metadata', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: makeValidPhaseYaml({ status: '[DONE]' }),
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'does not match heading').length > 0);
    });

    it('reports invalid phase goal', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: makeValidPhaseYaml({ goal: 'implementing', status: '[WIP]' }),
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, "Phase-level goal must be 'planning'").length > 0,
      );
    });

    it('reports invalid expansion for phase', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: makeValidPhaseYaml({ expansion: 'none', status: '[WIP]' }),
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, "Phase-level expansion must be 'steps'").length > 0,
      );
    });

    it('reports auto_expand not false for phase', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: makeValidPhaseYaml({ auto_expand: true, status: '[WIP]' }),
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'auto_expand must be false').length > 0);
    });

    it('reports invalid mode for phase', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: makeValidPhaseYaml({ mode: 'bad-mode', status: '[WIP]' }),
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, "mode must be 'fresh-session' or 'perpetual'")
          .length > 0,
      );
    });

    it('reports source_of_truth mismatch for phase', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: makeValidPhaseYaml({
          source_of_truth: 'plans/other.plans.md',
          status: '[WIP]',
        }),
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'source_of_truth must be').length > 0);
    });

    it('reports copy_paste not true for phase', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: makeValidPhaseYaml({ copy_paste: false, status: '[WIP]' }),
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'copy_paste must be true').length > 0);
    });

    it('reports empty skills list for phase', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: makeValidPhaseYaml({ skills: [], status: '[WIP]' }),
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'skills must be a non-empty list').length > 0,
      );
    });

    it('reports non-array skills for phase', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: {
          ...makeValidPhaseYaml({ status: '[WIP]' }),
          skills: 'not-array',
        },
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'skills must be a non-empty list').length > 0,
      );
    });

    it('reports empty validation list for phase', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: makeValidPhaseYaml({ validation: [], status: '[WIP]' }),
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'validation must be a non-empty list').length > 0,
      );
    });

    it('reports non-array validation for phase', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: { ...makeValidPhaseYaml({ status: '[WIP]' }), validation: 'bad' },
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'validation must be a non-empty list').length > 0,
      );
    });

    it('reports empty acceptance_criteria for phase', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: makeValidPhaseYaml({ acceptance_criteria: [], status: '[WIP]' }),
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'acceptance_criteria must be a non-empty list')
          .length > 0,
      );
    });

    it('reports empty placeholder_steps for phase', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: makeValidPhaseYaml({ placeholder_steps: [], status: '[WIP]' }),
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'placeholder_steps must be a non-empty list')
          .length > 0,
      );
    });

    it('reports missing required sections in non-DONE phase', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'WIP',
        phaseSections: null,
        yaml: makeValidPhaseYaml({ status: '[WIP]' }),
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'Missing required section').length > 0);
    });

    it('reports forbidden Copy-paste prompt section in phase', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'WIP',
        phaseSections: makePhaseSections('WIP'),
        extraBody: '**Copy-paste prompt:** Do stuff.',
        yaml: makeValidPhaseYaml({ status: '[WIP]' }),
        steps: [],
      };
      const planText = [
        '## Implementation phases',
        '',
        '### Phase A — Test Phase [WIP]',
        '',
        makePhaseSections('WIP'),
        '',
        '**Copy-paste prompt:** Do stuff.',
        '',
        yamlBlock(makeValidPhaseYaml({ status: '[WIP]' })),
        '',
        '## Validation gates',
      ].join('\n');
      const report = await validatePlanText(planText, PLAN_PATH);
      assert.ok(
        findIssues(report, 'must not contain a separate Copy-paste prompt')
          .length > 0,
      );
    });

    it('reports step not starting at 01 in active phase', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'WIP',
        phaseSections: makePhaseSections('WIP'),
        yaml: makeValidPhaseYaml({ status: '[WIP]', step: 1 }),
        steps: [
          {
            headingStep: 2,
            headingTitle: 'Step Two',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ step: 2, status: '[WIP]', next_step: 3 }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'must start with Step 01').length > 0);
    });

    it('reports wrong WIP step count in WIP phase', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'WIP',
        phaseSections: makePhaseSections('WIP'),
        yaml: makeValidPhaseYaml({ status: '[WIP]', step: 1 }),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ status: '[WIP]' }),
          },
          {
            headingStep: 2,
            headingTitle: 'Step Two',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ step: 2, status: '[WIP]', next_step: 3 }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'Expected exactly one [WIP] step').length > 0,
      );
    });

    it('reports WIP steps in non-WIP phase', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'PLANNED',
        phaseSections: makePhaseSections('PLANNED'),
        yaml: makeValidPhaseYaml({ status: '[PLANNED]', step: 1 }),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ status: '[WIP]' }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'Only [WIP] phases may contain [WIP] steps').length >
          0,
      );
    });

    it('reports step number mismatch', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'WIP',
        phaseSections: makePhaseSections('WIP'),
        yaml: makeValidPhaseYaml({ status: '[WIP]', step: 1 }),
        steps: [
          {
            headingStep: 3,
            headingTitle: 'Step Three',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ step: 3, status: '[WIP]', next_step: 4 }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'Expected step 01, found step 03').length > 0,
      );
    });

    it('reports missing YAML in non-DONE step', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'WIP',
        phaseSections: makePhaseSections('WIP'),
        yaml: makeValidPhaseYaml({ status: '[WIP]' }),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: null,
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'must keep a YAML metadata block').length > 0,
      );
    });

    it('reports legacy block in step', async () => {
      const planText = [
        '## Implementation phases',
        '',
        '### Phase A — Test Phase [WIP]',
        '',
        makePhaseSections('WIP'),
        '',
        yamlBlock(makeValidPhaseYaml({ status: '[WIP]' })),
        '',
        '#### Step 01 — Step One [WIP]',
        '',
        makeStepSections('WIP'),
        '',
        '```yaml',
        'phase: A',
        'step: 1',
        'title: Step One',
        'status: [WIP]',
        'agent: some-agent',
        'goal: implementing',
        'mode: fresh-session',
        'source_of_truth: plans/test.plans.md',
        'copy_paste: true',
        'next_step: 2',
        'skills:',
        '  - skill',
        'validation:',
        '  - test.ts',
        'acceptance_criteria:',
        '  - criterion',
        '```',
        '',
        '## Validation gates',
      ].join('\n');
      const report = await validatePlanText(planText, PLAN_PATH);
      assert.ok(findIssues(report, 'Legacy plan block detected').length > 0);
    });

    it('reports missing step metadata keys', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'WIP',
        phaseSections: makePhaseSections('WIP'),
        yaml: makeValidPhaseYaml({ status: '[WIP]' }),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              phase: 'A',
              step: 1,
              title: 'Step One',
              status: '[WIP]',
              expansion: 'none',
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'Missing step metadata key').length > 0);
    });

    it('reports unknown metadata keys in step', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              unknown_key: 'value',
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'Unexpected metadata key: unknown_key').length > 0,
      );
    });

    it('reports step title mismatch', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ title: 'Wrong Title', status: '[WIP]' }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'does not match heading title').length > 0);
    });

    it('reports step phase mismatch', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ phase: 'B', status: '[WIP]' }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'does not match phase heading').length > 0);
    });

    it('reports step number mismatch in metadata', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ step: 2, status: '[WIP]', next_step: 3 }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'does not match step heading').length > 0);
    });

    it('reports step status mismatch', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ status: '[DONE]' }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'does not match step heading').length > 0);
    });

    it('reports invalid step goal', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ goal: 'bad-goal', status: '[WIP]' }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, "Invalid goal 'bad-goal'").length > 0);
    });

    it('reports invalid tdd_sequence', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              tdd_sequence: 'bad-sequence',
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, "Invalid tdd_sequence 'bad-sequence'").length > 0,
      );
    });

    it('reports invalid step mode', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ mode: 'bad-mode', status: '[WIP]' }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, "mode must be 'fresh-session' or 'perpetual'")
          .length > 0,
      );
    });

    it('reports step source_of_truth mismatch', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({
              source_of_truth: 'plans/other.plans.md',
              status: '[WIP]',
            }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'source_of_truth must be').length > 0);
    });

    it('reports step copy_paste not true', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ copy_paste: false, status: '[WIP]' }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'copy_paste must be true').length > 0);
    });

    it('reports empty step skills', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ skills: [], status: '[WIP]' }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'skills must be a non-empty list').length > 0,
      );
    });

    it('reports empty step validation', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ validation: [], status: '[WIP]' }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'validation must be a non-empty list').length > 0,
      );
    });

    it('reports empty step acceptance_criteria in non-DONE step', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({
              acceptance_criteria: [],
              status: '[WIP]',
            }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'acceptance_criteria must be a non-empty list')
          .length > 0,
      );
    });

    it('reports invalid expansion value in step', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              expansion: 'bad-expansion',
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, "Invalid expansion 'bad-expansion'").length > 0,
      );
    });

    it('reports slices expansion without auto_expand: true', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              expansion: 'slices',
              auto_expand: false,
              tdd_sequence: 'red-green',
              slices: [
                {
                  slice_id: 'S1',
                  title: 'Slice 1',
                  status: 'WIP',
                  goal: 'red-testing',
                  estimate_hours: 2,
                  files_to_change: ['src/foo.ts'],
                  acceptance_criteria: ['AC-1'],
                  parallelizable: false,
                  dependencies: [],
                },
                {
                  slice_id: 'S2',
                  title: 'Slice 2',
                  status: 'WIP',
                  goal: 'green-testing',
                  estimate_hours: 1,
                  files_to_change: ['src/bar.ts'],
                  acceptance_criteria: ['AC-2'],
                  parallelizable: false,
                  dependencies: ['S1'],
                },
              ],
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, "Expansion 'slices' requires auto_expand: true")
          .length > 0,
      );
    });

    it('reports slices expansion without tdd_sequence', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              expansion: 'slices',
              auto_expand: true,
              slices: [
                {
                  slice_id: 'S1',
                  title: 'Slice 1',
                  status: 'WIP',
                  goal: 'red-testing',
                  estimate_hours: 2,
                  files_to_change: ['src/foo.ts'],
                  acceptance_criteria: ['AC-1'],
                  parallelizable: false,
                  dependencies: [],
                },
                {
                  slice_id: 'S2',
                  title: 'Slice 2',
                  status: 'WIP',
                  goal: 'green-testing',
                  estimate_hours: 1,
                  files_to_change: ['src/bar.ts'],
                  acceptance_criteria: ['AC-2'],
                  parallelizable: false,
                  dependencies: ['S1'],
                },
              ],
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, "Expansion 'slices' requires a tdd_sequence field")
          .length > 0,
      );
    });

    it('reports slices expansion with empty slices list', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              expansion: 'slices',
              auto_expand: true,
              tdd_sequence: 'red-green',
              slices: [],
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(
          report,
          "Expansion 'slices' requires a non-empty slices list",
        ).length > 0,
      );
    });

    it('reports missing slice keys', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              expansion: 'slices',
              auto_expand: true,
              tdd_sequence: 'red-green',
              slices: [{}],
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'Missing slice key').length > 0);
    });

    it('reports invalid slice goal', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              expansion: 'slices',
              auto_expand: true,
              tdd_sequence: 'red-green',
              slices: [
                {
                  slice_id: 'S1',
                  title: 'Slice 1',
                  status: 'WIP',
                  goal: 'bad-goal',
                  estimate_hours: 2,
                  files_to_change: ['src/foo.ts'],
                  acceptance_criteria: ['AC-1'],
                  parallelizable: false,
                  dependencies: [],
                },
              ],
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, "Invalid slice goal 'bad-goal'").length > 0);
    });

    it('reports files_to_change not a list in slice', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              expansion: 'slices',
              auto_expand: true,
              tdd_sequence: 'red-green',
              slices: [
                {
                  slice_id: 'S1',
                  title: 'Slice 1',
                  status: 'WIP',
                  goal: 'red-testing',
                  estimate_hours: 2,
                  files_to_change: 'not-a-list',
                  acceptance_criteria: ['AC-1'],
                  parallelizable: false,
                  dependencies: [],
                },
              ],
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'Slice files_to_change must be a list').length > 0,
      );
    });

    it('reports acceptance_criteria not a list in slice', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              expansion: 'slices',
              auto_expand: true,
              tdd_sequence: 'red-green',
              slices: [
                {
                  slice_id: 'S1',
                  title: 'Slice 1',
                  status: 'WIP',
                  goal: 'red-testing',
                  estimate_hours: 2,
                  files_to_change: ['src/foo.ts'],
                  acceptance_criteria: 'not-a-list',
                  parallelizable: false,
                  dependencies: [],
                },
              ],
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'Slice acceptance_criteria must be a list').length >
          0,
      );
    });

    it('reports parallelizable not boolean in slice', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              expansion: 'slices',
              auto_expand: true,
              tdd_sequence: 'red-green',
              slices: [
                {
                  slice_id: 'S1',
                  title: 'Slice 1',
                  status: 'WIP',
                  goal: 'red-testing',
                  estimate_hours: 2,
                  files_to_change: ['src/foo.ts'],
                  acceptance_criteria: ['AC-1'],
                  parallelizable: 'not-boolean',
                  dependencies: [],
                },
              ],
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'Slice parallelizable must be a boolean').length > 0,
      );
    });

    it('reports dependencies not a list in slice', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              expansion: 'slices',
              auto_expand: true,
              tdd_sequence: 'red-green',
              slices: [
                {
                  slice_id: 'S1',
                  title: 'Slice 1',
                  status: 'WIP',
                  goal: 'red-testing',
                  estimate_hours: 2,
                  files_to_change: ['src/foo.ts'],
                  acceptance_criteria: ['AC-1'],
                  parallelizable: false,
                  dependencies: 'not-a-list',
                },
              ],
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'Slice dependencies must be a list').length > 0,
      );
    });

    it('reports estimate_hours not a number in slice', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              expansion: 'slices',
              auto_expand: true,
              tdd_sequence: 'red-green',
              slices: [
                {
                  slice_id: 'S1',
                  title: 'Slice 1',
                  status: 'WIP',
                  goal: 'red-testing',
                  estimate_hours: 'not-a-number',
                  files_to_change: ['src/foo.ts'],
                  acceptance_criteria: ['AC-1'],
                  parallelizable: false,
                  dependencies: [],
                },
              ],
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'Slice estimate_hours must be a number').length > 0,
      );
    });

    it('reports first slice goal mismatch for red-green', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              expansion: 'slices',
              auto_expand: true,
              tdd_sequence: 'red-green',
              slices: [
                {
                  slice_id: 'S1',
                  title: 'Slice 1',
                  status: 'WIP',
                  goal: 'implementing',
                  estimate_hours: 2,
                  files_to_change: ['src/foo.ts'],
                  acceptance_criteria: ['AC-1'],
                  parallelizable: false,
                  dependencies: [],
                },
              ],
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, "Expected slice 0 goal to be 'red-testing'").length >
          0,
      );
    });

    it('reports first slice goal mismatch for green-only', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              expansion: 'slices',
              auto_expand: true,
              tdd_sequence: 'green-only',
              slices: [
                {
                  slice_id: 'S1',
                  title: 'Slice 1',
                  status: 'WIP',
                  goal: 'red-testing',
                  estimate_hours: 2,
                  files_to_change: ['src/foo.ts'],
                  acceptance_criteria: ['AC-1'],
                  parallelizable: false,
                  dependencies: [],
                },
              ],
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, "Expected slice 0 goal to be 'implementing'")
          .length > 0,
      );
    });

    it('reports last slice goal mismatch', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              expansion: 'slices',
              auto_expand: true,
              tdd_sequence: 'red-green',
              slices: [
                {
                  slice_id: 'S1',
                  title: 'Slice 1',
                  status: 'WIP',
                  goal: 'red-testing',
                  estimate_hours: 2,
                  files_to_change: ['src/foo.ts'],
                  acceptance_criteria: ['AC-1'],
                  parallelizable: false,
                  dependencies: [],
                },
                {
                  slice_id: 'S2',
                  title: 'Slice 2',
                  status: 'WIP',
                  goal: 'implementing',
                  estimate_hours: 1,
                  files_to_change: ['src/bar.ts'],
                  acceptance_criteria: ['AC-2'],
                  parallelizable: false,
                  dependencies: ['S1'],
                },
              ],
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, "Expected final slice goal to be 'green-testing'")
          .length > 0,
      );
    });

    it('validates valid slices with red-green sequence', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              expansion: 'slices',
              auto_expand: true,
              tdd_sequence: 'red-green',
              slices: [
                {
                  slice_id: 'S1',
                  title: 'Slice 1',
                  status: 'WIP',
                  goal: 'red-testing',
                  estimate_hours: 2,
                  files_to_change: ['src/foo.ts'],
                  acceptance_criteria: ['AC-1'],
                  parallelizable: false,
                  dependencies: [],
                },
                {
                  slice_id: 'S2',
                  title: 'Slice 2',
                  status: 'WIP',
                  goal: 'green-testing',
                  estimate_hours: 1,
                  files_to_change: ['src/bar.ts'],
                  acceptance_criteria: ['AC-2'],
                  parallelizable: false,
                  dependencies: ['S1'],
                },
              ],
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      const sliceErrors = report.issues.filter((i) =>
        i.path.includes('slice-'),
      );
      assert.equal(sliceErrors.length, 0);
    });

    it('reports missing required sections in non-DONE step', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: null,
            yaml: makeValidStepYaml({ status: '[WIP]' }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'Missing required section').length > 0);
    });

    it('reports forbidden Copy-paste prompt in step', async () => {
      const planText = [
        '## Implementation phases',
        '',
        '### Phase A — Test Phase [WIP]',
        '',
        makePhaseSections('WIP'),
        '',
        yamlBlock(makeValidPhaseYaml({ status: '[WIP]' })),
        '',
        '#### Step 01 — Step One [WIP]',
        '',
        makeStepSections('WIP'),
        '',
        '**Copy-paste prompt:** Do stuff.',
        '',
        yamlBlock(makeValidStepYaml({ status: '[WIP]' })),
        '',
        '## Validation gates',
      ].join('\n');
      const report = await validatePlanText(planText, PLAN_PATH);
      assert.ok(
        findIssues(report, 'must not contain a separate Copy-paste prompt')
          .length > 0,
      );
    });

    it('reports agent_file does not exist', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'DONE',
            stepSections: makeStepSections('DONE'),
            yaml: {
              ...makeValidStepYaml({ status: '[DONE]' }),
              agent_file: 'nonexistent/file.mjs',
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'agent_file does not exist').length > 0);
    });

    it('reports agent_file that does exist (no error)', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              agent_file: 'scripts/agent-customization/customization-utils.mjs',
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.equal(findIssues(report, 'agent_file does not exist').length, 0);
    });

    it('produces phase summary with goal from WIP step', async () => {
      const plan = buildPlan([validPhase('WIP')]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.equal(report.phases[0].goal, 'implementing');
    });

    it('produces phase summary with goal from phase metadata when no WIP step', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'PLANNED',
        phaseSections: makePhaseSections('PLANNED'),
        yaml: makeValidPhaseYaml({ status: '[PLANNED]', goal: 'planning' }),
        steps: [],
      };
      const plan = buildPlan([phase, validPhase('WIP')]);
      const report = await validatePlanText(plan, PLAN_PATH);
      const plannedPhase = report.phases.find((p) => p.status === 'PLANNED');
      assert.equal(plannedPhase.goal, 'planning');
    });

    it('produces phase summary with null goal when no metadata', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'DONE',
        phaseSections: null,
        yaml: null,
        steps: [],
      };
      const wipPhase = validPhase('WIP');
      const plan = buildPlan([phase, wipPhase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      const donePhase = report.phases.find((p) => p.status === 'DONE');
      assert.equal(donePhase.goal, null);
    });

    it('handles numeric phase labels', async () => {
      const phase1 = {
        ...validPhase('WIP'),
        headingPhase: '1',
        yaml: makeValidPhaseYaml({
          phase: '1',
          status: '[WIP]',
          next_phase: '2',
        }),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ phase: '1', status: '[WIP]' }),
          },
        ],
      };
      const phase2 = {
        ...validPhase('DONE'),
        headingPhase: '2',
        yaml: makeValidPhaseYaml({
          phase: '2',
          status: '[DONE]',
          next_phase: '3',
        }),
        steps: [],
      };
      const plan = buildPlan([phase1, phase2]);
      const report = await validatePlanText(plan, PLAN_PATH);
      // Should not report phase label mismatch for 1 -> 2
      const labelErrors = findIssues(report, 'Expected phase');
      assert.equal(labelErrors.length, 0);
    });

    it('handles numeric phase label 0 -> A transition', async () => {
      const phase0 = {
        ...validPhase('DONE'),
        headingPhase: '0',
        yaml: makeValidPhaseYaml({
          phase: '0',
          status: '[DONE]',
          next_phase: 'A',
        }),
        steps: [],
      };
      const phaseA = validPhase('WIP');
      const plan = buildPlan([phase0, phaseA]);
      const report = await validatePlanText(plan, PLAN_PATH);
      // Phase 0 should expect next phase A
      const labelErrors = findIssues(report, 'Expected phase A, found phase A');
      // Phase A is correct, so no error
      assert.equal(labelErrors.length, 0);
    });

    it('handles Z phase label (next is null)', async () => {
      const phaseA = {
        ...validPhase('DONE'),
        headingPhase: 'Y',
        yaml: makeValidPhaseYaml({
          phase: 'Y',
          status: '[DONE]',
          next_phase: 'Z',
        }),
        steps: [],
      };
      const phaseZ = {
        ...validPhase('WIP'),
        headingPhase: 'Z',
        yaml: makeValidPhaseYaml({
          phase: 'Z',
          status: '[WIP]',
          next_phase: 'null',
        }),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ phase: 'Z', status: '[WIP]' }),
          },
        ],
      };
      const plan = buildPlan([phaseA, phaseZ]);
      const report = await validatePlanText(plan, PLAN_PATH);
      // Z -> null means no expected next label, so no error for Z being last
      const labelErrors = findIssues(report, 'Expected phase');
      // Y expects Z, found Z -> correct
      assert.equal(labelErrors.length, 0);
    });

    it('handles invalid phase label format (no next label)', async () => {
      // Phase label that is neither numeric nor single alpha
      const phaseA = validPhase('DONE');
      const phaseBad = {
        ...validPhase('WIP'),
        headingPhase: 'AB',
        yaml: makeValidPhaseYaml({ phase: 'AB', status: '[WIP]' }),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ phase: 'AB', status: '[WIP]' }),
          },
        ],
      };
      const plan = buildPlan([phaseA, phaseBad]);
      const report = await validatePlanText(plan, PLAN_PATH);
      // getNextPhaseLabel('A') returns 'B', but phase is 'AB'
      // normalizePhaseLabel('AB') = 'AB', expectedPhaseLabel = 'B'
      // So there should be a mismatch error
      assert.ok(
        findIssues(report, 'Expected phase B, found phase AB').length > 0,
      );
    });

    it('handles DONE step skipping title and acceptance_criteria', async () => {
      const phase = {
        ...validPhase('DONE'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'DONE',
            stepSections: null,
            yaml: {
              phase: 'A',
              step: 1,
              status: '[DONE]',
              goal: 'implementing',
              mode: 'fresh-session',
              source_of_truth: PLAN_PATH,
              copy_paste: true,
              next_step: 2,
              skills: ['skill'],
              validation: ['test.ts'],
              // Missing title and acceptance_criteria — allowed for DONE
            },
          },
        ],
      };
      const plan = buildPlan([phase, validPhase('WIP')]);
      const report = await validatePlanText(plan, PLAN_PATH);
      const missingKeyErrors = findIssues(
        report,
        'Missing step metadata key: title',
      );
      assert.equal(missingKeyErrors.length, 0);
      const missingAcErrors = findIssues(
        report,
        'Missing step metadata key: acceptance_criteria',
      );
      assert.equal(missingAcErrors.length, 0);
    });

    it('handles expansion: none in step (no slices validation)', async () => {
      const phase = {
        ...validPhase('WIP'),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              expansion: 'none',
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      const expansionErrors = findIssues(report, 'Invalid expansion');
      assert.equal(expansionErrors.length, 0);
    });

    it('handles perpetual mode for phase and step', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: makeValidPhaseYaml({ mode: 'perpetual', status: '[WIP]' }),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ mode: 'perpetual', status: '[WIP]' }),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      const modeErrors = findIssues(report, 'mode must be');
      assert.equal(modeErrors.length, 0);
    });

    it('handles copy_paste as string "true"', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: {
          ...makeValidPhaseYaml({ status: '[WIP]' }),
          copy_paste: 'true',
        },
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: {
              ...makeValidStepYaml({ status: '[WIP]' }),
              copy_paste: 'true',
            },
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      const copyPasteErrors = findIssues(report, 'copy_paste must be true');
      assert.equal(copyPasteErrors.length, 0);
    });

    it('handles validation entries with stale --testPathPattern flag', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: makeValidPhaseYaml({
          validation: ['--testPathPattern=foo'],
          status: '[WIP]',
        }),
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'stale --testPathPattern flag').length > 0);
    });

    it('handles validation entries that do not look like file paths', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: makeValidPhaseYaml({
          validation: ['justaword'],
          status: '[WIP]',
        }),
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(
        findIssues(report, 'does not look like a file path').length > 0,
      );
    });

    it('handles validation entries that are not strings', async () => {
      const phase = {
        ...validPhase('WIP'),
        yaml: { ...makeValidPhaseYaml({ status: '[WIP]' }), validation: [123] },
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.ok(findIssues(report, 'must be a string').length > 0);
    });

    it('returns phases summary with schema step for step-packet phases', async () => {
      const planText = [
        '## Implementation phases',
        '',
        '### Phase A — Test Phase [WIP]',
        '',
        makePhaseSections('WIP'),
        '',
        yamlBlock(makeValidStepYaml({ status: '[WIP]' })),
        '',
        '## Validation gates',
      ].join('\n');
      const report = await validatePlanText(planText, PLAN_PATH);
      assert.equal(report.phases[0].schema, 'step');
    });

    it('returns phases summary with schema phase for phase-packet phases', async () => {
      const plan = buildPlan([validPhase('WIP')]);
      const report = await validatePlanText(plan, PLAN_PATH);
      assert.equal(report.phases[0].schema, 'phase');
    });

    it('does not error on multi-char phase label in sequence (getNextPhaseLabel returns null)', async () => {
      const phaseAB = {
        headingPhase: 'AB',
        headingTitle: 'Phase AB',
        headingStatus: 'WIP',
        phaseSections: makePhaseSections('WIP'),
        yaml: makeValidPhaseYaml({
          phase: 'AB',
          status: '[WIP]',
          next_phase: 'AC',
        }),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml({ phase: 'AB', status: '[WIP]' }),
          },
        ],
      };
      const phaseAC = {
        ...validPhase('DONE'),
        headingPhase: 'AC',
        headingTitle: 'Phase AC',
        yaml: makeValidPhaseYaml({
          phase: 'AC',
          status: '[DONE]',
          next_phase: 'AD',
        }),
      };
      const plan = buildPlan([phaseAB, phaseAC]);
      const report = await validatePlanText(plan, PLAN_PATH);
      // getNextPhaseLabel('AB') returns null (not digit, not single char),
      // so the expected-vs-actual check is skipped — no phase sequence error
      const seqErrors = findIssues(report, 'Expected phase');
      assert.equal(seqErrors.length, 0);
    });

    it('covers getNextPhaseLabel Z -> null when Z is not the last phase', async () => {
      const phaseZ = {
        ...validPhase('DONE'),
        headingPhase: 'Z',
        headingTitle: 'Phase Z',
        yaml: makeValidPhaseYaml({
          phase: 'Z',
          status: '[DONE]',
          next_phase: 'AA',
        }),
        steps: [],
      };
      const phaseAA = {
        ...validPhase('WIP'),
        headingPhase: 'AA',
        headingTitle: 'Phase AA',
        yaml: makeValidPhaseYaml({ phase: 'AA', status: '[WIP]' }),
      };
      const plan = buildPlan([phaseZ, phaseAA]);
      const report = await validatePlanText(plan, PLAN_PATH);
      // getNextPhaseLabel('Z') returns null, so no expected-vs-actual check
      const seqErrors = findIssues(report, 'Expected phase');
      assert.equal(seqErrors.length, 0);
    });

    it('covers DONE step-packet (shouldRequireFullStepSequence false) with DONE step no YAML', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'DONE',
        phaseSections: makePhaseSections('DONE'),
        yaml: makeValidPhaseYaml({
          step: 1,
          phase: 'A',
          status: '[DONE]',
          expansion: 'steps',
        }),
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'DONE',
            stepSections: makeStepSections('DONE'),
            yaml: null,
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      // DONE step-packet: shouldRequireFullStepSequence is false (DONE status),
      // expectedStep falls through to stepBlock.headingStep (line 585).
      // DONE step with no YAML: metadata === null, headingStatus === 'DONE',
      // so no "must keep a YAML metadata block" error (line 601 false branch).
      const yamlErrors = findIssues(report, 'must keep a YAML metadata block');
      assert.equal(yamlErrors.length, 0);
    });

    it('covers isLegacyBlock via missing expansion field (line 1035)', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'WIP',
        phaseSections: makePhaseSections('WIP'),
        yaml: {
          phase: 'A',
          title: 'Test Phase',
          status: '[WIP]',
          goal: 'planning',
        },
        steps: [],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      // No expansion field → isLegacyBlock returns true at line 1035
      const legacyErrors = findIssues(report, 'Legacy plan block detected');
      assert.ok(legacyErrors.length > 0);
    });

    it('covers booleanValue with string copy_paste (line 1041)', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'WIP',
        phaseSections: makePhaseSections('WIP'),
        yaml: {
          ...makeValidPhaseYaml(),
          copy_paste: "'true'",
        },
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml(),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      // copy_paste: 'true' (quoted string) → booleanValue("true") returns true
      // (typeof value === 'string' → value === 'true' → true)
      const copyPasteErrors = findIssues(report, 'copy_paste must be true');
      assert.equal(copyPasteErrors.length, 0);
    });

    it('covers normalizePhaseLabel(undefined) and stripStatus(undefined) (lines 1060, 1064)', async () => {
      const phase = {
        headingPhase: 'A',
        headingTitle: 'Test Phase',
        headingStatus: 'WIP',
        phaseSections: makePhaseSections('WIP'),
        // YAML without 'phase' and 'status' fields — normalizePhaseLabel(undefined)
        // returns null (line 1060), stripStatus(undefined) returns null (line 1064)
        yaml: {
          title: 'Test Phase',
          goal: 'planning',
          expansion: 'steps',
          auto_expand: false,
          mode: 'fresh-session',
          source_of_truth: PLAN_PATH,
          copy_paste: true,
          next_phase: 'B',
          skills: ['plan-alignment'],
          validation: ['scripts/test.test.ts'],
          acceptance_criteria: ['AC-001: Criterion'],
          placeholder_steps: ['Step 01: Placeholder'],
        },
        steps: [
          {
            headingStep: 1,
            headingTitle: 'Step One',
            headingStatus: 'WIP',
            stepSections: makeStepSections('WIP'),
            yaml: makeValidStepYaml(),
          },
        ],
      };
      const plan = buildPlan([phase]);
      const report = await validatePlanText(plan, PLAN_PATH);
      // Missing phase and status keys produce errors, but the code continues
      // to call normalizePhaseLabel(undefined) and stripStatus(undefined)
      const missingPhaseErrors = findIssues(
        report,
        'Missing phase metadata key: phase',
      );
      assert.ok(missingPhaseErrors.length > 0);
      const missingStatusErrors = findIssues(
        report,
        'Missing phase metadata key: status',
      );
      assert.ok(missingStatusErrors.length > 0);
    });
  });
});
