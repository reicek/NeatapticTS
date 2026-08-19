/**
 * @fileoverview Coverage tests for mcp-plan-utils.mjs — plan parser and snapshot builders.
 * Creates temporary plan files under plans/ and cleans them up after each test.
 */

import { writeFile, unlink, readFile } from 'node:fs/promises';
import path from 'node:path';

import {
  loadActivePlanContext,
  resolveEffectivePlanPath,
  createWorkflowSnapshot,
  createValidationAllowlistSnapshot,
} from './mcp-plan-utils.mjs';
import { MCP_REPO_ROOT } from './mcp-utils.mjs';

const PLANS_DIR = path.join(MCP_REPO_ROOT, 'plans');
const SESSION_OVERRIDE_PATH = path.join(
  MCP_REPO_ROOT,
  'data',
  'mcp-session-override.json',
);
const createdFiles = [];
let originalOverrideContent = null;
let overrideExisted = false;

/** Create a temporary plan file under plans/ and track it for cleanup. */
async function createPlanFile(name, content) {
  const filePath = path.join(PLANS_DIR, name);
  await writeFile(filePath, content, 'utf8');
  createdFiles.push(filePath);
  return `plans/${name}`;
}

/** Save the original session override file content. */
async function saveOverride() {
  try {
    originalOverrideContent = await readFile(SESSION_OVERRIDE_PATH, 'utf8');
    overrideExisted = true;
  } catch {
    originalOverrideContent = null;
    overrideExisted = false;
  }
}

/** Restore the original session override file content. */
async function restoreOverride() {
  if (overrideExisted && originalOverrideContent !== null) {
    await writeFile(SESSION_OVERRIDE_PATH, originalOverrideContent, 'utf8');
  } else {
    try {
      await unlink(SESSION_OVERRIDE_PATH);
    } catch {
      // already gone
    }
  }
}

/** Write a session override file. */
async function writeOverride(content) {
  await writeFile(SESSION_OVERRIDE_PATH, content, 'utf8');
}

/** Delete the session override file. */
async function deleteOverride() {
  try {
    await unlink(SESSION_OVERRIDE_PATH);
  } catch {
    // already gone
  }
}

beforeAll(saveOverride);

afterEach(async () => {
  for (const filePath of createdFiles) {
    try {
      await unlink(filePath);
    } catch {
      // already gone
    }
  }
  createdFiles.length = 0;
  await restoreOverride();
});

// ── Plan fixtures ──────────────────────────────────────────────────────────

const NORMAL_PLAN = `# Normal Test Plan

## Implementation phases

### Phase 1 — Test Phase [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — Test Step [WIP]

\`\`\`yaml
agent: '04-implementing'
goal: 'implement feature'
validation:
  - 'npm test'
  - 'npm run lint'
slices:
  - slice_id: S1
    title: 'Slice 1'
    status: '[WIP]'
    goal: 'Goal 1'
  - slice_id: S2
    title: 'Slice 2'
    status: '[PLANNED]'
    goal: 'Goal 2'
\`\`\`

**Step objective:** Implement the feature correctly.

**Required validation:**
- \`npm test\`
- \`npm run lint\`

**Files:** \`src/test.ts\`

## Validation gates
`;

const PHASE_ONLY_PLAN = `# Phase Only Test Plan

## Implementation phases

### Phase 2 — Phase Only [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — Planned Step [PLANNED]

\`\`\`yaml
agent: '04-implementing'
goal: 'will be implemented'
\`\`\`

**Step objective:** TBD.

## Validation gates
`;

const NO_WIP_STEP_PLAN = `# No WIP Step Test Plan

## Implementation phases

### Phase 3 — No WIP Step [WIP]

#### Step 01 — Planned Step [PLANNED]

**Step objective:** TBD.

## Validation gates
`;

const MULTI_WIP_STEP_PLAN = `# Multi WIP Step Test Plan

## Implementation phases

### Phase 4 — Multi WIP [WIP]

#### Step 01 — First WIP [WIP]

\`\`\`yaml
agent: '04-implementing'
\`\`\`

#### Step 02 — Second WIP [WIP]

\`\`\`yaml
agent: '04-implementing'
\`\`\`

## Validation gates
`;

const NO_WIP_PHASE_PLAN = `# No WIP Phase Test Plan

## Implementation phases

### Phase 5 — Done Phase [DONE]

\`\`\`yaml
expansion: steps
\`\`\`

## Validation gates
`;

const TWO_WIP_PHASES_PLAN = `# Two WIP Phases Test Plan

## Implementation phases

### Phase 6 — First WIP Phase [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

### Phase 7 — Second WIP Phase [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

## Validation gates
`;

const NO_IMPL_SECTION_PLAN = `# No Implementation Section

Some content but no implementation phases section.

## Validation gates
`;

const STEP_NO_YAML_PLAN = `# Step No YAML Test Plan

## Implementation phases

### Phase 8 — No YAML Step [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — No YAML [WIP]

**Step objective:** No YAML here.

## Validation gates
`;

const PLANNED_SLICES_PLAN = `# Planned Slices Test Plan

## Implementation phases

### Phase 9 — Planned Slices [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — Planned Slices Step [WIP]

\`\`\`yaml
agent: '04-implementing'
goal: 'implement'
slices:
  - slice_id: S1
    title: 'Slice 1'
    status: '[PLANNED]'
    goal: 'Goal 1'
  - slice_id: S2
    title: 'Slice 2'
    status: '[PLANNED]'
    goal: 'Goal 2'
\`\`\`

**Step objective:** Implement planned slices.

## Validation gates
`;

const NO_SLICES_PLAN = `# No Slices Test Plan

## Implementation phases

### Phase 10 — No Slices [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — No Slices Step [WIP]

\`\`\`yaml
agent: '04-implementing'
goal: 'implement'
\`\`\`

**Step objective:** Implement without slices.

## Validation gates
`;

const DONE_SLICES_PLAN = `# Done Slices Test Plan

## Implementation phases

### Phase 11 — Done Slices [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — Done Slices Step [WIP]

\`\`\`yaml
agent: '04-implementing'
goal: 'implement'
slices:
  - slice_id: S1
    title: 'Slice 1'
    status: '[DONE]'
    goal: 'Goal 1'
\`\`\`

**Step objective:** Implement with done slices.

## Validation gates
`;

const SINGLE_SECTION_PLAN = `# Single Section Test Plan

## Implementation phases

### Phase 12 — Single Section [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — Single Section Step [WIP]

\`\`\`yaml
agent: '04-implementing'
goal: 'implement'
\`\`\`

**Step objective:** This is the only section in the step body.

## Validation gates
`;

const EMPTY_OBJECTIVE_PLAN = `# Empty Objective Test Plan

## Implementation phases

### Phase 13 — Empty Objective [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — Empty Objective Step [WIP]

\`\`\`yaml
agent: '04-implementing'
goal: 'implement'
validation:
  - 'npm test'
\`\`\`

**Step objective:**
**Required validation:**
- \`npm test\`

## Validation gates
`;

const VAL_MISMATCH_PLAN = `# Validation Mismatch Test Plan

## Implementation phases

### Phase 14 — Val Mismatch [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — Val Mismatch Step [WIP]

\`\`\`yaml
agent: '04-implementing'
goal: 'implement'
validation:
  - 'npm test'
  - 'npm run lint'
  - 'npm run build'
\`\`\`

**Step objective:** Implement with mismatched validation.

**Required validation:**
- \`npm test\`

## Validation gates
`;

const LETTER_STEP_PLAN = `# Letter Step Test Plan

## Implementation phases

### Phase 15 — Letter Step [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step E1 — Letter Step [WIP]

\`\`\`yaml
agent: '04-implementing'
goal: 'implement'
\`\`\`

**Step objective:** Implement with letter-prefixed step.

## Validation gates
`;

const NO_AGENT_NO_GOAL_PLAN = `# No Agent No Goal Test Plan

## Implementation phases

### Phase 16 — No Agent Goal [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — No Agent Goal Step [WIP]

\`\`\`yaml
validation:
  - 'npm test'
\`\`\`

**Step objective:** Step with neither agent nor goal.

**Required validation:**
- \`npm test\`

## Validation gates
`;

const SPARSE_SLICES_PLAN = `# Sparse Slices Test Plan

## Implementation phases

### Phase 17 — Sparse Slices [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — Sparse Slices Step [WIP]

\`\`\`yaml
agent: '04-implementing'
goal: 'implement'
slices:
  - slice_id: S1
\`\`\`

**Step objective:** Step with sparse slice fields.

## Validation gates
`;

const NO_SLICE_ID_PLAN = `# No Slice ID Test Plan

## Implementation phases

### Phase 21 — No Slice ID [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — No Slice ID Step [WIP]

\`\`\`yaml
agent: '04-implementing'
goal: 'implement'
slices:
  - title: 'No ID Slice'
\`\`\`

**Step objective:** Step with slice missing slice_id.

## Validation gates
`;

const MULTI_STEP_PLAN = `# Multi Step Test Plan

## Implementation phases

### Phase 18 — Multi Step [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — Planned First [PLANNED]

\`\`\`yaml
agent: '04-implementing'
goal: 'planned'
\`\`\`

**Step objective:** First planned step.

#### Step 02 — WIP Second [WIP]

\`\`\`yaml
agent: '04-implementing'
goal: 'implement'
\`\`\`

**Step objective:** Second WIP step.

## Validation gates
`;

const MULTI_PHASE_PLAN = `# Multi Phase Test Plan

## Implementation phases

### Phase 19 — Done Phase [DONE]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — Done Step [DONE]

\`\`\`yaml
agent: '04-implementing'
goal: 'done'
\`\`\`

**Step objective:** Done step.

### Phase 20 — WIP Phase [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — WIP Step [WIP]

\`\`\`yaml
agent: '04-implementing'
goal: 'implement'
\`\`\`

**Step objective:** WIP step.

## Validation gates
`;

const LETTER_PHASE_PLAN = `# Letter Phase Test Plan

## Implementation phases

### Phase A — Letter Phase [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — Letter Phase Step [WIP]

\`\`\`yaml
agent: '04-implementing'
goal: 'implement'
\`\`\`

**Step objective:** Step with letter phase.

## Validation gates
`;

// ── loadActivePlanContext tests ─────────────────────────────────────────────

describe('loadActivePlanContext', () => {
  it('loads normal plan with WIP step, slices, and validation', async () => {
    const planPath = await createPlanFile(
      'test-cov-normal.plans.md',
      NORMAL_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    expect(ctx.planPath).toContain('test-cov-normal.plans.md');
    expect(ctx.activePhase.number).toBe(1);
    expect(ctx.activePhase.title).toBe('Test Phase');
    expect(ctx.activePhase.status).toBe('WIP');
    expect(ctx.activeStep.number).toBe(1);
    expect(ctx.activeStep.title).toBe('Test Step');
    expect(ctx.activeStep.status).toBe('WIP');
    expect(ctx.activeStep.metadata.agent).toBe('04-implementing');
    expect(ctx.activeStep.stepObjective).toBe(
      'Implement the feature correctly.',
    );
    expect(ctx.activeStep.validationCommands).toEqual([
      'npm test',
      'npm run lint',
    ]);
    expect(ctx.activeStep.requiredValidationCommands).toEqual([
      'npm test',
      'npm run lint',
    ]);
    expect(ctx.activeStep.validationCommandsMatch).toBe(true);
    expect(ctx.activeStep.activeSlice).toEqual({
      slice_id: 'S1',
      title: 'Slice 1',
      status: '[WIP]',
      goal: 'Goal 1',
    });
    expect(ctx.phaseMetadata).toEqual({
      expansion: 'steps',
      auto_expand: false,
    });
  });

  it('loads phase-only plan with auto_expand=false (no WIP step)', async () => {
    const planPath = await createPlanFile(
      'test-cov-phase-only.plans.md',
      PHASE_ONLY_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    expect(ctx.activeStep).toBeNull();
    expect(ctx.phaseMetadata).toEqual({
      expansion: 'steps',
      auto_expand: false,
    });
    expect(ctx.activePhase.number).toBe(2);
  });

  it('throws when WIP phase has no WIP step and auto_expand is not false', async () => {
    const planPath = await createPlanFile(
      'test-cov-no-wip-step.plans.md',
      NO_WIP_STEP_PLAN,
    );
    await expect(loadActivePlanContext(planPath)).rejects.toThrow(
      'is [WIP] but has no [WIP] step',
    );
  });

  it('throws when multiple WIP steps exist', async () => {
    const planPath = await createPlanFile(
      'test-cov-multi-wip.plans.md',
      MULTI_WIP_STEP_PLAN,
    );
    await expect(loadActivePlanContext(planPath)).rejects.toThrow(
      'Expected at most one [WIP] step',
    );
  });

  it('throws when no WIP phase exists (all DONE)', async () => {
    const planPath = await createPlanFile(
      'test-cov-no-wip-phase.plans.md',
      NO_WIP_PHASE_PLAN,
    );
    await expect(loadActivePlanContext(planPath)).rejects.toThrow(
      'Expected exactly one [WIP] phase',
    );
  });

  it('throws when multiple WIP phases exist', async () => {
    const planPath = await createPlanFile(
      'test-cov-two-wip-phases.plans.md',
      TWO_WIP_PHASES_PLAN,
    );
    await expect(loadActivePlanContext(planPath)).rejects.toThrow(
      'Expected exactly one [WIP] phase',
    );
  });

  it('throws when no implementation section exists', async () => {
    const planPath = await createPlanFile(
      'test-cov-no-impl.plans.md',
      NO_IMPL_SECTION_PLAN,
    );
    await expect(loadActivePlanContext(planPath)).rejects.toThrow(
      'Expected exactly one [WIP] phase',
    );
  });

  it('throws when WIP step has no YAML metadata block', async () => {
    const planPath = await createPlanFile(
      'test-cov-step-no-yaml.plans.md',
      STEP_NO_YAML_PLAN,
    );
    await expect(loadActivePlanContext(planPath)).rejects.toThrow(
      'missing a YAML metadata block',
    );
  });

  it('returns planned slice when no WIP slice exists', async () => {
    const planPath = await createPlanFile(
      'test-cov-planned-slices.plans.md',
      PLANNED_SLICES_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    expect(ctx.activeStep.activeSlice).toEqual({
      slice_id: 'S1',
      title: 'Slice 1',
      status: '[PLANNED]',
      goal: 'Goal 1',
    });
  });

  it('returns null activeSlice when step has no slices', async () => {
    const planPath = await createPlanFile(
      'test-cov-no-slices.plans.md',
      NO_SLICES_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    expect(ctx.activeStep.activeSlice).toBeNull();
  });

  it('returns first slice as fallback when all slices are DONE', async () => {
    const planPath = await createPlanFile(
      'test-cov-done-slices.plans.md',
      DONE_SLICES_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    expect(ctx.activeStep.activeSlice).toEqual({
      slice_id: 'S1',
      title: 'Slice 1',
      status: '[DONE]',
      goal: 'Goal 1',
    });
  });

  it('handles step with only one section (no next section match)', async () => {
    const planPath = await createPlanFile(
      'test-cov-single-section.plans.md',
      SINGLE_SECTION_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    expect(ctx.activeStep.stepObjective).toBe(
      'This is the only section in the step body.',
    );
    // No Required validation section → empty commands
    expect(ctx.activeStep.requiredValidationCommands).toEqual([]);
  });

  it('handles step with empty Step objective followed by Required validation', async () => {
    const planPath = await createPlanFile(
      'test-cov-empty-objective.plans.md',
      EMPTY_OBJECTIVE_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    // Step objective is empty (next section starts at index 0 of afterMarker)
    expect(ctx.activeStep.stepObjective).toBe('');
    expect(ctx.activeStep.requiredValidationCommands).toEqual(['npm test']);
  });

  it('detects validation command mismatch (YAML has more than prose)', async () => {
    const planPath = await createPlanFile(
      'test-cov-val-mismatch.plans.md',
      VAL_MISMATCH_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    expect(ctx.activeStep.validationCommands).toEqual([
      'npm test',
      'npm run lint',
      'npm run build',
    ]);
    expect(ctx.activeStep.requiredValidationCommands).toEqual(['npm test']);
    expect(ctx.activeStep.validationCommandsMatch).toBe(false);
  });

  it('handles letter-prefixed step identifier', async () => {
    const planPath = await createPlanFile(
      'test-cov-letter-step.plans.md',
      LETTER_STEP_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    expect(ctx.activeStep.number).toBe('E1');
  });

  it('throws ENOENT error for nonexistent plan file', async () => {
    await expect(
      loadActivePlanContext('plans/nonexistent-test-file-12345.plans.md'),
    ).rejects.toThrow('Plan file not found');
  });

  it('rethrows non-ENOENT errors (e.g., EISDIR for directory)', async () => {
    // Passing '.' resolves to MCP_REPO_ROOT which is a directory
    await expect(loadActivePlanContext('.')).rejects.toThrow();
  });

  it('uses null agent when both agent and goal are absent', async () => {
    const planPath = await createPlanFile(
      'test-cov-no-agent-no-goal.plans.md',
      NO_AGENT_NO_GOAL_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    const snapshot = createWorkflowSnapshot(ctx);
    expect(snapshot.activeStep.agent).toBeNull();
  });

  it('handles sparse slices with missing fields (covers ?? fallbacks in pickSliceFields)', async () => {
    const planPath = await createPlanFile(
      'test-cov-sparse-slices.plans.md',
      SPARSE_SLICES_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    expect(ctx.activeStep.activeSlice).toEqual({
      slice_id: 'S1',
      title: '',
      status: '',
      goal: '',
    });
  });

  it('handles slice with missing slice_id (covers ?? fallback for slice_id)', async () => {
    const planPath = await createPlanFile(
      'test-cov-no-slice-id.plans.md',
      NO_SLICE_ID_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    expect(ctx.activeStep.activeSlice).toEqual({
      slice_id: '',
      title: 'No ID Slice',
      status: '',
      goal: '',
    });
  });

  it('handles multi-step plan (covers nextStepMatch?.index branch)', async () => {
    const planPath = await createPlanFile(
      'test-cov-multi-step.plans.md',
      MULTI_STEP_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    expect(ctx.activeStep.number).toBe(2);
    expect(ctx.activeStep.title).toBe('WIP Second');
  });

  it('handles multi-phase plan with DONE + WIP (covers nextPhaseMatch?.index branch)', async () => {
    const planPath = await createPlanFile(
      'test-cov-multi-phase.plans.md',
      MULTI_PHASE_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    expect(ctx.activePhase.number).toBe(20);
    expect(ctx.activePhase.title).toBe('WIP Phase');
  });

  it('handles non-numeric phase label (covers ternary false branch)', async () => {
    const planPath = await createPlanFile(
      'test-cov-letter-phase.plans.md',
      LETTER_PHASE_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    expect(ctx.activePhase.number).toBe('A');
  });

  it('uses null agent in validation snapshot when both agent and goal absent', async () => {
    const planPath = await createPlanFile(
      'test-cov-no-agent-no-goal2.plans.md',
      NO_AGENT_NO_GOAL_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    const snapshot = createValidationAllowlistSnapshot(ctx);
    expect(snapshot.activeStep.agent).toBeNull();
  });
});

// ── resolveEffectivePlanPath tests ──────────────────────────────────────────

describe('resolveEffectivePlanPath', () => {
  beforeAll(saveOverride);
  afterAll(restoreOverride);

  it('uses plan_path argument when provided', async () => {
    const result = await resolveEffectivePlanPath(
      { plan_path: 'plans/test.plans.md' },
      'plans/fallback.plans.md',
    );
    expect(result).toBe('plans/test.plans.md');
  });

  it('throws when plan_path resolves outside plans/', async () => {
    await expect(
      resolveEffectivePlanPath(
        { plan_path: '../outside.md' },
        'plans/fallback.plans.md',
      ),
    ).rejects.toThrow('plan_path must resolve within plans/');
  });

  it('accepts absolute plan_path within plans/', async () => {
    const absPath = path.join(MCP_REPO_ROOT, 'plans', 'test-abs.plans.md');
    const result = await resolveEffectivePlanPath(
      { plan_path: absPath },
      'plans/fallback.plans.md',
    );
    expect(result).toBe('plans/test-abs.plans.md');
  });

  it('uses session override when no plan_path arg', async () => {
    await writeOverride(
      JSON.stringify({ plan_path: 'plans/from-override.plans.md' }),
    );
    const result = await resolveEffectivePlanPath(
      {},
      'plans/fallback.plans.md',
    );
    expect(result).toBe('plans/from-override.plans.md');
  });

  it('falls back to startup when session override has invalid JSON', async () => {
    await writeOverride('not valid json');
    const result = await resolveEffectivePlanPath(
      {},
      'plans/fallback.plans.md',
    );
    expect(result).toBe('plans/fallback.plans.md');
  });

  it('falls back to startup when plan_path is not a string', async () => {
    await writeOverride(JSON.stringify({ plan_path: 123 }));
    const result = await resolveEffectivePlanPath(
      {},
      'plans/fallback.plans.md',
    );
    expect(result).toBe('plans/fallback.plans.md');
  });

  it('falls back to startup when plan_path is empty/whitespace', async () => {
    await writeOverride(JSON.stringify({ plan_path: '   ' }));
    const result = await resolveEffectivePlanPath(
      {},
      'plans/fallback.plans.md',
    );
    expect(result).toBe('plans/fallback.plans.md');
  });

  it('falls back to startup when session override plan_path is outside plans/', async () => {
    await writeOverride(JSON.stringify({ plan_path: '../outside.md' }));
    const result = await resolveEffectivePlanPath(
      {},
      'plans/fallback.plans.md',
    );
    expect(result).toBe('plans/fallback.plans.md');
  });

  it('falls back to startup when session override file does not exist', async () => {
    await deleteOverride();
    const result = await resolveEffectivePlanPath(
      {},
      'plans/fallback.plans.md',
    );
    expect(result).toBe('plans/fallback.plans.md');
  });

  it('falls back to startup when no plan_path and no override and no startup', async () => {
    await deleteOverride();
    const result = await resolveEffectivePlanPath({}, undefined);
    expect(result).toBeUndefined();
  });
});

// ── createWorkflowSnapshot tests ────────────────────────────────────────────

describe('createWorkflowSnapshot', () => {
  it('builds snapshot with activeStep', async () => {
    const planPath = await createPlanFile(
      'test-cov-normal.plans.md',
      NORMAL_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    const snapshot = createWorkflowSnapshot(ctx);
    expect(snapshot.scope).toBe('repo-static');
    expect(snapshot.plan).toContain('test-cov-normal.plans.md');
    expect(snapshot.activePhase.number).toBe(1);
    expect(snapshot.activeStep).not.toBeNull();
    expect(snapshot.activeStep.number).toBe(1);
    expect(snapshot.activeStep.agent).toBe('04-implementing');
    expect(snapshot.activeStep.agentFile).toBeNull();
    expect(snapshot.activeStep.objective).toBe(
      'Implement the feature correctly.',
    );
    expect(snapshot.activeStep.nextStep).toBeNull();
    expect(snapshot.activeStep.validationCommands).toEqual([
      'npm test',
      'npm run lint',
    ]);
    expect(snapshot.activeStep.activeSlice.slice_id).toBe('S1');
    expect(snapshot.phaseMetadata).toEqual({
      expansion: 'steps',
      auto_expand: false,
    });
    expect(snapshot.allowlistAuthority).toBe('active-step.validation');
    expect(snapshot.sourceBoundary).toEqual([
      'plan metadata',
      'deterministic customization scripts',
    ]);
  });

  it('builds snapshot with null activeStep (phase-only)', async () => {
    const planPath = await createPlanFile(
      'test-cov-phase-only.plans.md',
      PHASE_ONLY_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    const snapshot = createWorkflowSnapshot(ctx);
    expect(snapshot.activeStep).toBeNull();
    expect(snapshot.activePhase.number).toBe(2);
    expect(snapshot.phaseMetadata).toEqual({
      expansion: 'steps',
      auto_expand: false,
    });
  });

  it('uses metadata.goal as agent when agent is absent', async () => {
    const planContent = NORMAL_PLAN.replace(
      "agent: '04-implementing'",
      '# agent removed',
    );
    const planPath = await createPlanFile(
      'test-cov-no-agent.plans.md',
      planContent,
    );
    const ctx = await loadActivePlanContext(planPath);
    const snapshot = createWorkflowSnapshot(ctx);
    expect(snapshot.activeStep.agent).toBe('implement feature');
  });
});

// ── createValidationAllowlistSnapshot tests ─────────────────────────────────

describe('createValidationAllowlistSnapshot', () => {
  it('builds allowlist with activeStep', async () => {
    const planPath = await createPlanFile(
      'test-cov-normal.plans.md',
      NORMAL_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    const snapshot = createValidationAllowlistSnapshot(ctx);
    expect(snapshot.scope).toBe('direct-MCP');
    expect(snapshot.plan).toContain('test-cov-normal.plans.md');
    expect(snapshot.activePhase.number).toBe(1);
    expect(snapshot.activeStep).not.toBeNull();
    expect(snapshot.activeStep.agent).toBe('04-implementing');
    expect(snapshot.activeSlice.slice_id).toBe('S1');
    expect(snapshot.allowlistAuthority).toBe('active-step.validation');
    expect(snapshot.validationCommands).toEqual(['npm test', 'npm run lint']);
    expect(snapshot.requiredValidationCommands).toEqual([
      'npm test',
      'npm run lint',
    ]);
    expect(snapshot.validationCommandsMatch).toBe(true);
  });

  it('builds allowlist with null activeStep (phase-only)', async () => {
    const planPath = await createPlanFile(
      'test-cov-phase-only.plans.md',
      PHASE_ONLY_PLAN,
    );
    const ctx = await loadActivePlanContext(planPath);
    const snapshot = createValidationAllowlistSnapshot(ctx);
    expect(snapshot.activeStep).toBeNull();
    expect(snapshot.activeSlice).toBeNull();
    expect(snapshot.validationCommands).toEqual([]);
    expect(snapshot.requiredValidationCommands).toEqual([]);
    expect(snapshot.validationCommandsMatch).toBe(false);
  });

  it('uses metadata.goal as agent when agent is absent', async () => {
    const planContent = NORMAL_PLAN.replace(
      "agent: '04-implementing'",
      '# agent removed',
    );
    const planPath = await createPlanFile(
      'test-cov-no-agent2.plans.md',
      planContent,
    );
    const ctx = await loadActivePlanContext(planPath);
    const snapshot = createValidationAllowlistSnapshot(ctx);
    expect(snapshot.activeStep.agent).toBe('implement feature');
  });
});
