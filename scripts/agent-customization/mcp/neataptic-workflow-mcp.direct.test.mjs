/**
 * @module neataptic-workflow-mcp.direct.test
 * @description Native-ESM red-phase contract tests for slice context retention.
 *
 * Issue 2: once a step is marked [DONE] and its details are archived to the
 * companion `.logs.md` file, `get_slice_context` currently returns
 * `{ notFound: true }`. These tests create real temporary plan/logs fixtures,
 * invoke the `get_slice_context` tool with an injected `searchContextFn`, and
 * assert that a [DONE] slice remains retrievable from the archive.
 *
 * Pure dependency injection is used: only the `searchContextFn` seam is mocked;
 * filesystem reads are exercised through ephemeral test fixtures so the test
 * exercises the same parsing path as production.
 */

import { writeFile, unlink } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createWorkflowTools } from './neataptic-workflow-mcp.mjs';
import {
  deriveLogsPath,
  findArchivedSliceDescriptor,
} from './slice-context-archive.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '../../../');

/**
 * Return an injectable `searchContextFn` that yields a deterministic empty
 * result. The red contract is about descriptor lookup, not RAG ranking, so the
 * returned corpus chunks are intentionally minimal.
 *
 * @param {Array<Record<string, unknown>>} [results=[]] - Optional ranked chunks.
 * @returns {Function} Injectable Cortex search_context replacement.
 */
function makeSearchContextFn(results = []) {
  return async () => ({
    dense_state: 'cold',
    results,
  });
}

/**
 * Write a temporary plan file under `plans/` and return its repo-relative path.
 *
 * @param {string} content - Markdown plan body.
 * @returns {Promise<string>} Repo-relative path of the temp plan.
 */
async function writeTempPlan(content) {
  const fileName = `__test-archive-${Date.now()}-${Math.random()
    .toString(36)
    .slice(2)}.plans.md`;
  const absolutePath = path.resolve(REPO_ROOT, 'plans', fileName);
  await writeFile(absolutePath, content.trim(), 'utf8');
  return path.relative(REPO_ROOT, absolutePath).replace(/\\/g, '/');
}

/**
 * Write the companion `.logs.md` file for a given plan path.
 *
 * @param {string} planPath - Repo-relative plan path (used to derive log path).
 * @param {string} content - Markdown logs body.
 * @returns {Promise<string>} Repo-relative path of the temp logs file.
 */
async function writeTempLogs(planPath, content) {
  const logsPath = planPath.replace(/\.plans\.md$/, '.logs.md');
  const absolutePath = path.resolve(REPO_ROOT, logsPath);
  await writeFile(absolutePath, content.trim(), 'utf8');
  return logsPath;
}

/**
 * Best-effort cleanup of a temp plan and its companion log file.
 *
 * @param {string} planPath - Repo-relative plan path.
 * @returns {Promise<void>}
 */
async function removeTempPlan(planPath) {
  for (const suffix of ['.plans.md', '.logs.md']) {
    const filePath = planPath.replace(/\.plans\.md$/, suffix);
    try {
      await unlink(path.resolve(REPO_ROOT, filePath));
    } catch {
      // Fixture may not exist; ignore cleanup failures.
    }
  }
}

describe('neataptic-workflow-mcp get_slice_context archive retention', () => {
  /**
   * Create a temporary plan with one active step and one compressed [DONE]
   * step, plus a companion `.logs.md` file containing the archived slice.
   * Invoke `get_slice_context` for the archived slice and return the tool
   * result. Cleans up both temp files even if the lookup throws.
   *
   * @returns {Promise<Record<string, unknown>>} The `get_slice_context` result.
   */
  async function runArchiveLookup() {
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
slices:
  - slice_id: active-slice
    title: Active slice
    status: '[WIP]'
\`\`\`

#### Step 02: Archived step [DONE]

\`\`\`yaml
phase: 1
step: 2
status: '[DONE]'
\`\`\`

## Validation gates

- none
`);

    await writeTempLogs(
      planPath,
      `
# Test Plan — Step Logs

## Phase 1 — Test

### Step 02: Archived step [DONE]

\`\`\`yaml
phase: 1
step: 2
title: 'Archived step'
slices:
  - slice_id: done-slice
    title: Archived slice
    status: '[DONE]'
    goal: 'Retrieve archived slice context'
    files_to_change:
      - scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs
      - scripts/agent-customization/mcp/slice-context-archive.mjs
    acceptance_criteria:
      - id: AC-ARCH-001
        text: 'get_slice_context returns the archived slice instead of notFound'
\`\`\`

**Validation evidence:**
- red test fails before implementation
`,
    );

    try {
      const tools = createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn(),
      });
      const tool = tools.find((t) => t.name === 'get_slice_context');

      return await tool.handler({
        slice_id: 'done-slice',
        plan_path: planPath,
      });
    } finally {
      await removeTempPlan(planPath);
    }
  }

  /**
   * Red contract for Issue 2: a completed slice whose details have been moved
   * to the companion `.logs.md` archive must still resolve through
   * `get_slice_context`.
   */
  it('resolves the archived [DONE] slice instead of returning notFound', async () => {
    const result = await runArchiveLookup();

    expect(result.notFound).toBeFalsy();
    expect(result.slice_id).toBe('done-slice');
    expect(result.title).toBe('Archived slice');
  });

  it('preserves the slice status and goal from the logs archive', async () => {
    const result = await runArchiveLookup();

    expect(result.status).toBe('[DONE]');
    expect(result.goal).toBe('Retrieve archived slice context');
  });

  it('preserves files_to_change and acceptance_criteria from the logs archive', async () => {
    const result = await runArchiveLookup();

    expect(result.files_to_change).toEqual([
      'scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs',
      'scripts/agent-customization/mcp/slice-context-archive.mjs',
    ]);
    expect(result.acceptance_criteria).toContainEqual({
      id: 'AC-ARCH-001',
      text: 'get_slice_context returns the archived slice instead of notFound',
      validation: null,
    });
  });

  it('marks archived responses with archived: true', async () => {
    const result = await runArchiveLookup();

    expect(result.archived).toBe(true);
  });
});

describe('findArchivedSliceDescriptor error paths', () => {
  /**
   * Build an injectable `readFile` that rejects as if the file is missing.
   *
   * @returns {Function} Stubbed readFile that throws an ENOENT error.
   */
  function makeMissingFileRead() {
    return async () => {
      const error = new Error('ENOENT: no such file or directory');
      error.code = 'ENOENT';
      throw error;
    };
  }

  it('returns null when the .logs.md file does not exist', async () => {
    const result = await findArchivedSliceDescriptor(
      'plans/missing.plans.md',
      'any-slice',
      { readFile: makeMissingFileRead() },
    );

    expect(result).toBeNull();
  });

  it('returns null when the .logs.md file contains malformed YAML blocks', async () => {
    const readFile = async () => `
# Log archive

\`\`\`yaml
this is not valid yaml: [unclosed
\`\`\`

\`\`\`yaml
slices:
  - slice_id: good-slice
    title: Good slice
\`\`\`
`;

    const result = await findArchivedSliceDescriptor(
      'plans/malformed.plans.md',
      'wanted-slice',
      { readFile },
    );

    expect(result).toBeNull();
  });
});

describe('deriveLogsPath edge cases', () => {
  it('replaces the .plans.md suffix with .logs.md', () => {
    expect(deriveLogsPath('plans/demo.plans.md')).toBe('plans/demo.logs.md');
  });

  it('returns the original path when it does not end in .plans.md', () => {
    expect(deriveLogsPath('plans/demo.md')).toBe('plans/demo.md');
  });
});

describe('findArchivedSliceDescriptor branch coverage', () => {
  /**
   * Build an injectable readFile that returns the supplied Markdown contents.
   *
   * @param {string} contents - Markdown/YAML fixture contents.
   * @returns {Function} Stubbed readFile.
   */
  function makeReadFile(contents) {
    return async () => contents;
  }

  it('returns null when a YAML block has no closing fence', async () => {
    const result = await findArchivedSliceDescriptor(
      'plans/unclosed.plans.md',
      'any-slice',
      {
        readFile: makeReadFile(`
# Logs

\`\`\`yaml
unclosed block: true
`),
      },
    );

    expect(result).toBeNull();
  });

  it('returns null when slices is not an array', async () => {
    const result = await findArchivedSliceDescriptor(
      'plans/scalar-slices.plans.md',
      'any-slice',
      {
        readFile: makeReadFile(`
\`\`\`yaml
slices: not-an-array
\`\`\`
`),
      },
    );

    expect(result).toBeNull();
  });

  it('skips slice objects that have no slice_id', async () => {
    const result = await findArchivedSliceDescriptor(
      'plans/no-id.plans.md',
      'wanted-slice',
      {
        readFile: makeReadFile(`
\`\`\`yaml
slices:
  - title: Slice without id
\`\`\`
`),
      },
    );

    expect(result).toBeNull();
  });

  it('builds a sparse archived descriptor with fallback defaults', async () => {
    const result = await findArchivedSliceDescriptor(
      'plans/sparse.plans.md',
      'sparse-slice',
      {
        readFile: makeReadFile(`
\`\`\`yaml
slices:
  - slice_id: sparse-slice
    title: Sparse slice
\`\`\`
`),
      },
    );

    expect(result).not.toBeNull();
    expect(result.archived).toBe(true);
    expect(result.stepNumber).toBe('');
    expect(result.stepMetadata.title).toBe('');
    expect(result.stepMetadata.status).toBe('DONE');
    expect(result.stepMetadata.skills).toEqual([]);
    expect(result.stepMetadata.validation).toEqual([]);
    expect(result.boundaryNotes.phase).toBeNull();
    expect(result.boundaryNotes.goal).toBeNull();
    expect(result.boundaryNotes.files_to_change).toEqual([]);
    expect(result.boundaryNotes.slice_history).toEqual([
      { slice_id: 'sparse-slice', status: '', title: 'Sparse slice' },
    ]);
  });

  it('falls back to stepPacket-level skills and validation', async () => {
    const result = await findArchivedSliceDescriptor(
      'plans/fallback.plans.md',
      'fallback-slice',
      {
        readFile: makeReadFile(`
\`\`\`yaml
phase: 2
step: 5
step_number: 5
status: WIP
skills:
  - implementation-executor
validation:
  - npx jest --testPathPattern=direct
slices:
  - slice_id: fallback-slice
    title: Fallback slice
\`\`\`
`),
      },
    );

    expect(result).not.toBeNull();
    expect(result.boundaryNotes.phase).toBe(2);
    expect(result.stepNumber).toBe('5');
    expect(result.stepMetadata.status).toBe('WIP');
    expect(result.stepMetadata.skills).toEqual(['implementation-executor']);
    expect(result.stepMetadata.validation).toEqual([
      'npx jest --testPathPattern=direct',
    ]);
    expect(result.boundaryNotes.slice_history).toHaveLength(1);
  });
});

describe('findArchivedSliceDescriptor branch coverage', () => {
  /**
   * Build an injectable readFile that returns the supplied Markdown contents.
   *
   * @param {string} contents - Markdown/YAML fixture contents.
   * @returns {Function} Stubbed readFile.
   */
  function makeReadFile(contents) {
    return async () => contents;
  }

  it('finds a slice in a later YAML block after scanning earlier blocks', async () => {
    const result = await findArchivedSliceDescriptor(
      'plans/multi-block.plans.md',
      'wanted-slice',
      {
        readFile: makeReadFile(`
\`\`\`yaml
slices:
  - slice_id: other-slice
    title: Other slice
\`\`\`

\`\`\`yaml
phase: 3
step: 7
slices:
  - slice_id: wanted-slice
    title: Wanted slice
\`\`\`
`),
      },
    );

    expect(result).not.toBeNull();
    expect(result.archived).toBe(true);
    expect(result.slice_id).toBe('wanted-slice');
    expect(result.stepNumber).toBe('7');
    expect(result.boundaryNotes.phase).toBe(3);
    expect(result.boundaryNotes.slice_history).toHaveLength(1);
  });

  it('uses stepPacket-level fallbacks and slice-level overrides', async () => {
    const result = await findArchivedSliceDescriptor(
      'plans/fallback-full.plans.md',
      'branch-slice',
      {
        readFile: makeReadFile(`
\`\`\`yaml
phase_number: 2
step_number: 3
phase_status: '[WIP]'
phase_title: 'Phase title'
title: 'Step title'
tdd_sequence: red-green-refactor
mode: tdd
skills:
  - skill-a
validation:
  - npx test
next_step: Step 04
slices:
  - slice_id: branch-slice
    title: Branch slice
    goal: Slice goal
    estimate_hours: 4
    parallelizable: false
    dependencies:
      - dep-slice
    next_slice: next-slice
    status: '[DONE]'
  - slice_id: other-slice
\`\`\`
`),
      },
    );

    expect(result).not.toBeNull();
    expect(result.stepNumber).toBe('3');
    expect(result.boundaryNotes.phase).toBe(2);
    expect(result.stepMetadata.title).toBe('Step title');
    expect(result.stepMetadata.status).toBe('[DONE]');
    expect(result.stepMetadata.tdd_sequence).toBe('red-green-refactor');
    expect(result.stepMetadata.mode).toBe('tdd');
    expect(result.stepMetadata.skills).toEqual(['skill-a']);
    expect(result.stepMetadata.validation).toEqual(['npx test']);
    expect(result.stepMetadata.next_step).toBe('Step 04');
    expect(result.sliceTitle).toBe('Branch slice');
    expect(result.boundaryNotes.goal).toBe('Slice goal');
    expect(result.boundaryNotes.phase_status).toBe('[WIP]');
    expect(result.boundaryNotes.phase_title).toBe('Phase title');
    expect(result.boundaryNotes.estimate_hours).toBe(4);
    expect(result.boundaryNotes.parallelizable).toBe(false);
    expect(result.boundaryNotes.dependencies).toEqual(['dep-slice']);
    expect(result.boundaryNotes.next_slice).toBe('next-slice');
    expect(result.boundaryNotes.slice_history).toEqual([
      { slice_id: 'branch-slice', status: '[DONE]', title: 'Branch slice' },
      { slice_id: 'other-slice', status: '', title: '' },
    ]);
  });

  it('falls back to DONE when status is absent and uses step number aliases', async () => {
    const result = await findArchivedSliceDescriptor(
      'plans/alias.plans.md',
      'alias-slice',
      {
        readFile: makeReadFile(`
\`\`\`yaml
phase: 1
step: 9
title: 'Alias step'
slices:
  - slice_id: alias-slice
    title: Alias slice
\`\`\`
`),
      },
    );

    expect(result).not.toBeNull();
    expect(result.stepMetadata.status).toBe('DONE');
    expect(result.boundaryNotes.status).toBe('DONE');
    expect(result.boundaryNotes.phase_status).toBe('DONE');
    expect(result.stepNumber).toBe('9');
  });

  it('uses slice-level skills, validation, title fallback, and empty slice ids', async () => {
    const result = await findArchivedSliceDescriptor(
      'plans/slice-level.skills.plans.md',
      'skills-slice',
      {
        readFile: makeReadFile(`
\`\`\`yaml
phase: 1
step: 2
step_number: '2'
phase_status: '[WIP]'
phase_title: 'Phase with skills'
title: 'Step with skills'
slices:
  - slice_id: skills-slice
    skills:
      - skill-x
    validation:
      - npx validate
    status: '[WIP]'
  - title: No-id slice
\`\`\`
`),
      },
    );

    expect(result).not.toBeNull();
    expect(result.slice_id).toBe('skills-slice');
    expect(result.sliceTitle).toBe('skills-slice');
    expect(result.stepMetadata.skills).toEqual(['skill-x']);
    expect(result.stepMetadata.validation).toEqual(['npx validate']);
    expect(result.boundaryNotes.title).toBe('skills-slice');
    expect(result.boundaryNotes.status).toBe('[WIP]');
    expect(result.boundaryNotes.slice_history).toEqual([
      { slice_id: 'skills-slice', status: '[WIP]', title: '' },
      { slice_id: '', status: '', title: 'No-id slice' },
    ]);
  });
});

describe('get_slice_context defensive branches', () => {
  /**
   * Build a deterministic searchContextFn that always throws.
   *
   * @returns {Function} Injectable searchContext replacement that rejects.
   */
  function makeFailingSearchContextFn() {
    return async () => {
      throw new Error('search_context unavailable in this test');
    };
  }

  it('returns notFound when the active plan cannot be loaded', async () => {
    const planPath = `plans/__missing-${Date.now()}.plans.md`;
    const tools = createWorkflowTools({
      planPath,
      searchContextFn: makeFailingSearchContextFn(),
    });
    const tool = tools.find((t) => t.name === 'get_slice_context');

    const result = await tool.handler({
      slice_id: 'missing-slice',
      plan_path: planPath,
    });

    expect(result.notFound).toBe(true);
    expect(result.slice_id).toBe('missing-slice');
  });

  it('returns a compact response when search_context rejects', async () => {
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Search failure test [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
slices:
  - slice_id: fail-slice
    title: Search failure slice
    status: '[WIP]'
    goal: 'Exercise the search_context catch path'
\`\`\`

## Validation gates

- none
`);

    try {
      const tools = createWorkflowTools({
        planPath,
        searchContextFn: makeFailingSearchContextFn(),
      });
      const tool = tools.find((t) => t.name === 'get_slice_context');

      const result = await tool.handler({
        slice_id: 'fail-slice',
        plan_path: planPath,
      });

      expect(result.notFound).toBeFalsy();
      expect(result.slice_id).toBe('fail-slice');
      expect(result.context.text).toBe('');
      expect(result.context.chunks).toEqual([]);
      expect(result.context.query).toBeTruthy();
    } finally {
      await removeTempPlan(planPath);
    }
  });
});
