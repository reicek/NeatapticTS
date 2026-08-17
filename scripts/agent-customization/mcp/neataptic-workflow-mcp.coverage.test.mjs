import { describe, it, expect } from '@jest/globals';
import { writeFile, unlink, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { createWorkflowTools } from './neataptic-workflow-mcp.mjs';

const REPO_ROOT = process.cwd();

async function writeTempPlan(content) {
  const fileName = `__test-cov-${Date.now()}-${Math.random().toString(36).slice(2)}.plans.md`;
  const absolutePath = path.resolve(REPO_ROOT, 'plans', fileName);
  await mkdir(path.dirname(absolutePath), { recursive: true });
  await writeFile(absolutePath, content.trim(), 'utf8');
  return path.relative(REPO_ROOT, absolutePath).replace(/\\/g, '/');
}

async function writeTempLogs(planPath, content) {
  const logsPath = planPath.replace(/\.plans\.md$/, '.logs.md');
  const absolutePath = path.resolve(REPO_ROOT, logsPath);
  await writeFile(absolutePath, content.trim(), 'utf8');
  return logsPath;
}

async function removeTempPlan(planPath) {
  for (const suffix of ['.plans.md', '.logs.md']) {
    const filePath = planPath.replace(/\.plans\.md$/, suffix);
    try {
      await unlink(path.resolve(REPO_ROOT, filePath));
    } catch {
      // ignore
    }
  }
}

async function callGetSliceContext(planPath, searchContextFn, sliceId) {
  const tools = createWorkflowTools({ planPath, searchContextFn });
  const tool = tools.find((t) => t.name === 'get_slice_context');
  return tool.handler({ slice_id: sliceId, plan_path: planPath });
}

function generateSlicesYaml(count) {
  const lines = [];
  for (let i = 0; i < count; i++) {
    lines.push(`  - slice_id: s-${i}`);
    lines.push(`    title: Slice ${i}`);
    lines.push(`    status: '[DONE]'`);
  }
  return lines.join('\n');
}

describe('neataptic-workflow-mcp coverage', () => {
  it('covers mapResultToChunk fallbacks, boost, follow-up refs, and partial file detection', async () => {
    const planContent = `
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
title: 'Active step'
tdd_sequence: 'red-green-refactor'
mode: 'strict'
skills:
  - skill-one
  - skill-two
validation:
  - 'npm test'
  - 'npm run lint'
next_step: 'Step 02'
slices:
  - slice_id: full-slice
    title: Full feature slice
    status: '[WIP]'
    goal: 'Implement full feature with comprehensive tests'
    estimate_hours: 4
    parallelizable: true
    next_slice: 'other-slice'
    files_to_change:
      - src/feature.ts
      - src/missing.ts
      - tests/feature.test.ts
      - tests/dropped.test.ts
      - tests/other.test.ts
    dependencies:
      - dep-slice-1
    acceptance_criteria:
      - id: AC-001
        text: 'Feature works correctly'
        validation: 'npm test -- feature'
      - text: 'No id provided here'
  - slice_id: other-slice
    title: Other slice
    status: '[DONE]'
\`\`\`

## Validation gates

- none
`;
    const planPath = await writeTempPlan(planContent);
    try {
      let callIndex = 0;
      const searchContextFn = async (options) => {
        callIndex++;
        if (options.budget === 500) {
          return {
            context: 'Test context from test query response',
            results: [
              { chunk_id: 'test-1', file_path: 'tests/feature.test.ts', text: 'test content', char_start: 400, char_end: 500, score: 10 },
            ],
            token_count: 100,
          };
        }
        if (callIndex === 3) {
          return null;
        }
        if (callIndex === 1) {
          return {
            results: [
              { chunk_id: 1, file_path: 'src/feature.ts', text: 'feature code', heading_path: 'My Heading', char_start: 0, char_end: 100, score: 100 },
              { metadata: { file_path: 'tests/feature.test.ts', char_start: 200, char_end: 300, heading_path: 'Section A', context_header: 'Ctx' }, content: 'test code', score: 60 },
              { file_path: 'scripts/agent-customization/mcp/slice-context-archive.mjs', snippet: 'archive code', char_start: 0, char_end: 50, score: 40 },
              { chunk_id: 2, truncated: true, file_path: 'tests/dropped.test.ts', text: 'dropped test', score: 90 },
              { chunk_id: 3, path: 'tests/feature.test.ts', text: 'path test', char_start: 0, char_end: 199, score: 60 },
              { file_path: 12345, text: 'non-string path', score: 5 },
              { chunk_id: 'fallback-1', score: 20 },
              { chunk_id: 'text-chunk', score: 15 },
              { score: 10, char_start: 999 },
              { metadata: { path: 'tests/other.test.ts' }, text: 'metadata path test', score: 60 },
              { chunk_id: 'no-path', truncated: true, text: 'no path truncated', score: 30 },
            ],
            context: {
              chunks: [
                { chunk_id: 1, file_path: 'src/feature.ts', content: 'full feature code', char_start: 0, char_end: 100 },
                { chunk_id: 'fallback-1', file_path: 'src/fallback.ts', content: 'fallback content', heading_path: 'Assembled Heading', char_start: 0, char_end: 100 },
                { chunk_id: 'text-chunk', file_path: 'src/text-chunk.ts', text: 'text chunk content', context_header: 'Assembled Context', char_start: 0 },
                { file_path: 'src/no-id.ts', content: 'no id content', char_start: 0 },
              ],
            },
            follow_up_refs: [{ tool: 'search_context', reason: 'More context needed' }],
            token_count: 500,
          };
        }
        return { results: [], token_count: 50 };
      };

      const result = await callGetSliceContext(planPath, searchContextFn, 'full-slice');

      // buildCompactSliceResponse always sets compact: true in basePayload
      expect(result.slice_id).toBe('full-slice');
      expect(result.compact).toBe(true);
      expect(result.instructions).toContain('VALIDATION:');
      expect(result.instructions).toContain('SKILLS:');
      expect(result.instructions).toContain('NEXT STEP:');
      expect(result.context.text).toBe('Test context from test query response');
      const refTools = result.context.follow_up_refs.map((r) => r.tool);
      expect(refTools).toContain('load_chunk');
      expect(refTools).toContain('search_context');
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('returns empty RAG context when searchContextFn is null', async () => {
    const planContent = `
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
slices:
  - slice_id: minimal-slice
\`\`\`

## Validation gates

- none
`;
    const planPath = await writeTempPlan(planContent);
    try {
      const result = await callGetSliceContext(planPath, null, 'minimal-slice');
      expect(result.slice_id).toBe('minimal-slice');
      expect(result.context.chunks).toEqual([]);
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('sets testResponse to null when test search throws', async () => {
    const planContent = `
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
tdd_sequence: 'red-green-refactor'
slices:
  - slice_id: tdd-slice
    title: TDD slice
    status: '[WIP]'
    goal: 'Test TDD slice'
    files_to_change:
      - src/feature.ts
      - tests/feature.test.ts
\`\`\`

## Validation gates

- none
`;
    const planPath = await writeTempPlan(planContent);
    try {
      const searchContextFn = async (options) => {
        if (options.budget === 500) {
          throw new Error('Test search failed');
        }
        return { results: [], token_count: 0 };
      };

      const result = await callGetSliceContext(planPath, searchContextFn, 'tdd-slice');
      expect(result.slice_id).toBe('tdd-slice');
      expect(result.context.text).toBe('');
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('truncates context text when payload exceeds budget', async () => {
    const planContent = `
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
slices:
  - slice_id: large-context-slice
    title: Large context slice
    status: '[WIP]'
    goal: 'Test large context truncation'
\`\`\`

## Validation gates

- none
`;
    const planPath = await writeTempPlan(planContent);
    try {
      const largeText = 'A'.repeat(20000);
      const searchContextFn = async () => ({
        context: largeText,
        results: [],
        token_count: 0,
      });

      const result = await callGetSliceContext(planPath, searchContextFn, 'large-context-slice');
      expect(result.slice_id).toBe('large-context-slice');
      expect(result.context.text.length).toBeLessThan(20000);
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('truncates instructions when fixed_part is large', async () => {
    const slicesYaml = generateSlicesYaml(250);
    const acLines = [];
    for (let i = 0; i < 30; i++) {
      const id = `AC-${String(i + 1).padStart(3, '0')}`;
      acLines.push(`      - id: ${id}`);
      acLines.push(`        text: 'Acceptance criterion ${i + 1} for testing truncation behavior with a long description'`);
      acLines.push(`        validation: 'npm test -- criterion-${i + 1}'`);
    }
    const planContent = `
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
slices:
${slicesYaml}
  - slice_id: target-slice
    title: Target slice
    status: '[WIP]'
    goal: 'Test instructions truncation'
    acceptance_criteria:
${acLines.join('\n')}
\`\`\`

## Validation gates

- none
`;
    const planPath = await writeTempPlan(planContent);
    try {
      const searchContextFn = async () => ({ results: [], token_count: 0 });
      const result = await callGetSliceContext(planPath, searchContextFn, 'target-slice');
      expect(result.slice_id).toBe('target-slice');
      expect(result.truncated).toBe(true);
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('uses compact fallback when payload cannot be truncated enough', async () => {
    const slicesYaml = generateSlicesYaml(400);
    const planContent = `
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
slices:
${slicesYaml}
  - slice_id: compact-slice
    title: Compact slice
    status: '[WIP]'
    goal: 'Compact fallback test goal'
\`\`\`

## Validation gates

- none
`;
    const planPath = await writeTempPlan(planContent);
    try {
      const searchContextFn = async () => ({ results: [], token_count: 0 });
      const result = await callGetSliceContext(planPath, searchContextFn, 'compact-slice');
      expect(result.slice_id).toBe('compact-slice');
      expect(result.compact).toBe(true);
      expect(result.truncated).toBe(true);
      expect(result.fallback_message).toContain('could not be trimmed');
      expect(result.goal).toBe('Compact fallback test goal');
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('builds descriptor from step when slice_id matches step label', async () => {
    const planContent = `
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step B1: Symbolic step [WIP]

\`\`\`yaml
phase: 1
step: B1
status: '[WIP]'
slices:
  - slice_id: some-slice
    title: Some slice
    status: '[WIP]'
  - slice_id: other-slice
    title: Other slice
    status: '[DONE]'
\`\`\`

## Validation gates

- none
`;
    const planPath = await writeTempPlan(planContent);
    try {
      const searchContextFn = async () => ({ results: [], token_count: 0 });
      const result = await callGetSliceContext(planPath, searchContextFn, 'B1');
      expect(result.slice_id).toBe('B1');
      expect(result.compact).toBe(true);
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('covers goal null branch via archived descriptor', async () => {
    const planContent = `
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

## Validation gates

- none
`;
    const logsContent = `
# Logs

\`\`\`yaml
step: 2
status: '[DONE]'
slices:
  - slice_id: archived-slice
    title: Archived slice
    status: '[DONE]'
\`\`\`
`;
    const planPath = await writeTempPlan(planContent);
    await writeTempLogs(planPath, logsContent);
    try {
      const searchContextFn = async () => ({ results: [], token_count: 0 });
      const result = await callGetSliceContext(planPath, searchContextFn, 'archived-slice');
      expect(result.slice_id).toBe('archived-slice');
      expect(result.goal).toBeNull();
      expect(result.instructions).toContain('GOAL: unspecified');
    } finally {
      await removeTempPlan(planPath);
    }
  });
});