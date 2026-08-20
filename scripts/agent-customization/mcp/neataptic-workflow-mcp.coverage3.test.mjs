/**
 * @module neataptic-workflow-mcp.coverage3.test
 * @description Coverage tests for remaining branches in neataptic-workflow-mcp.mjs.
 *
 * Targets the last set of uncovered branch lines to reach 100% branch coverage:
 * - createWorkflowTools() no-arg default (line 186)
 * - Self-check non-array tools (line 348)
 * - Step label match with no slices → empty slice_history (line 742)
 * - Duplicate results with different scores (line 885)
 * - Duplicate assembled chunk IDs (line 900)
 * - follow_up_refs with search_context ref (line 909)
 * - Partial file detection + dropped chunks (lines 1027-1041)
 * - Text stitching with partial_file (lines 1133-1136)
 * - Contained chunks in deduplication (line 1365)
 * - Multi-step plan extractStepPacketText nextMatch (line 1787)
 */

import { describe, it, expect, jest } from '@jest/globals';
import { writeFile, unlink, mkdir } from 'node:fs/promises';
import path from 'node:path';
import {
  createWorkflowTools,
  runWorkflowSelfCheck,
} from './neataptic-workflow-mcp.mjs';
import { MCP_PROTOCOL_VERSION } from './mcp-utils.mjs';

const REPO_ROOT = process.cwd();

async function writeTempPlan(content) {
  const fileName = `__test-cov3-${Date.now()}-${Math.random()
    .toString(36)
    .slice(2)}.plans.md`;
  const absolutePath = path.resolve(REPO_ROOT, 'plans', fileName);
  await mkdir(path.dirname(absolutePath), { recursive: true });
  await writeFile(absolutePath, content.trim(), 'utf8');
  return path.relative(REPO_ROOT, absolutePath).replace(/\\/g, '/');
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

async function writeTempSourceFile(content) {
  const fileName = `__cov3-src-${Date.now()}-${Math.random()
    .toString(36)
    .slice(2)}.ts`;
  const absolutePath = path.resolve(
    REPO_ROOT,
    'scripts/agent-customization/mcp',
    fileName,
  );
  await writeFile(absolutePath, content, 'utf8');
  return path.relative(REPO_ROOT, absolutePath).replace(/\\/g, '/');
}

async function removeTempSourceFile(filePath) {
  try {
    await unlink(path.resolve(REPO_ROOT, filePath));
  } catch {
    // ignore
  }
}

function makeWipPlan(sliceYaml) {
  return `
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
slices:
${sliceYaml}
\`\`\`

## Validation gates

- none
`;
}

describe('neataptic-workflow-mcp coverage part 3', () => {
  describe('createWorkflowTools no-arg default', () => {
    it('covers createWorkflowTools() with no arguments (line 186)', () => {
      const tools = createWorkflowTools();
      expect(tools).toHaveLength(3);
      expect(tools.map((t) => t.name)).toEqual(
        expect.arrayContaining([
          'get_active_workflow_snapshot',
          'get_customization_inventory',
          'get_slice_context',
        ]),
      );
    });
  });

  describe('self-check non-array tools', () => {
    it('covers self-check with non-array tools/list result (line 348)', async () => {
      const planPath = await writeTempPlan(
        makeWipPlan(`  - slice_id: non-array-slice
    title: Non-array slice
    status: '[WIP]'`),
      );
      try {
        const mockServer = {
          tools: [{ name: 't1' }, { name: 't2' }, { name: 't3' }],
          dispatch: async (request) => {
            if (request.method === 'initialize') {
              return { protocolVersion: MCP_PROTOCOL_VERSION };
            }
            if (request.method === 'tools/list') {
              return { tools: null };
            }
            if (request.method === 'tools/call') {
              if (request.params?.name === 'get_active_workflow_snapshot') {
                return {
                  structuredContent: {
                    activePhase: { number: 1 },
                    activeStep: { number: 1 },
                  },
                };
              }
              if (request.params?.name === 'get_customization_inventory') {
                return {
                  structuredContent: {
                    summary: { agents: 3, skills: 5 },
                  },
                };
              }
            }
            return {};
          },
        };
        const report = await runWorkflowSelfCheck({
          server: mockServer,
          planPath,
        });
        expect(report.ok).toBe(false);
        const toolIssue = report.issues.find((i) =>
          String(i.message).includes('found none'),
        );
        expect(toolIssue).toBeDefined();
      } finally {
        await removeTempPlan(planPath);
      }
    });
  });

  describe('step label match with no slices', () => {
    it('covers empty slice_history from step label match (line 742)', async () => {
      const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step B1: Step with no slices [WIP]

\`\`\`yaml
phase: 1
step: B1
status: '[WIP]'
goal: 'Test no slices'
\`\`\`

## Validation gates

- none
`);
      try {
        const searchContextFn = async () => ({
          results: [],
          token_count: 0,
        });
        const tools = createWorkflowTools({ planPath, searchContextFn });
        const tool = tools.find((t) => t.name === 'get_slice_context');
        const result = await tool.handler({
          slice_id: 'B1',
          plan_path: planPath,
        });
        expect(result.slice_id).toBe('B1');
        expect(result.slice_history).toEqual([]);
        // Instructions should not contain SLICE HISTORY since slice_history is empty
        expect(result.instructions).not.toContain('SLICE HISTORY');
      } finally {
        await removeTempPlan(planPath);
      }
    });
  });

  describe('duplicate results and assembled chunks', () => {
    it('covers duplicate results with different scores (line 885) and duplicate assembled IDs (line 900) and follow_up_refs (line 909)', async () => {
      const planPath = await writeTempPlan(
        makeWipPlan(`  - slice_id: dup-slice
    title: Dup slice
    status: '[WIP]'
    goal: 'Test duplicates'`),
      );
      try {
        const searchContextFn = async () => ({
          results: [
            {
              chunk_id: 'dup-id',
              file_path: 'src/dup.ts',
              text: 'First result with lower score',
              char_start: 0,
              char_end: 50,
              score: 30,
            },
            {
              chunk_id: 'dup-id',
              file_path: 'src/dup.ts',
              text: 'Second result with higher score',
              char_start: 0,
              char_end: 60,
              score: 80,
            },
            {
              chunk_id: 'dup-id',
              file_path: 'src/dup.ts',
              text: 'Third result with lower score again',
              char_start: 0,
              char_end: 70,
              score: 20,
            },
          ],
          context: {
            chunks: [
              {
                chunk_id: 'asm-dup',
                file_path: 'src/asm.ts',
                content: 'Assembled chunk 1',
                char_start: 0,
                char_end: 30,
              },
              {
                chunk_id: 'asm-dup',
                file_path: 'src/asm.ts',
                content: 'Assembled chunk 2 (duplicate ID)',
                char_start: 0,
                char_end: 40,
              },
            ],
          },
          follow_up_refs: [
            { tool: 'search_context', reason: 'Follow up for more context' },
            { tool: 'load_chunk', reason: 'Not a search_context ref' },
          ],
          token_count: 100,
        });
        const tools = createWorkflowTools({ planPath, searchContextFn });
        const tool = tools.find((t) => t.name === 'get_slice_context');
        const result = await tool.handler({
          slice_id: 'dup-slice',
          plan_path: planPath,
        });
        expect(result.slice_id).toBe('dup-slice');
        // The search_context ref should appear in follow_up_refs
        const searchRef = result.context.follow_up_refs.find(
          (r) =>
            r.tool === 'search_context' &&
            r.reason === 'Follow up for more context',
        );
        expect(searchRef).toBeDefined();
      } finally {
        await removeTempPlan(planPath);
      }
    });
  });

  describe('partial file detection and dropped chunks', () => {
    it('covers partial file detection (line 1027) and dropped chunks with chunk_id (line 1041)', async () => {
      // Create a temp source file with known content (longer than char_end)
      const fileContent = 'A'.repeat(500);
      const tempFilePath = await writeTempSourceFile(fileContent);
      const planPath = await writeTempPlan(
        makeWipPlan(`  - slice_id: partial-slice
    title: Partial slice
    status: '[WIP]'
    goal: 'Test partial file detection'
    files_to_change:
      - ${tempFilePath}`),
      );
      try {
        const searchContextFn = async () => ({
          results: [
            {
              chunk_id: 'partial-chunk',
              file_path: tempFilePath,
              text: 'A'.repeat(100),
              char_start: 0,
              char_end: 100,
              score: 90,
            },
            {
              chunk_id: 'dropped-chunk',
              file_path: 'src/nonexistent-dropped.ts',
              text: '',
              char_start: 0,
              char_end: 10,
              score: 10,
              truncated: true,
            },
          ],
          token_count: 200,
        });
        const tools = createWorkflowTools({ planPath, searchContextFn });
        const tool = tools.find((t) => t.name === 'get_slice_context');
        const result = await tool.handler({
          slice_id: 'partial-slice',
          plan_path: planPath,
        });
        expect(result.slice_id).toBe('partial-slice');
        // The partial chunk should have partial_file in the context chunks
        const partialChunk = result.context.chunks.find(
          (c) => c.chunk_id === 'partial-chunk',
        );
        expect(partialChunk).toBeDefined();
        expect(partialChunk.partial_file).toBe(true);
        // Dropped chunk should have a load_chunk follow-up ref
        const loadChunkRef = result.context.follow_up_refs.find(
          (r) =>
            r.tool === 'load_chunk' && r.args?.chunk_id === 'dropped-chunk',
        );
        expect(loadChunkRef).toBeDefined();
      } finally {
        await removeTempPlan(planPath);
        await removeTempSourceFile(tempFilePath);
      }
    });
  });

  describe('text stitching with partial_file', () => {
    it('covers partial_file note in text stitching (lines 1133-1136)', async () => {
      // Create a temp source file with known content
      const fileContent = 'B'.repeat(300);
      const tempFilePath = await writeTempSourceFile(fileContent);
      const planPath = await writeTempPlan(
        makeWipPlan(`  - slice_id: stitch-slice
    title: Stitch slice
    status: '[WIP]'
    goal: 'Test stitching with partial file'
    files_to_change:
      - ${tempFilePath}`),
      );
      try {
        // No assembled context string → triggers text stitching from chunks
        const searchContextFn = async () => ({
          results: [
            {
              chunk_id: 'partial-stitch',
              file_path: tempFilePath,
              text: 'First meaningful line of the file',
              char_start: 0,
              char_end: 50,
              score: 80,
            },
          ],
          // No context field → fullContextText will be empty → stitching path
          token_count: 50,
        });
        const tools = createWorkflowTools({ planPath, searchContextFn });
        const tool = tools.find((t) => t.name === 'get_slice_context');
        const result = await tool.handler({
          slice_id: 'stitch-slice',
          plan_path: planPath,
        });
        expect(result.slice_id).toBe('stitch-slice');
        // The context text should contain the PARTIAL note
        expect(result.context.text).toContain('[PARTIAL:');
        expect(result.context.text).toContain('First meaningful line');
      } finally {
        await removeTempPlan(planPath);
        await removeTempSourceFile(tempFilePath);
      }
    });
  });

  describe('contained chunks in deduplication', () => {
    it('covers contained chunk dropping (line 1365 falsy branch)', async () => {
      const planPath = await writeTempPlan(
        makeWipPlan(`  - slice_id: contained-slice
    title: Contained slice
    status: '[WIP]'
    goal: 'Test contained chunks'`),
      );
      try {
        const searchContextFn = async () => ({
          results: [
            {
              chunk_id: 'broad-chunk',
              file_path: 'src/contained.ts',
              text: 'Broad chunk covering a large range of the file',
              char_start: 0,
              char_end: 200,
              score: 70,
            },
            {
              chunk_id: 'narrow-chunk',
              file_path: 'src/contained.ts',
              text: 'Narrow chunk within the broad chunk range',
              char_start: 10,
              char_end: 50,
              score: 90,
            },
          ],
          token_count: 100,
        });
        const tools = createWorkflowTools({ planPath, searchContextFn });
        const tool = tools.find((t) => t.name === 'get_slice_context');
        const result = await tool.handler({
          slice_id: 'contained-slice',
          plan_path: planPath,
        });
        expect(result.slice_id).toBe('contained-slice');
        // The narrow chunk (contained within broad) should be dropped
        const chunkIds = result.context.chunks.map((c) => c.chunk_id);
        expect(chunkIds).toContain('broad-chunk');
        expect(chunkIds).not.toContain('narrow-chunk');
      } finally {
        await removeTempPlan(planPath);
      }
    });
  });

  describe('follow_up_refs with no matching ref', () => {
    it('covers if(ref) falsy branch (line 910)', async () => {
      const planPath = await writeTempPlan(
        makeWipPlan(`  - slice_id: no-ref-slice
    title: No ref slice
    status: '[WIP]'
    goal: 'Test no matching ref'`),
      );
      try {
        const searchContextFn = async () => ({
          results: [
            {
              chunk_id: 'no-ref-chunk',
              file_path: 'src/no-ref.ts',
              text: 'Some text',
              char_start: 0,
              char_end: 50,
              score: 50,
            },
          ],
          follow_up_refs: [
            { tool: 'load_chunk', reason: 'Not a search_context ref' },
            { tool: 'other', reason: 'Also not matching' },
          ],
          token_count: 50,
        });
        const tools = createWorkflowTools({ planPath, searchContextFn });
        const tool = tools.find((t) => t.name === 'get_slice_context');
        const result = await tool.handler({
          slice_id: 'no-ref-slice',
          plan_path: planPath,
        });
        expect(result.slice_id).toBe('no-ref-slice');
        // No search_context ref should be in follow_up_refs
        const searchRef = result.context.follow_up_refs.find(
          (r) => r.tool === 'search_context',
        );
        expect(searchRef).toBeUndefined();
      } finally {
        await removeTempPlan(planPath);
      }
    });
  });

  describe('partial file detection — full file', () => {
    it('covers char_end >= fileChars falsy branch (line 1028)', async () => {
      const fileContent = 'C'.repeat(50);
      const tempFilePath = await writeTempSourceFile(fileContent);
      const planPath = await writeTempPlan(
        makeWipPlan(`  - slice_id: full-file-slice
    title: Full file slice
    status: '[WIP]'
    goal: 'Test full file detection'
    files_to_change:
      - ${tempFilePath}`),
      );
      try {
        const searchContextFn = async () => ({
          results: [
            {
              chunk_id: 'full-chunk',
              file_path: tempFilePath,
              text: 'C'.repeat(50),
              char_start: 0,
              char_end: 50,
              score: 90,
            },
          ],
          token_count: 50,
        });
        const tools = createWorkflowTools({ planPath, searchContextFn });
        const tool = tools.find((t) => t.name === 'get_slice_context');
        const result = await tool.handler({
          slice_id: 'full-file-slice',
          plan_path: planPath,
        });
        expect(result.slice_id).toBe('full-file-slice');
        const chunk = result.context.chunks.find(
          (c) => c.chunk_id === 'full-chunk',
        );
        expect(chunk).toBeDefined();
        expect(chunk.partial_file).toBeUndefined();
      } finally {
        await removeTempPlan(planPath);
        await removeTempSourceFile(tempFilePath);
      }
    });
  });

  describe('dropped chunks without chunk_id', () => {
    it('covers chunk_id null falsy branch (line 1042)', async () => {
      const planPath = await writeTempPlan(
        makeWipPlan(`  - slice_id: no-id-slice
    title: No ID slice
    status: '[WIP]'
    goal: 'Test dropped chunk without chunk_id'`),
      );
      try {
        const searchContextFn = async () => ({
          results: [
            {
              // No chunk_id → will be null after mapResultToChunk
              file_path: 'src/no-id.ts',
              text: '',
              char_start: 0,
              char_end: 10,
              score: 10,
              truncated: true,
            },
            {
              chunk_id: 'survivor-chunk',
              file_path: 'src/survivor.ts',
              text: 'Survivor text to keep',
              char_start: 0,
              char_end: 30,
              score: 80,
            },
          ],
          token_count: 50,
        });
        const tools = createWorkflowTools({ planPath, searchContextFn });
        const tool = tools.find((t) => t.name === 'get_slice_context');
        const result = await tool.handler({
          slice_id: 'no-id-slice',
          plan_path: planPath,
        });
        expect(result.slice_id).toBe('no-id-slice');
        // The dropped chunk has no chunk_id, so no load_chunk ref for it
        const loadChunkRefs = result.context.follow_up_refs.filter(
          (r) => r.tool === 'load_chunk',
        );
        // No load_chunk refs because the only dropped chunk has no chunk_id
        expect(loadChunkRefs).toHaveLength(0);
      } finally {
        await removeTempPlan(planPath);
      }
    });
  });

  describe('multi-step plan extractStepPacketText', () => {
    it('covers nextMatch truthy branch in extractStepPacketText (line 1787)', async () => {
      const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: First step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
slices:
  - slice_id: first-step-slice
    title: First step slice
    status: '[WIP]'
    goal: 'Test first step'
\`\`\`

#### Step 02: Second step [DONE]

\`\`\`yaml
phase: 1
step: 2
status: '[DONE]'
\`\`\`

## Validation gates

- none
`);
      try {
        const searchContextFn = async () => ({
          results: [],
          token_count: 0,
        });
        const tools = createWorkflowTools({ planPath, searchContextFn });
        const tool = tools.find((t) => t.name === 'get_slice_context');
        // Request "01" as slice_id — this goes through the fallback loop
        // and extractStepPacketText finds step "01" which is NOT the last step,
        // so nextMatch is truthy.
        const result = await tool.handler({
          slice_id: '01',
          plan_path: planPath,
        });
        // The step label "01" matches the step, so buildDescriptorFromStep is called
        // Goal comes from step-level metadata, not slice-level
        expect(result.slice_id).toBe('01');
        expect(result.step_number).toBe(1);
      } finally {
        await removeTempPlan(planPath);
      }
    });
  });
});
