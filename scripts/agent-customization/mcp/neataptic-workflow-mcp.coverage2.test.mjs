/**
 * @module neataptic-workflow-mcp.coverage2.test
 * @description Additional coverage tests for neataptic-workflow-mcp.mjs.
 *
 * Covers tool handlers, self-check, main/bootstrapMain, utility function
 * branches, and remaining edge cases not exercised by the primary suite.
 */

import { describe, it, expect, jest } from '@jest/globals';
import { writeFile, unlink, mkdir } from 'node:fs/promises';
import path from 'node:path';
import {
  createWorkflowTools,
  runWorkflowSelfCheck,
  main,
  bootstrapMain,
} from './neataptic-workflow-mcp.mjs';
import { createMcpServer, MCP_PROTOCOL_VERSION } from './mcp-utils.mjs';

const REPO_ROOT = process.cwd();

async function writeTempPlan(content) {
  const fileName = `__test-cov2-${Date.now()}-${Math.random()
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

describe('neataptic-workflow-mcp coverage part 2', () => {
  describe('createWorkflowTools defaults and tool handlers', () => {
    it('covers default searchContextFn parameter (line 145)', () => {
      const tools = createWorkflowTools({ planPath: 'plans/test.md' });
      expect(tools).toHaveLength(3);
      expect(tools.map((t) => t.name)).toContain('get_slice_context');
    });

    it('covers get_active_workflow_snapshot success path', async () => {
      const planPath = await writeTempPlan(
        makeWipPlan(`  - slice_id: test-slice
    title: Test slice
    status: '[WIP]'`),
      );
      try {
        const tools = createWorkflowTools({ planPath });
        const snapshotTool = tools.find(
          (t) => t.name === 'get_active_workflow_snapshot',
        );
        const result = await snapshotTool.handler({ plan_path: planPath });
        expect(result.scope).toBe('repo-static');
        expect(result.activePhase).not.toBeNull();
        expect(result.activeStep).not.toBeNull();
      } finally {
        await removeTempPlan(planPath);
      }
    });

    it('covers get_active_workflow_snapshot no-WIP graceful degradation', async () => {
      const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [DONE]

#### Step 01: Done step [DONE]

\`\`\`yaml
phase: 1
step: 1
status: '[DONE]'
\`\`\`

## Validation gates

- none
`);
      try {
        const tools = createWorkflowTools({ planPath });
        const snapshotTool = tools.find(
          (t) => t.name === 'get_active_workflow_snapshot',
        );
        const result = await snapshotTool.handler({ plan_path: planPath });
        expect(result.scope).toBe('no-active-phase');
        expect(result.activePhase).toBeNull();
        expect(result.fallbackAdvice).toBeDefined();
      } finally {
        await removeTempPlan(planPath);
      }
    });

    it('covers get_active_workflow_snapshot non-WIP error rethrow', async () => {
      const tools = createWorkflowTools({
        planPath: 'plans/__nonexistent-cov2.plans.md',
      });
      const snapshotTool = tools.find(
        (t) => t.name === 'get_active_workflow_snapshot',
      );
      await expect(
        snapshotTool.handler({
          plan_path: 'plans/__nonexistent-cov2.plans.md',
        }),
      ).rejects.toThrow();
    });

    it('covers get_customization_inventory with inventoryCommandRunner (line 149, 211)', async () => {
      const mockRunner = jest.fn().mockResolvedValue({
        exitCode: 0,
        stdout: '{"summary":{"agents":5,"skills":10},"agents":[],"skills":[]}',
      });
      const tools = createWorkflowTools({
        planPath: 'plans/test.md',
        inventoryCommandRunner: mockRunner,
      });
      const inventoryTool = tools.find(
        (t) => t.name === 'get_customization_inventory',
      );
      const result = await inventoryTool.handler();
      expect(result.summary.agents).toBe(5);
      expect(mockRunner).toHaveBeenCalled();
    });

    it('covers loadCustomizationInventory exit code error', async () => {
      const mockRunner = jest
        .fn()
        .mockResolvedValue({ exitCode: 1, stdout: '' });
      const tools = createWorkflowTools({
        planPath: 'plans/test.md',
        inventoryCommandRunner: mockRunner,
      });
      const inventoryTool = tools.find(
        (t) => t.name === 'get_customization_inventory',
      );
      await expect(inventoryTool.handler()).rejects.toThrow(/exit code 1/);
    });

    it('covers loadCustomizationInventory invalid JSON error', async () => {
      const mockRunner = jest
        .fn()
        .mockResolvedValue({ exitCode: 0, stdout: 'not valid json' });
      const tools = createWorkflowTools({
        planPath: 'plans/test.md',
        inventoryCommandRunner: mockRunner,
      });
      const inventoryTool = tools.find(
        (t) => t.name === 'get_customization_inventory',
      );
      await expect(inventoryTool.handler()).rejects.toThrow(/valid JSON/);
    });
  });

  describe('runWorkflowSelfCheck', () => {
    it('covers full self-check with valid WIP plan and mock inventory', async () => {
      const planPath = await writeTempPlan(
        makeWipPlan(`  - slice_id: self-check-slice
    title: Self-check slice
    status: '[WIP]'
    goal: 'Test self-check'`),
      );
      try {
        const mockRunner = jest.fn().mockResolvedValue({
          exitCode: 0,
          stdout: '{"summary":{"agents":5,"skills":10}}',
        });
        const tools = createWorkflowTools({
          planPath,
          inventoryCommandRunner: mockRunner,
        });
        const server = createMcpServer({
          serverName: 'neataptic_workflow_mcp',
          serverVersion: '0.1.0',
          tools,
        });
        const report = await runWorkflowSelfCheck({
          server,
          planPath,
        });
        expect(report.name).toContain('self-check');
        expect(report.ok).toBe(true);
      } finally {
        await removeTempPlan(planPath);
      }
    });

    it('covers all self-check error branches with mock server', async () => {
      const planPath = await writeTempPlan(
        makeWipPlan(`  - slice_id: err-slice
    title: Error slice
    status: '[WIP]'
    goal: 'Test errors'`),
      );
      try {
        const mockServer = {
          tools: [{ name: 'tool1' }, { name: 'tool2' }],
          dispatch: async (request) => {
            if (request.method === 'initialize') {
              return { protocolVersion: 'wrong-version' };
            }
            if (request.method === 'tools/list') {
              return { tools: [{ name: 'tool1' }, { name: 'tool2' }] };
            }
            if (request.method === 'tools/call') {
              if (request.params?.name === 'get_active_workflow_snapshot') {
                return {
                  isError: true,
                  structuredContent: {
                    activePhase: { number: 999 },
                    activeStep: { number: 999 },
                    selectedActiveAgent: 'bad',
                  },
                };
              }
              if (request.params?.name === 'get_customization_inventory') {
                return { isError: true, structuredContent: {} };
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
        expect(report.issues.length).toBeGreaterThanOrEqual(7);
      } finally {
        await removeTempPlan(planPath);
      }
    });

    it('covers self-check no-WIP-step error branch', async () => {
      const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

\`\`\`yaml
expansion: steps
auto_expand: false
\`\`\`

#### Step 01: Done step [DONE]

\`\`\`yaml
phase: 1
step: 1
status: '[DONE]'
\`\`\`

## Validation gates

- none
`);
      try {
        const mockServer = {
          tools: [{ name: 'tool1' }, { name: 'tool2' }, { name: 'tool3' }],
          dispatch: async (request) => {
            if (request.method === 'initialize') {
              return { protocolVersion: MCP_PROTOCOL_VERSION };
            }
            if (request.method === 'tools/list') {
              return {
                tools: [
                  { name: 'tool1' },
                  { name: 'tool2' },
                  { name: 'tool3' },
                ],
              };
            }
            if (request.method === 'tools/call') {
              if (request.params?.name === 'get_active_workflow_snapshot') {
                return { structuredContent: {} };
              }
              if (request.params?.name === 'get_customization_inventory') {
                return {
                  structuredContent: {
                    summary: { agents: 5, skills: 10 },
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
      } finally {
        await removeTempPlan(planPath);
      }
    });
  });

  describe('main and bootstrapMain', () => {
    it('covers main --help path', async () => {
      const exitSpy = jest
        .spyOn(process, 'exit')
        .mockImplementation(() => undefined);
      const logSpy = jest
        .spyOn(console, 'log')
        .mockImplementation(() => undefined);
      try {
        await main(['--help']);
        expect(exitSpy.mock.calls[0]?.[0]).toBe(0);
        expect(logSpy.mock.calls.length).toBeGreaterThan(0);
      } finally {
        exitSpy.mockRestore();
        logSpy.mockRestore();
      }
    });

    it('covers main --self-check path with injected runSelfCheck', async () => {
      const runSelfCheckMock = jest
        .fn()
        .mockResolvedValue({ ok: true, name: 'test' });
      const writeSpy = jest
        .spyOn(process.stdout, 'write')
        .mockImplementation(() => true);
      const originalExitCode = process.exitCode;
      try {
        await main(['--self-check', '--plan=plans/test.plans.md'], {
          runSelfCheck: runSelfCheckMock,
        });
        expect(runSelfCheckMock.mock.calls.length).toBe(1);
        expect(process.exitCode).toBe(0);
      } finally {
        writeSpy.mockRestore();
        process.exitCode = originalExitCode;
      }
    });

    it('covers main --self-check with failing self-check (exit code 1)', async () => {
      const runSelfCheckMock = jest
        .fn()
        .mockResolvedValue({ ok: false, name: 'test' });
      const writeSpy = jest
        .spyOn(process.stdout, 'write')
        .mockImplementation(() => true);
      const originalExitCode = process.exitCode;
      try {
        await main(['--self-check', '--plan=plans/test.plans.md'], {
          runSelfCheck: runSelfCheckMock,
        });
        expect(process.exitCode).toBe(1);
      } finally {
        writeSpy.mockRestore();
        process.exitCode = originalExitCode;
      }
    });

    it('covers main stdio path with injected runStdio', async () => {
      const runStdioMock = jest.fn().mockResolvedValue(undefined);
      await main(['--plan=plans/test.plans.md'], {
        runStdio: runStdioMock,
      });
      expect(runStdioMock.mock.calls.length).toBe(1);
    });

    it('covers bootstrapMain error catch path', async () => {
      const errorSpy = jest
        .spyOn(console, 'error')
        .mockImplementation(() => undefined);
      const originalExitCode = process.exitCode;
      try {
        process.exitCode = undefined;
        const promise = bootstrapMain();
        await promise;
        expect(process.exitCode).toBe(1);
        expect(errorSpy.mock.calls.length).toBeGreaterThan(0);
      } finally {
        errorSpy.mockRestore();
        process.exitCode = originalExitCode ?? 0;
      }
    });

    it('covers bootstrapMain success path (no error)', async () => {
      const errorSpy = jest
        .spyOn(console, 'error')
        .mockImplementation(() => undefined);
      const originalExitCode = process.exitCode;
      const originalArgv = process.argv;
      try {
        process.exitCode = undefined;
        process.argv = ['node', 'script.mjs', '--plan=plans/test.plans.md'];
        const runStdioMock = jest.fn().mockResolvedValue(undefined);
        const promise = bootstrapMain({ runStdio: runStdioMock });
        await promise;
        expect(process.exitCode).toBeUndefined();
        expect(errorSpy.mock.calls.length).toBe(0);
      } finally {
        errorSpy.mockRestore();
        process.exitCode = originalExitCode ?? 0;
        process.argv = originalArgv;
      }
    });
  });

  describe('fetchSliceContext edge cases', () => {
    it('covers assembled.context string property (line 1081)', async () => {
      const planPath = await writeTempPlan(
        makeWipPlan(`  - slice_id: assembled-slice
    title: Assembled slice
    status: '[WIP]'
    goal: 'Test assembled context'`),
      );
      try {
        const searchContextFn = async () => ({
          context: { context: 'Assembled context text from cortex' },
          results: [],
          token_count: 50,
        });
        const tools = createWorkflowTools({ planPath, searchContextFn });
        const tool = tools.find((t) => t.name === 'get_slice_context');
        const result = await tool.handler({
          slice_id: 'assembled-slice',
          plan_path: planPath,
        });
        expect(result.context.text).toBe('Assembled context text from cortex');
      } finally {
        await removeTempPlan(planPath);
      }
    });

    it('covers fallback context stitching from chunks (lines 1096-1102)', async () => {
      const planPath = await writeTempPlan(
        makeWipPlan(`  - slice_id: fallback-slice
    title: Fallback slice
    status: '[WIP]'
    goal: 'Test fallback stitching'
    files_to_change:
      - src/feature.ts`),
      );
      try {
        const searchContextFn = async () => ({
          results: [
            {
              chunk_id: 'r1',
              file_path: 'src/feature.ts',
              text: 'First line of code\nSecond line',
              char_start: 0,
              char_end: 100,
              score: 50,
            },
            {
              chunk_id: 'r2',
              file_path: 'src/partial.ts',
              text: 'Partial content here',
              char_start: 0,
              char_end: 50,
              score: 40,
            },
          ],
          token_count: 100,
        });
        const tools = createWorkflowTools({ planPath, searchContextFn });
        const tool = tools.find((t) => t.name === 'get_slice_context');
        const result = await tool.handler({
          slice_id: 'fallback-slice',
          plan_path: planPath,
        });
        expect(result.context.text).toContain('feature.ts');
        expect(result.context.text).toContain('First line of code');
      } finally {
        await removeTempPlan(planPath);
      }
    });

    it('covers extractQueryTokens with empty text (line 1221)', async () => {
      const planPath = await writeTempPlan(
        makeWipPlan(`  - slice_id: empty-ac-slice
    title: Empty AC slice
    status: '[WIP]'
    goal: 'Test empty AC'
    acceptance_criteria:
      - id: AC-EMPTY`),
      );
      try {
        const searchContextFn = async () => ({
          results: [],
          token_count: 0,
        });
        const tools = createWorkflowTools({ planPath, searchContextFn });
        const tool = tools.find((t) => t.name === 'get_slice_context');
        const result = await tool.handler({
          slice_id: 'empty-ac-slice',
          plan_path: planPath,
        });
        expect(result.slice_id).toBe('empty-ac-slice');
      } finally {
        await removeTempPlan(planPath);
      }
    });
  });

  describe('truncation edge cases', () => {
    it('covers instructions truncation success (line 615)', async () => {
      const slicesYaml = [];
      for (let i = 0; i < 170; i++) {
        slicesYaml.push('  - slice_id: s-' + i);
        slicesYaml.push('    title: Slice ' + i);
        slicesYaml.push("    status: '[DONE]'");
      }
      const acLines = [];
      for (let i = 0; i < 30; i++) {
        const id = 'AC-' + String(i + 1).padStart(3, '0');
        acLines.push('      - id: ' + id);
        acLines.push(
          "        text: 'Acceptance criterion " +
            (i + 1) +
            " for testing truncation behavior with a long description'",
        );
        acLines.push(
          "        validation: 'npm test -- criterion-" + (i + 1) + "'",
        );
      }
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
${slicesYaml.join('\n')}
  - slice_id: instr-trunc-slice
    title: Instructions truncation slice
    status: '[WIP]'
    goal: 'Test instructions truncation success path'
    acceptance_criteria:
${acLines.join('\n')}
\`\`\`

## Validation gates

- none
`);
      try {
        const largeText = 'A'.repeat(5000);
        const searchContextFn = async () => ({
          context: largeText,
          results: [],
          token_count: 100,
        });
        const tools = createWorkflowTools({ planPath, searchContextFn });
        const tool = tools.find((t) => t.name === 'get_slice_context');
        const result = await tool.handler({
          slice_id: 'instr-trunc-slice',
          plan_path: planPath,
        });
        expect(result.truncated).toBe(true);
        expect(result.fallback_message).toContain(
          'context.text and instructions were truncated',
        );
      } finally {
        await removeTempPlan(planPath);
      }
    });

    it('covers truncateStringToBytes truncation path (lines 1510-1517)', async () => {
      const slicesYaml = [];
      for (let i = 0; i < 400; i++) {
        slicesYaml.push('  - slice_id: s-' + i);
        slicesYaml.push('    title: Slice ' + i);
        slicesYaml.push("    status: '[DONE]'");
      }
      const longGoal = 'G'.repeat(300);
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
${slicesYaml.join('\n')}
  - slice_id: trunc-bytes-slice
    title: Trunc bytes slice
    status: '[WIP]'
    goal: '${longGoal}'
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
          slice_id: 'trunc-bytes-slice',
          plan_path: planPath,
        });
        expect(result.compact).toBe(true);
        expect(result.truncated).toBe(true);
        expect(result.fallback_message).toContain('could not be trimmed');
        expect(Buffer.byteLength(result.goal, 'utf8')).toBeLessThanOrEqual(256);
      } finally {
        await removeTempPlan(planPath);
      }
    });
  });

  describe('findSliceDescriptor edge cases', () => {
    it('covers normalizeStepNumber with number input (line 1572)', async () => {
      const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: my-slice — Title [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
slices:
  - slice_id: my-slice
    title: My slice
    status: '[WIP]'
    goal: 'Test normalizeStepNumber with number'
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
          slice_id: 'my-slice',
          plan_path: planPath,
        });
        expect(result.slice_id).toBe('my-slice');
        expect(result.goal).toBe('Test normalizeStepNumber with number');
      } finally {
        await removeTempPlan(planPath);
      }
    });

    it('covers findSliceDescriptor loop finding non-active step (lines 1662-1664, 1685)', async () => {
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

#### Step 02: Other step [DONE]

\`\`\`yaml
phase: 1
step: 2
status: '[DONE]'
slices:
  - slice_id: 02-slice
    title: Other slice
    status: '[DONE]'
    goal: 'Found via loop'
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
          slice_id: '02-slice',
          plan_path: planPath,
        });
        expect(result.slice_id).toBe('02-slice');
        expect(result.goal).toBe('Found via loop');
      } finally {
        await removeTempPlan(planPath);
      }
    });

    it('covers findSliceDescriptor return null (line 1707)', async () => {
      const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: my-slice — Title [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
slices:
  - slice_id: different-slice
    title: Different slice
    status: '[WIP]'
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
          slice_id: 'my-slice',
          plan_path: planPath,
        });
        expect(result.notFound).toBe(true);
      } finally {
        await removeTempPlan(planPath);
      }
    });

    it('covers parseStepPacketMetadata no YAML block (line 1753)', async () => {
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

#### Step A1: NoYaml step [DONE]

This step has no YAML metadata block at all.

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
          slice_id: 'missing-slice',
          plan_path: planPath,
        });
        expect(result.notFound).toBe(true);
      } finally {
        await removeTempPlan(planPath);
      }
    });
  });
});
