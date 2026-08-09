/**
 * @module neataptic-workflow-mcp.test
 * @description Red-phase contract tests for the upcoming `get_slice_context` tool.
 *
 * The workflow MCP entry point was refactored so it is directly importable in
 * addition to its stdio production runtime path. The stdio tests exercise the
 * spawned server; direct-import tests verify the exported seams and the
 * `get_slice_context` tool with an injectable `searchContextFn` mock.
 */
import { spawn } from 'node:child_process';
import { writeFile, unlink } from 'node:fs/promises';
import path from 'node:path';

const REPO_ROOT = path.resolve(__dirname, '../../../');
const SERVER_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs',
);

interface JsonRpcRequest {
  jsonrpc: '2.0';
  id: number;
  method: string;
  params?: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc: '2.0';
  id?: number;
  result?: unknown;
  error?: { code: number; message: string };
}

interface WorkflowTool {
  name: string;
  handler: (args: Record<string, unknown>) => Promise<unknown>;
}

interface SliceContextResult {
  compact: boolean;
  slice_id: string;
  plan: string;
  phase: number | null;
  step_number: number | null;
  step_title: string;
  title: string;
  status: string;
  goal: string | null;
  estimate_hours: number | null;
  parallelizable: boolean | null;
  tdd_sequence: string | null;
  mode: string | null;
  skills: string[];
  validation: string[];
  files_to_change: string[];
  acceptance_criteria: Array<{
    id: string;
    text: string;
    validation: string | null;
  }>;
  dependencies: string[];
  next_slice: string | null;
  next_step: string | null;
  instructions: string;
  context: {
    query: string | null;
    text: string;
    chunks: Array<Record<string, unknown>>;
    token_count: number;
    dense_state: string | null;
    truncated: boolean;
    follow_up_refs: Array<Record<string, unknown>>;
  };
  notFound?: boolean;
  message?: string;
  truncated?: boolean;
  fallback_message?: string;
}

interface SearchContextResponse {
  dense_state?: unknown;
  results?: unknown;
}

interface SelfCheckIssue {
  severity: string;
  path?: string;
  message: string;
}

interface SelfCheckReport {
  ok: boolean;
  issues: SelfCheckIssue[];
}

async function runMcpSession(
  requests: JsonRpcRequest[],
  timeoutMs = 10000,
): Promise<JsonRpcResponse[]> {
  const child = spawn(
    process.execPath,
    [SERVER_PATH, '--plan=plans/mcp-active-binding.plans.md'],
    {
      cwd: REPO_ROOT,
      stdio: ['pipe', 'pipe', 'pipe'],
    },
  );

  const responses: JsonRpcResponse[] = [];
  const stdoutBuffer: string[] = [];
  const expectedResponseCount = requests.length + 1; // initialize + requests

  return await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      child.kill();
      reject(
        new Error(
          `Timed out after ${timeoutMs}ms waiting for ${expectedResponseCount} responses`,
        ),
      );
    }, timeoutMs);

    child.stdout?.on('data', (chunk: Buffer) => {
      stdoutBuffer.push(chunk.toString('utf8'));
      const lines = stdoutBuffer.join('').split('\n');
      stdoutBuffer.length = 0;
      // Keep any trailing incomplete line for the next chunk.
      const trailing = lines.pop();
      if (trailing !== undefined && trailing !== '') {
        stdoutBuffer.push(trailing);
      }
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) {
          continue;
        }
        try {
          responses.push(JSON.parse(trimmed) as JsonRpcResponse);
        } catch {
          // Ignore non-JSON diagnostics.
        }
      }
      if (responses.length >= expectedResponseCount) {
        clearTimeout(timeout);
        child.stdin?.end();
        resolve(responses.slice(1)); // drop initialize response
      }
    });

    child.on('error', (error) => {
      clearTimeout(timeout);
      reject(error);
    });

    child.on('close', () => {
      clearTimeout(timeout);
      resolve(responses.slice(1));
    });

    // Handshake, then the caller's requests.
    child.stdin?.write(
      JSON.stringify({
        jsonrpc: '2.0',
        id: 0,
        method: 'initialize',
        params: {
          protocolVersion: '2024-11-05',
          capabilities: {},
          clientInfo: { name: 'test', version: '0.1.0' },
        },
      }) + '\n',
    );
    for (const request of requests) {
      child.stdin?.write(JSON.stringify(request) + '\n');
    }
  });
}

describe('neataptic-workflow-mcp get_slice_context contract', () => {
  it('registers get_slice_context in the workflow MCP tool list', async () => {
    const [response] = await runMcpSession([
      { jsonrpc: '2.0', id: 1, method: 'tools/list' },
    ]);
    const tools = (
      (response.result as { tools?: Array<{ name: string }> } | undefined)
        ?.tools ?? []
    ).map((tool) => tool.name);

    expect(tools).toContain('get_slice_context');
  });

  it('returns a JSON context window for a valid slice_id', async () => {
    const [response] = await runMcpSession([
      {
        jsonrpc: '2.0',
        id: 1,
        method: 'tools/call',
        params: {
          name: 'get_slice_context',
          arguments: {
            slice_id: 'E2-green',
            plan_path:
              'plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md',
          },
        },
      },
    ]);

    expect({
      hasError: response.error !== undefined,
      hasResult: response.result !== undefined,
    }).toEqual({ hasError: false, hasResult: true });
  });

  it('includes the compact slice summary fields in the context window', async () => {
    const [response] = await runMcpSession(
      [
        {
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: {
            name: 'get_slice_context',
            arguments: {
              slice_id: 'B3-impl',
              plan_path: 'plans/completed/orchestration-fixes.plans.md',
            },
          },
        },
      ],
      60000,
    );
    const result = (response.result ?? {}) as {
      structuredContent?: Record<string, unknown>;
    };

    expect({
      hasError: response.error !== undefined,
      hasStructuredContent: result.structuredContent !== undefined,
      keys: Object.keys(result.structuredContent ?? {}),
    }).toEqual({
      hasError: false,
      hasStructuredContent: true,
      keys: expect.arrayContaining([
        'compact',
        'slice_id',
        'plan',
        'step_number',
        'title',
        'status',
        'goal',
        'files_to_change',
        'acceptance_criteria',
        'dependencies',
        'next_slice',
      ]),
    });
  });

  it('returns a not-found contract for an unknown slice_id', async () => {
    const [response] = await runMcpSession([
      {
        jsonrpc: '2.0',
        id: 1,
        method: 'tools/call',
        params: {
          name: 'get_slice_context',
          arguments: { slice_id: 'UNKNOWN' },
        },
      },
    ]);
    const result = (response.result ?? {}) as {
      structuredContent?: Record<string, unknown>;
    };

    expect({
      hasError: response.error !== undefined,
      notFound: result.structuredContent?.notFound,
    }).toEqual({ hasError: false, notFound: true });
  });
});

describe('neataptic-workflow-mcp direct import contract', () => {
  afterEach(() => {
    process.exitCode = undefined;
  });

  afterAll(() => {
    process.exitCode = 0;
  });

  async function loadModule() {
    return await import(SERVER_PATH);
  }

  function makeSearchContextFn(
    results: Array<Record<string, unknown>> = [],
    context?: Record<string, unknown>,
  ) {
    return jest.fn().mockResolvedValue({
      dense_state: 'cold',
      results,
      ...(context !== undefined ? { context } : {}),
    });
  }

  async function createWorkflowServer(
    planPath?: string,
    searchContextFn?: jest.Mock,
  ) {
    const mod = await loadModule();
    const utils = await import(
      path.resolve(REPO_ROOT, 'scripts/agent-customization/mcp/mcp-utils.mjs')
    );
    return utils.createMcpServer({
      serverName: 'test-workflow',
      serverVersion: '0.1.0',
      tools: mod.createWorkflowTools({ planPath, searchContextFn }),
    });
  }

  it('exports the expected entry points and tool factory', async () => {
    const mod = await loadModule();

    expect([
      mod.createWorkflowTools,
      mod.runWorkflowSelfCheck,
      mod.main,
      mod.bootstrapMain,
    ]).toEqual(expect.arrayContaining([expect.any(Function)]));
  });

  it('createWorkflowTools registers three workflow tools including get_slice_context', async () => {
    const mod = await loadModule();
    const tools = mod.createWorkflowTools({
      planPath: 'plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md',
      searchContextFn: makeSearchContextFn(),
    }) as WorkflowTool[];
    const names = tools.map((tool) => tool.name);

    expect(names).toEqual([
      'get_active_workflow_snapshot',
      'get_customization_inventory',
      'get_slice_context',
    ]);
  });

  it('get_slice_context assembles a compact summary for the active step label', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step E2 — Active step [WIP]

\`\`\`yaml
phase: 1
step: E2
title: 'E2: Active step'
slices:
  - slice_id: E2
    title: E2 slice
    status: '[WIP]'
    goal: test goal
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: 'E2',
        plan_path: planPath,
      })) as SliceContextResult;

      expect({
        compact: result.compact,
        slice_id: result.slice_id,
        title: result.title,
        status: result.status,
        goal: result.goal,
      }).toEqual({
        compact: true,
        slice_id: 'E2',
        title: 'E2 slice',
        status: '[WIP]',
        goal: 'test goal',
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context returns a not-found contract for unknown slice_id', async () => {
    const mod = await loadModule();
    const tools = mod.createWorkflowTools({
      planPath: 'plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md',
      searchContextFn: makeSearchContextFn([]),
    }) as WorkflowTool[];
    const tool = tools.find((t) => t.name === 'get_slice_context');
    const result = (await (tool as WorkflowTool).handler({
      slice_id: 'UNKNOWN-SLICE',
    })) as SliceContextResult;

    expect(result.notFound).toBe(true);
  });

  it('get_active_workflow_snapshot returns active step metadata for a WIP plan', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
\`\`\`

#### Step 02: Next step [PLANNED]

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_active_workflow_snapshot');
      const result = (await (tool as WorkflowTool).handler({
        plan_path: planPath,
      })) as Record<string, unknown>;

      expect(result.activeStep).toEqual(
        expect.objectContaining({
          number: 1,
          status: 'WIP',
          title: 'Active step',
        }),
      );
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_active_workflow_snapshot returns null activeStep when no phase is WIP', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [DONE]

#### Step 01: First step [DONE]

\`\`\`yaml
phase: 1
step: 1
status: '[DONE]'
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({ planPath }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_active_workflow_snapshot');
      const result = (await (tool as WorkflowTool).handler({
        plan_path: planPath,
      })) as Record<string, unknown>;

      expect(result.activeStep).toBeNull();
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_customization_inventory returns deterministic inventory via handler', async () => {
    const mod = await loadModule();
    const tools = mod.createWorkflowTools({}) as WorkflowTool[];
    const tool = tools.find((t) => t.name === 'get_customization_inventory');
    const result = (await (tool as WorkflowTool).handler({})) as Record<
      string,
      unknown
    >;

    expect({
      ok: result.ok,
      agentsType: typeof (result.summary as Record<string, number>)?.agents,
      agentsIsArray: Array.isArray(result.agents),
    }).toEqual({
      ok: true,
      agentsType: 'number',
      agentsIsArray: true,
    });
  });

  it('get_customization_inventory surfaces a non-zero exit code', async () => {
    const mod = await loadModule();
    const runner = jest.fn().mockResolvedValue({
      exitCode: 1,
      stdout: '',
      stderr: '',
    });
    const tools = mod.createWorkflowTools({
      inventoryCommandRunner: runner,
    }) as WorkflowTool[];
    const tool = tools.find((t) => t.name === 'get_customization_inventory');

    await expect((tool as WorkflowTool).handler({})).rejects.toThrow(
      'exit code 1',
    );
  });

  it('get_customization_inventory invokes the inventory runner once', async () => {
    const mod = await loadModule();
    const runner = jest.fn().mockResolvedValue({
      exitCode: 1,
      stdout: '',
      stderr: '',
    });
    const tools = mod.createWorkflowTools({
      inventoryCommandRunner: runner,
    }) as WorkflowTool[];
    const tool = tools.find((t) => t.name === 'get_customization_inventory');

    await (tool as WorkflowTool).handler({}).catch(() => undefined);
    expect(runner).toHaveBeenCalledTimes(1);
  });

  it('get_customization_inventory surfaces invalid JSON output', async () => {
    const mod = await loadModule();
    const runner = jest.fn().mockResolvedValue({
      exitCode: 0,
      stdout: 'not-json',
      stderr: '',
    });
    const tools = mod.createWorkflowTools({
      inventoryCommandRunner: runner,
    }) as WorkflowTool[];
    const tool = tools.find((t) => t.name === 'get_customization_inventory');

    await expect((tool as WorkflowTool).handler({})).rejects.toThrow(
      'valid JSON',
    );
  });

  it('runWorkflowSelfCheck reports an active step when the plan has one', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
\`\`\`

## Validation gates

- none
`);
    const server = await createWorkflowServer(planPath);
    const logSpy = jest.spyOn(console, 'log').mockImplementation();
    const errorSpy = jest.spyOn(console, 'error').mockImplementation();
    try {
      const report = await mod.runWorkflowSelfCheck({ server, planPath });
      expect(report.ok).toBe(true);
    } finally {
      logSpy.mockRestore();
      errorSpy.mockRestore();
      await removeTempPlan(planPath);
    }
  });

  it('runWorkflowSelfCheck reports a missing active WIP step', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

\`\`\`yaml
phase: 1
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
    const server = await createWorkflowServer(planPath);
    const logSpy = jest.spyOn(console, 'log').mockImplementation();
    const errorSpy = jest.spyOn(console, 'error').mockImplementation();
    try {
      const report = (await mod.runWorkflowSelfCheck({
        server,
        planPath,
      })) as Record<string, unknown>;
      expect({
        ok: report.ok,
        issueMessages: (report.issues as Array<{ message: string }>).map(
          (issue) => issue.message,
        ),
      }).toEqual({
        ok: false,
        issueMessages: expect.arrayContaining([
          expect.stringContaining('No active WIP phase or step'),
        ]),
      });
    } finally {
      logSpy.mockRestore();
      errorSpy.mockRestore();
      await removeTempPlan(planPath);
    }
  });

  it('main --help exits 0', async () => {
    const mod = await loadModule();
    const exitSpy = jest.spyOn(process, 'exit').mockImplementation(() => {
      throw new Error('PROCESS_EXIT');
    });
    const originalArgv = process.argv;
    process.argv = ['node', 'script', '--help'];
    try {
      await expect(mod.main()).rejects.toThrow('PROCESS_EXIT');
    } finally {
      process.argv = originalArgv;
      exitSpy.mockRestore();
    }
  });

  it('main --help prints usage', async () => {
    const mod = await loadModule();
    const logSpy = jest.spyOn(console, 'log').mockImplementation();
    const exitSpy = jest.spyOn(process, 'exit').mockImplementation(() => {
      throw new Error('PROCESS_EXIT');
    });
    const originalArgv = process.argv;
    process.argv = ['node', 'script', '--help'];
    try {
      await mod.main().catch(() => undefined);
      expect(logSpy).toHaveBeenCalledWith(
        expect.stringContaining('workflow facts as a direct MCP server'),
      );
    } finally {
      process.argv = originalArgv;
      logSpy.mockRestore();
      exitSpy.mockRestore();
    }
  });

  it('main --self-check runs a local audit and exits 0 for a valid plan', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
\`\`\`

## Validation gates

- none
`);
    const originalExitCode = process.exitCode;
    const logSpy = jest.spyOn(console, 'log').mockImplementation();
    const originalArgv = process.argv;
    process.argv = ['node', 'script', '--self-check', `--plan=${planPath}`];
    try {
      await mod.main();
      expect(process.exitCode).toBe(0);
    } finally {
      process.argv = originalArgv;
      process.exitCode = originalExitCode;
      logSpy.mockRestore();
      await removeTempPlan(planPath);
    }
  });

  it('bootstrapMain catches a startup error and sets exitCode', async () => {
    const mod = await loadModule();
    const errorSpy = jest.spyOn(console, 'error').mockImplementation();
    const originalExitCode = process.exitCode;
    process.exitCode = 0;
    try {
      mod.bootstrapMain();
      await new Promise((resolve) => setImmediate(resolve));
      expect(process.exitCode).toBe(1);
    } finally {
      process.exitCode = originalExitCode;
      errorSpy.mockRestore();
    }
  });

  it('bootstrapMain logs a startup error', async () => {
    const mod = await loadModule();
    const errorSpy = jest.spyOn(console, 'error').mockImplementation();
    try {
      mod.bootstrapMain();
      await new Promise((resolve) => setImmediate(resolve));
      expect(errorSpy).toHaveBeenCalled();
    } finally {
      errorSpy.mockRestore();
    }
  });

  it('main starts the stdio server in default mode', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
\`\`\`

## Validation gates

- none
`);
    const runStdio = jest.fn().mockResolvedValue(undefined);
    const originalArgv = process.argv;
    process.argv = ['node', 'script', `--plan=${planPath}`];
    try {
      await mod.main(process.argv.slice(2), { runStdio });
      expect(runStdio.mock.calls).toEqual([
        [expect.objectContaining({ tools: expect.any(Array) })],
      ]);
    } finally {
      process.argv = originalArgv;
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context resolves an active numeric step by step number and title', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
title: '01: Active step'
acceptance_criteria:
  - id: AC-1
    text: 'Build the thing'
    validation: 'npx jest'
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: '01',
        plan_path: planPath,
      })) as SliceContextResult;

      expect({
        step_number: result.step_number,
        title: result.title,
        acceptance_criteria: result.acceptance_criteria,
      }).toEqual({
        step_number: 1,
        title: '01',
        acceptance_criteria: [
          { id: 'AC-1', text: 'Build the thing', validation: 'npx jest' },
        ],
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context resolves an exact slice descriptor when slices are declared', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
title: '01: Active step'
slices:
  - slice_id: slice-a
    title: Slice A
    status: '[WIP]'
    goal: Build it
    files_to_change:
      - src/a.ts
    dependencies:
      - dep-1
    next_slice: slice-b
    acceptance_criteria:
      - id: AC-SLICE
        text: 'Slice passes'
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: 'slice-a',
        plan_path: planPath,
      })) as SliceContextResult;

      expect({
        compact: result.compact,
        slice_id: result.slice_id,
        step_number: result.step_number,
        title: result.title,
        status: result.status,
        goal: result.goal,
        files_to_change: result.files_to_change,
        acceptance_criteria: result.acceptance_criteria,
        dependencies: result.dependencies,
        next_slice: result.next_slice,
      }).toEqual({
        compact: true,
        slice_id: 'slice-a',
        step_number: 1,
        title: 'Slice A',
        status: '[WIP]',
        goal: 'Build it',
        files_to_change: ['src/a.ts'],
        acceptance_criteria: [
          { id: 'AC-SLICE', text: 'Slice passes', validation: null },
        ],
        dependencies: ['dep-1'],
        next_slice: 'slice-b',
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context handles a step packet without a YAML block', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
\`\`\`

#### Step 02: No yaml step [PLANNED]

This step has no YAML block.

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: '02',
        plan_path: planPath,
      })) as SliceContextResult;

      expect({
        step_number: result.step_number,
        title: result.title,
        acceptance_criteria: result.acceptance_criteria,
      }).toEqual({
        step_number: 2,
        title: '02',
        acceptance_criteria: [],
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context handles an invalid YAML block gracefully', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
\`\`\`

#### Step 02: Bad yaml step [PLANNED]

This step has an invalid YAML block.

\`\`\`yaml
not: valid: yaml: [
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: '02',
        plan_path: planPath,
      })) as SliceContextResult;

      expect({
        step_number: result.step_number,
        title: result.title,
        acceptance_criteria: result.acceptance_criteria,
      }).toEqual({
        step_number: 2,
        title: '02',
        acceptance_criteria: [],
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context resolves the active step by exact numeric id', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: '1',
        plan_path: planPath,
      })) as SliceContextResult;

      expect({
        step_number: result.step_number,
        title: result.title,
      }).toEqual({
        step_number: 1,
        title: '1',
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context returns a step descriptor when slice_id matches the step title', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Deploy feature [WIP]

\`\`\`yaml
phase: 1
step: 1
title: 'Deploy feature'
status: '[WIP]'
acceptance_criteria:
  - id: AC-1
    text: 'Deploy passes'
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: 'Deploy',
        plan_path: planPath,
      })) as SliceContextResult;

      expect({
        step_number: result.step_number,
        title: result.title,
        acceptance_criteria: result.acceptance_criteria,
      }).toEqual({
        step_number: 1,
        title: 'Deploy',
        acceptance_criteria: [
          { id: 'AC-1', text: 'Deploy passes', validation: null },
        ],
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context derives a step label from a dashed slice_id to match the step title', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Deploy feature [WIP]

\`\`\`yaml
phase: 1
step: 1
title: 'Deploy feature'
status: '[WIP]'
acceptance_criteria:
  - id: AC-1
    text: 'Deploy passes'
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: 'Deploy-feature',
        plan_path: planPath,
      })) as SliceContextResult;

      expect({
        step_number: result.step_number,
        title: result.title,
        acceptance_criteria: result.acceptance_criteria,
      }).toEqual({
        step_number: 1,
        title: 'Deploy-feature',
        acceptance_criteria: [
          { id: 'AC-1', text: 'Deploy passes', validation: null },
        ],
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context evaluates title fallbacks when the YAML block omits title', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: No title step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: 'No',
        plan_path: planPath,
      })) as SliceContextResult;

      expect({
        notFound: result.notFound,
        slice_id: result.slice_id,
      }).toEqual({
        notFound: true,
        slice_id: 'No',
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context builds a compact summary for a slice missing title', async () => {
    const mod = await loadModule();
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
  - slice_id: slice-notitle
    status: '[WIP]'
    goal: finish it
    files_to_change:
      - src/foo.ts
    acceptance_criteria: []
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: 'slice-notitle',
        plan_path: planPath,
      })) as SliceContextResult;

      expect({
        slice_id: result.slice_id,
        title: result.title,
        files_to_change: result.files_to_change,
      }).toEqual({
        slice_id: 'slice-notitle',
        title: '',
        files_to_change: ['src/foo.ts'],
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context returns notFound when the step title does not match metadata', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Deploy feature [WIP]

\`\`\`yaml
phase: 1
step: 1
title: 'Build feature'
status: '[WIP]'
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: 'Deploy',
        plan_path: planPath,
      })) as SliceContextResult;

      expect(result.notFound).toBe(true);
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context builds a compact summary for a slice with minimal fields', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
title: '01: Active step'
slices:
  - slice_id: slice-min
    title: Minimal slice
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: 'slice-min',
        plan_path: planPath,
      })) as SliceContextResult;

      expect({
        compact: result.compact,
        slice_id: result.slice_id,
        step_number: result.step_number,
        title: result.title,
        status: result.status,
        goal: result.goal,
        files_to_change: result.files_to_change,
        acceptance_criteria: result.acceptance_criteria,
        dependencies: result.dependencies,
        next_slice: result.next_slice,
      }).toEqual({
        compact: true,
        slice_id: 'slice-min',
        step_number: 1,
        title: 'Minimal slice',
        status: '',
        goal: '',
        files_to_change: [],
        acceptance_criteria: [],
        dependencies: [],
        next_slice: null,
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context normalizes test contracts with missing fields', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
title: '01: Active step'
acceptance_criteria:
  - text: 'No id contract'
  - id: 'ID-ONLY'
  - {}
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: '01',
        plan_path: planPath,
      })) as SliceContextResult;

      expect(result.acceptance_criteria).toEqual([
        { id: '', text: 'No id contract', validation: null },
        { id: 'ID-ONLY', text: '', validation: null },
        { id: '', text: '', validation: null },
      ]);
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_active_workflow_snapshot rethrows unexpected load errors', async () => {
    const mod = await loadModule();
    const missingPlan = path.resolve(
      REPO_ROOT,
      'plans/__missing-test.plans.md',
    );
    const tools = mod.createWorkflowTools({
      planPath: missingPlan,
    }) as WorkflowTool[];
    const tool = tools.find((t) => t.name === 'get_active_workflow_snapshot');

    await expect(
      (tool as WorkflowTool).handler({ plan_path: missingPlan }),
    ).rejects.toThrow('Plan file not found');
  });

  it('runWorkflowSelfCheck surfaces snapshot and inventory mismatches', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
\`\`\`

## Validation gates

- none
`);
    const mockServer = {
      tools: [
        { name: 'get_active_workflow_snapshot' },
        { name: 'get_customization_inventory' },
        { name: 'get_slice_context' },
      ],
      dispatch: jest.fn().mockImplementation((request) => {
        if (request.method === 'initialize') {
          return { protocolVersion: 'invalid-version' };
        }
        if (request.method === 'tools/list') {
          return { tools: [{ name: 'only-one' }] };
        }
        if (
          request.method === 'tools/call' &&
          request.params?.name === 'get_active_workflow_snapshot'
        ) {
          return {
            structuredContent: {
              activePhase: { number: 99 },
              activeStep: { number: 99 },
              hookObservations: [],
            },
          };
        }
        if (
          request.method === 'tools/call' &&
          request.params?.name === 'get_customization_inventory'
        ) {
          return { structuredContent: {} };
        }
        return {};
      }),
    };
    try {
      const report = (await mod.runWorkflowSelfCheck({
        server: mockServer,
        planPath,
      })) as Record<string, unknown>;
      expect({
        ok: report.ok,
        messages: (report.issues as Array<{ message: string }>).map(
          (issue) => issue.message,
        ),
      }).toEqual({
        ok: false,
        messages: expect.arrayContaining([
          expect.stringContaining('protocol version'),
          expect.stringContaining('Expected 3 workflow tools'),
          expect.stringContaining('phase did not match'),
          expect.stringContaining('step did not match'),
          expect.stringContaining('bridge-required'),
          expect.stringContaining('expected summary counts'),
        ]),
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('runWorkflowSelfCheck reports an inventory tool error', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
\`\`\`

## Validation gates

- none
`);
    const mockServer = {
      tools: [
        { name: 'get_active_workflow_snapshot' },
        { name: 'get_customization_inventory' },
        { name: 'get_slice_context' },
      ],
      dispatch: jest.fn().mockImplementation((request) => {
        if (request.method === 'initialize') {
          return { protocolVersion: '2024-11-05' };
        }
        if (request.method === 'tools/list') {
          return {
            tools: [
              { name: 'get_active_workflow_snapshot' },
              { name: 'get_customization_inventory' },
              { name: 'get_slice_context' },
            ],
          };
        }
        if (
          request.method === 'tools/call' &&
          request.params?.name === 'get_active_workflow_snapshot'
        ) {
          return {
            structuredContent: {
              activePhase: { number: 1 },
              activeStep: { number: 1 },
            },
          };
        }
        if (
          request.method === 'tools/call' &&
          request.params?.name === 'get_customization_inventory'
        ) {
          return { isError: true, error: { message: 'inventory down' } };
        }
        return {};
      }),
    };
    try {
      const report = (await mod.runWorkflowSelfCheck({
        server: mockServer,
        planPath,
      })) as Record<string, unknown>;
      expect({
        ok: report.ok,
        messages: (report.issues as Array<{ message: string }>).map(
          (issue) => issue.message,
        ),
      }).toEqual({
        ok: false,
        messages: expect.arrayContaining([
          expect.stringContaining('inventory tool returned an error'),
        ]),
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('runWorkflowSelfCheck reports a snapshot tool error', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
\`\`\`

## Validation gates

- none
`);
    const mockServer = {
      tools: [
        { name: 'get_active_workflow_snapshot' },
        { name: 'get_customization_inventory' },
        { name: 'get_slice_context' },
      ],
      dispatch: jest.fn().mockImplementation((request) => {
        if (request.method === 'initialize') {
          return { protocolVersion: '2024-11-05' };
        }
        if (request.method === 'tools/list') {
          return {
            tools: [
              { name: 'get_active_workflow_snapshot' },
              { name: 'get_customization_inventory' },
              { name: 'get_slice_context' },
            ],
          };
        }
        if (
          request.method === 'tools/call' &&
          request.params?.name === 'get_active_workflow_snapshot'
        ) {
          return { isError: true, error: { message: 'snapshot down' } };
        }
        if (
          request.method === 'tools/call' &&
          request.params?.name === 'get_customization_inventory'
        ) {
          return { structuredContent: { summary: { agents: 1, skills: 1 } } };
        }
        return {};
      }),
    };
    try {
      const report = (await mod.runWorkflowSelfCheck({
        server: mockServer,
        planPath,
      })) as Record<string, unknown>;
      expect({
        ok: report.ok,
        messages: (report.issues as Array<{ message: string }>).map(
          (issue) => issue.message,
        ),
      }).toEqual({
        ok: false,
        messages: expect.arrayContaining([
          expect.stringContaining('snapshot tool returned an error'),
        ]),
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('createWorkflowTools can be called with no arguments', async () => {
    const mod = await loadModule();
    const tools = mod.createWorkflowTools();
    expect({
      length: tools.length,
      names: tools.map((t: WorkflowTool) => t.name),
    }).toEqual({
      length: 3,
      names: expect.arrayContaining([
        'get_active_workflow_snapshot',
        'get_customization_inventory',
        'get_slice_context',
      ]),
    });
  });

  it('get_slice_context returns notFound when search results are not an array', async () => {
    const mod = await loadModule();
    const tools = mod.createWorkflowTools({
      planPath: 'plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md',
      searchContextFn: async () =>
        ({ results: 'not-array' }) as unknown as SearchContextResponse,
    }) as WorkflowTool[];
    const tool = tools.find((t) => t.name === 'get_slice_context');
    const result = (await (tool as WorkflowTool).handler({
      slice_id: 'does-not-exist',
    })) as SliceContextResult;
    expect(result.notFound).toBe(true);
  });

  it('get_slice_context preserves array files_to_change in compact output', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
title: 'Active step'
status: '[WIP]'
source_boundary:
  - src/foo.ts
files_to_change:
  - src/bar.ts
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: 'Active',
        plan_path: planPath,
      })) as SliceContextResult;
      expect({
        title: result.title,
        files_to_change: result.files_to_change,
      }).toEqual({
        title: 'Active',
        files_to_change: ['src/bar.ts'],
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context ignores a step packet whose slices field is not an array', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
slices: not-a-list
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: 'missing-slice',
        plan_path: planPath,
      })) as SliceContextResult;
      expect(result.notFound).toBe(true);
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context normalizes non-array files_to_change in compact output', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
title: 'Active step'
status: '[WIP]'
source_boundary: not-array
files_to_change: also-not-array
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: 'Active',
        plan_path: planPath,
      })) as SliceContextResult;
      expect(result.files_to_change).toEqual([]);
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('runWorkflowSelfCheck reports a mismatched tool count', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
\`\`\`

## Validation gates

- none
`);
    const mockServer = {
      tools: [
        { name: 'get_active_workflow_snapshot' },
        { name: 'get_customization_inventory' },
        { name: 'get_slice_context' },
      ],
      dispatch: jest.fn(async (request: JsonRpcRequest) => {
        if (request.method === 'initialize') {
          return { protocolVersion: '2024-11-05' };
        }
        if (request.method === 'tools/list') {
          return { tools: 'not-an-array' };
        }
        if (
          request.method === 'tools/call' &&
          request.params?.name === 'get_active_workflow_snapshot'
        ) {
          return { activeStep: { number: 1, title: 'Active step' } };
        }
        if (
          request.method === 'tools/call' &&
          request.params?.name === 'get_customization_inventory'
        ) {
          return { structuredContent: { summary: { agents: 1, skills: 1 } } };
        }
        return {};
      }),
    };
    try {
      const report = (await mod.runWorkflowSelfCheck({
        server: mockServer,
        planPath,
      })) as Record<string, unknown>;
      expect({
        ok: report.ok,
        messages: (report.issues as Array<{ message: string }>).map(
          (issue) => issue.message,
        ),
      }).toEqual({
        ok: false,
        messages: expect.arrayContaining([
          expect.stringContaining('Expected 3 workflow tools'),
        ]),
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('main --self-check exits 1 when the self-check reports issues', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
\`\`\`

## Validation gates

- none
`);
    const originalExitCode = process.exitCode;
    process.exitCode = 0;
    try {
      await mod.main(['--self-check', `--plan=${planPath}`], {
        runSelfCheck: async () =>
          ({
            ok: false,
            issues: [
              { severity: 'error', path: planPath, message: 'fake issue' },
            ],
          }) as unknown as SelfCheckReport,
      });
      expect(process.exitCode).toBe(1);
    } finally {
      process.exitCode = originalExitCode;
      await removeTempPlan(planPath);
    }
  });

  it('bootstrapMain awaits a successful self-check', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
\`\`\`

## Validation gates

- none
`);
    const originalArgv = process.argv.slice();
    const originalExitCode = process.exitCode;
    process.exitCode = 0;
    process.argv = [
      'node',
      'neataptic-workflow-mcp.mjs',
      '--self-check',
      `--plan=${planPath}`,
    ];
    try {
      await mod.bootstrapMain({
        runSelfCheck: async () => ({ ok: true, issues: [] }),
      });
      expect(process.exitCode).toBe(0);
    } finally {
      process.argv = originalArgv;
      process.exitCode = originalExitCode;
      await removeTempPlan(planPath);
    }
  });

  it('process exitCode is clean after bootstrapMain test', () => {
    expect(process.exitCode).toBeUndefined();
  });

  it('get_slice_context input schema does not advertise the full option', async () => {
    const mod = await loadModule();
    const tools = mod.createWorkflowTools() as Array<
      WorkflowTool & { inputSchema: Record<string, unknown> }
    >;
    const tool = tools.find((t) => t.name === 'get_slice_context');

    expect(tool?.inputSchema).toEqual(
      expect.objectContaining({
        type: 'object',
        properties: expect.objectContaining({
          slice_id: expect.objectContaining({ type: 'string' }),
          plan_path: expect.objectContaining({ type: 'string' }),
        }),
        required: ['slice_id'],
        additionalProperties: false,
      }),
    );
    expect(tool?.inputSchema).not.toHaveProperty('properties.full');
  });

  it('get_slice_context default compact mode returns only essential slice fields', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
title: '01: Active step'
slices:
  - slice_id: compact-slice
    title: Compact slice
    status: '[WIP]'
    goal: compact goal
    files_to_change:
      - src/a.ts
    dependencies:
      - other-slice
    next_slice: next-slice
    acceptance_criteria:
      - id: AC-1
        text: first criterion
      - id: AC-2
        text: second criterion
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: 'compact-slice',
        plan_path: planPath,
      })) as Record<string, unknown>;
      const expectedPlan = path
        .relative(REPO_ROOT, planPath)
        .split(path.sep)
        .join('/');

      expect(result).toEqual(
        expect.objectContaining({
          compact: true,
          slice_id: 'compact-slice',
          plan: expectedPlan,
          phase: 1,
          step_number: 1,
          step_title: '01: Active step',
          title: 'Compact slice',
          status: '[WIP]',
          goal: 'compact goal',
          estimate_hours: null,
          parallelizable: null,
          tdd_sequence: null,
          mode: null,
          skills: [],
          validation: [],
          files_to_change: ['src/a.ts'],
          acceptance_criteria: [
            { id: 'AC-1', text: 'first criterion', validation: null },
            { id: 'AC-2', text: 'second criterion', validation: null },
          ],
          dependencies: ['other-slice'],
          next_slice: 'next-slice',
          next_step: null,
          instructions: expect.stringContaining('compact-slice'),
          context: expect.objectContaining({
            query: expect.any(String),
            text: '',
            chunks: [],
            token_count: 0,
            truncated: false,
          }),
        }),
      );
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context default compact mode omits the full step packet and source chunks', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
title: '01: Active step'
slices:
  - slice_id: omit-slice
    title: Omit slice
    status: '[WIP]'
    goal: omit goal
    files_to_change:
      - src/a.ts
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: async () =>
          ({
            results: [{ chunk_id: 1, text: 'chunk text' }],
          }) as unknown as SearchContextResponse,
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: 'omit-slice',
        plan_path: planPath,
      })) as Record<string, unknown>;

      expect({
        title: result.title,
        hasStepPacket: 'stepPacket' in result,
        hasSourceChunks: 'sourceChunks' in result,
        hasBoundaryNotes: 'boundaryNotes' in result,
        hasTestContracts: 'testContracts' in result,
        hasCompactFlag: result.compact,
      }).toEqual({
        title: 'Omit slice',
        hasStepPacket: false,
        hasSourceChunks: false,
        hasBoundaryNotes: false,
        hasTestContracts: false,
        hasCompactFlag: true,
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context default compact mode keeps the formatted envelope under 16 KB', async () => {
    const mod = await loadModule();
    const utils = await import(
      path.resolve(REPO_ROOT, 'scripts/agent-customization/mcp/mcp-utils.mjs')
    );
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
title: '01: Active step'
slices:
  - slice_id: small-slice
    title: Small slice
    status: '[WIP]'
    goal: small goal
    files_to_change:
      - src/a.ts
\`\`\`

## Validation gates

- none
`);
    try {
      const searchContextFn = makeSearchContextFn(
        [
          {
            chunk_id: 'chunk-1',
            file_path: 'src/a.ts',
            text: 'First relevant chunk body from src/a.ts.',
          },
          {
            chunk_id: 'chunk-2',
            file_path: 'src/a.ts',
            text: 'Second relevant chunk body from src/a.ts.',
          },
        ],
        {
          context:
            '# Context from src/a.ts\n\nFirst relevant chunk body from src/a.ts.\n\nSecond relevant chunk body from src/a.ts.',
          chunks: [
            {
              chunk_id: 'chunk-1',
              file_path: 'src/a.ts',
              content: 'First relevant chunk body from src/a.ts.',
              char_start: 0,
              char_end: 40,
            },
            {
              chunk_id: 'chunk-2',
              file_path: 'src/a.ts',
              content: 'Second relevant chunk body from src/a.ts.',
              char_start: 42,
              char_end: 86,
            },
          ],
          tokenCount: 24,
        },
      );
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn,
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const rawResult = (await (tool as WorkflowTool).handler({
        slice_id: 'small-slice',
        plan_path: planPath,
      })) as Record<string, unknown>;
      const envelope = utils.formatToolResult(rawResult);

      const envelopeBytes = Buffer.byteLength(JSON.stringify(envelope), 'utf8');
      expect(envelopeBytes).toBeLessThanOrEqual(16_384);
      expect(Array.isArray(envelope.content)).toBe(true);
      expect(envelope.structuredContent).toBeDefined();

      const payload = envelope.structuredContent as Record<string, unknown>;
      const chunks = (payload.context as Record<string, unknown>)?.chunks as
        Array<Record<string, unknown>> | undefined;
      expect(chunks?.length).toBeGreaterThan(0);
      for (const chunk of chunks ?? []) {
        expect(typeof chunk.file_path).toBe('string');
        expect((chunk.file_path as string).length).toBeGreaterThan(0);
        expect(typeof chunk.chunk_id).not.toBe('undefined');
      }
      expect(searchContextFn).toHaveBeenCalledWith(
        expect.objectContaining({
          query: expect.stringContaining('small slice') as string,
          context_format: 'json',
          expand_query: true,
          use_rerank: true,
          metadata: {
            filter: {
              op: 'or',
              predicates: [
                expect.objectContaining({
                  op: 'like',
                  field: 'file_path',
                  value: expect.stringContaining('__test-active-') as string,
                }),
                {
                  op: 'like',
                  field: 'file_path',
                  value: '%src/a.ts%',
                },
              ],
            },
          },
        }),
      );
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context default compact mode truncates payloads that exceed 16 KB', async () => {
    const mod = await loadModule();
    const hugeGoal = 'x'.repeat(20_000);
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
title: '01: Active step'
slices:
  - slice_id: huge-slice
    title: Huge slice
    status: '[WIP]'
    goal: ${hugeGoal}
    files_to_change:
      - src/a.ts
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: 'huge-slice',
        plan_path: planPath,
      })) as Record<string, unknown>;

      expect({
        truncated: result.truncated,
        hasFallbackMessage: typeof result.fallback_message === 'string',
        hasHugeGoal: result.goal === hugeGoal,
      }).toEqual({
        truncated: true,
        hasFallbackMessage: true,
        hasHugeGoal: false,
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  it('get_slice_context default notFound response is compact', async () => {
    const mod = await loadModule();
    const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
\`\`\`

## Validation gates

- none
`);
    try {
      const tools = mod.createWorkflowTools({
        planPath,
        searchContextFn: makeSearchContextFn([]),
      }) as WorkflowTool[];
      const tool = tools.find((t) => t.name === 'get_slice_context');
      const result = (await (tool as WorkflowTool).handler({
        slice_id: 'missing-slice',
        plan_path: planPath,
      })) as Record<string, unknown>;

      expect({
        compact: result.compact,
        notFound: result.notFound,
        hasSourceChunks: 'sourceChunks' in result,
      }).toEqual({
        compact: true,
        notFound: true,
        hasSourceChunks: false,
      });
    } finally {
      await removeTempPlan(planPath);
    }
  });

  describe('get_slice_context RAG assembly fixes', () => {
    it('deduplicates chunks with identical or overlapping file ranges', async () => {
      const mod = await loadModule();
      const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01 — RAG step [WIP]

\`\`\`yaml
phase: 1
step: 1
title: '01: RAG step'
slices:
  - slice_id: dedup-slice
    title: 'deduplicate chunks'
    status: '[WIP]'
    goal: test deduplication
    files_to_change:
      - src/constants.ts
    acceptance_criteria:
      - id: AC-1
        text: 'overlapping ranges share same file path'
\`\`\`

## Validation gates

- none
`);
      try {
        const results = [
          {
            chunk_id: 'dedup-a',
            file_path: 'src/constants.ts',
            char_start: 0,
            char_end: 100,
            text: 'chunk a body',
            score: 0.9,
          },
          {
            chunk_id: 'dedup-b',
            file_path: 'src/constants.ts',
            char_start: 0,
            char_end: 120,
            text: 'chunk b body',
            score: 0.8,
          },
          {
            chunk_id: 'dedup-c',
            file_path: 'src/constants.ts',
            char_start: 0,
            char_end: 100,
            text: 'chunk c body',
            score: 0.7,
          },
          {
            chunk_id: 'dedup-d',
            file_path: 'src/constants.ts',
            char_start: 200,
            char_end: 300,
            text: 'chunk d body',
            score: 0.6,
          },
        ];
        const tools = mod.createWorkflowTools({
          planPath,
          searchContextFn: makeSearchContextFn(
            results as Array<Record<string, unknown>>,
          ),
        }) as WorkflowTool[];
        const tool = tools.find((t) => t.name === 'get_slice_context');
        const result = (await (tool as WorkflowTool).handler({
          slice_id: 'dedup-slice',
          plan_path: planPath,
        })) as Record<string, unknown>;
        const ctx = result.context as Record<string, unknown>;
        const chunks = ctx.chunks as Array<Record<string, unknown>>;
        const starts = chunks
          .map((c) => c.char_start as number)
          .sort((a, b) => a - b);

        expect({
          chunkCount: chunks.length,
          starts,
          uniqueIds: new Set(chunks.map((c) => c.chunk_id)).size,
        }).toEqual({
          chunkCount: 2,
          starts: [0, 200],
          uniqueIds: 2,
        });
      } finally {
        await removeTempPlan(planPath);
      }
    });

    it('issues one search_context call per acceptance criterion and merges results', async () => {
      const mod = await loadModule();
      const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01 — RAG step [WIP]

\`\`\`yaml
phase: 1
step: 1
title: '01: RAG step'
slices:
  - slice_id: multiquery-slice
    title: 'player health ammo'
    status: '[WIP]'
    goal: implement hero state
    files_to_change:
      - src/hero.ts
      - src/damage.ts
      - src/dash.ts
    acceptance_criteria:
      - id: AC-207
        text: 'damage decreases health by exact amount'
      - id: AC-208
        text: 'dash grants invulnerability frames during dash'
\`\`\`

## Validation gates

- none
`);
      try {
        const searchContextFn = jest
          .fn()
          .mockImplementation(async (opts: Record<string, unknown>) => {
            const q = String(opts.query ?? '');
            // Order matters: AC queries also contain generic words such as
            // 'health', so match the more specific AC terms first.
            if (q.includes('damage')) {
              return {
                dense_state: 'cold',
                results: [
                  {
                    chunk_id: 'ac-damage',
                    file_path: 'src/damage.ts',
                    char_start: 0,
                    char_end: 10,
                    text: 'damage chunk',
                    score: 0.8,
                  },
                ],
              };
            }
            if (q.includes('dash')) {
              return {
                dense_state: 'cold',
                results: [
                  {
                    chunk_id: 'ac-dash',
                    file_path: 'src/dash.ts',
                    char_start: 0,
                    char_end: 10,
                    text: 'dash chunk',
                    score: 0.8,
                  },
                ],
              };
            }
            if (
              q.includes('player') ||
              q.includes('health') ||
              q.includes('ammo')
            ) {
              return {
                dense_state: 'cold',
                results: [
                  {
                    chunk_id: 'primary',
                    file_path: 'src/hero.ts',
                    char_start: 0,
                    char_end: 10,
                    text: 'primary chunk',
                    score: 0.9,
                  },
                ],
              };
            }
            return { dense_state: 'cold', results: [] };
          }) as jest.Mock;
        const tools = mod.createWorkflowTools({
          planPath,
          searchContextFn: searchContextFn as unknown as jest.Mock,
        }) as WorkflowTool[];
        const tool = tools.find((t) => t.name === 'get_slice_context');
        const result = (await (tool as WorkflowTool).handler({
          slice_id: 'multiquery-slice',
          plan_path: planPath,
        })) as Record<string, unknown>;
        const ctx = result.context as Record<string, unknown>;
        const chunks = ctx.chunks as Array<Record<string, unknown>>;
        const paths = chunks.map((c) => c.file_path as string).sort();

        expect({
          calls: searchContextFn.mock.calls.length,
          paths,
        }).toEqual({
          calls: 3,
          paths: ['src/damage.ts', 'src/dash.ts', 'src/hero.ts'],
        });
      } finally {
        await removeTempPlan(planPath);
      }
    });

    it('prioritizes *.test.ts chunks in TDD slices even with low base score', async () => {
      const mod = await loadModule();
      const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01 — TDD step [WIP]

\`\`\`yaml
phase: 1
step: 1
title: '01: TDD step'
tdd_sequence: 'red-green'
slices:
  - slice_id: tdd-priority-slice
    title: 'state test priority'
    status: '[WIP]'
    goal: verify state behavior
    files_to_change:
      - src/constants.ts
      - src/state.test.ts
    acceptance_criteria:
      - id: AC-1
        text: 'state transitions are deterministic'
\`\`\`

## Validation gates

- none
`);
      try {
        const results = [
          {
            chunk_id: 'test-low',
            file_path: 'src/state.test.ts',
            char_start: 0,
            char_end: 10,
            text: 'test chunk',
            score: 1,
          },
          {
            chunk_id: 'constants-high',
            file_path: 'src/constants.ts',
            char_start: 0,
            char_end: 10,
            text: 'constants chunk',
            score: 100,
          },
          {
            chunk_id: 'other-1',
            file_path: 'src/other1.ts',
            char_start: 0,
            char_end: 10,
            text: 'other 1',
            score: 91,
          },
          {
            chunk_id: 'other-2',
            file_path: 'src/other2.ts',
            char_start: 0,
            char_end: 10,
            text: 'other 2',
            score: 90,
          },
          {
            chunk_id: 'other-3',
            file_path: 'src/other3.ts',
            char_start: 0,
            char_end: 10,
            text: 'other 3',
            score: 89,
          },
          {
            chunk_id: 'other-4',
            file_path: 'src/other4.ts',
            char_start: 0,
            char_end: 10,
            text: 'other 4',
            score: 88,
          },
          {
            chunk_id: 'other-5',
            file_path: 'src/other5.ts',
            char_start: 0,
            char_end: 10,
            text: 'other 5',
            score: 87,
          },
          {
            chunk_id: 'other-6',
            file_path: 'src/other6.ts',
            char_start: 0,
            char_end: 10,
            text: 'other 6',
            score: 86,
          },
          {
            chunk_id: 'other-7',
            file_path: 'src/other7.ts',
            char_start: 0,
            char_end: 10,
            text: 'other 7',
            score: 85,
          },
          {
            chunk_id: 'other-8',
            file_path: 'src/other8.ts',
            char_start: 0,
            char_end: 10,
            text: 'other 8',
            score: 84,
          },
          {
            chunk_id: 'other-9',
            file_path: 'src/other9.ts',
            char_start: 0,
            char_end: 10,
            text: 'other 9',
            score: 82,
          },
        ];
        const tools = mod.createWorkflowTools({
          planPath,
          searchContextFn: makeSearchContextFn(
            results as Array<Record<string, unknown>>,
          ),
        }) as WorkflowTool[];
        const tool = tools.find((t) => t.name === 'get_slice_context');
        const result = (await (tool as WorkflowTool).handler({
          slice_id: 'tdd-priority-slice',
          plan_path: planPath,
        })) as Record<string, unknown>;
        const ctx = result.context as Record<string, unknown>;
        const chunks = ctx.chunks as Array<Record<string, unknown>>;
        const paths = chunks.map((c) => c.file_path as string);

        expect(paths).toContain('src/state.test.ts');
      } finally {
        await removeTempPlan(planPath);
      }
    });

    it('emits a load_chunk follow-up ref for missing test files', async () => {
      const mod = await loadModule();
      const planPath = await writeTempPlan(`
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01 — TDD step [WIP]

\`\`\`yaml
phase: 1
step: 1
title: '01: TDD step'
tdd_sequence: 'red-green'
slices:
  - slice_id: tdd-missing-slice
    title: 'missing test file'
    status: '[WIP]'
    goal: implement state tests
    files_to_change:
      - src/state.test.ts
    acceptance_criteria:
      - id: AC-1
        text: 'state transitions are deterministic'
\`\`\`

## Validation gates

- none
`);
      try {
        const searchContextFn = jest
          .fn()
          .mockImplementation(async (opts: Record<string, unknown>) => {
            const q = String(opts.query ?? '');
            if (q.includes('state.test.ts')) {
              return {
                dense_state: 'cold',
                results: [
                  {
                    chunk_id: 'missing-test',
                    file_path: 'src/state.test.ts',
                    char_start: 0,
                    char_end: 0,
                    text: '',
                    score: 0.5,
                  },
                ],
              };
            }
            return { dense_state: 'cold', results: [] };
          }) as jest.Mock;
        const tools = mod.createWorkflowTools({
          planPath,
          searchContextFn: searchContextFn as unknown as jest.Mock,
        }) as WorkflowTool[];
        const tool = tools.find((t) => t.name === 'get_slice_context');
        const result = (await (tool as WorkflowTool).handler({
          slice_id: 'tdd-missing-slice',
          plan_path: planPath,
        })) as Record<string, unknown>;
        const ctx = result.context as Record<string, unknown>;
        const refs = ctx.follow_up_refs as
          Array<Record<string, unknown>> | undefined;
        const loadChunkRef = refs?.find(
          (ref) =>
            ref.tool === 'load_chunk' &&
            (ref.args as Record<string, unknown>)?.chunk_id === 'missing-test',
        );

        expect(loadChunkRef).toBeDefined();
      } finally {
        await removeTempPlan(planPath);
      }
    });
  });

  async function writeTempPlan(content: string): Promise<string> {
    const fileName = `__test-active-${Date.now()}-${Math.random()
      .toString(36)
      .slice(2)}.plans.md`;
    const absolutePath = path.resolve(REPO_ROOT, 'plans', fileName);
    await writeFile(absolutePath, content.trim(), 'utf-8');
    return absolutePath;
  }

  async function removeTempPlan(absolutePath: string): Promise<void> {
    try {
      await unlink(absolutePath);
    } catch {
      // Best-effort cleanup of the temporary plan fixture.
    }
  }
});
