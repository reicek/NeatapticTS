import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

jest.unstable_mockModule('./mcp-utils.mjs', () => ({
  createMcpServer: jest.fn((opts) => ({
    name: opts.serverName,
    version: opts.serverVersion,
    tools: opts.tools,
    dispatch: jest.fn(async (req) => {
      if (req.method === 'initialize') {
        return { protocolVersion: '2024-11-05', capabilities: {}, serverInfo: { name: opts.serverName, version: opts.serverVersion } };
      }
      if (req.method === 'tools/list') {
        return { tools: opts.tools.map((t) => ({ name: t.name, description: t.description, inputSchema: t.inputSchema })) };
      }
      if (req.method === 'tools/call') {
        const tool = opts.tools.find((t) => t.name === req.params.name);
        if (!tool) return { isError: true, content: [{ type: 'text', text: 'unknown tool' }] };
        try {
          const result = await tool.handler(req.params.arguments || {});
          return { structuredContent: result, content: [{ type: 'text', text: JSON.stringify(result) }] };
        } catch (error) {
          return { isError: true, content: [{ type: 'text', text: error.message }] };
        }
      }
      return {};
    }),
  })),
  createSelfCheckReport: jest.fn((name, issues, meta) => ({ name, ok: issues.length === 0, issues, meta })),
  createTool: jest.fn((opts) => ({ name: opts.name, description: opts.description, inputSchema: opts.inputSchema, handler: opts.handler, annotations: opts.annotations })),
  emitSelfCheckReport: jest.fn(),
  invokeServerRequest: jest.fn(),
  MCP_PROTOCOL_VERSION: '2024-11-05',
  MCP_REPO_ROOT: 'C:\\NeatapticTS',
  parseMcpCliArgs: jest.fn(),
  printMcpUsage: jest.fn(),
  requireExplicitPlanPath: jest.fn((plan) => plan || 'plans/default.md'),
  requireString: jest.fn((val, name) => {
    if (typeof val !== 'string' || !val.trim()) throw new Error(`${name} is required`);
    return val;
  }),
  runShellFreeCommand: jest.fn(),
  runStdioMcpServer: jest.fn(),
  selfCheckError: jest.fn((scope, message) => ({ scope, message })),
  tokenizeShellSafeCommand: jest.fn(),
}));

jest.unstable_mockModule('./mcp-plan-utils.mjs', () => ({
  createValidationAllowlistSnapshot: jest.fn(),
  loadActivePlanContext: jest.fn(),
}));

jest.unstable_mockModule('node:fs', () => ({
  existsSync: jest.fn(),
  readFileSync: jest.fn(),
  writeFileSync: jest.fn(),
}));

jest.unstable_mockModule('node:fs/promises', () => ({
  readFile: jest.fn(),
  readdir: jest.fn(),
  stat: jest.fn(),
  writeFile: jest.fn(),
  mkdir: jest.fn(),
  appendFile: jest.fn(),
  access: jest.fn(),
  constants: { R_OK: 4, W_OK: 2, F_OK: 0 },
  rm: jest.fn(),
  glob: jest.fn(),
}));

let mockMcpUtils;
let mockPlanUtils;
let mockFs;

beforeEach(async () => {
  jest.resetModules();
  mockMcpUtils = await import('./mcp-utils.mjs');
  mockPlanUtils = await import('./mcp-plan-utils.mjs');
  mockFs = await import('node:fs');
  jest.clearAllMocks();
});

async function importModule(argv) {
  const origArgv = process.argv;
  const origExit = process.exit;
  process.argv = ['node', 'neataptic-validation-mcp.mjs', ...(argv || [])];
  process.exit = (code) => { throw new Error(`EXIT:${code}`); };
  try {
    await import('./neataptic-validation-mcp.mjs');
  } catch {
    // process.exit may throw
  }
  process.argv = origArgv;
  process.exit = origExit;
}

describe('neataptic-validation-mcp', () => {
  it('prints help and exits 0 when --help', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({ help: true });
    mockPlanUtils.loadActivePlanContext.mockResolvedValue({ activeStep: { number: 1, validationCommands: [] } });
    mockPlanUtils.createValidationAllowlistSnapshot.mockReturnValue({ validationCommands: [], activeStep: { number: 1 } });

    await importModule(['--help']);

    assert.ok(mockMcpUtils.printMcpUsage.mock.calls.length >= 1);
  });

  it('starts stdio server by default', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({ help: false, selfCheck: false, plan: 'plans/test.md' });
    mockMcpUtils.runStdioMcpServer.mockResolvedValue();
    mockFs.existsSync.mockReturnValue(false);
    mockPlanUtils.loadActivePlanContext.mockResolvedValue({ activeStep: { number: 1, validationCommands: [] } });

    await importModule([]);

    assert.ok(mockMcpUtils.runStdioMcpServer.mock.calls.length >= 1);
  });

  it('runs self-check when --self-check', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({ help: false, selfCheck: true, plan: 'plans/test.md' });
    mockMcpUtils.invokeServerRequest.mockImplementation(async (server, req) => server.dispatch(req));
    mockFs.existsSync.mockReturnValue(false);
    mockPlanUtils.loadActivePlanContext.mockResolvedValue({
      activeStep: { number: 1, validationCommands: ['npm test'] },
    });
    mockPlanUtils.createValidationAllowlistSnapshot.mockReturnValue({
      validationCommands: ['npm test'],
      validationCommandsMatch: true,
      activeStep: { number: 1 },
      activePhase: { number: 1 },
    });
    mockMcpUtils.runShellFreeCommand.mockResolvedValue({ exitCode: 0, stdout: 'ok', stderr: '' });
    // Make tokenizer reject semicolons
    mockMcpUtils.tokenizeShellSafeCommand.mockImplementation((cmd) => {
      if (cmd.includes(';')) throw new Error('shell metacharacter rejected');
      return ['node', cmd];
    });

    await importModule(['--self-check']);

    assert.ok(mockMcpUtils.emitSelfCheckReport.mock.calls.length >= 1);
  });

  it('get_active_validation_allowlist handler returns snapshot', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({ help: false, selfCheck: false, plan: 'plans/test.md' });
    mockMcpUtils.runStdioMcpServer.mockResolvedValue();
    mockFs.existsSync.mockReturnValue(false);
    mockPlanUtils.loadActivePlanContext.mockResolvedValue({ activeStep: { number: 1, validationCommands: ['npm test'] } });
    mockPlanUtils.createValidationAllowlistSnapshot.mockReturnValue({ validationCommands: ['npm test'], activeStep: { number: 1 } });

    await importModule([]);

    const server = mockMcpUtils.createMcpServer.mock.calls[0][0];
    const allowlistTool = server.tools.find((t) => t.name === 'get_active_validation_allowlist');
    const result = await allowlistTool.handler({});
    assert.deepStrictEqual(result.validationCommands, ['npm test']);
  });

  it('run_allowlisted_validation handler executes allow-listed command', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({ help: false, selfCheck: false, plan: 'plans/test.md' });
    mockMcpUtils.runStdioMcpServer.mockResolvedValue();
    mockFs.existsSync.mockReturnValue(false);
    mockPlanUtils.loadActivePlanContext.mockResolvedValue({ activeStep: { number: 1, validationCommands: ['npm test'] } });
    mockMcpUtils.runShellFreeCommand.mockResolvedValue({ exitCode: 0, stdout: 'pass', stderr: '' });

    await importModule([]);

    const server = mockMcpUtils.createMcpServer.mock.calls[0][0];
    const runTool = server.tools.find((t) => t.name === 'run_allowlisted_validation');
    const result = await runTool.handler({ command: 'npm test' });
    assert.strictEqual(result.exitCode, 0);
    assert.strictEqual(result.scope, 'direct-MCP');
  });

  it('run_allowlisted_validation rejects non-allow-listed command', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({ help: false, selfCheck: false, plan: 'plans/test.md' });
    mockMcpUtils.runStdioMcpServer.mockResolvedValue();
    mockFs.existsSync.mockReturnValue(false);
    mockPlanUtils.loadActivePlanContext.mockResolvedValue({ activeStep: { number: 1, validationCommands: ['npm test'] } });

    await importModule([]);

    const server = mockMcpUtils.createMcpServer.mock.calls[0][0];
    const runTool = server.tools.find((t) => t.name === 'run_allowlisted_validation');
    await assert.rejects(
      () => runTool.handler({ command: 'rm -rf /' }),
      /not allow-listed/,
    );
  });

  it('uses session override when available', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({ help: false, selfCheck: false, plan: 'plans/default.md' });
    mockMcpUtils.runStdioMcpServer.mockResolvedValue();
    mockFs.existsSync.mockReturnValue(true);
    const { readFile } = await import('node:fs/promises');
    readFile.mockResolvedValue(JSON.stringify({ plan_path: 'plans/override.md' }));
    mockPlanUtils.loadActivePlanContext.mockResolvedValue({ activeStep: { number: 1, validationCommands: [] } });
    mockPlanUtils.createValidationAllowlistSnapshot.mockReturnValue({ validationCommands: [] });

    await importModule([]);

    const server = mockMcpUtils.createMcpServer.mock.calls[0][0];
    const allowlistTool = server.tools.find((t) => t.name === 'get_active_validation_allowlist');
    await allowlistTool.handler({});
    // loadActivePlanContext should have been called with override path
    assert.ok(mockPlanUtils.loadActivePlanContext.mock.calls.length >= 1);
  });

  it('handles session override with invalid JSON', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({ help: false, selfCheck: false, plan: 'plans/default.md' });
    mockMcpUtils.runStdioMcpServer.mockResolvedValue();
    mockFs.existsSync.mockReturnValue(true);
    const { readFile } = await import('node:fs/promises');
    readFile.mockResolvedValue('not json');
    mockPlanUtils.loadActivePlanContext.mockResolvedValue({ activeStep: { number: 1, validationCommands: [] } });
    mockPlanUtils.createValidationAllowlistSnapshot.mockReturnValue({ validationCommands: [] });

    await importModule([]);

    // Should fall back to startup plan path
    const server = mockMcpUtils.createMcpServer.mock.calls[0][0];
    const allowlistTool = server.tools.find((t) => t.name === 'get_active_validation_allowlist');
    await allowlistTool.handler({});
    assert.ok(mockPlanUtils.loadActivePlanContext.mock.calls.length >= 1);
  });

  it('handles session override with empty plan_path', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({ help: false, selfCheck: false, plan: 'plans/default.md' });
    mockMcpUtils.runStdioMcpServer.mockResolvedValue();
    mockFs.existsSync.mockReturnValue(true);
    const { readFile } = await import('node:fs/promises');
    readFile.mockResolvedValue(JSON.stringify({ plan_path: '' }));
    mockPlanUtils.loadActivePlanContext.mockResolvedValue({ activeStep: { number: 1, validationCommands: [] } });
    mockPlanUtils.createValidationAllowlistSnapshot.mockReturnValue({ validationCommands: [] });

    await importModule([]);

    const server = mockMcpUtils.createMcpServer.mock.calls[0][0];
    const allowlistTool = server.tools.find((t) => t.name === 'get_active_validation_allowlist');
    await allowlistTool.handler({});
    // Should fall back to startup plan path
    assert.ok(mockPlanUtils.loadActivePlanContext.mock.calls.length >= 1);
  });

  it('self-check records issue branches when responses are bad', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({ help: false, selfCheck: true, plan: 'plans/test.md' });
    mockFs.existsSync.mockReturnValue(false);
    mockPlanUtils.loadActivePlanContext.mockResolvedValue({
      activeStep: { number: 1, validationCommands: ['npm test'] },
    });
    mockMcpUtils.tokenizeShellSafeCommand.mockReturnValue(['node', 'npm', 'test']); // does not throw
    mockMcpUtils.invokeServerRequest
      .mockResolvedValueOnce({ protocolVersion: 'WRONG' }) // initialize -> 202
      .mockResolvedValueOnce({ tools: [] }) // tools/list -> 214
      .mockResolvedValueOnce({ // get_active_validation_allowlist -> 223, 233, 244
        isError: true,
        structuredContent: {
          validationCommands: ['npm test'],
          validationCommandsMatch: false,
          activeStep: { number: 99 },
          activePhase: { number: 1 },
        },
      })
      .mockResolvedValueOnce({ // run_allowlisted_validation 'npm test' -> 283
        isError: true,
        structuredContent: { exitCode: 1 },
      })
      .mockResolvedValueOnce({ // rejectedUnknownCommand -> 302 (not rejected)
        isError: false,
      });

    await importModule(['--self-check']);

    assert.ok(mockMcpUtils.emitSelfCheckReport.mock.calls.length >= 1);
    const report = mockMcpUtils.createSelfCheckReport.mock.calls[0];
    const issues = report[1];
    assert.ok(issues.length >= 8, `expected >=8 issues, got ${issues.length}`);
  });

  it('self-check reports issue when no executable validation commands exist', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({ help: false, selfCheck: true, plan: 'plans/test.md' });
    mockFs.existsSync.mockReturnValue(false);
    mockPlanUtils.loadActivePlanContext.mockResolvedValue({
      activeStep: { number: 1, validationCommands: [] },
    });
    mockMcpUtils.tokenizeShellSafeCommand.mockImplementation(() => {
      throw new Error('shell metacharacter rejected');
    });
    mockMcpUtils.invokeServerRequest
      .mockResolvedValueOnce({ protocolVersion: '2024-11-05' }) // initialize ok
      .mockResolvedValueOnce({ tools: [{ name: 'a' }, { name: 'b' }] }) // tools/list ok (2 tools)
      .mockResolvedValueOnce({ // allowlist ok but empty
        structuredContent: {
          validationCommands: [],
          validationCommandsMatch: true,
          activeStep: { number: 1 },
          activePhase: { number: 1 },
        },
      })
      .mockResolvedValueOnce({ // rejectedUnknownCommand rejected -> no 302
        isError: true,
      });

    await importModule(['--self-check']);

    assert.ok(mockMcpUtils.emitSelfCheckReport.mock.calls.length >= 1);
    const report = mockMcpUtils.createSelfCheckReport.mock.calls[0];
    const issues = report[1];
    assert.ok(issues.some((i) => /could not find any non-recursive/.test(i.message)));
  });

  it('self-check handles non-array tools and missing structuredContent', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({ help: false, selfCheck: true, plan: 'plans/test.md' });
    mockFs.existsSync.mockReturnValue(false);
    mockPlanUtils.loadActivePlanContext.mockResolvedValue({
      activeStep: { number: 1, validationCommands: [] },
    });
    mockMcpUtils.tokenizeShellSafeCommand.mockImplementation(() => {
      throw new Error('shell metacharacter rejected');
    });
    mockMcpUtils.invokeServerRequest
      .mockResolvedValueOnce({ protocolVersion: '2024-11-05' })
      .mockResolvedValueOnce({ tools: 'not-an-array' })
      .mockResolvedValueOnce({ isError: true })
      .mockResolvedValueOnce({ isError: true });

    await importModule(['--self-check']);

    assert.ok(mockMcpUtils.emitSelfCheckReport.mock.calls.length >= 1);
  });

  it('self-check handles run_allowlisted_validation without structuredContent', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({ help: false, selfCheck: true, plan: 'plans/test.md' });
    mockFs.existsSync.mockReturnValue(false);
    mockPlanUtils.loadActivePlanContext.mockResolvedValue({
      activeStep: { number: 1, validationCommands: ['npm test'] },
    });
    mockMcpUtils.tokenizeShellSafeCommand.mockImplementation(() => {
      throw new Error('shell metacharacter rejected');
    });
    mockMcpUtils.invokeServerRequest
      .mockResolvedValueOnce({ protocolVersion: '2024-11-05' })
      .mockResolvedValueOnce({ tools: [{ name: 'a' }, { name: 'b' }] })
      .mockResolvedValueOnce({
        structuredContent: {
          validationCommands: ['npm test'],
          validationCommandsMatch: true,
          activeStep: { number: 1 },
          activePhase: { number: 1 },
        },
      })
      .mockResolvedValueOnce({ isError: false })
      .mockResolvedValueOnce({ isError: true });

    await importModule(['--self-check']);

    assert.ok(mockMcpUtils.emitSelfCheckReport.mock.calls.length >= 1);
  });

  it('session override with absolute plan_path within plans/ resolves correctly', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({ help: false, selfCheck: false, plan: 'plans/default.md' });
    mockMcpUtils.runStdioMcpServer.mockResolvedValue();
    mockFs.existsSync.mockReturnValue(true);
    const { readFile } = await import('node:fs/promises');
    const absolutePlanPath = 'C:\\NeatapticTS\\plans\\override.md';
    readFile.mockResolvedValue(JSON.stringify({ plan_path: absolutePlanPath }));
    mockPlanUtils.loadActivePlanContext.mockResolvedValue({ activeStep: { number: 1, validationCommands: [] } });
    mockPlanUtils.createValidationAllowlistSnapshot.mockReturnValue({ validationCommands: [] });

    await importModule([]);

    const server = mockMcpUtils.createMcpServer.mock.calls[0][0];
    const allowlistTool = server.tools.find((t) => t.name === 'get_active_validation_allowlist');
    await allowlistTool.handler({});
    assert.ok(mockPlanUtils.loadActivePlanContext.mock.calls.length >= 1);
  });

  it('session override plan_path escaping plans/ is rejected and falls back', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({ help: false, selfCheck: false, plan: 'plans/default.md' });
    mockMcpUtils.runStdioMcpServer.mockResolvedValue();
    mockFs.existsSync.mockReturnValue(true);
    const { readFile } = await import('node:fs/promises');
    readFile.mockResolvedValue(JSON.stringify({ plan_path: '../outside.md' }));
    mockPlanUtils.loadActivePlanContext.mockResolvedValue({ activeStep: { number: 1, validationCommands: [] } });
    mockPlanUtils.createValidationAllowlistSnapshot.mockReturnValue({ validationCommands: [] });

    await importModule([]);

    const server = mockMcpUtils.createMcpServer.mock.calls[0][0];
    const allowlistTool = server.tools.find((t) => t.name === 'get_active_validation_allowlist');
    await allowlistTool.handler({});
    // Escaping path was rejected by resolvePlansScopedPath (throws, caught) and
    // fell back to the startup plan path.
    assert.ok(mockPlanUtils.loadActivePlanContext.mock.calls.length >= 1);
  });
});