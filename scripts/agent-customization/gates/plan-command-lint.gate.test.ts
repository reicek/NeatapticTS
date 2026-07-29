/**
 * @module plan-command-lint.gate.test
 * @description Green tests for the plan-command-lint gate contract.
 *
 * The plan-command-lint gate validates shell commands referenced in plan YAML
 * validation lists against the actual project CLI (e.g. `jest --help`) so that
 * flag drift (such as `--testPathPattern` vs `--testPathPatterns`) is caught
 * before plan updates are accepted. It must tolerate comments and path-only
 * entries that may appear in validation lists.
 */
import {
  describe,
  expect,
  it,
  jest,
  beforeEach,
  afterEach,
} from '@jest/globals';
import path from 'node:path';
import { exec } from 'node:child_process';
import { readFile, readdir } from 'node:fs/promises';

jest.mock('node:child_process', () => ({
  exec: jest.fn(),
}));

jest.mock('node:fs/promises', () => ({
  readFile: jest.fn(),
  readdir: jest.fn(),
}));

interface LintContext {
  helpCache: Map<string, string>;
  warnings: string[];
}

interface LintCommandInfo {
  raw: string;
  command: string;
  source: string;
  plan: string;
}

interface LintIssue {
  plan: string;
  source: string;
  command: string;
  cli: string;
  invalidFlags: string[];
}

interface LintReport {
  pass: boolean;
  evidence: Record<string, unknown>;
  fixHint: string | null;
  owner: string;
}

interface PlanFile {
  filePath: string;
  text: string;
}

interface GateModule {
  runPlanCommandLintGate: (planPath?: string | null) => Promise<LintReport>;
  main: (argv?: string[]) => Promise<LintReport>;
  resolvePlanFiles: (
    planPath: string | null,
    context: LintContext,
  ) => Promise<PlanFile[]>;
  extractCommandsFromPlan: (text: string, plan: string) => LintCommandInfo[];
  validateCommand: (
    command: LintCommandInfo,
    context: LintContext,
  ) => Promise<LintIssue | null>;
  fetchHelp: (cli: string, context: LintContext) => Promise<string>;
  looksLikeShellCommand: (cmd: string) => boolean;
  stripInlineComment: (cmd: string) => string;
}

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const DEFAULT_PLAN = 'plans/orchestration-fixes.plans.md';

const mockExec = exec as unknown as jest.Mock<any>;
const mockReadFile = readFile as unknown as jest.Mock<any>;
const mockReaddir = readdir as unknown as jest.Mock<any>;

const DEFAULT_HELP: Record<string, string> = {
  jest: [
    '--help',
    '--config',
    '--testPathPattern',
    '--runInBand',
    '--coverage',
    '--collect-coverage',
    '--verbose',
    '--no-cache',
    '--selectProjects',
    '--testPathIgnorePatterns',
  ].join(' '),
  tsc: ['--help', '--all', '--skipLibCheck', '--noEmit'].join(' '),
  eslint: '--help',
  prettier: ['--help', '--check', '--write'].join(' '),
};

function stubExec(helpByCli: Record<string, string> = DEFAULT_HELP) {
  mockExec.mockImplementation((...args: any[]) => {
    const cmd = args[0] as string;
    const callback = args[2] as (
      err: Error | null,
      result?: { stdout: string; stderr: string },
    ) => void;
    const parts = cmd.trim().split(/\s+/);
    const cli = parts[1];
    if (cli && helpByCli[cli] !== undefined) {
      callback(null, { stdout: helpByCli[cli], stderr: '' });
      return;
    }
    callback(new Error(`No exec stub for ${cmd}`));
  });
}

function failExec() {
  mockExec.mockImplementation((...args: any[]) => {
    const cmd = args[0] as string;
    const callback = args[2] as (
      err: Error | null,
      result?: { stdout: string; stderr: string },
    ) => void;
    callback(new Error(`exec failed: ${cmd}`));
  });
}

function emptyExec() {
  mockExec.mockImplementation((...args: any[]) => {
    const callback = args[2] as (
      err: Error | null,
      result?: { stdout: string; stderr: string },
    ) => void;
    callback(null, { stdout: '', stderr: '' });
  });
}

function makeContext(): LintContext {
  return { helpCache: new Map(), warnings: [] };
}

function planText(commands: string[]) {
  return [
    '# Test Plan',
    '',
    '## Validation',
    '',
    ...commands.map((cmd) => `- '${cmd}'`),
    '',
  ].join('\n');
}

function mockDirent(name: string, isFile: boolean) {
  return { name, isFile: () => isFile };
}

/**
 * Lazily load the ESM gate module inside each test so Jest does not try to
 * transform its top-level await through the mjs-to-cjs transformer.
 */
async function loadGate(): Promise<GateModule> {
  // @ts-ignore - tested module is authored in plain ESM without a declaration file.
  return import('./plan-command-lint.gate.mjs');
}

describe('plan-command-lint.gate.mjs', () => {
  beforeAll(() => {
    // Force the module-level `process.argv[1] ?? ''` fallback to be evaluated
    // so that the default-argument branch is covered by code coverage.
    const originalScript = process.argv[1];
    process.argv[1] = undefined as unknown as string;
    return () => {
      process.argv[1] = originalScript;
    };
  });

  beforeEach(() => {
    jest.clearAllMocks();
    process.exitCode = 0;
    mockReadFile.mockResolvedValue('# Plan\n');
    mockReaddir.mockResolvedValue([]);
  });

  afterEach(() => {
    process.exitCode = 0;
  });

  it('exports the core gate orchestration functions', async () => {
    const gate = await loadGate();

    expect(typeof gate.runPlanCommandLintGate).toBe('function');
    expect(typeof gate.main).toBe('function');
    expect(typeof gate.resolvePlanFiles).toBe('function');
  });

  it('exports the command extraction and validation helpers', async () => {
    const gate = await loadGate();

    expect(typeof gate.extractCommandsFromPlan).toBe('function');
    expect(typeof gate.validateCommand).toBe('function');
    expect(typeof gate.fetchHelp).toBe('function');
  });

  it('exports the string utility helpers', async () => {
    const gate = await loadGate();

    expect(typeof gate.looksLikeShellCommand).toBe('function');
    expect(typeof gate.stripInlineComment).toBe('function');
  });

  describe('runPlanCommandLintGate', () => {
    it('passes when the plan contains no commands', async () => {
      const { runPlanCommandLintGate } = await loadGate();
      mockReadFile.mockResolvedValue('# Empty plan\n');

      const result = (await runPlanCommandLintGate(
        'plans/test.plans.md',
      )) as LintReport;

      expect(result.pass).toBe(true);
      expect(result.evidence.commandsChecked).toBe(0);
      expect(result.evidence.issues).toEqual([]);
    });

    it('reports no warnings when the plan contains no commands', async () => {
      const { runPlanCommandLintGate } = await loadGate();
      mockReadFile.mockResolvedValue('# Empty plan\n');

      const result = (await runPlanCommandLintGate(
        'plans/test.plans.md',
      )) as LintReport;

      expect(result.evidence.warnings).toEqual([]);
    });

    it('uses the default plan when no planPath is provided', async () => {
      const { runPlanCommandLintGate } = await loadGate();
      const expectedPath = path.join(REPO_ROOT, DEFAULT_PLAN);
      mockReadFile.mockResolvedValue('# Default plan\n');
      stubExec();

      await runPlanCommandLintGate();

      expect(mockReadFile).toHaveBeenCalledWith(expectedPath, 'utf8');
    });

    it('warns and passes when the plan file is missing', async () => {
      const { runPlanCommandLintGate } = await loadGate();
      const error = new Error('ENOENT: no such file');
      (error as NodeJS.ErrnoException).code = 'ENOENT';
      mockReadFile.mockRejectedValue(error);

      const result = (await runPlanCommandLintGate(
        'plans/missing.plans.md',
      )) as LintReport;

      expect(result.pass).toBe(true);
      expect(result.evidence.scannedPlans).toEqual([]);
      expect(result.evidence.note).toBe(
        'Plan file not found: plans/missing.plans.md',
      );
    });

    it('records a warning when the plan file is missing', async () => {
      const { runPlanCommandLintGate } = await loadGate();
      const error = new Error('ENOENT: no such file');
      (error as NodeJS.ErrnoException).code = 'ENOENT';
      mockReadFile.mockRejectedValue(error);

      const result = (await runPlanCommandLintGate(
        'plans/missing.plans.md',
      )) as LintReport;

      expect(result.evidence.warnings).toEqual(
        expect.arrayContaining([
          expect.stringContaining(
            'Could not read plan file plans/missing.plans.md',
          ),
        ]),
      );
    });

    it('scans every .plans.md file when planPath is null', async () => {
      const { runPlanCommandLintGate } = await loadGate();
      mockReaddir.mockResolvedValue([
        mockDirent('alpha.plans.md', true),
        mockDirent('beta.plans.md', true),
        mockDirent('other.md', true),
        mockDirent('folder', false),
      ]);
      mockReadFile.mockImplementation((...args: any[]) => {
        const filePath = args[0] as string;
        if (filePath.includes('alpha.plans.md')) {
          return Promise.resolve(planText(['npx jest --help']));
        }
        if (filePath.includes('beta.plans.md')) {
          return Promise.resolve(planText(['npx tsc --noEmit']));
        }
        return Promise.resolve('# Plan\n');
      });
      stubExec();

      const result = (await runPlanCommandLintGate(null)) as LintReport;

      expect(result.pass).toBe(true);
      expect(result.evidence.scannedPlans).toEqual([
        path.join('plans', 'alpha.plans.md'),
        path.join('plans', 'beta.plans.md'),
      ]);
      expect(result.evidence.commandsChecked).toBe(2);
    });

    it('warns and passes when the plans directory cannot be read', async () => {
      const { runPlanCommandLintGate } = await loadGate();
      mockReaddir.mockRejectedValue(new Error('permission denied'));

      const result = (await runPlanCommandLintGate(null)) as LintReport;

      expect(result.pass).toBe(true);
      expect(result.evidence.note).toBe('No root-level .plans.md files found.');
      expect(result.evidence.warnings).toEqual(
        expect.arrayContaining([
          expect.stringContaining('Could not list plans directory'),
        ]),
      );
    });

    it('reports an invalid flag', async () => {
      const { runPlanCommandLintGate } = await loadGate();
      mockReadFile.mockResolvedValue(planText(['npx jest --bad-flag']));
      stubExec();

      const result = (await runPlanCommandLintGate(
        'plans/test.plans.md',
      )) as LintReport;

      expect(result.pass).toBe(false);
      expect(result.evidence.issues).toHaveLength(1);
      const issue = result.evidence.issues as LintIssue[];
      expect(issue[0].cli).toBe('jest');
    });

    it('describes the invalid flag in the issue and fixHint', async () => {
      const { runPlanCommandLintGate } = await loadGate();
      mockReadFile.mockResolvedValue(planText(['npx jest --bad-flag']));
      stubExec();

      const result = (await runPlanCommandLintGate(
        'plans/test.plans.md',
      )) as LintReport;

      const issue = result.evidence.issues as LintIssue[];
      expect(issue[0].invalidFlags).toEqual(['--bad-flag']);
      expect(result.fixHint).toContain('--bad-flag');
    });

    it('passes when all long flags are valid', async () => {
      const { runPlanCommandLintGate } = await loadGate();
      mockReadFile.mockResolvedValue(
        planText([
          'npx jest --help --config=jest.config.mjs',
          'npx tsc --noEmit --skipLibCheck',
        ]),
      );
      stubExec();

      const result = (await runPlanCommandLintGate(
        'plans/test.plans.md',
      )) as LintReport;

      expect(result.pass).toBe(true);
      expect(result.evidence.issues).toEqual([]);
    });

    it('ignores commands without long flags', async () => {
      const { runPlanCommandLintGate } = await loadGate();
      mockReadFile.mockResolvedValue(planText(['npx jest', 'node script.js']));

      const result = (await runPlanCommandLintGate(
        'plans/test.plans.md',
      )) as LintReport;

      expect(result.pass).toBe(true);
      expect(result.evidence.commandsChecked).toBe(2);
      expect(result.evidence.issues).toEqual([]);
    });

    it('deduplicates repeated commands', async () => {
      const { runPlanCommandLintGate } = await loadGate();
      mockReadFile.mockResolvedValue(
        planText(['npx jest --help', '- "npx jest --help"']),
      );
      stubExec();

      const result = (await runPlanCommandLintGate(
        'plans/test.plans.md',
      )) as LintReport;

      expect(result.evidence.commandsChecked).toBe(1);
    });

    it('records a warning when help cannot be retrieved', async () => {
      const { runPlanCommandLintGate } = await loadGate();
      mockReadFile.mockResolvedValue(planText(['npx jest --help']));
      failExec();

      const result = (await runPlanCommandLintGate(
        'plans/test.plans.md',
      )) as LintReport;

      expect(result.pass).toBe(true);
      expect(result.evidence.warnings).toEqual(
        expect.arrayContaining([
          expect.stringContaining('Could not retrieve help for jest'),
        ]),
      );
    });

    it('does not flag commands when help text is empty', async () => {
      const { runPlanCommandLintGate } = await loadGate();
      mockReadFile.mockResolvedValue(planText(['npx jest --bad-flag']));
      emptyExec();

      const result = (await runPlanCommandLintGate(
        'plans/test.plans.md',
      )) as LintReport;

      expect(result.pass).toBe(true);
      expect(result.evidence.issues).toEqual([]);
    });
  });

  describe('main CLI surface', () => {
    it('shows help and exits when --help is passed', async () => {
      const { main } = await loadGate();
      const exitSpy = jest.spyOn(process, 'exit').mockImplementation(() => {
        throw new Error('exit');
      });
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});

      await expect(main(['--help'])).rejects.toThrow('exit');

      expect(exitSpy).toHaveBeenCalledWith(0);
      expect(logSpy).toHaveBeenCalledWith(
        expect.stringContaining('plan-command-lint gate'),
      );

      exitSpy.mockRestore();
      logSpy.mockRestore();
    });

    it('emits JSON output with --json', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      mockReadFile.mockResolvedValue(planText(['npx jest --help']));
      stubExec();

      const result = (await main([
        '--json',
        '--plan=plans/test.plans.md',
      ])) as LintReport;

      expect(result.pass).toBe(true);
      expect(logSpy).toHaveBeenCalledTimes(1);

      logSpy.mockRestore();
    });

    it('logs a valid JSON report with --json', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      mockReadFile.mockResolvedValue(planText(['npx jest --help']));
      stubExec();

      await main(['--json', '--plan=plans/test.plans.md']);

      const logged = JSON.parse(
        logSpy.mock.calls[0][0] as string,
      ) as LintReport;
      expect(logged.pass).toBe(true);
      expect(logged.owner).toBe('plan-command-lint.gate.mjs');

      logSpy.mockRestore();
    });

    it('emits plain PASS output without --json', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      mockReadFile.mockResolvedValue(planText(['npx jest --help']));
      stubExec();

      await main(['--plan=plans/test.plans.md']);

      expect(logSpy).toHaveBeenCalledWith('PASS', 'plan-command-lint gate');
      expect(process.exitCode).toBe(0);

      logSpy.mockRestore();
    });

    it('emits FAIL, fixHint, and sets exitCode when issues are found', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      mockReadFile.mockResolvedValue(planText(['npx jest --bad-flag']));
      stubExec();

      await main(['--plan=plans/test.plans.md']);

      expect(logSpy).toHaveBeenCalledWith('FAIL', 'plan-command-lint gate');
      expect(logSpy).toHaveBeenCalledWith(
        'fixHint:',
        expect.stringContaining('--bad-flag'),
      );
      expect(process.exitCode).toBe(1);

      logSpy.mockRestore();
    });

    it('prints warnings in plain mode when help retrieval fails', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      mockReadFile.mockResolvedValue(planText(['npx jest --help']));
      failExec();

      await main(['--plan=plans/test.plans.md']);

      expect(logSpy).toHaveBeenCalledWith('PASS', 'plan-command-lint gate');
      expect(logSpy).toHaveBeenCalledWith(
        'WARNING:',
        expect.stringContaining('Could not retrieve help for jest'),
      );

      logSpy.mockRestore();
    });

    it('scans all plans with --all', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      mockReaddir.mockResolvedValue([
        mockDirent('alpha.plans.md', true),
        mockDirent('beta.plans.md', true),
      ]);
      mockReadFile.mockImplementation((...args: any[]) => {
        const filePath = args[0] as string;
        if (filePath.includes('alpha.plans.md')) {
          return Promise.resolve(planText(['npx jest --help']));
        }
        if (filePath.includes('beta.plans.md')) {
          return Promise.resolve(planText(['npx tsc --noEmit']));
        }
        return Promise.resolve('# Plan\n');
      });
      stubExec();

      const result = (await main(['--all', '--json'])) as LintReport;

      expect(result.pass).toBe(true);
      expect(result.evidence.scannedPlans).toEqual(
        expect.arrayContaining([
          path.join('plans', 'alpha.plans.md'),
          path.join('plans', 'beta.plans.md'),
        ]),
      );

      logSpy.mockRestore();
    });

    it('uses process.argv when no argv is passed', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      mockReadFile.mockResolvedValue(planText(['npx jest --help']));
      stubExec();

      const originalArgv = process.argv;
      process.argv = [
        'node',
        'plan-command-lint.gate.mjs',
        '--json',
        '--plan=plans/test.plans.md',
      ];

      try {
        const result = (await main()) as LintReport;
        expect(result.pass).toBe(true);
      } finally {
        process.argv = originalArgv;
        logSpy.mockRestore();
      }
    });

    it('falls back to the default plan when no --plan or --all is given', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      mockReadFile.mockResolvedValue('# Default plan\n');
      stubExec();

      const result = (await main([])) as LintReport;

      expect(result.pass).toBe(true);
      expect(mockReadFile).toHaveBeenCalledWith(
        path.join(REPO_ROOT, DEFAULT_PLAN),
        'utf8',
      );

      logSpy.mockRestore();
    });
  });

  describe('resolvePlanFiles', () => {
    it('reads a single plan file', async () => {
      const { resolvePlanFiles } = await loadGate();
      const context = makeContext();
      mockReadFile.mockResolvedValue('plan text');

      const files = (await resolvePlanFiles(
        'plans/foo.plans.md',
        context,
      )) as PlanFile[];

      expect(files).toHaveLength(1);
      expect(files[0].filePath).toBe('plans/foo.plans.md');
      expect(files[0].text).toBe('plan text');
    });

    it('returns an empty list and a warning for a missing single plan', async () => {
      const { resolvePlanFiles } = await loadGate();
      const context = makeContext();
      mockReadFile.mockRejectedValue(new Error('ENOENT'));

      const files = (await resolvePlanFiles(
        'plans/missing.plans.md',
        context,
      )) as PlanFile[];

      expect(files).toEqual([]);
      expect(context.warnings).toHaveLength(1);
    });

    it('reads every .plans.md file in the plans directory', async () => {
      const { resolvePlanFiles } = await loadGate();
      const context = makeContext();
      mockReaddir.mockResolvedValue([
        mockDirent('a.plans.md', true),
        mockDirent('b.plans.md', true),
        mockDirent('c.md', true),
      ]);
      mockReadFile.mockResolvedValue('text');

      const files = (await resolvePlanFiles(null, context)) as PlanFile[];

      expect(files.map((f: PlanFile) => f.filePath)).toEqual([
        path.join('plans', 'a.plans.md'),
        path.join('plans', 'b.plans.md'),
      ]);
    });

    it('returns an empty list and a warning when readdir fails', async () => {
      const { resolvePlanFiles } = await loadGate();
      const context = makeContext();
      mockReaddir.mockRejectedValue(new Error('boom'));

      const files = (await resolvePlanFiles(null, context)) as PlanFile[];

      expect(files).toEqual([]);
      expect(context.warnings).toHaveLength(1);
    });
  });

  describe('extractCommandsFromPlan', () => {
    it('extracts commands from YAML lists, inline values, and backticks', async () => {
      const { extractCommandsFromPlan } = await loadGate();
      const text = [
        '# Plan',
        '',
        '- "npx jest --help"',
        "validation: 'npx tsc --noEmit'",
        'Run `npx eslint --help` now.',
        '- npx prettier --check .',
      ].join('\n');

      const commands = extractCommandsFromPlan(
        text,
        'plans/test.plans.md',
      ) as LintCommandInfo[];
      const rawCommands = commands.map((c: LintCommandInfo) => c.raw);

      expect(rawCommands).toEqual(
        expect.arrayContaining([
          'npx jest --help',
          'npx tsc --noEmit',
          'npx eslint --help',
          'npx prettier --check .',
        ]),
      );
    });

    it('deduplicates identical commands from the same source', async () => {
      const { extractCommandsFromPlan } = await loadGate();
      const text = ['- "npx jest --help"', '- "npx jest --help"'].join('\n');

      const commands = extractCommandsFromPlan(
        text,
        'plans/test.plans.md',
      ) as LintCommandInfo[];

      expect(commands).toHaveLength(1);
    });

    it('skips entries that do not look like shell commands', async () => {
      const { extractCommandsFromPlan } = await loadGate();
      const text = [
        '- "python script.py"',
        '- "bash do.sh"',
        '- "npx jest --help"',
      ].join('\n');

      const commands = extractCommandsFromPlan(
        text,
        'plans/test.plans.md',
      ) as LintCommandInfo[];

      expect(commands).toHaveLength(1);
      expect(commands[0].raw).toBe('npx jest --help');
    });
  });

  describe('looksLikeShellCommand', () => {
    it('returns true for recognized shell binaries', async () => {
      const { looksLikeShellCommand } = await loadGate();

      expect(looksLikeShellCommand('npx jest --help')).toBe(true);
      expect(looksLikeShellCommand('node script.js')).toBe(true);
      expect(looksLikeShellCommand('npm run build')).toBe(true);
    });

    it('returns false for empty or unrecognized commands', async () => {
      const { looksLikeShellCommand } = await loadGate();

      expect(looksLikeShellCommand('')).toBe(false);
      expect(looksLikeShellCommand('python script.py')).toBe(false);
      expect(looksLikeShellCommand('bash do.sh')).toBe(false);
    });
  });

  describe('stripInlineComment', () => {
    it('removes trailing comments from a command string', async () => {
      const { stripInlineComment } = await loadGate();

      expect(stripInlineComment("npx jest --help # run tests'")).toBe(
        'npx jest --help',
      );
    });
  });

  describe('validateCommand', () => {
    it('returns null for commands without long flags', async () => {
      const { validateCommand } = await loadGate();
      const context = makeContext();

      const issue = (await validateCommand(
        {
          raw: 'npx jest',
          command: 'npx jest',
          source: 'yaml-list',
          plan: 'plans/test.plans.md',
        },
        context,
      )) as LintIssue | null;

      expect(issue).toBeNull();
    });

    it('returns null for an unknown CLI', async () => {
      const { validateCommand } = await loadGate();
      const context = makeContext();

      const issue = (await validateCommand(
        {
          raw: 'node script.js --flag',
          command: 'node script.js --flag',
          source: 'yaml-list',
          plan: 'plans/test.plans.md',
        },
        context,
      )) as LintIssue | null;

      expect(issue).toBeNull();
    });

    it('unwraps npx and validates the wrapped CLI', async () => {
      const { validateCommand } = await loadGate();
      const context = makeContext();
      context.helpCache.set('jest', '--help --config');

      const issue = (await validateCommand(
        {
          raw: 'npx jest --help',
          command: 'npx jest --help',
          source: 'yaml-list',
          plan: 'plans/test.plans.md',
        },
        context,
      )) as LintIssue | null;

      expect(issue).toBeNull();
    });

    it('returns null when npx has no positional CLI', async () => {
      const { validateCommand } = await loadGate();
      const context = makeContext();

      const issue = (await validateCommand(
        {
          raw: 'npx --help',
          command: 'npx --help',
          source: 'yaml-list',
          plan: 'plans/test.plans.md',
        },
        context,
      )) as LintIssue | null;

      expect(issue).toBeNull();
    });

    it('returns null for npm wrappers that do not resolve to a known CLI', async () => {
      const { validateCommand } = await loadGate();
      const context = makeContext();

      const issue = (await validateCommand(
        {
          raw: 'npm run jest --help',
          command: 'npm run jest --help',
          source: 'yaml-list',
          plan: 'plans/test.plans.md',
        },
        context,
      )) as LintIssue | null;

      expect(issue).toBeNull();
    });

    it('reports an invalid long flag', async () => {
      const { validateCommand } = await loadGate();
      const context = makeContext();
      context.helpCache.set('jest', '--help --config');

      const issue = (await validateCommand(
        {
          raw: 'npx jest --bad-flag',
          command: 'npx jest --bad-flag',
          source: 'yaml-list',
          plan: 'plans/test.plans.md',
        },
        context,
      )) as LintIssue | null;

      expect(issue).not.toBeNull();
      expect((issue as LintIssue).invalidFlags).toEqual(['--bad-flag']);
    });

    it('returns null when help text is empty', async () => {
      const { validateCommand } = await loadGate();
      const context = makeContext();
      context.helpCache.set('jest', '');

      const issue = (await validateCommand(
        {
          raw: 'npx jest --bad-flag',
          command: 'npx jest --bad-flag',
          source: 'yaml-list',
          plan: 'plans/test.plans.md',
        },
        context,
      )) as LintIssue | null;

      expect(issue).toBeNull();
    });
  });

  describe('fetchHelp', () => {
    it('caches help text and returns it on subsequent calls', async () => {
      const { fetchHelp } = await loadGate();
      const context = makeContext();
      stubExec();

      const first = await fetchHelp('jest', context);
      const second = await fetchHelp('jest', context);

      expect(first).toBe(DEFAULT_HELP.jest);
      expect(second).toBe(first);
      expect(mockExec).toHaveBeenCalledTimes(1);
    });

    it('warns and returns an empty string for an unknown CLI', async () => {
      const { fetchHelp } = await loadGate();
      const context = makeContext();

      const help = await fetchHelp('unknown', context);

      expect(help).toBe('');
      expect(context.warnings).toEqual(
        expect.arrayContaining([
          'Unknown CLI "unknown"; cannot validate flags.',
        ]),
      );
    });

    it('warns and returns an empty string when exec fails', async () => {
      const { fetchHelp } = await loadGate();
      const context = makeContext();
      failExec();

      const help = await fetchHelp('jest', context);

      expect(help).toBe('');
      expect(context.warnings).toEqual(
        expect.arrayContaining([
          expect.stringContaining('Could not retrieve help for jest'),
        ]),
      );
    });
  });
});
