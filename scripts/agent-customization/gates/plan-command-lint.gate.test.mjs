/**
 * @module plan-command-lint.gate.test
 * @description Coverage tests for plan-command-lint.gate.mjs.
 *
 * Tests exported functions directly (pure functions) and main() with mocked
 * dependencies for filesystem and child_process.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';

let mockReaddir;
let mockReadFile;
let mockExecStdout;
let mockExecError;
let mockParseArgsResult;

jest.unstable_mockModule('node:fs/promises', () => ({
  readdir: (...args) => mockReaddir(...args),
  readFile: (...args) => mockReadFile(...args),
}));

jest.unstable_mockModule('node:child_process', () => ({
  exec: (cmd, opts, cb) => {
    if (typeof opts === 'function') {
      cb = opts;
      opts = undefined;
    }
    if (cb) {
      if (mockExecError) {
        cb(mockExecError, { stdout: '', stderr: '' });
      } else {
        cb(null, { stdout: mockExecStdout ?? '', stderr: '' });
      }
    }
  },
}));

jest.unstable_mockModule('../customization-utils.mjs', () => ({
  parseArgs: (argv) =>
    mockParseArgsResult ?? { json: false, help: false, all: false, plan: null },
  repoRoot: path.resolve(),
}));

async function loadGate() {
  jest.resetModules();
  return import('./plan-command-lint.gate.mjs');
}

describe('plan-command-lint gate', () => {
  beforeEach(() => {
    mockReaddir = jest.fn(async () => []);
    mockReadFile = jest.fn(async () => '');
    mockExecStdout = '';
    mockExecError = null;
    mockParseArgsResult = null;
    jest.resetModules();
  });

  describe('stripInlineComment', () => {
    it('removes inline comment after whitespace #', async () => {
      const { stripInlineComment } = await loadGate();
      assert.equal(
        stripInlineComment('jest --config=x # comment'),
        'jest --config=x',
      );
    });

    it('returns trimmed string when no comment', async () => {
      const { stripInlineComment } = await loadGate();
      assert.equal(stripInlineComment('  jest --help  '), 'jest --help');
    });
  });

  describe('looksLikeShellCommand', () => {
    it('recognizes npx', async () => {
      const { looksLikeShellCommand } = await loadGate();
      assert.equal(looksLikeShellCommand('npx jest --help'), true);
    });

    it('recognizes npm', async () => {
      const { looksLikeShellCommand } = await loadGate();
      assert.equal(looksLikeShellCommand('npm test'), true);
    });

    it('recognizes node', async () => {
      const { looksLikeShellCommand } = await loadGate();
      assert.equal(looksLikeShellCommand('node script.mjs'), true);
    });

    it('recognizes jest', async () => {
      const { looksLikeShellCommand } = await loadGate();
      assert.equal(looksLikeShellCommand('jest --help'), true);
    });

    it('recognizes tsc', async () => {
      const { looksLikeShellCommand } = await loadGate();
      assert.equal(looksLikeShellCommand('tsc --help'), true);
    });

    it('recognizes eslint', async () => {
      const { looksLikeShellCommand } = await loadGate();
      assert.equal(looksLikeShellCommand('eslint --fix'), true);
    });

    it('recognizes prettier', async () => {
      const { looksLikeShellCommand } = await loadGate();
      assert.equal(looksLikeShellCommand('prettier --write'), true);
    });

    it('rejects unknown command', async () => {
      const { looksLikeShellCommand } = await loadGate();
      assert.equal(looksLikeShellCommand('python script.py'), false);
    });

    it('rejects empty string', async () => {
      const { looksLikeShellCommand } = await loadGate();
      assert.equal(looksLikeShellCommand(''), false);
    });

    it('rejects non-string', async () => {
      const { looksLikeShellCommand } = await loadGate();
      assert.equal(looksLikeShellCommand(null), false);
      assert.equal(looksLikeShellCommand(undefined), false);
      assert.equal(looksLikeShellCommand(123), false);
    });
  });

  describe('extractCommandsFromPlan', () => {
    it('extracts YAML list item commands', async () => {
      const { extractCommandsFromPlan } = await loadGate();
      const text = '- npx jest --help\n- npx tsc --help';
      const commands = extractCommandsFromPlan(text, 'plan.md');
      assert.equal(commands.length, 2);
      assert.equal(commands[0].command, 'npx jest --help');
      assert.equal(commands[0].source, 'yaml-list');
      assert.equal(commands[0].plan, 'plan.md');
    });

    it('extracts inline YAML value commands', async () => {
      const { extractCommandsFromPlan } = await loadGate();
      const text = 'validate: npx jest --config=jest.config.mjs';
      const commands = extractCommandsFromPlan(text, 'plan.md');
      assert.equal(commands.length, 1);
      assert.equal(commands[0].source, 'yaml-inline');
      assert.equal(commands[0].command, 'npx jest --config=jest.config.mjs');
    });

    it('extracts backtick commands', async () => {
      const { extractCommandsFromPlan } = await loadGate();
      const text = 'Run `npx eslint --help` to check flags.';
      const commands = extractCommandsFromPlan(text, 'plan.md');
      assert.equal(commands.length, 1);
      assert.equal(commands[0].source, 'backtick');
      assert.equal(commands[0].command, 'npx eslint --help');
    });

    it('strips inline comments from commands', async () => {
      const { extractCommandsFromPlan } = await loadGate();
      const text = '- npx jest --help # check flags';
      const commands = extractCommandsFromPlan(text, 'plan.md');
      assert.equal(commands.length, 1);
      assert.equal(commands[0].command, 'npx jest --help');
    });

    it('filters out non-shell commands', async () => {
      const { extractCommandsFromPlan } = await loadGate();
      const text = '- some text here\n- npx jest --help';
      const commands = extractCommandsFromPlan(text, 'plan.md');
      assert.equal(commands.length, 1);
      assert.equal(commands[0].command, 'npx jest --help');
    });

    it('deduplicates commands with same source and command', async () => {
      const { extractCommandsFromPlan } = await loadGate();
      const text = '- npx jest --help\n- npx jest --help';
      const commands = extractCommandsFromPlan(text, 'plan.md');
      assert.equal(commands.length, 1);
    });

    it('handles empty text', async () => {
      const { extractCommandsFromPlan } = await loadGate();
      const commands = extractCommandsFromPlan('', 'plan.md');
      assert.equal(commands.length, 0);
    });

    it('preserves raw command with comment in raw field', async () => {
      const { extractCommandsFromPlan } = await loadGate();
      const text = '- npx jest --help # comment';
      const commands = extractCommandsFromPlan(text, 'plan.md');
      assert.equal(commands[0].raw, 'npx jest --help # comment');
      assert.equal(commands[0].command, 'npx jest --help');
    });

    it('deduplicates across sources but keeps different sources', async () => {
      const { extractCommandsFromPlan } = await loadGate();
      const text = '- npx jest --help\nRun `npx jest --help` here.';
      const commands = extractCommandsFromPlan(text, 'plan.md');
      assert.equal(commands.length, 2);
      assert.equal(commands[0].source, 'yaml-list');
      assert.equal(commands[1].source, 'backtick');
    });
  });

  describe('validateCommand', () => {
    it('returns null when no long flags', async () => {
      const { validateCommand } = await loadGate();
      const context = { helpCache: new Map(), warnings: [] };
      const result = await validateCommand(
        {
          raw: 'npx jest',
          command: 'npx jest',
          source: 'yaml-list',
          plan: 'p.md',
        },
        context,
      );
      assert.equal(result, null);
    });

    it('returns null for unknown CLI', async () => {
      const { validateCommand } = await loadGate();
      const context = { helpCache: new Map(), warnings: [] };
      const result = await validateCommand(
        {
          raw: 'python --flag',
          command: 'python --flag',
          source: 'yaml-list',
          plan: 'p.md',
        },
        context,
      );
      assert.equal(result, null);
    });

    it('returns null for npx wrapper with no CLI (flags only)', async () => {
      const { validateCommand } = await loadGate();
      const context = { helpCache: new Map(), warnings: [] };
      const result = await validateCommand(
        {
          raw: 'npx --some-flag',
          command: 'npx --some-flag',
          source: 'yaml-list',
          plan: 'p.md',
        },
        context,
      );
      assert.equal(result, null);
    });

    it('returns null for npm wrapper with no CLI (flags only)', async () => {
      const { validateCommand } = await loadGate();
      const context = { helpCache: new Map(), warnings: [] };
      const result = await validateCommand(
        {
          raw: 'npm --flag',
          command: 'npm --flag',
          source: 'yaml-list',
          plan: 'p.md',
        },
        context,
      );
      assert.equal(result, null);
    });

    it('returns null when flags are valid in help', async () => {
      mockExecStdout = '--config <path>  Path to config\n--json  JSON output';
      const { validateCommand } = await loadGate();
      const context = { helpCache: new Map(), warnings: [] };
      const result = await validateCommand(
        {
          raw: 'npx jest --config=x --json',
          command: 'npx jest --config=x --json',
          source: 'yaml-list',
          plan: 'p.md',
        },
        context,
      );
      assert.equal(result, null);
    });

    it('returns issue when flags are NOT in help', async () => {
      mockExecStdout = '--config <path>  Path to config';
      const { validateCommand } = await loadGate();
      const context = { helpCache: new Map(), warnings: [] };
      const result = await validateCommand(
        {
          raw: 'npx jest --badflag',
          command: 'npx jest --badflag',
          source: 'yaml-list',
          plan: 'p.md',
        },
        context,
      );
      assert.ok(result);
      assert.equal(result.cli, 'jest');
      assert.ok(result.invalidFlags.includes('--badflag'));
    });

    it('returns null when fetchHelp returns empty', async () => {
      mockExecError = new Error('command not found');
      const { validateCommand } = await loadGate();
      const context = { helpCache: new Map(), warnings: [] };
      const result = await validateCommand(
        {
          raw: 'npx jest --someflag',
          command: 'npx jest --someflag',
          source: 'yaml-list',
          plan: 'p.md',
        },
        context,
      );
      assert.equal(result, null);
      assert.ok(context.warnings.length > 0);
    });

    it('handles npx wrapper with flags before CLI', async () => {
      mockExecStdout = '--silent';
      const { validateCommand } = await loadGate();
      const context = { helpCache: new Map(), warnings: [] };
      const result = await validateCommand(
        {
          raw: 'npx --silent jest --silent',
          command: 'npx --silent jest --silent',
          source: 'yaml-list',
          plan: 'p.md',
        },
        context,
      );
      assert.equal(result, null);
    });

    it('strips =value from flags when checking help', async () => {
      mockExecStdout = '--config <path>';
      const { validateCommand } = await loadGate();
      const context = { helpCache: new Map(), warnings: [] };
      const result = await validateCommand(
        {
          raw: 'npx jest --config=foo.json',
          command: 'npx jest --config=foo.json',
          source: 'yaml-list',
          plan: 'p.md',
        },
        context,
      );
      assert.equal(result, null);
    });
  });

  describe('fetchHelp', () => {
    it('returns cached help on cache hit', async () => {
      const { fetchHelp } = await loadGate();
      const context = {
        helpCache: new Map([['jest', 'cached help']]),
        warnings: [],
      };
      const result = await fetchHelp('jest', context);
      assert.equal(result, 'cached help');
    });

    it('returns empty string for unknown CLI and records warning', async () => {
      const { fetchHelp } = await loadGate();
      const context = { helpCache: new Map(), warnings: [] };
      const result = await fetchHelp('unknown-cli', context);
      assert.equal(result, '');
      assert.ok(context.warnings.some((w) => w.includes('Unknown CLI')));
      assert.equal(context.helpCache.get('unknown-cli'), '');
    });

    it('fetches and caches help on exec success', async () => {
      mockExecStdout = 'jest help output';
      const { fetchHelp } = await loadGate();
      const context = { helpCache: new Map(), warnings: [] };
      const result = await fetchHelp('jest', context);
      assert.equal(result, 'jest help output');
      assert.equal(context.helpCache.get('jest'), 'jest help output');
    });

    it('returns empty string on exec failure and records warning', async () => {
      mockExecError = new Error('exec failed');
      const { fetchHelp } = await loadGate();
      const context = { helpCache: new Map(), warnings: [] };
      const result = await fetchHelp('jest', context);
      assert.equal(result, '');
      assert.ok(context.warnings.some((w) => w.includes('Could not retrieve')));
      assert.equal(context.helpCache.get('jest'), '');
    });
  });

  describe('resolvePlanFiles', () => {
    it('returns single file when planPath exists', async () => {
      mockReadFile = jest.fn(async () => 'plan content');
      const { resolvePlanFiles } = await loadGate();
      const context = { warnings: [] };
      const result = await resolvePlanFiles('plans/test.plans.md', context);
      assert.equal(result.length, 1);
      assert.equal(result[0].filePath, 'plans/test.plans.md');
      assert.equal(result[0].text, 'plan content');
    });

    it('returns empty array and warning when planPath file does not exist', async () => {
      mockReadFile = jest.fn(async () => {
        throw new Error('ENOENT');
      });
      const { resolvePlanFiles } = await loadGate();
      const context = { warnings: [] };
      const result = await resolvePlanFiles('plans/missing.plans.md', context);
      assert.equal(result.length, 0);
      assert.ok(
        context.warnings.some((w) => w.includes('Could not read plan file')),
      );
    });

    it('returns all plan files when planPath is null', async () => {
      mockReaddir = jest.fn(async () => [
        { isFile: () => true, name: 'a.plans.md' },
        { isFile: () => true, name: 'b.plans.md' },
        { isFile: () => true, name: 'not-plan.txt' },
        { isFile: () => false, name: 'dir.plans.md' },
      ]);
      mockReadFile = jest.fn(async () => 'content');
      const { resolvePlanFiles } = await loadGate();
      const context = { warnings: [] };
      const result = await resolvePlanFiles(null, context);
      assert.equal(result.length, 2);
      assert.equal(result[0].filePath, path.join('plans', 'a.plans.md'));
      assert.equal(result[1].filePath, path.join('plans', 'b.plans.md'));
    });

    it('returns empty array and warning when readdir fails', async () => {
      mockReaddir = jest.fn(async () => {
        throw new Error('ENOENT');
      });
      const { resolvePlanFiles } = await loadGate();
      const context = { warnings: [] };
      const result = await resolvePlanFiles(null, context);
      assert.equal(result.length, 0);
      assert.ok(
        context.warnings.some((w) => w.includes('Could not list plans')),
      );
    });
  });

  describe('runPlanCommandLintGate', () => {
    it('passes with note when no plan files found (specific path)', async () => {
      mockReadFile = jest.fn(async () => {
        throw new Error('ENOENT');
      });
      const { runPlanCommandLintGate } = await loadGate();
      const result = await runPlanCommandLintGate('plans/missing.plans.md');
      assert.equal(result.pass, true);
      assert.ok(result.evidence.note.includes('Plan file not found'));
      assert.equal(
        result.fixHint,
        'Verify the plan path: plans/missing.plans.md',
      );
    });

    it('passes with note when no plan files found (null path)', async () => {
      mockReaddir = jest.fn(async () => []);
      const { runPlanCommandLintGate } = await loadGate();
      const result = await runPlanCommandLintGate(null);
      assert.equal(result.pass, true);
      assert.ok(result.evidence.note.includes('No root-level .plans.md'));
      assert.equal(result.fixHint, null);
    });

    it('passes when plan has valid commands', async () => {
      mockReadFile = jest.fn(async () => '- npx jest --help');
      mockExecStdout = '--help  Show help';
      const { runPlanCommandLintGate } = await loadGate();
      const result = await runPlanCommandLintGate('plans/test.plans.md');
      assert.equal(result.pass, true);
      assert.equal(result.evidence.commandsChecked, 1);
      assert.equal(result.evidence.issues.length, 0);
    });

    it('fails when plan has invalid flags', async () => {
      mockReadFile = jest.fn(async () => '- npx jest --badflag');
      mockExecStdout = '--help  Show help';
      const { runPlanCommandLintGate } = await loadGate();
      const result = await runPlanCommandLintGate('plans/test.plans.md');
      assert.equal(result.pass, false);
      assert.equal(result.evidence.issues.length, 1);
      assert.ok(result.fixHint);
      assert.ok(result.fixHint.includes('--badflag'));
    });

    it('passes when plan has no commands', async () => {
      mockReadFile = jest.fn(async () => '# Just a title\nNo commands here.');
      const { runPlanCommandLintGate } = await loadGate();
      const result = await runPlanCommandLintGate('plans/test.plans.md');
      assert.equal(result.pass, true);
      assert.equal(result.evidence.commandsChecked, 0);
    });

    it('runPlanCommandLintGate() with no args uses DEFAULT_PLAN', async () => {
      mockReadFile = jest.fn(async () => '# No commands');
      const { runPlanCommandLintGate } = await loadGate();
      const result = await runPlanCommandLintGate();
      assert.equal(result.pass, true);
    });
  });

  describe('main', () => {
    it('--help prints usage and exits 0', async () => {
      mockParseArgsResult = { json: false, help: true, all: false, plan: null };
      const { main } = await loadGate();
      const logs = [];
      const originalLog = console.log;
      const originalExit = process.exit;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      process.exit = (code) => {
        const err = new Error(`EXIT:${code}`);
        err.code = code;
        throw err;
      };
      try {
        await main(['--help']);
      } catch (err) {
        assert.equal(err.message, 'EXIT:0');
      } finally {
        console.log = originalLog;
        process.exit = originalExit;
      }
      assert.ok(logs.some((l) => l.includes('plan-command-lint gate')));
    });

    it('no --plan, no --all → uses DEFAULT_PLAN', async () => {
      mockParseArgsResult = { json: true, help: false, all: false, plan: null };
      mockReadFile = jest.fn(async () => '- npx jest --help');
      mockExecStdout = '--help  Show help';
      const { main } = await loadGate();
      const logs = [];
      const originalLog = console.log;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      try {
        await main(['--json']);
      } finally {
        console.log = originalLog;
      }
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(
        parsed.evidence.scannedPlans[0],
        'plans/orchestration-fixes.plans.md',
      );
    });

    it('--plan=path → specific plan', async () => {
      mockParseArgsResult = {
        json: true,
        help: false,
        all: false,
        plan: 'plans/my-plan.plans.md',
      };
      mockReadFile = jest.fn(async () => '- npx jest --help');
      mockExecStdout = '--help  Show help';
      const { main } = await loadGate();
      const logs = [];
      const originalLog = console.log;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      try {
        await main(['--json', '--plan=plans/my-plan.plans.md']);
      } finally {
        console.log = originalLog;
      }
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.scannedPlans[0], 'plans/my-plan.plans.md');
    });

    it('--all → all plans', async () => {
      mockParseArgsResult = { json: true, help: false, all: true, plan: null };
      mockReaddir = jest.fn(async () => [
        { isFile: () => true, name: 'a.plans.md' },
      ]);
      mockReadFile = jest.fn(async () => '- npx jest --help');
      mockExecStdout = '--help  Show help';
      const { main } = await loadGate();
      const logs = [];
      const originalLog = console.log;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      try {
        await main(['--json', '--all']);
      } finally {
        console.log = originalLog;
      }
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.scannedPlans.length, 1);
    });

    it('text output includes PASS', async () => {
      mockParseArgsResult = {
        json: false,
        help: false,
        all: false,
        plan: null,
      };
      mockReadFile = jest.fn(async () => '# No commands');
      const { main } = await loadGate();
      const logs = [];
      const originalLog = console.log;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      try {
        await main([]);
      } finally {
        console.log = originalLog;
      }
      assert.ok(logs.some((l) => l.includes('PASS')));
    });

    it('main() with no args uses process.argv.slice(2) default', async () => {
      const { main } = await loadGate();
      const logs = [];
      const originalLog = console.log;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      try {
        await main();
      } finally {
        console.log = originalLog;
      }
      assert.ok(logs.some((l) => l.includes('PASS')));
    });

    it('text output includes FAIL and fixHint', async () => {
      mockParseArgsResult = {
        json: false,
        help: false,
        all: false,
        plan: 'plans/test.plans.md',
      };
      mockReadFile = jest.fn(async () => '- npx jest --badflag');
      mockExecStdout = '--help  Show help';
      const { main } = await loadGate();
      const logs = [];
      const originalLog = console.log;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      try {
        await main(['--plan=plans/test.plans.md']);
      } finally {
        console.log = originalLog;
      }
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.ok(logs.some((l) => l.includes('fixHint:')));
    });

    it('text output includes warnings', async () => {
      mockParseArgsResult = {
        json: false,
        help: false,
        all: false,
        plan: 'plans/missing.plans.md',
      };
      mockReadFile = jest.fn(async () => {
        throw new Error('ENOENT');
      });
      const { main } = await loadGate();
      const logs = [];
      const originalLog = console.log;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      try {
        await main(['--plan=plans/missing.plans.md']);
      } finally {
        console.log = originalLog;
      }
      assert.ok(logs.some((l) => l.includes('WARNING:')));
    });

    it('--plan flag (token form) is detected', async () => {
      mockParseArgsResult = {
        json: true,
        help: false,
        all: false,
        plan: 'plans/token.plans.md',
      };
      mockReadFile = jest.fn(async () => '# No commands');
      const { main } = await loadGate();
      const logs = [];
      const originalLog = console.log;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      try {
        await main(['--json', '--plan', 'plans/token.plans.md']);
      } finally {
        console.log = originalLog;
      }
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.scannedPlans[0], 'plans/token.plans.md');
    });
  });

  describe('isMain guard', () => {
    it('covers process.argv[1] nullish coalescing in isMain check', async () => {
      const originalArgv1 = process.argv[1];
      process.argv[1] = undefined;
      try {
        const mod = await loadGate();
        assert.equal(typeof mod.main, 'function');
      } finally {
        process.argv[1] = originalArgv1;
      }
    });
  });
});
