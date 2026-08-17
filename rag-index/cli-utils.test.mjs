/**
 * @module cli-utils.test
 * @description Comprehensive tests for cli-utils.mjs targeting 100% coverage.
 */
import { jest } from '@jest/globals';
import path from 'node:path';
import {
  parseCliArgs,
  printHelp,
  writeJsonOrText,
  toRepoRelative,
  fail,
} from './cli-utils.mjs';

// ---------------------------------------------------------------------------
// parseCliArgs
// ---------------------------------------------------------------------------

describe('parseCliArgs', () => {
  it('parses --key=value', () => {
    const flags = parseCliArgs(['--alpha=0.5']);
    expect(flags.alpha).toBe('0.5');
  });

  it('parses --key value (separate)', () => {
    const flags = parseCliArgs(['--alpha', '0.5']);
    expect(flags.alpha).toBe('0.5');
  });

  it('parses positional arguments', () => {
    const flags = parseCliArgs(['pos1', '--flag', 'val', 'pos2']);
    expect(flags._).toEqual(['pos1', 'pos2']);
  });

  it('parses boolean flags (no value)', () => {
    const flags = parseCliArgs(['--json']);
    expect(flags.json).toBe(true);
  });

  it('overwrites non-repeatable flags (last wins)', () => {
    const flags = parseCliArgs(['--key=a', '--key=b']);
    expect(flags.key).toBe('b');
  });

  it('accumulates repeatable flags with first occurrence as string', () => {
    const flags = parseCliArgs(['--condition=a'], {
      repeatableFlags: ['condition'],
    });
    expect(flags.condition).toBe('a');
  });

  it('accumulates repeatable flags into array on second occurrence', () => {
    const flags = parseCliArgs(['--condition=a', '--condition=b'], {
      repeatableFlags: ['condition'],
    });
    expect(flags.condition).toEqual(['a', 'b']);
  });

  it('accumulates repeatable flags with three occurrences', () => {
    const flags = parseCliArgs(
      ['--condition=a', '--condition=b', '--condition=c'],
      { repeatableFlags: ['condition'] },
    );
    expect(flags.condition).toEqual(['a', 'b', 'c']);
  });

  it('handles repeatable flags mixing = and space syntax', () => {
    const flags = parseCliArgs(['--condition=a', '--condition', 'b'], {
      repeatableFlags: ['condition'],
    });
    expect(flags.condition).toEqual(['a', 'b']);
  });

  it('handles default options (no options parameter)', () => {
    const flags = parseCliArgs(['--key=val']);
    expect(flags.key).toBe('val');
  });

  it('handles --key followed by another --flag (boolean for first)', () => {
    const flags = parseCliArgs(['--alpha', '--json']);
    expect(flags.alpha).toBe(true);
    expect(flags.json).toBe(true);
  });

  it('handles empty argv', () => {
    const flags = parseCliArgs([]);
    expect(flags._).toEqual([]);
  });

  it('handles argument that is just --', () => {
    const flags = parseCliArgs(['--']);
    expect(flags['']).toBe(true);
  });

  it('handles nullish repeatableFlags via ??', () => {
    const flags = parseCliArgs(['--key=val'], { repeatableFlags: undefined });
    expect(flags.key).toBe('val');
  });
});

// ---------------------------------------------------------------------------
// printHelp
// ---------------------------------------------------------------------------

describe('printHelp', () => {
  it('prints formatted help text', () => {
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    printHelp({
      title: 'Test CLI',
      usage: 'node test.mjs [options]',
      options: ['--json   JSON output', '--help   Show help'],
    });
    expect(logSpy).toHaveBeenCalledTimes(1);
    const output = logSpy.mock.calls[0][0];
    expect(output).toContain('Test CLI');
    expect(output).toContain('Usage: node test.mjs [options]');
    expect(output).toContain('Options:');
    expect(output).toContain('--json   JSON output');
    expect(output).toContain('--help   Show help');
    logSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// writeJsonOrText
// ---------------------------------------------------------------------------

describe('writeJsonOrText', () => {
  it('writes JSON when json=true', () => {
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    const payload = { pass: true, score: 0.5 };
    writeJsonOrText(payload, true, (p) => `text: ${p.score}`);
    expect(logSpy).toHaveBeenCalledTimes(1);
    expect(logSpy.mock.calls[0][0]).toBe(JSON.stringify(payload, null, 2));
    logSpy.mockRestore();
  });

  it('writes formatted text when json=false', () => {
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    const payload = { pass: true, score: 0.5 };
    writeJsonOrText(payload, false, (p) => `text: ${p.score}`);
    expect(logSpy).toHaveBeenCalledTimes(1);
    expect(logSpy.mock.calls[0][0]).toBe('text: 0.5');
    logSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// toRepoRelative
// ---------------------------------------------------------------------------

describe('toRepoRelative', () => {
  it('converts absolute path to repo-relative with forward slashes', () => {
    const result = toRepoRelative(path.join(process.cwd(), 'src', 'foo.ts'));
    expect(result).toBe('src/foo.ts');
  });

  it('handles path with subdirectory', () => {
    const result = toRepoRelative(
      path.join(process.cwd(), 'rag-index', 'eval-runner.mjs'),
    );
    expect(result).toBe('rag-index/eval-runner.mjs');
  });
});

// ---------------------------------------------------------------------------
// fail
// ---------------------------------------------------------------------------

describe('fail', () => {
  it('logs JSON error and sets exitCode when json=true', () => {
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    const origExitCode = process.exitCode;
    process.exitCode = 0;
    fail('something went wrong', true, { code: 42 });
    expect(logSpy).toHaveBeenCalledTimes(1);
    const parsed = JSON.parse(logSpy.mock.calls[0][0]);
    expect(parsed.pass).toBe(false);
    expect(parsed.ok).toBe(false);
    expect(parsed.error).toBe('something went wrong');
    expect(parsed.code).toBe(42);
    expect(process.exitCode).toBe(1);
    logSpy.mockRestore();
    process.exitCode = origExitCode;
  });

  it('logs JSON error without extra details', () => {
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    const origExitCode = process.exitCode;
    process.exitCode = 0;
    fail('error message', true);
    expect(logSpy).toHaveBeenCalledTimes(1);
    const parsed = JSON.parse(logSpy.mock.calls[0][0]);
    expect(parsed.error).toBe('error message');
    expect(process.exitCode).toBe(1);
    logSpy.mockRestore();
    process.exitCode = origExitCode;
  });

  it('logs to stderr when json=false', () => {
    const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    const origExitCode = process.exitCode;
    process.exitCode = 0;
    fail('error message', false);
    expect(errorSpy).toHaveBeenCalledTimes(1);
    expect(errorSpy.mock.calls[0][0]).toBe('error message');
    expect(process.exitCode).toBe(1);
    errorSpy.mockRestore();
    process.exitCode = origExitCode;
  });

  it('defaults json to false and details to empty', () => {
    const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    const origExitCode = process.exitCode;
    process.exitCode = 0;
    fail('default error');
    expect(errorSpy).toHaveBeenCalledTimes(1);
    expect(process.exitCode).toBe(1);
    errorSpy.mockRestore();
    process.exitCode = origExitCode;
  });
});