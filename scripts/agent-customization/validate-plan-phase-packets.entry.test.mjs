/**
 * @module validate-plan-phase-packets.entry.test
 * @description Direct coverage tests for validate-plan-phase-packets.mjs
 *   module-level help (107-118), main() (130-148), and entry-point guard (229-232).
 *   Uses cache-busting dynamic imports with manipulated process.argv.
 */
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const SCRIPT_PATH = path.resolve(
  'scripts',
  'agent-customization',
  'validate-plan-phase-packets.mjs',
);

describe('validate-plan-phase-packets entry-point coverage', () => {
  it('module-level --help prints usage, main() help returns, entry guard calls main', async () => {
    const originalArgv = process.argv;
    const originalExit = process.exit;
    const originalLog = console.log;

    const logChunks = [];
    process.argv = [process.execPath, SCRIPT_PATH, '--help'];
    // Mock process.exit as no-op so module evaluation continues past line 118
    process.exit = () => {};
    console.log = (...args) => {
      logChunks.push(args.join(' '));
    };

    try {
      // Cache-busting import re-executes module-level code with --help in argv.
      // Module-level help (107-118) runs printUsage + process.exit(0) (no-op).
      // Then entry guard (229) matches because argv[1] === scriptPath.
      // main() is called, options.help is true → main() help path (130-142)
      // calls printUsage again and returns.
      await import(
        `./validate-plan-phase-packets.mjs?help-entry=${Date.now()}`
      );
      // Allow async main().then() to settle
      await new Promise((resolve) => setTimeout(resolve, 500));
    } finally {
      process.argv = originalArgv;
      process.exit = originalExit;
      console.log = originalLog;
      process.exitCode = 0;
    }

    const output = logChunks.join('\n');
    // printUsage was called at least once (module-level or main)
    assert.ok(
      output.includes('Validate copy-pasteable plan phase/step packets.'),
      `Expected usage text in output: ${output.slice(0, 200)}`,
    );
  });

  it('main() reads plan, validates, and writes JSON report via entry guard', async () => {
    const originalArgv = process.argv;
    const originalWrite = process.stdout.write.bind(process.stdout);
    const originalErr = console.error;

    const stdoutChunks = [];
    process.argv = [process.execPath, SCRIPT_PATH, '--json'];
    process.stdout.write = (chunk) => {
      stdoutChunks.push(
        typeof chunk === 'string' ? chunk : chunk.toString('utf8'),
      );
      return true;
    };
    console.error = () => {};

    try {
      await import(
        `./validate-plan-phase-packets.mjs?entry-json=${Date.now()}`
      );
      // Allow async main() to complete (reads file, validates, writes report)
      await new Promise((resolve) => setTimeout(resolve, 2000));
    } finally {
      process.argv = originalArgv;
      process.stdout.write = originalWrite;
      console.error = originalErr;
      process.exitCode = 0;
    }

    const output = stdoutChunks.join('');
    const parsed = JSON.parse(output.trim());
    assert.equal(typeof parsed.ok, 'boolean');
    assert.ok(Array.isArray(parsed.issues));
  });

  it('main() sets exitCode to 1 when report.ok is false', async () => {
    const originalArgv = process.argv;
    const originalWrite = process.stdout.write.bind(process.stdout);
    const originalExitCode = process.exitCode;

    const fs = await import('node:fs/promises');
    const invalidPlanRel = 'test-invalid-plan-for-coverage.plans.md';
    const invalidPlanAbs = path.resolve(invalidPlanRel);
    // Minimal invalid plan: a WIP phase with incomplete YAML → report.ok = false
    const invalidPlan = [
      '## Implementation phases',
      '',
      '### Phase A — Test Phase [WIP]',
      '',
      '**Phase objective:** Test.',
      '**Stop conditions:** Done.',
      '**Required validation:** Tests pass.',
      '',
      '```yaml',
      'phase: A',
      'title: Test Phase',
      'status: [WIP]',
      'expansion: steps',
      '```',
      '',
      '## Validation gates',
      '',
      '- All tests pass.',
    ].join('\n');
    await fs.writeFile(invalidPlanAbs, invalidPlan, 'utf8');

    const stdoutChunks = [];
    process.argv = [
      process.execPath,
      SCRIPT_PATH,
      '--json',
      `--plan=${invalidPlanRel}`,
    ];
    process.stdout.write = (chunk) => {
      stdoutChunks.push(
        typeof chunk === 'string' ? chunk : chunk.toString('utf8'),
      );
      return true;
    };
    process.exitCode = undefined;

    let capturedExitCode = 0;
    try {
      await import(
        `./validate-plan-phase-packets.mjs?entry-fail=${Date.now()}`
      );
      // Allow async main() to complete (reads file, validates, writes report, sets exitCode)
      await new Promise((resolve) => setTimeout(resolve, 2000));
      capturedExitCode = process.exitCode;
    } finally {
      process.argv = originalArgv;
      process.stdout.write = originalWrite;
      process.exitCode = originalExitCode ?? 0;
      await fs.unlink(invalidPlanAbs).catch(() => {});
    }

    const output = stdoutChunks.join('');
    const parsed = JSON.parse(output.trim());
    assert.equal(parsed.ok, false);
    assert.equal(capturedExitCode, 1);
  });

  it('entry guard error handler logs error and sets exit code on main() failure', async () => {
    const originalArgv = process.argv;
    const originalErr = console.error;

    const errorMessages = [];
    process.argv = [
      process.execPath,
      SCRIPT_PATH,
      '--plan=plans/nonexistent-test-plan-12345.plans.md',
    ];
    console.error = (...args) => {
      errorMessages.push(args.map(String).join(' '));
    };

    try {
      await import(
        `./validate-plan-phase-packets.mjs?entry-error=${Date.now()}`
      );
      // Allow async main().catch() to settle
      await new Promise((resolve) => setTimeout(resolve, 2000));
    } finally {
      process.argv = originalArgv;
      console.error = originalErr;
      process.exitCode = 0;
    }

    // The .catch() handler at lines 231-232 should have logged the error
    assert.ok(
      errorMessages.length > 0,
      'Expected error to be logged by entry guard catch handler',
    );
  });
});
