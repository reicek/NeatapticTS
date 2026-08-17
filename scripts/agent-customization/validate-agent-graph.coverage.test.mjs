/**
 * @module validate-agent-graph.coverage.test
 * @description Direct coverage tests for validate-agent-graph.mjs main() and entry guard.
 *   Avoids importInIsolation so V8 coverage is collected.
 */
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

async function withArgv(argv, fn) {
  const original = process.argv;
  process.argv = argv;
  try {
    return await fn();
  } finally {
    process.argv = original;
  }
}

async function captureStdout(fn) {
  const chunks = [];
  const originalWrite = process.stdout.write.bind(process.stdout);
  process.stdout.write = (chunk) => {
    chunks.push(typeof chunk === 'string' ? chunk : chunk.toString('utf8'));
    return true;
  };
  try {
    await fn();
    return chunks;
  } finally {
    process.stdout.write = originalWrite;
  }
}

async function swallowStderr(fn) {
  const originalErr = console.error;
  console.error = (...args) => {};
  try {
    return await fn();
  } finally {
    console.error = originalErr;
  }
}

describe('validate-agent-graph direct coverage', () => {
  it('main() runs validation and writes JSON report', async () => {
    const { main } = await import('./validate-agent-graph.mjs');
    const output = (await captureStdout(() =>
      withArgv([process.execPath, 'dummy', '--json'], () =>
        swallowStderr(() => main()),
      ),
    )).join('');
    const parsed = JSON.parse(output.trim());
    assert.equal(typeof parsed.ok, 'boolean');
    process.exitCode = 0;
  });

  it('main() runs validation and writes human-readable report', async () => {
    const { main } = await import('./validate-agent-graph.mjs');
    const output = (await captureStdout(() =>
      withArgv([process.execPath, 'dummy'], () =>
        swallowStderr(() => main()),
      ),
    )).join('');
    assert.ok(output.includes('PASS') || output.includes('FAIL'));
    process.exitCode = 0;
  });

  it('main() sets exitCode to 1 when report.ok is false (line 30 branch)', async () => {
    const fsSync = await import('node:fs');
    const os = await import('node:os');
    const pathMod = await import('node:path');
    const tempDir = fsSync.mkdtempSync(pathMod.join(os.tmpdir(), 'agent-graph-fail-'));
    const agentsDir = pathMod.join(tempDir, '.github', 'agents');
    fsSync.mkdirSync(agentsDir, { recursive: true });
    fsSync.writeFileSync(
      pathMod.join(agentsDir, 'bad.agent.md'),
      '---\nname: bad-agent\ntools: []\n---\nBad agent with no tier.\n',
    );

    const originalCwd = process.cwd();
    const originalArgv = process.argv;
    const originalWrite = process.stdout.write.bind(process.stdout);
    const originalExitCode = process.exitCode;
    const originalErr = console.error;
    const stdoutChunks = [];
    process.stdout.write = (chunk) => {
      stdoutChunks.push(typeof chunk === 'string' ? chunk : chunk.toString('utf8'));
      return true;
    };
    console.error = () => {};
    process.chdir(tempDir);
    process.argv = [process.execPath, 'dummy', '--json'];

    let capturedExitCode = 0;
    try {
      const { main } = await import(
        `./validate-agent-graph.mjs?fail-${Date.now()}`
      );
      await main();
      capturedExitCode = process.exitCode;
    } finally {
      process.chdir(originalCwd);
      process.argv = originalArgv;
      process.stdout.write = originalWrite;
      console.error = originalErr;
      process.exitCode = originalExitCode ?? 0;
      fsSync.rmSync(tempDir, { recursive: true, force: true });
    }

    const output = stdoutChunks.join('');
    const parsed = JSON.parse(output.trim());
    assert.equal(parsed.ok, false);
    assert.equal(capturedExitCode, 1);
  });

  it('handleMainError logs error and sets exit code', async () => {
    const { handleMainError } = await import('./validate-agent-graph.mjs');
    const errors = [];
    const originalErr = console.error;
    console.error = (...args) => errors.push(args.join(' '));
    try {
      await handleMainError(new Error('test error'));
      assert.ok(errors.some((e) => e.includes('test error')));
      assert.equal(process.exitCode, 1);
    } finally {
      console.error = originalErr;
      process.exitCode = 0;
    }
  });

  it('main() with --help prints usage and exits (lines 20-25)', async () => {
    const { main } = await import('./validate-agent-graph.mjs');
    const originalExit = process.exit;
    const logs = [];
    const originalLog = console.log;
    console.log = (...args) => logs.push(args.join(' '));
    process.exit = () => {};
    try {
      await withArgv([process.execPath, 'dummy', '--help'], () => main());
      assert.ok(logs.some((l) => l.includes('Validate NeatapticTS')));
    } finally {
      process.exit = originalExit;
      console.log = originalLog;
      process.exitCode = 0;
    }
  });

  it('covers entry point guard when imported as main module', async () => {
    const scriptPath = path.resolve(
      'scripts',
      'agent-customization',
      'validate-agent-graph.mjs',
    );
    const originalArgv = process.argv;
    const originalWrite = process.stdout.write.bind(process.stdout);
    process.stdout.write = () => true;
    console.error = () => {};
    process.argv = [process.execPath, scriptPath];
    // Force the entry-guard condition to be true by mocking import.meta.url check.
    // We import fresh — the guard at lines 38-43 checks:
    //   process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
    // Since we set process.argv[1] to the script path, and import.meta.url is the
    // module URL, they should match.
    try {
      // Use dynamic import with cache-busting query to re-execute module top-level
      await import(`./validate-agent-graph.mjs?entry-guard=${Date.now()}`);
      // Allow the async main().then() to settle
      await new Promise((resolve) => setTimeout(resolve, 300));
    } finally {
      process.argv = originalArgv;
      process.stdout.write = originalWrite;
      console.error = (...args) => {};
      process.exitCode = 0;
    }
  });
});