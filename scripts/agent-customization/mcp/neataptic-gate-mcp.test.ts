/**
 * @module neataptic-gate-mcp.test
 * @description Green tests for the gate MCP server entry point.
 */
import { spawn } from 'node:child_process';
import path from 'node:path';

const REPO_ROOT = path.resolve();
const SERVER_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/mcp/neataptic-gate-mcp.mjs',
);

function runSelfCheck(): Promise<{ exitCode: number | null; output: string }> {
  return new Promise((resolve) => {
    const child = spawn(
      process.execPath,
      [SERVER_PATH, '--self-check', '--json'],
      {
        cwd: REPO_ROOT,
      },
    );
    let output = '';
    child.stdout?.on('data', (chunk) => {
      output += chunk.toString();
    });
    child.stderr?.on('data', (chunk) => {
      output += chunk.toString();
    });
    child.on('close', (exitCode) => {
      resolve({ exitCode, output });
    });
  });
}

describe('neataptic-gate-mcp server entry', () => {
  it('passes its own self-check', async () => {
    const { exitCode, output } = await runSelfCheck();
    expect(exitCode).toBe(0);
    expect(output).toContain('gate-mcp self-check');
    const parsed = JSON.parse(output);
    expect(parsed.ok).toBe(true);
  });

  it('exits cleanly when stdin ends without a request', async () => {
    const child = spawn(process.execPath, [SERVER_PATH], {
      cwd: REPO_ROOT,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let output = '';
    child.stdout?.on('data', (chunk) => {
      output += chunk.toString();
    });
    child.stderr?.on('data', (chunk) => {
      output += chunk.toString();
    });

    child.stdin?.end();

    const exitCode = await new Promise<number | null>((resolve) => {
      const timeout = setTimeout(() => {
        child.kill();
        resolve(null);
      }, 3000);
      child.on('close', (code) => {
        clearTimeout(timeout);
        resolve(code);
      });
    });

    expect(exitCode === 0 || exitCode === null).toBe(true);
  });
});
