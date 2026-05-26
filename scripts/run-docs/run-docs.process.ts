/*
 * Low-level child-process helpers for the docs-pipeline runner.
 *
 * `createNpmRunCommand` normalises across three environments:
 *   1. npm lifecycle scripts (npm_execpath is set) — invoke via node directly.
 *   2. Windows cmd.exe shells without npm_execpath.
 *   3. POSIX shells without npm_execpath.
 *
 * `stopProcess` / `stopSiblingProcesses` provide best-effort teardown when
 * any task in a parallel batch fails. On Windows, `taskkill /t` is required to
 * reach grandchild processes; `SIGTERM` has no effect there.
 */

import { spawn, type ChildProcess } from 'node:child_process';
import type { RunningTask, SpawnedCommand } from './run-docs.types.js';

/**
 * Builds the platform-appropriate `SpawnedCommand` for `npm run <scriptName>`.
 *
 * Resolution order:
 * 1. If `npm_execpath` is set (inside an npm lifecycle script), delegate to
 *    the same Node binary so the correct npm version is used.
 * 2. If running on Windows without `npm_execpath`, invoke via `cmd.exe` so
 *    `npm.cmd` resolves correctly through `%PATH%`.
 * 3. Otherwise invoke `npm` directly.
 *
 * @param scriptName - npm script name passed verbatim to `npm run`.
 * @returns Fully resolved `SpawnedCommand` ready for `spawn()`.
 */
export function createNpmRunCommand(scriptName: string): SpawnedCommand {
  const npmExecutablePath = process.env.npm_execpath;

  if (npmExecutablePath) {
    const argumentsToPass = [npmExecutablePath, 'run', scriptName];

    return {
      command: `${process.execPath} ${argumentsToPass.join(' ')}`,
      executable: process.execPath,
      argumentsToPass,
    };
  }

  if (process.platform === 'win32') {
    const executable = process.env.ComSpec ?? 'cmd.exe';
    const argumentsToPass = ['/d', '/s', '/c', 'npm.cmd', 'run', scriptName];

    return {
      command: `${executable} ${argumentsToPass.join(' ')}`,
      executable,
      argumentsToPass,
    };
  }

  const executable = 'npm';
  const argumentsToPass = ['run', scriptName];

  return {
    command: `${executable} ${argumentsToPass.join(' ')}`,
    executable,
    argumentsToPass,
  };
}

/**
 * Terminates every running task in the batch except for the process that
 * already exited or errored.
 *
 * Errors from individual `stopProcess` calls are swallowed — teardown is
 * best-effort and should not mask the original failure.
 *
 * @param runningTasks - All tasks that were spawned in this parallel batch.
 * @param excludedProcess - The process that triggered the failure; skipped so
 *   it is not double-killed.
 */
export function stopSiblingProcesses(
  runningTasks: readonly RunningTask[],
  excludedProcess: ChildProcess,
): void {
  for (const { childProcess } of runningTasks) {
    if (childProcess === excludedProcess || childProcess.killed) {
      continue;
    }

    try {
      stopProcess(childProcess);
    } catch {
      // Best-effort shutdown only.
    }
  }
}

/**
 * Terminates a single child process.
 *
 * On Windows, `taskkill /t /f` is used so that the entire process tree
 * (including grandchildren such as nested npm scripts) is killed. On POSIX,
 * the standard `SIGTERM` via `childProcess.kill()` is sufficient.
 *
 * @param childProcess - The child process to terminate.
 */
export function stopProcess(childProcess: ChildProcess): void {
  const processId = childProcess.pid;

  if (processId === undefined) {
    childProcess.kill();
    return;
  }

  if (process.platform === 'win32') {
    spawn('taskkill', ['/pid', String(processId), '/t', '/f'], {
      stdio: 'ignore',
    }).unref();
    return;
  }

  childProcess.kill();
}
