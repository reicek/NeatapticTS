/*
 * Parallel and sequential task execution for the docs-pipeline runner.
 *
 * `runScriptTasksInParallel` spawns every task in the given batch
 * simultaneously and resolves only when all succeed. The first failure kills
 * every surviving sibling and rejects the promise so the outer `main()` can
 * surface a clean error message and non-zero exit code.
 *
 * `runScriptTask` is a convenience wrapper for single-task invocations that
 * must be sequenced relative to the surrounding pipeline steps.
 */

import { type ChildProcess, spawn } from 'node:child_process';
import { createNpmRunCommand, stopSiblingProcesses } from './run-docs.process.js';
import type { RunningTask, ScriptTask } from './run-docs.types.js';

/** Shared mutable state threaded through both event handlers of one task batch. */
interface BatchState {
  completedTaskCount: number;
  hasSettled: boolean;
}

/**
 * Builds an error-event handler for one spawned task in a parallel batch.
 *
 * The handler marks the batch as settled on first call, kills all siblings,
 * and rejects the batch promise with a descriptive spawn-failure message.
 *
 * @param scriptTask - The task whose child process emitted the error.
 * @param runningTasks - All running tasks in the current batch.
 * @param childProcess - The failing child process (excluded from sibling kill).
 * @param state - Shared mutable settlement state for the batch.
 * @param reject - Batch promise rejection callback.
 * @returns Error event handler suitable for `childProcess.once('error', ...)`.
 */
function buildErrorHandler(
  scriptTask: ScriptTask,
  runningTasks: RunningTask[],
  childProcess: ChildProcess,
  state: BatchState,
  reject: (error: Error) => void,
): (error: Error) => void {
  return (error) => {
    if (state.hasSettled) return;
    state.hasSettled = true;
    stopSiblingProcesses(runningTasks, childProcess);
    reject(
      new Error(
        `Failed to start ${scriptTask.label} (${scriptTask.scriptName}): ${error.message}`,
        { cause: error },
      ),
    );
  };
}

/**
 * Builds an exit-event handler for one spawned task in a parallel batch.
 *
 * On a zero exit code the handler increments the completed-task counter and
 * resolves the batch promise when all tasks have finished. On any non-zero exit
 * or signal it marks the batch as settled, kills siblings, and rejects.
 *
 * @param scriptTask - The task whose child process exited.
 * @param totalTasks - Total number of tasks in the current batch.
 * @param runningTasks - All running tasks in the current batch.
 * @param childProcess - The exiting child process (excluded from sibling kill).
 * @param state - Shared mutable settlement state for the batch.
 * @param resolve - Batch promise resolution callback.
 * @param reject - Batch promise rejection callback.
 * @returns Exit event handler suitable for `childProcess.once('exit', ...)`.
 */
function buildExitHandler(
  scriptTask: ScriptTask,
  totalTasks: number,
  runningTasks: RunningTask[],
  childProcess: ChildProcess,
  state: BatchState,
  resolve: () => void,
  reject: (error: Error) => void,
): (exitCode: number | null, signal: NodeJS.Signals | null) => void {
  return (exitCode, signal) => {
    if (state.hasSettled) return;
    if (exitCode === 0) {
      state.completedTaskCount += 1;
      if (state.completedTaskCount === totalTasks) {
        state.hasSettled = true;
        resolve();
      }
      return;
    }
    state.hasSettled = true;
    stopSiblingProcesses(runningTasks, childProcess);
    reject(
      new Error(
        `${scriptTask.label} (${scriptTask.scriptName}) failed with exit code ${exitCode ?? 'null'}${signal ? ` and signal ${signal}` : ''}.`,
      ),
    );
  };
}

/**
 * Spawns all `scriptTasks` in parallel and resolves when every task exits with
 * code `0`.
 *
 * If any task fails (non-zero exit, signal termination, or spawn error), all
 * surviving siblings are killed and the returned promise rejects with a
 * descriptive error. The rejection message includes the task label, npm script
 * name, exit code, and signal so the operator knows which step broke.
 *
 * Resolves immediately when `scriptTasks` is empty.
 *
 * @param scriptTasks - Ordered list of npm scripts to run concurrently.
 * @returns A promise that resolves when all tasks complete successfully.
 */
export async function runScriptTasksInParallel(
  scriptTasks: readonly ScriptTask[],
): Promise<void> {
  if (scriptTasks.length === 0) {
    return;
  }

  await new Promise<void>((resolve, reject) => {
    const runningTasks: RunningTask[] = [];
    const state: BatchState = { completedTaskCount: 0, hasSettled: false };

    for (const scriptTask of scriptTasks) {
      const spawnedCommand = createNpmRunCommand(scriptTask.scriptName);
      const childProcess = spawn(
        spawnedCommand.executable,
        spawnedCommand.argumentsToPass,
        { stdio: 'inherit' },
      );
      runningTasks.push({ childProcess, command: spawnedCommand.command });

      childProcess.once(
        'error',
        buildErrorHandler(scriptTask, runningTasks, childProcess, state, reject),
      );
      childProcess.once(
        'exit',
        buildExitHandler(
          scriptTask,
          scriptTasks.length,
          runningTasks,
          childProcess,
          state,
          resolve,
          reject,
        ),
      );
    }
  });
}

/**
 * Runs a single npm script task, waiting for it to complete before resolving.
 *
 * Delegates to `runScriptTasksInParallel` with a one-element array so error
 * handling and process teardown remain consistent across sequential and
 * parallel invocations.
 *
 * @param scriptTask - The npm script task to execute.
 * @returns A promise that resolves when the task exits successfully.
 */
export async function runScriptTask(scriptTask: ScriptTask): Promise<void> {
  await runScriptTasksInParallel([scriptTask]);
}
