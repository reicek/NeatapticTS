/*
 * Shared contracts for the docs-pipeline runner boundary.
 *
 * Keeping interfaces here prevents circular imports between the process,
 * runner, and workflow modules and gives IDEs a single hover-navigation target.
 */

import type { ChildProcess } from 'node:child_process';

/**
 * A named npm script that the runner should execute as a child process.
 *
 * The `label` is used only in human-readable error messages. The `scriptName`
 * is passed verbatim to `npm run <scriptName>`.
 */
export interface ScriptTask {
  /** Human-readable description used in error messages. */
  label: string;
  /** npm script name to execute (e.g. `"build:hello-network"`). */
  scriptName: string;
}

/**
 * A script task that has been spawned and is currently running.
 *
 * The `command` string is kept solely for diagnostic output; the
 * `childProcess` handle is what gets killed on sibling failure.
 */
export interface RunningTask {
  /** The live child-process handle. */
  childProcess: ChildProcess;
  /** Full command string (executable + arguments) used for diagnostics. */
  command: string;
}

/**
 * Resolved executable and argument list ready for `spawn()`.
 *
 * `command` is the human-readable join of `executable` and
 * `argumentsToPass`; it is not passed to the OS directly.
 */
export interface SpawnedCommand {
  /** Full command string for display and error messages. */
  command: string;
  /** Executable passed as the first argument to `spawn()`. */
  executable: string;
  /** Argument array passed as the second argument to `spawn()`. */
  argumentsToPass: string[];
}
