/*
 * Orchestrates the docs pipeline with one docs-scripts build and parallelized
 * independent steps.
 *
 * This keeps the user-facing `npm run docs` and `npm run docs:folders`
 * commands simple while removing repeated `docs:build-scripts` work and
 * parallelizing the safe parts of the pipeline.
 */

import { spawn, type ChildProcess } from 'node:child_process';

const ALL_MODE = 'all';
const FOLDERS_MODE = 'folders';
const SUPPORTED_MODES = new Set([ALL_MODE, FOLDERS_MODE]);

interface ScriptTask {
  label: string;
  scriptName: string;
}

interface RunningTask {
  childProcess: ChildProcess;
  command: string;
}

interface SpawnedCommand {
  command: string;
  executable: string;
  argumentsToPass: string[];
}

async function main(): Promise<void> {
  const requestedMode = process.argv[2] ?? ALL_MODE;
  ensureSupportedMode(requestedMode);

  if (requestedMode === FOLDERS_MODE) {
    await runFoldersWorkflow();
    return;
  }

  await runFullDocsWorkflow();
}

async function runFullDocsWorkflow(): Promise<void> {
  // Step 1: Build the browser example bundles in parallel.
  await runScriptTasksInParallel([
    { label: 'Hello Network bundle', scriptName: 'build:hello-network' },
    { label: 'Evolve XOR bundle', scriptName: 'build:evolve-xor' },
    { label: 'Sequence Reset bundle', scriptName: 'build:sequence-reset' },
    { label: 'ASCII Maze bundles', scriptName: 'build:ascii-maze' },
    { label: 'Flappy Bird bundles', scriptName: 'build:flappy-bird' },
    { label: 'NEATchat bundle', scriptName: 'build:neat-chat' },
    { label: 'Semantic snapshot', scriptName: 'index:build-snapshot' },
  ]);

  // Step 2: Generate copied example assets and folder docs in parallel.
  await runScriptTasksInParallel([
    { label: 'Examples copy', scriptName: 'docs:examples:built' },
    { label: 'Source folder docs', scriptName: 'docs:folders:src:built' },
    {
      label: 'ASCII Maze folder docs',
      scriptName: 'docs:folders:asciiMaze:built',
    },
    {
      label: 'Flappy Bird folder docs',
      scriptName: 'docs:folders:flappy-bird:built',
    },
  ]);

  // Step 3: Render the final HTML site after content generation finishes.
  await runScriptTask({
    label: 'HTML docs render',
    scriptName: 'docs:html:built',
  });
}

async function runFoldersWorkflow(): Promise<void> {
  await runScriptTasksInParallel([
    { label: 'Source folder docs', scriptName: 'docs:folders:src:built' },
    {
      label: 'ASCII Maze folder docs',
      scriptName: 'docs:folders:asciiMaze:built',
    },
    {
      label: 'Flappy Bird folder docs',
      scriptName: 'docs:folders:flappy-bird:built',
    },
  ]);
}

function ensureSupportedMode(
  requestedMode: string,
): asserts requestedMode is 'all' | 'folders' {
  if (!SUPPORTED_MODES.has(requestedMode)) {
    throw new Error(
      `Unsupported docs mode "${requestedMode}". Expected one of: ${[...SUPPORTED_MODES].join(', ')}`,
    );
  }
}

async function runScriptTasksInParallel(
  scriptTasks: readonly ScriptTask[],
): Promise<void> {
  if (scriptTasks.length === 0) {
    return;
  }

  await new Promise<void>((resolve, reject) => {
    const runningTasks: RunningTask[] = [];
    let completedTaskCount = 0;
    let hasSettled = false;

    for (const scriptTask of scriptTasks) {
      const spawnedCommand = createNpmRunCommand(scriptTask.scriptName);
      const childProcess = spawn(
        spawnedCommand.executable,
        spawnedCommand.argumentsToPass,
        {
          stdio: 'inherit',
        },
      );
      runningTasks.push({ childProcess, command: spawnedCommand.command });

      childProcess.once('error', (error) => {
        if (hasSettled) {
          return;
        }

        hasSettled = true;
        stopSiblingProcesses(runningTasks, childProcess);
        reject(
          new Error(
            `Failed to start ${scriptTask.label} (${scriptTask.scriptName}): ${error.message}`,
            { cause: error },
          ),
        );
      });

      childProcess.once('exit', (exitCode, signal) => {
        if (hasSettled) {
          return;
        }

        if (exitCode === 0) {
          completedTaskCount += 1;
          if (completedTaskCount === scriptTasks.length) {
            hasSettled = true;
            resolve();
          }

          return;
        }

        hasSettled = true;
        stopSiblingProcesses(runningTasks, childProcess);
        reject(
          new Error(
            `${scriptTask.label} (${scriptTask.scriptName}) failed with exit code ${exitCode ?? 'null'}${signal ? ` and signal ${signal}` : ''}.`,
          ),
        );
      });
    }
  });
}

async function runScriptTask(scriptTask: ScriptTask): Promise<void> {
  await runScriptTasksInParallel([scriptTask]);
}

function createNpmRunCommand(scriptName: string): SpawnedCommand {
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

function stopSiblingProcesses(
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

function stopProcess(childProcess: ChildProcess): void {
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

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
