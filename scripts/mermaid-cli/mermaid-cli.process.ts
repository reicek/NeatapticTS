/**
 * @module mermaid-cli/mermaid-cli.process
 *
 * Child-process lifecycle helpers for the Mermaid CLI wrapper.
 *
 * `runMermaidCli` spawns the mmdc entry point as a child Node.js process and
 * propagates stdio directly to the terminal, so the caller sees real-time
 * Mermaid output.  The remaining helpers manage temporary directories that
 * must exist before mmdc writes its output files.
 */

import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { MERMAID_TEMP_DIRECTORY_PREFIX } from './mermaid-cli.constants.js';
import type { MermaidCliInvocation } from './mermaid-cli.types.js';

/**
 * Runs Mermaid CLI as a child process with the prepared argument list.
 *
 * Stdio is inherited so mmdc output is visible to the caller in real time.
 * Rejects with an error when mmdc exits with a non-zero status code.
 *
 * @param cliPath - Absolute path to the mmdc entry point.
 * @param argumentsToPass - Argument list to forward to mmdc.
 * @returns Resolves when mmdc exits with code `0`.
 */
export async function runMermaidCli(
  cliPath: string,
  argumentsToPass: string[],
): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const childProcess = spawn(
      process.execPath,
      [cliPath, ...argumentsToPass],
      {
        stdio: 'inherit',
      },
    );

    childProcess.once('exit', (exitCode) => {
      if (exitCode === 0) {
        resolve();
        return;
      }

      reject(new Error(`Mermaid CLI exited with code ${exitCode ?? 'null'}.`));
    });
    childProcess.once('error', reject);
  });
}

/**
 * Creates a temporary Mermaid output directory inside the OS temp folder.
 *
 * @returns Absolute path to the created temporary directory.
 */
export async function createMermaidTempDirectory(): Promise<string> {
  return mkdtemp(path.join(os.tmpdir(), MERMAID_TEMP_DIRECTORY_PREFIX));
}

/**
 * Ensures the parent directory for a given file path exists, creating it
 * recursively when absent.
 *
 * @param filePath - Target file path whose parent directory must exist.
 * @returns Resolves when the parent directory exists.
 */
export async function ensureParentDirectoryExists(
  filePath: string,
): Promise<void> {
  await mkdir(path.dirname(path.resolve(filePath)), { recursive: true });
}

/**
 * Cleans up a prepared Mermaid invocation and an optional extra temp directory.
 *
 * Runs the invocation's own cleanup callback first, then removes the extra
 * directory (e.g. the temporary validation output directory).
 *
 * @param mermaidCliInvocation - Prepared Mermaid invocation with a cleanup callback.
 * @param extraDirectoryPath - Additional temporary directory to remove.
 * @returns Resolves after all cleanup completes.
 */
export async function cleanupInvocation(
  mermaidCliInvocation: MermaidCliInvocation,
  extraDirectoryPath: string,
): Promise<void> {
  await mermaidCliInvocation.cleanup();
  await rm(extraDirectoryPath, { recursive: true, force: true });
}
