/**
 * @module mermaid-cli/mermaid-cli.puppeteer
 *
 * Puppeteer configuration injection for Linux CI environments.
 *
 * On Linux CI (detected via `CI=true` or `GITHUB_ACTIONS=true`) Chromium
 * refuses to start unless the `--no-sandbox` flag is supplied, because the
 * kernel SUID sandbox is unavailable inside most containers.  This module
 * writes a temporary `puppeteer-config.json` and prepends
 * `--puppeteerConfigFile <path>` to the mmdc argument list so that the caller
 * never has to know the detail.
 *
 * @see {@link https://chromium.googlesource.com/chromium/src/+/HEAD/docs/linux/suid_sandbox_development.md | Chromium SUID sandbox}
 */

import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  PUPPETEER_CI_LINUX_ARGS,
  PUPPETEER_CONFIG_FILE_NAME,
  PUPPETEER_TEMP_DIRECTORY_PREFIX,
} from './mermaid-cli.constants.js';
import type { MermaidCliInvocation, ParsedArguments } from './mermaid-cli.types.js';

/**
 * Builds the Mermaid CLI invocation, injecting a CI-safe Puppeteer config when
 * running on Linux CI without an explicit caller-supplied config.
 *
 * @param parsedArguments - Parsed CLI arguments (used to check for an existing Puppeteer config).
 * @param baseArguments - Base Mermaid CLI arguments to wrap or forward.
 * @returns Prepared Mermaid CLI invocation with argument list and cleanup callback.
 */
export async function buildMermaidCliInvocation(
  parsedArguments: ParsedArguments,
  baseArguments: string[],
): Promise<MermaidCliInvocation> {
  if (!shouldInjectCiLinuxNoSandbox(parsedArguments)) {
    return createDirectInvocation(baseArguments);
  }

  const temporaryPuppeteerDirectoryPath = await createPuppeteerTempDirectory();
  const puppeteerConfigFilePath = path.join(
    temporaryPuppeteerDirectoryPath,
    PUPPETEER_CONFIG_FILE_NAME,
  );

  await writePuppeteerConfigFile(puppeteerConfigFilePath);

  return {
    argumentsToPass: buildPuppeteerConfigArguments(
      puppeteerConfigFilePath,
      baseArguments,
    ),
    cleanup: async () => {
      await rm(temporaryPuppeteerDirectoryPath, {
        recursive: true,
        force: true,
      });
    },
  };
}

/**
 * Creates a direct Mermaid CLI invocation with no cleanup work.
 *
 * @param baseArguments - Base Mermaid CLI arguments to forward unchanged.
 * @returns Direct Mermaid CLI invocation whose cleanup is a no-op.
 */
export function createDirectInvocation(
  baseArguments: string[],
): MermaidCliInvocation {
  return {
    argumentsToPass: baseArguments,
    cleanup: async () => {},
  };
}

/**
 * Writes the injected Puppeteer config file used for Linux CI browser launches.
 *
 * @param puppeteerConfigFilePath - Target file path for the generated config.
 * @returns Resolves when the config file has been written.
 */
export async function writePuppeteerConfigFile(
  puppeteerConfigFilePath: string,
): Promise<void> {
  await writeFile(
    puppeteerConfigFilePath,
    JSON.stringify({ args: PUPPETEER_CI_LINUX_ARGS }),
  );
}

/**
 * Prepends the `--puppeteerConfigFile` flag and injected config path to the
 * base mmdc argument list so Chromium launches with the CI sandbox workaround.
 *
 * @param puppeteerConfigFilePath - Absolute path to the generated Puppeteer config file.
 * @param baseArguments - Base Mermaid CLI arguments to extend with the config flag.
 * @returns Extended mmdc argument list with the Puppeteer config file flag prepended.
 */
export function buildPuppeteerConfigArguments(
  puppeteerConfigFilePath: string,
  baseArguments: string[],
): string[] {
  return ['--puppeteerConfigFile', puppeteerConfigFilePath, ...baseArguments];
}

/**
 * Determines whether the Linux CI sandbox workaround should be injected.
 *
 * Injection occurs only when all three conditions hold: running on Linux,
 * running inside CI, and the caller has not supplied an explicit Puppeteer
 * config.
 *
 * @param parsedArguments - Parsed CLI arguments.
 * @returns `true` when the wrapper should inject a Puppeteer config.
 */
export function shouldInjectCiLinuxNoSandbox(
  parsedArguments: ParsedArguments,
): boolean {
  return (
    isLinuxPlatform() &&
    isContinuousIntegrationEnvironment() &&
    !hasExplicitPuppeteerConfig(parsedArguments)
  );
}

/**
 * Returns `true` when `process.platform` equals `'linux'`, indicating that the
 * current execution environment is a Linux-based operating system.
 *
 * @returns `true` when the active platform is Linux.
 */
export function isLinuxPlatform(): boolean {
  return process.platform === 'linux';
}

/**
 * Determines whether the current process is running in a CI environment.
 *
 * Recognises both the generic `CI=true` convention and the GitHub Actions
 * `GITHUB_ACTIONS=true` variable.
 *
 * @returns `true` when the environment is identified as CI.
 */
export function isContinuousIntegrationEnvironment(): boolean {
  return process.env.CI === 'true' || process.env.GITHUB_ACTIONS === 'true';
}

/**
 * Determines whether the caller already supplied a Puppeteer config override.
 *
 * Checks both named arguments (`--puppeteerConfigFile`, `-p`) and raw
 * pass-through tokens to avoid double-injection.
 *
 * @param parsedArguments - Parsed CLI arguments.
 * @returns `true` when an explicit Puppeteer config is already present.
 */
export function hasExplicitPuppeteerConfig(
  parsedArguments: ParsedArguments,
): boolean {
  return (
    parsedArguments.named.puppeteerConfigFile !== undefined ||
    parsedArguments.named.p !== undefined ||
    parsedArguments.passthrough.includes('--puppeteerConfigFile') ||
    parsedArguments.passthrough.includes('-p')
  );
}

/**
 * Creates a temporary Puppeteer config directory inside the OS temp folder.
 *
 * @returns Absolute path to the created temporary directory.
 */
export async function createPuppeteerTempDirectory(): Promise<string> {
  return mkdtemp(path.join(os.tmpdir(), PUPPETEER_TEMP_DIRECTORY_PREFIX));
}
