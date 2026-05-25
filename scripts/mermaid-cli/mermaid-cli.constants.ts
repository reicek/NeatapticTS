/**
 * @module mermaid-cli/mermaid-cli.constants
 *
 * Named constants for the Mermaid CLI wrapper.
 * Centralising string literals here ensures every module uses the same
 * values and avoids magic-string drift across the codebase.
 */

/** Sub-command name passed to the Mermaid wrapper to render a diagram to an output file. */
export const EXPORT_COMMAND = 'export' as const;

/** Sub-command name passed to the Mermaid wrapper to validate a diagram without producing an output file. */
export const VALIDATE_COMMAND = 'validate' as const;

/** Complete set of sub-command strings the CLI entry point accepts, used for early input validation. */
export const SUPPORTED_COMMANDS = new Set([
  EXPORT_COMMAND,
  VALIDATE_COMMAND,
] as const);

/** `mkdtemp` prefix for temporary directories that hold Mermaid SVG output during diagram validation. */
export const MERMAID_TEMP_DIRECTORY_PREFIX = 'neatapticts-mermaid-';

/** `mkdtemp` prefix for temporary directories that hold the injected Puppeteer configuration on Linux CI. */
export const PUPPETEER_TEMP_DIRECTORY_PREFIX = 'neatapticts-mermaid-puppeteer-';

/** File name of the Puppeteer configuration JSON written into the temporary directory on Linux CI. */
export const PUPPETEER_CONFIG_FILE_NAME = 'puppeteer-config.json';

/**
 * Chromium launch flags injected on Linux CI to disable the sandbox that
 * is unavailable in most containerised environments.
 *
 * @see {@link https://chromium.googlesource.com/chromium/src/+/HEAD/docs/linux/suid_sandbox_development.md | Chromium SUID sandbox}
 */
export const PUPPETEER_CI_LINUX_ARGS: readonly string[] = [
  '--no-sandbox',
  '--disable-setuid-sandbox',
];
