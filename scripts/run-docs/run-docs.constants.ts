/*
 * Named constants for the docs-pipeline runner.
 *
 * Centralising the mode strings here means every switch/comparison goes
 * through the same literal and a typo surfaces as a TypeScript error instead
 * of a silent runtime mismatch.
 */

/** CLI mode token that triggers the full docs build pipeline, including browser bundles and HTML rendering. */
export const ALL_MODE = 'all' as const;

/** CLI mode token that triggers only the folder-level README generation steps, skipping browser bundles and HTML. */
export const FOLDERS_MODE = 'folders' as const;

/**
 * Complete set of mode strings accepted by the CLI entry point.
 *
 * Used by `ensureSupportedMode` to validate `process.argv[2]` before routing
 * to a workflow function.
 */
export const SUPPORTED_MODES: ReadonlySet<string> = new Set([
  ALL_MODE,
  FOLDERS_MODE,
]);
