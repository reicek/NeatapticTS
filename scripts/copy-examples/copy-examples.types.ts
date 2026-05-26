/*
 * Public contracts for the copy-examples boundary.
 *
 * These types are shared across all sub-modules. Keeping them in one place
 * prevents circular imports and lets each sub-module import only what it needs.
 */

/**
 * Category bucket that controls which section of the landing page an example
 * is rendered into.
 *
 * - `'flagship'` — larger, browser-hosted demos shown in the "Flagship Demos" section.
 * - `'starter'`  — focused single-concept walkthroughs shown in the "Starter Examples" section.
 */
export type ExampleCategory = 'flagship' | 'starter';

/**
 * Static definition of one example registered in the copy-examples workflow.
 *
 * Every field except `runCommand` is required because the publishing pipeline
 * always needs a destination directory name, a landing-page label, and source
 * root to probe for an `index.html`.
 */
export interface ExampleDefinition {
  /** Landing-page category that governs section placement. */
  category: ExampleCategory;
  /** One-sentence description rendered on the landing page and source-first page. */
  description: string;
  /** Output folder name under `docs/examples/` and link target on the landing page. */
  dirName: string;
  /** Short identifier used for the `examples/<label>` badge on the card. */
  label: string;
  /** Optional shell command shown as the local run snippet. */
  runCommand?: string;
  /** Full page title used in `<title>` and `<h1>` elements. */
  title: string;
  /** Absolute filesystem path to the source example folder. */
  sourceDir: string;
}

/**
 * Metadata for an example that was successfully copied (or generated) into
 * `docs/examples/` during the current run.
 *
 * The workflow returns `PublishedExample | null` per entry; callers narrow to
 * `PublishedExample` before passing to landing-page builders.
 */
export interface PublishedExample {
  /** Landing-page category that governs section placement. */
  category: ExampleCategory;
  /** One-sentence description rendered on the landing page card. */
  description: string;
  /** Output folder name under `docs/examples/`. */
  dirName: string;
  /**
   * `true` when the source `index.html` was found and copied verbatim
   * (with asset-path rewrites); `false` when a source-first placeholder was
   * generated instead.
   */
  hasBrowserEntry: boolean;
  /** Short identifier used for the `examples/<label>` badge on the card. */
  label: string;
  /** Optional shell command shown as the local run snippet. */
  runCommand?: string;
  /** Full page title used in the landing-page card heading. */
  title: string;
}
