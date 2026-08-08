/*
 * Copies browser-viewable examples into docs/examples/* so GitHub Pages can
 * publish them alongside the generated documentation site.
 *
 * The script intentionally keeps one narrow contract: copy each example's
 * browser entrypoint into its published docs folder, then rebuild the shared
 * examples landing page from the demos that were actually published.
 *
 * Heavy logic lives in the `scripts/copy-examples/` boundary:
 *   copy-examples.types.ts       — ExampleCategory, ExampleDefinition, PublishedExample
 *   copy-examples.constants.ts   — path and URL constants
 *   copy-examples.definitions.ts — EXAMPLE_DEFINITIONS registry
 *   copy-examples.io.ts          — pathExists, isPublishedExample
 *   copy-examples.copy.ts        — copyExampleEntryPoint, removeRetiredPublishedExamples
 *   copy-examples.html.ts        — all HTML builder functions
 *   copy-examples.landing.ts     — writeExamplesLandingPage
 */

import {
  copyExampleDocs,
  copyExampleEntryPoint,
  copyExampleExtraAssets,
  removeRetiredPublishedExamples,
} from './copy-examples/copy-examples.copy.js';
import { EXAMPLE_DEFINITIONS } from './copy-examples/copy-examples.definitions.js';
import { isPublishedExample } from './copy-examples/copy-examples.io.js';
import { writeExamplesLandingPage } from './copy-examples/copy-examples.landing.js';
import { DOCS_EXAMPLES_LOG_PREFIX } from './copy-examples/copy-examples.constants.js';

/**
 * Runs the example-copy workflow.
 *
 * Step 1 copies published browser entrypoints for all known examples.
 * Step 2 removes stale docs/examples folders from previous runs.
 * Step 3 rebuilds the landing page from the demos that were actually copied.
 *
 * @returns Promise resolved when copying, retirement, and landing-page generation complete.
 */
async function main(): Promise<void> {
  const publishedExamples = (
    await Promise.all(
      EXAMPLE_DEFINITIONS.map(async (exampleDefinition) => {
        const publishedExample = await copyExampleEntryPoint(exampleDefinition);
        // Extra assets (supplementary browser pages and their data
        // dependencies) are copied verbatim into the published folder so they
        // can be served from the same URL path as the generated index.html.
        await copyExampleExtraAssets(exampleDefinition);
        // Static docs are copied regardless of whether the example has a browser
        // entrypoint, so educational markdown files under <example>/docs/ are
        // always published alongside the generated folder READMEs.
        await copyExampleDocs(exampleDefinition);
        return publishedExample;
      }),
    )
  ).filter(isPublishedExample);

  await removeRetiredPublishedExamples(publishedExamples);
  await writeExamplesLandingPage(publishedExamples);
}

main().catch((error: unknown) => {
  console.error(`${DOCS_EXAMPLES_LOG_PREFIX} Failed:`, error);
  process.exit(1);
});
