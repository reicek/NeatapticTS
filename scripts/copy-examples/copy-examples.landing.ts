/*
 * Landing-page write step for the copy-examples boundary.
 *
 * This module owns the single I/O operation that commits the shared
 * docs/examples/index.html landing page. HTML construction is delegated to
 * copy-examples.html.ts so that this layer contains only the file-write concern.
 */

import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import {
  DOCS_EXAMPLES_DIR,
  DOCS_EXAMPLES_LOG_PREFIX,
  EXAMPLE_ENTRY_FILE_NAME,
} from './copy-examples.constants.js';
import { buildExamplesLandingPageHtml } from './copy-examples.html.js';
import type { PublishedExample } from './copy-examples.types.js';

/**
 * Writes the shared `docs/examples/index.html` landing page.
 *
 * Creates `docs/examples/` if it does not yet exist, then renders and writes
 * the landing page from the list of examples that were successfully published
 * in the current run. The HTML is built by `buildExamplesLandingPageHtml`.
 *
 * @param publishedExamples - Examples that were published in the current run.
 * @returns Promise resolved when the landing page has been written to disk.
 */
export async function writeExamplesLandingPage(
  publishedExamples: readonly PublishedExample[],
): Promise<void> {
  await mkdir(DOCS_EXAMPLES_DIR, { recursive: true });

  const examplesLandingPageHtml = buildExamplesLandingPageHtml(publishedExamples);
  await writeFile(
    path.join(DOCS_EXAMPLES_DIR, EXAMPLE_ENTRY_FILE_NAME),
    examplesLandingPageHtml,
    'utf8',
  );

  console.log(`${DOCS_EXAMPLES_LOG_PREFIX} Wrote examples landing page`);
}
