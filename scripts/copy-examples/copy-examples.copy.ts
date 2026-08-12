/*
 * File-copy and retirement operations for the copy-examples boundary.
 *
 * This module owns all destructive I/O: writing published example pages into
 * docs/examples/ and removing stale folders that are no longer in the registry.
 * Pure HTML construction is delegated to copy-examples.html.ts.
 */

import { mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { copy } from 'fs-extra';
import {
  DOCS_EXAMPLES_DIR,
  DOCS_EXAMPLES_LOG_PREFIX,
  EXAMPLE_ENTRY_FILE_NAME,
} from './copy-examples.constants.js';
import {
  addGeneratedExamplePageComment,
  buildSourceFirstExamplePageHtml,
  rewriteExamplePageForDocs,
} from './copy-examples.html.js';
import { pathExists } from './copy-examples.io.js';
import type {
  ExampleDefinition,
  PublishedExample,
} from './copy-examples.types.js';

/**
 * Copies one example entrypoint into `docs/examples/<name>/index.html`.
 *
 * When the source folder exists and contains an `index.html`, the file is
 * read, rewritten for the docs asset layout, stamped with a generated-file
 * comment, and written to the destination. When the source folder exists but
 * has no `index.html`, a source-first placeholder page is generated instead.
 * When the source folder is missing entirely, the example is skipped and
 * `null` is returned.
 *
 * @param exampleDefinition - Example publishing definition from `EXAMPLE_DEFINITIONS`.
 * @returns Published example metadata when the example was successfully written,
 *   otherwise `null`.
 */
export async function copyExampleEntryPoint(
  exampleDefinition: ExampleDefinition,
): Promise<PublishedExample | null> {
  const hasSourceDirectory = await pathExists(exampleDefinition.sourceDir);
  if (!hasSourceDirectory) {
    console.warn(
      `${DOCS_EXAMPLES_LOG_PREFIX} ${exampleDefinition.dirName} source directory not found, skipping`,
    );
    return null;
  }

  const sourceIndexPath = path.join(
    exampleDefinition.sourceDir,
    EXAMPLE_ENTRY_FILE_NAME,
  );
  const hasSourceIndex = await pathExists(sourceIndexPath);

  const destinationDirectoryPath = path.join(
    DOCS_EXAMPLES_DIR,
    exampleDefinition.dirName,
  );
  const destinationIndexPath = path.join(
    destinationDirectoryPath,
    EXAMPLE_ENTRY_FILE_NAME,
  );

  await mkdir(destinationDirectoryPath, { recursive: true });

  if (hasSourceIndex) {
    const sourceIndexHtml = await readFile(sourceIndexPath, 'utf8');
    const docsReadyIndexHtml = addGeneratedExamplePageComment(
      exampleDefinition,
      rewriteExamplePageForDocs(sourceIndexHtml),
    );

    await writeFile(destinationIndexPath, docsReadyIndexHtml, 'utf8');

    console.log(
      `${DOCS_EXAMPLES_LOG_PREFIX} Copied ${exampleDefinition.dirName} ${EXAMPLE_ENTRY_FILE_NAME}`,
    );
  } else {
    const generatedExamplePageHtml = addGeneratedExamplePageComment(
      exampleDefinition,
      buildSourceFirstExamplePageHtml(exampleDefinition),
    );

    await writeFile(destinationIndexPath, generatedExamplePageHtml, 'utf8');

    console.log(
      `${DOCS_EXAMPLES_LOG_PREFIX} Generated source-first page for ${exampleDefinition.dirName}`,
    );
  }

  return {
    category: exampleDefinition.category,
    description: exampleDefinition.description,
    dirName: exampleDefinition.dirName,
    hasBrowserEntry: hasSourceIndex,
    label: exampleDefinition.label,
    runCommand: exampleDefinition.runCommand,
    title: exampleDefinition.title,
  };
}

/**
 * Copies supplementary asset files declared on an example definition into the
 * published `docs/examples/<name>/` folder.
 *
 * Unlike the primary entrypoint, extra assets are copied verbatim with no
 * path rewriting or generated-file comment stamping. This is the intended
 * behavior for supplementary browser pages (such as a sprite preview) and
 * their data dependencies that must be served from the same folder as the
 * generated `index.html` but are authored directly in the source tree.
 *
 * Files listed in `exampleDefinition.extraAssets` that do not exist in the
 * source directory are skipped with a warning so that the pipeline does not
 * fail when an optional asset is absent.
 *
 * @param exampleDefinition - Example publishing definition with optional `extraAssets`.
 * @returns Promise resolved when all declared extra assets have been copied or skipped.
 */
export async function copyExampleExtraAssets(
  exampleDefinition: ExampleDefinition,
): Promise<void> {
  if (
    !exampleDefinition.extraAssets ||
    exampleDefinition.extraAssets.length === 0
  ) {
    return;
  }

  const destinationDirectoryPath = path.join(
    DOCS_EXAMPLES_DIR,
    exampleDefinition.dirName,
  );

  for (const assetFileName of exampleDefinition.extraAssets) {
    const sourceAssetPath = path.join(
      exampleDefinition.sourceDir,
      assetFileName,
    );
    if (!(await pathExists(sourceAssetPath))) {
      console.warn(
        `${DOCS_EXAMPLES_LOG_PREFIX} ${exampleDefinition.dirName} extra asset not found, skipping: ${assetFileName}`,
      );
      continue;
    }

    const destinationAssetPath = path.join(
      destinationDirectoryPath,
      assetFileName,
    );
    await copy(sourceAssetPath, destinationAssetPath, { overwrite: true });

    console.log(
      `${DOCS_EXAMPLES_LOG_PREFIX} Copied ${exampleDefinition.dirName} extra asset: ${assetFileName}`,
    );
  }
}

/**
 * Copies static documentation files from an example's `docs/` folder into the
 * published docs tree.
 *
 * The generator owns per-directory README files and the root README mirror, but
 * hand-written educational markdown files placed under `examples/<name>/docs/`
 * are not automatically published.  This function copies them so links from
 * the generated root README resolve in both the source tree and the published
 * docs site.
 *
 * README.md files are skipped because they are produced by the folder docs
 * generator and must not be overwritten by hand-written copies.
 *
 * @param exampleDefinition - Example publishing definition.
 * @returns Nothing.
 */
export async function copyExampleDocs(
  exampleDefinition: ExampleDefinition,
): Promise<void> {
  const sourceDocsDir = path.join(exampleDefinition.sourceDir, 'docs');
  if (!(await pathExists(sourceDocsDir))) {
    return;
  }

  const destinationDocsDir = path.join(
    DOCS_EXAMPLES_DIR,
    exampleDefinition.dirName,
    'docs',
  );
  await mkdir(destinationDocsDir, { recursive: true });

  await copy(sourceDocsDir, destinationDocsDir, {
    overwrite: true,
    filter: (src) => path.basename(src) !== 'README.md',
  });

  console.log(
    `${DOCS_EXAMPLES_LOG_PREFIX} Copied docs for ${exampleDefinition.dirName}`,
  );
}

/**
 * Removes stale `docs/examples/<name>/` folders for examples that are no
 * longer in the current published set.
 *
 * Staleness is determined by comparing the names of all subdirectories found
 * in `docs/examples/` against the `dirName` values in `publishedExamples`.
 * Any directory not in the published set is recursively removed. The function
 * is a no-op when `docs/examples/` does not yet exist.
 *
 * @param publishedExamples - Examples kept in the current docs publication set.
 * @returns Promise resolved when all stale folders have been removed.
 */
export async function removeRetiredPublishedExamples(
  publishedExamples: readonly PublishedExample[],
): Promise<void> {
  const hasDocsExamplesDirectory = await pathExists(DOCS_EXAMPLES_DIR);
  if (!hasDocsExamplesDirectory) {
    return;
  }

  const publishedExampleDirectoryNames = new Set(
    publishedExamples.map((publishedExample) => publishedExample.dirName),
  );
  const docsExamplesEntries = await readdir(DOCS_EXAMPLES_DIR, {
    withFileTypes: true,
  });

  await Promise.all(
    docsExamplesEntries
      .filter((docsExamplesEntry) => docsExamplesEntry.isDirectory())
      .filter((docsExamplesEntry) => {
        return !publishedExampleDirectoryNames.has(docsExamplesEntry.name);
      })
      .map(async (docsExamplesEntry) => {
        await rm(path.join(DOCS_EXAMPLES_DIR, docsExamplesEntry.name), {
          recursive: true,
          force: true,
        });

        console.log(
          `${DOCS_EXAMPLES_LOG_PREFIX} Removed stale published example ${docsExamplesEntry.name}`,
        );
      }),
  );
}
