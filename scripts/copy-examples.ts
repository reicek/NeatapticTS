/*
 * Copies browser-viewable examples into docs/examples/* so GitHub Pages can
 * publish them alongside the generated documentation site.
 *
 * The script intentionally keeps one narrow contract: copy each example's
 * browser entrypoint into its published docs folder, then rebuild the shared
 * examples landing page from the demos that were actually published.
 */

import { access, copyFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const DOCS_EXAMPLES_LOG_PREFIX = '[docs:examples]';
const DOCS_EXAMPLES_DIR = path.resolve('docs', 'examples');
const EXAMPLE_ENTRY_FILE_NAME = 'index.html';
const DOCS_THEME_STYLESHEET_PATH = '../assets/theme.css';
const DOCS_HOME_PATH = '../index.html';
const EXAMPLES_HOME_PATH = './index.html';
const GITHUB_REPOSITORY_URL = 'https://github.com/reicek/NeatapticTS';

interface ExampleDefinition {
  dirName: string;
  label: string;
  title: string;
  sourceDir: string;
}

interface PublishedExample {
  dirName: string;
  label: string;
  title: string;
}

const EXAMPLE_DEFINITIONS: readonly ExampleDefinition[] = [
  {
    dirName: 'asciiMaze',
    label: 'asciiMaze',
    title: 'ASCII Maze (NeatapticTS)',
    sourceDir: path.resolve('test', 'examples', 'asciiMaze'),
  },
  {
    dirName: 'flappy_bird',
    label: 'flappy_bird',
    title: 'Flappy Bird (NeatapticTS)',
    sourceDir: path.resolve('test', 'examples', 'flappy_bird'),
  },
];

/**
 * Runs the example-copy workflow.
 *
 * Step 1 copies published browser entrypoints for all known examples.
 * Step 2 rebuilds the landing page from the demos that were actually copied.
 *
 * @returns Promise resolved when copying and landing-page generation complete.
 */
async function main(): Promise<void> {
  const publishedExamples = (
    await Promise.all(EXAMPLE_DEFINITIONS.map(copyExampleEntryPoint))
  ).filter(isPublishedExample);

  await writeExamplesLandingPage(publishedExamples);
}

/**
 * Copies one example entrypoint into `docs/examples/<name>/index.html`.
 *
 * @param exampleDefinition - Example publishing definition.
 * @returns Published example metadata when the example entrypoint was copied,
 * otherwise `null`.
 */
async function copyExampleEntryPoint(
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
  if (!hasSourceIndex) {
    console.warn(
      `${DOCS_EXAMPLES_LOG_PREFIX} ${exampleDefinition.dirName} ${EXAMPLE_ENTRY_FILE_NAME} missing`,
    );
    return null;
  }

  const destinationDirectoryPath = path.join(
    DOCS_EXAMPLES_DIR,
    exampleDefinition.dirName,
  );
  const destinationIndexPath = path.join(
    destinationDirectoryPath,
    EXAMPLE_ENTRY_FILE_NAME,
  );

  await mkdir(destinationDirectoryPath, { recursive: true });
  await copyFile(sourceIndexPath, destinationIndexPath);

  console.log(
    `${DOCS_EXAMPLES_LOG_PREFIX} Copied ${exampleDefinition.dirName} ${EXAMPLE_ENTRY_FILE_NAME}`,
  );

  return {
    dirName: exampleDefinition.dirName,
    label: exampleDefinition.label,
    title: exampleDefinition.title,
  };
}

/**
 * Writes the shared docs/examples landing page.
 *
 * @param publishedExamples - Examples that were published in the current run.
 * @returns Promise resolved when the landing page has been written.
 */
async function writeExamplesLandingPage(
  publishedExamples: readonly PublishedExample[],
): Promise<void> {
  await mkdir(DOCS_EXAMPLES_DIR, { recursive: true });

  const examplesLandingPageHtml =
    buildExamplesLandingPageHtml(publishedExamples);
  await writeFile(
    path.join(DOCS_EXAMPLES_DIR, EXAMPLE_ENTRY_FILE_NAME),
    examplesLandingPageHtml,
    'utf8',
  );

  console.log(`${DOCS_EXAMPLES_LOG_PREFIX} Wrote examples landing page`);
}

/**
 * Builds the shared examples landing page HTML.
 *
 * @param publishedExamples - Examples that were published in the current run.
 * @returns Rendered landing page HTML.
 */
function buildExamplesLandingPageHtml(
  publishedExamples: readonly PublishedExample[],
): string {
  const examplesListMarkup = buildExamplesListMarkup(publishedExamples);

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Examples • NeatapticTS Docs</title>
    <link rel="stylesheet" href="${DOCS_THEME_STYLESHEET_PATH}" />
  </head>
  <body>
    <header class="topbar">
      <div class="inner">
        <div class="brand"><a href="${DOCS_HOME_PATH}">NeatapticTS</a></div>
        <nav class="main-nav">
          <a href="${DOCS_HOME_PATH}">Home</a>
          <a href="${DOCS_HOME_PATH}">Docs</a>
          <a href="${EXAMPLES_HOME_PATH}" class="active">Examples</a>
          <a href="${GITHUB_REPOSITORY_URL}" target="_blank" rel="noopener">GitHub</a>
        </nav>
      </div>
    </header>
    <div class="layout">
      <main class="content">
        <h1>Examples</h1>
        <p>Interactive browser demos built from this repository:</p>
        ${examplesListMarkup}
        <footer class="site-footer">Generated from source JSDoc • <a href="${GITHUB_REPOSITORY_URL}">GitHub</a></footer>
      </main>
    </div>
  </body>
</html>`;
}

/**
 * Builds the published examples list markup.
 *
 * @param publishedExamples - Examples that were published in the current run.
 * @returns HTML list markup.
 */
function buildExamplesListMarkup(
  publishedExamples: readonly PublishedExample[],
): string {
  if (publishedExamples.length === 0) {
    return '<p>No interactive examples were published in this run.</p>';
  }

  const listItemsMarkup = publishedExamples
    .toSorted((leftExample, rightExample) =>
      leftExample.title.localeCompare(rightExample.title),
    )
    .map(
      (publishedExample) =>
        `<li><a href="./${publishedExample.dirName}/${EXAMPLE_ENTRY_FILE_NAME}">${publishedExample.title}</a> <span class="demo-path">(examples/${publishedExample.label})</span></li>`,
    )
    .join('');

  return `<ul>${listItemsMarkup}</ul>`;
}

/**
 * Resolves whether a filesystem path exists.
 *
 * @param targetPath - Filesystem path to check.
 * @returns True when the path exists.
 */
async function pathExists(targetPath: string): Promise<boolean> {
  try {
    await access(targetPath);
    return true;
  } catch {
    return false;
  }
}

/**
 * Narrows `Promise.all` output to the published example shape.
 *
 * @param publishedExample - Candidate example result.
 * @returns `true` when the example was published successfully.
 */
function isPublishedExample(
  publishedExample: PublishedExample | null,
): publishedExample is PublishedExample {
  return publishedExample !== null;
}

main().catch((error: unknown) => {
  console.error(`${DOCS_EXAMPLES_LOG_PREFIX} Failed:`, error);
  process.exit(1);
});
