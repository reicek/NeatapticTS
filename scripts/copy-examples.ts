/*
 * Copies browser-viewable examples into docs/examples/* so GitHub Pages can
 * publish them alongside the generated documentation site.
 *
 * The script intentionally keeps one narrow contract: copy each example's
 * browser entrypoint into its published docs folder, then rebuild the shared
 * examples landing page from the demos that were actually published.
 */

import { access, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';

const DOCS_EXAMPLES_LOG_PREFIX = '[docs:examples]';
const DOCS_EXAMPLES_DIR = path.resolve('docs', 'examples');
const EXAMPLE_ENTRY_FILE_NAME = 'index.html';
const DOCS_THEME_STYLESHEET_PATH = '../assets/theme.css';
const REPO_EXAMPLE_THEME_STYLESHEET_PATH = '../../docs/assets/theme.css';
const REPO_EXAMPLE_TOOLTIP_RUNTIME_PATH = '../../docs/assets/theme-tooltips.js';
const DOCS_HOME_PATH = '../index.html';
const EXAMPLES_HOME_PATH = './index.html';
const EXAMPLE_PAGE_THEME_STYLESHEET_PATH = '../../assets/theme.css';
const EXAMPLE_PAGE_TOOLTIP_RUNTIME_PATH = '../../assets/theme-tooltips.js';
const EXAMPLE_PAGE_DOCS_HOME_PATH = '../../index.html';
const EXAMPLE_PAGE_EXAMPLES_HOME_PATH = '../index.html';
const GITHUB_REPOSITORY_URL = 'https://github.com/reicek/NeatapticTS';
const GITHUB_MAIN_BRANCH_URL = `${GITHUB_REPOSITORY_URL}/tree/main`;

type ExampleCategory = 'flagship' | 'starter';

interface ExampleDefinition {
  category: ExampleCategory;
  description: string;
  dirName: string;
  label: string;
  runCommand?: string;
  title: string;
  sourceDir: string;
}

interface PublishedExample {
  category: ExampleCategory;
  description: string;
  dirName: string;
  hasBrowserEntry: boolean;
  label: string;
  runCommand?: string;
  title: string;
}

const EXAMPLE_DEFINITIONS: readonly ExampleDefinition[] = [
  {
    category: 'starter',
    description:
      'The smallest public-network walkthrough: build one compact feed-forward network, run one inference pass, and inspect the result shape immediately.',
    dirName: 'helloNetwork',
    label: 'helloNetwork',
    runCommand: 'npm run example:hello-network',
    title: 'Hello Network (NeatapticTS)',
    sourceDir: path.resolve('examples', 'helloNetwork'),
  },
  {
    category: 'starter',
    description:
      'A bounded feed-forward NEAT run on XOR that now reaches a solved state and shows the smallest end-to-end evolutionary loop in the repo.',
    dirName: 'evolveXor',
    label: 'evolveXor',
    runCommand: 'npm run example:evolve-xor',
    title: 'Evolve XOR (NeatapticTS)',
    sourceDir: path.resolve('examples', 'evolveXor'),
  },
  {
    category: 'starter',
    description:
      'A tiny LSTM example that feeds the same input sequence three times to show state accumulation, the effect of clear() after a run, and the carryover effect when clear() is skipped.',
    dirName: 'sequenceReset',
    label: 'sequenceReset',
    runCommand: 'npx tsx examples/sequenceReset/run.ts',
    title: 'Sequence Reset (NeatapticTS)',
    sourceDir: path.resolve('examples', 'sequenceReset'),
  },
  {
    category: 'flagship',
    description:
      'A compact navigation lab with browser playback, reward shaping, telemetry-rich search, and a deliberately small observation budget.',
    dirName: 'asciiMaze',
    label: 'asciiMaze',
    title: 'ASCII Maze (NeatapticTS)',
    sourceDir: path.resolve('examples', 'asciiMaze'),
  },
  {
    category: 'flagship',
    description:
      'A fast browser neuroevolution system with worker-backed playback, temporal observations, and a full inspectable runtime architecture.',
    dirName: 'flappy_bird',
    label: 'flappy_bird',
    title: 'Flappy Bird (NeatapticTS)',
    sourceDir: path.resolve('examples', 'flappy_bird'),
  },
  {
    category: 'flagship',
    description:
      'A published browser contract preview for the tiny sequence-learning chat demo, with a visible chat shell, staged progress, and the reused Flappy visualizer boundary.',
    dirName: 'neatChat',
    label: 'neatChat',
    runCommand: 'npx tsx examples/neatChat/run.ts',
    title: 'NEATchat (NeatapticTS)',
    sourceDir: path.resolve('examples', 'neatChat'),
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

  await removeRetiredPublishedExamples(publishedExamples);
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
    const docsReadyIndexHtml = rewriteExamplePageForDocs(sourceIndexHtml);

    await writeFile(destinationIndexPath, docsReadyIndexHtml, 'utf8');

    console.log(
      `${DOCS_EXAMPLES_LOG_PREFIX} Copied ${exampleDefinition.dirName} ${EXAMPLE_ENTRY_FILE_NAME}`,
    );
  } else {
    await writeFile(
      destinationIndexPath,
      buildSourceFirstExamplePageHtml(exampleDefinition),
      'utf8',
    );

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
 * Removes stale docs/examples folders for examples that are no longer published.
 *
 * @param publishedExamples - Examples kept in the current docs publication set.
 * @returns Promise resolved when stale folders are removed.
 */
async function removeRetiredPublishedExamples(
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

/**
 * Builds the shared examples landing page HTML.
 *
 * @param publishedExamples - Examples that were published in the current run.
 * @returns Rendered landing page HTML.
 */
function buildExamplesLandingPageHtml(
  publishedExamples: readonly PublishedExample[],
): string {
  const examplesSectionsMarkup = buildExamplesSectionsMarkup(publishedExamples);

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Examples • NeatapticTS Docs</title>
    <link rel="stylesheet" href="${DOCS_THEME_STYLESHEET_PATH}" />
    <script defer src="../assets/theme-tooltips.js"></script>
  </head>
  <body class="demo-page">
    <header class="topbar">
      <div class="inner topbar-inner-connected">
        <div class="brand"><a href="${DOCS_HOME_PATH}">NeatapticTS</a></div>
        <nav class="main-nav">
          <a href="${DOCS_HOME_PATH}">Home</a>
          <a href="${DOCS_HOME_PATH}">Docs</a>
          <a href="${EXAMPLES_HOME_PATH}" class="active">Examples</a>
          <a href="${GITHUB_REPOSITORY_URL}" target="_blank" rel="noopener">GitHub</a>
        </nav>
      </div>
    </header>
    <main class="demo-main">
      <section class="hero">
        <p class="eyebrow">Neon Demo Arcade</p>
        <h1>Examples</h1>
        <p>Every browser-facing example currently shipped from this repository, from the smallest starter walkthroughs to the larger flagship demos.</p>
      </section>
      <section class="example-hub">
        ${examplesSectionsMarkup}
      </section>
      <footer class="site-footer">Generated from source JSDoc • <a href="${GITHUB_REPOSITORY_URL}">GitHub</a></footer>
    </main>
  </body>
</html>`;
}

/**
 * Builds grouped examples markup for the landing page.
 *
 * @param publishedExamples - Examples that were published in the current run.
 * @returns HTML section markup.
 */
function buildExamplesSectionsMarkup(
  publishedExamples: readonly PublishedExample[],
): string {
  if (publishedExamples.length === 0) {
    return '<p>No interactive examples were published in this run.</p>';
  }

  return [
    buildExamplesCategoryMarkup(
      'Starter Examples',
      'Small, focused examples that teach one public API story at a time.',
      publishedExamples.filter(
        (publishedExample) => publishedExample.category === 'starter',
      ),
    ),
    buildExamplesCategoryMarkup(
      'Flagship Demos',
      'Larger browser-facing systems that show the library under more realistic pressure.',
      publishedExamples.filter(
        (publishedExample) => publishedExample.category === 'flagship',
      ),
    ),
  ].join('');
}

/**
 * Builds one category section for the examples landing page.
 *
 * @param heading - Section heading.
 * @param description - Section description.
 * @param publishedExamples - Published examples for the category.
 * @returns HTML section markup.
 */
function buildExamplesCategoryMarkup(
  heading: string,
  description: string,
  publishedExamples: readonly PublishedExample[],
): string {
  if (publishedExamples.length === 0) {
    return '';
  }

  const listItemsMarkup = publishedExamples
    .toSorted((leftExample, rightExample) =>
      leftExample.title.localeCompare(rightExample.title),
    )
    .map(buildExampleListItemMarkup)
    .join('');

  return `<section class="demo-category"><div class="demo-category-header"><h2>${heading}</h2><p>${description}</p></div><div class="demo-card-grid">${listItemsMarkup}</div></section>`;
}

/**
 * Builds one examples list item.
 *
 * @param publishedExample - Published example metadata.
 * @returns HTML list item markup.
 */
function buildExampleListItemMarkup(publishedExample: PublishedExample): string {
  const localExampleLink = `./${publishedExample.dirName}/${EXAMPLE_ENTRY_FILE_NAME}`;
  const actionLabel = publishedExample.hasBrowserEntry
    ? 'Open browser demo'
    : 'Open example page';
  const runCommandMarkup = publishedExample.runCommand
    ? `<p class="demo-card-command"><strong>Run locally:</strong> <code>${publishedExample.runCommand}</code></p>`
    : '';
  const sourceUrl = `${GITHUB_MAIN_BRANCH_URL}/examples/${publishedExample.dirName}`;

  return `<article class="demo-card"><span class="demo-card-path">examples/${publishedExample.label}</span><h3>${publishedExample.title}</h3><p>${publishedExample.description}</p>${runCommandMarkup}<div class="demo-card-actions"><a class="demo-link-button" href="${localExampleLink}">${actionLabel}</a><a href="${sourceUrl}" target="_blank" rel="noopener">Source</a></div></article>`;
}

/**
 * Builds the local page for a source-first example without a browser host.
 *
 * @param exampleDefinition - Example publishing definition.
 * @returns Rendered HTML page.
 */
function buildSourceFirstExamplePageHtml(
  exampleDefinition: ExampleDefinition,
): string {
  const sourceUrl = `${GITHUB_MAIN_BRANCH_URL}/examples/${exampleDefinition.dirName}`;
  const runCommandMarkup = exampleDefinition.runCommand
    ? `<p><strong>Run from the repo root:</strong></p><pre><code>${exampleDefinition.runCommand}</code></pre>`
    : '';

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${exampleDefinition.title}</title>
    <link rel="stylesheet" href="${EXAMPLE_PAGE_THEME_STYLESHEET_PATH}" />
    <script defer src="${EXAMPLE_PAGE_TOOLTIP_RUNTIME_PATH}"></script>
  </head>
  <body class="demo-page">
    <header class="topbar">
      <div class="inner topbar-inner-connected">
        <div class="brand"><a href="${EXAMPLE_PAGE_DOCS_HOME_PATH}">NeatapticTS</a></div>
        <nav class="main-nav">
          <a href="${EXAMPLE_PAGE_DOCS_HOME_PATH}">Home</a>
          <a href="${EXAMPLE_PAGE_DOCS_HOME_PATH}">Docs</a>
          <a href="${EXAMPLE_PAGE_EXAMPLES_HOME_PATH}" class="active">Examples</a>
          <a href="${GITHUB_REPOSITORY_URL}" target="_blank" rel="noopener">GitHub</a>
        </nav>
      </div>
    </header>
    <main class="demo-main">
      <section class="hero">
        <p class="eyebrow">Source-First Example</p>
        <h1>${exampleDefinition.title}</h1>
        <p>${exampleDefinition.description}</p>
      </section>
      <section class="demo-category">
        <div class="demo-category-header">
          <h2>Open It Locally</h2>
          <p>This example is source-first rather than browser-hosted, so this page points you at the local run command and source entrypoint instead of an in-browser simulation.</p>
        </div>
        <div class="demo-card-grid">
          <article class="demo-card">
            <span class="demo-card-path">examples/${exampleDefinition.label}</span>
            <h3>Repository entrypoint</h3>
            ${runCommandMarkup}
            <div class="demo-card-actions">
              <a class="demo-link-button" href="${sourceUrl}" target="_blank" rel="noopener">Open source folder</a>
              <a href="${EXAMPLE_PAGE_EXAMPLES_HOME_PATH}">Back to all examples</a>
            </div>
          </article>
        </div>
      </section>
    </main>
  </body>
</html>`;
}

/**
 * Rewrite source example HTML so the published docs copy points at docs-owned assets.
 *
 * @param examplePageHtml - Source example host HTML.
 * @returns HTML rewritten for docs/examples publication.
 */
function rewriteExamplePageForDocs(examplePageHtml: string): string {
  return examplePageHtml
    .replaceAll(
      REPO_EXAMPLE_THEME_STYLESHEET_PATH,
      EXAMPLE_PAGE_THEME_STYLESHEET_PATH,
    )
    .replaceAll(
      REPO_EXAMPLE_TOOLTIP_RUNTIME_PATH,
      EXAMPLE_PAGE_TOOLTIP_RUNTIME_PATH,
    )
    .replaceAll(
      "? '../../docs/assets/",
      "? '../../assets/",
    );
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
