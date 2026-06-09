/*
 * HTML builders and source-page rewriters for the copy-examples boundary.
 *
 * All functions here are pure (no I/O). They consume example metadata and
 * return HTML strings that the copy and landing-page layers then write to disk.
 */

import type {
  ExampleDefinition,
  PublishedExample,
} from './copy-examples.types.js';
import {
  DOCS_HOME_PATH,
  DOCS_THEME_STYLESHEET_PATH,
  EXAMPLE_ENTRY_FILE_NAME,
  EXAMPLE_PAGE_DOCS_HOME_PATH,
  EXAMPLE_PAGE_EXAMPLES_HOME_PATH,
  EXAMPLE_PAGE_THEME_STYLESHEET_PATH,
  EXAMPLE_PAGE_TOOLTIP_RUNTIME_PATH,
  EXAMPLES_HOME_PATH,
  GENERATED_EXAMPLE_PAGE_COMMENT_PREFIX,
  GITHUB_MAIN_BRANCH_URL,
  GITHUB_REPOSITORY_URL,
  REPO_EXAMPLE_THEME_STYLESHEET_PATH,
  REPO_EXAMPLE_TOOLTIP_RUNTIME_PATH,
} from './copy-examples.constants.js';

/**
 * Rewrite source example HTML so the published docs copy points at docs-owned
 * assets rather than the repo-relative asset paths used during local development.
 *
 * Three substitutions are performed:
 * 1. The source theme stylesheet path is replaced with the docs-level path.
 * 2. The source tooltip runtime path is replaced with the docs-level path.
 * 3. Dynamic asset references that prefix `'../../docs/assets/` are rewritten
 *    to `'../../assets/` to match the published layout.
 *
 * @param examplePageHtml - Raw HTML read from the source `examples/<name>/index.html`.
 * @returns HTML with all repo-local asset paths rewritten for docs publication.
 */
export function rewriteExamplePageForDocs(examplePageHtml: string): string {
  return examplePageHtml
    .replaceAll(
      REPO_EXAMPLE_THEME_STYLESHEET_PATH,
      EXAMPLE_PAGE_THEME_STYLESHEET_PATH,
    )
    .replaceAll(
      REPO_EXAMPLE_TOOLTIP_RUNTIME_PATH,
      EXAMPLE_PAGE_TOOLTIP_RUNTIME_PATH,
    )
    .replaceAll("? '../../docs/assets/", "? '../../assets/");
}

/**
 * Prepends a generated-file warning comment to a published example page.
 *
 * The comment directs maintainers to edit the source file rather than the
 * generated docs copy. The check for the prefix prevents double-stamping when
 * the same page is processed more than once in a single run.
 *
 * @param exampleDefinition - Example publishing definition used to build the
 *   source and destination paths shown in the comment.
 * @param examplePageHtml - Final HTML that will be written into `docs/examples/`.
 * @returns HTML prefixed with the generated-file warning comment.
 */
export function addGeneratedExamplePageComment(
  exampleDefinition: ExampleDefinition,
  examplePageHtml: string,
): string {
  if (examplePageHtml.startsWith(GENERATED_EXAMPLE_PAGE_COMMENT_PREFIX)) {
    return examplePageHtml;
  }

  const sourceEntryPath = `examples/${exampleDefinition.dirName}/${EXAMPLE_ENTRY_FILE_NAME}`;
  const generatedEntryPath = `docs/examples/${exampleDefinition.dirName}/${EXAMPLE_ENTRY_FILE_NAME}`;

  return `${GENERATED_EXAMPLE_PAGE_COMMENT_PREFIX} Edit ${sourceEntryPath}, not ${generatedEntryPath}. -->\n${examplePageHtml}`;
}

/**
 * Builds the local placeholder page for a source-first example that has no
 * browser-hosted `index.html`.
 *
 * The generated page shows the example description, an optional local run
 * command, and a link to the source folder on GitHub rather than an in-browser
 * simulation.
 *
 * @param exampleDefinition - Example publishing definition.
 * @returns Rendered HTML page string.
 */
export function buildSourceFirstExamplePageHtml(
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
 * Builds the shared `docs/examples/index.html` landing page HTML.
 *
 * The page groups published examples into "Starter Examples" and "Flagship
 * Demos" sections, each sorted alphabetically by title within the section.
 *
 * @param publishedExamples - Examples that were published in the current run.
 * @returns Rendered landing page HTML string.
 */
export function buildExamplesLandingPageHtml(
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
 * Builds the combined section markup for both example categories.
 *
 * Returns a no-examples message when `publishedExamples` is empty, which
 * guards against rendering an entirely blank landing page.
 *
 * @param publishedExamples - Examples that were published in the current run.
 * @returns HTML string containing zero, one, or two category sections.
 */
export function buildExamplesSectionsMarkup(
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
 * Returns an empty string when the category has no published examples so that
 * empty sections are cleanly elided from the output.
 *
 * @param heading - Section `<h2>` heading text.
 * @param description - One-sentence description rendered below the heading.
 * @param publishedExamples - Published examples belonging to this category.
 * @returns HTML section markup, or an empty string when the category is empty.
 */
export function buildExamplesCategoryMarkup(
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
 * Builds the demo-card markup for one published example.
 *
 * The card includes the example title, description, an optional local run
 * command snippet, a primary action link (browser demo or example page), and
 * a source link to GitHub.
 *
 * @param publishedExample - Published example metadata.
 * @returns HTML `<article>` card markup.
 */
export function buildExampleListItemMarkup(
  publishedExample: PublishedExample,
): string {
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
