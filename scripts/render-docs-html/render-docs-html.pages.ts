/*
 * Page discovery, markdown rendering, and HTML shell emission for generated docs.
 *
 * This chapter owns the page-level pipeline: find README files, collect their
 * metadata, render markdown with stable anchors, assemble the left rail and TOC,
 * and emit the final HTML shell.
 */

import { setTimeout } from 'node:timers/promises';
import path from 'path';
import fg from 'fast-glob';
import fs from 'fs-extra';
import { marked } from 'marked';

import {
  buildDocsSidebarHtml,
  buildExamplesTocLinksHtml,
} from './render-docs-html.navigation.js';
import {
  buildMermaidBootstrapScript,
  collectMermaidBlocks,
  renderMermaidDiagramPlaceholder,
} from './render-docs-html.mermaid.js';
import {
  buildRelativeDocsHref,
  DOCS_DIR,
  escapeHtml,
  hasPublishedDocsPage,
  rewriteCompiledDocsContentLinks,
  slugify,
} from './render-docs-html.shared.js';
import type {
  MermaidBlockReference,
  PageMeta,
} from './render-docs-html.types.js';

const RETRIABLE_FILE_SYSTEM_ERROR_CODES = new Set([
  'UNKNOWN',
  'EPERM',
  'EBUSY',
  'EACCES',
]);
const DOCS_WRITE_MAX_ATTEMPTS = 6;
const DOCS_WRITE_INITIAL_RETRY_DELAY_MS = 120;

interface TocFileHeading {
  file: string;
  anchor: string;
  symbols: { name: string; anchor: string }[];
}

/** Collects markdown page metadata and Mermaid fences from the docs tree. */
export async function collectDocsPages(): Promise<{
  pages: PageMeta[];
  mermaidBlocks: MermaidBlockReference[];
}> {
  const readmes = await fg(['**/README.md'], { cwd: DOCS_DIR, absolute: true });
  const pages: PageMeta[] = [];
  const mermaidBlocks: MermaidBlockReference[] = [];

  for (const markdownFilePath of readmes) {
    const markdown = await fs.readFile(markdownFilePath, 'utf8');
    const title =
      mdTitle(markdown) ||
      path.relative(DOCS_DIR, path.dirname(markdownFilePath)) ||
      'Documentation';
    const relDir = path
      .relative(DOCS_DIR, path.dirname(markdownFilePath))
      .replace(/\\/g, '/');

    pages.push({
      abs: markdownFilePath,
      relDir,
      title,
      markdown,
    });
    mermaidBlocks.push(...collectMermaidBlocks(markdown, markdownFilePath));
  }

  return { pages, mermaidBlocks };
}

/** Emits the generated HTML page for every discovered markdown page. */
export async function emitDocsPages(pages: readonly PageMeta[]): Promise<void> {
  const generatedPageDirectories = new Set(pages.map((page) => page.relDir));

  for (const page of pages) {
    const pageHtml = renderDocsPage(page, pages, generatedPageDirectories);
    const outputFilePath = path.join(path.dirname(page.abs), 'index.html');
    await writeFileWithRetry(outputFilePath, pageHtml, 'utf8');
  }

  console.log('HTML docs generated.');
}

/** Renders one docs page shell. */
function renderDocsPage(
  page: PageMeta,
  pages: readonly PageMeta[],
  generatedPageDirectories: ReadonlySet<string>,
): string {
  const tableOfContents = extractFileHeadings(page.markdown);
  const renderedBody = renderMarkdownToHtml(page.markdown);
  const htmlBody = rewriteCompiledDocsContentLinks(
    renderedBody.htmlBody,
    page.relDir,
    generatedPageDirectories,
  );
  const rootExamplesTocHtml = buildExamplesTocLinksHtml({
    currentDir: page.relDir,
    generatedPageDirectories,
    hasPublishedDocsPage,
    buildRelativeDocsHref,
    escapeHtml,
  });
  const sidebarHtml = buildDocsSidebarHtml({
    currentDir: page.relDir,
    pages,
    generatedPageDirectories,
    hasPublishedDocsPage,
    buildRelativeDocsHref,
    escapeHtml,
  });
  const tocHtml = buildPageTocHtml(
    page.relDir,
    tableOfContents,
    rootExamplesTocHtml,
  );
  const relativePathToRoot = path
    .relative(path.dirname(page.abs), DOCS_DIR)
    .replace(/\\/g, '/');
  const cssHref =
    (relativePathToRoot ? `${relativePathToRoot}/` : '') + 'assets/theme.css';
  const tooltipRuntimeHref =
    (relativePathToRoot ? `${relativePathToRoot}/` : '') +
    'assets/theme-tooltips.js';
  const mermaidBootstrapScript = renderedBody.hasMermaidDiagram
    ? buildMermaidBootstrapScript(relativePathToRoot)
    : '';
  const examplesHref = `${relativePathToRoot || '.'}/examples/index.html`;
  const onExamplesPage = page.relDir.startsWith('examples');
  const docsActiveClass = !onExamplesPage ? ' class="active"' : '';
  const examplesActiveClass = onExamplesPage ? ' class="active"' : '';

  return `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>${
    page.title
  } – NeatapticTS Docs</title><meta name="viewport" content="width=device-width,initial-scale=1">\n<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Raleway:wght@400;600;700&family=Open+Sans:wght@400;600&display=swap" rel="stylesheet">\n<link rel="stylesheet" href="${cssHref}"><script defer src="${tooltipRuntimeHref}"></script></head><body class="${
    page.relDir === '' ? 'is-root' : ''
  }">\n<header class="topbar"><div class="inner topbar-inner-connected"><div class="brand"><a class="brand-link" href="${
    relativePathToRoot || '.'
  }/index.html"><span class="brand-title-lockup" aria-hidden="true"><span class="brand-title-row brand-title-row-top"><span class="brand-title-corner">╔══</span><span class="brand-label">NeatapticTS</span><span class="brand-title-corner">══╗</span></span></span><span class="brand-title-accessible">NeatapticTS</span></a></div><nav class="main-nav"><a href="${
    relativePathToRoot || '.'
  }/index.html">Home</a><a href="${
    relativePathToRoot || '.'
  }/index.html"${docsActiveClass}>Docs</a><a href="${examplesHref}"${examplesActiveClass}>Examples</a><a href="https://github.com/reicek/NeatapticTS" target="_blank" rel="noopener">GitHub</a></nav></div></header>\n<div class="layout"><aside class="sidebar docs-panel docs-panel-left">${sidebarHtml}</aside><main class="content docs-panel docs-panel-center">${htmlBody}<footer class="site-footer">Generated from source JSDoc • <a href="https://github.com/reicek/NeatapticTS">GitHub</a></footer></main><aside class="toc docs-panel docs-panel-right">${tocHtml}</aside></div>${buildDocsLayoutInteractionScript()}${mermaidBootstrapScript}</body></html>`;
}

/** Extracts file and symbol headings for the page table of contents. */
function extractFileHeadings(markdown: string): TocFileHeading[] {
  const fileHeadings: TocFileHeading[] = [];
  const markdownLines = markdown.split(/\r?\n/);
  let currentFileHeading: TocFileHeading | null = null;

  for (const markdownLine of markdownLines) {
    const fileMatch = /^##\s+(.+\.ts)\s*$/.exec(markdownLine);
    if (fileMatch) {
      const fileName = fileMatch[1];
      currentFileHeading = {
        file: fileName,
        anchor: slugify(fileName),
        symbols: [],
      };
      fileHeadings.push(currentFileHeading);
      continue;
    }

    const symbolMatch = /^###\s+(.+)\s*$/.exec(markdownLine);
    if (symbolMatch && currentFileHeading) {
      const symbolName = symbolMatch[1];
      currentFileHeading.symbols.push({
        name: symbolName,
        anchor: slugify(symbolName),
      });
    }
  }

  return fileHeadings;
}

/** Renders markdown to HTML with stable heading ids and Mermaid placeholders. */
function renderMarkdownToHtml(markdown: string): {
  htmlBody: string;
  hasMermaidDiagram: boolean;
} {
  const renderer = new marked.Renderer();
  const originalCodeRenderer = renderer.code?.bind(renderer);
  let hasMermaidDiagram = false;

  (renderer as unknown as { heading: (token: any) => string }).heading = ({
    text,
    depth,
    raw,
  }: any) => {
    const source = (raw ?? text ?? '')
      .toString()
      .replace(/<[^>]+>/g, '')
      .trim();
    const id = slugify(source);
    return `<h${depth} id="${id}">${text}</h${depth}>`;
  };

  (renderer as unknown as { code: (token: any) => string }).code = (
    token: any,
  ) => {
    const { text, lang } = token;
    if ((lang ?? '').toString().trim().toLowerCase() === 'mermaid') {
      hasMermaidDiagram = true;
      return renderMermaidDiagramPlaceholder((text ?? '').toString());
    }

    return originalCodeRenderer?.(token) ?? '';
  };

  return {
    htmlBody: marked.parse(markdown, { async: false, renderer }),
    hasMermaidDiagram,
  };
}

/** Builds the right-rail TOC for one page. */
function buildPageTocHtml(
  currentDir: string,
  fileHeadings: readonly TocFileHeading[],
  rootExamplesTocHtml: string,
): string {
  if (fileHeadings.length > 0) {
    return `<div class="page-toc"><h2>Files</h2>${fileHeadings
      .map(
        (fileHeading) =>
          `<div class="toc-file"><a href="#${fileHeading.anchor}">${fileHeading.file}</a>${
            fileHeading.symbols.length > 0
              ? `<ul>${fileHeading.symbols
                  .map(
                    (symbol) =>
                      `<li><a href="#${symbol.anchor}">${symbol.name}</a></li>`,
                  )
                  .join('')}</ul>`
              : ''
          }</div>`,
      )
      .join('')}</div>`;
  }

  if (currentDir === '' && rootExamplesTocHtml) {
    return `<div class="page-toc"><h2>Examples</h2><div class="toc-file">${rootExamplesTocHtml}</div></div>`;
  }

  return '';
}

/** Keeps the panel-expansion interaction script out of the root entrypoint. */
function buildDocsLayoutInteractionScript(): string {
  return `<script>
const docsPanels = Array.from(document.querySelectorAll('.docs-panel'));

function setActiveDocsPanel(nextPanel) {
  docsPanels.forEach((panel) => {
    panel.classList.toggle('is-active', panel === nextPanel);
  });
}

document.addEventListener('click', (event) => {
  const target = event.target;
  if (!(target instanceof Element)) {
    setActiveDocsPanel(null);
    return;
  }

  const activePanel = target.closest('.docs-panel');
  setActiveDocsPanel(activePanel instanceof HTMLElement ? activePanel : null);
});

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    setActiveDocsPanel(null);
  }
});
</script>`;
}

/** Returns the first markdown heading or undefined when absent. */
function mdTitle(markdown: string): string | undefined {
  return markdown.match(/^#\s+(.+)$/m)?.[1];
}

/** Narrows unknown errors to retryable filesystem failures. */
function isRetriableFileSystemError(
  error: unknown,
): error is NodeJS.ErrnoException {
  if (!(error instanceof Error)) {
    return false;
  }

  const code = (error as NodeJS.ErrnoException).code;
  return Boolean(code && RETRIABLE_FILE_SYSTEM_ERROR_CODES.has(code));
}

/** Writes a page file with bounded retry logic for transient Windows locks. */
async function writeFileWithRetry(
  filePath: string,
  content: string,
  encoding: BufferEncoding,
): Promise<void> {
  let retryDelayMilliseconds = DOCS_WRITE_INITIAL_RETRY_DELAY_MS;

  for (
    let attemptNumber = 1;
    attemptNumber <= DOCS_WRITE_MAX_ATTEMPTS;
    attemptNumber += 1
  ) {
    try {
      await fs.writeFile(filePath, content, encoding);
      return;
    } catch (error: unknown) {
      const canRetry =
        isRetriableFileSystemError(error) &&
        attemptNumber < DOCS_WRITE_MAX_ATTEMPTS;
      if (!canRetry) {
        throw error;
      }

      await setTimeout(retryDelayMilliseconds);
      retryDelayMilliseconds = Math.min(retryDelayMilliseconds * 2, 1_000);
    }
  }
}
