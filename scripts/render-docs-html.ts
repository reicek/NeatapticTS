/*
 * Converts every README.md inside docs/ (including root copy) into an index.html in the same directory.
 * Usage: npm run docs:html
 */
import fg from 'fast-glob';
import path from 'path';
import fs from 'fs-extra';
import { marked } from 'marked';

const DOCS_DIR = path.resolve('docs');
const THEME_CSS_SOURCE_PATH = path.resolve('scripts', 'assets', 'theme.css');
const THEME_CSS_OUTPUT_PATH = path.join(DOCS_DIR, 'assets', 'theme.css');
const NN_IMAGE_SOURCE_PATH = path.resolve('nn.jpg');
const NN_IMAGE_FALLBACK_SOURCE_PATH = path.resolve('scripts', 'assets', 'nn.jpg');
const NN_IMAGE_OUTPUT_PATH = path.join(DOCS_DIR, 'nn.jpg');
const EXAMPLE_DEMOS = [
  { dir: 'examples/asciiMaze', label: 'asciiMaze' },
  { dir: 'examples/flappy_bird', label: 'flappy_bird' },
];

const RETRIABLE_FILE_SYSTEM_ERROR_CODES = new Set([
  'UNKNOWN',
  'EPERM',
  'EBUSY',
  'EACCES',
]);
const DOCS_WRITE_MAX_ATTEMPTS = 6;
const DOCS_WRITE_INITIAL_RETRY_DELAY_MS = 120;

function isRetriableFileSystemError(error: unknown): error is NodeJS.ErrnoException {
  if (!(error instanceof Error)) return false;
  const code = (error as NodeJS.ErrnoException).code;
  if (!code) return false;
  return RETRIABLE_FILE_SYSTEM_ERROR_CODES.has(code);
}

async function waitForRetryDelay(delayMilliseconds: number): Promise<void> {
  await new Promise((resolve) => {
    setTimeout(resolve, delayMilliseconds);
  });
}

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

      await waitForRetryDelay(retryDelayMilliseconds);
      retryDelayMilliseconds = Math.min(retryDelayMilliseconds * 2, 1_000);
    }
  }
}

async function ensureThemeCss(): Promise<void> {
  // Step 1: Ensure destination directory exists.
  await fs.ensureDir(path.dirname(THEME_CSS_OUTPUT_PATH));

  // Step 2: Copy static theme stylesheet used by generated docs pages.
  await fs.copyFile(THEME_CSS_SOURCE_PATH, THEME_CSS_OUTPUT_PATH);
}

async function ensureStaticDocsAssets(): Promise<void> {
  // Step 1: Ensure core theme assets are present.
  await ensureThemeCss();

  // Step 2: Ensure the README hero image resolves when served from `/docs/`.
  // The root README is copied into `docs/README.md`, so `<img src="nn.jpg">`
  // becomes a request for `/docs/nn.jpg` when hosted under that base path.
  const hasRootImage = await fs.pathExists(NN_IMAGE_SOURCE_PATH);
  const hasFallbackImage = await fs.pathExists(NN_IMAGE_FALLBACK_SOURCE_PATH);
  const sourcePath = hasRootImage
    ? NN_IMAGE_SOURCE_PATH
    : hasFallbackImage
      ? NN_IMAGE_FALLBACK_SOURCE_PATH
      : undefined;

  if (!sourcePath) return;
  await fs.copyFile(sourcePath, NN_IMAGE_OUTPUT_PATH);
}

function slugify(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/-{2,}/g, '-');
}

function buildExamplesLinksHtml(currentDir: string): string {
  return EXAMPLE_DEMOS.map((entry) => {
    const copiedExampleAbs = path.resolve(DOCS_DIR, entry.dir, 'index.html');
    if (!fs.existsSync(copiedExampleAbs)) return '';

    const relLink = path.posix.relative(currentDir || '.', entry.dir) || '.';
    const href = (relLink === '.' ? '.' : relLink) + '/index.html';
    return `<li><a href="${href}">${entry.label}</a></li>`;
  })
    .filter(Boolean)
    .join('');
}

async function main() {
  await ensureStaticDocsAssets();
  const readmes = await fg(['**/README.md'], { cwd: DOCS_DIR, absolute: true });

  // Collect metadata for navigation
  interface PageMeta {
    abs: string;
    relDir: string;
    title: string;
  }
  const pages: PageMeta[] = [];
  for (const mdFile of readmes) {
    const md = await fs.readFile(mdFile, 'utf8');
    const title =
      md.match(/^#\s+(.+)$/m)?.[1] ||
      path.relative(DOCS_DIR, path.dirname(mdFile)) ||
      'Documentation';
    const relDir = path
      .relative(DOCS_DIR, path.dirname(mdFile))
      .replace(/\\/g, '/');
    pages.push({ abs: mdFile, relDir, title });
  }

  // Build nav list; group top-level folders similar to original sections.
  const navHtmlFor = (currentDir: string) => {
    // Group by first segment
    interface Group {
      name: string;
      items: PageMeta[];
    }
    const groupsMap = new Map<string, Group>();
    for (const p of pages) {
      const seg = p.relDir.split('/')[0] || 'root';
      if (!groupsMap.has(seg)) groupsMap.set(seg, { name: seg, items: [] });
      groupsMap.get(seg)!.items.push(p);
    }
    const order = [
      'root',
      'architecture',
      'methods',
      'neat',
      'multithreading',
      'examples',
    ];
    const makeLink = (page: PageMeta) => {
      const isCurrent = page.relDir === currentDir;
      const relLink =
        path.posix.relative(currentDir || '.', page.relDir || '.') || '.';
      const href = (relLink === '.' ? '.' : relLink) + '/index.html';
      const label =
        page.relDir === '' ? 'Overview' : page.relDir.replace(/\\/g, '/');
      return `<li${
        isCurrent ? ' class="current"' : ''
      }><a href="${href}">${label}${isCurrent ? '' : ''}</a></li>`;
    };
    const demoLinksHtml = buildExamplesLinksHtml(currentDir);
    const groups = Array.from(groupsMap.values());
    if (demoLinksHtml && !groupsMap.has('examples')) {
      groups.push({ name: 'examples', items: [] });
    }
    const groupsHtml = groups
      .sort((a, b) => {
        const leftOrder = order.indexOf(a.name);
        const rightOrder = order.indexOf(b.name);
        const leftRank = leftOrder === -1 ? Number.MAX_SAFE_INTEGER : leftOrder;
        const rightRank = rightOrder === -1 ? Number.MAX_SAFE_INTEGER : rightOrder;
        return leftRank - rightRank || a.name.localeCompare(b.name);
      })
      .map((g) => {
        const items = g.items.sort((a, b) => a.relDir.localeCompare(b.relDir));
        if (g.name === 'root')
          return makeLink(items.find((i) => i.relDir === '')!);
        return `<li class="group"><div class="g-head">${
          g.name
        }</div><ul>${items.map(makeLink).join('')}${
          g.name === 'examples' ? demoLinksHtml : ''
        }</ul></li>`;
      })
      .join('');
    return `<ul class="sidebar-sections">${groupsHtml}</ul>`;
  };

  for (const meta of pages) {
    const md = await fs.readFile(meta.abs, 'utf8');
    // Extract headings for TOC (## file, ### symbol)
    const fileHeadings: {
      file: string;
      anchor: string;
      symbols: { name: string; anchor: string }[];
    }[] = [];
    const lines = md.split(/\r?\n/);
    let currentFile: {
      file: string;
      anchor: string;
      symbols: { name: string; anchor: string }[];
    } | null = null;
    for (const line of lines) {
      const fileMatch = /^##\s+(.+\.ts)\s*$/.exec(line);
      if (fileMatch) {
        const fileName = fileMatch[1];
        const anchor = slugify(fileName);
        currentFile = { file: fileName, anchor, symbols: [] };
        fileHeadings.push(currentFile);
        continue;
      }
      const symMatch = /^###\s+([A-Za-z0-9_]+)\s*$/.exec(line);
      if (symMatch && currentFile) {
        const sym = symMatch[1];
        currentFile.symbols.push({ name: sym, anchor: slugify(sym) });
      }
    }
    // Configure marked renderer with deterministic heading IDs so anchors match our TOC.
    const renderer = new marked.Renderer();
    const originalHeading = renderer.heading?.bind(renderer);
    // Marked >= v16 passes a single Heading token object { text, depth, raw, tokens }
    // See: https://marked.js.org/using_pro#renderer for updated signature.
    (renderer as any).heading = ({ text, depth, raw }: any) => {
      const source = (raw ?? text ?? '')
        .toString()
        .replace(/<[^>]+>/g, '')
        .trim();
      const id = slugify(source);
      return `<h${depth} id="${id}">${text}</h${depth}>`;
    };
    marked.use({ renderer });
    const htmlBody = marked.parse(md, { async: false });
    const rootExamplesTocHtml = buildExamplesLinksHtml(meta.relDir);
    const toc = fileHeadings.length
      ? `<div class="page-toc"><h2>Files</h2>${fileHeadings
          .map(
            (f) =>
              `<div class=\"toc-file\"><a href=\"#${f.anchor}\">${f.file}</a>${
                f.symbols.length
                  ? `<ul>${f.symbols
                      .map((s) => `<li><a href=#${s.anchor}>${s.name}</a></li>`)
                      .join('')}</ul>`
                  : ''
              }</div>`
          )
          .join('')}</div>`
            : meta.relDir === '' && rootExamplesTocHtml
            ? `<div class="page-toc"><h2>Examples</h2><div class="toc-file"><ul>${rootExamplesTocHtml}</ul></div></div>`
            : '';
    const outFile = path.join(path.dirname(meta.abs), 'index.html');
    const relToRoot = path
      .relative(path.dirname(meta.abs), DOCS_DIR)
      .replace(/\\/g, '/');
    const cssHref = (relToRoot ? relToRoot + '/' : '') + 'assets/theme.css';
    // Add Examples top-level nav; active when current dir starts with examples
    const examplesHref = (relToRoot || '.') + '/examples/index.html';
    const onExamples = meta.relDir.startsWith('examples');
    const docsActive = !onExamples ? ' class="active"' : '';
    const examplesActive = onExamples ? ' class="active"' : '';
    const page = `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>${
      meta.title
    } – NeatapticTS Docs</title><meta name="viewport" content="width=device-width,initial-scale=1">\n<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Raleway:wght@400;600;700&family=Open+Sans:wght@400;600&display=swap" rel="stylesheet">\n<link rel="stylesheet" href="${cssHref}"></head><body class="${
      meta.relDir === '' ? 'is-root' : ''
    }">\n<header class="topbar"><div class="inner"><div class="brand"><a href="${
      relToRoot || '.'
    }/index.html">NeatapticTS</a></div><nav class="main-nav"><a href="${
      relToRoot || '.'
    }/index.html">Home</a><a href="${
      relToRoot || '.'
    }/index.html"${docsActive}>Docs</a><a href="${examplesHref}"${examplesActive}>Examples</a><a href="https://github.com/reicek/NeatapticTS" target="_blank" rel="noopener">GitHub</a></nav></div></header>\n<div class="layout"><aside class="sidebar">${navHtmlFor(
      meta.relDir
    )}</aside><main class="content">${htmlBody}<footer class="site-footer">Generated from source JSDoc • <a href="https://github.com/reicek/NeatapticTS">GitHub</a></footer></main><aside class="toc">${toc}</aside></div></body></html>`;
    await writeFileWithRetry(outFile, page, 'utf8');
  }
  console.log('HTML docs generated.');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
