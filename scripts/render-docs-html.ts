/*
 * Converts every README.md inside docs/ (including root copy) into an index.html in the same directory.
 * Usage: npm run docs:html
 */
import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import fg from 'fast-glob';
import path from 'path';
import fs from 'fs-extra';
import { marked } from 'marked';

const DOCS_DIR = path.resolve('docs');
const THEME_CSS_SOURCE_PATH = path.resolve('scripts', 'assets', 'theme.css');
const THEME_CSS_OUTPUT_PATH = path.join(DOCS_DIR, 'assets', 'theme.css');
const MERMAID_MODULE_SOURCE_PATH = path.resolve(
  'node_modules',
  'mermaid',
  'dist',
  'mermaid.esm.min.mjs',
);
const MERMAID_DIST_SOURCE_DIR = path.resolve('node_modules', 'mermaid', 'dist');
const MERMAID_MODULE_OUTPUT_PATH = path.join(
  DOCS_DIR,
  'assets',
  'vendor',
  'mermaid.esm.min.mjs',
);
const MERMAID_DIST_OUTPUT_DIR = path.join(DOCS_DIR, 'assets', 'vendor');
const MERMAID_CLI_SCRIPT_PATH = path.resolve('scripts', 'mermaid-cli.mjs');
const NN_IMAGE_SOURCE_PATH = path.resolve('nn.jpg');
const NN_IMAGE_FALLBACK_SOURCE_PATH = path.resolve(
  'scripts',
  'assets',
  'nn.jpg',
);
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

interface PageMeta {
  abs: string;
  relDir: string;
  title: string;
  markdown: string;
}

interface MermaidBlockReference {
  readmePath: string;
  blockNumber: number;
  diagram: string;
}

function isRetriableFileSystemError(
  error: unknown,
): error is NodeJS.ErrnoException {
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

async function ensureMermaidModule(): Promise<void> {
  // Step 1: Skip Mermaid bootstrapping when the browser bundle is unavailable.
  const hasMermaidModule = await fs.pathExists(MERMAID_MODULE_SOURCE_PATH);
  if (!hasMermaidModule) return;

  // Step 2: Copy the full Mermaid dist bundle so transitive chunk imports load.
  await fs.ensureDir(MERMAID_DIST_OUTPUT_DIR);
  await fs.copy(MERMAID_DIST_SOURCE_DIR, MERMAID_DIST_OUTPUT_DIR, {
    overwrite: true,
  });
}

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

async function ensureStaticDocsAssets(): Promise<void> {
  // Step 1: Ensure core theme assets are present.
  await ensureThemeCss();
  await ensureMermaidModule();

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

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function buildMermaidBootstrapScript(relToRoot: string): string {
  const mermaidModuleHref =
    (relToRoot ? `${relToRoot}/` : '') + 'assets/vendor/mermaid.esm.min.mjs';

  return `<script type="module">
import mermaid from "${mermaidModuleHref}";

mermaid.initialize({
  startOnLoad: false,
  securityLevel: 'loose',
  theme: 'base',
  themeVariables: {
    darkMode: true,
    background: '#121a24',
    primaryColor: '#102131',
    primaryTextColor: '#d7e4f3',
    primaryBorderColor: '#5ec8ff',
    secondaryColor: '#0f1722',
    secondaryTextColor: '#d7e4f3',
    secondaryBorderColor: '#5ec8ff',
    tertiaryColor: '#16283a',
    tertiaryTextColor: '#d7e4f3',
    tertiaryBorderColor: '#ffbf69',
    mainBkg: '#0f1722',
    nodeBorder: '#5ec8ff',
    clusterBkg: '#102131',
    clusterBorder: '#5ec8ff',
    lineColor: '#5ec8ff',
    edgeLabelBackground: '#0f1722',
    textColor: '#d7e4f3',
    fontFamily: 'Open Sans, sans-serif'
  }
});

try {
  await mermaid.run({ querySelector: '.mermaid-diagram' });
} catch (error) {
  console.error('[docs] Mermaid render failed.', error);
}
</script>`;
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

function collectMermaidBlocks(
  markdown: string,
  readmePath: string,
): MermaidBlockReference[] {
  const mermaidBlocks: MermaidBlockReference[] = [];
  const mermaidFencePattern = /^```mermaid[^\n]*\r?\n([\s\S]*?)^```\s*$/gm;
  let mermaidMatch = mermaidFencePattern.exec(markdown);

  while (mermaidMatch) {
    mermaidBlocks.push({
      readmePath,
      blockNumber: mermaidBlocks.length + 1,
      diagram: mermaidMatch[1].trim(),
    });
    mermaidMatch = mermaidFencePattern.exec(markdown);
  }

  return mermaidBlocks;
}

async function validateMermaidBlocks(
  mermaidBlocks: readonly MermaidBlockReference[],
): Promise<void> {
  if (mermaidBlocks.length === 0) {
    return;
  }

  const tempDirectoryPath = await mkdtemp(
    path.join(os.tmpdir(), 'neatapticts-docs-mermaid-'),
  );

  try {
    for (const mermaidBlock of mermaidBlocks) {
      const tempInputPath = path.join(
        tempDirectoryPath,
        `diagram-${mermaidBlock.blockNumber}.mmd`,
      );

      // Step 1: Materialize the Mermaid block for CLI validation.
      await fs.writeFile(tempInputPath, mermaidBlock.diagram, 'utf8');

      // Step 2: Treat broken Mermaid as a docs build failure instead of a browser-only error.
      await runMermaidValidation(tempInputPath, mermaidBlock);
    }
  } finally {
    await rm(tempDirectoryPath, { recursive: true, force: true });
  }
}

async function runMermaidValidation(
  tempInputPath: string,
  mermaidBlock: MermaidBlockReference,
): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const childProcess = spawn(
      process.execPath,
      [MERMAID_CLI_SCRIPT_PATH, 'validate', '--input', tempInputPath],
      { stdio: 'inherit' },
    );

    childProcess.once('exit', (exitCode) => {
      if (exitCode === 0) {
        resolve();
        return;
      }

      const relativeReadmePath = path.relative(process.cwd(), mermaidBlock.readmePath);
      reject(
        new Error(
          `Invalid Mermaid diagram in ${relativeReadmePath} (block ${mermaidBlock.blockNumber}). Mermaid CLI exited with code ${exitCode ?? 'null'}.`,
        ),
      );
    });
    childProcess.once('error', reject);
  });
}

async function main() {
  await ensureStaticDocsAssets();
  const readmes = await fg(['**/README.md'], { cwd: DOCS_DIR, absolute: true });

  // Collect metadata for navigation
  const pages: PageMeta[] = [];
  const mermaidBlocks: MermaidBlockReference[] = [];
  for (const mdFile of readmes) {
    const md = await fs.readFile(mdFile, 'utf8');
    const title =
      md.match(/^#\s+(.+)$/m)?.[1] ||
      path.relative(DOCS_DIR, path.dirname(mdFile)) ||
      'Documentation';
    const relDir = path
      .relative(DOCS_DIR, path.dirname(mdFile))
      .replace(/\\/g, '/');
    pages.push({ abs: mdFile, relDir, title, markdown: md });
    mermaidBlocks.push(...collectMermaidBlocks(md, mdFile));
  }

  // Step 1: Validate Mermaid across the docs tree before emitting HTML pages.
  await validateMermaidBlocks(mermaidBlocks);

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
        const rightRank =
          rightOrder === -1 ? Number.MAX_SAFE_INTEGER : rightOrder;
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
    const md = meta.markdown;
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
    const originalCode = renderer.code?.bind(renderer);
    let hasMermaidDiagram = false;
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
    (renderer as any).code = (codeToken: any) => {
      const { text, lang } = codeToken;
      if ((lang ?? '').toString().trim().toLowerCase() === 'mermaid') {
        hasMermaidDiagram = true;
        return `<pre class="mermaid mermaid-diagram">${escapeHtml(
          (text ?? '').toString(),
        )}</pre>`;
      }

      return originalCode?.(codeToken) ?? '';
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
              }</div>`,
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
    const mermaidBootstrapScript = hasMermaidDiagram
      ? buildMermaidBootstrapScript(relToRoot)
      : '';
    const docsLayoutInteractionScript = buildDocsLayoutInteractionScript();
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
    }/index.html"${docsActive}>Docs</a><a href="${examplesHref}"${examplesActive}>Examples</a><a href="https://github.com/reicek/NeatapticTS" target="_blank" rel="noopener">GitHub</a></nav></div></header>\n<div class="layout"><aside class="sidebar docs-panel docs-panel-left">${navHtmlFor(
      meta.relDir,
    )}</aside><main class="content">${htmlBody}<footer class="site-footer">Generated from source JSDoc • <a href="https://github.com/reicek/NeatapticTS">GitHub</a></footer></main><aside class="toc docs-panel docs-panel-right">${toc}</aside></div>${docsLayoutInteractionScript}${mermaidBootstrapScript}</body></html>`;
    await writeFileWithRetry(outFile, page, 'utf8');
  }
  console.log('HTML docs generated.');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
