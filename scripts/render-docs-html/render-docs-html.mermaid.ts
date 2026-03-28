/*
 * Mermaid asset bootstrapping and validation for generated HTML docs.
 *
 * This chapter owns both sides of Mermaid handling: copying the browser bundle
 * needed at runtime and treating invalid fenced diagrams as a build failure
 * during docs generation.
 */

import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'path';
import fs from 'fs-extra';

import { escapeHtml, DOCS_DIR } from './render-docs-html.shared.js';
import type { MermaidBlockReference } from './render-docs-html.types.js';

const MERMAID_MODULE_SOURCE_PATH = path.resolve(
  'node_modules',
  'mermaid',
  'dist',
  'mermaid.esm.min.mjs',
);
const MERMAID_DIST_SOURCE_DIR = path.resolve('node_modules', 'mermaid', 'dist');
const MERMAID_DIST_OUTPUT_DIR = path.join(DOCS_DIR, 'assets', 'vendor');
const MERMAID_CLI_SCRIPT_PATH = path.resolve(
  'dist-docs',
  'scripts',
  'mermaid-cli.js',
);
const MERMAID_VALIDATION_CONCURRENCY = Math.max(
  1,
  Math.min(
    4,
    typeof os.availableParallelism === 'function'
      ? os.availableParallelism() - 1
      : 4,
  ),
);

interface MermaidValidationTask {
  mermaidBlock: MermaidBlockReference;
  tempInputPath: string;
}

/** Ensures the Mermaid browser bundle is copied into the published docs tree. */
export async function ensureMermaidBrowserAssets(): Promise<void> {
  const hasMermaidModule = await fs.pathExists(MERMAID_MODULE_SOURCE_PATH);
  if (!hasMermaidModule) {
    return;
  }

  await fs.ensureDir(MERMAID_DIST_OUTPUT_DIR);
  await fs.copy(MERMAID_DIST_SOURCE_DIR, MERMAID_DIST_OUTPUT_DIR, {
    overwrite: true,
  });
}

/** Builds the client-side Mermaid bootstrap script for one generated page. */
export function buildMermaidBootstrapScript(relToRoot: string): string {
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

/** Collects Mermaid fenced blocks from one markdown page. */
export function collectMermaidBlocks(
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

/** Validates every collected Mermaid block before HTML generation proceeds. */
export async function validateMermaidBlocks(
  mermaidBlocks: readonly MermaidBlockReference[],
): Promise<void> {
  if (mermaidBlocks.length === 0) {
    return;
  }

  const uniqueMermaidBlocks = resolveUniqueMermaidBlocks(mermaidBlocks);
  const tempDirectoryPath = await mkdtemp(
    path.join(os.tmpdir(), 'neatapticts-docs-mermaid-'),
  );

  try {
    const mermaidValidationTasks = uniqueMermaidBlocks.map(
      (mermaidBlock, blockIndex) => ({
        mermaidBlock,
        tempInputPath: path.join(
          tempDirectoryPath,
          `diagram-${blockIndex + 1}.mmd`,
        ),
      }),
    );

    await Promise.all(
      mermaidValidationTasks.map((mermaidValidationTask) =>
        fs.writeFile(
          mermaidValidationTask.tempInputPath,
          mermaidValidationTask.mermaidBlock.diagram,
          'utf8',
        ),
      ),
    );
    await runMermaidValidationPool(mermaidValidationTasks);
  } finally {
    await rm(tempDirectoryPath, { recursive: true, force: true });
  }
}

/**
 * Resolves one representative block for each unique Mermaid source string.
 *
 * Many generated README surfaces repeat the same diagrams, so validating each
 * unique source only once avoids redundant parser work.
 *
 * @param mermaidBlocks - Mermaid blocks collected from generated docs surfaces.
 * @returns Representative Mermaid blocks keyed by unique diagram text.
 */
function resolveUniqueMermaidBlocks(
  mermaidBlocks: readonly MermaidBlockReference[],
): MermaidBlockReference[] {
  const uniqueMermaidBlocksByDiagram = new Map<string, MermaidBlockReference>();

  for (const mermaidBlock of mermaidBlocks) {
    if (!uniqueMermaidBlocksByDiagram.has(mermaidBlock.diagram)) {
      uniqueMermaidBlocksByDiagram.set(mermaidBlock.diagram, mermaidBlock);
    }
  }

  return [...uniqueMermaidBlocksByDiagram.values()];
}

/** Validates Mermaid diagrams through a bounded parallel worker pool. */
async function runMermaidValidationPool(
  mermaidValidationTasks: readonly MermaidValidationTask[],
): Promise<void> {
  let nextTaskIndex = 0;
  const workerCount = Math.min(
    MERMAID_VALIDATION_CONCURRENCY,
    mermaidValidationTasks.length,
  );

  await Promise.all(
    Array.from({ length: workerCount }, async () => {
      while (nextTaskIndex < mermaidValidationTasks.length) {
        const mermaidValidationTask = mermaidValidationTasks[nextTaskIndex];
        nextTaskIndex += 1;
        await runMermaidValidation(mermaidValidationTask);
      }
    }),
  );
}

/** Runs the Mermaid CLI validator for one diagram source file. */
async function runMermaidValidation(
  mermaidValidationTask: MermaidValidationTask,
): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const childProcess = spawn(
      process.execPath,
      [
        MERMAID_CLI_SCRIPT_PATH,
        'validate',
        '--input',
        mermaidValidationTask.tempInputPath,
      ],
      { stdio: 'inherit' },
    );

    childProcess.once('exit', (exitCode, signal) => {
      if (exitCode === 0) {
        resolve();
        return;
      }

      const relativeReadmePath = path.relative(
        process.cwd(),
        mermaidValidationTask.mermaidBlock.readmePath,
      );
      reject(
        new Error(
          `Invalid Mermaid diagram in ${relativeReadmePath} (block ${mermaidValidationTask.mermaidBlock.blockNumber}). Mermaid CLI exited with code ${exitCode ?? 'null'}${signal ? ` and signal ${signal}` : ''}.`,
        ),
      );
    });
    childProcess.once('error', reject);
  });
}

/** Renders Mermaid fenced blocks as safe placeholder HTML before client boot. */
export function renderMermaidDiagramPlaceholder(diagram: string): string {
  return `<pre class="mermaid mermaid-diagram">${escapeHtml(diagram)}</pre>`;
}
