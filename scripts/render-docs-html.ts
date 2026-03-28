/*
 * Converts every README.md inside docs/ into an index.html in the same directory.
 * Usage: npm run docs:html
 *
 * This file intentionally stays orchestration-first. The folder-owned boundary
 * under `scripts/render-docs-html/` owns navigation, Markdown page rendering,
 * static assets, and Mermaid validation.
 */

import { ensureRenderDocsAssets } from './render-docs-html/render-docs-html.assets.js';
import {
  ensureMermaidBrowserAssets,
  validateMermaidBlocks,
} from './render-docs-html/render-docs-html.mermaid.js';
import {
  collectDocsPages,
  emitDocsPages,
} from './render-docs-html/render-docs-html.pages.js';

async function main(): Promise<void> {
  // Step 1: Ensure static assets and Mermaid browser bundles are published.
  await ensureRenderDocsAssets();
  await ensureMermaidBrowserAssets();

  // Step 2: Discover markdown pages and collect Mermaid fences.
  const { pages, mermaidBlocks } = await collectDocsPages();

  // Step 3: Validate all Mermaid diagrams before emitting HTML.
  await validateMermaidBlocks(mermaidBlocks);

  // Step 4: Render the full HTML page set.
  await emitDocsPages(pages);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
