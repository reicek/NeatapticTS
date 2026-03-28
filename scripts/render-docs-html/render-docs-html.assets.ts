/*
 * Static asset bootstrapping for generated HTML docs.
 *
 * Theme CSS and the landing-page hero image are stable assets that should be
 * prepared before markdown pages are rendered. Keeping that work here prevents
 * the root renderer from mixing file-copy policy with page assembly.
 */

import path from 'path';
import fs from 'fs-extra';

import { DOCS_DIR } from './render-docs-html.shared.js';

const THEME_CSS_SOURCE_PATH = path.resolve('scripts', 'assets', 'theme.css');
const THEME_CSS_OUTPUT_PATH = path.join(DOCS_DIR, 'assets', 'theme.css');
const NN_IMAGE_SOURCE_PATH = path.resolve('nn.jpg');
const NN_IMAGE_FALLBACK_SOURCE_PATH = path.resolve(
  'scripts',
  'assets',
  'nn.jpg',
);
const NN_IMAGE_OUTPUT_PATH = path.join(DOCS_DIR, 'nn.jpg');

/** Ensures static non-Mermaid assets exist under the generated docs tree. */
export async function ensureRenderDocsAssets(): Promise<void> {
  await ensureThemeCss();
  await ensureRootHeroImage();
}

/** Copies the docs theme stylesheet into the published assets tree. */
async function ensureThemeCss(): Promise<void> {
  await fs.ensureDir(path.dirname(THEME_CSS_OUTPUT_PATH));
  await fs.copyFile(THEME_CSS_SOURCE_PATH, THEME_CSS_OUTPUT_PATH);
}

/**
 * Copies the root README hero image so `/docs/README.md` can keep using the
 * same `nn.jpg` reference after publication.
 */
async function ensureRootHeroImage(): Promise<void> {
  const hasRootImage = await fs.pathExists(NN_IMAGE_SOURCE_PATH);
  const hasFallbackImage = await fs.pathExists(NN_IMAGE_FALLBACK_SOURCE_PATH);
  const sourcePath = hasRootImage
    ? NN_IMAGE_SOURCE_PATH
    : hasFallbackImage
      ? NN_IMAGE_FALLBACK_SOURCE_PATH
      : undefined;

  if (!sourcePath) {
    return;
  }

  await fs.copyFile(sourcePath, NN_IMAGE_OUTPUT_PATH);
}
