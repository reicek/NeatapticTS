/*
 * Centralizes stable docs-generator constants and target definitions.
 *
 * Keeping these values in one file makes it easier to audit what the generator
 * considers a target, a published output root, or an ordering-control file
 * without mixing that policy into traversal or rendering code.
 */

import * as path from 'path';

import type { DocsTargetConfig } from './generate-docs.types.js';

/** Root docs output directory mirrored into the published docs site. */
export const DOCS_DIR = path.resolve('docs');

/** Default docs target when no CLI or environment override is supplied. */
export const DEFAULT_DOCS_TARGET = 'src';

/** Synthetic symbol name used to store file-level summary content. */
export const FILE_SUMMARY_SYMBOL_NAME = '__file_summary__';

/** Source-file glob used to discover docs-generator inputs. */
export const SOURCE_FILE_GLOBS = ['**/*.ts'];

/** Ignore rules for files that should never become generated docs sources. */
export const SOURCE_FILE_IGNORE_GLOBS = ['**/*.d.ts'];

/** Generated docs landing page file name for the folder index. */
export const FOLDER_INDEX_FILE_NAME = 'FOLDERS.md';

/** Per-directory ordering config file name. */
export const DOCS_ORDER_CONFIG_FILE_NAME = 'docs.order.json';

/** Supported keys accepted inside `docs.order.json`. */
export const DOCS_ORDER_SUPPORTED_KEYS = [
  'introFile',
  'fileOrder',
  'symbolOrder',
  'folderOrder',
  'hiddenFiles',
  'hiddenSymbols',
] as const;

/** Workspace root used to rewrite absolute ts-morph type paths into repo paths. */
export const WORKSPACE_ROOT_DIR = path.resolve('.');

/**
 * Docs targets supported by the folder README generator.
 *
 * Each target maps one source tree into one published docs tree while keeping
 * any target-specific root README mirroring rules local to the target.
 */
export const DOCS_TARGETS: Record<string, DocsTargetConfig> = {
  src: {
    name: 'src',
    sourceDir: path.resolve('src'),
    docsDir: DOCS_DIR,
    rootDocsDir: path.join(DOCS_DIR, 'src'),
    rootReadmeSource: path.resolve('README.md'),
    rootReadmeDestination: path.join(DOCS_DIR, 'README.md'),
    includeFolderIndex: true,
  },
  asciiMaze: {
    name: 'asciiMaze',
    sourceDir: path.resolve('test', 'examples', 'asciiMaze'),
    docsDir: path.join(DOCS_DIR, 'examples', 'asciiMaze', 'docs'),
    rootDocsDir: path.join(DOCS_DIR, 'examples', 'asciiMaze', 'docs'),
    rootReadmeSource: path.resolve(
      'test',
      'examples',
      'asciiMaze',
      'README.md',
    ),
    rootReadmeDestination: path.join(
      DOCS_DIR,
      'examples',
      'asciiMaze',
      'docs',
      'README.md',
    ),
    excludeRootSourceFiles: true,
    publishedRootDir: path.join(DOCS_DIR, 'examples', 'asciiMaze'),
    preservePublishedEntries: ['index.html'],
  },
  'flappy-bird': {
    name: 'flappy-bird',
    sourceDir: path.resolve('test', 'examples', 'flappy_bird'),
    docsDir: path.join(DOCS_DIR, 'examples', 'flappy_bird', 'docs'),
    rootDocsDir: path.join(DOCS_DIR, 'examples', 'flappy_bird', 'docs'),
    rootReadmeSource: path.resolve(
      'test',
      'examples',
      'flappy_bird',
      'README.md',
    ),
    rootReadmeDestination: path.join(
      DOCS_DIR,
      'examples',
      'flappy_bird',
      'docs',
      'README.md',
    ),
    excludeRootSourceFiles: true,
    publishedRootDir: path.join(DOCS_DIR, 'examples', 'flappy_bird'),
    preservePublishedEntries: ['index.html'],
  },
};
