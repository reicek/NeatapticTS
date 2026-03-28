/*
 * Owns CLI target selection and source-tree preparation for one docs run.
 *
 * This chapter decides what tree is being documented and which source files are
 * eligible before any ts-morph symbol work or markdown rendering begins.
 */

import fg from 'fast-glob';
import fs from 'fs-extra';
import * as path from 'path';
import type { SourceFile } from 'ts-morph';

import {
  DEFAULT_DOCS_TARGET,
  DOCS_TARGETS,
  SOURCE_FILE_GLOBS,
  SOURCE_FILE_IGNORE_GLOBS,
} from './generate-docs.constants.js';
import type {
  DocsTargetConfig,
  GenerateDocsState,
} from './generate-docs.types.js';

/**
 * Resolves the active docs target from CLI or environment input.
 *
 * Lookup order:
 * 1. `--target=...`
 * 2. `DOCS_TARGET`
 * 3. default target
 *
 * @param rawArguments - CLI arguments passed after the executable path.
 * @returns Concrete docs target configuration.
 */
export function resolveDocsTarget(
  rawArguments: readonly string[],
): DocsTargetConfig {
  const cliTarget = rawArguments
    .find((argument) => argument.startsWith('--target='))
    ?.slice('--target='.length);
  const requestedTarget =
    cliTarget ?? process.env.DOCS_TARGET ?? DEFAULT_DOCS_TARGET;
  const resolvedTarget = DOCS_TARGETS[requestedTarget];

  if (!resolvedTarget) {
    const supportedTargets = Object.keys(DOCS_TARGETS).toSorted().join(', ');
    throw new Error(
      `Unknown docs target "${requestedTarget}". Supported targets: ${supportedTargets}`,
    );
  }

  return resolvedTarget;
}

/**
 * Prepares the output tree for the selected target.
 *
 * @param target - Target configuration.
 * @returns Nothing.
 */
export async function initializeDocsTarget(
  target: DocsTargetConfig,
): Promise<void> {
  await cleanupPublishedRoot(target);
  await fs.ensureDir(target.docsDir);

  if (
    !target.rootReadmeSource ||
    !target.rootReadmeDestination ||
    !(await fs.pathExists(target.rootReadmeSource))
  ) {
    return;
  }

  await fs.ensureDir(path.dirname(target.rootReadmeDestination));
  await fs.copyFile(target.rootReadmeSource, target.rootReadmeDestination);
}

/**
 * Loads target source files into the shared ts-morph project.
 *
 * @param state - Shared docs-generator state.
 * @param target - Target configuration.
 * @returns Filtered source files that belong to the target tree.
 */
export async function loadTargetSourceFiles(
  state: GenerateDocsState,
  target: DocsTargetConfig,
): Promise<SourceFile[]> {
  const filePaths = await fg(SOURCE_FILE_GLOBS, {
    cwd: target.sourceDir,
    absolute: true,
    ignore: SOURCE_FILE_IGNORE_GLOBS,
  });

  for (const filePath of filePaths) {
    if (!shouldIncludeSourceFile(filePath, target)) {
      continue;
    }

    state.project.addSourceFileAtPath(filePath);
  }

  const rawSourceFiles = state.project.getSourceFiles();
  const sourceFiles = rawSourceFiles.filter((sourceFile) => {
    const filePath = sourceFile.getFilePath();
    return !filePath.endsWith('.d.ts') && !/node_modules/.test(filePath);
  });

  console.log(
    `[docs:${target.name}] Loaded ${sourceFiles.length} source files (raw: ${rawSourceFiles.length})`,
  );

  return sourceFiles;
}

/**
 * Determines whether a discovered file should participate in docs generation.
 *
 * @param filePath - Absolute source file path.
 * @param target - Target configuration.
 * @returns True when the file belongs in the target set.
 */
function shouldIncludeSourceFile(
  filePath: string,
  target: DocsTargetConfig,
): boolean {
  if (/\.test\.ts$/i.test(filePath)) {
    return false;
  }

  if (!target.excludeRootSourceFiles) {
    return true;
  }

  const relativeDirectory = path.dirname(
    path.relative(target.sourceDir, filePath),
  );
  return relativeDirectory !== '' && relativeDirectory !== '.';
}

/**
 * Removes previously published generated files for targets that mirror output
 * into a browsable published directory.
 *
 * @param target - Target configuration.
 * @returns Nothing.
 */
async function cleanupPublishedRoot(target: DocsTargetConfig): Promise<void> {
  if (!target.publishedRootDir) {
    return;
  }

  await fs.ensureDir(target.publishedRootDir);
  const preservedEntries = new Set(target.preservePublishedEntries ?? []);
  const childEntries = await fs.readdir(target.publishedRootDir);

  for (const childEntry of childEntries) {
    if (preservedEntries.has(childEntry)) {
      continue;
    }

    await fs.remove(path.join(target.publishedRootDir, childEntry));
  }
}
