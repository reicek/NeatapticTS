/*
 * Emits docs.order.json warning messages scoped to the output boundary.
 *
 * Ordering helpers depend on these checks to explain stale config references
 * without mixing repetitive warning logic back into the main README and index
 * rendering flow.
 */

import * as path from 'path';

import { warnDirectoryDocsOrderConfig } from '../generate-docs.order.js';
import type { RenderedSymbol } from '../generate-docs.types.js';

/**
 * Warns when `fileOrder` references files missing from the current folder.
 *
 * @param configuredFileOrder - Configured file order entries.
 * @param filePaths - Actual files in the current directory README.
 * @param configPath - Absolute config path.
 * @returns Nothing.
 */
export function warnForUnknownConfiguredDirectoryFiles(
  configuredFileOrder: readonly string[],
  filePaths: readonly string[],
  configPath: string | undefined,
): void {
  if (!configPath) {
    return;
  }

  const knownFileNames = new Set(
    filePaths.map((filePath) => path.basename(filePath)),
  );
  const unknownFileNames = configuredFileOrder.filter(
    (fileName) => !knownFileNames.has(fileName),
  );

  if (unknownFileNames.length === 0) {
    return;
  }

  warnDirectoryDocsOrderConfig(
    configPath,
    `Field "fileOrder" references unknown files for this folder: ${unknownFileNames.join(', ')}`,
  );
}

/**
 * Warns when `hiddenFiles` references files missing from the current folder.
 *
 * @param configuredHiddenFiles - Configured hidden file entries.
 * @param filePaths - Actual files in the current directory README.
 * @param configPath - Absolute config path.
 * @returns Nothing.
 */
export function warnForUnknownConfiguredHiddenFiles(
  configuredHiddenFiles: readonly string[],
  filePaths: readonly string[],
  configPath: string | undefined,
): void {
  if (!configPath) {
    return;
  }

  const knownFileNames = new Set(
    filePaths.map((filePath) => path.basename(filePath)),
  );
  const unknownFileNames = configuredHiddenFiles.filter(
    (fileName) => !knownFileNames.has(fileName),
  );

  if (unknownFileNames.length === 0) {
    return;
  }

  warnDirectoryDocsOrderConfig(
    configPath,
    `Field "hiddenFiles" references unknown files for this folder: ${unknownFileNames.join(', ')}`,
  );
}

/**
 * Warns when `introFile` references a file missing from the current folder.
 *
 * @param configuredIntroFile - Configured intro file name.
 * @param filePaths - Actual files in the current directory README.
 * @param configPath - Absolute config path.
 * @returns Nothing.
 */
export function warnForUnknownConfiguredDirectoryIntroFile(
  configuredIntroFile: string,
  filePaths: readonly string[],
  configPath: string | undefined,
): void {
  if (!configPath) {
    return;
  }

  const knownFileNames = new Set(
    filePaths.map((filePath) => path.basename(filePath)),
  );
  if (knownFileNames.has(configuredIntroFile)) {
    return;
  }

  warnDirectoryDocsOrderConfig(
    configPath,
    `Field "introFile" references an unknown file for this folder: ${configuredIntroFile}`,
  );
}

/**
 * Warns when `symbolOrder` references top-level symbols missing from one file
 * section.
 *
 * @param configuredSymbolOrder - Configured symbol order entries.
 * @param topLevelSymbols - Top-level symbols rendered for the current file.
 * @param configPath - Absolute config path.
 * @param filePath - Absolute source file path for warning context.
 * @returns Nothing.
 */
export function warnForUnknownConfiguredFileSectionSymbols(
  configuredSymbolOrder: readonly string[],
  topLevelSymbols: readonly RenderedSymbol[],
  configPath: string | undefined,
  filePath: string,
): void {
  if (!configPath) {
    return;
  }

  const knownSymbolNames = new Set(
    topLevelSymbols.map((symbol) => symbol.name),
  );
  const unknownSymbolNames = configuredSymbolOrder.filter(
    (symbolName) => !knownSymbolNames.has(symbolName),
  );

  if (unknownSymbolNames.length === 0) {
    return;
  }

  warnDirectoryDocsOrderConfig(
    configPath,
    `Field "symbolOrder.${path.basename(filePath)}" references unknown top-level symbols for this file section: ${unknownSymbolNames.join(', ')}`,
  );
}

/**
 * Warns when `hiddenSymbols` references symbols missing from the current
 * directory README.
 *
 * @param configuredHiddenSymbols - Configured hidden symbol entries.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @param configPath - Absolute config path.
 * @returns Nothing.
 */
export function warnForUnknownConfiguredHiddenSymbols(
  configuredHiddenSymbols: readonly string[],
  fileSymbolMap: Map<string, RenderedSymbol[]>,
  configPath: string | undefined,
): void {
  if (!configPath) {
    return;
  }

  const knownSymbolNames = new Set(
    [...fileSymbolMap.values()].flatMap((renderedSymbols) =>
      renderedSymbols.map((symbol) => symbol.name),
    ),
  );
  const unknownSymbolNames = configuredHiddenSymbols.filter(
    (symbolName) => !knownSymbolNames.has(symbolName),
  );

  if (unknownSymbolNames.length === 0) {
    return;
  }

  warnDirectoryDocsOrderConfig(
    configPath,
    `Field "hiddenSymbols" references unknown symbols for this folder: ${unknownSymbolNames.join(', ')}`,
  );
}

/**
 * Warns when `folderOrder` references child folders missing from one folder
 * index node.
 *
 * @param configuredFolderOrder - Configured folder order entries.
 * @param childFolderNames - Actual child folder names under the current node.
 * @param configPath - Absolute config path.
 * @param nodePath - Relative folder-index node path for warning context.
 * @returns Nothing.
 */
export function warnForUnknownConfiguredFolderIndexChildren(
  configuredFolderOrder: readonly string[],
  childFolderNames: readonly string[],
  configPath: string | undefined,
  nodePath: string,
): void {
  if (!configPath) {
    return;
  }

  const knownFolderNames = new Set(childFolderNames);
  const unknownFolderNames = configuredFolderOrder.filter(
    (folderName) => !knownFolderNames.has(folderName),
  );

  if (unknownFolderNames.length === 0) {
    return;
  }

  warnDirectoryDocsOrderConfig(
    configPath,
    `Field "folderOrder" references unknown child folders for ${nodePath}: ${unknownFolderNames.join(', ')}`,
  );
}
