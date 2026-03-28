/*
 * Resolves ordering and visibility policy for generated output.
 *
 * This chapter folds docs.order.json preferences together with the stable
 * fallback heuristics used to order files, symbols, and folder-index entries.
 */

import * as path from 'path';

import { FILE_SUMMARY_SYMBOL_NAME } from '../generate-docs.constants.js';
import { getCachedDirectoryDocsOrderConfig } from '../generate-docs.order.js';
import type {
  FolderIndexNode,
  GenerateDocsState,
  RenderedSymbol,
} from '../generate-docs.types.js';
import {
  warnForUnknownConfiguredDirectoryFiles,
  warnForUnknownConfiguredDirectoryIntroFile,
  warnForUnknownConfiguredFileSectionSymbols,
  warnForUnknownConfiguredFolderIndexChildren,
  warnForUnknownConfiguredHiddenFiles,
  warnForUnknownConfiguredHiddenSymbols,
} from './generate-docs.output.warnings.utils.js';

/**
 * Resolves directory file order using config-first ordering with stable
 * fallback to the existing heuristic rank.
 *
 * After explicit `fileOrder` entries are placed, fallback ranking prefers:
 * index-style entrypoints, files with promoted summaries, non-utility files,
 * type-focused files, then shorter file names before lexical tie-breaks.
 *
 * @param state - Shared docs-generator state.
 * @param directoryPath - Absolute directory path.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @returns Files sorted for one directory README.
 */
export function resolveSortedDirectoryFiles(
  state: GenerateDocsState,
  directoryPath: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): string[] {
  const loadedConfig = getCachedDirectoryDocsOrderConfig(state, directoryPath);
  const configuredFileOrder = loadedConfig?.config.fileOrder;

  if (configuredFileOrder?.length) {
    warnForUnknownConfiguredDirectoryFiles(
      configuredFileOrder,
      [...fileSymbolMap.keys()],
      loadedConfig?.configPath,
    );
  }

  const explicitFileOrderIndex = new Map(
    (configuredFileOrder ?? []).map(
      (fileName, index) => [fileName, index] as const,
    ),
  );

  return [...fileSymbolMap.keys()].toSorted((leftFile, rightFile) => {
    const leftExplicitOrder = explicitFileOrderIndex.get(
      path.basename(leftFile),
    );
    const rightExplicitOrder = explicitFileOrderIndex.get(
      path.basename(rightFile),
    );

    if (leftExplicitOrder !== undefined || rightExplicitOrder !== undefined) {
      if (leftExplicitOrder === undefined) {
        return 1;
      }

      if (rightExplicitOrder === undefined) {
        return -1;
      }

      if (leftExplicitOrder !== rightExplicitOrder) {
        return leftExplicitOrder - rightExplicitOrder;
      }
    }

    return compareDirectoryReadmeFiles(leftFile, rightFile, fileSymbolMap);
  });
}

/**
 * Resolves visible files for one directory README after applying optional
 * `hiddenFiles` filtering.
 *
 * @param state - Shared docs-generator state.
 * @param directoryPath - Absolute directory path.
 * @param sortedFiles - Already sorted directory files.
 * @returns Visible files that should remain in the generated output.
 */
export function resolveVisibleDirectoryFiles(
  state: GenerateDocsState,
  directoryPath: string,
  sortedFiles: readonly string[],
): string[] {
  const loadedConfig = getCachedDirectoryDocsOrderConfig(state, directoryPath);
  const configuredHiddenFiles = loadedConfig?.config.hiddenFiles;

  if (configuredHiddenFiles?.length) {
    warnForUnknownConfiguredHiddenFiles(
      configuredHiddenFiles,
      sortedFiles,
      loadedConfig?.configPath,
    );
  }

  const hiddenFileNames = new Set(configuredHiddenFiles ?? []);
  return sortedFiles.filter(
    (filePath) => !hiddenFileNames.has(path.basename(filePath)),
  );
}

/**
 * Resolves hidden symbol names for one directory README.
 *
 * @param state - Shared docs-generator state.
 * @param directoryPath - Absolute directory path.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @returns Hidden symbol-name set for render-time filtering.
 */
export function resolveHiddenDirectorySymbolNames(
  state: GenerateDocsState,
  directoryPath: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): ReadonlySet<string> {
  const loadedConfig = getCachedDirectoryDocsOrderConfig(state, directoryPath);
  const configuredHiddenSymbols = loadedConfig?.config.hiddenSymbols;

  if (configuredHiddenSymbols?.length) {
    warnForUnknownConfiguredHiddenSymbols(
      configuredHiddenSymbols,
      fileSymbolMap,
      loadedConfig?.configPath,
    );
  }

  return new Set(configuredHiddenSymbols ?? []);
}

/**
 * Resolves top-level symbol order for one rendered file section.
 *
 * @param state - Shared docs-generator state.
 * @param filePath - Absolute source file path.
 * @param topLevelSymbols - Stable fallback-sorted top-level symbols.
 * @returns Top-level symbols ordered for rendering.
 */
export function resolveSortedTopLevelFileSectionSymbols(
  state: GenerateDocsState,
  filePath: string,
  topLevelSymbols: readonly RenderedSymbol[],
): RenderedSymbol[] {
  const directoryPath = path.dirname(filePath);
  const loadedConfig = getCachedDirectoryDocsOrderConfig(state, directoryPath);
  const configuredSymbolOrder =
    loadedConfig?.config.symbolOrder?.[path.basename(filePath)];

  if (!configuredSymbolOrder?.length) {
    return [...topLevelSymbols];
  }

  warnForUnknownConfiguredFileSectionSymbols(
    configuredSymbolOrder,
    topLevelSymbols,
    loadedConfig?.configPath,
    filePath,
  );

  const explicitSymbolOrderIndex = new Map(
    configuredSymbolOrder.map(
      (symbolName, index) => [symbolName, index] as const,
    ),
  );

  return [...topLevelSymbols].toSorted((leftSymbol, rightSymbol) => {
    const leftExplicitOrder = explicitSymbolOrderIndex.get(leftSymbol.name);
    const rightExplicitOrder = explicitSymbolOrderIndex.get(rightSymbol.name);

    if (leftExplicitOrder !== undefined || rightExplicitOrder !== undefined) {
      if (leftExplicitOrder === undefined) {
        return 1;
      }

      if (rightExplicitOrder === undefined) {
        return -1;
      }

      if (leftExplicitOrder !== rightExplicitOrder) {
        return leftExplicitOrder - rightExplicitOrder;
      }
    }

    return 0;
  });
}

/**
 * Resolves child-folder order for one folder-index node.
 *
 * @param state - Shared docs-generator state.
 * @param node - Folder index node whose children should be ordered.
 * @returns Sorted child folder names.
 */
export function resolveSortedFolderIndexChildNames(
  state: GenerateDocsState,
  node: FolderIndexNode,
): string[] {
  const loadedConfig = getCachedDirectoryDocsOrderConfig(
    state,
    node.sourceDirectoryPath,
  );
  const configuredFolderOrder = loadedConfig?.config.folderOrder;

  if (configuredFolderOrder?.length) {
    warnForUnknownConfiguredFolderIndexChildren(
      configuredFolderOrder,
      [...node.children.keys()],
      loadedConfig?.configPath,
      node.sourceDirectoryPath,
    );
  }

  const explicitFolderOrderIndex = new Map(
    (configuredFolderOrder ?? []).map(
      (folderName, index) => [folderName, index] as const,
    ),
  );

  return [...node.children.keys()].toSorted((leftName, rightName) => {
    const leftExplicitOrder = explicitFolderOrderIndex.get(leftName);
    const rightExplicitOrder = explicitFolderOrderIndex.get(rightName);

    if (leftExplicitOrder !== undefined || rightExplicitOrder !== undefined) {
      if (leftExplicitOrder === undefined) {
        return 1;
      }

      if (rightExplicitOrder === undefined) {
        return -1;
      }

      if (leftExplicitOrder !== rightExplicitOrder) {
        return leftExplicitOrder - rightExplicitOrder;
      }
    }

    return leftName.localeCompare(rightName);
  });
}

/**
 * Warns when `introFile` references a file missing from the current folder.
 *
 * @param state - Shared docs-generator state.
 * @param directoryPath - Absolute directory path.
 * @param configuredIntroFile - Configured intro file name.
 * @param sortedFiles - Actual files in the current directory README.
 * @returns Nothing.
 */
export function warnForConfiguredIntroFileIfNeeded(
  state: GenerateDocsState,
  directoryPath: string,
  configuredIntroFile: string,
  sortedFiles: readonly string[],
): void {
  const loadedConfig = getCachedDirectoryDocsOrderConfig(state, directoryPath);
  warnForUnknownConfiguredDirectoryIntroFile(
    configuredIntroFile,
    sortedFiles,
    loadedConfig?.configPath,
  );
}

/**
 * Ranks files within a directory README so entrypoints appear before helpers.
 *
 * The goal is not semantic perfection; it is a deterministic educational order
 * that usually reads well for humans when no explicit `docs.order.json`
 * override exists.
 *
 * @param filePath - Absolute source file path.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @returns Sort rank tuple.
 */
function rankFileForDirectoryReadme(
  filePath: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): number[] {
  const fileName = path.basename(filePath).toLowerCase();
  const renderedSymbols = fileSymbolMap.get(filePath) ?? [];

  const hasFileSummary = renderedSymbols.some((renderedSymbol) => {
    if (
      renderedSymbol.kind !== 'File' ||
      renderedSymbol.name !== FILE_SUMMARY_SYMBOL_NAME
    ) {
      return false;
    }

    return Boolean(renderedSymbol.jsdoc.description?.trim());
  });

  const isIndexFile = fileName === 'index.ts';
  const isTypesFile =
    fileName.includes('.types.') || fileName.endsWith('.types.ts');
  const isUtilityLike =
    /(\.utils\.|\.export-|\.import-|\.internal\.|\.private\.)/i.test(fileName);
  const isEntrypointLike = !isUtilityLike;

  return [
    isIndexFile ? 0 : 1,
    hasFileSummary ? 0 : 1,
    isEntrypointLike ? 0 : 1,
    isTypesFile ? 0 : 1,
    fileName.length,
  ];
}

/**
 * Compares two files using the existing directory README heuristic.
 *
 * @param leftFile - Left file path.
 * @param rightFile - Right file path.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @returns Negative when left sorts before right.
 */
function compareDirectoryReadmeFiles(
  leftFile: string,
  rightFile: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): number {
  const leftRank = rankFileForDirectoryReadme(leftFile, fileSymbolMap);
  const rightRank = rankFileForDirectoryReadme(rightFile, fileSymbolMap);

  for (
    let index = 0;
    index < Math.max(leftRank.length, rightRank.length);
    index += 1
  ) {
    const leftValue = leftRank[index] ?? 0;
    const rightValue = rightRank[index] ?? 0;
    if (leftValue !== rightValue) {
      return leftValue - rightValue;
    }
  }

  return leftFile.localeCompare(rightFile);
}
