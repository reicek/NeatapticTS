/*
 * Builds per-directory README markdown for the output boundary.
 *
 * This chapter owns the readable docs narrative: promoted file summaries,
 * file section ordering, symbol rendering, and the markdown block structure
 * that turns collected symbols into educational folder READMEs.
 */

import * as path from 'path';

import { getCachedDirectoryDocsOrderConfig } from '../generate-docs.order.js';
import type {
  GenerateDocsState,
  RenderedFileSummary,
  RenderedSymbol,
} from '../generate-docs.types.js';
import { renderSignatureBlock } from '../symbols/generate-docs.symbols.js';
import {
  resolveHiddenDirectorySymbolNames,
  resolveSortedDirectoryFiles,
  resolveSortedTopLevelFileSectionSymbols,
  resolveVisibleDirectoryFiles,
  warnForConfiguredIntroFileIfNeeded,
} from './generate-docs.output.ordering.utils.js';

/**
 * Builds the README markdown for one directory.
 *
 * One file summary may be promoted into the directory opening so the folder
 * README starts with the strongest module-level narrative instead of repeating
 * the same text inside the first file section.
 *
 * @param state - Shared docs-generator state.
 * @param relativeDirectory - Directory path relative to the target root.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @param sourceDir - Absolute source root.
 * @returns Markdown README content.
 */
export function buildDirectoryReadme(
  state: GenerateDocsState,
  relativeDirectory: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
  sourceDir: string,
): string {
  const title = (relativeDirectory || path.basename(sourceDir)).replace(
    /\\/g,
    '/',
  );
  const directoryBaseName = path.basename(relativeDirectory || sourceDir);
  const directoryPath =
    relativeDirectory === ''
      ? sourceDir
      : path.join(sourceDir, relativeDirectory);
  const lines = [`# ${title}`, ''];

  const sortedFiles = resolveSortedDirectoryFiles(
    state,
    directoryPath,
    fileSymbolMap,
  );
  const hiddenSymbolNames = resolveHiddenDirectorySymbolNames(
    state,
    directoryPath,
    fileSymbolMap,
  );
  const visibleSortedFiles = resolveVisibleDirectoryFiles(
    state,
    directoryPath,
    sortedFiles,
  );

  const primaryDirectoryIntro = resolvePrimaryDirectoryIntro(
    state,
    directoryPath,
    sortedFiles,
    fileSymbolMap,
    directoryBaseName,
  );

  if (primaryDirectoryIntro?.summary.description) {
    lines.push(primaryDirectoryIntro.summary.description, '');
  }

  if (primaryDirectoryIntro?.summary.examples?.length) {
    lines.push(
      primaryDirectoryIntro.summary.examples.length === 1
        ? 'Example:'
        : 'Examples:',
      '',
    );

    for (const example of primaryDirectoryIntro.summary.examples) {
      lines.push(example, '');
    }
  }

  for (const filePath of visibleSortedFiles) {
    renderFileReadmeSection(
      state,
      lines,
      filePath,
      fileSymbolMap,
      sourceDir,
      hiddenSymbolNames,
      primaryDirectoryIntro?.filePath === filePath,
    );
  }

  return `${lines.join('\n').trim()}\n`;
}

/**
 * Renders the README section for a single file.
 *
 * @param state - Shared docs-generator state.
 * @param lines - Output line buffer.
 * @param filePath - Absolute source file path.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @param sourceDir - Absolute source root.
 * @param hiddenSymbolNames - Hidden symbol-name set for the current directory.
 * @param suppressFileSummary - Whether the file summary was already promoted above.
 * @returns Nothing.
 */
function renderFileReadmeSection(
  state: GenerateDocsState,
  lines: string[],
  filePath: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
  sourceDir: string,
  hiddenSymbolNames: ReadonlySet<string>,
  suppressFileSummary: boolean = false,
): void {
  const visibleFileSymbols = (fileSymbolMap.get(filePath) ?? []).filter(
    (symbol) => !hiddenSymbolNames.has(symbol.name),
  );
  const fileSummarySymbol = visibleFileSymbols.find(
    (symbol) => symbol.kind === 'File',
  );
  const nonFileSummarySymbols = visibleFileSymbols.filter(
    (symbol) => symbol.kind !== 'File',
  );

  if (
    !fileSummarySymbol?.jsdoc.description &&
    !fileSummarySymbol?.jsdoc.examples?.length &&
    nonFileSummarySymbols.length === 0
  ) {
    return;
  }

  const relativeFilePath = path
    .relative(sourceDir, filePath)
    .replace(/\\/g, '/');
  lines.push(`## ${relativeFilePath}`, '');

  if (!suppressFileSummary && fileSummarySymbol?.jsdoc.description) {
    lines.push(fileSummarySymbol.jsdoc.description, '');
  }

  if (!suppressFileSummary && fileSummarySymbol?.jsdoc.examples?.length) {
    lines.push(
      fileSummarySymbol.jsdoc.examples.length === 1 ? 'Example:' : 'Examples:',
      '',
    );

    for (const example of fileSummarySymbol.jsdoc.examples) {
      lines.push(example, '');
    }
  }

  const fileBaseName = path.basename(filePath, '.ts');
  const sortedTopLevelCandidates = nonFileSummarySymbols
    .filter((symbol) => !symbol.parent)
    .toSorted((left, right) =>
      compareFileSectionSymbols(left, right, fileBaseName),
    );
  const topLevelSymbols = resolveSortedTopLevelFileSectionSymbols(
    state,
    filePath,
    sortedTopLevelCandidates,
  );
  const symbolsByParent = groupSymbolsByParent(nonFileSummarySymbols);

  for (const topLevelSymbol of topLevelSymbols) {
    renderSymbolBlock(lines, topLevelSymbol, 3);

    const childSymbols = symbolsByParent.get(topLevelSymbol.name);
    if (!childSymbols) {
      continue;
    }

    childSymbols.sort((left, right) => left.name.localeCompare(right.name));
    for (const childSymbol of childSymbols) {
      renderSymbolBlock(lines, childSymbol, 4);
    }
  }

  for (const [parentName, childSymbols] of symbolsByParent) {
    if (topLevelSymbols.some((symbol) => symbol.name === parentName)) {
      continue;
    }

    lines.push(`### ${parentName}`, '');
    childSymbols.sort((left, right) => left.name.localeCompare(right.name));
    for (const childSymbol of childSymbols) {
      renderSymbolBlock(lines, childSymbol, 4);
    }
  }
}

/**
 * Resolves the file-summary block that should be promoted into the directory
 * README opening.
 *
 * Selection priority is:
 * 1. `docs.order.json` `introFile` when configured and the referenced file has
 *    a renderable summary,
 * 2. a directory entrypoint file matching the folder name or `index.ts`,
 * 3. the first remaining sorted file with a renderable summary.
 *
 * @param state - Shared docs-generator state.
 * @param directoryPath - Absolute directory path.
 * @param sortedFiles - Files already ordered for directory README rendering.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @param directoryBaseName - Base directory name used for entrypoint prioritization.
 * @returns File path and summary when a promotable directory intro exists.
 */
function resolvePrimaryDirectoryIntro(
  state: GenerateDocsState,
  directoryPath: string,
  sortedFiles: readonly string[],
  fileSymbolMap: Map<string, RenderedSymbol[]>,
  directoryBaseName: string,
): { filePath: string; summary: RenderedFileSummary } | undefined {
  const configuredIntroFile = getCachedDirectoryDocsOrderConfig(
    state,
    directoryPath,
  )?.config.introFile;
  if (configuredIntroFile) {
    warnForConfiguredIntroFileIfNeeded(
      state,
      directoryPath,
      configuredIntroFile,
      sortedFiles,
    );

    const configuredIntroFilePath = sortedFiles.find(
      (filePath) => path.basename(filePath) === configuredIntroFile,
    );
    if (configuredIntroFilePath) {
      const configuredIntroSummary = resolveRenderedFileSummary(
        configuredIntroFilePath,
        fileSymbolMap,
      );

      if (
        configuredIntroSummary.description ||
        configuredIntroSummary.examples?.length
      ) {
        return {
          filePath: configuredIntroFilePath,
          summary: configuredIntroSummary,
        };
      }
    }
  }

  const prioritizedFiles = sortedFiles.toSorted((leftFile, rightFile) => {
    const leftBaseName = path.basename(leftFile, '.ts');
    const rightBaseName = path.basename(rightFile, '.ts');
    const leftIsDirectoryEntry =
      leftBaseName === directoryBaseName || leftBaseName === 'index';
    const rightIsDirectoryEntry =
      rightBaseName === directoryBaseName || rightBaseName === 'index';

    if (leftIsDirectoryEntry !== rightIsDirectoryEntry) {
      return leftIsDirectoryEntry ? -1 : 1;
    }

    return sortedFiles.indexOf(leftFile) - sortedFiles.indexOf(rightFile);
  });

  for (const filePath of prioritizedFiles) {
    const summary = resolveRenderedFileSummary(filePath, fileSymbolMap);

    if (summary.description || summary.examples?.length) {
      return { filePath, summary };
    }
  }

  return undefined;
}

/**
 * Resolves the rendered file-summary block for one collected source file.
 *
 * @param filePath - Absolute source file path.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @returns Rendered file summary values when present.
 */
function resolveRenderedFileSummary(
  filePath: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): RenderedFileSummary {
  const fileSummarySymbol = (fileSymbolMap.get(filePath) ?? []).find(
    (renderedSymbol) => renderedSymbol.kind === 'File',
  );

  return {
    description: fileSummarySymbol?.jsdoc.description,
    examples: fileSummarySymbol?.jsdoc.examples,
  };
}

/**
 * Groups symbols that have parent names.
 *
 * @param renderedSymbols - Symbols for one file.
 * @returns Parent-keyed symbol map.
 */
function groupSymbolsByParent(
  renderedSymbols: readonly RenderedSymbol[],
): Map<string, RenderedSymbol[]> {
  const symbolsByParent = new Map<string, RenderedSymbol[]>();

  for (const renderedSymbol of renderedSymbols.filter(
    (symbol) => symbol.parent,
  )) {
    const childSymbols = symbolsByParent.get(renderedSymbol.parent!) ?? [];
    childSymbols.push(renderedSymbol);
    symbolsByParent.set(renderedSymbol.parent!, childSymbols);
  }

  return symbolsByParent;
}

/**
 * Renders one symbol block into the README line buffer.
 *
 * @param lines - Output line buffer.
 * @param renderedSymbol - Symbol to render.
 * @param headingLevel - Markdown heading level.
 * @returns Nothing.
 */
function renderSymbolBlock(
  lines: string[],
  renderedSymbol: RenderedSymbol,
  headingLevel: number,
): void {
  const symbolDescription =
    renderedSymbol.jsdoc.description || renderedSymbol.jsdoc.summary;

  lines.push(`${'#'.repeat(headingLevel)} ${renderedSymbol.name}`);

  if (renderedSymbol.signature) {
    lines.push(
      '',
      ...renderSignatureBlock(renderedSymbol.name, renderedSymbol.signature),
    );
  }

  if (symbolDescription) {
    lines.push('', symbolDescription);
  }

  if (renderedSymbol.jsdoc.deprecated) {
    lines.push('', `**Deprecated:** ${renderedSymbol.jsdoc.deprecated}`);
  }

  if (renderedSymbol.jsdoc.params?.length) {
    lines.push('', 'Parameters:');
    for (const parameter of renderedSymbol.jsdoc.params) {
      const normalizedParamDoc = normalizeParameterDoc(parameter.doc);
      lines.push(
        `- \`${parameter.name}\`${normalizedParamDoc ? ` - ${normalizedParamDoc}` : ''}`,
      );
    }
  }

  if (renderedSymbol.jsdoc.returns) {
    lines.push('', `Returns: ${renderedSymbol.jsdoc.returns}`);
  }

  if (renderedSymbol.jsdoc.examples?.length) {
    lines.push(
      '',
      renderedSymbol.jsdoc.examples.length === 1 ? 'Example:' : 'Examples:',
    );

    for (const example of renderedSymbol.jsdoc.examples) {
      lines.push('', example);
    }
  }

  lines.push('');
}

/**
 * Normalizes parameter documentation extracted from JSDoc.
 *
 * The generator renders parameter bullets as `- \`name\` - description`.
 * If the extracted doc already begins with `-` (common when authors write
 * `@param name - description`), the README ends up with a doubled dash.
 *
 * @param doc Raw parameter doc string from the JSDoc parser.
 * @returns Normalized parameter doc string safe for bullet rendering.
 */
function normalizeParameterDoc(doc: string | undefined): string {
  if (!doc) {
    return '';
  }

  const trimmed = doc.trim();
  return trimmed.replace(/^-(\s+)/, '');
}

/**
 * Compares file-section symbols using the stable fallback order.
 *
 * @param left - Left rendered symbol.
 * @param right - Right rendered symbol.
 * @param fileBaseName - Base file name without extension.
 * @returns Negative when left sorts before right.
 */
function compareFileSectionSymbols(
  left: RenderedSymbol,
  right: RenderedSymbol,
  fileBaseName: string,
): number {
  const leftIsPrimary = !left.parent && left.name === fileBaseName;
  const rightIsPrimary = !right.parent && right.name === fileBaseName;
  if (leftIsPrimary !== rightIsPrimary) {
    return leftIsPrimary ? -1 : 1;
  }

  return (
    (left.parent || left.name).localeCompare(right.parent || right.name) ||
    left.name.localeCompare(right.name)
  );
}
