/*
 * Normalizes and deduplicates collected renderable symbols.
 *
 * The collector is intentionally permissive so it can gather multiple doc
 * entry points. This chapter folds that raw set into stable names and merged
 * JSDoc fields before README rendering begins.
 */

import * as path from 'path';

import { FILE_SUMMARY_SYMBOL_NAME } from '../generate-docs.constants.js';
import type {
  DirectorySymbolMap,
  RenderedSymbol,
} from '../generate-docs.types.js';

/**
 * Tidies and deduplicates collected symbols before README rendering.
 *
 * @param directorySymbolMap - Directory symbol map to normalize.
 * @returns Nothing.
 */
export function dedupeDirectorySymbols(
  directorySymbolMap: DirectorySymbolMap,
): void {
  for (const [, fileSymbolMap] of directorySymbolMap) {
    for (const [filePath, renderedSymbols] of fileSymbolMap) {
      fileSymbolMap.set(filePath, dedupeFileSymbols(renderedSymbols, filePath));
    }
  }
}

/**
 * Deduplicates symbols for a single file while conservatively merging JSDoc.
 *
 * @param renderedSymbols - Symbols collected for one file.
 * @param filePath - Absolute source file path.
 * @returns Deduplicated symbol list.
 */
function dedupeFileSymbols(
  renderedSymbols: readonly RenderedSymbol[],
  filePath: string,
): RenderedSymbol[] {
  const seen = new Map<string, RenderedSymbol>();
  const dedupedSymbols: RenderedSymbol[] = [];

  for (const renderedSymbol of renderedSymbols) {
    const normalizedName = normalizeName(renderedSymbol, filePath);
    const dedupeKey = `${renderedSymbol.parent || ''}::${normalizedName}::${renderedSymbol.signature || ''}`;
    const existingSymbol = seen.get(dedupeKey);

    if (existingSymbol) {
      mergeRenderedSymbols(existingSymbol, renderedSymbol);
      continue;
    }

    const clonedSymbol: RenderedSymbol = {
      ...renderedSymbol,
      name: normalizedName,
      jsdoc: {
        ...renderedSymbol.jsdoc,
        params: renderedSymbol.jsdoc.params?.slice(),
        examples: renderedSymbol.jsdoc.examples?.slice(),
      },
    };

    seen.set(dedupeKey, clonedSymbol);
    dedupedSymbols.push(clonedSymbol);
  }

  return dedupedSymbols;
}

/**
 * Merges missing JSDoc fields from a duplicate symbol into the retained one.
 *
 * @param target - Existing retained symbol.
 * @param incoming - Duplicate symbol carrying possible extra data.
 * @returns Nothing.
 */
function mergeRenderedSymbols(
  target: RenderedSymbol,
  incoming: RenderedSymbol,
): void {
  target.jsdoc.description =
    target.jsdoc.description || incoming.jsdoc.description;
  target.jsdoc.summary = target.jsdoc.summary || incoming.jsdoc.summary;
  target.jsdoc.deprecated =
    target.jsdoc.deprecated || incoming.jsdoc.deprecated;

  if (incoming.jsdoc.examples?.length) {
    target.jsdoc.examples = [
      ...(target.jsdoc.examples || []),
      ...incoming.jsdoc.examples.filter(
        (incomingExample) =>
          !(target.jsdoc.examples || []).includes(incomingExample),
      ),
    ];
  }

  if (incoming.jsdoc.params?.length) {
    target.jsdoc.params = (target.jsdoc.params || []).slice();

    for (const parameter of incoming.jsdoc.params) {
      const alreadyPresent = target.jsdoc.params.some(
        (existingParameter) => existingParameter.name === parameter.name,
      );
      if (!alreadyPresent) {
        target.jsdoc.params.push(parameter);
      }
    }
  }

  if (!target.signature && incoming.signature) {
    target.signature = incoming.signature;
  }
}

/**
 * Normalizes rendered symbol names so generated headings remain stable.
 *
 * @param renderedSymbol - Symbol being normalized.
 * @param filePath - Source file path used as fallback context.
 * @returns Stable display name.
 */
function normalizeName(
  renderedSymbol: RenderedSymbol,
  filePath: string,
): string {
  let name = String(renderedSymbol.name || '');

  if (!name || name === 'default' || name === FILE_SUMMARY_SYMBOL_NAME) {
    const fileBaseName = path.basename(filePath || '', '.ts');

    if (renderedSymbol.kind === 'File' || name === FILE_SUMMARY_SYMBOL_NAME) {
      return fileBaseName;
    }

    if (renderedSymbol.parent) {
      return `${renderedSymbol.parent}.${fileBaseName}`;
    }

    if (renderedSymbol.signature) {
      return `${fileBaseName}${renderedSymbol.signature.split(')')[0]})`;
    }

    return fileBaseName;
  }

  return name.replace(/^function\s+/, '').replace(/\(\)$/, '');
}
