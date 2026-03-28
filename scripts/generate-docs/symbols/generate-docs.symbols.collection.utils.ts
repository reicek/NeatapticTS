/*
 * Collects docs-ready symbols from ts-morph source files.
 *
 * This chapter owns source-file scanning, export walking, and the bookkeeping
 * needed to accumulate one directory/file symbol map without pulling JSDoc
 * parsing or signature formatting concerns into the traversal flow.
 */

import * as path from 'path';

import type { SourceFile } from 'ts-morph';

import { FILE_SUMMARY_SYMBOL_NAME } from '../generate-docs.constants.js';
import type {
  DirectorySymbolMap,
  RenderedSymbol,
} from '../generate-docs.types.js';
import { resolveFileSummary } from './generate-docs.symbols.jsdoc.utils.js';
import {
  renderDeclaration,
  renderSymbol,
} from './generate-docs.symbols.render.utils.js';

/**
 * Collects renderable symbols grouped by directory and then by file.
 *
 * @param sourceFiles - Loaded source files to scan.
 * @returns Nested symbol map keyed by directory and then source file path.
 */
export function collectDirectorySymbols(
  sourceFiles: readonly SourceFile[],
): DirectorySymbolMap {
  const directorySymbolMap: DirectorySymbolMap = new Map();

  for (const sourceFile of sourceFiles) {
    collectSourceFileSymbols(sourceFile, directorySymbolMap);
  }

  return directorySymbolMap;
}

/**
 * Collects every renderable symbol from a single source file.
 *
 * @param sourceFile - Source file being scanned.
 * @param directorySymbolMap - Aggregate directory symbol map.
 * @returns Nothing.
 */
function collectSourceFileSymbols(
  sourceFile: SourceFile,
  directorySymbolMap: DirectorySymbolMap,
): void {
  const filePath = sourceFile.getFilePath();
  const fileSymbolMap = resolveFileSymbolMap(directorySymbolMap, filePath);

  addFileSummarySymbol(sourceFile, fileSymbolMap);
  collectExportedDeclarations(sourceFile, fileSymbolMap);
  collectDocumentedTopLevelDeclarations(sourceFile, fileSymbolMap);
}

/**
 * Resolves the mutable file-symbol map for the file's directory.
 *
 * @param directorySymbolMap - Aggregate directory symbol map.
 * @param filePath - Absolute source file path.
 * @returns Per-directory file symbol map.
 */
function resolveFileSymbolMap(
  directorySymbolMap: DirectorySymbolMap,
  filePath: string,
): Map<string, RenderedSymbol[]> {
  const directoryPath = path.dirname(filePath);
  let fileSymbolMap = directorySymbolMap.get(directoryPath);

  if (!fileSymbolMap) {
    fileSymbolMap = new Map();
    directorySymbolMap.set(directoryPath, fileSymbolMap);
  }

  return fileSymbolMap;
}

/**
 * Adds a synthetic file-summary symbol when the file starts with a file-level
 * JSDoc block.
 *
 * @param sourceFile - Source file being scanned.
 * @param fileSymbolMap - Mutable symbol map for the file's directory.
 * @returns Nothing.
 */
function addFileSummarySymbol(
  sourceFile: SourceFile,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): void {
  const fileSummary = resolveFileSummary(sourceFile);
  if (!fileSummary.description && !fileSummary.examples?.length) {
    return;
  }

  appendRenderedSymbol(fileSymbolMap, sourceFile.getFilePath(), {
    kind: 'File',
    name: FILE_SUMMARY_SYMBOL_NAME,
    filePath: sourceFile.getFilePath(),
    jsdoc: {
      description: fileSummary.description,
      examples: fileSummary.examples,
    },
  });

  console.log(`[docs] Added file summary for ${sourceFile.getBaseName()}`);
}

/**
 * Collects exported declarations and supported child members.
 *
 * @param sourceFile - Source file being scanned.
 * @param fileSymbolMap - Mutable symbol map for the file's directory.
 * @returns Nothing.
 */
function collectExportedDeclarations(
  sourceFile: SourceFile,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): void {
  const exportedDeclarations = sourceFile.getExportedDeclarations();
  if (exportedDeclarations.size === 0) {
    return;
  }

  console.log(
    `[docs] ${sourceFile.getBaseName()} exports: ${exportedDeclarations.size}`,
  );

  for (const [, declarations] of exportedDeclarations) {
    for (const declaration of declarations) {
      const symbol = declaration.getSymbol();
      if (!symbol) {
        continue;
      }

      const renderedSymbol = renderSymbol(
        symbol,
        declaration.getKindName(),
        sourceFile.getFilePath(),
      );
      if (renderedSymbol) {
        appendRenderedSymbol(
          fileSymbolMap,
          sourceFile.getFilePath(),
          renderedSymbol,
        );
      }

      collectExportedObjectLiteralMembers(
        declaration as unknown as Record<string, unknown>,
        symbol.getName(),
        sourceFile.getFilePath(),
        fileSymbolMap,
      );
      collectExportedClassMembers(
        declaration as unknown as Record<string, unknown>,
        symbol.getName(),
        sourceFile.getFilePath(),
        fileSymbolMap,
      );
    }
  }
}

/**
 * Collects documented properties from exported object-literal variables.
 *
 * @param declaration - Candidate declaration.
 * @param parentName - Parent exported symbol name.
 * @param filePath - Absolute source file path.
 * @param fileSymbolMap - Mutable symbol map for the file's directory.
 * @returns Nothing.
 */
function collectExportedObjectLiteralMembers(
  declaration: Record<string, unknown>,
  parentName: string,
  filePath: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): void {
  try {
    const declarationKindName = (
      declaration.getKindName as (() => string) | undefined
    )?.();
    if (declarationKindName !== 'VariableDeclaration') {
      return;
    }

    const initializer = (
      declaration.getInitializer as (() => any) | undefined
    )?.();
    if (initializer?.getKindName?.() !== 'ObjectLiteralExpression') {
      return;
    }

    const properties = (initializer.getProperties?.() as any[]) || [];
    for (const property of properties) {
      const propertyName =
        property.getName?.() || property.getSymbol?.()?.getName?.();
      const renderedProperty = renderDeclaration(
        property,
        String(propertyName),
        property.getKindName?.() || 'Property',
        filePath,
        parentName,
      );

      if (renderedProperty) {
        appendRenderedSymbol(fileSymbolMap, filePath, renderedProperty);
      }
    }
  } catch {
    // Best-effort introspection only.
  }
}

/**
 * Collects documented members from exported classes.
 *
 * @param declaration - Candidate declaration.
 * @param parentName - Parent exported symbol name.
 * @param filePath - Absolute source file path.
 * @param fileSymbolMap - Mutable symbol map for the file's directory.
 * @returns Nothing.
 */
function collectExportedClassMembers(
  declaration: Record<string, unknown>,
  parentName: string,
  filePath: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): void {
  try {
    const declarationKindName = (
      declaration.getKindName as (() => string) | undefined
    )?.();
    if (declarationKindName !== 'ClassDeclaration') {
      return;
    }

    const members =
      (declaration.getMembers as (() => any[]) | undefined)?.() || [];
    for (const member of members) {
      const memberName = member.getName?.();
      if (!memberName) {
        continue;
      }

      const renderedMember = renderDeclaration(
        member,
        String(memberName),
        member.getKindName?.() || 'ClassMember',
        filePath,
        parentName,
      );

      if (renderedMember) {
        appendRenderedSymbol(fileSymbolMap, filePath, renderedMember);
      }
    }
  } catch {
    // Best-effort introspection only.
  }
}

/**
 * Collects documented top-level non-exported declarations.
 *
 * @param sourceFile - Source file being scanned.
 * @param fileSymbolMap - Mutable symbol map for the file's directory.
 * @returns Nothing.
 */
function collectDocumentedTopLevelDeclarations(
  sourceFile: SourceFile,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): void {
  try {
    const statements =
      (
        sourceFile as unknown as { getStatements?: () => any[] }
      ).getStatements?.() || [];

    for (const statement of statements) {
      const declarations = statement.getDeclarations?.() || [statement];

      for (const declaration of declarations) {
        const jsDocs = declaration.getJsDocs?.() || [];
        if (jsDocs.length === 0) {
          continue;
        }

        const name =
          declaration.getName?.() || declaration.getSymbol?.()?.getName?.();
        if (!name) {
          continue;
        }

        if (
          hasRenderedSymbol(
            fileSymbolMap,
            sourceFile.getFilePath(),
            String(name),
          )
        ) {
          continue;
        }

        const renderedDeclaration = renderDeclaration(
          declaration,
          String(name),
          declaration.getKindName?.() || 'Declaration',
          sourceFile.getFilePath(),
        );

        if (renderedDeclaration) {
          appendRenderedSymbol(
            fileSymbolMap,
            sourceFile.getFilePath(),
            renderedDeclaration,
          );
        }
      }
    }
  } catch {
    // Best-effort introspection only.
  }
}

/**
 * Checks whether a symbol name has already been collected for a file.
 *
 * @param fileSymbolMap - Mutable symbol map for the directory.
 * @param filePath - Absolute source file path.
 * @param name - Candidate symbol name.
 * @returns True when the symbol already exists in the collected set.
 */
function hasRenderedSymbol(
  fileSymbolMap: Map<string, RenderedSymbol[]>,
  filePath: string,
  name: string,
): boolean {
  return (fileSymbolMap.get(filePath) ?? []).some(
    (symbol) => symbol.name === name,
  );
}

/**
 * Appends a rendered symbol to the per-file collection.
 *
 * @param fileSymbolMap - Mutable symbol map for the directory.
 * @param filePath - Absolute source file path.
 * @param renderedSymbol - Symbol to append.
 * @returns Nothing.
 */
function appendRenderedSymbol(
  fileSymbolMap: Map<string, RenderedSymbol[]>,
  filePath: string,
  renderedSymbol: RenderedSymbol,
): void {
  const renderedSymbols = fileSymbolMap.get(filePath) ?? [];
  renderedSymbols.push(renderedSymbol);
  fileSymbolMap.set(filePath, renderedSymbols);
}
