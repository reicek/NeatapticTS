/*
 * Renders declarations and exported symbols into the docs-ready symbol model.
 *
 * This chapter isolates ts-morph declaration quirks from the collector so the
 * rest of the boundary can work with one normalized RenderedSymbol contract.
 */

import type { JSDocTag, Symbol as MorphSymbol } from 'ts-morph';

import type { RenderedSymbol } from '../generate-docs.types.js';
import {
  extractExampleDocs,
  extractParamDocs,
  getTagCommentText,
  hasInternalTag,
  resolveRenderableJsDocs,
} from './generate-docs.symbols.jsdoc.utils.js';
import { resolveCallSignature } from './generate-docs.symbols.signature.utils.js';

/**
 * Renders one exported symbol when it belongs to a supported declaration kind
 * and is not marked internal.
 *
 * @param symbol - Exported symbol.
 * @param fallbackKind - Declaration kind used as a fallback label.
 * @param filePath - Absolute source file path.
 * @returns Rendered symbol or null when not supported.
 */
export function renderSymbol(
  symbol: MorphSymbol,
  fallbackKind: string,
  filePath: string,
): RenderedSymbol | null {
  const declaration = symbol.getDeclarations()[0];
  if (!declaration) {
    return null;
  }

  const declarationKind = declaration.getKindName();
  const supportedKinds =
    /ClassDeclaration|FunctionDeclaration|InterfaceDeclaration|EnumDeclaration|TypeAliasDeclaration|VariableDeclaration/;
  if (!supportedKinds.test(declarationKind)) {
    return null;
  }

  const jsDocs = resolveRenderableJsDocs(declaration);
  if (hasInternalTag(jsDocs)) {
    return null;
  }

  const primaryJsDoc = jsDocs.at(-1);
  const fullDescription = primaryJsDoc?.getDescription().trim();

  return {
    kind: fallbackKind || declarationKind,
    name: symbol.getName(),
    parent: undefined,
    filePath,
    signature: resolveCallSignature(declaration),
    jsdoc: {
      summary: fullDescription?.split(/\r?\n\r?\n/)[0]?.trim(),
      description: fullDescription,
      params: extractParamDocs(primaryJsDoc?.getTags() || []),
      examples: extractExampleDocs(primaryJsDoc?.getTags() || []),
      returns: getTagCommentText(
        primaryJsDoc
          ?.getTags()
          .find(
            (tag: JSDocTag) =>
              tag.getTagName() === 'returns' || tag.getTagName() === 'return',
          ),
      ),
      deprecated: getTagCommentText(
        primaryJsDoc
          ?.getTags()
          .find((tag: JSDocTag) => tag.getTagName() === 'deprecated'),
      ),
    },
  };
}

/**
 * Renders a declaration-like node that may not have a directly usable symbol.
 *
 * @param declaration - Declaration-like node.
 * @param forcedName - Optional forced display name.
 * @param forcedKind - Optional forced kind label.
 * @param filePath - Absolute source file path.
 * @param parentName - Optional parent display name.
 * @returns Rendered symbol or null when not renderable.
 */
export function renderDeclaration(
  declaration: any,
  forcedName?: string,
  forcedKind?: string,
  filePath?: string,
  parentName?: string,
): RenderedSymbol | null {
  if (!declaration) {
    return null;
  }

  try {
    const jsDocs = resolveRenderableJsDocs(declaration);
    if (jsDocs.length === 0 || hasInternalTag(jsDocs)) {
      return null;
    }

    const primaryJsDoc = jsDocs.at(-1);
    const fullDescription = primaryJsDoc?.getDescription?.()?.trim();

    return {
      kind:
        forcedKind ||
        declaration.getKindName?.() ||
        declaration.getKind?.() ||
        'Declaration',
      name:
        forcedName ||
        declaration.getName?.() ||
        declaration.getSymbol?.()?.getName?.() ||
        (parentName ? `${parentName}.${forcedName}` : ''),
      parent: parentName || undefined,
      filePath: filePath || '',
      signature: resolveCallSignature(declaration),
      jsdoc: {
        summary: fullDescription?.split(/\r?\n\r?\n/)[0]?.trim(),
        description: fullDescription,
        params: extractParamDocs(primaryJsDoc?.getTags() || []),
        examples: extractExampleDocs(primaryJsDoc?.getTags() || []),
        returns: getTagCommentText(
          primaryJsDoc
            ?.getTags()
            .find(
              (tag: JSDocTag) =>
                tag.getTagName() === 'returns' || tag.getTagName() === 'return',
            ),
        ),
        deprecated: getTagCommentText(
          primaryJsDoc
            ?.getTags()
            .find((tag: JSDocTag) => tag.getTagName() === 'deprecated'),
        ),
      },
    };
  } catch {
    return null;
  }
}
