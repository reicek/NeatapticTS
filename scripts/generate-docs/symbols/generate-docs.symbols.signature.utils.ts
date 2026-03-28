/*
 * Formats and parses callable signature text for generated README output.
 *
 * ts-morph returns rich type text that is useful but often noisy for docs.
 * This chapter normalizes those signatures into stable stored strings and
 * re-expands them into readable fenced blocks when rendering markdown.
 */

import * as path from 'path';

import { WORKSPACE_ROOT_DIR } from '../generate-docs.constants.js';

/**
 * Formats a stored call signature into a human-friendly TypeScript block.
 *
 * @param symbolName - Symbol name shown in the heading.
 * @param signature - Stored canonical call signature.
 * @returns Fenced TypeScript block lines.
 */
export function renderSignatureBlock(
  symbolName: string,
  signature: string,
): string[] {
  const parsedSignature = parseCallSignature(signature);
  if (!parsedSignature) {
    return ['```ts', `${symbolName}${signature}`, '```'];
  }

  if (parsedSignature.parameters.length === 0) {
    return ['```ts', `${symbolName}(): ${parsedSignature.returnType}`, '```'];
  }

  return [
    '```ts',
    `${symbolName}(`,
    ...parsedSignature.parameters.map((parameter) => `  ${parameter},`),
    `): ${parsedSignature.returnType}`,
    '```',
  ];
}

/**
 * Resolves a call signature string from a declaration-like node.
 *
 * The stored format is the canonical round-trip shape used by this boundary:
 * `(param: Type, other: Type) => ReturnType`. `renderSignatureBlock` later
 * parses that value back into a fenced TypeScript block for generated README
 * sections.
 *
 * Non-callable declarations intentionally return `undefined` so symbol
 * collection can stay best-effort instead of treating missing signatures as a
 * hard failure.
 *
 * @param declaration - Declaration-like node.
 * @returns Signature string when the declaration is callable.
 * @example
 * const signature = resolveCallSignature(declaration);
 * // => "(value: number) => string"
 */
export function resolveCallSignature(declaration: any): string | undefined {
  try {
    const declarationType =
      declaration.getType?.() || declaration.getSymbol?.()?.getType?.();
    const callSignature = declarationType?.getCallSignatures?.()[0];
    if (!callSignature) {
      return undefined;
    }

    const renderedParameters = callSignature
      .getParameters()
      .map((parameter: any) => {
        const parameterDeclarations = parameter.getDeclarations();
        const parameterType = normalizeRenderedTypeText(
          parameter
            .getTypeAtLocation(parameterDeclarations[0] || declaration)
            .getText(),
        );
        return `${parameter.getName()}: ${parameterType}`;
      })
      .join(', ');

    const returnType = normalizeRenderedTypeText(
      callSignature.getReturnType().getText(),
    );
    return `(${renderedParameters}) => ${returnType}`;
  } catch {
    return undefined;
  }
}

/**
 * Extracts the balanced parameter-list portion of a stored call signature.
 *
 * @param signature - Stored canonical call signature.
 * @returns Parameter-list text including the outer parentheses when recognized.
 * @example
 * extractCallSignatureParameterList('(value: number, cb: (x: number) => void) => string');
 * // => '(value: number, cb: (x: number) => void)'
 */
export function extractCallSignatureParameterList(
  signature: string,
): string | undefined {
  if (!signature.startsWith('(')) {
    return undefined;
  }

  const closingParenthesisIndex = findMatchingDelimiter(signature, 0, '(', ')');
  if (closingParenthesisIndex === -1) {
    return undefined;
  }

  return signature.slice(0, closingParenthesisIndex + 1);
}

/**
 * Parses a canonical stored call signature.
 *
 * @param signature - Stored canonical call signature.
 * @returns Parsed parameters and return type when recognized.
 */
function parseCallSignature(
  signature: string,
): { parameters: string[]; returnType: string } | undefined {
  const parameterList = extractCallSignatureParameterList(signature);
  if (!parameterList) {
    return undefined;
  }

  const parameterListText = parameterList.slice(1, -1);
  const returnTypePrefix = signature.slice(parameterList.length).trimStart();
  if (!returnTypePrefix.startsWith('=>')) {
    return undefined;
  }

  return {
    parameters: splitTopLevelCommaSeparated(parameterListText),
    returnType: returnTypePrefix.slice(2).trim(),
  };
}

/**
 * Splits a comma-separated type list while respecting nested delimiters.
 *
 * @param value - Comma-separated text.
 * @returns Top-level list entries.
 */
function splitTopLevelCommaSeparated(value: string): string[] {
  if (!value.trim()) {
    return [];
  }

  const segments: string[] = [];
  let segmentStartIndex = 0;
  let parenthesisDepth = 0;
  let bracketDepth = 0;
  let braceDepth = 0;
  let angleDepth = 0;

  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    if (character === '(') {
      parenthesisDepth += 1;
      continue;
    }

    if (character === ')') {
      parenthesisDepth -= 1;
      continue;
    }

    if (character === '[') {
      bracketDepth += 1;
      continue;
    }

    if (character === ']') {
      bracketDepth -= 1;
      continue;
    }

    if (character === '{') {
      braceDepth += 1;
      continue;
    }

    if (character === '}') {
      braceDepth -= 1;
      continue;
    }

    if (character === '<') {
      angleDepth += 1;
      continue;
    }

    if (character === '>') {
      angleDepth = Math.max(0, angleDepth - 1);
      continue;
    }

    const isTopLevelComma =
      character === ',' &&
      parenthesisDepth === 0 &&
      bracketDepth === 0 &&
      braceDepth === 0 &&
      angleDepth === 0;
    if (!isTopLevelComma) {
      continue;
    }

    segments.push(value.slice(segmentStartIndex, index).trim());
    segmentStartIndex = index + 1;
  }

  segments.push(value.slice(segmentStartIndex).trim());
  return segments.filter(Boolean);
}

/**
 * Finds the matching closing delimiter for a starting delimiter.
 *
 * @param value - Text to inspect.
 * @param startIndex - Opening delimiter index.
 * @param openingDelimiter - Opening delimiter character.
 * @param closingDelimiter - Closing delimiter character.
 * @returns Closing delimiter index when found.
 */
function findMatchingDelimiter(
  value: string,
  startIndex: number,
  openingDelimiter: string,
  closingDelimiter: string,
): number {
  let delimiterDepth = 0;

  for (let index = startIndex; index < value.length; index += 1) {
    const character = value[index];
    if (character === openingDelimiter) {
      delimiterDepth += 1;
      continue;
    }

    if (character !== closingDelimiter) {
      continue;
    }

    delimiterDepth -= 1;
    if (delimiterDepth === 0) {
      return index;
    }
  }

  return -1;
}

/**
 * Rewrites absolute import paths from ts-morph type text into repo-relative paths.
 *
 * @param typeText - Raw type text returned by ts-morph.
 * @returns Normalized type text safe for generated docs.
 */
function normalizeRenderedTypeText(typeText: string): string {
  const normalizedImportPaths = typeText.replace(
    /import\((['"])([^'"]+)\1\)/g,
    (_match, quote: string, importPath: string) => {
      const normalizedImportPath = path.normalize(importPath);
      if (!path.isAbsolute(normalizedImportPath)) {
        return `import(${quote}${importPath}${quote})`;
      }

      const relativeImportPath = path
        .relative(WORKSPACE_ROOT_DIR, normalizedImportPath)
        .replace(/\\/g, '/');

      const portableImportPath = relativeImportPath.startsWith('..')
        ? importPath.replace(/\\/g, '/')
        : relativeImportPath;

      return `import(${quote}${portableImportPath}${quote})`;
    },
  );

  return normalizedImportPaths
    .replace(
      /\btypeof\s+import\((['"])([^'"]+)\1\)\.([A-Za-z_$][\w$]*)/g,
      'typeof $3',
    )
    .replace(/import\((['"])([^'"]+)\1\)\.([A-Za-z_$][\w$]*)/g, '$3');
}
