/*
 * Parses file-summary and declaration-level JSDoc for the symbols boundary.
 *
 * The collector and renderer both depend on these helpers, so this chapter is
 * the shared JSDoc vocabulary for symbol extraction, file-summary promotion,
 * and tag normalization.
 */

import type { JSDocTag, SourceFile } from 'ts-morph';

import type {
  RenderedFileSummary,
  RenderedParameter,
} from '../generate-docs.types.js';

/**
 * Resolves the renderable file-summary JSDoc for a source file.
 *
 * Precedence is intentionally stable so generated folder intros are
 * predictable:
 * 1. a leading top-of-file JSDoc block,
 * 2. attached module-summary blocks that appear before the first declaration,
 * 3. the first ordinary JSDoc description reachable from the file node.
 *
 * @param sourceFile - Source file to inspect.
 * @returns File summary description/examples when present.
 */
export function resolveFileSummary(
  sourceFile: SourceFile,
): RenderedFileSummary {
  const leadingFileSummary = extractLeadingFileJsDocSummary(sourceFile);
  if (leadingFileSummary.description || leadingFileSummary.examples?.length) {
    return leadingFileSummary;
  }

  const attachedFileSummary = extractAttachedFileJsDocSummary(sourceFile);
  if (attachedFileSummary.description || attachedFileSummary.examples?.length) {
    return attachedFileSummary;
  }

  return {
    description: getFirstJsDocDescription(
      sourceFile as unknown as { getJsDocs?: () => unknown[] },
    ),
  };
}

/**
 * Returns JSDoc blocks from a declaration-like node.
 *
 * @param node - Declaration-like node.
 * @returns JSDoc array.
 */
export function getJsDocs(node: any): any[] {
  return (node?.getJsDocs?.() as any[]) || [];
}

/**
 * Resolves the JSDoc blocks that should be used for a renderable declaration.
 *
 * Exported constants are commonly documented on their enclosing
 * `VariableStatement` instead of on the inner `VariableDeclaration`. This
 * helper falls back to that enclosing statement so generated README entries for
 * constants keep their descriptions.
 *
 * @param node - Declaration-like node.
 * @returns JSDoc array, preferring direct docs and falling back when needed.
 */
export function resolveRenderableJsDocs(node: any): any[] {
  const directJsDocs = getJsDocs(node);
  if (directJsDocs.length > 0) {
    return directJsDocs;
  }

  if (node?.getKindName?.() !== 'VariableDeclaration') {
    return directJsDocs;
  }

  const variableStatement =
    node.getFirstAncestorByKindName?.('VariableStatement') ||
    node.getVariableStatement?.();
  return getJsDocs(variableStatement);
}

/**
 * Resolves the first JSDoc description from a declaration-like node.
 *
 * @param node - Declaration-like node.
 * @returns Description text when present.
 */
export function getFirstJsDocDescription(node: {
  getJsDocs?: () => unknown[];
}): string | undefined {
  const firstJsDoc = getJsDocs(node)[0];
  return firstJsDoc?.getDescription?.()?.trim() as string | undefined;
}

/**
 * Checks whether any provided JSDoc block contains an internal tag.
 *
 * @param jsDocs - JSDoc array to inspect.
 * @returns True when the declaration is marked internal.
 */
export function hasInternalTag(jsDocs: readonly any[]): boolean {
  return jsDocs.some((jsDoc) =>
    jsDoc
      .getTags()
      .some((tag: { getTagName(): string }) => tag.getTagName() === 'internal'),
  );
}

/**
 * Extracts rendered parameter docs from JSDoc tags.
 *
 * @param tags - JSDoc tags to inspect.
 * @returns Parameter docs when present.
 */
export function extractParamDocs(
  tags: readonly JSDocTag[],
): RenderedParameter[] | undefined {
  const parameterDocs = tags
    .filter((tag) => tag.getTagName() === 'param')
    .map((tag) => {
      const match = tag.getText().match(/@param\s+(\w+)/);
      return {
        name: match?.[1] || '',
        doc: getTagCommentText(tag),
      } satisfies RenderedParameter;
    })
    .filter((parameter) => Boolean(parameter.name));

  return parameterDocs.length > 0 ? parameterDocs : undefined;
}

/**
 * Extracts rendered example blocks from JSDoc tags.
 *
 * @param tags - JSDoc tags to inspect.
 * @returns Example blocks when present.
 */
export function extractExampleDocs(
  tags: readonly JSDocTag[],
): string[] | undefined {
  const examples = tags
    .filter((tag) => tag.getTagName() === 'example')
    .map((tag) => getExampleTagText(tag))
    .filter((example): example is string => Boolean(example));

  return examples.length > 0 ? examples : undefined;
}

/**
 * Resolves a tag comment into plain text.
 *
 * @param tag - JSDoc tag to inspect.
 * @returns Flattened comment string when present.
 */
export function getTagCommentText(
  tag: JSDocTag | undefined,
): string | undefined {
  const rawComment = tag?.getComment();
  if (typeof rawComment === 'string') {
    return sanitizeTagCommentText(rawComment);
  }

  if (Array.isArray(rawComment)) {
    return sanitizeTagCommentText(
      rawComment
        .map(
          (commentPart) =>
            (commentPart as { getText?: () => string }).getText?.() ||
            String(commentPart),
        )
        .join(' ')
        .trim(),
    );
  }

  return undefined;
}

/**
 * Normalizes multiline tag blocks while preserving markdown structure.
 *
 * @param blockText - Multiline tag block text.
 * @returns Cleaned block text or undefined when empty.
 */
export function sanitizeTagBlockText(
  blockText: string | undefined,
): string | undefined {
  const normalizedBlock = blockText
    ?.replace(/\r\n/g, '\n')
    .replace(/^\s*\* ?/gm, '')
    .replace(/^\n+/, '')
    .replace(/\n+$/, '')
    .trim();

  return normalizedBlock ? normalizedBlock : undefined;
}

/**
 * Remove ts-morph JSDoc tag artifacts such as standalone trailing asterisks.
 *
 * @param commentText - Flattened tag comment text.
 * @returns Cleaned comment text or undefined when nothing meaningful remains.
 */
function sanitizeTagCommentText(
  commentText: string | undefined,
): string | undefined {
  const normalizedComment = commentText
    ?.replace(/\r\n/g, '\n')
    .replace(/\n\s*\*\s*$/g, '')
    .replace(/^\s*\*\s*$/g, '')
    .trim();

  return normalizedComment ? normalizedComment : undefined;
}

/**
 * Extracts and parses the leading file-level JSDoc block from raw source text.
 *
 * @param sourceFile - Source file to inspect.
 * @returns Parsed top-of-file summary description/examples when present.
 */
function extractLeadingFileJsDocSummary(
  sourceFile: SourceFile,
): RenderedFileSummary {
  const text = sourceFile.getFullText();
  const match = text.match(/^\s*(?:\uFEFF)?\/\*\*([\s\S]*?)\*\//);
  if (!match) {
    return {};
  }

  const cleanedText = match[1]
    .split(/\r?\n/)
    .map((line) => line.replace(/^\s*\*\s?/, ''))
    .join('\n')
    .trim();

  if (!cleanedText) {
    return {};
  }

  if (
    cleanedText.split('\n').some((line) => line.trim().startsWith('@internal'))
  ) {
    return {};
  }

  return parseFileSummaryBlock(cleanedText);
}

/**
 * Parses a cleaned file-summary JSDoc block into description and examples.
 *
 * @param cleanedText - Cleaned raw file-summary block text.
 * @returns Parsed description/examples.
 */
function parseFileSummaryBlock(cleanedText: string): RenderedFileSummary {
  const tagStartIndex = cleanedText.search(/^@\w+/m);
  const description = sanitizeTagBlockText(
    tagStartIndex >= 0 ? cleanedText.slice(0, tagStartIndex) : cleanedText,
  );
  const examples = extractFileSummaryExamples(cleanedText);

  return {
    description,
    examples,
  };
}

/**
 * Extracts file-summary example blocks from a cleaned raw JSDoc block.
 *
 * @param cleanedText - Cleaned raw file-summary block text.
 * @returns Example blocks when present.
 */
function extractFileSummaryExamples(cleanedText: string): string[] | undefined {
  const exampleMatches = [
    ...cleanedText.matchAll(/(^|\n)@example\s*([\s\S]*?)(?=\n@\w+|$)/g),
  ];

  const examples = exampleMatches
    .map((match) => sanitizeTagBlockText(match[2]))
    .filter((example): example is string => Boolean(example));

  return examples.length > 0 ? examples : undefined;
}

/**
 * Extracts module-summary JSDoc blocks that are attached to the first
 * top-level declaration after imports/exports.
 *
 * @param sourceFile - Source file to inspect.
 * @returns Parsed description/examples when attached module-summary blocks exist.
 */
function extractAttachedFileJsDocSummary(
  sourceFile: SourceFile,
): RenderedFileSummary {
  const firstStatementWithMultipleJsDocs = sourceFile
    .getStatements()
    .find((statement) => {
      const statementJsDocs = getJsDocs(statement);
      return statementJsDocs.length > 1 && !hasInternalTag(statementJsDocs);
    });

  if (!firstStatementWithMultipleJsDocs) {
    return {};
  }

  const summaryJsDocs = getJsDocs(firstStatementWithMultipleJsDocs).slice(
    0,
    -1,
  );
  return mergeRenderedFileSummaries(summaryJsDocs.map(renderJsDocSummaryBlock));
}

/**
 * Resolves an example tag into markdown-friendly multiline text.
 *
 * @param tag - JSDoc example tag.
 * @returns Example block content when present.
 */
function getExampleTagText(tag: JSDocTag | undefined): string | undefined {
  const rawTagText = tag?.getText()?.replace(/\r\n/g, '\n');
  if (!rawTagText) {
    return undefined;
  }

  const exampleBody = rawTagText.replace(/^@example\s*/, '');
  return sanitizeTagBlockText(exampleBody);
}

/**
 * Renders one JSDoc block into file-summary description/examples.
 *
 * @param jsDoc - JSDoc block to render.
 * @returns Parsed description/examples.
 */
function renderJsDocSummaryBlock(jsDoc: any): RenderedFileSummary {
  const description = sanitizeTagBlockText(jsDoc?.getDescription?.()?.trim());
  const examples = extractExampleDocs(jsDoc?.getTags?.() || []);

  return {
    description,
    examples,
  };
}

/**
 * Merges multiple parsed file-summary blocks.
 *
 * @param summaries - Parsed summary blocks.
 * @returns One merged file summary.
 */
function mergeRenderedFileSummaries(
  summaries: readonly RenderedFileSummary[],
): RenderedFileSummary {
  const descriptions = summaries
    .map((summary) => summary.description)
    .filter((description): description is string => Boolean(description));
  const examples = summaries
    .flatMap((summary) => summary.examples || [])
    .filter(
      (example, index, allExamples) => allExamples.indexOf(example) === index,
    );

  return {
    description:
      descriptions.length > 0 ? descriptions.join('\n\n') : undefined,
    examples: examples.length > 0 ? examples : undefined,
  };
}
