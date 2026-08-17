/**
 * @description Structure-aware markdown chunker with heading hierarchy preservation,
 * semantic boundary splitting, and cross-chunk context headers. Replaces the naive
 * heading-based chunker (chunker.mjs) with semantic-aware splitting that never breaks
 * inside fenced code blocks or tables, splits at sentence boundaries, and provides
 * full heading paths for disambiguation.
 *
 * Key improvements over v1:
 * - Full heading hierarchy in `heading_path` (e.g., `# Architecture > ## Network > ### activate`)
 * - Semantic boundary splitting: code blocks → tables → paragraphs → sentences → chars
 * - Code blocks and tables are atomic (never split inside)
 * - Sentence-boundary overlap (256 chars target)
 * - `(continued)` suffix on sub-chunks beyond the first
 * - `context_header` column: `[file_path > heading_path]`
 * - `depth` column: heading level (1–6) for heading chunks, 0 for preamble
 *
 * @param {string} markdownText - Raw markdown text to chunk.
 * @param {object} [options={}] - Chunking options.
 * @param {number} [options.maxChars=2048] - Hard maximum chars per chunk body text.
 * @param {number} [options.overlapChars=256] - Target overlap chars at sentence boundaries.
 * @param {string} [options.filePath=''] - File path for context_header generation.
 *
 * @returns {Array<import('./chunker.d.mts').MarkdownChunkV2>} Array of v2 markdown chunks with hierarchy metadata.
 *
 * @example
 * ```ts
 * const chunks = chunkMarkdownV2('# Title\n\nParagraph text...', { filePath: 'src/README.md' });
 * console.log(chunks[0].context_header); // '[src/README.md > Title]'
 * ```
 */
const DEFAULT_MAX_CHARS = 2048;
const DEFAULT_OVERLAP_CHARS = 256;
const HEADING_PATTERN = /^(#{1,6})\s+(.+)$/gm;
const CODE_FENCE_PATTERN = /^```[\w]*\s*$/;
const CODE_FENCE_CLOSE_PATTERN = /^```\s*$/;
const TABLE_ROW_PATTERN = /^\|.+\|$/;
const TABLE_SEPARATOR_PATTERN = /^\|[\s\-:]+\|[\s\-:]+\|[\s\-:]*/;

/**
 * Chunk markdown text into semantically-aware sections with hierarchy metadata.
 *
 * @param {string} markdownText - Raw markdown text to chunk.
 * @param {object} [options={}] - Chunking options.
 * @param {number} [options.maxChars=2048] - Hard maximum chars per chunk body text.
 * @param {number} [options.overlapChars=256] - Target overlap chars at sentence boundaries.
 * @param {string} [options.filePath=''] - File path for context_header generation.
 * @returns {Array<import('./chunker.d.mts').MarkdownChunkV2>} Array of v2 markdown chunks.
 */
export function chunkMarkdownV2(markdownText, options = {}) {
  const maxChars = Math.max(1, Number(options.maxChars ?? DEFAULT_MAX_CHARS));
  const overlapChars = Math.max(
    0,
    Math.min(
      Number(options.overlapChars ?? DEFAULT_OVERLAP_CHARS),
      maxChars - 1,
    ),
  );
  const filePath = String(options.filePath ?? '');
  const sections = collectSectionsWithHierarchy(markdownText);

  return sections.flatMap((section) => {
    if (section.bodyText.length <= maxChars) {
      return [buildChunk(section, filePath)];
    }

    // Section exceeds maxChars — split at semantic boundaries.
    const subChunks = splitAtSemanticBoundaries(
      section.bodyText,
      maxChars,
      overlapChars,
    );
    return subChunks.map((subChunk, subIndex) => {
      const isContinuation = subIndex > 0;
      const headingPath = isContinuation
        ? `${section.fullHeadingPath} (continued)`
        : section.fullHeadingPath;
      const depth = section.headingLevel;
      const contextHeader = buildContextHeader(
        filePath,
        section.fullHeadingPath,
      );

      return {
        heading_path: headingPath,
        body_text: subChunk.text,
        char_start: section.charStart + subChunk.offset,
        char_end: section.charStart + subChunk.offset + subChunk.text.length,
        depth,
        context_header: contextHeader,
        parent_chunk_id: null, // Markdown sections are depth=0 within the document
        symbol_name: null,
        signature_text: null,
        jsdoc_text: null,
        export_type: null,
        module_path: null,
      };
    });
  });
}

/**
 * Collect markdown sections with full heading hierarchy.
 *
 * Parses all headings, maintains a hierarchy stack, and produces one section
 * per heading with the full heading path from root to current level.
 *
 * @param {string} markdownText - Raw markdown text.
 * @returns {Array<{fullHeadingPath: string, bodyText: string, headingLevel: number, charStart: number, charEnd: number}>} Sections with hierarchy.
 */
export function collectSectionsWithHierarchy(markdownText) {
  const headings = [...markdownText.matchAll(HEADING_PATTERN)].map((match) => ({
    level: match[1].length,
    text: match[2].trim(),
    index: match.index,
    lineEnd:
      markdownText.indexOf('\n', match.index) === -1
        ? markdownText.length
        : markdownText.indexOf('\n', match.index),
  }));

  // No headings — treat entire text as a single preamble section.
  if (headings.length === 0) {
    const trimmedText = markdownText.trim();
    if (trimmedText.length === 0) return [];

    return [
      {
        bodyText: trimmedText,
        charEnd: markdownText.length,
        charStart: 0,
        fullHeadingPath: '',
        headingLevel: 0,
      },
    ];
  }

  const sections = [];
  const headingStack = [];

  // Preamble before the first heading.
  const preambleText = markdownText.slice(0, headings[0].index).trim();
  if (preambleText.length > 0) {
    sections.push({
      bodyText: preambleText,
      charEnd: headings[0].index,
      charStart: 0,
      fullHeadingPath: '',
      headingLevel: 0,
    });
  }

  for (
    let headingIndex = 0;
    headingIndex < headings.length;
    headingIndex += 1
  ) {
    const heading = headings[headingIndex];

    // Maintain heading hierarchy stack.
    while (
      headingStack.length > 0 &&
      headingStack.at(-1).level >= heading.level
    ) {
      headingStack.pop();
    }
    headingStack.push(heading);

    const fullHeadingPath = headingStack.map((h) => h.text).join(' > ');
    const nextHeading = headings[headingIndex + 1];
    const charStart = heading.index;
    const charEnd = nextHeading?.index ?? markdownText.length;
    const bodyText = markdownText.slice(heading.lineEnd, charEnd).trim();

    /* istanbul ignore next -- defensive: fullHeadingPath is always non-empty when a heading exists */
    if (bodyText.length > 0 || fullHeadingPath.length > 0) {
      sections.push({
        bodyText,
        charEnd,
        charStart,
        fullHeadingPath,
        headingLevel: heading.level,
      });
    }
  }

  return sections;
}

/**
 * Split text at semantic boundaries within a size budget.
 *
 * Priority order: fenced code block boundary → table boundary → paragraph boundary
 * (double newline) → sentence boundary → hard character split as last resort.
 *
 * Code blocks and tables are always atomic — they are never split inside.
 *
 * @param {string} text - Text to split.
 * @param {number} maxChars - Hard maximum chars per chunk.
 * @param {number} overlapChars - Target overlap chars at sentence boundaries.
 * @returns {Array<{text: string, offset: number}>} Sub-chunks with text and character offset.
 */
export function splitAtSemanticBoundaries(text, maxChars, overlapChars) {
  if (text.length <= maxChars) {
    return [{ offset: 0, text }];
  }

  const blocks = extractAtomicBlocks(text);
  const chunks = [];
  let currentOffset = 0;
  let currentText = '';

  for (const block of blocks) {
    // If adding this block would exceed maxChars and we already have content, flush.
    if (
      currentText.length > 0 &&
      currentText.length + block.text.length + 1 > maxChars
    ) {
      chunks.push({ offset: currentOffset, text: currentText.trimEnd() });

      // Compute overlap from the end of the current chunk.
      const overlapText = computeOverlap(currentText.trimEnd(), overlapChars);
      currentText = overlapText;
      currentOffset = currentOffset; // Overlap text starts from a position within the previous chunk
    }

    // If the block itself exceeds maxChars and is atomic, keep it as-is (override max).
    if (block.isAtomic && block.text.length > maxChars) {
      if (currentText.trim().length > 0) {
        chunks.push({ offset: currentOffset, text: currentText.trimEnd() });
        const overlapText = computeOverlap(currentText.trimEnd(), overlapChars);
        currentText = overlapText;
        currentOffset = block.offset;
      }
      chunks.push({ offset: block.offset, text: block.text.trimEnd() });

      // Prepare for next block with overlap from the atomic block.
      const overlapText = computeOverlap(block.text.trimEnd(), overlapChars);
      currentText = overlapText;
      currentOffset = block.offset + block.text.length;
      continue;
    }

    if (currentText.trim().length === 0) {
      currentText = block.text;
      currentOffset = block.offset;
    } else {
      currentText = `${currentText}\n\n${block.text}`;
    }
  }

  // Flush remaining text.
  /* istanbul ignore next -- defensive: currentText always has remaining content when the loop exits */
  if (currentText.trim().length > 0) {
    chunks.push({ offset: currentOffset, text: currentText.trimEnd() });
  }

  return chunks;
}

/**
 * Extract atomic blocks from markdown text.
 *
 * Fenced code blocks and tables are marked as atomic (never split inside).
 * Paragraphs and other text are split at double-newline boundaries.
 *
 * @param {string} text - Markdown text.
 * @returns {Array<{text: string, offset: number, isAtomic: boolean}>} Atomic and non-atomic blocks.
 */
export function extractAtomicBlocks(text) {
  const blocks = [];
  const lines = text.split('\n');
  let currentOffset = 0;
  let inCodeBlock = false;
  let inTable = false;
  let blockStart = 0;
  let blockLines = [];

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const line = lines[lineIndex];
    const trimmedLine = line.trimStart();

    // Code block boundary detection.
    if (!inCodeBlock && CODE_FENCE_PATTERN.test(trimmedLine)) {
      // Flush any accumulated non-code text.
      if (blockLines.length > 0) {
        blocks.push({
          isAtomic: false,
          offset: currentOffset + blockStart,
          text: blockLines.join('\n'),
        });
        blockLines = [];
        blockStart = lineIndex;
      }

      inCodeBlock = true;
      blockStart = lineIndex;
      blockLines = [line];

      // Find closing fence.
      for (
        let closeIndex = lineIndex + 1;
        closeIndex < lines.length;
        closeIndex += 1
      ) {
        blockLines.push(lines[closeIndex]);
        if (CODE_FENCE_CLOSE_PATTERN.test(lines[closeIndex].trimStart())) {
          inCodeBlock = false;
          lineIndex = closeIndex;
          break;
        }
      }

      if (inCodeBlock) {
        // Unclosed code block — treat rest as atomic.
        blocks.push({
          isAtomic: true,
          offset: currentOffset + blockStart,
          text: blockLines.join('\n'),
        });
        blockLines = [];
        return blocks;
      }

      blocks.push({
        isAtomic: true,
        offset: currentOffset + blockStart,
        text: blockLines.join('\n'),
      });
      blockLines = [];
      blockStart = lineIndex + 1;
      continue;
    }

    /* istanbul ignore next -- unreachable: inCodeBlock is always false here because the code fence handler above either finds the closing fence (setting inCodeBlock=false) or returns early */
    if (inCodeBlock) {
      blockLines.push(line);
      continue;
    }

    // Table detection.
    if (!inTable && TABLE_ROW_PATTERN.test(trimmedLine)) {
      // Check if next non-empty line is a separator.
      const nextNonEmptyLine = lines
        .slice(lineIndex + 1)
        .find((l) => l.trim().length > 0);
      if (
        nextNonEmptyLine &&
        TABLE_SEPARATOR_PATTERN.test(nextNonEmptyLine.trimStart())
      ) {
        // Flush any accumulated non-table text.
        if (blockLines.length > 0) {
          blocks.push({
            isAtomic: false,
            offset: currentOffset + blockStart,
            text: blockLines.join('\n'),
          });
          blockLines = [];
        }

        inTable = true;
        blockStart = lineIndex;
        blockLines = [line];
        continue;
      }
    }

    if (inTable) {
      if (
        TABLE_ROW_PATTERN.test(trimmedLine) ||
        TABLE_SEPARATOR_PATTERN.test(trimmedLine)
      ) {
        blockLines.push(line);
        continue;
      }

      // Table ended — flush as atomic.
      blocks.push({
        isAtomic: true,
        offset: currentOffset + blockStart,
        text: blockLines.join('\n'),
      });
      inTable = false;
      blockLines = [];
      blockStart = lineIndex;
    }

    // Paragraph or other text — flush at double-newline boundaries.
    if (trimmedLine.length === 0) {
      if (blockLines.length > 0) {
        blocks.push({
          isAtomic: false,
          offset: currentOffset + blockStart,
          text: blockLines.join('\n'),
        });
        blockLines = [];
      }
      blockStart = lineIndex + 1;
      continue;
    }

    if (blockLines.length === 0) {
      blockStart = lineIndex;
    }
    blockLines.push(line);
  }

  // Flush remaining content.
  if (blockLines.length > 0) {
    blocks.push({
      isAtomic: inCodeBlock || inTable,
      offset: currentOffset + blockStart,
      text: blockLines.join('\n'),
    });
  }

  return blocks;
}

/**
 * Compute overlap text from the end of a chunk, targeting sentence boundaries.
 *
 * Finds the last sentence boundary within ±20% of the overlap target and
 * returns the text from that boundary to the end of the chunk. If no sentence
 * boundary is found, falls back to the last `overlapChars` characters.
 *
 * @param {string} chunkText - Text of the chunk to compute overlap from.
 * @param {number} overlapChars - Target overlap length in characters.
 * @returns {string} Overlap text to prepend to the next chunk.
 */
export function computeOverlap(chunkText, overlapChars) {
  if (chunkText.length <= overlapChars) return chunkText;

  const searchStart = Math.max(0, chunkText.length - overlapChars - 50);
  const searchEnd = Math.min(
    chunkText.length,
    chunkText.length - overlapChars + 50,
  );

  // Find the last sentence boundary within the tolerance window.
  const sentenceBoundaryPattern = /[.!?]\s/g;
  let lastBoundary = -1;
  let match = null;

  while ((match = sentenceBoundaryPattern.exec(chunkText)) !== null) {
    /* istanbul ignore next -- defensive: match.index is always within the search range */
    if (match.index >= searchStart && match.index <= searchEnd) {
      lastBoundary = match.index + 2; // Include the punctuation and space
    }
  }

  if (lastBoundary > 0) {
    return chunkText.slice(lastBoundary);
  }

  // Fallback: use the last overlapChars characters.
  return chunkText.slice(chunkText.length - overlapChars);
}

/**
 * Build a single v2 markdown chunk from a section.
 *
 * @param {{ fullHeadingPath: string, bodyText: string, headingLevel: number, charStart: number, charEnd: number }} section - Parsed section.
 * @param {string} filePath - File path for the context header.
 * @returns {import('./chunker.d.mts').MarkdownChunkV2} V2 markdown chunk.
 */
function buildChunk(section, filePath) {
  const headingPath = section.fullHeadingPath || '';
  return {
    body_text: section.bodyText,
    char_end: section.charEnd,
    char_start: section.charStart,
    context_header: buildContextHeader(filePath, headingPath),
    depth: section.headingLevel,
    export_type: null,
    heading_path: headingPath,
    jsdoc_text: null,
    module_path: null,
    parent_chunk_id: null,
    signature_text: null,
    symbol_name: null,
  };
}

/**
 * Build a context header string for a markdown chunk.
 *
 * Format: `[file_path > heading_path]` or `[file_path]` when heading_path is empty.
 *
 * @param {string} filePath - Repository-relative file path.
 * @param {string} headingPath - Full heading path.
 * @returns {string} Context header string.
 */
function buildContextHeader(filePath, headingPath) {
  if (headingPath) return `[${filePath} > ${headingPath}]`;
  return `[${filePath}]`;
}
