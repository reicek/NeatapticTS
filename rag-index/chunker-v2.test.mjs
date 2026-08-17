import { describe, it, expect, jest } from '@jest/globals';

import {
  chunkMarkdownV2,
  collectSectionsWithHierarchy,
  splitAtSemanticBoundaries,
  extractAtomicBlocks,
  computeOverlap,
} from './chunker-v2.mjs';

describe('chunkMarkdownV2', () => {
  it('chunks simple markdown with heading hierarchy', () => {
    const text = '# Title\n\nSome body text here.\n';
    const chunks = chunkMarkdownV2(text, { filePath: 'README.md' });
    expect(chunks).toHaveLength(1);
    expect(chunks[0].heading_path).toBe('Title');
    expect(chunks[0].context_header).toBe('[README.md > Title]');
    expect(chunks[0].depth).toBe(1);
    expect(chunks[0].body_text).toBe('Some body text here.');
  });

  it('handles preamble before first heading', () => {
    const text = 'Preamble text.\n\n# Heading\n\nBody.\n';
    const chunks = chunkMarkdownV2(text, { filePath: 'doc.md' });
    expect(chunks.length).toBe(2);
    expect(chunks[0].heading_path).toBe('');
    expect(chunks[0].depth).toBe(0);
    expect(chunks[0].context_header).toBe('[doc.md]');
    expect(chunks[1].heading_path).toBe('Heading');
  });

  it('adds (continued) suffix for split sections', () => {
    const body = 'A'.repeat(60) + '\n\n' + 'B'.repeat(60);
    const text = `# Big\n\n${body}`;
    const chunks = chunkMarkdownV2(text, { maxChars: 100, filePath: 'f.md' });
    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks[0].heading_path).toBe('Big');
    expect(chunks[1].heading_path).toContain('(continued)');
  });

  it('uses default filePath when not provided', () => {
    const text = '# H\n\nBody.\n';
    const chunks = chunkMarkdownV2(text);
    expect(chunks[0].context_header).toBe('[ > H]');
  });

  it('uses default maxChars and overlapChars', () => {
    const text = '# H\n\nSmall body.\n';
    const chunks = chunkMarkdownV2(text);
    expect(chunks).toHaveLength(1);
    expect(chunks[0].body_text).toBe('Small body.');
  });

  it('returns empty array for empty text', () => {
    const chunks = chunkMarkdownV2('');
    expect(chunks).toEqual([]);
  });

  it('returns empty array for whitespace-only text', () => {
    const chunks = chunkMarkdownV2('   \n\n  \n  ');
    expect(chunks).toEqual([]);
  });

  it('produces all v2 fields in chunk objects', () => {
    const text = '# H\n\nBody.\n';
    const chunks = chunkMarkdownV2(text, { filePath: 'test.md' });
    const chunk = chunks[0];
    expect(chunk).toHaveProperty('body_text');
    expect(chunk).toHaveProperty('char_start');
    expect(chunk).toHaveProperty('char_end');
    expect(chunk).toHaveProperty('context_header');
    expect(chunk).toHaveProperty('depth');
    expect(chunk).toHaveProperty('export_type');
    expect(chunk).toHaveProperty('heading_path');
    expect(chunk).toHaveProperty('jsdoc_text');
    expect(chunk).toHaveProperty('module_path');
    expect(chunk).toHaveProperty('parent_chunk_id');
    expect(chunk).toHaveProperty('signature_text');
    expect(chunk).toHaveProperty('symbol_name');
  });

  it('handles nested heading hierarchy', () => {
    const text = [
      '# Top',
      'Body top.',
      '## Sub',
      'Body sub.',
      '# Another',
      'Body another.',
    ].join('\n\n');
    const chunks = chunkMarkdownV2(text, { filePath: 'f.md' });
    expect(chunks.length).toBe(3);
    expect(chunks[0].heading_path).toBe('Top');
    expect(chunks[1].heading_path).toBe('Top > Sub');
    expect(chunks[2].heading_path).toBe('Another');
  });
});

describe('collectSectionsWithHierarchy', () => {
  it('returns empty array for empty text', () => {
    expect(collectSectionsWithHierarchy('')).toEqual([]);
  });

  it('returns empty array for whitespace-only text', () => {
    expect(collectSectionsWithHierarchy('   \n\n  ')).toEqual([]);
  });

  it('returns single preamble section for text with no headings', () => {
    const sections = collectSectionsWithHierarchy('Just some text.');
    expect(sections).toHaveLength(1);
    expect(sections[0].fullHeadingPath).toBe('');
    expect(sections[0].headingLevel).toBe(0);
    expect(sections[0].bodyText).toBe('Just some text.');
  });

  it('extracts preamble before first heading', () => {
    const text = 'Preamble.\n\n# Heading\n\nBody.\n';
    const sections = collectSectionsWithHierarchy(text);
    expect(sections.length).toBe(2);
    expect(sections[0].fullHeadingPath).toBe('');
    expect(sections[0].bodyText).toBe('Preamble.');
    expect(sections[1].fullHeadingPath).toBe('Heading');
  });

  it('skips empty preamble', () => {
    const text = '# Heading\n\nBody.\n';
    const sections = collectSectionsWithHierarchy(text);
    expect(sections).toHaveLength(1);
    expect(sections[0].fullHeadingPath).toBe('Heading');
  });

  it('builds heading hierarchy path', () => {
    const text = [
      '# A',
      'a body',
      '## B',
      'b body',
      '### C',
      'c body',
      '# D',
      'd body',
    ].join('\n\n');
    const sections = collectSectionsWithHierarchy(text);
    expect(sections).toHaveLength(4);
    expect(sections[0].fullHeadingPath).toBe('A');
    expect(sections[1].fullHeadingPath).toBe('A > B');
    expect(sections[2].fullHeadingPath).toBe('A > B > C');
    expect(sections[3].fullHeadingPath).toBe('D');
  });

  it('includes sections with heading path even if body is empty', () => {
    const text = '# H1\n\n# H2\n\nBody.\n';
    const sections = collectSectionsWithHierarchy(text);
    expect(sections.length).toBe(2);
    expect(sections[0].fullHeadingPath).toBe('H1');
    expect(sections[0].bodyText).toBe('');
    expect(sections[1].fullHeadingPath).toBe('H2');
  });

  it('handles heading at end of text with no newline', () => {
    const text = '# Heading at end';
    const sections = collectSectionsWithHierarchy(text);
    expect(sections).toHaveLength(1);
    expect(sections[0].fullHeadingPath).toBe('Heading at end');
  });
});

describe('splitAtSemanticBoundaries', () => {
  it('returns single chunk when text fits in maxChars', () => {
    const result = splitAtSemanticBoundaries('Short text.', 100, 10);
    expect(result).toHaveLength(1);
    expect(result[0].text).toBe('Short text.');
    expect(result[0].offset).toBe(0);
  });

  it('splits text with multiple paragraphs', () => {
    const text = 'A'.repeat(60) + '\n\n' + 'B'.repeat(60) + '\n\n' + 'C'.repeat(60);
    const result = splitAtSemanticBoundaries(text, 100, 20);
    expect(result.length).toBeGreaterThan(1);
  });

  it('handles atomic block exceeding maxChars', () => {
    const code = '```js\n' + 'X'.repeat(200) + '\n```';
    const text = 'Intro.\n\n' + code + '\n\nOutro.';
    const result = splitAtSemanticBoundaries(text, 100, 20);
    expect(result.length).toBeGreaterThan(1);
    // The atomic code block should be one chunk
    const codeChunk = result.find((r) => r.text.includes('```'));
    expect(codeChunk).toBeDefined();
  });

  it('handles atomic block exceeding maxChars with existing content', () => {
    const code = '```js\n' + 'X'.repeat(200) + '\n```';
    const text = 'Some intro text here.\n\n' + code + '\n\nAfter.';
    const result = splitAtSemanticBoundaries(text, 80, 20);
    expect(result.length).toBeGreaterThan(1);
  });

  it('flushes remaining text at end', () => {
    const text = 'A'.repeat(40) + '\n\n' + 'B'.repeat(40);
    const result = splitAtSemanticBoundaries(text, 100, 20);
    expect(result.length).toBe(1);
  });

  it('handles text starting with atomic block', () => {
    const code = '```js\n' + 'X'.repeat(200) + '\n```';
    const text = code + '\n\nAfter text.';
    const result = splitAtSemanticBoundaries(text, 100, 20);
    expect(result.length).toBeGreaterThan(1);
  });
});

describe('extractAtomicBlocks', () => {
  it('extracts simple paragraphs', () => {
    const text = 'Paragraph 1.\n\nParagraph 2.';
    const blocks = extractAtomicBlocks(text);
    expect(blocks).toHaveLength(2);
    expect(blocks[0].isAtomic).toBe(false);
    expect(blocks[0].text).toBe('Paragraph 1.');
    expect(blocks[1].text).toBe('Paragraph 2.');
  });

  it('extracts fenced code blocks as atomic', () => {
    const text = 'Before.\n\n```js\nconst x = 1;\n```\n\nAfter.';
    const blocks = extractAtomicBlocks(text);
    expect(blocks.length).toBe(3);
    const codeBlock = blocks.find((b) => b.isAtomic);
    expect(codeBlock).toBeDefined();
    expect(codeBlock.text).toContain('```');
  });

  it('handles unclosed code block as atomic', () => {
    const text = 'Before.\n\n```\nunclosed code';
    const blocks = extractAtomicBlocks(text);
    // Unclosed code block → returns early with atomic block
    const codeBlock = blocks.find((b) => b.isAtomic && b.text.includes('unclosed'));
    expect(codeBlock).toBeDefined();
  });

  it('extracts tables as atomic', () => {
    const text = 'Before.\n\n| Col1 | Col2 |\n| --- | --- |\n| A | B |\n\nAfter.';
    const blocks = extractAtomicBlocks(text);
    const tableBlock = blocks.find((b) => b.isAtomic && b.text.includes('|'));
    expect(tableBlock).toBeDefined();
    expect(tableBlock.text).toContain('| Col1 | Col2 |');
  });

  it('flushes non-code text before code block', () => {
    const text = 'Some text line.\n```js\ncode\n```';
    const blocks = extractAtomicBlocks(text);
    expect(blocks.length).toBe(2);
    expect(blocks[0].isAtomic).toBe(false);
    expect(blocks[0].text).toBe('Some text line.');
    expect(blocks[1].isAtomic).toBe(true);
  });

  it('flushes non-table text before table', () => {
    const text = 'Some text.\n| Col1 | Col2 |\n| --- | --- |\n| A | B |\nAfter.';
    const blocks = extractAtomicBlocks(text);
    expect(blocks.length).toBe(3);
    expect(blocks[0].isAtomic).toBe(false);
    expect(blocks[0].text).toBe('Some text.');
    expect(blocks[1].isAtomic).toBe(true);
    expect(blocks[2].isAtomic).toBe(false);
  });

  it('handles table at end of text', () => {
    const text = '| A | B |\n| --- | --- |\n| 1 | 2 |';
    const blocks = extractAtomicBlocks(text);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].isAtomic).toBe(true);
  });

  it('handles non-table row that looks like table but no separator', () => {
    const text = '| not a table |\nJust text.';
    const blocks = extractAtomicBlocks(text);
    // Should be treated as normal text since no separator follows
    expect(blocks.every((b) => !b.isAtomic)).toBe(true);
  });

  it('handles empty lines between content', () => {
    const text = 'A.\n\n\n\nB.';
    const blocks = extractAtomicBlocks(text);
    expect(blocks).toHaveLength(2);
    expect(blocks[0].text).toBe('A.');
    expect(blocks[1].text).toBe('B.');
  });

  it('handles code block at start with no preceding text', () => {
    const text = '```\ncode\n```\nAfter.';
    const blocks = extractAtomicBlocks(text);
    expect(blocks.length).toBe(2);
    expect(blocks[0].isAtomic).toBe(true);
    expect(blocks[1].isAtomic).toBe(false);
  });

  it('flushes remaining content at end of text', () => {
    const text = 'Just one paragraph with no trailing newline.';
    const blocks = extractAtomicBlocks(text);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].text).toBe('Just one paragraph with no trailing newline.');
  });

  it('handles code block followed immediately by another code block', () => {
    const text = '```\ncode1\n```\n```\ncode2\n```';
    const blocks = extractAtomicBlocks(text);
    expect(blocks.length).toBe(2);
    expect(blocks[0].isAtomic).toBe(true);
    expect(blocks[1].isAtomic).toBe(true);
  });

  it('handles table followed by text on same line', () => {
    const text = '| A | B |\n| --- | --- |\n| 1 | 2 |\nNot a table row.';
    const blocks = extractAtomicBlocks(text);
    const tableBlock = blocks.find((b) => b.isAtomic);
    expect(tableBlock).toBeDefined();
    const textBlock = blocks.find((b) => !b.isAtomic);
    expect(textBlock).toBeDefined();
  });

  it('handles multiple consecutive empty lines', () => {
    const text = 'A.\n\n\nB.\n\n\nC.';
    const blocks = extractAtomicBlocks(text);
    expect(blocks).toHaveLength(3);
  });
});

describe('computeOverlap', () => {
  it('returns full text when shorter than overlapChars', () => {
    expect(computeOverlap('Short.', 100)).toBe('Short.');
  });

  it('returns full text when equal to overlapChars', () => {
    expect(computeOverlap('12345', 5)).toBe('12345');
  });

  it('finds sentence boundary within tolerance window', () => {
    const text = 'First sentence. Second sentence here.';
    const result = computeOverlap(text, 15);
    // Should find the sentence boundary and return from there
    expect(result.length).toBeLessThanOrEqual(text.length);
    expect(result.length).toBeGreaterThan(0);
  });

  it('falls back to last overlapChars when no sentence boundary found', () => {
    const text = 'A'.repeat(200);
    const result = computeOverlap(text, 50);
    expect(result.length).toBe(50);
    expect(result).toBe('A'.repeat(50));
  });

  it('handles text with multiple sentence boundaries', () => {
    const text = 'Sentence one! Sentence two? Sentence three. End.';
    const result = computeOverlap(text, 20);
    expect(result.length).toBeGreaterThan(0);
  });
});