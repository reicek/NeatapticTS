import { describe, it, expect } from '@jest/globals';

import { chunkMarkdown } from './chunker.mjs';

describe('chunkMarkdown', () => {
  it('returns a single chunk for text with no headings', () => {
    const text = 'This is some plain text without headings.';
    const chunks = chunkMarkdown(text);
    expect(chunks).toHaveLength(1);
    expect(chunks[0].heading_path).toBe('');
    expect(chunks[0].body_text).toBe(text);
    expect(chunks[0].char_start).toBe(0);
    expect(chunks[0].char_end).toBe(text.length);
  });

  it('returns zero chunks for empty string', () => {
    const chunks = chunkMarkdown('');
    expect(chunks).toHaveLength(0);
  });

  it('handles heading at end of text with no trailing newline', () => {
    const text = '# Heading at end';
    const chunks = chunkMarkdown(text);
    expect(chunks).toHaveLength(1);
    expect(chunks[0].heading_path).toBe('# Heading at end');
  });

  it('builds heading hierarchy with multiple levels', () => {
    const text = [
      '# Top',
      'Body for top.',
      '## Sub',
      'Body for sub.',
      '### Deep',
      'Body for deep.',
      '## Another Sub',
      'Body for another.',
    ].join('\n');
    const chunks = chunkMarkdown(text);
    expect(chunks.length).toBe(4);
    expect(chunks[0].heading_path).toBe('# Top');
    expect(chunks[1].heading_path).toBe('# Top > ## Sub');
    expect(chunks[2].heading_path).toBe('# Top > ## Sub > ### Deep');
    expect(chunks[3].heading_path).toBe('# Top > ## Another Sub');
  });

  it('heading stack pops correctly when going back to lower level', () => {
    const text = [
      '# A',
      'body a',
      '## B',
      'body b',
      '# C',
      'body c',
    ].join('\n');
    const chunks = chunkMarkdown(text);
    expect(chunks).toHaveLength(3);
    expect(chunks[0].heading_path).toBe('# A');
    expect(chunks[1].heading_path).toBe('# A > ## B');
    expect(chunks[2].heading_path).toBe('# C');
  });

  it('splits a section larger than maxChars into overlapping chunks', () => {
    const body = 'A'.repeat(100);
    const text = `# Heading\n${body}`;
    const chunks = chunkMarkdown(text, { maxChars: 30, overlapChars: 10 });
    expect(chunks.length).toBeGreaterThan(1);
    // Verify overlap: second chunk should start before the end of first
    const firstChunkEnd = chunks[0].char_end;
    const secondChunkStart = chunks[1].char_start;
    expect(secondChunkStart).toBeLessThan(firstChunkEnd);
  });

  it('uses heading_path as fallback when body_text is empty', () => {
    const text = '# Heading Only\n## Next';
    const chunks = chunkMarkdown(text);
    // First heading has no body, but heading_path is non-empty so it's kept
    const headingOnly = chunks.find((c) => c.heading_path === '# Heading Only');
    expect(headingOnly).toBeDefined();
    expect(headingOnly.body_text).toBe('# Heading Only');
  });

  it('respects custom maxChars and overlapChars', () => {
    const body = 'X'.repeat(50);
    const text = `# H\n${body}`;
    const chunks = chunkMarkdown(text, { maxChars: 20, overlapChars: 5 });
    expect(chunks.length).toBeGreaterThan(1);
    for (const chunk of chunks) {
      expect(chunk.body_text.length).toBeLessThanOrEqual(20);
    }
  });

  it('clamps overlapChars to maxChars - 1', () => {
    const body = 'Y'.repeat(100);
    const text = `# H\n${body}`;
    const chunks = chunkMarkdown(text, { maxChars: 10, overlapChars: 100 });
    // overlapChars should be clamped to 9
    expect(chunks.length).toBeGreaterThan(1);
    // Each chunk body should be <= 10
    for (const chunk of chunks) {
      expect(chunk.body_text.length).toBeLessThanOrEqual(10);
    }
  });

  it('uses default options when options is empty', () => {
    const text = '# H\nSome text here.';
    const chunks = chunkMarkdown(text, {});
    expect(chunks).toHaveLength(1);
    expect(chunks[0].body_text).toBe('Some text here.');
  });

  it('uses maxChars=1 when maxChars is 0', () => {
    const text = 'AB';
    const chunks = chunkMarkdown(text, { maxChars: 0 });
    // Math.max(1, 0) = 1, so each chunk is 1 char
    expect(chunks.length).toBe(2);
    expect(chunks[0].body_text).toBe('A');
    expect(chunks[1].body_text).toBe('B');
  });

  it('uses overlapChars=0 when overlapChars is negative', () => {
    const body = 'Z'.repeat(100);
    const text = `# H\n${body}`;
    const chunks = chunkMarkdown(text, { maxChars: 20, overlapChars: -5 });
    // Math.max(0, Math.min(-5, 19)) = 0
    expect(chunks.length).toBeGreaterThan(1);
    // With 0 overlap, chunks should not overlap
    for (let i = 1; i < chunks.length; i++) {
      expect(chunks[i].char_start).toBeGreaterThanOrEqual(chunks[i - 1].char_end);
    }
  });

  it('handles multiple headings at same level', () => {
    const text = [
      '# First',
      'body1',
      '# Second',
      'body2',
    ].join('\n');
    const chunks = chunkMarkdown(text);
    expect(chunks).toHaveLength(2);
    expect(chunks[0].heading_path).toBe('# First');
    expect(chunks[1].heading_path).toBe('# Second');
  });

  it('handles non-string input by coercing to string', () => {
    const chunks = chunkMarkdown(12345);
    expect(chunks).toHaveLength(1);
    expect(chunks[0].body_text).toBe('12345');
  });

  it('produces correct char positions for multi-chunk sections', () => {
    const body = 'A'.repeat(100);
    const text = `# H\n${body}`;
    const chunks = chunkMarkdown(text, { maxChars: 30, overlapChars: 10 });
    // First chunk should start at heading position
    expect(chunks[0].char_start).toBe(0);
    // All char positions should be within text bounds
    for (const chunk of chunks) {
      expect(chunk.char_start).toBeGreaterThanOrEqual(0);
      expect(chunk.char_end).toBeLessThanOrEqual(text.length);
    }
  });
});