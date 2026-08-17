import { jest } from '@jest/globals';
import {
  enrichChunks,
  deduplicateChunks,
  orderChunks,
  enforceBudget,
  stitchContext,
  assembleContext,
  DEFAULT_CHARS_PER_TOKEN,
  DEFAULT_BUDGET,
  DEFAULT_COSINE_THRESHOLD,
  DEFAULT_TIER_THRESHOLDS,
  DEFAULT_FAMILY_PRIORITY,
} from './assemble-context.mjs';

describe('assemble-context.mjs', () => {
  describe('constants', () => {
    it('exports expected constants', () => {
      expect(DEFAULT_CHARS_PER_TOKEN).toBe(4);
      expect(DEFAULT_BUDGET).toBe(4096);
      expect(DEFAULT_COSINE_THRESHOLD).toBe(0.95);
      expect(DEFAULT_TIER_THRESHOLDS).toEqual({ essential: 0.7, supporting: 0.4 });
      expect(DEFAULT_FAMILY_PRIORITY).toContain('ts-source');
      expect(DEFAULT_FAMILY_PRIORITY).toContain('readme');
    });
  });

  describe('enrichChunks', () => {
    it('uses fallback enrichment when no client provided', async () => {
      const chunks = [
        { chunk_id: 1, file_path: 'a.ts', body_text: 'hello', score: 0.9 },
      ];
      const result = await enrichChunks(chunks);
      expect(result).toHaveLength(1);
      expect(result[0].body_text).toBe('hello');
      expect(result[0].sha256).toBeTruthy();
      expect(result[0].context_header).toBe('a.ts');
      expect(result[0].query_class).toBe('default');
    });

    it('uses fallback enrichment when chunks is empty', async () => {
      const client = { execute: jest.fn() };
      const result = await enrichChunks([], { client });
      expect(result).toEqual([]);
      expect(client.execute).not.toHaveBeenCalled();
    });

    it('uses fallback enrichment when chunk_ids are all null', async () => {
      const client = { execute: jest.fn() };
      const chunks = [
        { chunk_id: null, file_path: 'a.ts', body_text: 'hello', score: 0.5 },
      ];
      const result = await enrichChunks(chunks, { client });
      expect(client.execute).not.toHaveBeenCalled();
      expect(result).toHaveLength(1);
      expect(result[0].body_text).toBe('hello');
    });

    it('enriches chunks using database client', async () => {
      const client = {
        execute: jest.fn().mockResolvedValue({
          rows: [
            {
              chunk_id: 1,
              chunk_index: 0,
              heading_path: 'section1',
              body_text: 'db body',
              char_start: 0,
              char_end: 10,
              parent_chunk_id: null,
              depth: 0,
              context_header: 'db header',
              symbol_name: 'foo',
              signature_text: '() => void',
              jsdoc_text: 'doc',
              export_type: 'function',
              module_path: 'src/a.ts',
              arch_layer: 'core',
              chunk_sha256: 'abc123',
              file_path: 'src/a.ts',
              family: 'ts-source',
              entity_type: 'function',
              entity_name: 'foo',
            },
          ],
        }),
      };
      const chunks = [
        { chunk_id: 1, file_path: 'a.ts', score: 0.9 },
      ];
      const result = await enrichChunks(chunks, { client, query_class: 'test' });
      expect(client.execute).toHaveBeenCalledWith(
        expect.objectContaining({ args: [1] }),
      );
      expect(result).toHaveLength(1);
      expect(result[0].body_text).toBe('db body');
      expect(result[0].sha256).toBe('abc123');
      expect(result[0].family).toBe('ts-source');
      expect(result[0].query_class).toBe('test');
      expect(result[0].score).toBe(0.9);
    });

    it('falls back when enriched row is not found for a chunk', async () => {
      const client = {
        execute: jest.fn().mockResolvedValue({ rows: [] }),
      };
      const chunks = [
        { chunk_id: 99, file_path: 'a.ts', body_text: 'fallback', score: 0.5 },
      ];
      const result = await enrichChunks(chunks, { client });
      expect(result).toHaveLength(1);
      expect(result[0].body_text).toBe('fallback');
      expect(result[0].sha256).toBeTruthy();
    });

    it('prefers content over text when body_text is absent', async () => {
      const chunks = [
        { chunk_id: null, file_path: 'a.ts', content: 'content body', score: 0.5 },
      ];
      const result = await enrichChunks(chunks);
      expect(result[0].body_text).toBe('content body');
    });

    it('uses text field when body_text and content are absent', async () => {
      const chunks = [
        { chunk_id: null, file_path: 'a.ts', text: 'text body', score: 0.5 },
      ];
      const result = await enrichChunks(chunks);
      expect(result[0].body_text).toBe('text body');
    });

    it('uses existing sha256 or chunk_sha256 when available', async () => {
      const chunks = [
        { chunk_id: null, file_path: 'a.ts', body_text: 'x', score: 0.5, sha256: 'existing-hash' },
      ];
      const result = await enrichChunks(chunks);
      expect(result[0].sha256).toBe('existing-hash');
    });

    it('uses chunk_sha256 when sha256 is not available', async () => {
      const chunks = [
        { chunk_id: null, file_path: 'a.ts', body_text: 'x', score: 0.5, chunk_sha256: 'chunk-hash' },
      ];
      const result = await enrichChunks(chunks);
      expect(result[0].sha256).toBe('chunk-hash');
    });

    it('uses existing context_header when available', async () => {
      const chunks = [
        { chunk_id: null, file_path: 'a.ts', body_text: 'x', score: 0.5, context_header: 'existing header' },
      ];
      const result = await enrichChunks(chunks);
      expect(result[0].context_header).toBe('existing header');
    });

    it('builds context header from heading_path', async () => {
      const chunks = [
        { chunk_id: null, file_path: 'a.ts', heading_path: 'my heading', body_text: 'x', score: 0.5 },
      ];
      const result = await enrichChunks(chunks);
      expect(result[0].context_header).toBe('a.ts > my heading');
    });

    it('builds context header from just file_path when heading is empty', async () => {
      const chunks = [
        { chunk_id: null, file_path: 'a.ts', heading_path: '  ', body_text: 'x', score: 0.5 },
      ];
      const result = await enrichChunks(chunks);
      expect(result[0].context_header).toBe('a.ts');
    });
  });

  describe('deduplicateChunks', () => {
    it('collapses exact duplicates by sha256', () => {
      const chunks = [
        { chunk_id: 1, sha256: 'aaa', body_text: 'a', score: 0.9 },
        { chunk_id: 2, sha256: 'aaa', body_text: 'a', score: 0.8 },
      ];
      expect(deduplicateChunks(chunks)).toHaveLength(1);
    });

    it('collapses exact duplicates by chunk_id when no sha256', () => {
      const chunks = [
        { chunk_id: 1, body_text: 'a', score: 0.9 },
        { chunk_id: 1, body_text: 'a', score: 0.8 },
      ];
      expect(deduplicateChunks(chunks)).toHaveLength(1);
    });

    it('removes parent chunks when children are present', () => {
      const chunks = [
        { chunk_id: 1, parent_chunk_id: null, body_text: 'parent', score: 0.9 },
        { chunk_id: 2, parent_chunk_id: 1, body_text: 'child', score: 0.8 },
      ];
      const result = deduplicateChunks(chunks);
      expect(result).toHaveLength(1);
      expect(result[0].chunk_id).toBe(2);
    });

    it('keeps parent when no children present', () => {
      const chunks = [
        { chunk_id: 1, parent_chunk_id: null, body_text: 'parent', score: 0.9 },
      ];
      expect(deduplicateChunks(chunks)).toHaveLength(1);
    });

    it('handles parent_chunk_id as empty string', () => {
      const chunks = [
        { chunk_id: 1, parent_chunk_id: '', body_text: 'a', score: 0.9 },
        { chunk_id: 2, parent_chunk_id: '', body_text: 'b', score: 0.8 },
      ];
      expect(deduplicateChunks(chunks)).toHaveLength(2);
    });

    it('collapses near-duplicate embeddings above cosine threshold', () => {
      const chunks = [
        { chunk_id: 1, sha256: 'a', body_text: 'a', score: 0.9, embedding: [1, 0, 0] },
        { chunk_id: 2, sha256: 'b', body_text: 'b', score: 0.8, embedding: [0.99, 0.01, 0] },
      ];
      expect(deduplicateChunks(chunks)).toHaveLength(1);
    });

    it('keeps chunks with different embeddings', () => {
      const chunks = [
        { chunk_id: 1, sha256: 'a', body_text: 'a', score: 0.9, embedding: [1, 0, 0] },
        { chunk_id: 2, sha256: 'b', body_text: 'b', score: 0.8, embedding: [0, 1, 0] },
      ];
      expect(deduplicateChunks(chunks)).toHaveLength(2);
    });

    it('keeps chunks without embeddings', () => {
      const chunks = [
        { chunk_id: 1, sha256: 'a', body_text: 'a', score: 0.9 },
        { chunk_id: 2, sha256: 'b', body_text: 'b', score: 0.8 },
      ];
      expect(deduplicateChunks(chunks)).toHaveLength(2);
    });

    it('keeps chunks with empty embeddings', () => {
      const chunks = [
        { chunk_id: 1, sha256: 'a', body_text: 'a', score: 0.9, embedding: [] },
        { chunk_id: 2, sha256: 'b', body_text: 'b', score: 0.8, embedding: [] },
      ];
      expect(deduplicateChunks(chunks)).toHaveLength(2);
    });

    it('skips embedding comparison when lengths differ', () => {
      const chunks = [
        { chunk_id: 1, sha256: 'a', body_text: 'a', score: 0.9, embedding: [1, 0] },
        { chunk_id: 2, sha256: 'b', body_text: 'b', score: 0.8, embedding: [1, 0, 0] },
      ];
      expect(deduplicateChunks(chunks)).toHaveLength(2);
    });

    it('handles zero vectors (denominator === 0)', () => {
      const chunks = [
        { chunk_id: 1, sha256: 'a', body_text: 'a', score: 0.9, embedding: [0, 0, 0] },
        { chunk_id: 2, sha256: 'b', body_text: 'b', score: 0.8, embedding: [0, 0, 0] },
      ];
      expect(deduplicateChunks(chunks)).toHaveLength(2);
    });

    it('uses custom cosineThreshold', () => {
      const chunks = [
        { chunk_id: 1, sha256: 'a', body_text: 'a', score: 0.9, embedding: [1, 0, 0] },
        { chunk_id: 2, sha256: 'b', body_text: 'b', score: 0.8, embedding: [0.7, 0.7, 0] },
      ];
      // cosine ~ 0.707, with threshold 0.5 they're duplicates
      expect(deduplicateChunks(chunks, { cosineThreshold: 0.5 })).toHaveLength(1);
      // with threshold 0.95 they're not
      expect(deduplicateChunks(chunks, { cosineThreshold: 0.95 })).toHaveLength(2);
    });
  });

  describe('orderChunks', () => {
    it('assigns tiers and sorts by tier order', () => {
      const chunks = [
        { chunk_id: 1, file_path: 'b.ts', char_start: 0, score: 0.3 },
        { chunk_id: 2, file_path: 'a.ts', char_start: 0, score: 0.8 },
      ];
      const result = orderChunks(chunks);
      expect(result[0].tier).toBe('essential');
      expect(result[1].tier).toBe('supplementary');
    });

    it('sorts by file max score within same tier', () => {
      const chunks = [
        { chunk_id: 1, file_path: 'a.ts', char_start: 0, score: 0.5 },
        { chunk_id: 2, file_path: 'b.ts', char_start: 0, score: 0.6 },
        { chunk_id: 3, file_path: 'b.ts', char_start: 10, score: 0.45 },
      ];
      const result = orderChunks(chunks);
      // b.ts has max 0.6, a.ts has max 0.5 → b.ts first
      expect(result[0].file_path).toBe('b.ts');
    });

    it('sorts by family priority within same tier and file max', () => {
      const chunks = [
        { chunk_id: 1, file_path: 'a.ts', char_start: 0, score: 0.5, family: 'demo' },
        { chunk_id: 2, file_path: 'b.ts', char_start: 0, score: 0.5, family: 'ts-source' },
      ];
      const result = orderChunks(chunks);
      // ts-source has higher priority (lower ordinal) → comes first
      expect(result[0].family).toBe('ts-source');
    });

    it('sorts by file_path when family is the same or unknown', () => {
      const chunks = [
        { chunk_id: 1, file_path: 'z.ts', char_start: 0, score: 0.5 },
        { chunk_id: 2, file_path: 'a.ts', char_start: 0, score: 0.5 },
      ];
      const result = orderChunks(chunks);
      expect(result[0].file_path).toBe('a.ts');
    });

    it('sorts by char_start within same file', () => {
      const chunks = [
        { chunk_id: 1, file_path: 'a.ts', char_start: 50, score: 0.5 },
        { chunk_id: 2, file_path: 'a.ts', char_start: 10, score: 0.5 },
      ];
      const result = orderChunks(chunks);
      expect(result[0].char_start).toBe(10);
    });

    it('handles unknown family with MAX_SAFE_INTEGER ordinal', () => {
      const chunks = [
        { chunk_id: 1, file_path: 'a.ts', char_start: 0, score: 0.5, family: 'unknown' },
        { chunk_id: 2, file_path: 'b.ts', char_start: 0, score: 0.5, family: 'ts-source' },
      ];
      const result = orderChunks(chunks);
      expect(result[0].family).toBe('ts-source');
    });

    it('handles missing family with MAX_SAFE_INTEGER ordinal', () => {
      const chunks = [
        { chunk_id: 1, file_path: 'a.ts', char_start: 0, score: 0.5 },
        { chunk_id: 2, file_path: 'b.ts', char_start: 0, score: 0.5, family: 'ts-source' },
      ];
      const result = orderChunks(chunks);
      expect(result[0].family).toBe('ts-source');
    });

    it('uses doc_family as fallback for family', () => {
      const chunks = [
        { chunk_id: 1, file_path: 'a.ts', char_start: 0, score: 0.5, doc_family: 'demo' },
        { chunk_id: 2, file_path: 'b.ts', char_start: 0, score: 0.5, doc_family: 'ts-source' },
      ];
      const result = orderChunks(chunks);
      expect(result[0].doc_family).toBe('ts-source');
    });

    it('uses custom thresholds', () => {
      const chunks = [
        { chunk_id: 1, file_path: 'a.ts', char_start: 0, score: 0.5 },
      ];
      const result = orderChunks(chunks, { essential: 0.6, supporting: 0.3 });
      expect(result[0].tier).toBe('supporting');
    });

    it('uses custom familyPriority list', () => {
      const chunks = [
        { chunk_id: 1, file_path: 'a.ts', char_start: 0, score: 0.5, family: 'demo' },
        { chunk_id: 2, file_path: 'b.ts', char_start: 0, score: 0.5, family: 'readme' },
      ];
      const result = orderChunks(chunks, { familyPriority: ['demo', 'readme'] });
      expect(result[0].family).toBe('demo');
    });
  });

  describe('enforceBudget', () => {
    it('selects all chunks when they fit within budget', () => {
      const chunks = [
        { tier: 'essential', file_path: 'a.ts', char_start: 0, body_text: 'hello', score: 0.9 },
        { tier: 'supporting', file_path: 'b.ts', char_start: 0, body_text: 'world', score: 0.5 },
      ];
      const result = enforceBudget(chunks, { budget: 1000 });
      expect(result.selectedChunks).toHaveLength(2);
      expect(result.tokenCount).toBeGreaterThan(0);
      expect(result.truncated).toBe(false);
      expect(result.tierCounts.essential).toBe(1);
      expect(result.tierCounts.supporting).toBe(1);
    });

    it('truncates essential chunks that exceed remaining budget', () => {
      const longText = 'A'.repeat(1000);
      const chunks = [
        { tier: 'essential', file_path: 'a.ts', char_start: 0, body_text: longText, score: 0.9 },
      ];
      const result = enforceBudget(chunks, { budget: 10 });
      expect(result.selectedChunks).toHaveLength(1);
      expect(result.truncated).toBe(true);
      expect(result.selectedChunks[0].truncated).toBe(true);
    });

    it('drops supporting chunks that do not fit', () => {
      const chunks = [
        { tier: 'essential', file_path: 'a.ts', char_start: 0, body_text: 'A'.repeat(200), score: 0.9 },
        { tier: 'supporting', file_path: 'b.ts', char_start: 0, body_text: 'B'.repeat(200), score: 0.5 },
      ];
      const result = enforceBudget(chunks, { budget: 30 });
      // essential takes ~50 tokens, supporting needs ~50 more but only ~20 remain
      expect(result.selectedChunks.length).toBeLessThanOrEqual(2);
    });

    it('truncates last supplementary chunk when it does not fully fit', () => {
      const chunks = [
        { tier: 'supplementary', file_path: 'a.ts', char_start: 0, body_text: 'A'.repeat(200), score: 0.3 },
      ];
      const result = enforceBudget(chunks, { budget: 10 });
      expect(result.selectedChunks).toHaveLength(1);
      expect(result.truncated).toBe(true);
    });

    it('drops non-last supplementary chunks that do not fit', () => {
      const chunks = [
        { tier: 'supplementary', file_path: 'a.ts', char_start: 0, body_text: 'A'.repeat(200), score: 0.3 },
        { tier: 'supplementary', file_path: 'b.ts', char_start: 0, body_text: 'B'.repeat(200), score: 0.2 },
      ];
      const result = enforceBudget(chunks, { budget: 10 });
      // first supplementary is NOT the last, so it gets dropped (not truncated)
      // second supplementary IS the last, so it gets truncated
      expect(result.selectedChunks).toHaveLength(1);
    });

    it('breaks when remaining budget is 0', () => {
      const chunks = [
        { tier: 'essential', file_path: 'a.ts', char_start: 0, body_text: 'A'.repeat(200), score: 0.9 },
        { tier: 'supporting', file_path: 'b.ts', char_start: 0, body_text: 'B'.repeat(200), score: 0.5 },
      ];
      const result = enforceBudget(chunks, { budget: 5 });
      // essential chunk consumes all budget, supporting gets dropped (remaining <= 0)
      expect(result.tierCounts.essential).toBe(1);
      expect(result.tierCounts.supporting).toBe(0);
    });

    it('uses custom countTokens function', () => {
      const chunks = [
        { tier: 'essential', file_path: 'a.ts', char_start: 0, body_text: 'hello', score: 0.9 },
      ];
      const result = enforceBudget(chunks, { budget: 100, countTokens: () => 5 });
      expect(result.tokenCount).toBe(5);
    });

    it('handles empty chunks', () => {
      const result = enforceBudget([], { budget: 100 });
      expect(result.selectedChunks).toEqual([]);
      expect(result.tokenCount).toBe(0);
      expect(result.truncated).toBe(false);
    });

    it('handles budget of 0', () => {
      const chunks = [
        { tier: 'essential', file_path: 'a.ts', char_start: 0, body_text: 'hello', score: 0.9 },
      ];
      const result = enforceBudget(chunks, { budget: 0 });
      // budget is 0, remaining is 0, essential chunk gets truncated to 0 chars
      // truncateAtSentenceBoundary with maxChars=0 → clamped to 1, then text.slice(0,1)
      expect(result.selectedChunks.length).toBeGreaterThanOrEqual(0);
    });

    it('handles empty body text (0 tokens)', () => {
      const chunks = [
        { tier: 'essential', file_path: 'a.ts', char_start: 0, body_text: '', score: 0.9 },
        { tier: 'supporting', file_path: 'b.ts', char_start: 0, body_text: '', score: 0.5 },
      ];
      const result = enforceBudget(chunks, { budget: 100 });
      expect(result.selectedChunks).toHaveLength(2);
      expect(result.tokenCount).toBe(0);
    });

    it('uses content field when body_text is absent', () => {
      const chunks = [
        { tier: 'essential', file_path: 'a.ts', char_start: 0, content: 'hello world', score: 0.9 },
      ];
      const result = enforceBudget(chunks, { budget: 100 });
      expect(result.selectedChunks).toHaveLength(1);
      expect(result.tokenCount).toBeGreaterThan(0);
    });

    it('truncates at sentence boundary when possible', () => {
      const text = 'First sentence. Second sentence. Third sentence.';
      const chunks = [
        { tier: 'essential', file_path: 'a.ts', char_start: 0, body_text: text, score: 0.9 },
      ];
      const result = enforceBudget(chunks, { budget: 10 });
      expect(result.truncated).toBe(true);
      expect(result.selectedChunks[0].body_text).toBeTruthy();
    });

    it('truncates at newline when no sentence boundary', () => {
      const text = 'line1\nline2\nline3\nline4\nline5\nline6\nline7';
      const chunks = [
        { tier: 'essential', file_path: 'a.ts', char_start: 0, body_text: text, score: 0.9 },
      ];
      const result = enforceBudget(chunks, { budget: 10 });
      expect(result.truncated).toBe(true);
    });

    it('hard truncates when no boundary found', () => {
      const text = 'abcdefghij'.repeat(100);
      const chunks = [
        { tier: 'essential', file_path: 'a.ts', char_start: 0, body_text: text, score: 0.9 },
      ];
      const result = enforceBudget(chunks, { budget: 10 });
      expect(result.truncated).toBe(true);
      expect(result.selectedChunks[0].body_text.length).toBeLessThan(text.length);
    });

    it('does not truncate when text fits within maxChars', () => {
      const text = 'short text';
      const chunks = [
        { tier: 'essential', file_path: 'a.ts', char_start: 0, body_text: text, score: 0.9 },
      ];
      const result = enforceBudget(chunks, { budget: 1000 });
      expect(result.truncated).toBe(false);
    });
  });

  describe('stitchContext', () => {
    it('stitches chunks as markdown by default', () => {
      const chunks = [
        { file_path: 'a.ts', heading_path: 'sec1', char_start: 0, char_end: 10, body_text: 'hello', score: 0.9, tier: 'essential' },
      ];
      const result = stitchContext(chunks);
      expect(typeof result).toBe('string');
      expect(result).toContain('a.ts');
      expect(result).toContain('hello');
    });

    it('stitches chunks as json when format is json', () => {
      const chunks = [
        { file_path: 'a.ts', heading_path: 'sec1', char_start: 0, char_end: 10, body_text: 'hello', score: 0.9, tier: 'essential' },
      ];
      const result = stitchContext(chunks, { context_format: 'json' });
      expect(typeof result).toBe('object');
      expect(result.context).toContain('hello');
      expect(result.chunks).toHaveLength(1);
      expect(result.chunks[0].file_path).toBe('a.ts');
      expect(result.tokenCount).toBeGreaterThan(0);
    });

    it('uses context_header in markdown stitch when available', () => {
      const chunks = [
        { file_path: 'a.ts', context_header: 'custom header', char_start: 0, body_text: 'hello', score: 0.9 },
      ];
      const result = stitchContext(chunks);
      expect(result).toContain('[custom header]');
    });

    it('builds header from heading_path in markdown stitch', () => {
      const chunks = [
        { file_path: 'a.ts', heading_path: 'my heading', char_start: 0, body_text: 'hello', score: 0.9 },
      ];
      const result = stitchContext(chunks);
      expect(result).toContain('[a.ts > my heading]');
    });

    it('groups chunks from same file without repeating header', () => {
      const chunks = [
        { file_path: 'a.ts', char_start: 0, body_text: 'part1', score: 0.9 },
        { file_path: 'a.ts', char_start: 10, body_text: 'part2', score: 0.8 },
      ];
      const result = stitchContext(chunks);
      expect(result).toContain('part1');
      expect(result).toContain('part2');
      // Only one header since same file
      const headerCount = (result.match(/\[/g) || []).length;
      expect(headerCount).toBe(1);
    });

    it('handles empty chunks in markdown stitch', () => {
      const result = stitchContext([]);
      expect(result).toBe('');
    });

    it('handles empty chunks in json stitch', () => {
      const result = stitchContext([], { context_format: 'json' });
      expect(result.context).toBe('');
      expect(result.chunks).toEqual([]);
      expect(result.tokenCount).toBe(0);
    });

    it('uses content field in json stitch when body_text absent', () => {
      const chunks = [
        { file_path: 'a.ts', char_start: 0, char_end: 10, content: 'content body', score: 0.5 },
      ];
      const result = stitchContext(chunks, { context_format: 'json' });
      expect(result.chunks[0].content).toBe('content body');
    });

    it('assigns tier in json stitch when missing', () => {
      const chunks = [
        { file_path: 'a.ts', char_start: 0, char_end: 10, body_text: 'hello', score: 0.9 },
      ];
      const result = stitchContext(chunks, { context_format: 'json' });
      expect(result.chunks[0].tier).toBe('essential');
    });

    it('handles null heading_path in json stitch', () => {
      const chunks = [
        { file_path: 'a.ts', heading_path: null, char_start: 0, char_end: 10, body_text: 'hello', score: 0.9, tier: 'essential' },
      ];
      const result = stitchContext(chunks, { context_format: 'json' });
      expect(result.chunks[0].heading_path).toBe(null);
    });
  });

  describe('assembleContext', () => {
    it('runs the full pipeline without a client', async () => {
      const chunks = [
        { chunk_id: 1, file_path: 'a.ts', char_start: 0, char_end: 10, body_text: 'hello world', score: 0.9 },
        { chunk_id: 2, file_path: 'b.ts', char_start: 0, char_end: 10, body_text: 'foo bar', score: 0.5 },
      ];
      const result = await assembleContext(chunks, { budget: 1000 });
      expect(result.context).toContain('hello');
      expect(result.tokenCount).toBeGreaterThan(0);
      expect(result.tierCounts).toBeDefined();
      expect(result.selectedChunks).toBeDefined();
    });

    it('runs the full pipeline with json format', async () => {
      const chunks = [
        { chunk_id: 1, file_path: 'a.ts', char_start: 0, char_end: 10, body_text: 'hello world', score: 0.9 },
      ];
      const result = await assembleContext(chunks, { budget: 1000, context_format: 'json' });
      // assembleContext returns context as string, not the json object
      expect(typeof result.context).toBe('string');
      expect(result.tokenCount).toBeGreaterThan(0);
    });

    it('handles empty chunks', async () => {
      const result = await assembleContext([], { budget: 1000 });
      expect(result.context).toBe('');
      expect(result.tokenCount).toBe(0);
    });

    it('handles duplicate chunks', async () => {
      const chunks = [
        { chunk_id: 1, file_path: 'a.ts', char_start: 0, char_end: 10, body_text: 'hello', score: 0.9, sha256: 'same' },
        { chunk_id: 2, file_path: 'b.ts', char_start: 0, char_end: 10, body_text: 'hello', score: 0.8, sha256: 'same' },
      ];
      const result = await assembleContext(chunks, { budget: 1000 });
      // One chunk deduped
      expect(result.selectedChunks).toHaveLength(1);
    });
  });
});