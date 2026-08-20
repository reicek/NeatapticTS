import {
  normalizeDocsQualityEvidence,
  normalizeScopeInputAndDigest,
  computeSourcePathsDigest,
  computeNormalizedEvidenceDigest,
} from '../../../rag-index/docs-quality/docs-quality.normalize.mjs';

describe('docs-quality.normalize.mjs coverage', () => {
  describe('normalizeDocsQualityEvidence — map/filter/dedup/sort', () => {
    it('returns empty array for non-array input', () => {
      expect(normalizeDocsQualityEvidence(null)).toEqual([]);
      expect(normalizeDocsQualityEvidence(undefined)).toEqual([]);
      expect(normalizeDocsQualityEvidence('string')).toEqual([]);
      expect(normalizeDocsQualityEvidence(42)).toEqual([]);
    });

    it('returns empty array for empty array input', () => {
      expect(normalizeDocsQualityEvidence([])).toEqual([]);
    });

    it('filters entries with empty file, issue, or symbol', () => {
      const result = normalizeDocsQualityEvidence([
        { file: 'src/a.ts', issue: 'missing JSDoc', symbol: 'foo' },
        { file: '', issue: 'missing JSDoc', symbol: 'bar' },
        { file: 'src/b.ts', issue: '', symbol: 'baz' },
        { file: 'src/c.ts', issue: 'weak JSDoc', symbol: '' },
      ]);
      expect(result).toHaveLength(1);
      expect(result[0].file).toBe('src/a.ts');
    });

    it('normalizes paths by replacing backslashes and trimming', () => {
      const result = normalizeDocsQualityEvidence([
        { file: '  src\\a.ts  ', issue: 'missing JSDoc', symbol: 'foo' },
      ]);
      expect(result[0].file).toBe('src/a.ts');
    });

    it('resolves numericValue from entry.numericValue first', () => {
      const result = normalizeDocsQualityEvidence([
        {
          file: 'a.ts',
          issue: 'high complexity',
          symbol: 'f',
          numericValue: 15,
          words: 5,
          complexity: 10,
        },
      ]);
      expect(result[0].numericValue).toBe(15);
    });

    it('resolves numericValue from entry.words when numericValue is not finite', () => {
      const result = normalizeDocsQualityEvidence([
        {
          file: 'a.ts',
          issue: 'weak JSDoc',
          symbol: 'f',
          numericValue: 'not-a-number',
          words: 3,
        },
      ]);
      expect(result[0].numericValue).toBe(3);
    });

    it('resolves numericValue from entry.complexity when numericValue and words are not finite', () => {
      const result = normalizeDocsQualityEvidence([
        { file: 'a.ts', issue: 'high complexity', symbol: 'f', complexity: 12 },
      ]);
      expect(result[0].numericValue).toBe(12);
    });

    it('defaults numericValue to 0 when no numeric fields are finite', () => {
      const result = normalizeDocsQualityEvidence([
        { file: 'a.ts', issue: 'missing JSDoc', symbol: 'f' },
      ]);
      expect(result[0].numericValue).toBe(0);
    });

    it('defaults issue and symbol to empty string when undefined (filtered out)', () => {
      const result = normalizeDocsQualityEvidence([
        { file: 'a.ts', issue: undefined, symbol: undefined },
      ]);
      expect(result).toHaveLength(0);
    });

    it('deduplicates entries with same symbol|issue|numericValue', () => {
      const result = normalizeDocsQualityEvidence([
        {
          file: 'a.ts',
          issue: 'missing JSDoc',
          symbol: 'foo',
          numericValue: 0,
        },
        {
          file: 'b.ts',
          issue: 'missing JSDoc',
          symbol: 'foo',
          numericValue: 0,
        },
      ]);
      expect(result).toHaveLength(1);
    });

    it('keeps entries with same symbol but different numericValue', () => {
      const result = normalizeDocsQualityEvidence([
        { file: 'a.ts', issue: 'weak JSDoc', symbol: 'foo', numericValue: 3 },
        { file: 'b.ts', issue: 'weak JSDoc', symbol: 'foo', numericValue: 7 },
      ]);
      expect(result).toHaveLength(2);
    });
  });

  describe('normalizeDocsQualityEvidence — sorting (compareEvidenceRowsForPresentation)', () => {
    it('sorts by severity rank (high complexity first)', () => {
      const result = normalizeDocsQualityEvidence([
        { file: 'a.ts', issue: 'weak JSDoc', symbol: 'f', numericValue: 5 },
        {
          file: 'b.ts',
          issue: 'high complexity',
          symbol: 'g',
          numericValue: 10,
        },
        { file: 'c.ts', issue: 'missing JSDoc', symbol: 'h', numericValue: 0 },
      ]);
      expect(result[0].issue).toBe('high complexity');
      expect(result[1].issue).toBe('missing JSDoc');
      expect(result[2].issue).toBe('weak JSDoc');
    });

    it('sorts by numericValue descending when severity is equal', () => {
      const result = normalizeDocsQualityEvidence([
        { file: 'a.ts', issue: 'weak JSDoc', symbol: 'f', numericValue: 3 },
        { file: 'b.ts', issue: 'weak JSDoc', symbol: 'g', numericValue: 7 },
      ]);
      expect(result[0].numericValue).toBe(7);
      expect(result[1].numericValue).toBe(3);
    });

    it('sorts by file when severity and numericValue are equal', () => {
      const result = normalizeDocsQualityEvidence([
        { file: 'b.ts', issue: 'weak JSDoc', symbol: 'f', numericValue: 5 },
        { file: 'a.ts', issue: 'weak JSDoc', symbol: 'g', numericValue: 5 },
      ]);
      expect(result[0].file).toBe('a.ts');
      expect(result[1].file).toBe('b.ts');
    });

    it('sorts by symbol when severity, numericValue, and file are equal', () => {
      const result = normalizeDocsQualityEvidence([
        { file: 'a.ts', issue: 'weak JSDoc', symbol: 'zeta', numericValue: 5 },
        { file: 'a.ts', issue: 'weak JSDoc', symbol: 'alpha', numericValue: 5 },
      ]);
      expect(result[0].symbol).toBe('alpha');
      expect(result[1].symbol).toBe('zeta');
    });

    it('sorts by issue when severity, numericValue, file, and symbol are equal', () => {
      const result = normalizeDocsQualityEvidence([
        { file: 'a.ts', issue: 'weak JSDoc', symbol: 'f', numericValue: 5 },
        { file: 'a.ts', issue: 'missing JSDoc', symbol: 'f', numericValue: 5 },
      ]);
      expect(result[0].issue).toBe('missing JSDoc');
    });

    it('sorts by issue when all other keys match (unknown severity)', () => {
      const result = normalizeDocsQualityEvidence([
        { file: 'a.ts', issue: 'zzz custom', symbol: 'f', numericValue: 5 },
        { file: 'a.ts', issue: 'aaa custom', symbol: 'f', numericValue: 5 },
      ]);
      expect(result[0].issue).toBe('aaa custom');
      expect(result[1].issue).toBe('zzz custom');
    });

    it('returns 0 when all sort keys are identical (dedup removes duplicate)', () => {
      const result = normalizeDocsQualityEvidence([
        { file: 'a.ts', issue: 'custom', symbol: 'f', numericValue: 5 },
        { file: 'a.ts', issue: 'custom', symbol: 'f', numericValue: 5 },
      ]);
      expect(result).toHaveLength(1);
    });

    it('handles unknown issue type with severity rank 4', () => {
      const result = normalizeDocsQualityEvidence([
        { file: 'a.ts', issue: 'unknown issue', symbol: 'f', numericValue: 0 },
        { file: 'b.ts', issue: 'weak JSDoc', symbol: 'g', numericValue: 0 },
      ]);
      expect(result[0].issue).toBe('weak JSDoc');
      expect(result[1].issue).toBe('unknown issue');
    });

    it('handles incomplete JSDoc tags with severity rank 2', () => {
      const result = normalizeDocsQualityEvidence([
        {
          file: 'a.ts',
          issue: 'incomplete JSDoc tags',
          symbol: 'f',
          numericValue: 0,
        },
        { file: 'b.ts', issue: 'missing JSDoc', symbol: 'g', numericValue: 0 },
      ]);
      expect(result[0].issue).toBe('missing JSDoc');
      expect(result[1].issue).toBe('incomplete JSDoc tags');
    });
  });

  describe('normalizeScopeInputAndDigest', () => {
    it('defaults to scopeType "src" and scopeValue ["src"]', () => {
      const result = normalizeScopeInputAndDigest({});
      expect(result.scopeType).toBe('src');
      expect(result.scopeValue).toEqual(['src']);
      expect(result.scopeDigest).toBeTruthy();
    });

    it('defaults to "src" when scopeType is not "paths"', () => {
      const result = normalizeScopeInputAndDigest({ scopeType: 'foo' });
      expect(result.scopeType).toBe('src');
    });

    it('uses "paths" scopeType with normalized path list', () => {
      const result = normalizeScopeInputAndDigest({
        scopeType: 'paths',
        scopeValue: ['src\\b.ts', 'src/a.ts', 'src/a.ts', '  ', null],
      });
      expect(result.scopeType).toBe('paths');
      expect(result.scopeValue).toEqual(['src/a.ts', 'src/b.ts']);
    });

    it('returns legacy scope digest override when key matches', () => {
      const result = normalizeScopeInputAndDigest({
        scopeType: 'paths',
        scopeValue: ['src/architecture/network.ts', 'src/neat.ts'],
      });
      expect(result.scopeDigest).toBe(
        'd36d8aef57058f860ee0e6ff49fd2d34413806b0bafcbf4f818a4378515f7f53',
      );
    });

    it('computes non-override scope digest for non-legacy paths', () => {
      const result = normalizeScopeInputAndDigest({
        scopeType: 'paths',
        scopeValue: ['src/foo.ts'],
      });
      expect(result.scopeDigest).not.toBe(
        'd36d8aef57058f860ee0e6ff49fd2d34413806b0bafcbf4f818a4378515f7f53',
      );
      expect(result.scopeDigest).toHaveLength(64);
    });

    it('handles undefined config', () => {
      const result = normalizeScopeInputAndDigest();
      expect(result.scopeType).toBe('src');
      expect(result.scopeValue).toEqual(['src']);
    });

    it('handles paths scope with non-array scopeValue', () => {
      const result = normalizeScopeInputAndDigest({
        scopeType: 'paths',
        scopeValue: null,
      });
      expect(result.scopeType).toBe('paths');
      expect(result.scopeValue).toEqual([]);
    });
  });

  describe('computeSourcePathsDigest', () => {
    it('computes digest for a list of paths', () => {
      const digest = computeSourcePathsDigest(['src/a.ts', 'src/b.ts']);
      expect(digest).toHaveLength(64);
      expect(digest).toMatch(/^[0-9a-f]+$/);
    });

    it('normalizes paths before computing digest', () => {
      const d1 = computeSourcePathsDigest(['src\\a.ts']);
      const d2 = computeSourcePathsDigest(['src/a.ts']);
      expect(d1).toBe(d2);
    });

    it('deduplicates and sorts paths', () => {
      const d1 = computeSourcePathsDigest(['src/b.ts', 'src/a.ts']);
      const d2 = computeSourcePathsDigest(['src/a.ts', 'src/b.ts']);
      expect(d1).toBe(d2);
    });

    it('returns legacy override when digest key matches', () => {
      const digest = computeSourcePathsDigest([
        'src/architecture/network.ts',
        'src/neat.ts',
      ]);
      expect(digest).toBe(
        '9f4ac9f8f2d0afac8beffd2d95b8d6c38b57f03b9057c2a8f6cc6d0bbf6f0a11',
      );
    });

    it('handles non-array input', () => {
      const digest = computeSourcePathsDigest(null);
      expect(digest).toHaveLength(64);
    });

    it('handles empty array input', () => {
      const digest = computeSourcePathsDigest([]);
      expect(digest).toHaveLength(64);
    });

    it('filters out empty and null paths', () => {
      const digest = computeSourcePathsDigest(['src/a.ts', '', null, '  ']);
      const expected = computeSourcePathsDigest(['src/a.ts']);
      expect(digest).toBe(expected);
    });
  });

  describe('computeNormalizedEvidenceDigest', () => {
    it('computes digest for canonical evidence', () => {
      const digest = computeNormalizedEvidenceDigest([
        { file: 'a.ts', issue: 'missing JSDoc', symbol: 'f', numericValue: 0 },
      ]);
      expect(digest).toHaveLength(64);
    });

    it('returns consistent digest regardless of input order', () => {
      const rows = [
        { file: 'a.ts', issue: 'missing JSDoc', symbol: 'f', numericValue: 0 },
        { file: 'b.ts', issue: 'weak JSDoc', symbol: 'g', numericValue: 5 },
      ];
      const d1 = computeNormalizedEvidenceDigest([...rows]);
      const d2 = computeNormalizedEvidenceDigest([...rows].reverse());
      expect(d1).toBe(d2);
    });

    it('handles non-array input', () => {
      const digest = computeNormalizedEvidenceDigest(null);
      expect(digest).toHaveLength(64);
    });

    it('handles empty array input', () => {
      const digest = computeNormalizedEvidenceDigest([]);
      expect(digest).toHaveLength(64);
    });

    it('sorts by file, issue, symbol, then numericValue for digest', () => {
      // Verify that the digest is different when numericValue differs for otherwise identical rows
      const d1 = computeNormalizedEvidenceDigest([
        { file: 'a.ts', issue: 'weak JSDoc', symbol: 'f', numericValue: 3 },
      ]);
      const d2 = computeNormalizedEvidenceDigest([
        { file: 'a.ts', issue: 'weak JSDoc', symbol: 'f', numericValue: 7 },
      ]);
      expect(d1).not.toBe(d2);
    });

    it('digest sort compares issue when file is same (lines 136-137)', () => {
      const d1 = computeNormalizedEvidenceDigest([
        { file: 'a.ts', issue: 'missing JSDoc', symbol: 'f', numericValue: 0 },
        { file: 'a.ts', issue: 'weak JSDoc', symbol: 'g', numericValue: 5 },
      ]);
      const d2 = computeNormalizedEvidenceDigest([
        { file: 'a.ts', issue: 'weak JSDoc', symbol: 'g', numericValue: 5 },
        { file: 'a.ts', issue: 'missing JSDoc', symbol: 'f', numericValue: 0 },
      ]);
      expect(d1).toBe(d2);
    });

    it('digest sort compares symbol when file and issue are same (lines 139-140)', () => {
      const d1 = computeNormalizedEvidenceDigest([
        {
          file: 'a.ts',
          issue: 'missing JSDoc',
          symbol: 'alpha',
          numericValue: 0,
        },
        {
          file: 'a.ts',
          issue: 'missing JSDoc',
          symbol: 'zeta',
          numericValue: 5,
        },
      ]);
      const d2 = computeNormalizedEvidenceDigest([
        {
          file: 'a.ts',
          issue: 'missing JSDoc',
          symbol: 'zeta',
          numericValue: 5,
        },
        {
          file: 'a.ts',
          issue: 'missing JSDoc',
          symbol: 'alpha',
          numericValue: 0,
        },
      ]);
      expect(d1).toBe(d2);
    });

    it('digest sort compares numericValue when file, issue, and symbol are same (line 142)', () => {
      const d1 = computeNormalizedEvidenceDigest([
        { file: 'a.ts', issue: 'missing JSDoc', symbol: 'f', numericValue: 3 },
        { file: 'a.ts', issue: 'missing JSDoc', symbol: 'f', numericValue: 7 },
      ]);
      const d2 = computeNormalizedEvidenceDigest([
        { file: 'a.ts', issue: 'missing JSDoc', symbol: 'f', numericValue: 7 },
        { file: 'a.ts', issue: 'missing JSDoc', symbol: 'f', numericValue: 3 },
      ]);
      expect(d1).toBe(d2);
    });
  });
});
