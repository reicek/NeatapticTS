/**
 * @module metadata-filter.coverage.test
 * @description Coverage tests targeting rag-index/metadata-filter.mjs — exercises
 * validateFilter, compileFilterToSql, compileFilterToSqlAliased, applyPostRetrievalFilter,
 * and every error branch.
 */
import { validateFilter, compileFilterToSql, compileFilterToSqlAliased, applyPostRetrievalFilter, FilterError } from '../../../rag-index/metadata-filter.mjs';

describe('metadata-filter — FilterError', () => {
  it('creates an error with the FilterError name', () => {
    const err = new FilterError('test message');
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe('FilterError');
    expect(err.message).toBe('test message');
  });
});

describe('metadata-filter — validateFilter success cases', () => {
  it('validates eq predicate', () => {
    expect(() => validateFilter({ op: 'eq', field: 'arch_layer', value: 'network' })).not.toThrow();
  });

  it('validates neq predicate', () => {
    expect(() => validateFilter({ op: 'neq', field: 'arch_layer', value: 'network' })).not.toThrow();
  });

  it('validates eq with null value on numeric field', () => {
    expect(() => validateFilter({ op: 'eq', field: 'depth', value: null })).not.toThrow();
  });

  it('validates eq with null value on enum field', () => {
    expect(() => validateFilter({ op: 'eq', field: 'jsdoc_quality', value: null })).not.toThrow();
  });

  it('validates in predicate', () => {
    expect(() => validateFilter({ op: 'in', field: 'arch_layer', values: ['network', 'methods'] })).not.toThrow();
  });

  it('validates not_in predicate', () => {
    expect(() => validateFilter({ op: 'not_in', field: 'arch_layer', values: ['network'] })).not.toThrow();
  });

  it('validates gt predicate on numeric field', () => {
    expect(() => validateFilter({ op: 'gt', field: 'depth', value: 5 })).not.toThrow();
  });

  it('validates gte predicate', () => {
    expect(() => validateFilter({ op: 'gte', field: 'depth', value: 5 })).not.toThrow();
  });

  it('validates lt predicate', () => {
    expect(() => validateFilter({ op: 'lt', field: 'depth', value: 5 })).not.toThrow();
  });

  it('validates lte predicate', () => {
    expect(() => validateFilter({ op: 'lte', field: 'depth', value: 5 })).not.toThrow();
  });

  it('validates like predicate', () => {
    expect(() => validateFilter({ op: 'like', field: 'module_path', value: 'src/%' })).not.toThrow();
  });

  it('validates is_null predicate', () => {
    expect(() => validateFilter({ op: 'is_null', field: 'arch_layer' })).not.toThrow();
  });

  it('validates is_not_null predicate', () => {
    expect(() => validateFilter({ op: 'is_not_null', field: 'arch_layer' })).not.toThrow();
  });

  it('validates and predicate', () => {
    expect(() =>
      validateFilter({
        op: 'and',
        predicates: [
          { op: 'eq', field: 'arch_layer', value: 'network' },
          { op: 'eq', field: 'export_type', value: 'class' },
        ],
      }),
    ).not.toThrow();
  });

  it('validates or predicate', () => {
    expect(() =>
      validateFilter({
        op: 'or',
        predicates: [
          { op: 'eq', field: 'arch_layer', value: 'network' },
          { op: 'eq', field: 'arch_layer', value: 'methods' },
        ],
      }),
    ).not.toThrow();
  });

  it('validates not predicate', () => {
    expect(() =>
      validateFilter({ op: 'not', predicate: { op: 'eq', field: 'arch_layer', value: 'network' } }),
    ).not.toThrow();
  });

  it('validates eq with enum value', () => {
    expect(() => validateFilter({ op: 'eq', field: 'jsdoc_quality', value: 'good' })).not.toThrow();
  });

  it('validates eq with undefined value on enum field', () => {
    expect(() => validateFilter({ op: 'eq', field: 'jsdoc_quality', value: undefined })).not.toThrow();
  });

  it('validates in with numeric values on numeric field', () => {
    expect(() => validateFilter({ op: 'in', field: 'depth', values: [1, 2, 3] })).not.toThrow();
  });
});

describe('metadata-filter — validateFilter error cases', () => {
  it('throws on depth exceeded', () => {
    let predicate = { op: 'eq', field: 'arch_layer', value: 'x' };
    for (let i = 0; i < 12; i++) {
      predicate = { op: 'not', predicate };
    }
    expect(() => validateFilter(predicate)).toThrow(FilterError);
  });

  it('throws on predicate count exceeded (and-specific limit)', () => {
    const predicates = [];
    for (let i = 0; i < 51; i++) {
      predicates.push({ op: 'eq', field: 'arch_layer', value: `v${i}` });
    }
    expect(() => validateFilter({ op: 'and', predicates })).toThrow(FilterError);
  });

  it('throws on total predicate count exceeded via recursive validation', () => {
    const predicates = [];
    for (let i = 0; i < 50; i++) {
      predicates.push({ op: 'eq', field: 'arch_layer', value: `v${i}` });
    }
    // 50 predicates + 1 for the 'and' itself = 51 total, exceeding MAX_PREDICATE_COUNT (50)
    // This triggers the counter check at line 197 instead of the and-specific length check
    expect(() => validateFilter({ op: 'and', predicates })).toThrow(FilterError);
  });

  it('throws on non-object predicate', () => {
    expect(() => validateFilter(null)).toThrow(FilterError);
    expect(() => validateFilter('string')).toThrow(FilterError);
  });

  it('throws on unknown operator', () => {
    expect(() => validateFilter({ op: 'bogus', field: 'arch_layer' })).toThrow(FilterError);
  });

  it('throws on invalid field in eq', () => {
    expect(() => validateFilter({ op: 'eq', field: 'bogus', value: 'x' })).toThrow(FilterError);
  });

  it('throws on non-numeric value for numeric field in eq', () => {
    expect(() => validateFilter({ op: 'eq', field: 'depth', value: 'string' })).toThrow(FilterError);
  });

  it('throws on invalid enum value in eq', () => {
    expect(() => validateFilter({ op: 'eq', field: 'jsdoc_quality', value: 'bogus' })).toThrow(FilterError);
  });

  it('throws on invalid field in in predicate', () => {
    expect(() => validateFilter({ op: 'in', field: 'bogus', values: ['x'] })).toThrow(FilterError);
  });

  it('throws when in values is not an array', () => {
    expect(() => validateFilter({ op: 'in', field: 'arch_layer', values: 'not-array' })).toThrow(FilterError);
  });

  it('throws when in values is too long', () => {
    const values = Array(101).fill('x');
    expect(() => validateFilter({ op: 'in', field: 'arch_layer', values })).toThrow(FilterError);
  });

  it('throws when in values is empty', () => {
    expect(() => validateFilter({ op: 'in', field: 'arch_layer', values: [] })).toThrow(FilterError);
  });

  it('throws when in value is non-numeric for numeric field', () => {
    expect(() => validateFilter({ op: 'in', field: 'depth', values: ['string'] })).toThrow(FilterError);
  });

  it('throws when in value is invalid enum', () => {
    expect(() => validateFilter({ op: 'in', field: 'jsdoc_quality', values: ['bogus'] })).toThrow(FilterError);
  });

  it('throws on range op for non-numeric field', () => {
    expect(() => validateFilter({ op: 'gt', field: 'arch_layer', value: 5 })).toThrow(FilterError);
  });

  it('throws on non-numeric range value', () => {
    expect(() => validateFilter({ op: 'gt', field: 'depth', value: 'string' })).toThrow(FilterError);
  });

  it('throws on invalid field in like', () => {
    expect(() => validateFilter({ op: 'like', field: 'bogus', value: 'x%' })).toThrow(FilterError);
  });

  it('throws on non-string like value', () => {
    expect(() => validateFilter({ op: 'like', field: 'arch_layer', value: 123 })).toThrow(FilterError);
  });

  it('throws on invalid characters in like pattern', () => {
    expect(() => validateFilter({ op: 'like', field: 'arch_layer', value: 'bad|pattern' })).toThrow(FilterError);
  });

  it('throws on invalid field in is_null', () => {
    expect(() => validateFilter({ op: 'is_null', field: 'bogus' })).toThrow(FilterError);
  });

  it('throws on invalid field in is_not_null', () => {
    expect(() => validateFilter({ op: 'is_not_null', field: 'bogus' })).toThrow(FilterError);
  });

  it('throws when and has fewer than 2 predicates', () => {
    expect(() => validateFilter({ op: 'and', predicates: [{ op: 'eq', field: 'arch_layer', value: 'x' }] })).toThrow(FilterError);
  });

  it('throws when and has too many predicates', () => {
    const predicates = [];
    for (let i = 0; i < 51; i++) {
      predicates.push({ op: 'eq', field: 'arch_layer', value: `v${i}` });
    }
    expect(() => validateFilter({ op: 'and', predicates })).toThrow(FilterError);
  });

  it('throws when not predicate is missing inner', () => {
    expect(() => validateFilter({ op: 'not', predicate: null })).toThrow(FilterError);
  });

  it('throws when not predicate inner is not object', () => {
    expect(() => validateFilter({ op: 'not', predicate: 'string' })).toThrow(FilterError);
  });
});

describe('metadata-filter — compileFilterToSql', () => {
  it('compiles eq predicate', () => {
    const { sql, params } = compileFilterToSql({ op: 'eq', field: 'arch_layer', value: 'network' });
    expect(sql).toBe('chunks.arch_layer = ?');
    expect(params).toEqual(['network']);
  });

  it('compiles neq predicate', () => {
    const { sql, params } = compileFilterToSql({ op: 'neq', field: 'arch_layer', value: 'network' });
    expect(sql).toBe('chunks.arch_layer != ?');
    expect(params).toEqual(['network']);
  });

  it('compiles eq with null value as IS NULL', () => {
    const { sql } = compileFilterToSql({ op: 'eq', field: 'arch_layer', value: null });
    expect(sql).toBe('chunks.arch_layer IS NULL');
  });

  it('compiles neq with null value as IS NOT NULL', () => {
    const { sql } = compileFilterToSql({ op: 'neq', field: 'arch_layer', value: null });
    expect(sql).toBe('chunks.arch_layer IS NOT NULL');
  });

  it('compiles in predicate', () => {
    const { sql, params } = compileFilterToSql({ op: 'in', field: 'arch_layer', values: ['a', 'b'] });
    expect(sql).toBe('chunks.arch_layer IN (?, ?)');
    expect(params).toEqual(['a', 'b']);
  });

  it('compiles not_in predicate', () => {
    const { sql, params } = compileFilterToSql({ op: 'not_in', field: 'arch_layer', values: ['a', 'b'] });
    expect(sql).toBe('chunks.arch_layer NOT IN (?, ?)');
    expect(params).toEqual(['a', 'b']);
  });

  it('compiles gt predicate', () => {
    const { sql, params } = compileFilterToSql({ op: 'gt', field: 'depth', value: 5 });
    expect(sql).toBe('chunks.depth > ?');
    expect(params).toEqual([5]);
  });

  it('compiles gte predicate', () => {
    const { sql } = compileFilterToSql({ op: 'gte', field: 'depth', value: 5 });
    expect(sql).toBe('chunks.depth >= ?');
  });

  it('compiles lt predicate', () => {
    const { sql } = compileFilterToSql({ op: 'lt', field: 'depth', value: 5 });
    expect(sql).toBe('chunks.depth < ?');
  });

  it('compiles lte predicate', () => {
    const { sql } = compileFilterToSql({ op: 'lte', field: 'depth', value: 5 });
    expect(sql).toBe('chunks.depth <= ?');
  });

  it('compiles like predicate', () => {
    const { sql, params } = compileFilterToSql({ op: 'like', field: 'module_path', value: 'src/%' });
    expect(sql).toBe('chunks.module_path LIKE ?');
    expect(params).toEqual(['src/%']);
  });

  it('compiles is_null predicate', () => {
    const { sql } = compileFilterToSql({ op: 'is_null', field: 'arch_layer' });
    expect(sql).toBe('chunks.arch_layer IS NULL');
  });

  it('compiles is_not_null predicate', () => {
    const { sql } = compileFilterToSql({ op: 'is_not_null', field: 'arch_layer' });
    expect(sql).toBe('chunks.arch_layer IS NOT NULL');
  });

  it('compiles and predicate', () => {
    const { sql, params } = compileFilterToSql({
      op: 'and',
      predicates: [
        { op: 'eq', field: 'arch_layer', value: 'network' },
        { op: 'eq', field: 'export_type', value: 'class' },
      ],
    });
    expect(sql).toBe('(chunks.arch_layer = ? AND chunks.export_type = ?)');
    expect(params).toEqual(['network', 'class']);
  });

  it('compiles or predicate', () => {
    const { sql } = compileFilterToSql({
      op: 'or',
      predicates: [
        { op: 'eq', field: 'arch_layer', value: 'network' },
        { op: 'eq', field: 'arch_layer', value: 'methods' },
      ],
    });
    expect(sql).toBe('(chunks.arch_layer = ? OR chunks.arch_layer = ?)');
  });

  it('compiles not predicate', () => {
    const { sql } = compileFilterToSql({
      op: 'not',
      predicate: { op: 'eq', field: 'arch_layer', value: 'network' },
    });
    expect(sql).toBe('NOT (chunks.arch_layer = ?)');
  });

  it('compiles family (document-level) field', () => {
    const { sql, params } = compileFilterToSql({ op: 'eq', field: 'family', value: 'ts-source' });
    expect(sql).toBe('documents.doc_family = ?');
    expect(params).toEqual(['ts-source']);
  });

  it('compiles file_path (document-level) field', () => {
    const { sql } = compileFilterToSql({ op: 'eq', field: 'file_path', value: 'src/x.ts' });
    expect(sql).toBe('documents.file_path = ?');
  });

  it('throws on unknown operator in compile', () => {
    expect(() => compileFilterToSql({ op: 'bogus', field: 'arch_layer' })).toThrow(FilterError);
  });

  it('falls back to chunks.<field> for unmapped fields', () => {
    const { sql } = compileFilterToSql({ op: 'eq', field: 'unknown_field', value: 'x' });
    expect(sql).toBe('chunks.unknown_field = ?');
  });
});

describe('metadata-filter — compileFilterToSqlAliased', () => {
  it('compiles eq with aliases', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'eq', field: 'arch_layer', value: 'network' });
    expect(sql).toBe('c.arch_layer = ?');
    expect(params).toEqual(['network']);
  });

  it('compiles neq with aliases', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'neq', field: 'arch_layer', value: 'network' });
    expect(sql).toBe('c.arch_layer != ?');
    expect(params).toEqual(['network']);
  });

  it('compiles eq null with aliases (IS NULL)', () => {
    const { sql } = compileFilterToSqlAliased({ op: 'eq', field: 'arch_layer', value: null });
    expect(sql).toBe('c.arch_layer IS NULL');
  });

  it('compiles neq null with aliases (IS NOT NULL)', () => {
    const { sql } = compileFilterToSqlAliased({ op: 'neq', field: 'arch_layer', value: null });
    expect(sql).toBe('c.arch_layer IS NOT NULL');
  });

  it('compiles in with aliases', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'in', field: 'arch_layer', values: ['a', 'b'] });
    expect(sql).toBe('c.arch_layer IN (?, ?)');
    expect(params).toEqual(['a', 'b']);
  });

  it('compiles not_in with aliases', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'not_in', field: 'arch_layer', values: ['a', 'b'] });
    expect(sql).toBe('c.arch_layer NOT IN (?, ?)');
    expect(params).toEqual(['a', 'b']);
  });

  it('compiles gt with aliases', () => {
    const { sql } = compileFilterToSqlAliased({ op: 'gt', field: 'depth', value: 5 });
    expect(sql).toBe('c.depth > ?');
  });

  it('compiles gte with aliases', () => {
    const { sql } = compileFilterToSqlAliased({ op: 'gte', field: 'depth', value: 5 });
    expect(sql).toBe('c.depth >= ?');
  });

  it('compiles lt with aliases', () => {
    const { sql } = compileFilterToSqlAliased({ op: 'lt', field: 'depth', value: 5 });
    expect(sql).toBe('c.depth < ?');
  });

  it('compiles lte with aliases', () => {
    const { sql } = compileFilterToSqlAliased({ op: 'lte', field: 'depth', value: 5 });
    expect(sql).toBe('c.depth <= ?');
  });

  it('compiles like with aliases', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'like', field: 'module_path', value: 'src/%' });
    expect(sql).toBe('c.module_path LIKE ?');
    expect(params).toEqual(['src/%']);
  });

  it('compiles is_null with aliases', () => {
    const { sql } = compileFilterToSqlAliased({ op: 'is_null', field: 'arch_layer' });
    expect(sql).toBe('c.arch_layer IS NULL');
  });

  it('compiles is_not_null with aliases', () => {
    const { sql } = compileFilterToSqlAliased({ op: 'is_not_null', field: 'arch_layer' });
    expect(sql).toBe('c.arch_layer IS NOT NULL');
  });

  it('compiles and with aliases', () => {
    const { sql } = compileFilterToSqlAliased({
      op: 'and',
      predicates: [
        { op: 'eq', field: 'arch_layer', value: 'network' },
        { op: 'eq', field: 'export_type', value: 'class' },
      ],
    });
    expect(sql).toBe('(c.arch_layer = ? AND c.export_type = ?)');
  });

  it('compiles or with aliases', () => {
    const { sql } = compileFilterToSqlAliased({
      op: 'or',
      predicates: [
        { op: 'eq', field: 'arch_layer', value: 'network' },
        { op: 'eq', field: 'arch_layer', value: 'methods' },
      ],
    });
    expect(sql).toBe('(c.arch_layer = ? OR c.arch_layer = ?)');
  });

  it('compiles not with aliases', () => {
    const { sql } = compileFilterToSqlAliased({
      op: 'not',
      predicate: { op: 'eq', field: 'arch_layer', value: 'network' },
    });
    expect(sql).toBe('NOT (c.arch_layer = ?)');
  });

  it('compiles family (document-level) with d alias', () => {
    const { sql } = compileFilterToSqlAliased({ op: 'eq', field: 'family', value: 'ts-source' });
    expect(sql).toBe('d.doc_family = ?');
  });

  it('compiles file_path (document-level) with d alias', () => {
    const { sql } = compileFilterToSqlAliased({ op: 'eq', field: 'file_path', value: 'src/x.ts' });
    expect(sql).toBe('d.file_path = ?');
  });

  it('throws on unknown operator in aliased compile', () => {
    expect(() => compileFilterToSqlAliased({ op: 'bogus', field: 'arch_layer' })).toThrow(FilterError);
  });

  it('falls back to c.<field> for unmapped fields', () => {
    const { sql } = compileFilterToSqlAliased({ op: 'eq', field: 'unknown_field', value: 'x' });
    expect(sql).toBe('c.unknown_field = ?');
  });
});

describe('metadata-filter — applyPostRetrievalFilter', () => {
  const candidates = [
    { arch_layer: 'network', export_type: 'class', depth: 5, family: 'ts-source', file_path: 'src/a.ts', jsdoc_quality: 'good' },
    { arch_layer: 'methods', export_type: 'function', depth: 3, family: 'ts-source', file_path: 'src/b.ts', jsdoc_quality: 'none' },
    { arch_layer: 'network', export_type: 'function', depth: 10, family: 'md-doc', file_path: 'docs/c.md', jsdoc_quality: 'weak' },
  ];

  it('filters with eq', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'eq', field: 'arch_layer', value: 'network' });
    expect(result).toHaveLength(2);
  });

  it('filters with neq', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'neq', field: 'arch_layer', value: 'network' });
    expect(result).toHaveLength(1);
  });

  it('filters with eq null (field is null/undefined)', () => {
    const result = applyPostRetrievalFilter(
      [{ arch_layer: null }, { arch_layer: 'x' }],
      { op: 'eq', field: 'arch_layer', value: null },
    );
    expect(result).toHaveLength(1);
  });

  it('filters with neq null', () => {
    const result = applyPostRetrievalFilter(
      [{ arch_layer: null }, { arch_layer: 'x' }],
      { op: 'neq', field: 'arch_layer', value: null },
    );
    expect(result).toHaveLength(1);
  });

  it('filters with in', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'in', field: 'arch_layer', values: ['network', 'methods'] });
    expect(result).toHaveLength(3);
  });

  it('filters with not_in', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'not_in', field: 'arch_layer', values: ['network'] });
    expect(result).toHaveLength(1);
  });

  it('filters with gt', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'gt', field: 'depth', value: 4 });
    expect(result).toHaveLength(2);
  });

  it('filters with gte', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'gte', field: 'depth', value: 5 });
    expect(result).toHaveLength(2);
  });

  it('filters with lt', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'lt', field: 'depth', value: 5 });
    expect(result).toHaveLength(1);
  });

  it('filters with lte', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'lte', field: 'depth', value: 5 });
    expect(result).toHaveLength(2);
  });

  it('filters with like (% wildcard)', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'like', field: 'file_path', value: 'src/%' });
    expect(result).toHaveLength(2);
  });

  it('filters with like (_ wildcard)', () => {
    const result = applyPostRetrievalFilter(
      [{ file_path: 'abc' }, { file_path: 'abcd' }],
      { op: 'like', field: 'file_path', value: 'ab_' },
    );
    expect(result).toHaveLength(1);
    expect(result[0].file_path).toBe('abc');
  });

  it('filters with is_null', () => {
    const result = applyPostRetrievalFilter(
      [{ arch_layer: 'x' }, { arch_layer: null }, { other: 1 }],
      { op: 'is_null', field: 'arch_layer' },
    );
    expect(result).toHaveLength(2);
  });

  it('filters with is_not_null', () => {
    const result = applyPostRetrievalFilter(
      [{ arch_layer: 'x' }, { arch_layer: null }, { other: 1 }],
      { op: 'is_not_null', field: 'arch_layer' },
    );
    expect(result).toHaveLength(1);
  });

  it('filters with and', () => {
    const result = applyPostRetrievalFilter(candidates, {
      op: 'and',
      predicates: [
        { op: 'eq', field: 'arch_layer', value: 'network' },
        { op: 'eq', field: 'export_type', value: 'class' },
      ],
    });
    expect(result).toHaveLength(1);
  });

  it('filters with or', () => {
    const result = applyPostRetrievalFilter(candidates, {
      op: 'or',
      predicates: [
        { op: 'eq', field: 'arch_layer', value: 'network' },
        { op: 'eq', field: 'export_type', value: 'function' },
      ],
    });
    expect(result).toHaveLength(3);
  });

  it('filters with not', () => {
    const result = applyPostRetrievalFilter(candidates, {
      op: 'not',
      predicate: { op: 'eq', field: 'arch_layer', value: 'network' },
    });
    expect(result).toHaveLength(1);
  });

  it('returns false for unknown operator', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'bogus' });
    expect(result).toHaveLength(0);
  });

  it('uses doc_family fallback for family field', () => {
    const result = applyPostRetrievalFilter(
      [{ doc_family: 'ts-source' }, { doc_family: 'md-doc' }],
      { op: 'eq', field: 'family', value: 'ts-source' },
    );
    expect(result).toHaveLength(1);
  });

  it('uses metadata sub-object for field values', () => {
    const result = applyPostRetrievalFilter(
      [{ metadata: { arch_layer: 'network' } }, { metadata: { arch_layer: 'methods' } }],
      { op: 'eq', field: 'arch_layer', value: 'network' },
    );
    expect(result).toHaveLength(1);
  });

  it('like with regex special chars in pattern', () => {
    const result = applyPostRetrievalFilter(
      [{ file_path: 'src/file.test.ts' }, { file_path: 'src/other.ts' }],
      { op: 'like', field: 'file_path', value: 'src/file.test.ts' },
    );
    expect(result).toHaveLength(1);
  });

  it('like on candidate missing the field returns empty string (?? branch)', () => {
    const result = applyPostRetrievalFilter(
      [{ other: 1 }, { file_path: 'src/a.ts' }],
      { op: 'like', field: 'file_path', value: 'src/%' },
    );
    expect(result).toHaveLength(1);
    expect(result[0].file_path).toBe('src/a.ts');
  });

  it('eq on family where both family and doc_family are missing (?? null branch)', () => {
    const result = applyPostRetrievalFilter(
      [{ other: 1 }, { family: 'ts-source' }],
      { op: 'eq', field: 'family', value: 'ts-source' },
    );
    expect(result).toHaveLength(1);
  });

  it('eq on file_path where file_path is missing (?? null branch)', () => {
    const result = applyPostRetrievalFilter(
      [{ other: 1 }, { file_path: 'src/x.ts' }],
      { op: 'eq', field: 'file_path', value: 'src/x.ts' },
    );
    expect(result).toHaveLength(1);
  });
});