import { FilterError, validateFilter, compileFilterToSql, compileFilterToSqlAliased, applyPostRetrievalFilter } from './metadata-filter.mjs';

describe('FilterError', () => {
  it('creates an error with name FilterError', () => {
    const err = new FilterError('test message');
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(FilterError);
    expect(err.message).toBe('test message');
    expect(err.name).toBe('FilterError');
  });
});

describe('validateFilter', () => {
  it('validates eq predicate for string field', () => {
    expect(() => validateFilter({ op: 'eq', field: 'arch_layer', value: 'network' })).not.toThrow();
  });

  it('validates eq predicate for numeric field with number value', () => {
    expect(() => validateFilter({ op: 'eq', field: 'depth', value: 5 })).not.toThrow();
  });

  it('validates eq predicate for numeric field with null value', () => {
    expect(() => validateFilter({ op: 'eq', field: 'depth', value: null })).not.toThrow();
  });

  it('throws on eq for numeric field with string value', () => {
    expect(() => validateFilter({ op: 'eq', field: 'depth', value: 'five' })).toThrow(FilterError);
    expect(() => validateFilter({ op: 'eq', field: 'depth', value: 'five' })).toThrow('requires a numeric value');
  });

  it('throws on invalid field name', () => {
    expect(() => validateFilter({ op: 'eq', field: 'invalid_field', value: 'x' })).toThrow(FilterError);
    expect(() => validateFilter({ op: 'eq', field: 'invalid_field', value: 'x' })).toThrow('Invalid field');
  });

  it('throws on invalid enum value for jsdoc_quality', () => {
    expect(() => validateFilter({ op: 'eq', field: 'jsdoc_quality', value: 'bad' })).toThrow(FilterError);
    expect(() => validateFilter({ op: 'eq', field: 'jsdoc_quality', value: 'bad' })).toThrow('must be one of');
  });

  it('allows null for enum-validated field in eq', () => {
    expect(() => validateFilter({ op: 'eq', field: 'jsdoc_quality', value: null })).not.toThrow();
  });

  it('allows undefined for enum-validated field', () => {
    expect(() => validateFilter({ op: 'eq', field: 'jsdoc_quality', value: undefined })).not.toThrow();
  });

  it('validates neq predicate', () => {
    expect(() => validateFilter({ op: 'neq', field: 'arch_layer', value: 'network' })).not.toThrow();
  });

  it('validates in predicate', () => {
    expect(() => validateFilter({ op: 'in', field: 'arch_layer', values: ['network', 'neat'] })).not.toThrow();
  });

  it('validates not_in predicate', () => {
    expect(() => validateFilter({ op: 'not_in', field: 'arch_layer', values: ['network'] })).not.toThrow();
  });

  it('throws on in without values array', () => {
    expect(() => validateFilter({ op: 'in', field: 'arch_layer', values: null })).toThrow(FilterError);
    expect(() => validateFilter({ op: 'in', field: 'arch_layer' })).toThrow('requires a values array');
  });

  it('throws on in with too many values', () => {
    const values = Array.from({ length: 101 }, (_, i) => `v${i}`);
    expect(() => validateFilter({ op: 'in', field: 'arch_layer', values })).toThrow(FilterError);
    expect(() => validateFilter({ op: 'in', field: 'arch_layer', values })).toThrow('Too many values');
  });

  it('throws on in with empty values', () => {
    expect(() => validateFilter({ op: 'in', field: 'arch_layer', values: [] })).toThrow(FilterError);
    expect(() => validateFilter({ op: 'in', field: 'arch_layer', values: [] })).toThrow('at least one value');
  });

  it('throws on in with invalid field', () => {
    expect(() => validateFilter({ op: 'in', field: 'bad', values: ['x'] })).toThrow('Invalid field');
  });

  it('throws on in with numeric field but non-number values', () => {
    expect(() => validateFilter({ op: 'in', field: 'depth', values: ['x'] })).toThrow('requires numeric values');
  });

  it('throws on in with invalid enum value', () => {
    expect(() => validateFilter({ op: 'in', field: 'jsdoc_quality', values: ['bad'] })).toThrow('must be one of');
  });

  it('validates gt, gte, lt, lte for numeric fields', () => {
    expect(() => validateFilter({ op: 'gt', field: 'depth', value: 5 })).not.toThrow();
    expect(() => validateFilter({ op: 'gte', field: 'depth', value: 5 })).not.toThrow();
    expect(() => validateFilter({ op: 'lt', field: 'depth', value: 5 })).not.toThrow();
    expect(() => validateFilter({ op: 'lte', field: 'depth', value: 5 })).not.toThrow();
  });

  it('throws on gt for non-numeric field', () => {
    expect(() => validateFilter({ op: 'gt', field: 'arch_layer', value: 5 })).toThrow(FilterError);
    expect(() => validateFilter({ op: 'gt', field: 'arch_layer', value: 5 })).toThrow('does not support range');
  });

  it('throws on gt with non-number value', () => {
    expect(() => validateFilter({ op: 'gt', field: 'depth', value: 'five' })).toThrow('Range value must be a number');
  });

  it('validates like predicate', () => {
    expect(() => validateFilter({ op: 'like', field: 'module_path', value: 'src/%' })).not.toThrow();
    expect(() => validateFilter({ op: 'like', field: 'file_path', value: 'src/network/_ct.ts' })).not.toThrow();
  });

  it('throws on like with invalid field', () => {
    expect(() => validateFilter({ op: 'like', field: 'bad', value: '%' })).toThrow('Invalid field');
  });

  it('throws on like with non-string value', () => {
    expect(() => validateFilter({ op: 'like', field: 'module_path', value: 123 })).toThrow('LIKE value must be a string');
  });

  it('throws on like with invalid characters', () => {
    expect(() => validateFilter({ op: 'like', field: 'module_path', value: 'src/[bad]' })).toThrow('invalid characters');
  });

  it('validates is_null and is_not_null', () => {
    expect(() => validateFilter({ op: 'is_null', field: 'arch_layer' })).not.toThrow();
    expect(() => validateFilter({ op: 'is_not_null', field: 'arch_layer' })).not.toThrow();
  });

  it('throws on is_null with invalid field', () => {
    expect(() => validateFilter({ op: 'is_null', field: 'bad' })).toThrow('Invalid field');
  });

  it('validates and/or with 2+ predicates', () => {
    expect(() => validateFilter({ op: 'and', predicates: [
      { op: 'eq', field: 'arch_layer', value: 'network' },
      { op: 'eq', field: 'family', value: 'ts-source' },
    ] })).not.toThrow();
    expect(() => validateFilter({ op: 'or', predicates: [
      { op: 'eq', field: 'arch_layer', value: 'network' },
      { op: 'eq', field: 'arch_layer', value: 'neat' },
    ] })).not.toThrow();
  });

  it('throws on and with fewer than 2 predicates', () => {
    expect(() => validateFilter({ op: 'and', predicates: [{ op: 'eq', field: 'arch_layer', value: 'network' }] })).toThrow('at least 2 predicates');
    expect(() => validateFilter({ op: 'and', predicates: 'not array' })).toThrow('at least 2 predicates');
  });

  it('throws on and with too many predicates', () => {
    const preds = Array.from({ length: 51 }, () => ({ op: 'eq', field: 'arch_layer', value: 'network' }));
    expect(() => validateFilter({ op: 'and', predicates: preds })).toThrow('exceeds maximum');
  });

  it('throws on or with too many predicates', () => {
    const preds = Array.from({ length: 51 }, () => ({ op: 'eq', field: 'arch_layer', value: 'network' }));
    expect(() => validateFilter({ op: 'or', predicates: preds })).toThrow('exceeds maximum');
  });

  it('validates not predicate', () => {
    expect(() => validateFilter({ op: 'not', predicate: { op: 'eq', field: 'arch_layer', value: 'network' } })).not.toThrow();
  });

  it('throws on not without inner predicate', () => {
    expect(() => validateFilter({ op: 'not', predicate: null })).toThrow('requires an inner predicate');
    expect(() => validateFilter({ op: 'not', predicate: 'not object' })).toThrow('requires an inner predicate');
  });

  it('throws on unknown operator', () => {
    expect(() => validateFilter({ op: 'unknown', field: 'arch_layer' })).toThrow('Unknown operator');
  });

  it('throws on non-object predicate', () => {
    expect(() => validateFilter(null)).toThrow('non-null object');
    expect(() => validateFilter('string')).toThrow('non-null object');
    expect(() => validateFilter(42)).toThrow('non-null object');
  });

  it('throws when depth exceeds maximum', () => {
    let filter = { op: 'eq', field: 'arch_layer', value: 'network' };
    for (let i = 0; i < 12; i++) {
      filter = { op: 'not', predicate: filter };
    }
    expect(() => validateFilter(filter)).toThrow('nesting depth');
  });

  it('throws when predicate count exceeds maximum', () => {
    const preds = Array.from({ length: 51 }, () => ({ op: 'eq', field: 'arch_layer', value: 'network' }));
    expect(() => validateFilter({ op: 'and', predicates: preds })).toThrow('exceeds maximum');
  });

  it('throws when nested predicate count exceeds maximum via and', () => {
    // Create a filter that exceeds count via nested and
    const preds1 = Array.from({ length: 25 }, () => ({ op: 'eq', field: 'arch_layer', value: 'network' }));
    const preds2 = Array.from({ length: 26 }, () => ({ op: 'eq', field: 'arch_layer', value: 'neat' }));
    expect(() => validateFilter({ op: 'and', predicates: [
      { op: 'and', predicates: preds1 },
      { op: 'and', predicates: preds2 },
    ] })).toThrow('predicate count');
  });
});

describe('compileFilterToSql', () => {
  it('compiles eq for chunk field', () => {
    const { sql, params } = compileFilterToSql({ op: 'eq', field: 'arch_layer', value: 'network' });
    expect(sql).toBe('chunks.arch_layer = ?');
    expect(params).toEqual(['network']);
  });

  it('compiles neq for chunk field', () => {
    const { sql, params } = compileFilterToSql({ op: 'neq', field: 'arch_layer', value: 'network' });
    expect(sql).toBe('chunks.arch_layer != ?');
    expect(params).toEqual(['network']);
  });

  it('compiles eq for document field (family)', () => {
    const { sql, params } = compileFilterToSql({ op: 'eq', field: 'family', value: 'ts-source' });
    expect(sql).toBe('documents.doc_family = ?');
    expect(params).toEqual(['ts-source']);
  });

  it('compiles eq for document field (file_path)', () => {
    const { sql, params } = compileFilterToSql({ op: 'eq', field: 'file_path', value: 'src/test.ts' });
    expect(sql).toBe('documents.file_path = ?');
    expect(params).toEqual(['src/test.ts']);
  });

  it('compiles eq with null value as IS NULL', () => {
    const { sql, params } = compileFilterToSql({ op: 'eq', field: 'arch_layer', value: null });
    expect(sql).toBe('chunks.arch_layer IS NULL');
    expect(params).toEqual([]);
  });

  it('compiles neq with null value as IS NOT NULL', () => {
    const { sql, params } = compileFilterToSql({ op: 'neq', field: 'arch_layer', value: null });
    expect(sql).toBe('chunks.arch_layer IS NOT NULL');
    expect(params).toEqual([]);
  });

  it('compiles gt, gte, lt, lte', () => {
    expect(compileFilterToSql({ op: 'gt', field: 'depth', value: 5 }).sql).toBe('chunks.depth > ?');
    expect(compileFilterToSql({ op: 'gte', field: 'depth', value: 5 }).sql).toBe('chunks.depth >= ?');
    expect(compileFilterToSql({ op: 'lt', field: 'depth', value: 5 }).sql).toBe('chunks.depth < ?');
    expect(compileFilterToSql({ op: 'lte', field: 'depth', value: 5 }).sql).toBe('chunks.depth <= ?');
  });

  it('compiles in predicate', () => {
    const { sql, params } = compileFilterToSql({ op: 'in', field: 'arch_layer', values: ['network', 'neat'] });
    expect(sql).toBe('chunks.arch_layer IN (?, ?)');
    expect(params).toEqual(['network', 'neat']);
  });

  it('compiles not_in predicate', () => {
    const { sql, params } = compileFilterToSql({ op: 'not_in', field: 'arch_layer', values: ['network'] });
    expect(sql).toBe('chunks.arch_layer NOT IN (?)');
    expect(params).toEqual(['network']);
  });

  it('compiles like predicate', () => {
    const { sql, params } = compileFilterToSql({ op: 'like', field: 'module_path', value: 'src/%' });
    expect(sql).toBe('chunks.module_path LIKE ?');
    expect(params).toEqual(['src/%']);
  });

  it('compiles is_null', () => {
    const { sql, params } = compileFilterToSql({ op: 'is_null', field: 'arch_layer' });
    expect(sql).toBe('chunks.arch_layer IS NULL');
    expect(params).toEqual([]);
  });

  it('compiles is_not_null', () => {
    const { sql, params } = compileFilterToSql({ op: 'is_not_null', field: 'arch_layer' });
    expect(sql).toBe('chunks.arch_layer IS NOT NULL');
    expect(params).toEqual([]);
  });

  it('compiles and', () => {
    const { sql, params } = compileFilterToSql({ op: 'and', predicates: [
      { op: 'eq', field: 'arch_layer', value: 'network' },
      { op: 'eq', field: 'family', value: 'ts-source' },
    ] });
    expect(sql).toBe('(chunks.arch_layer = ? AND documents.doc_family = ?)');
    expect(params).toEqual(['network', 'ts-source']);
  });

  it('compiles or', () => {
    const { sql, params } = compileFilterToSql({ op: 'or', predicates: [
      { op: 'eq', field: 'arch_layer', value: 'network' },
      { op: 'eq', field: 'arch_layer', value: 'neat' },
    ] });
    expect(sql).toBe('(chunks.arch_layer = ? OR chunks.arch_layer = ?)');
    expect(params).toEqual(['network', 'neat']);
  });

  it('compiles not', () => {
    const { sql, params } = compileFilterToSql({ op: 'not', predicate: { op: 'eq', field: 'arch_layer', value: 'network' } });
    expect(sql).toBe('NOT (chunks.arch_layer = ?)');
    expect(params).toEqual(['network']);
  });

  it('throws on unknown operator in compile', () => {
    expect(() => compileFilterToSql({ op: 'unknown' })).toThrow('Unknown operator');
  });
});

describe('applyPostRetrievalFilter', () => {
  const candidates = [
    { arch_layer: 'network', family: 'ts-source', export_type: 'function', depth: 3 },
    { arch_layer: 'neat', family: 'ts-source', export_type: 'class', depth: 5 },
    { arch_layer: 'network', doc_family: 'plan', export_type: 'function', depth: 0 },
  ];

  it('filters by eq', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'eq', field: 'arch_layer', value: 'network' });
    expect(result).toHaveLength(2);
  });

  it('filters by neq', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'neq', field: 'arch_layer', value: 'network' });
    expect(result).toHaveLength(1);
    expect(result[0].arch_layer).toBe('neat');
  });

  it('filters by in', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'in', field: 'arch_layer', values: ['neat'] });
    expect(result).toHaveLength(1);
  });

  it('filters by not_in', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'not_in', field: 'arch_layer', values: ['neat'] });
    expect(result).toHaveLength(2);
  });

  it('filters by gt', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'gt', field: 'depth', value: 2 });
    expect(result).toHaveLength(2);
  });

  it('filters by gte', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'gte', field: 'depth', value: 3 });
    expect(result).toHaveLength(2);
  });

  it('filters by lt', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'lt', field: 'depth', value: 4 });
    expect(result).toHaveLength(2);
  });

  it('filters by lte', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'lte', field: 'depth', value: 3 });
    expect(result).toHaveLength(2);
  });

  it('filters by like', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'like', field: 'arch_layer', value: 'net%' });
    expect(result).toHaveLength(2);
  });

  it('filters by like with _ wildcard', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'like', field: 'arch_layer', value: 'ne_t' });
    expect(result).toHaveLength(1);
  });

  it('filters by is_null', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'is_null', field: 'module_path' });
    expect(result).toHaveLength(3);
  });

  it('filters by is_not_null', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'is_not_null', field: 'arch_layer' });
    expect(result).toHaveLength(3);
  });

  it('filters by and', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'and', predicates: [
      { op: 'eq', field: 'arch_layer', value: 'network' },
      { op: 'eq', field: 'export_type', value: 'function' },
    ] });
    expect(result).toHaveLength(2);
  });

  it('filters by or', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'or', predicates: [
      { op: 'eq', field: 'arch_layer', value: 'neat' },
      { op: 'eq', field: 'depth', value: 0 },
    ] });
    expect(result).toHaveLength(2);
  });

  it('filters by not', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'not', predicate: { op: 'eq', field: 'arch_layer', value: 'network' } });
    expect(result).toHaveLength(1);
  });

  it('returns false for unknown operator in evaluate', () => {
    const result = applyPostRetrievalFilter(candidates, { op: 'unknown' });
    expect(result).toHaveLength(0);
  });

  it('eq with null predicate value matches null/undefined field', () => {
    const cands = [{ arch_layer: null }, { arch_layer: 'network' }];
    const result = applyPostRetrievalFilter(cands, { op: 'eq', field: 'arch_layer', value: null });
    expect(result).toHaveLength(1);
  });

  it('neq with null predicate value', () => {
    const cands = [{ arch_layer: null }, { arch_layer: 'network' }];
    const result = applyPostRetrievalFilter(cands, { op: 'neq', field: 'arch_layer', value: null });
    expect(result).toHaveLength(1);
    expect(result[0].arch_layer).toBe('network');
  });

  it('family field maps to family property then doc_family', () => {
    const cands = [
      { family: 'ts-source', arch_layer: 'x' },
      { doc_family: 'plan', arch_layer: 'y' },
    ];
    expect(applyPostRetrievalFilter(cands, { op: 'eq', field: 'family', value: 'ts-source' })).toHaveLength(1);
    expect(applyPostRetrievalFilter(cands, { op: 'eq', field: 'family', value: 'plan' })).toHaveLength(1);
  });

  it('family field returns null when neither family nor doc_family set', () => {
    const cands = [{ arch_layer: 'x' }];
    expect(applyPostRetrievalFilter(cands, { op: 'is_null', field: 'family' })).toHaveLength(1);
  });

  it('file_path field', () => {
    const cands = [{ file_path: 'src/test.ts' }, { file_path: 'plans/test.md' }];
    expect(applyPostRetrievalFilter(cands, { op: 'eq', field: 'file_path', value: 'src/test.ts' })).toHaveLength(1);
  });

  it('file_path field returns null when file_path not set', () => {
    const cands = [{ arch_layer: 'x' }];
    expect(applyPostRetrievalFilter(cands, { op: 'is_null', field: 'file_path' })).toHaveLength(1);
  });

  it('metadata sub-object is checked for field values', () => {
    const cands = [{ metadata: { arch_layer: 'network' } }, { arch_layer: 'neat' }];
    expect(applyPostRetrievalFilter(cands, { op: 'eq', field: 'arch_layer', value: 'network' })).toHaveLength(1);
  });

  it('like on null/undefined field value converts to empty string', () => {
    const cands = [{ arch_layer: null }];
    const result = applyPostRetrievalFilter(cands, { op: 'like', field: 'arch_layer', value: '%' });
    expect(result).toHaveLength(1);
  });

  it('like with special regex chars in pattern', () => {
    const cands = [{ module_path: 'src/network.test.ts' }];
    const result = applyPostRetrievalFilter(cands, { op: 'like', field: 'module_path', value: 'src/network.test.ts' });
    expect(result).toHaveLength(1);
  });

  it('is_null with undefined field value', () => {
    const cands = [{ arch_layer: undefined }];
    expect(applyPostRetrievalFilter(cands, { op: 'is_null', field: 'arch_layer' })).toHaveLength(1);
  });

  it('is_not_null with undefined field value', () => {
    const cands = [{ arch_layer: undefined }];
    expect(applyPostRetrievalFilter(cands, { op: 'is_not_null', field: 'arch_layer' })).toHaveLength(0);
  });

  it('is_null with null field value', () => {
    const cands = [{ arch_layer: null }];
    expect(applyPostRetrievalFilter(cands, { op: 'is_null', field: 'arch_layer' })).toHaveLength(1);
  });

  it('is_not_null with null field value', () => {
    const cands = [{ arch_layer: null }];
    expect(applyPostRetrievalFilter(cands, { op: 'is_not_null', field: 'arch_layer' })).toHaveLength(0);
  });
});

describe('compileFilterToSql unknown field', () => {
  it('uses chunks.<field> for unknown field in compileFilterToSql', () => {
    const { sql, params } = compileFilterToSql({ op: 'eq', field: 'unknown_col', value: 'x' });
    expect(sql).toBe('chunks.unknown_col = ?');
    expect(params).toEqual(['x']);
  });
});

describe('compileFilterToSqlAliased', () => {
  it('compiles eq for chunk field', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'eq', field: 'arch_layer', value: 'network' });
    expect(sql).toBe('c.arch_layer = ?');
    expect(params).toEqual(['network']);
  });

  it('compiles neq for chunk field', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'neq', field: 'arch_layer', value: 'network' });
    expect(sql).toBe('c.arch_layer != ?');
    expect(params).toEqual(['network']);
  });

  it('compiles eq for document field (family)', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'eq', field: 'family', value: 'ts-source' });
    expect(sql).toBe('d.doc_family = ?');
    expect(params).toEqual(['ts-source']);
  });

  it('compiles eq for document field (file_path)', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'eq', field: 'file_path', value: 'src/test.ts' });
    expect(sql).toBe('d.file_path = ?');
    expect(params).toEqual(['src/test.ts']);
  });

  it('compiles eq with null value as IS NULL', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'eq', field: 'arch_layer', value: null });
    expect(sql).toBe('c.arch_layer IS NULL');
    expect(params).toEqual([]);
  });

  it('compiles neq with null value as IS NOT NULL', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'neq', field: 'arch_layer', value: null });
    expect(sql).toBe('c.arch_layer IS NOT NULL');
    expect(params).toEqual([]);
  });

  it('compiles gt, gte, lt, lte', () => {
    expect(compileFilterToSqlAliased({ op: 'gt', field: 'depth', value: 5 }).sql).toBe('c.depth > ?');
    expect(compileFilterToSqlAliased({ op: 'gte', field: 'depth', value: 5 }).sql).toBe('c.depth >= ?');
    expect(compileFilterToSqlAliased({ op: 'lt', field: 'depth', value: 5 }).sql).toBe('c.depth < ?');
    expect(compileFilterToSqlAliased({ op: 'lte', field: 'depth', value: 5 }).sql).toBe('c.depth <= ?');
  });

  it('compiles in predicate', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'in', field: 'arch_layer', values: ['network', 'neat'] });
    expect(sql).toBe('c.arch_layer IN (?, ?)');
    expect(params).toEqual(['network', 'neat']);
  });

  it('compiles not_in predicate', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'not_in', field: 'arch_layer', values: ['network'] });
    expect(sql).toBe('c.arch_layer NOT IN (?)');
    expect(params).toEqual(['network']);
  });

  it('compiles like predicate', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'like', field: 'module_path', value: 'src/%' });
    expect(sql).toBe('c.module_path LIKE ?');
    expect(params).toEqual(['src/%']);
  });

  it('compiles is_null', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'is_null', field: 'arch_layer' });
    expect(sql).toBe('c.arch_layer IS NULL');
    expect(params).toEqual([]);
  });

  it('compiles is_not_null', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'is_not_null', field: 'arch_layer' });
    expect(sql).toBe('c.arch_layer IS NOT NULL');
    expect(params).toEqual([]);
  });

  it('compiles and', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'and', predicates: [
      { op: 'eq', field: 'arch_layer', value: 'network' },
      { op: 'eq', field: 'family', value: 'ts-source' },
    ] });
    expect(sql).toBe('(c.arch_layer = ? AND d.doc_family = ?)');
    expect(params).toEqual(['network', 'ts-source']);
  });

  it('compiles or', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'or', predicates: [
      { op: 'eq', field: 'arch_layer', value: 'network' },
      { op: 'eq', field: 'arch_layer', value: 'neat' },
    ] });
    expect(sql).toBe('(c.arch_layer = ? OR c.arch_layer = ?)');
    expect(params).toEqual(['network', 'neat']);
  });

  it('compiles not', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'not', predicate: { op: 'eq', field: 'arch_layer', value: 'network' } });
    expect(sql).toBe('NOT (c.arch_layer = ?)');
    expect(params).toEqual(['network']);
  });

  it('uses c.<field> for unknown field', () => {
    const { sql, params } = compileFilterToSqlAliased({ op: 'eq', field: 'unknown_col', value: 'x' });
    expect(sql).toBe('c.unknown_col = ?');
    expect(params).toEqual(['x']);
  });

  it('throws on unknown operator', () => {
    expect(() => compileFilterToSqlAliased({ op: 'unknown' })).toThrow('Unknown operator');
  });
});