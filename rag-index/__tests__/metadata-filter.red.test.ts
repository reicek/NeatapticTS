/**
 * @module metadata-filter.red.test
 * @description Red tests for the metadata filter grammar module.
 *
 * Uses runModuleEvaluation to call .mjs module functions from .ts test files,
 * matching the established pattern in classify-query.red.test.ts.
 *
 * Covers validateFilter, compileFilterToSql, compileFilterToSqlAliased,
 * applyPostRetrievalFilter, and FilterError across five categories:
 * predicate types, validation errors, SQL generation, aliased SQL generation,
 * and in-memory evaluation.
 */
import { execFileSync } from 'node:child_process';
import path from 'node:path';

// ---------------------------------------------------------------------------
// Types for filter results
// ---------------------------------------------------------------------------

interface SqlResult {
  sql: string;
  params: Array<string | number | null>;
}

interface ErrorResult {
  name: string;
  message: string;
}

// ---------------------------------------------------------------------------
// Helper: evaluate .mjs modules via subprocess (matching established pattern)
// ---------------------------------------------------------------------------

const runModuleEvaluation = <Result>(source: string): Result => {
  const output = execFileSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: path.resolve(process.cwd()),
      encoding: 'utf8',
    },
  );
  return JSON.parse(output) as Result;
};

// ---------------------------------------------------------------------------
// validateFilter — predicate types (all should pass validation)
// ---------------------------------------------------------------------------

describe('metadata-filter validateFilter — predicate types', () => {
  it('accepts eq with valid string field', () => {
    const result = runModuleEvaluation<{ ok: boolean }>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      validateFilter({ op: 'eq', field: 'arch_layer', value: 'network' });
      console.log(JSON.stringify({ ok: true }));
    `);
    expect(result.ok).toBe(true);
  });

  it('accepts eq with valid numeric field', () => {
    const result = runModuleEvaluation<{ ok: boolean }>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      validateFilter({ op: 'eq', field: 'jsdoc_word_count', value: 10 });
      console.log(JSON.stringify({ ok: true }));
    `);
    expect(result.ok).toBe(true);
  });

  it('accepts neq with valid string field', () => {
    const result = runModuleEvaluation<{ ok: boolean }>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      validateFilter({ op: 'neq', field: 'export_type', value: 'function' });
      console.log(JSON.stringify({ ok: true }));
    `);
    expect(result.ok).toBe(true);
  });

  it('accepts in with valid string field and values array', () => {
    const result = runModuleEvaluation<{ ok: boolean }>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      validateFilter({ op: 'in', field: 'export_type', values: ['function', 'class'] });
      console.log(JSON.stringify({ ok: true }));
    `);
    expect(result.ok).toBe(true);
  });

  it('accepts not_in with valid string field and values array', () => {
    const result = runModuleEvaluation<{ ok: boolean }>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      validateFilter({ op: 'not_in', field: 'export_type', values: ['function', 'class'] });
      console.log(JSON.stringify({ ok: true }));
    `);
    expect(result.ok).toBe(true);
  });

  it('accepts gt with numeric field', () => {
    const result = runModuleEvaluation<{ ok: boolean }>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      validateFilter({ op: 'gt', field: 'jsdoc_word_count', value: 5 });
      console.log(JSON.stringify({ ok: true }));
    `);
    expect(result.ok).toBe(true);
  });

  it('accepts gte with numeric field', () => {
    const result = runModuleEvaluation<{ ok: boolean }>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      validateFilter({ op: 'gte', field: 'jsdoc_word_count', value: 5 });
      console.log(JSON.stringify({ ok: true }));
    `);
    expect(result.ok).toBe(true);
  });

  it('accepts lt with numeric field', () => {
    const result = runModuleEvaluation<{ ok: boolean }>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      validateFilter({ op: 'lt', field: 'depth', value: 3 });
      console.log(JSON.stringify({ ok: true }));
    `);
    expect(result.ok).toBe(true);
  });

  it('accepts lte with numeric field', () => {
    const result = runModuleEvaluation<{ ok: boolean }>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      validateFilter({ op: 'lte', field: 'depth', value: 3 });
      console.log(JSON.stringify({ ok: true }));
    `);
    expect(result.ok).toBe(true);
  });

  it('accepts like with valid pattern', () => {
    const result = runModuleEvaluation<{ ok: boolean }>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      validateFilter({ op: 'like', field: 'module_path', value: '%network%' });
      console.log(JSON.stringify({ ok: true }));
    `);
    expect(result.ok).toBe(true);
  });

  it('accepts is_null with valid field', () => {
    const result = runModuleEvaluation<{ ok: boolean }>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      validateFilter({ op: 'is_null', field: 'jsdoc_quality' });
      console.log(JSON.stringify({ ok: true }));
    `);
    expect(result.ok).toBe(true);
  });

  it('accepts is_not_null with valid field', () => {
    const result = runModuleEvaluation<{ ok: boolean }>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      validateFilter({ op: 'is_not_null', field: 'jsdoc_quality' });
      console.log(JSON.stringify({ ok: true }));
    `);
    expect(result.ok).toBe(true);
  });

  it('accepts and with two predicates', () => {
    const result = runModuleEvaluation<{ ok: boolean }>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      validateFilter({ op: 'and', predicates: [
        { op: 'eq', field: 'family', value: 'ts-source' },
        { op: 'eq', field: 'arch_layer', value: 'network' },
      ]});
      console.log(JSON.stringify({ ok: true }));
    `);
    expect(result.ok).toBe(true);
  });

  it('accepts or with two predicates', () => {
    const result = runModuleEvaluation<{ ok: boolean }>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      validateFilter({ op: 'or', predicates: [
        { op: 'eq', field: 'family', value: 'ts-source' },
        { op: 'eq', field: 'arch_layer', value: 'network' },
      ]});
      console.log(JSON.stringify({ ok: true }));
    `);
    expect(result.ok).toBe(true);
  });

  it('accepts not with inner predicate', () => {
    const result = runModuleEvaluation<{ ok: boolean }>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      validateFilter({ op: 'not', predicate: { op: 'eq', field: 'arch_layer', value: 'network' } });
      console.log(JSON.stringify({ ok: true }));
    `);
    expect(result.ok).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// validateFilter — validation errors
// ---------------------------------------------------------------------------

describe('metadata-filter validateFilter — validation errors', () => {
  it('throws FilterError for invalid field name', () => {
    const result = runModuleEvaluation<ErrorResult>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      try {
        validateFilter({ op: 'eq', field: 'invalid_field', value: 'x' });
        console.log(JSON.stringify({ name: '', message: '' }));
      } catch (e) {
        console.log(JSON.stringify({ name: e.name, message: e.message }));
      }
    `);
    expect(result).toEqual({
      name: 'FilterError',
      message: 'Invalid field: invalid_field',
    });
  });

  it('throws FilterError when numeric field receives string value', () => {
    const result = runModuleEvaluation<ErrorResult>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      try {
        validateFilter({ op: 'eq', field: 'jsdoc_word_count', value: 'not_a_number' });
        console.log(JSON.stringify({ name: '', message: '' }));
      } catch (e) {
        console.log(JSON.stringify({ name: e.name, message: e.message }));
      }
    `);
    expect(result).toEqual({
      name: 'FilterError',
      message: 'Field jsdoc_word_count requires a numeric value',
    });
  });

  it('throws FilterError when string field uses range operator', () => {
    const result = runModuleEvaluation<ErrorResult>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      try {
        validateFilter({ op: 'gt', field: 'arch_layer', value: 5 });
        console.log(JSON.stringify({ name: '', message: '' }));
      } catch (e) {
        console.log(JSON.stringify({ name: e.name, message: e.message }));
      }
    `);
    expect(result).toEqual({
      name: 'FilterError',
      message: 'Field arch_layer does not support range operators',
    });
  });

  it('throws FilterError when nesting depth exceeds maximum', () => {
    const result = runModuleEvaluation<ErrorResult>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      let predicate = { op: 'eq', field: 'depth', value: 0 };
      for (let i = 0; i < 11; i++) {
        predicate = { op: 'not', predicate: predicate };
      }
      try {
        validateFilter(predicate);
        console.log(JSON.stringify({ name: '', message: '' }));
      } catch (e) {
        console.log(JSON.stringify({ name: e.name, message: e.message }));
      }
    `);
    expect(result).toEqual({
      name: 'FilterError',
      message: 'Filter nesting depth exceeds maximum (10)',
    });
  });

  it('throws FilterError when predicate count exceeds maximum', () => {
    const result = runModuleEvaluation<ErrorResult>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      const predicates = [];
      for (let i = 0; i < 51; i++) {
        predicates.push({ op: 'eq', field: 'depth', value: i });
      }
      try {
        validateFilter({ op: 'and', predicates: predicates });
        console.log(JSON.stringify({ name: '', message: '' }));
      } catch (e) {
        console.log(JSON.stringify({ name: e.name, message: e.message }));
      }
    `);
    expect(result).toEqual({
      name: 'FilterError',
      message: 'and exceeds maximum 50 predicates',
    });
  });

  it('throws FilterError for empty in values array', () => {
    const result = runModuleEvaluation<ErrorResult>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      try {
        validateFilter({ op: 'in', field: 'export_type', values: [] });
        console.log(JSON.stringify({ name: '', message: '' }));
      } catch (e) {
        console.log(JSON.stringify({ name: e.name, message: e.message }));
      }
    `);
    expect(result).toEqual({
      name: 'FilterError',
      message: 'in requires at least one value',
    });
  });

  it('throws FilterError for LIKE pattern with invalid characters', () => {
    const result = runModuleEvaluation<ErrorResult>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      try {
        validateFilter({ op: 'like', field: 'module_path', value: '%test;drop%' });
        console.log(JSON.stringify({ name: '', message: '' }));
      } catch (e) {
        console.log(JSON.stringify({ name: e.name, message: e.message }));
      }
    `);
    expect(result).toEqual({
      name: 'FilterError',
      message:
        'LIKE pattern contains invalid characters (only alphanumeric, %, _, /, ., - are allowed)',
    });
  });

  it('throws FilterError for unknown operator', () => {
    const result = runModuleEvaluation<ErrorResult>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      try {
        validateFilter({ op: 'unknown', field: 'arch_layer', value: 'x' });
        console.log(JSON.stringify({ name: '', message: '' }));
      } catch (e) {
        console.log(JSON.stringify({ name: e.name, message: e.message }));
      }
    `);
    expect(result).toEqual({
      name: 'FilterError',
      message: 'Unknown operator: unknown',
    });
  });

  it('throws FilterError for non-object predicate', () => {
    const result = runModuleEvaluation<ErrorResult>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      try {
        validateFilter(null);
        console.log(JSON.stringify({ name: '', message: '' }));
      } catch (e) {
        console.log(JSON.stringify({ name: e.name, message: e.message }));
      }
    `);
    expect(result).toEqual({
      name: 'FilterError',
      message: 'Predicate must be a non-null object',
    });
  });

  it('throws FilterError when and has fewer than 2 predicates', () => {
    const result = runModuleEvaluation<ErrorResult>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      try {
        validateFilter({ op: 'and', predicates: [{ op: 'eq', field: 'arch_layer', value: 'network' }] });
        console.log(JSON.stringify({ name: '', message: '' }));
      } catch (e) {
        console.log(JSON.stringify({ name: e.name, message: e.message }));
      }
    `);
    expect(result).toEqual({
      name: 'FilterError',
      message: 'and requires at least 2 predicates',
    });
  });

  it('throws FilterError when or has fewer than 2 predicates', () => {
    const result = runModuleEvaluation<ErrorResult>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      try {
        validateFilter({ op: 'or', predicates: [{ op: 'eq', field: 'arch_layer', value: 'network' }] });
        console.log(JSON.stringify({ name: '', message: '' }));
      } catch (e) {
        console.log(JSON.stringify({ name: e.name, message: e.message }));
      }
    `);
    expect(result).toEqual({
      name: 'FilterError',
      message: 'or requires at least 2 predicates',
    });
  });

  it('throws FilterError when not has no inner predicate', () => {
    const result = runModuleEvaluation<ErrorResult>(`
      import { validateFilter } from './rag-index/metadata-filter.mjs';
      try {
        validateFilter({ op: 'not' });
        console.log(JSON.stringify({ name: '', message: '' }));
      } catch (e) {
        console.log(JSON.stringify({ name: e.name, message: e.message }));
      }
    `);
    expect(result).toEqual({
      name: 'FilterError',
      message: 'not predicate requires an inner predicate',
    });
  });
});

// ---------------------------------------------------------------------------
// compileFilterToSql — SQL generation
// ---------------------------------------------------------------------------

describe('metadata-filter compileFilterToSql — SQL generation', () => {
  it('compiles eq on chunk field to parameterized SQL', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'eq', field: 'arch_layer', value: 'network' })));
    `);
    expect(result).toEqual({
      sql: 'chunks.arch_layer = ?',
      params: ['network'],
    });
  });

  it('compiles eq on document field to simple column reference SQL', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'eq', field: 'family', value: 'ts-source' })));
    `);
    expect(result.sql).toContain('documents.doc_family = ?');
    expect(result.params).toEqual(['ts-source']);
  });

  it('compiles neq on chunk field to parameterized SQL', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'neq', field: 'arch_layer', value: 'network' })));
    `);
    expect(result).toEqual({
      sql: 'chunks.arch_layer != ?',
      params: ['network'],
    });
  });

  it('compiles in to parameterized IN clause', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'in', field: 'export_type', values: ['function', 'class'] })));
    `);
    expect(result).toEqual({
      sql: 'chunks.export_type IN (?, ?)',
      params: ['function', 'class'],
    });
  });

  it('compiles not_in to parameterized NOT IN clause', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'not_in', field: 'export_type', values: ['function', 'class'] })));
    `);
    expect(result).toEqual({
      sql: 'chunks.export_type NOT IN (?, ?)',
      params: ['function', 'class'],
    });
  });

  it('compiles gt on numeric field to parameterized SQL', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'gt', field: 'jsdoc_word_count', value: 5 })));
    `);
    expect(result).toEqual({ sql: 'chunks.jsdoc_word_count > ?', params: [5] });
  });

  it('compiles gte on numeric field to parameterized SQL', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'gte', field: 'jsdoc_word_count', value: 5 })));
    `);
    expect(result).toEqual({
      sql: 'chunks.jsdoc_word_count >= ?',
      params: [5],
    });
  });

  it('compiles lt on numeric field to parameterized SQL', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'lt', field: 'depth', value: 3 })));
    `);
    expect(result).toEqual({ sql: 'chunks.depth < ?', params: [3] });
  });

  it('compiles lte on numeric field to parameterized SQL', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'lte', field: 'depth', value: 3 })));
    `);
    expect(result).toEqual({ sql: 'chunks.depth <= ?', params: [3] });
  });

  it('compiles like to parameterized LIKE clause', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'like', field: 'module_path', value: '%network%' })));
    `);
    expect(result).toEqual({
      sql: 'chunks.module_path LIKE ?',
      params: ['%network%'],
    });
  });

  it('compiles is_null to IS NULL clause', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'is_null', field: 'jsdoc_quality' })));
    `);
    expect(result).toEqual({ sql: 'chunks.jsdoc_quality IS NULL', params: [] });
  });

  it('compiles is_not_null to IS NOT NULL clause', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'is_not_null', field: 'jsdoc_quality' })));
    `);
    expect(result).toEqual({
      sql: 'chunks.jsdoc_quality IS NOT NULL',
      params: [],
    });
  });

  it('compiles and to AND-combined SQL', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'and', predicates: [
        { op: 'eq', field: 'arch_layer', value: 'network' },
        { op: 'eq', field: 'export_type', value: 'function' },
      ]})));
    `);
    expect(result).toEqual({
      sql: '(chunks.arch_layer = ? AND chunks.export_type = ?)',
      params: ['network', 'function'],
    });
  });

  it('compiles or to OR-combined SQL', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'or', predicates: [
        { op: 'eq', field: 'arch_layer', value: 'network' },
        { op: 'eq', field: 'export_type', value: 'function' },
      ]})));
    `);
    expect(result).toEqual({
      sql: '(chunks.arch_layer = ? OR chunks.export_type = ?)',
      params: ['network', 'function'],
    });
  });

  it('compiles not to NOT-wrapped SQL', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'not', predicate: { op: 'eq', field: 'arch_layer', value: 'network' } })));
    `);
    expect(result).toEqual({
      sql: 'NOT (chunks.arch_layer = ?)',
      params: ['network'],
    });
  });

  it('compiles eq with null value to IS NULL', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'eq', field: 'jsdoc_quality', value: null })));
    `);
    expect(result).toEqual({ sql: 'chunks.jsdoc_quality IS NULL', params: [] });
  });

  it('compiles neq with null value to IS NOT NULL', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSql } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSql({ op: 'neq', field: 'jsdoc_quality', value: null })));
    `);
    expect(result).toEqual({
      sql: 'chunks.jsdoc_quality IS NOT NULL',
      params: [],
    });
  });
});

// ---------------------------------------------------------------------------
// compileFilterToSqlAliased — aliased SQL generation
// ---------------------------------------------------------------------------

describe('metadata-filter compileFilterToSqlAliased — aliased SQL generation', () => {
  it('compiles eq on chunk field with c. alias', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSqlAliased } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSqlAliased({ op: 'eq', field: 'arch_layer', value: 'network' })));
    `);
    expect(result).toEqual({ sql: 'c.arch_layer = ?', params: ['network'] });
  });

  it('compiles eq on document field with d. alias', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSqlAliased } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSqlAliased({ op: 'eq', field: 'family', value: 'ts-source' })));
    `);
    expect(result).toEqual({ sql: 'd.doc_family = ?', params: ['ts-source'] });
  });

  it('compiles in with aliased references', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSqlAliased } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSqlAliased({ op: 'in', field: 'export_type', values: ['function', 'class'] })));
    `);
    expect(result).toEqual({
      sql: 'c.export_type IN (?, ?)',
      params: ['function', 'class'],
    });
  });

  it('compiles and/or/not with aliased references', () => {
    const result = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSqlAliased } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSqlAliased({ op: 'and', predicates: [
        { op: 'eq', field: 'family', value: 'ts-source' },
        { op: 'eq', field: 'arch_layer', value: 'network' },
      ]})));
    `);
    expect(result).toEqual({
      sql: '(d.doc_family = ? AND c.arch_layer = ?)',
      params: ['ts-source', 'network'],
    });
  });

  it('compiles is_null and is_not_null with aliased references', () => {
    const isNullResult = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSqlAliased } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSqlAliased({ op: 'is_null', field: 'jsdoc_quality' })));
    `);
    expect(isNullResult).toEqual({
      sql: 'c.jsdoc_quality IS NULL',
      params: [],
    });

    const isNotNullResult = runModuleEvaluation<SqlResult>(`
      import { compileFilterToSqlAliased } from './rag-index/metadata-filter.mjs';
      console.log(JSON.stringify(compileFilterToSqlAliased({ op: 'is_not_null', field: 'jsdoc_quality' })));
    `);
    expect(isNotNullResult).toEqual({
      sql: 'c.jsdoc_quality IS NOT NULL',
      params: [],
    });
  });
});

// ---------------------------------------------------------------------------
// applyPostRetrievalFilter — in-memory evaluation
// ---------------------------------------------------------------------------

describe('metadata-filter applyPostRetrievalFilter — in-memory evaluation', () => {
  it('eq matches exact value', () => {
    const result = runModuleEvaluation<Array<{ arch_layer: string }>>(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [{ arch_layer: 'network' }, { arch_layer: 'mutate' }];
      const predicate = { op: 'eq', field: 'arch_layer', value: 'network' };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([{ arch_layer: 'network' }]);
  });

  it('eq with null matches null and undefined field values', () => {
    const result = runModuleEvaluation<
      Array<{ jsdoc_quality: string | null }>
    >(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [{ jsdoc_quality: null }, { jsdoc_quality: 'good' }, {}];
      const predicate = { op: 'eq', field: 'jsdoc_quality', value: null };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([{ jsdoc_quality: null }, {}]);
  });

  it('neq excludes matching value', () => {
    const result = runModuleEvaluation<Array<{ arch_layer: string }>>(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [{ arch_layer: 'network' }, { arch_layer: 'mutate' }];
      const predicate = { op: 'neq', field: 'arch_layer', value: 'network' };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([{ arch_layer: 'mutate' }]);
  });

  it('in matches any value in the array', () => {
    const result = runModuleEvaluation<Array<{ export_type: string }>>(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [{ export_type: 'function' }, { export_type: 'class' }, { export_type: 'variable' }];
      const predicate = { op: 'in', field: 'export_type', values: ['function', 'class'] };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([
      { export_type: 'function' },
      { export_type: 'class' },
    ]);
  });

  it('not_in excludes any value in the array', () => {
    const result = runModuleEvaluation<Array<{ export_type: string }>>(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [{ export_type: 'function' }, { export_type: 'class' }, { export_type: 'variable' }];
      const predicate = { op: 'not_in', field: 'export_type', values: ['function', 'class'] };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([{ export_type: 'variable' }]);
  });

  it('gt matches values greater than threshold', () => {
    const result = runModuleEvaluation<Array<{ jsdoc_word_count: number }>>(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [{ jsdoc_word_count: 10 }, { jsdoc_word_count: 3 }];
      const predicate = { op: 'gt', field: 'jsdoc_word_count', value: 5 };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([{ jsdoc_word_count: 10 }]);
  });

  it('gte matches values greater than or equal to threshold', () => {
    const result = runModuleEvaluation<Array<{ jsdoc_word_count: number }>>(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [{ jsdoc_word_count: 5 }, { jsdoc_word_count: 3 }];
      const predicate = { op: 'gte', field: 'jsdoc_word_count', value: 5 };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([{ jsdoc_word_count: 5 }]);
  });

  it('lt matches values less than threshold', () => {
    const result = runModuleEvaluation<Array<{ jsdoc_word_count: number }>>(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [{ jsdoc_word_count: 3 }, { jsdoc_word_count: 10 }];
      const predicate = { op: 'lt', field: 'jsdoc_word_count', value: 5 };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([{ jsdoc_word_count: 3 }]);
  });

  it('lte matches values less than or equal to threshold', () => {
    const result = runModuleEvaluation<Array<{ jsdoc_word_count: number }>>(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [{ jsdoc_word_count: 5 }, { jsdoc_word_count: 10 }];
      const predicate = { op: 'lte', field: 'jsdoc_word_count', value: 5 };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([{ jsdoc_word_count: 5 }]);
  });

  it('like with percent wildcard matches substring', () => {
    const result = runModuleEvaluation<Array<{ module_path: string }>>(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [{ module_path: 'src/architecture/network' }, { module_path: 'src/neat/mutate' }];
      const predicate = { op: 'like', field: 'module_path', value: '%network%' };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([{ module_path: 'src/architecture/network' }]);
  });

  it('like with underscore wildcard matches single character', () => {
    const result = runModuleEvaluation<Array<{ module_path: string }>>(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [{ module_path: 'network' }, { module_path: 'src/network' }];
      const predicate = { op: 'like', field: 'module_path', value: 'net_ork' };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([{ module_path: 'network' }]);
  });

  it('is_null matches null and undefined field values', () => {
    const result = runModuleEvaluation<
      Array<{ jsdoc_quality: string | null }>
    >(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [{ jsdoc_quality: null }, { jsdoc_quality: 'good' }];
      const predicate = { op: 'is_null', field: 'jsdoc_quality' };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([{ jsdoc_quality: null }]);
  });

  it('is_not_null matches non-null field values', () => {
    const result = runModuleEvaluation<
      Array<{ jsdoc_quality: string | null }>
    >(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [{ jsdoc_quality: null }, { jsdoc_quality: 'good' }];
      const predicate = { op: 'is_not_null', field: 'jsdoc_quality' };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([{ jsdoc_quality: 'good' }]);
  });

  it('and requires all predicates to match', () => {
    const result = runModuleEvaluation<
      Array<{ arch_layer: string; export_type: string }>
    >(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [
        { arch_layer: 'network', export_type: 'function' },
        { arch_layer: 'network', export_type: 'class' },
        { arch_layer: 'mutate', export_type: 'function' },
      ];
      const predicate = { op: 'and', predicates: [
        { op: 'eq', field: 'arch_layer', value: 'network' },
        { op: 'eq', field: 'export_type', value: 'function' },
      ]};
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([
      { arch_layer: 'network', export_type: 'function' },
    ]);
  });

  it('or requires any predicate to match', () => {
    const result = runModuleEvaluation<
      Array<{ arch_layer: string; export_type: string }>
    >(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [
        { arch_layer: 'network', export_type: 'function' },
        { arch_layer: 'network', export_type: 'class' },
        { arch_layer: 'mutate', export_type: 'function' },
      ];
      const predicate = { op: 'or', predicates: [
        { op: 'eq', field: 'arch_layer', value: 'network' },
        { op: 'eq', field: 'export_type', value: 'function' },
      ]};
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([
      { arch_layer: 'network', export_type: 'function' },
      { arch_layer: 'network', export_type: 'class' },
      { arch_layer: 'mutate', export_type: 'function' },
    ]);
  });

  it('not negates inner predicate', () => {
    const result = runModuleEvaluation<Array<{ arch_layer: string }>>(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [{ arch_layer: 'network' }, { arch_layer: 'mutate' }];
      const predicate = { op: 'not', predicate: { op: 'eq', field: 'arch_layer', value: 'network' } };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([{ arch_layer: 'mutate' }]);
  });

  it('family field maps from doc_family or family property', () => {
    const result = runModuleEvaluation<
      Array<{ family?: string; doc_family?: string; arch_layer: string }>
    >(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [
        { family: 'ts-source', arch_layer: 'network' },
        { doc_family: 'ts-source', arch_layer: 'mutate' },
        { family: 'plan', arch_layer: 'network' },
      ];
      const predicate = { op: 'eq', field: 'family', value: 'ts-source' };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([
      { family: 'ts-source', arch_layer: 'network' },
      { doc_family: 'ts-source', arch_layer: 'mutate' },
    ]);
  });

  it('file_path field maps from file_path property', () => {
    const result = runModuleEvaluation<
      Array<{ file_path: string; arch_layer: string }>
    >(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [
        { file_path: 'src/neat.ts', arch_layer: 'network' },
        { file_path: 'src/arch.ts', arch_layer: 'mutate' },
      ];
      const predicate = { op: 'eq', field: 'file_path', value: 'src/neat.ts' };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([
      { file_path: 'src/neat.ts', arch_layer: 'network' },
    ]);
  });

  it('metadata sub-object is checked for field values', () => {
    const result = runModuleEvaluation<
      Array<{ metadata?: { jsdoc_quality: string }; jsdoc_quality?: string }>
    >(`
      import { applyPostRetrievalFilter } from './rag-index/metadata-filter.mjs';
      const candidates = [
        { metadata: { jsdoc_quality: 'good' } },
        { jsdoc_quality: 'good' },
        { metadata: { jsdoc_quality: 'weak' } },
      ];
      const predicate = { op: 'eq', field: 'jsdoc_quality', value: 'good' };
      console.log(JSON.stringify(applyPostRetrievalFilter(candidates, predicate)));
    `);
    expect(result).toEqual([
      { metadata: { jsdoc_quality: 'good' } },
      { jsdoc_quality: 'good' },
    ]);
  });
});
