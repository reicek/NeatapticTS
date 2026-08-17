import { jest } from '@jest/globals';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import {
  resolveArchLayer,
  countJsdocWords,
  classifyJsdocQuality,
  classifyTestCoverage,
  resolveSourcePathPattern,
  computeCyclomaticComplexity,
  loadCoverageReport,
  enrichChunkMetadata,
  enrichDocumentMetadata,
} from './metadata-enrichment.mjs';

describe('resolveArchLayer', () => {
  it('matches module_path prefix rules', () => {
    expect(resolveArchLayer('src/architecture/network', null)).toBe('network');
    expect(resolveArchLayer('src/neat/neat.ts', null)).toBe('neat');
    expect(resolveArchLayer('src/methods/methods.ts', null)).toBe('methods');
    expect(resolveArchLayer('src/multithreading/worker', null)).toBe('multithreading');
    expect(resolveArchLayer('src/config/config.ts', null)).toBe('config');
  });

  it('returns utils for src/ module_path not matching any rule', () => {
    expect(resolveArchLayer('src/utilities', null)).toBe('utils');
  });

  it('falls back to file_path when module_path is null', () => {
    expect(resolveArchLayer(null, 'src/neat/neat.ts')).toBe('neat');
    expect(resolveArchLayer(null, '.github/skills/test/SKILL.md')).toBe('skill');
    expect(resolveArchLayer(null, '.github/agents/test.agent.md')).toBe('agent');
    expect(resolveArchLayer(null, 'plans/test.plans.md')).toBe('plan');
  });

  it('returns utils for src/ file_path not matching any fallback rule', () => {
    expect(resolveArchLayer(null, 'src/utilities/format.ts')).toBe('utils');
  });

  it('returns doc when both module_path and file_path are null', () => {
    expect(resolveArchLayer(null, null)).toBe('doc');
  });

  it('returns doc for non-src file_path not matching fallback rules', () => {
    expect(resolveArchLayer(null, 'README.md')).toBe('doc');
  });

  it('checks module_path first even when file_path would match fallback', () => {
    expect(resolveArchLayer('src/architecture/network', '.github/skills/test/SKILL.md')).toBe('network');
  });

  it('returns doc when module_path is empty string and file_path is null', () => {
    expect(resolveArchLayer('', null)).toBe('doc');
  });
});

describe('countJsdocWords', () => {
  it('returns 0 for null', () => {
    expect(countJsdocWords(null)).toBe(0);
  });

  it('returns 0 for non-string input', () => {
    expect(countJsdocWords(123)).toBe(0);
  });

  it('returns 0 for empty string', () => {
    expect(countJsdocWords('')).toBe(0);
  });

  it('counts words in simple text', () => {
    expect(countJsdocWords('Activate the network')).toBe(3);
  });

  it('strips leading * from multi-line JSDoc and counts words', () => {
    const jsdoc = '* Activate the network\n* with inputs';
    expect(countJsdocWords(jsdoc)).toBe(5);
  });

  it('counts @param tags as words', () => {
    expect(countJsdocWords('@param input - The input values')).toBe(5);
  });
});

describe('classifyJsdocQuality', () => {
  it('returns none for null', () => {
    expect(classifyJsdocQuality(null)).toBe('none');
  });

  it('returns none for empty string', () => {
    expect(classifyJsdocQuality('')).toBe('none');
  });

  it('returns none for whitespace-only string', () => {
    expect(classifyJsdocQuality('   ')).toBe('none');
  });

  it('returns weak for fewer than 10 words', () => {
    expect(classifyJsdocQuality('Activate')).toBe('weak');
  });

  it('returns adequate for 10+ words without @param and @returns', () => {
    expect(classifyJsdocQuality('Activate the network forward pass with some input values here now')).toBe('adequate');
  });

  it('returns adequate for 10+ words with @param but no @returns', () => {
    expect(classifyJsdocQuality('Activate the network forward pass with some @param input values here now')).toBe('adequate');
  });

  it('returns good for 10+ words with both @param and @returns', () => {
    expect(classifyJsdocQuality('Activate. @param input - values. @returns out values here now')).toBe('good');
  });

  it('returns good for 10+ words with @param and @return (singular)', () => {
    expect(classifyJsdocQuality('Activate. @param input - values. @return out values here now')).toBe('good');
  });
});

describe('classifyTestCoverage', () => {
  it('returns unknown for null coverage report', () => {
    expect(classifyTestCoverage('src/neat.ts', null)).toBe('unknown');
  });

  it('returns unknown for empty filePath', () => {
    expect(classifyTestCoverage('', { 'src/neat.ts': { statements: { pct: 100 } } })).toBe('unknown');
  });

  it('returns unknown for null filePath', () => {
    expect(classifyTestCoverage(null, { 'src/neat.ts': { statements: { pct: 100 } } })).toBe('unknown');
  });

  it('returns unknown when file is not in coverage report', () => {
    expect(classifyTestCoverage('src/missing.ts', { 'src/neat.ts': { statements: { pct: 100 } } })).toBe('unknown');
  });

  it('returns full for 100% coverage', () => {
    expect(classifyTestCoverage('src/neat.ts', { 'src/neat.ts': { statements: { pct: 100 } } })).toBe('full');
  });

  it('returns partial for >0 and <100 coverage', () => {
    expect(classifyTestCoverage('src/neat.ts', { 'src/neat.ts': { statements: { pct: 50 } } })).toBe('partial');
  });

  it('returns none for 0% coverage', () => {
    expect(classifyTestCoverage('src/neat.ts', { 'src/neat.ts': { statements: { pct: 0 } } })).toBe('none');
  });

  it('returns none when statements.pct is missing (defaults to 0)', () => {
    expect(classifyTestCoverage('src/neat.ts', { 'src/neat.ts': {} })).toBe('none');
  });
});

describe('resolveSourcePathPattern', () => {
  it('returns empty string for empty filePath', () => {
    expect(resolveSourcePathPattern('')).toBe('');
  });

  it('returns directory + /** for src/ files', () => {
    expect(resolveSourcePathPattern('src/architecture/network/activate/network.activate.ts')).toBe('src/architecture/network/activate/**');
  });

  it('returns top-level directory + /** for non-src files with 2+ segments', () => {
    expect(resolveSourcePathPattern('.github/skills/coverage-guard/SKILL.md')).toBe('.github/skills/**');
  });

  it('returns directory + /** for single-segment non-src paths', () => {
    expect(resolveSourcePathPattern('README.md')).toBe('**');
  });

  it('handles plans path', () => {
    expect(resolveSourcePathPattern('plans/test-repair.plans.md')).toBe('plans/**');
  });
});

describe('computeCyclomaticComplexity', () => {
  it('returns null for null input', () => {
    expect(computeCyclomaticComplexity(null)).toBeNull();
  });

  it('returns null for non-string input', () => {
    expect(computeCyclomaticComplexity(123)).toBeNull();
  });

  it('returns 1 for simple function with no decision points', () => {
    expect(computeCyclomaticComplexity('function add(a, b) { return a + b; }')).toBe(1);
  });

  it('counts if statements', () => {
    expect(computeCyclomaticComplexity('function max(a, b) { if (a > b) return a; return b; }')).toBe(2);
  });

  it('counts else if and subtracts double-count', () => {
    const code = 'function f(x) { if (x > 0) return 1; else if (x < 0) return -1; return 0; }';
    // if: 2, else if: 1 → total 3, subtract 1 for else-if double count → 2 + 1(base) = 3
    // Actually: if matches 2, else if matches 1 → complexity = 1 + 2 + 1 = 4, subtract 1 = 3
    expect(computeCyclomaticComplexity(code)).toBe(3);
  });

  it('counts for, while, case, catch, &&, ||, ??', () => {
    const code = 'function f() { for (let i = 0; i < 10; i++) { if (i && i || i) { while (i) { switch (i) { case 1: break; } } } } try {} catch (e) {} }';
    const result = computeCyclomaticComplexity(code);
    expect(result).toBeGreaterThan(1);
  });

  it('counts ternary operator', () => {
    const code = 'function f(x) { return x > 0 ? x : -x; }';
    // ternary: 1 → complexity = 1 (base) + 1 = 2
    expect(computeCyclomaticComplexity(code)).toBe(2);
  });

  it('counts nullish coalescing', () => {
    const code = 'function f(x) { return x ?? 0; }';
    expect(computeCyclomaticComplexity(code)).toBe(2);
  });
});

describe('loadCoverageReport', () => {
  let tempDir;

  beforeEach(() => {
    tempDir = mkdtempSync(path.join(tmpdir(), 'meta-enrich-'));
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('returns parsed coverage report when file exists', async () => {
    const coverageDir = path.join(tempDir, 'coverage');
    const fs = await import('node:fs');
    fs.mkdirSync(coverageDir, { recursive: true });
    const report = { 'src/neat.ts': { statements: { pct: 100 } } };
    fs.writeFileSync(path.join(coverageDir, 'coverage-summary.json'), JSON.stringify(report));
    const result = await loadCoverageReport(tempDir);
    expect(result).toEqual(report);
  });

  it('returns null when coverage file is missing', async () => {
    const result = await loadCoverageReport(tempDir);
    expect(result).toBeNull();
  });

  it('returns null when coverage file is invalid JSON', async () => {
    const coverageDir = path.join(tempDir, 'coverage');
    const fs = await import('node:fs');
    fs.mkdirSync(coverageDir, { recursive: true });
    fs.writeFileSync(path.join(coverageDir, 'coverage-summary.json'), 'invalid json{');
    const result = await loadCoverageReport(tempDir);
    expect(result).toBeNull();
  });
});

describe('enrichChunkMetadata', () => {
  it('enriches a ts-source chunk with all fields', () => {
    const result = enrichChunkMetadata({
      jsdoc_text: 'Activate. @param x - input. @returns result value here',
      export_type: 'function',
      module_path: 'src/architecture/network/activate',
      file_path: 'src/architecture/network/activate/network.activate.ts',
      family: 'ts-source',
      body_text: 'function activate(x) { if (x > 0) return x; return 0; }',
    }, null);
    expect(result.arch_layer).toBe('network');
    expect(result.jsdoc_quality).toBe('good');
    expect(result.jsdoc_word_count).toBe(8);
    expect(result.cyclomatic_complexity).toBe(2);
    expect(result.test_coverage).toBe('unknown');
    expect(result.source_path_pattern).toBe('src/architecture/network/activate/**');
  });

  it('sets cyclomatic_complexity to null for non-ts-source family', () => {
    const result = enrichChunkMetadata({
      jsdoc_text: null,
      family: 'plan',
      file_path: 'plans/test.md',
      module_path: null,
    }, null);
    expect(result.cyclomatic_complexity).toBeNull();
    expect(result.test_coverage).toBe('unknown');
  });

  it('sets cyclomatic_complexity to null when body_text is missing', () => {
    const result = enrichChunkMetadata({
      family: 'ts-source',
      file_path: 'src/neat.ts',
      module_path: 'src/neat',
      body_text: null,
    }, null);
    expect(result.cyclomatic_complexity).toBeNull();
  });

  it('classifies test coverage from report for ts-source', () => {
    const coverageReport = { 'src/neat.ts': { statements: { pct: 100 } } };
    const result = enrichChunkMetadata({
      family: 'ts-source',
      file_path: 'src/neat.ts',
      module_path: 'src/neat',
    }, coverageReport);
    expect(result.test_coverage).toBe('full');
  });

  it('handles missing module_path with file_path fallback', () => {
    const result = enrichChunkMetadata({
      family: 'plan',
      file_path: 'plans/test.md',
      module_path: null,
    }, null);
    expect(result.arch_layer).toBe('plan');
  });

  it('handles missing file_path', () => {
    const result = enrichChunkMetadata({
      family: 'plan',
      file_path: null,
      module_path: null,
    }, null);
    expect(result.arch_layer).toBe('doc');
    expect(result.source_path_pattern).toBe('');
  });
});

describe('enrichDocumentMetadata', () => {
  it('enriches a ts-source document', () => {
    const coverageReport = { 'src/neat.ts': { statements: { pct: 50 } } };
    const result = enrichDocumentMetadata({
      file_path: 'src/neat.ts',
      family: 'ts-source',
    }, coverageReport);
    expect(result.arch_layer).toBe('neat');
    expect(result.test_coverage).toBe('partial');
    expect(result.source_path_pattern).toBe('src/**');
  });

  it('returns unknown test_coverage for non-ts-source', () => {
    const result = enrichDocumentMetadata({
      file_path: 'plans/test.md',
      family: 'plan',
    }, null);
    expect(result.test_coverage).toBe('unknown');
    expect(result.arch_layer).toBe('plan');
  });

  it('handles missing file_path', () => {
    const result = enrichDocumentMetadata({
      file_path: null,
      family: 'plan',
    }, null);
    expect(result.arch_layer).toBe('doc');
    expect(result.source_path_pattern).toBe('');
  });
});