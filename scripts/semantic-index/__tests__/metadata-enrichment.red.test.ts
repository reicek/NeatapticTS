/**
 * @module metadata-enrichment.red.test
 * @description Red tests for the metadata enrichment module.
 *
 * Uses runModuleEvaluation to call .mjs module functions from .ts test files,
 * matching the established pattern in classify-query.red.test.ts.
 *
 * Covers resolveArchLayer, classifyJsdocQuality, countJsdocWords,
 * computeCyclomaticComplexity, classifyTestCoverage, resolveSourcePathPattern,
 * enrichChunkMetadata, enrichDocumentMetadata, and loadCoverageReport across
 * nine categories: architectural layer resolution, JSDoc quality classification,
 * JSDoc word counting, cyclomatic complexity, test coverage classification,
 * source path pattern resolution, chunk metadata enrichment,
 * document metadata enrichment, and coverage report loading.
 */
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Types for enrichment results
// ---------------------------------------------------------------------------

interface ArchLayerResult {
  layer: string;
}

interface JsdocQualityResult {
  quality: string;
}

interface WordCountResult {
  count: number;
}

interface ComplexityResult {
  complexity: number | null;
}

interface CoverageResult {
  coverage: string;
}

interface PathPatternResult {
  pattern: string;
}

interface ChunkMetadataResult {
  arch_layer: string;
  jsdoc_quality: string;
  jsdoc_word_count: number;
  cyclomatic_complexity: number | null;
  test_coverage: string;
  source_path_pattern: string;
}

interface DocumentMetadataResult {
  arch_layer: string;
  test_coverage: string;
  source_path_pattern: string;
}

interface CoverageReportResult {
  report: Record<string, { statements: { pct: number } }> | null;
}

// ---------------------------------------------------------------------------
// Helper: evaluate .mjs modules via subprocess (matching established pattern)
// ---------------------------------------------------------------------------

const runModuleEvaluation = <Result>(source: string): Result => {
  const output = execFileSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: path.resolve(__dirname, '..', '..', '..'),
      encoding: 'utf8',
    },
  );
  return JSON.parse(output) as Result;
};

// ---------------------------------------------------------------------------
// resolveArchLayer
// ---------------------------------------------------------------------------

describe('resolveArchLayer', () => {
  it('resolves network layer from module path', () => {
    const result = runModuleEvaluation<ArchLayerResult>(`
      import { resolveArchLayer } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ layer: resolveArchLayer('src/architecture/network/activate', null) }));
    `);
    expect(result.layer).toBe('network');
  });

  it('resolves neat layer from module path', () => {
    const result = runModuleEvaluation<ArchLayerResult>(`
      import { resolveArchLayer } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ layer: resolveArchLayer('src/neat/mutation', null) }));
    `);
    expect(result.layer).toBe('neat');
  });

  it('resolves methods layer from module path', () => {
    const result = runModuleEvaluation<ArchLayerResult>(`
      import { resolveArchLayer } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ layer: resolveArchLayer('src/methods/activation', null) }));
    `);
    expect(result.layer).toBe('methods');
  });

  it('resolves multithreading layer from module path', () => {
    const result = runModuleEvaluation<ArchLayerResult>(`
      import { resolveArchLayer } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ layer: resolveArchLayer('src/multithreading/worker', null) }));
    `);
    expect(result.layer).toBe('multithreading');
  });

  it('resolves config layer from module path', () => {
    const result = runModuleEvaluation<ArchLayerResult>(`
      import { resolveArchLayer } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ layer: resolveArchLayer('src/config/settings', null) }));
    `);
    expect(result.layer).toBe('config');
  });

  it('resolves utils layer from unmatched src/ module path', () => {
    const result = runModuleEvaluation<ArchLayerResult>(`
      import { resolveArchLayer } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ layer: resolveArchLayer('src/utilities/format', null) }));
    `);
    expect(result.layer).toBe('utils');
  });

  it('resolves skill layer from .github/skills/ file path', () => {
    const result = runModuleEvaluation<ArchLayerResult>(`
      import { resolveArchLayer } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ layer: resolveArchLayer(null, '.github/skills/coverage-guard/SKILL.md') }));
    `);
    expect(result.layer).toBe('skill');
  });

  it('resolves agent layer from .github/agents/ file path', () => {
    const result = runModuleEvaluation<ArchLayerResult>(`
      import { resolveArchLayer } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ layer: resolveArchLayer(null, '.github/agents/04-implementing.agent.md') }));
    `);
    expect(result.layer).toBe('agent');
  });

  it('resolves plan layer from plans/ file path', () => {
    const result = runModuleEvaluation<ArchLayerResult>(`
      import { resolveArchLayer } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ layer: resolveArchLayer(null, 'plans/test-repair.plans.md') }));
    `);
    expect(result.layer).toBe('plan');
  });

  it('resolves doc layer from unmatched non-src file path', () => {
    const result = runModuleEvaluation<ArchLayerResult>(`
      import { resolveArchLayer } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ layer: resolveArchLayer(null, 'README.md') }));
    `);
    expect(result.layer).toBe('doc');
  });

  it('resolves doc layer when both paths are null', () => {
    const result = runModuleEvaluation<ArchLayerResult>(`
      import { resolveArchLayer } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ layer: resolveArchLayer(null, null) }));
    `);
    expect(result.layer).toBe('doc');
  });

  it('resolves utils layer when module path is null but file path starts with src/', () => {
    // When modulePath is null, the function falls back to filePath.
    // The fallback only checks .github/skills/, .github/agents/, plans/ —
    // any src/ path that reaches the fallback returns 'utils' regardless of
    // which architectural layer the file actually belongs to.
    const result = runModuleEvaluation<ArchLayerResult>(`
      import { resolveArchLayer } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ layer: resolveArchLayer(null, 'src/architecture/network/activate/network.activate.ts') }));
    `);
    expect(result.layer).toBe('utils');
  });
});

// ---------------------------------------------------------------------------
// classifyJsdocQuality
// ---------------------------------------------------------------------------

describe('classifyJsdocQuality', () => {
  it('returns none for null input', () => {
    const result = runModuleEvaluation<JsdocQualityResult>(`
      import { classifyJsdocQuality } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ quality: classifyJsdocQuality(null) }));
    `);
    expect(result.quality).toBe('none');
  });

  it('returns none for empty string', () => {
    const result = runModuleEvaluation<JsdocQualityResult>(`
      import { classifyJsdocQuality } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ quality: classifyJsdocQuality('') }));
    `);
    expect(result.quality).toBe('none');
  });

  it('returns none for whitespace-only string', () => {
    const result = runModuleEvaluation<JsdocQualityResult>(`
      import { classifyJsdocQuality } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ quality: classifyJsdocQuality('   ') }));
    `);
    expect(result.quality).toBe('none');
  });

  it('returns weak for short text with fewer than 10 words', () => {
    const result = runModuleEvaluation<JsdocQualityResult>(`
      import { classifyJsdocQuality } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ quality: classifyJsdocQuality('Activate the network') }));
    `);
    expect(result.quality).toBe('weak');
  });

  it('returns adequate for text with 10 or more words without @param and @returns', () => {
    const result = runModuleEvaluation<JsdocQualityResult>(`
      import { classifyJsdocQuality } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ quality: classifyJsdocQuality('Activate the network forward pass with given input values and produce output') }));
    `);
    expect(result.quality).toBe('adequate');
  });

  it('returns adequate for text with @param only (missing @returns)', () => {
    const result = runModuleEvaluation<JsdocQualityResult>(`
      import { classifyJsdocQuality } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ quality: classifyJsdocQuality('Activate the network forward pass. @param input - The input values to process') }));
    `);
    expect(result.quality).toBe('adequate');
  });

  it('returns good for text with both @param and @returns', () => {
    const result = runModuleEvaluation<JsdocQualityResult>(`
      import { classifyJsdocQuality } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ quality: classifyJsdocQuality('Activate the network forward pass. @param input - The input values. @returns The output values from the network activation') }));
    `);
    expect(result.quality).toBe('good');
  });
});

// ---------------------------------------------------------------------------
// countJsdocWords
// ---------------------------------------------------------------------------

describe('countJsdocWords', () => {
  it('returns 0 for null input', () => {
    const result = runModuleEvaluation<WordCountResult>(`
      import { countJsdocWords } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ count: countJsdocWords(null) }));
    `);
    expect(result.count).toBe(0);
  });

  it('returns 0 for empty string', () => {
    const result = runModuleEvaluation<WordCountResult>(`
      import { countJsdocWords } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ count: countJsdocWords('') }));
    `);
    expect(result.count).toBe(0);
  });

  it('returns correct count for multi-word text', () => {
    // "Activate the network forward pass" = 5 words
    const result = runModuleEvaluation<WordCountResult>(`
      import { countJsdocWords } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ count: countJsdocWords('Activate the network forward pass') }));
    `);
    expect(result.count).toBe(5);
  });

  it('returns correct count for multi-line JSDoc with leading asterisks', () => {
    // After stripping leading * and whitespace: "Activate the network" = 3 words
    const result = runModuleEvaluation<WordCountResult>(`
      import { countJsdocWords } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ count: countJsdocWords('* Activate the\\n * network') }));
    `);
    expect(result.count).toBe(3);
  });

  it('includes @param and @returns tags in word count', () => {
    // "@param input - The input values @returns The output" = 9 words
    // (hyphen "-" is counted as a word token)
    const result = runModuleEvaluation<WordCountResult>(`
      import { countJsdocWords } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ count: countJsdocWords('@param input - The input values @returns The output') }));
    `);
    expect(result.count).toBe(9);
  });
});

// ---------------------------------------------------------------------------
// computeCyclomaticComplexity
// ---------------------------------------------------------------------------

describe('computeCyclomaticComplexity', () => {
  it('returns null for null input', () => {
    const result = runModuleEvaluation<ComplexityResult>(`
      import { computeCyclomaticComplexity } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ complexity: computeCyclomaticComplexity(null) }));
    `);
    expect(result.complexity).toBeNull();
  });

  it('returns 1 for simple function with no decision points', () => {
    const result = runModuleEvaluation<ComplexityResult>(`
      import { computeCyclomaticComplexity } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ complexity: computeCyclomaticComplexity('function add(a, b) { return a + b; }') }));
    `);
    expect(result.complexity).toBe(1);
  });

  it('returns 2 for function with single if statement', () => {
    // Base 1 + if (1) = 2
    const result = runModuleEvaluation<ComplexityResult>(`
      import { computeCyclomaticComplexity } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ complexity: computeCyclomaticComplexity('function max(a, b) { if (a > b) { return a; } return b; }') }));
    `);
    expect(result.complexity).toBe(2);
  });

  it('subtracts double-counted else-if patterns', () => {
    // Base 1 + if (1) + else if (1) + else if (1) = 3 decision points, but:
    // "if" regex matches all 3 if occurrences, "else if" regex matches 2 else-if occurrences
    // So: base 1 + 3 (if) + 2 (else if) = 6, minus 2 (double-counted else if) = 4
    const result = runModuleEvaluation<ComplexityResult>(`
      import { computeCyclomaticComplexity } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ complexity: computeCyclomaticComplexity('function classify(x) { if (x > 10) return 1; else if (x > 5) return 2; else if (x > 0) return 3; return 0; }') }));
    `);
    expect(result.complexity).toBe(4);
  });

  it('returns correct count for ternary operator', () => {
    // Base 1 + ternary (1) = 2
    const result = runModuleEvaluation<ComplexityResult>(`
      import { computeCyclomaticComplexity } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ complexity: computeCyclomaticComplexity('function getLabel(flag) { return flag ? "yes" : "no"; }') }));
    `);
    expect(result.complexity).toBe(2);
  });

  it('returns correct count for && operator', () => {
    // Base 1 + && (1) = 2
    const result = runModuleEvaluation<ComplexityResult>(`
      import { computeCyclomaticComplexity } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ complexity: computeCyclomaticComplexity('function isValid(x) { return x && x.length > 0; }') }));
    `);
    expect(result.complexity).toBe(2);
  });

  it('returns null for non-string input', () => {
    const result = runModuleEvaluation<ComplexityResult>(`
      import { computeCyclomaticComplexity } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ complexity: computeCyclomaticComplexity(42) }));
    `);
    expect(result.complexity).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// classifyTestCoverage
// ---------------------------------------------------------------------------

describe('classifyTestCoverage', () => {
  it('returns full for 100% statement coverage', () => {
    const result = runModuleEvaluation<CoverageResult>(`
      import { classifyTestCoverage } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ coverage: classifyTestCoverage('src/neat/neat.ts', { 'src/neat/neat.ts': { statements: { pct: 100 } } }) }));
    `);
    expect(result.coverage).toBe('full');
  });

  it('returns partial for greater than 0% and less than 100% statement coverage', () => {
    const result = runModuleEvaluation<CoverageResult>(`
      import { classifyTestCoverage } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ coverage: classifyTestCoverage('src/neat/neat.ts', { 'src/neat/neat.ts': { statements: { pct: 85 } } }) }));
    `);
    expect(result.coverage).toBe('partial');
  });

  it('returns none for 0% statement coverage', () => {
    const result = runModuleEvaluation<CoverageResult>(`
      import { classifyTestCoverage } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ coverage: classifyTestCoverage('src/neat/neat.ts', { 'src/neat/neat.ts': { statements: { pct: 0 } } }) }));
    `);
    expect(result.coverage).toBe('none');
  });

  it('returns unknown for null coverage report', () => {
    const result = runModuleEvaluation<CoverageResult>(`
      import { classifyTestCoverage } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ coverage: classifyTestCoverage('src/neat/neat.ts', null) }));
    `);
    expect(result.coverage).toBe('unknown');
  });

  it('returns unknown for file not present in coverage report', () => {
    const result = runModuleEvaluation<CoverageResult>(`
      import { classifyTestCoverage } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ coverage: classifyTestCoverage('src/missing/file.ts', { 'src/neat/neat.ts': { statements: { pct: 100 } } }) }));
    `);
    expect(result.coverage).toBe('unknown');
  });

  it('returns unknown for null file path', () => {
    const result = runModuleEvaluation<CoverageResult>(`
      import { classifyTestCoverage } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ coverage: classifyTestCoverage(null, { 'src/neat/neat.ts': { statements: { pct: 100 } } }) }));
    `);
    expect(result.coverage).toBe('unknown');
  });

  it('returns unknown for empty file path', () => {
    const result = runModuleEvaluation<CoverageResult>(`
      import { classifyTestCoverage } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ coverage: classifyTestCoverage('', { 'src/neat/neat.ts': { statements: { pct: 100 } } }) }));
    `);
    expect(result.coverage).toBe('unknown');
  });
});

// ---------------------------------------------------------------------------
// resolveSourcePathPattern
// ---------------------------------------------------------------------------

describe('resolveSourcePathPattern', () => {
  it('produces directory/** for src/ paths', () => {
    const result = runModuleEvaluation<PathPatternResult>(`
      import { resolveSourcePathPattern } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ pattern: resolveSourcePathPattern('src/architecture/network/activate/network.activate.ts') }));
    `);
    expect(result.pattern).toBe('src/architecture/network/activate/**');
  });

  it('produces .github/skills/** for .github/skills/ path', () => {
    const result = runModuleEvaluation<PathPatternResult>(`
      import { resolveSourcePathPattern } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ pattern: resolveSourcePathPattern('.github/skills/coverage-guard/SKILL.md') }));
    `);
    expect(result.pattern).toBe('.github/skills/**');
  });

  it('produces plans/** for plans/ path', () => {
    const result = runModuleEvaluation<PathPatternResult>(`
      import { resolveSourcePathPattern } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ pattern: resolveSourcePathPattern('plans/test-repair.plans.md') }));
    `);
    expect(result.pattern).toBe('plans/**');
  });

  it('produces empty string for null input', () => {
    const result = runModuleEvaluation<PathPatternResult>(`
      import { resolveSourcePathPattern } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ pattern: resolveSourcePathPattern(null) }));
    `);
    expect(result.pattern).toBe('');
  });

  it('produces empty string for empty string input', () => {
    const result = runModuleEvaluation<PathPatternResult>(`
      import { resolveSourcePathPattern } from './scripts/semantic-index/metadata-enrichment.mjs';
      console.log(JSON.stringify({ pattern: resolveSourcePathPattern('') }));
    `);
    expect(result.pattern).toBe('');
  });
});

// ---------------------------------------------------------------------------
// enrichChunkMetadata
// ---------------------------------------------------------------------------

describe('enrichChunkMetadata', () => {
  it('returns all 6 metadata fields', () => {
    const chunk = {
      jsdoc_text:
        'Activate the network forward pass. @param input - The input values. @returns The output values.',
      export_type: 'function',
      module_path: 'src/architecture/network/activate',
      file_path: 'src/architecture/network/activate/network.activate.ts',
      family: 'ts-source',
      body_text: 'function activate(input) { return input; }',
    };
    const result = runModuleEvaluation<ChunkMetadataResult>(`
      import { enrichChunkMetadata } from './scripts/semantic-index/metadata-enrichment.mjs';
      const chunk = ${JSON.stringify(chunk)};
      console.log(JSON.stringify(enrichChunkMetadata(chunk, null)));
    `);
    expect(result).toHaveProperty('arch_layer');
    expect(result).toHaveProperty('jsdoc_quality');
    expect(result).toHaveProperty('jsdoc_word_count');
    expect(result).toHaveProperty('cyclomatic_complexity');
    expect(result).toHaveProperty('test_coverage');
    expect(result).toHaveProperty('source_path_pattern');
  });

  it('computes arch_layer from module_path', () => {
    const chunk = {
      jsdoc_text: null,
      export_type: 'function',
      module_path: 'src/architecture/network/activate',
      file_path: 'src/architecture/network/activate/network.activate.ts',
      family: 'ts-source',
      body_text: null,
    };
    const result = runModuleEvaluation<ChunkMetadataResult>(`
      import { enrichChunkMetadata } from './scripts/semantic-index/metadata-enrichment.mjs';
      const chunk = ${JSON.stringify(chunk)};
      console.log(JSON.stringify(enrichChunkMetadata(chunk, null)));
    `);
    expect(result.arch_layer).toBe('network');
  });

  it('computes jsdoc_quality from jsdoc_text', () => {
    const chunk = {
      jsdoc_text:
        'Activate the network forward pass. @param input - The input values. @returns The output values.',
      export_type: 'function',
      module_path: 'src/architecture/network/activate',
      file_path: 'src/architecture/network/activate/network.activate.ts',
      family: 'ts-source',
      body_text: null,
    };
    const result = runModuleEvaluation<ChunkMetadataResult>(`
      import { enrichChunkMetadata } from './scripts/semantic-index/metadata-enrichment.mjs';
      const chunk = ${JSON.stringify(chunk)};
      console.log(JSON.stringify(enrichChunkMetadata(chunk, null)));
    `);
    expect(result.jsdoc_quality).toBe('good');
  });

  it('computes jsdoc_word_count from jsdoc_text', () => {
    // "Activate the network forward pass" = 5 words (weak, < 10)
    const chunk = {
      jsdoc_text: 'Activate the network forward pass',
      export_type: 'function',
      module_path: 'src/neat/mutation',
      file_path: 'src/neat/mutation/neat.mutation.ts',
      family: 'ts-source',
      body_text: null,
    };
    const result = runModuleEvaluation<ChunkMetadataResult>(`
      import { enrichChunkMetadata } from './scripts/semantic-index/metadata-enrichment.mjs';
      const chunk = ${JSON.stringify(chunk)};
      console.log(JSON.stringify(enrichChunkMetadata(chunk, null)));
    `);
    expect(result.jsdoc_word_count).toBe(5);
  });

  it('computes cyclomatic_complexity from body_text for ts-source family', () => {
    // "function add(a, b) { return a + b; }" → complexity 1 (no decision points)
    const chunk = {
      jsdoc_text: null,
      export_type: 'function',
      module_path: 'src/methods/activation',
      file_path: 'src/methods/activation/methods.activation.ts',
      family: 'ts-source',
      body_text: 'function add(a, b) { return a + b; }',
    };
    const result = runModuleEvaluation<ChunkMetadataResult>(`
      import { enrichChunkMetadata } from './scripts/semantic-index/metadata-enrichment.mjs';
      const chunk = ${JSON.stringify(chunk)};
      console.log(JSON.stringify(enrichChunkMetadata(chunk, null)));
    `);
    expect(result.cyclomatic_complexity).toBe(1);
  });

  it('sets cyclomatic_complexity to null for non-ts-source family', () => {
    const chunk = {
      jsdoc_text: 'Plan description for checkpointing persistence.',
      export_type: null,
      module_path: null,
      file_path: 'plans/test-repair.plans.md',
      family: 'plan',
      body_text: 'Some body text',
    };
    const result = runModuleEvaluation<ChunkMetadataResult>(`
      import { enrichChunkMetadata } from './scripts/semantic-index/metadata-enrichment.mjs';
      const chunk = ${JSON.stringify(chunk)};
      console.log(JSON.stringify(enrichChunkMetadata(chunk, null)));
    `);
    expect(result.cyclomatic_complexity).toBeNull();
  });

  it('computes test_coverage from coverage report for ts-source family', () => {
    const chunk = {
      jsdoc_text: null,
      export_type: 'function',
      module_path: 'src/neat/mutation',
      file_path: 'src/neat/mutation/neat.mutation.ts',
      family: 'ts-source',
      body_text: null,
    };
    const coverageReport = {
      'src/neat/mutation/neat.mutation.ts': { statements: { pct: 100 } },
    };
    const result = runModuleEvaluation<ChunkMetadataResult>(`
      import { enrichChunkMetadata } from './scripts/semantic-index/metadata-enrichment.mjs';
      const chunk = ${JSON.stringify(chunk)};
      const report = ${JSON.stringify(coverageReport)};
      console.log(JSON.stringify(enrichChunkMetadata(chunk, report)));
    `);
    expect(result.test_coverage).toBe('full');
  });

  it('sets test_coverage to unknown for non-ts-source family', () => {
    const chunk = {
      jsdoc_text: 'Skill description.',
      export_type: null,
      module_path: null,
      file_path: '.github/skills/coverage-guard/SKILL.md',
      family: 'skill',
      body_text: null,
    };
    const coverageReport = {
      '.github/skills/coverage-guard/SKILL.md': { statements: { pct: 100 } },
    };
    const result = runModuleEvaluation<ChunkMetadataResult>(`
      import { enrichChunkMetadata } from './scripts/semantic-index/metadata-enrichment.mjs';
      const chunk = ${JSON.stringify(chunk)};
      const report = ${JSON.stringify(coverageReport)};
      console.log(JSON.stringify(enrichChunkMetadata(chunk, report)));
    `);
    expect(result.test_coverage).toBe('unknown');
  });

  it('computes source_path_pattern from file_path', () => {
    const chunk = {
      jsdoc_text: null,
      export_type: 'function',
      module_path: 'src/neat/mutation',
      file_path: 'src/neat/mutation/neat.mutation.ts',
      family: 'ts-source',
      body_text: null,
    };
    const result = runModuleEvaluation<ChunkMetadataResult>(`
      import { enrichChunkMetadata } from './scripts/semantic-index/metadata-enrichment.mjs';
      const chunk = ${JSON.stringify(chunk)};
      console.log(JSON.stringify(enrichChunkMetadata(chunk, null)));
    `);
    expect(result.source_path_pattern).toBe('src/neat/mutation/**');
  });
});

// ---------------------------------------------------------------------------
// enrichDocumentMetadata
// ---------------------------------------------------------------------------

describe('enrichDocumentMetadata', () => {
  it('returns arch_layer, test_coverage, and source_path_pattern', () => {
    const document = {
      file_path: 'src/architecture/network/activate/network.activate.ts',
      family: 'ts-source',
    };
    const result = runModuleEvaluation<DocumentMetadataResult>(`
      import { enrichDocumentMetadata } from './scripts/semantic-index/metadata-enrichment.mjs';
      const doc = ${JSON.stringify(document)};
      console.log(JSON.stringify(enrichDocumentMetadata(doc, null)));
    `);
    expect(result).toHaveProperty('arch_layer');
    expect(result).toHaveProperty('test_coverage');
    expect(result).toHaveProperty('source_path_pattern');
  });

  it('computes arch_layer from file_path via fallback (src/ paths resolve to utils)', () => {
    // enrichDocumentMetadata calls resolveArchLayer(null, filePath), which only
    // checks fallback rules (.github/skills/, .github/agents/, plans/) for file paths.
    // Any src/ file path that reaches the fallback returns 'utils'.
    const document = {
      file_path: 'src/neat/mutation/neat.mutation.ts',
      family: 'ts-source',
    };
    const result = runModuleEvaluation<DocumentMetadataResult>(`
      import { enrichDocumentMetadata } from './scripts/semantic-index/metadata-enrichment.mjs';
      const doc = ${JSON.stringify(document)};
      console.log(JSON.stringify(enrichDocumentMetadata(doc, null)));
    `);
    expect(result.arch_layer).toBe('utils');
  });

  it('computes test_coverage for ts-source family', () => {
    const document = {
      file_path: 'src/neat/mutation/neat.mutation.ts',
      family: 'ts-source',
    };
    const coverageReport = {
      'src/neat/mutation/neat.mutation.ts': { statements: { pct: 85 } },
    };
    const result = runModuleEvaluation<DocumentMetadataResult>(`
      import { enrichDocumentMetadata } from './scripts/semantic-index/metadata-enrichment.mjs';
      const doc = ${JSON.stringify(document)};
      const report = ${JSON.stringify(coverageReport)};
      console.log(JSON.stringify(enrichDocumentMetadata(doc, report)));
    `);
    expect(result.test_coverage).toBe('partial');
  });

  it('sets test_coverage to unknown for non-ts-source family', () => {
    const document = {
      file_path: 'plans/test-repair.plans.md',
      family: 'plan',
    };
    const result = runModuleEvaluation<DocumentMetadataResult>(`
      import { enrichDocumentMetadata } from './scripts/semantic-index/metadata-enrichment.mjs';
      const doc = ${JSON.stringify(document)};
      console.log(JSON.stringify(enrichDocumentMetadata(doc, null)));
    `);
    expect(result.test_coverage).toBe('unknown');
  });
});

// ---------------------------------------------------------------------------
// loadCoverageReport
// ---------------------------------------------------------------------------

describe('loadCoverageReport', () => {
  it('returns null when coverage file does not exist', async () => {
    const result = runModuleEvaluation<CoverageReportResult>(`
      import { loadCoverageReport } from './scripts/semantic-index/metadata-enrichment.mjs';
      const report = await loadCoverageReport('/nonexistent/path/that/does/not/exist');
      console.log(JSON.stringify({ report }));
    `);
    expect(result.report).toBeNull();
  });
});
