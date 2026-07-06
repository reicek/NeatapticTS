/**
 * @module classify-query.red.test
 * @description Red tests for the rule-based query classifier.
 *
 * Uses runModuleEvaluation to call .mjs module functions from .ts test files,
 * matching the established pattern in semantic-index.red.test.ts.
 */
import { execFileSync } from 'node:child_process';
import path from 'node:path';

// ---------------------------------------------------------------------------
// Types for classification results
// ---------------------------------------------------------------------------

interface ClassificationResult {
  query_class: string;
  confidence: number;
}

interface SearchCorpusResult {
  query_class: string;
  confidence: number;
  alpha: number;
  family: string | null;
  classification_fallback: boolean;
}

interface PatternDetectionResult {
  result: boolean;
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
// Pattern detection helpers
// ---------------------------------------------------------------------------

describe('classify-query pattern detection', () => {
  describe('hasPlanHints', () => {
    it('matches "design of checkpointing"', () => {
      const result = runModuleEvaluation<PatternDetectionResult>(`
        import { hasPlanHints } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify({ result: hasPlanHints('design of checkpointing') }));
      `);
      expect(result.result).toBe(true);
    });

    it('matches "architecture plan"', () => {
      const result = runModuleEvaluation<PatternDetectionResult>(`
        import { hasPlanHints } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify({ result: hasPlanHints('architecture plan') }));
      `);
      expect(result.result).toBe(true);
    });

    it('does not match "network activate method"', () => {
      const result = runModuleEvaluation<PatternDetectionResult>(`
        import { hasPlanHints } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify({ result: hasPlanHints('network activate method') }));
      `);
      expect(result.result).toBe(false);
    });

    it('is case-insensitive when query is pre-lowered', () => {
      // hasPlanHints expects lowercased input; classifyQuery lowercases before calling
      const result = runModuleEvaluation<PatternDetectionResult>(`
        import { hasPlanHints } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify({ result: hasPlanHints('design of the system') }));
      `);
      expect(result.result).toBe(true);
    });
  });

  describe('hasCodeHints', () => {
    it('matches "implementation of crossover"', () => {
      const result = runModuleEvaluation<PatternDetectionResult>(`
        import { hasCodeHints } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify({ result: hasCodeHints('implementation of crossover') }));
      `);
      expect(result.result).toBe(true);
    });

    it('does not match "what is the design"', () => {
      const result = runModuleEvaluation<PatternDetectionResult>(`
        import { hasCodeHints } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify({ result: hasCodeHints('what is the design') }));
      `);
      expect(result.result).toBe(false);
    });
  });

  describe('hasMultiHopIndicators', () => {
    it('matches "activate and then mutate" (and then pattern)', () => {
      const result = runModuleEvaluation<PatternDetectionResult>(`
        import { hasMultiHopIndicators } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify({ result: hasMultiHopIndicators('activate and then mutate') }));
      `);
      expect(result.result).toBe(true);
    });

    it('matches "functions that also call activate" (that also pattern)', () => {
      const result = runModuleEvaluation<PatternDetectionResult>(`
        import { hasMultiHopIndicators } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify({ result: hasMultiHopIndicators('functions that also call activate') }));
      `);
      expect(result.result).toBe(true);
    });

    it('does not match "network activate"', () => {
      const result = runModuleEvaluation<PatternDetectionResult>(`
        import { hasMultiHopIndicators } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify({ result: hasMultiHopIndicators('network activate') }));
      `);
      expect(result.result).toBe(false);
    });
  });

  describe('hasCrossFamilyIndicators', () => {
    it('matches "relationship between crossover and mutation"', () => {
      const result = runModuleEvaluation<PatternDetectionResult>(`
        import { hasCrossFamilyIndicators } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify({ result: hasCrossFamilyIndicators('relationship between crossover and mutation') }));
      `);
      expect(result.result).toBe(true);
    });

    it('does not match "network activate"', () => {
      const result = runModuleEvaluation<PatternDetectionResult>(`
        import { hasCrossFamilyIndicators } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify({ result: hasCrossFamilyIndicators('network activate') }));
      `);
      expect(result.result).toBe(false);
    });
  });

  describe('hasExploratoryHints', () => {
    it('matches "how does the training pipeline work"', () => {
      const result = runModuleEvaluation<PatternDetectionResult>(`
        import { hasExploratoryHints } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify({ result: hasExploratoryHints('how does the training pipeline work') }));
      `);
      expect(result.result).toBe(true);
    });

    it('does not match "network.activate"', () => {
      const result = runModuleEvaluation<PatternDetectionResult>(`
        import { hasExploratoryHints } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify({ result: hasExploratoryHints('network.activate') }));
      `);
      expect(result.result).toBe(false);
    });
  });

  describe('hasFamilyHints', () => {
    it('returns true when plan hints are present', () => {
      const result = runModuleEvaluation<PatternDetectionResult>(`
        import { hasFamilyHints } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify({ result: hasFamilyHints('what is the checkpointing design') }));
      `);
      expect(result.result).toBe(true);
    });

    it('returns true when code hints are present', () => {
      const result = runModuleEvaluation<PatternDetectionResult>(`
        import { hasFamilyHints } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify({ result: hasFamilyHints('implementation of crossover') }));
      `);
      expect(result.result).toBe(true);
    });

    it('does not match "random query" with no family hints', () => {
      const result = runModuleEvaluation<PatternDetectionResult>(`
        import { hasFamilyHints } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify({ result: hasFamilyHints('random query') }));
      `);
      expect(result.result).toBe(false);
    });
  });
});

// ---------------------------------------------------------------------------
// classifyQuery — all 6 classes
// ---------------------------------------------------------------------------

describe('classifyQuery', () => {
  it('classifies short direct queries as simple_lookup with confidence 0.9', () => {
    const result = runModuleEvaluation<ClassificationResult>(`
      import { classifyQuery } from './rag-index/classify-query.mjs';
      const r = classifyQuery('crossover');
      console.log(JSON.stringify(r));
    `);
    expect(result.query_class).toBe('simple_lookup');
    expect(result.confidence).toBe(0.9);
  });

  it('classifies code-specific queries with confidence 0.80', () => {
    const result = runModuleEvaluation<ClassificationResult>(`
      import { classifyQuery } from './rag-index/classify-query.mjs';
      console.log(JSON.stringify(classifyQuery('implementation of crossover in NEAT')));
    `);
    expect(result.query_class).toBe('code_specific');
    expect(result.confidence).toBe(0.8);
  });

  it('classifies plan-specific queries with confidence 0.85', () => {
    const result = runModuleEvaluation<ClassificationResult>(`
      import { classifyQuery } from './rag-index/classify-query.mjs';
      console.log(JSON.stringify(classifyQuery('what is the checkpointing design')));
    `);
    expect(result.query_class).toBe('plan_specific');
    expect(result.confidence).toBe(0.85);
  });

  it('classifies multi-hop queries with confidence 0.75', () => {
    const result = runModuleEvaluation<ClassificationResult>(`
      import { classifyQuery } from './rag-index/classify-query.mjs';
      console.log(JSON.stringify(classifyQuery('functions that also call activate')));
    `);
    expect(result.query_class).toBe('multi_hop');
    expect(result.confidence).toBe(0.75);
  });

  it('classifies cross-boundary queries with confidence 0.70', () => {
    const result = runModuleEvaluation<ClassificationResult>(`
      import { classifyQuery } from './rag-index/classify-query.mjs';
      console.log(JSON.stringify(classifyQuery('relationship between crossover and mutation')));
    `);
    expect(result.query_class).toBe('cross_boundary');
    expect(result.confidence).toBe(0.7);
  });

  it('classifies exploratory queries with confidence 0.65', () => {
    const result = runModuleEvaluation<ClassificationResult>(`
      import { classifyQuery } from './rag-index/classify-query.mjs';
      console.log(JSON.stringify(classifyQuery('how does the training pipeline work')));
    `);
    expect(result.query_class).toBe('exploratory');
    expect(result.confidence).toBe(0.65);
  });

  // -------------------------------------------------------------------------
  // Priority ordering
  // -------------------------------------------------------------------------

  describe('priority ordering', () => {
    it('plan beats code when both hints present', () => {
      // "design" is plan, "implementation" is code — plan wins
      const result = runModuleEvaluation<ClassificationResult>(`
        import { classifyQuery } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify(classifyQuery('design implementation of crossover')));
      `);
      expect(result.query_class).toBe('plan_specific');
    });

    it('code beats multi-hop when both hints present', () => {
      // "implementation" is code, "that also" is multi-hop — code wins
      const result = runModuleEvaluation<ClassificationResult>(`
        import { classifyQuery } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify(classifyQuery('implementation that also calls activate')));
      `);
      expect(result.query_class).toBe('code_specific');
    });

    it('multi-hop beats cross-boundary when both present', () => {
      // "and then" is multi-hop, "relationship" is cross-boundary — multi-hop wins
      const result = runModuleEvaluation<ClassificationResult>(`
        import { classifyQuery } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify(classifyQuery('relationship and then connection between crossover')));
      `);
      expect(result.query_class).toBe('multi_hop');
    });

    it('cross-boundary beats exploratory when both present', () => {
      // "relationship" is cross-boundary, "how does" is exploratory — cross-boundary wins
      const result = runModuleEvaluation<ClassificationResult>(`
        import { classifyQuery } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify(classifyQuery('how does the relationship between crossover and mutation work')));
      `);
      expect(result.query_class).toBe('cross_boundary');
    });
  });

  // -------------------------------------------------------------------------
  // Determinism
  // -------------------------------------------------------------------------

  describe('determinism', () => {
    it('returns identical results on repeated calls', () => {
      const first = runModuleEvaluation<ClassificationResult>(`
        import { classifyQuery } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify(classifyQuery('implementation of crossover in NEAT')));
      `);
      const second = runModuleEvaluation<ClassificationResult>(`
        import { classifyQuery } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify(classifyQuery('implementation of crossover in NEAT')));
      `);
      expect(first).toEqual(second);
    });
  });

  // -------------------------------------------------------------------------
  // Edge cases
  // -------------------------------------------------------------------------

  describe('edge cases', () => {
    it('classifies empty string as simple_lookup via short-query path (1 token < 5)', () => {
      // Empty string becomes [""] after trim().split(/\s+/), length 1 < 5, no family hints
      const result = runModuleEvaluation<ClassificationResult>(`
        import { classifyQuery } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify(classifyQuery('')));
      `);
      expect(result.query_class).toBe('simple_lookup');
      expect(result.confidence).toBe(0.9);
    });

    it('classifies single-word query as simple_lookup', () => {
      const result = runModuleEvaluation<ClassificationResult>(`
        import { classifyQuery } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify(classifyQuery('activate')));
      `);
      expect(result.query_class).toBe('simple_lookup');
    });

    it('classifies long pattern-free query as simple_lookup via fallback (0.50)', () => {
      // 9+ tokens with no family hints → falls through all checks → fallback
      const result = runModuleEvaluation<ClassificationResult>(`
        import { classifyQuery } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify(classifyQuery('the weather is sunny today and I feel great about everything')));
      `);
      expect(result.query_class).toBe('simple_lookup');
      expect(result.confidence).toBe(0.5);
    });
  });
});

// ---------------------------------------------------------------------------
// classifyForSearchCorpus — alpha/family defaults
// ---------------------------------------------------------------------------

describe('classifyForSearchCorpus', () => {
  it('returns per-class alpha for simple_lookup', () => {
    const result = runModuleEvaluation<SearchCorpusResult>(`
      import { classifyForSearchCorpus } from './rag-index/classify-query.mjs';
      console.log(JSON.stringify(classifyForSearchCorpus('crossover')));
    `);
    expect(result.alpha).toBe(0.75);
    expect(result.family).toBeNull();
  });

  it('returns per-class alpha for exploratory', () => {
    const result = runModuleEvaluation<SearchCorpusResult>(`
      import { classifyForSearchCorpus } from './rag-index/classify-query.mjs';
      console.log(JSON.stringify(classifyForSearchCorpus('how does training work')));
    `);
    expect(result.alpha).toBe(0.3);
    expect(result.family).toBeNull();
  });

  it('returns per-class alpha and family for code_specific', () => {
    const result = runModuleEvaluation<SearchCorpusResult>(`
      import { classifyForSearchCorpus } from './rag-index/classify-query.mjs';
      console.log(JSON.stringify(classifyForSearchCorpus('implementation of crossover')));
    `);
    expect(result.alpha).toBe(0.7);
    expect(result.family).toBe('ts-source');
  });

  it('returns per-class alpha and family for plan_specific', () => {
    const result = runModuleEvaluation<SearchCorpusResult>(`
      import { classifyForSearchCorpus } from './rag-index/classify-query.mjs';
      console.log(JSON.stringify(classifyForSearchCorpus('what is the design plan')));
    `);
    expect(result.alpha).toBe(0.7);
    expect(result.family).toBe('plan,completed-plan');
  });

  it('returns null family for classes without default family', () => {
    const result = runModuleEvaluation<SearchCorpusResult>(`
      import { classifyForSearchCorpus } from './rag-index/classify-query.mjs';
      console.log(JSON.stringify(classifyForSearchCorpus('relationship between crossover and mutation')));
    `);
    expect(result.family).toBeNull();
  });

  it('fallback simple_lookup has confidence 0.50 but uses per-class alpha (no degradation)', () => {
    // Long query with no patterns → fallback simple_lookup with confidence 0.50
    // confidence 0.50 is NOT below threshold 0.50, so no degradation occurs
    const result = runModuleEvaluation<SearchCorpusResult>(`
      import { classifyForSearchCorpus } from './rag-index/classify-query.mjs';
      console.log(JSON.stringify(classifyForSearchCorpus('the weather is sunny today and I feel great about everything')));
    `);
    expect(result.query_class).toBe('simple_lookup');
    expect(result.confidence).toBe(0.5);
    // No degradation: alpha comes from EMBEDDED_ALPHA_DEFAULTS['simple_lookup'] = 0.75
    expect(result.alpha).toBe(0.75);
    expect(result.classification_fallback).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Embedded defaults sync with canonical routing table
// ---------------------------------------------------------------------------

describe('embedded defaults sync', () => {
  it('EMBEDDED_ALPHA_DEFAULTS matches canonical DEFAULTS', () => {
    const embedded = runModuleEvaluation<Record<string, number>>(`
      import { EMBEDDED_ALPHA_DEFAULTS } from './rag-index/classify-query.mjs';
      console.log(JSON.stringify(EMBEDDED_ALPHA_DEFAULTS));
    `);
    const canonical = runModuleEvaluation<Record<string, number>>(`
      import { DEFAULTS } from './rag-index/routing-table.mjs';
      console.log(JSON.stringify(DEFAULTS));
    `);
    expect(embedded).toEqual(canonical);
  });

  it('EMBEDDED_FAMILY_DEFAULTS matches canonical ROUTING families', () => {
    const embedded = runModuleEvaluation<Record<string, string | null>>(`
      import { EMBEDDED_FAMILY_DEFAULTS } from './rag-index/classify-query.mjs';
      console.log(JSON.stringify(EMBEDDED_FAMILY_DEFAULTS));
    `);
    const routing = runModuleEvaluation<
      Record<string, { family: string | null }>
    >(`
      import { ROUTING } from './rag-index/routing-table.mjs';
      console.log(JSON.stringify(ROUTING));
    `);
    // Extract family values from ROUTING and compare
    const routingFamilies: Record<string, string | null> = {};
    for (const [key, val] of Object.entries(routing)) {
      routingFamilies[key] = val.family;
    }
    expect(embedded).toEqual(routingFamilies);
  });
});
