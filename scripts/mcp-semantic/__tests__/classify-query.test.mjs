/**
 * @module classify-query.test
 * @description Coverage tests for rag-index/classify-query.mjs — pure-function query classifier.
 */

import {
  hasPlanHints,
  hasCodeHints,
  hasCodeIdentifiers,
  hasMultiHopIndicators,
  hasCrossFamilyIndicators,
  hasExploratoryHints,
  hasFamilyHints,
  classifyQuery,
  classifyForSearchCorpus,
  EMBEDDED_ALPHA_DEFAULTS,
  EMBEDDED_FAMILY_DEFAULTS,
} from '../../../rag-index/classify-query.mjs';

// ---------------------------------------------------------------------------
// hasPlanHints
// ---------------------------------------------------------------------------

describe('classify-query: hasPlanHints', () => {
  it('returns true for plan keywords', () => {
    expect(hasPlanHints('what is the checkpointing design')).toBe(true);
    expect(hasPlanHints('show me the architecture plan')).toBe(true);
    expect(hasPlanHints('roadmap for next sprint')).toBe(true);
    expect(hasPlanHints('decision record')).toBe(true);
    expect(hasPlanHints('specification document')).toBe(true);
  });

  it('returns false for non-plan queries', () => {
    expect(hasPlanHints('how does crossover work')).toBe(false);
    expect(hasPlanHints('network activation')).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// hasCodeHints
// ---------------------------------------------------------------------------

describe('classify-query: hasCodeHints', () => {
  it('returns true for code keywords', () => {
    expect(hasCodeHints('implementation of crossover')).toBe(true);
    expect(hasCodeHints('code for the network module')).toBe(true);
    expect(hasCodeHints('source of the activation function')).toBe(true);
    expect(hasCodeHints('how to implement a neuron')).toBe(true);
    expect(hasCodeHints('function body of activate')).toBe(true);
    expect(hasCodeHints('typescript types')).toBe(true);
  });

  it('returns false for non-code queries', () => {
    expect(hasCodeHints('how does crossover work')).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// hasCodeIdentifiers
// ---------------------------------------------------------------------------

describe('classify-query: hasCodeIdentifiers', () => {
  it('returns true for dotted identifiers', () => {
    expect(hasCodeIdentifiers('network.activate')).toBe(true);
  });

  it('returns true for snake_case identifiers', () => {
    expect(hasCodeIdentifiers('snake_case_function')).toBe(true);
  });

  it('returns true for file-extension identifiers', () => {
    expect(hasCodeIdentifiers('network.ts')).toBe(true);
  });

  it('returns true for camelCase identifiers', () => {
    expect(hasCodeIdentifiers('findTheNEATSelectionCode')).toBe(true);
  });

  it('returns false for plain words', () => {
    expect(hasCodeIdentifiers('how does NEAT work')).toBe(false);
    expect(hasCodeIdentifiers('crossover')).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// hasMultiHopIndicators
// ---------------------------------------------------------------------------

describe('classify-query: hasMultiHopIndicators', () => {
  it('returns true for multi-hop patterns', () => {
    expect(hasMultiHopIndicators('functions that also call activate')).toBe(true);
    expect(hasMultiHopIndicators('which also uses slab')).toBe(true);
    expect(hasMultiHopIndicators('call activate and then return')).toBe(true);
    expect(hasMultiHopIndicators('call activate that returns')).toBe(true);
    expect(hasMultiHopIndicators('where also used')).toBe(true);
  });

  it('returns false for non-multi-hop queries', () => {
    expect(hasMultiHopIndicators('what is crossover')).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// hasCrossFamilyIndicators
// ---------------------------------------------------------------------------

describe('classify-query: hasCrossFamilyIndicators', () => {
  it('returns true for cross-family keywords', () => {
    expect(hasCrossFamilyIndicators('relationship between crossover and mutation')).toBe(true);
    expect(hasCrossFamilyIndicators('connection between modules')).toBe(true);
    expect(hasCrossFamilyIndicators('between two systems')).toBe(true);
  });

  it('returns false for non-cross-family queries', () => {
    expect(hasCrossFamilyIndicators('how does crossover work')).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// hasExploratoryHints
// ---------------------------------------------------------------------------

describe('classify-query: hasExploratoryHints', () => {
  it('returns true for exploratory keywords', () => {
    expect(hasExploratoryHints('how does the training pipeline work')).toBe(true);
    expect(hasExploratoryHints('explain the mutation algorithm')).toBe(true);
    expect(hasExploratoryHints('overview of the NEAT system')).toBe(true);
    expect(hasExploratoryHints('describe the selection process')).toBe(true);
    expect(hasExploratoryHints('what is NEAT')).toBe(true);
    expect(hasExploratoryHints('tell me about crossover')).toBe(true);
    expect(hasExploratoryHints('how do neurons work')).toBe(true);
  });

  it('returns false for non-exploratory queries', () => {
    expect(hasExploratoryHints('network.activate')).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// hasFamilyHints
// ---------------------------------------------------------------------------

describe('classify-query: hasFamilyHints', () => {
  it('returns true when any family hint is present', () => {
    expect(hasFamilyHints('what is the checkpointing design')).toBe(true);
    expect(hasFamilyHints('implementation of crossover')).toBe(true);
    expect(hasFamilyHints('how does the training work')).toBe(true);
    expect(hasFamilyHints('relationship between modules')).toBe(true);
    expect(hasFamilyHints('functions that also call activate')).toBe(true);
  });

  it('returns false when no family hints are present', () => {
    expect(hasFamilyHints('short api call')).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// classifyQuery
// ---------------------------------------------------------------------------

describe('classify-query: classifyQuery', () => {
  it('classifies short queries as simple_lookup', () => {
    const result = classifyQuery('crossover');
    expect(result.query_class).toBe('simple_lookup');
    expect(result.confidence).toBe(0.9);
    expect(result.hints.short_query).toBe(true);
  });

  it('classifies plan-specific queries', () => {
    const result = classifyQuery('what is the checkpointing design');
    expect(result.query_class).toBe('plan_specific');
    expect(result.confidence).toBe(0.85);
    expect(result.hints.family_filter).toBe('plan,completed-plan');
  });

  it('classifies code-identifier queries as code_specific', () => {
    const result = classifyQuery('network.activate');
    expect(result.query_class).toBe('code_specific');
    expect(result.confidence).toBe(0.85);
    expect(result.hints.family_filter).toBe('ts-source');
  });

  it('classifies code-hint queries as code_specific', () => {
    const result = classifyQuery('implementation of crossover in NEAT');
    expect(result.query_class).toBe('code_specific');
    expect(result.confidence).toBe(0.8);
    expect(result.hints.family_filter).toBe('ts-source');
  });

  it('classifies multi-hop queries', () => {
    const result = classifyQuery('functions that call activate and also use slab');
    expect(result.query_class).toBe('multi_hop');
    expect(result.confidence).toBe(0.75);
    expect(result.hints.multi_hop).toBe(true);
  });

  it('classifies cross-boundary queries', () => {
    const result = classifyQuery('relationship between crossover and mutation');
    expect(result.query_class).toBe('cross_boundary');
    expect(result.confidence).toBe(0.7);
    expect(result.hints.multi_family).toBe(true);
  });

  it('classifies exploratory queries', () => {
    const result = classifyQuery('how does the training pipeline work');
    expect(result.query_class).toBe('exploratory');
    expect(result.confidence).toBe(0.65);
    expect(result.hints.broad_retrieval).toBe(true);
  });

  it('falls back to simple_lookup for unmatched long queries', () => {
    const result = classifyQuery('the quick brown fox jumps over the lazy dog');
    expect(result.query_class).toBe('simple_lookup');
    expect(result.confidence).toBe(0.5);
    expect(result.hints.fallback).toBe(true);
  });

  it('does not classify short query as simple_lookup when it has family hints', () => {
    const result = classifyQuery('design');
    expect(result.query_class).not.toBe('simple_lookup');
  });

  it('does not classify short query as simple_lookup when it has code identifiers', () => {
    const result = classifyQuery('a.b');
    expect(result.query_class).not.toBe('simple_lookup');
  });
});

// ---------------------------------------------------------------------------
// classifyForSearchCorpus
// ---------------------------------------------------------------------------

describe('classify-query: classifyForSearchCorpus', () => {
  it('returns alpha and family for simple_lookup', () => {
    const result = classifyForSearchCorpus('crossover');
    expect(result.query_class).toBe('simple_lookup');
    expect(result.alpha).toBe(EMBEDDED_ALPHA_DEFAULTS.simple_lookup);
    expect(result.family).toBe(EMBEDDED_FAMILY_DEFAULTS.simple_lookup);
    expect(result.classification_fallback).toBe(false);
  });

  it('returns alpha and family for plan_specific', () => {
    const result = classifyForSearchCorpus('what is the checkpointing design');
    expect(result.query_class).toBe('plan_specific');
    expect(result.alpha).toBe(EMBEDDED_ALPHA_DEFAULTS.plan_specific);
    expect(result.family).toBe(EMBEDDED_FAMILY_DEFAULTS.plan_specific);
  });

  it('returns alpha and family for code_specific', () => {
    const result = classifyForSearchCorpus('network.activate');
    expect(result.query_class).toBe('code_specific');
    expect(result.alpha).toBe(EMBEDDED_ALPHA_DEFAULTS.code_specific);
    expect(result.family).toBe(EMBEDDED_FAMILY_DEFAULTS.code_specific);
  });

  it('returns alpha and family for multi_hop', () => {
    const result = classifyForSearchCorpus('functions that call activate and also use slab');
    expect(result.query_class).toBe('multi_hop');
    expect(result.alpha).toBe(EMBEDDED_ALPHA_DEFAULTS.multi_hop);
    expect(result.family).toBe(EMBEDDED_FAMILY_DEFAULTS.multi_hop);
  });

  it('returns alpha and family for cross_boundary', () => {
    const result = classifyForSearchCorpus('relationship between crossover and mutation');
    expect(result.query_class).toBe('cross_boundary');
    expect(result.alpha).toBe(EMBEDDED_ALPHA_DEFAULTS.cross_boundary);
    expect(result.family).toBe(EMBEDDED_FAMILY_DEFAULTS.cross_boundary);
  });

  it('returns alpha and family for exploratory', () => {
    const result = classifyForSearchCorpus('how does the training pipeline work');
    expect(result.query_class).toBe('exploratory');
    expect(result.alpha).toBe(EMBEDDED_ALPHA_DEFAULTS.exploratory);
    expect(result.family).toBe(EMBEDDED_FAMILY_DEFAULTS.exploratory);
  });

  it('returns alpha and family for fallback', () => {
    const result = classifyForSearchCorpus('the quick brown fox jumps over the lazy dog');
    expect(result.query_class).toBe('simple_lookup');
    expect(result.alpha).toBe(EMBEDDED_ALPHA_DEFAULTS.simple_lookup);
    expect(result.classification_fallback).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// EMBEDDED_ALPHA_DEFAULTS and EMBEDDED_FAMILY_DEFAULTS
// ---------------------------------------------------------------------------

describe('classify-query: embedded defaults', () => {
  it('has alpha defaults for all six classes', () => {
    expect(EMBEDDED_ALPHA_DEFAULTS).toHaveProperty('simple_lookup');
    expect(EMBEDDED_ALPHA_DEFAULTS).toHaveProperty('cross_boundary');
    expect(EMBEDDED_ALPHA_DEFAULTS).toHaveProperty('multi_hop');
    expect(EMBEDDED_ALPHA_DEFAULTS).toHaveProperty('exploratory');
    expect(EMBEDDED_ALPHA_DEFAULTS).toHaveProperty('code_specific');
    expect(EMBEDDED_ALPHA_DEFAULTS).toHaveProperty('plan_specific');
  });

  it('has family defaults for all six classes', () => {
    expect(EMBEDDED_FAMILY_DEFAULTS).toHaveProperty('simple_lookup');
    expect(EMBEDDED_FAMILY_DEFAULTS).toHaveProperty('code_specific');
    expect(EMBEDDED_FAMILY_DEFAULTS).toHaveProperty('plan_specific');
  });
});