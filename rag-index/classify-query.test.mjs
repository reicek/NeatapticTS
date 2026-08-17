import {
  EMBEDDED_ALPHA_DEFAULTS,
  EMBEDDED_FAMILY_DEFAULTS,
  hasPlanHints,
  hasCodeHints,
  hasCodeIdentifiers,
  hasMultiHopIndicators,
  hasCrossFamilyIndicators,
  hasExploratoryHints,
  hasFamilyHints,
  classifyQuery,
  classifyForSearchCorpus,
} from './classify-query.mjs';

describe('EMBEDDED_ALPHA_DEFAULTS', () => {
  it('has alpha for all 6 classes', () => {
    expect(EMBEDDED_ALPHA_DEFAULTS.simple_lookup).toBe(0.75);
    expect(EMBEDDED_ALPHA_DEFAULTS.cross_boundary).toBe(0.5);
    expect(EMBEDDED_ALPHA_DEFAULTS.multi_hop).toBe(0.35);
    expect(EMBEDDED_ALPHA_DEFAULTS.exploratory).toBe(0.3);
    expect(EMBEDDED_ALPHA_DEFAULTS.code_specific).toBe(0.7);
    expect(EMBEDDED_ALPHA_DEFAULTS.plan_specific).toBe(0.7);
  });
});

describe('EMBEDDED_FAMILY_DEFAULTS', () => {
  it('has family for all 6 classes', () => {
    expect(EMBEDDED_FAMILY_DEFAULTS.simple_lookup).toBeNull();
    expect(EMBEDDED_FAMILY_DEFAULTS.cross_boundary).toBeNull();
    expect(EMBEDDED_FAMILY_DEFAULTS.multi_hop).toBeNull();
    expect(EMBEDDED_FAMILY_DEFAULTS.exploratory).toBeNull();
    expect(EMBEDDED_FAMILY_DEFAULTS.code_specific).toBe('ts-source');
    expect(EMBEDDED_FAMILY_DEFAULTS.plan_specific).toBe('plan,completed-plan');
  });
});

describe('hasPlanHints', () => {
  it('returns true for plan keywords', () => {
    expect(hasPlanHints('what is the plan')).toBe(true);
    expect(hasPlanHints('design document')).toBe(true);
    expect(hasPlanHints('architecture overview')).toBe(true);
    expect(hasPlanHints('roadmap for v2')).toBe(true);
    expect(hasPlanHints('decision record')).toBe(true);
    expect(hasPlanHints('specification doc')).toBe(true);
  });

  it('returns false for no plan keywords', () => {
    expect(hasPlanHints('how does crossover work')).toBe(false);
  });
});

describe('hasCodeHints', () => {
  it('returns true for code keywords', () => {
    expect(hasCodeHints('implementation of crossover')).toBe(true);
    expect(hasCodeHints('code for activate')).toBe(true);
    expect(hasCodeHints('source of network')).toBe(true);
    expect(hasCodeHints('typescript function')).toBe(true);
    expect(hasCodeHints('implement the method')).toBe(true);
    expect(hasCodeHints('function body of activate')).toBe(true);
  });

  it('returns false for no code keywords', () => {
    expect(hasCodeHints('how does crossover work')).toBe(false);
  });
});

describe('hasCodeIdentifiers', () => {
  it('returns true for dotted identifiers', () => {
    expect(hasCodeIdentifiers('network.activate')).toBe(true);
  });

  it('returns true for snake_case identifiers', () => {
    expect(hasCodeIdentifiers('snake_case_function')).toBe(true);
  });

  it('returns true for camelCase identifiers', () => {
    expect(hasCodeIdentifiers('findTheNEATSelectionCode')).toBe(true);
  });

  it('returns true for file extension hints', () => {
    expect(hasCodeIdentifiers('code.ts')).toBe(true);
  });

  it('returns false for plain words', () => {
    expect(hasCodeIdentifiers('how does NEAT work')).toBe(false);
  });

  it('returns false for all-caps acronyms', () => {
    expect(hasCodeIdentifiers('NEAT')).toBe(false);
  });
});

describe('hasMultiHopIndicators', () => {
  it('returns true for multi-hop patterns', () => {
    expect(hasMultiHopIndicators('functions that also use slab')).toBe(true);
    expect(hasMultiHopIndicators('which also includes')).toBe(true);
    expect(hasMultiHopIndicators('and then process')).toBe(true);
    expect(hasMultiHopIndicators('call something that returns')).toBe(true);
    expect(hasMultiHopIndicators('where also defined')).toBe(true);
  });

  it('returns false for no multi-hop patterns', () => {
    expect(hasMultiHopIndicators('what is crossover')).toBe(false);
  });
});

describe('hasCrossFamilyIndicators', () => {
  it('returns true for cross-family keywords', () => {
    expect(hasCrossFamilyIndicators('relationship between crossover and mutation')).toBe(true);
    expect(hasCrossFamilyIndicators('connection between modules')).toBe(true);
    expect(hasCrossFamilyIndicators('between the layers')).toBe(true);
  });

  it('returns false for no cross-family keywords', () => {
    expect(hasCrossFamilyIndicators('how does crossover work')).toBe(false);
  });
});

describe('hasExploratoryHints', () => {
  it('returns true for exploratory keywords', () => {
    expect(hasExploratoryHints('how does the training pipeline work')).toBe(true);
    expect(hasExploratoryHints('explain the algorithm')).toBe(true);
    expect(hasExploratoryHints('overview of the system')).toBe(true);
    expect(hasExploratoryHints('describe the architecture')).toBe(true);
    expect(hasExploratoryHints('what is crossover')).toBe(true);
    expect(hasExploratoryHints('tell me about NEAT')).toBe(true);
    expect(hasExploratoryHints('how do neurons work')).toBe(true);
  });

  it('returns false for no exploratory keywords', () => {
    expect(hasExploratoryHints('network.activate')).toBe(false);
  });
});

describe('hasFamilyHints', () => {
  it('returns true when any family hint is present', () => {
    expect(hasFamilyHints('what is the plan')).toBe(true);
    expect(hasFamilyHints('implementation of crossover')).toBe(true);
    expect(hasFamilyHints('functions that also use slab')).toBe(true);
    expect(hasFamilyHints('relationship between modules')).toBe(true);
    expect(hasFamilyHints('how does crossover work')).toBe(true);
  });

  it('returns false when no family hints are present', () => {
    expect(hasFamilyHints('short api call')).toBe(false);
  });
});

describe('classifyQuery', () => {
  it('classifies short queries as simple_lookup', () => {
    const result = classifyQuery('crossover');
    expect(result.query_class).toBe('simple_lookup');
    expect(result.confidence).toBe(0.9);
    expect(result.hints.short_query).toBe(true);
  });

  it('classifies short query with family hints as non-simple', () => {
    const result = classifyQuery('plan design');
    expect(result.query_class).toBe('plan_specific');
  });

  it('classifies short query with code identifiers as code_specific', () => {
    const result = classifyQuery('network.activate');
    expect(result.query_class).toBe('code_specific');
    expect(result.confidence).toBe(0.85);
    expect(result.hints.family_filter).toBe('ts-source');
  });

  it('classifies plan-specific queries', () => {
    const result = classifyQuery('what is the checkpointing design and architecture');
    expect(result.query_class).toBe('plan_specific');
    expect(result.confidence).toBe(0.85);
    expect(result.hints.family_filter).toBe('plan,completed-plan');
  });

  it('classifies code-specific via identifiers', () => {
    const result = classifyQuery('findTheNEATSelectionCode in the codebase');
    expect(result.query_class).toBe('code_specific');
    expect(result.confidence).toBe(0.85);
  });

  it('classifies code-specific via keywords (no identifiers)', () => {
    const result = classifyQuery('implementation of crossover in the NEAT system');
    expect(result.query_class).toBe('code_specific');
    expect(result.confidence).toBe(0.8);
  });

  it('classifies multi-hop queries', () => {
    const result = classifyQuery('functions that call activate and then use slab in the network');
    expect(result.query_class).toBe('multi_hop');
    expect(result.confidence).toBe(0.75);
    expect(result.hints.multi_hop).toBe(true);
  });

  it('classifies cross-boundary queries', () => {
    const result = classifyQuery('the relationship between crossover and mutation in the system');
    expect(result.query_class).toBe('cross_boundary');
    expect(result.confidence).toBe(0.7);
    expect(result.hints.multi_family).toBe(true);
  });

  it('classifies exploratory queries', () => {
    const result = classifyQuery('how does the training pipeline work with the data');
    expect(result.query_class).toBe('exploratory');
    expect(result.confidence).toBe(0.65);
    expect(result.hints.broad_retrieval).toBe(true);
  });

  it('falls back to simple_lookup for unmatched long queries', () => {
    const result = classifyQuery('find all the things in the system that are not related to anything else here');
    expect(result.query_class).toBe('simple_lookup');
    expect(result.confidence).toBe(0.5);
    expect(result.hints.fallback).toBe(true);
  });
});

describe('classifyForSearchCorpus', () => {
  it('returns alpha and family for simple_lookup', () => {
    const result = classifyForSearchCorpus('crossover');
    expect(result.alpha).toBe(0.75);
    expect(result.family).toBeNull();
    expect(result.query_class).toBe('simple_lookup');
    expect(result.classification_fallback).toBe(false);
  });

  it('returns alpha and family for code_specific', () => {
    const result = classifyForSearchCorpus('network.activate');
    expect(result.alpha).toBe(0.7);
    expect(result.family).toBe('ts-source');
    expect(result.query_class).toBe('code_specific');
  });

  it('returns alpha and family for plan_specific', () => {
    const result = classifyForSearchCorpus('what is the checkpointing design and architecture');
    expect(result.alpha).toBe(0.7);
    expect(result.family).toBe('plan,completed-plan');
  });

  it('returns alpha and family for exploratory', () => {
    const result = classifyForSearchCorpus('how does the training pipeline work with data');
    expect(result.alpha).toBe(0.3);
    expect(result.family).toBeNull();
  });

  it('returns alpha and family for multi_hop', () => {
    const result = classifyForSearchCorpus('functions that call activate and then use slab in network');
    expect(result.alpha).toBe(0.35);
    expect(result.family).toBeNull();
  });

  it('returns alpha and family for cross_boundary', () => {
    const result = classifyForSearchCorpus('the relationship between crossover and mutation in system');
    expect(result.alpha).toBe(0.5);
    expect(result.family).toBeNull();
  });

  it('returns alpha and family for fallback simple_lookup', () => {
    const result = classifyForSearchCorpus('find all the things in the system that are not related to anything else here');
    expect(result.alpha).toBe(0.75);
    expect(result.family).toBeNull();
    expect(result.classification_fallback).toBe(false);
  });
});