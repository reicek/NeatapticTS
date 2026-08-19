/**
 * @module rag-index-expand-query.test
 * @description Coverage tests for rag-index/expand-query.mjs — query expansion pipeline.
 */

import { jest } from '@jest/globals';
import {
  expandQuery,
  expansionBehaviorForClass,
  loadDomainAssociations,
  lookupDomainAssociations,
  findNearestTermsServerSide,
  expansionRelevance,
  deduplicateExpansions,
  selectExpansions,
  buildExpandedFtsQuery,
  computeExpandedEmbedding,
  invalidateDomainAssociationsCache,
  MAX_EXPANDED_TERMS,
  MIN_EXPANSION_RELEVANCE,
  MIN_SIMILARITY,
  MAX_NEAREST_TERMS,
  DEFAULT_DOMAIN_ASSOCIATIONS_PATH,
} from '../../../rag-index/expand-query.mjs';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

describe('expand-query: constants', () => {
  it('exports expected constants', () => {
    expect(MAX_EXPANDED_TERMS).toBe(3);
    expect(MIN_EXPANSION_RELEVANCE).toBe(0.55);
    expect(MIN_SIMILARITY).toBe(0.65);
    expect(MAX_NEAREST_TERMS).toBe(5);
    expect(typeof DEFAULT_DOMAIN_ASSOCIATIONS_PATH).toBe('string');
  });
});

// ---------------------------------------------------------------------------
// expansionBehaviorForClass
// ---------------------------------------------------------------------------

describe('expand-query: expansionBehaviorForClass', () => {
  it('returns false for simple_lookup', () => {
    expect(expansionBehaviorForClass('simple_lookup')).toBe(false);
  });

  it('returns domain-only for code_specific', () => {
    expect(expansionBehaviorForClass('code_specific')).toBe('domain-only');
  });

  it('returns domain-only for plan_specific', () => {
    expect(expansionBehaviorForClass('plan_specific')).toBe('domain-only');
  });

  it('returns true for cross_boundary', () => {
    expect(expansionBehaviorForClass('cross_boundary')).toBe(true);
  });

  it('returns true for multi_hop', () => {
    expect(expansionBehaviorForClass('multi_hop')).toBe(true);
  });

  it('returns true for exploratory', () => {
    expect(expansionBehaviorForClass('exploratory')).toBe(true);
  });

  it('returns false for unknown class', () => {
    expect(expansionBehaviorForClass('unknown')).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// loadDomainAssociations
// ---------------------------------------------------------------------------

describe('expand-query: loadDomainAssociations', () => {
  it('returns cached associations on subsequent calls', async () => {
    invalidateDomainAssociationsCache();
    const result1 = await loadDomainAssociations();
    const result2 = await loadDomainAssociations();
    expect(result1).toBe(result2); // Same reference (cached)
  });

  it('returns empty dictionary when file is not found', async () => {
    invalidateDomainAssociationsCache();
    const result = await loadDomainAssociations(
      '/nonexistent/path/to/file.json',
    );
    expect(result.version).toBe(1);
    expect(result.associations).toEqual([]);
  });

  it('invalidateDomainAssociationsCache resets the cache', async () => {
    await loadDomainAssociations();
    invalidateDomainAssociationsCache();
    // Next call should reload
    const result = await loadDomainAssociations();
    expect(result).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// lookupDomainAssociations
// ---------------------------------------------------------------------------

describe('expand-query: lookupDomainAssociations', () => {
  it('finds matching associations', () => {
    const dictionary = {
      version: 1,
      associations: [
        {
          term: 'crossover',
          expansions: ['recombination', 'gene swap'],
          source: 'genetics',
          confidence: 0.9,
        },
      ],
    };
    const results = lookupDomainAssociations(['crossover'], dictionary);
    expect(results.length).toBe(2);
    expect(results[0].expanded).toBe('recombination');
    expect(results[0].source).toBe('genetics');
    expect(results[0].confidence).toBe(0.9);
    expect(results[0].type).toBe('domain-association');
  });

  it('is case-insensitive', () => {
    const dictionary = {
      version: 1,
      associations: [
        {
          term: 'Crossover',
          expansions: ['recombination'],
          source: 'test',
          confidence: 0.8,
        },
      ],
    };
    const results = lookupDomainAssociations(['crossover'], dictionary);
    expect(results.length).toBe(1);
    expect(results[0].expanded).toBe('recombination');
  });

  it('returns empty for no matches', () => {
    const dictionary = { version: 1, associations: [] };
    const results = lookupDomainAssociations(['unknown'], dictionary);
    expect(results).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// expansionRelevance
// ---------------------------------------------------------------------------

describe('expand-query: expansionRelevance', () => {
  it('computes relevance from similarity', () => {
    const score = expansionRelevance({
      similarity: 0.9,
      frequency: 10,
      expanded: 'test',
    });
    expect(score).toBeGreaterThan(0);
  });

  it('computes relevance from confidence', () => {
    const score = expansionRelevance({
      confidence: 0.8,
      frequency: 5,
      expanded: 'test',
    });
    expect(score).toBeGreaterThan(0);
  });

  it('uses frequency penalty', () => {
    const lowFreq = expansionRelevance({
      similarity: 0.9,
      frequency: 1,
      expanded: 'test',
    });
    const highFreq = expansionRelevance({
      similarity: 0.9,
      frequency: 1000,
      expanded: 'test',
    });
    expect(lowFreq).toBeGreaterThan(highFreq);
  });

  it('gives length bonus for multi-word expansions', () => {
    const shortExp = expansionRelevance({
      similarity: 0.9,
      frequency: 1,
      expanded: 'word',
    });
    const longExp = expansionRelevance({
      similarity: 0.9,
      frequency: 1,
      expanded: 'three word phrase',
    });
    expect(longExp).toBeGreaterThan(shortExp);
  });

  it('defaults frequency to 1 when absent', () => {
    const score = expansionRelevance({ similarity: 0.9, expanded: 'test' });
    expect(score).toBeGreaterThan(0);
  });

  it('defaults base score to 0 when neither similarity nor confidence', () => {
    const score = expansionRelevance({ frequency: 1, expanded: 'test' });
    expect(score).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// deduplicateExpansions
// ---------------------------------------------------------------------------

describe('expand-query: deduplicateExpansions', () => {
  it('removes duplicates keeping higher score', () => {
    const candidates = [
      { expanded: 'term1', relevanceScore: 0.6 },
      { expanded: 'term1', relevanceScore: 0.8 },
      { expanded: 'term2', relevanceScore: 0.7 },
    ];
    const result = deduplicateExpansions(candidates);
    expect(result.length).toBe(2);
    expect(result[0].relevanceScore).toBe(0.8);
    expect(result[1].relevanceScore).toBe(0.7);
  });

  it('is case-insensitive', () => {
    const candidates = [
      { expanded: 'Term1', relevanceScore: 0.5 },
      { expanded: 'term1', relevanceScore: 0.6 },
    ];
    const result = deduplicateExpansions(candidates);
    expect(result.length).toBe(1);
    expect(result[0].relevanceScore).toBe(0.6);
  });
});

// ---------------------------------------------------------------------------
// selectExpansions
// ---------------------------------------------------------------------------

describe('expand-query: selectExpansions', () => {
  it('merges and selects top-K expansions', () => {
    const embeddingExpansions = [
      { expanded: 'term1', similarity: 0.9, frequency: 5, relevanceScore: 0.8 },
      { expanded: 'term2', similarity: 0.8, frequency: 3, relevanceScore: 0.7 },
    ];
    const domainExpansions = [
      {
        expanded: 'term3',
        confidence: 0.85,
        frequency: 1,
        relevanceScore: 0.75,
      },
    ];
    const result = selectExpansions(embeddingExpansions, domainExpansions);
    expect(result.length).toBeLessThanOrEqual(MAX_EXPANDED_TERMS);
    expect(result[0].relevanceScore).toBeGreaterThanOrEqual(
      result[1]?.relevanceScore ?? 0,
    );
  });

  it('computes relevanceScore for candidates missing it', () => {
    const embeddingExpansions = [
      { expanded: 'term1', similarity: 0.9, frequency: 5 },
    ];
    const result = selectExpansions(embeddingExpansions, []);
    expect(result.length).toBeGreaterThan(0);
    expect(result[0].relevanceScore).toBeDefined();
  });

  it('filters out candidates below MIN_EXPANSION_RELEVANCE', () => {
    const embeddingExpansions = [
      {
        expanded: 'low',
        similarity: 0.1,
        frequency: 1000,
        relevanceScore: 0.1,
      },
    ];
    const result = selectExpansions(embeddingExpansions, []);
    expect(result).toEqual([]);
  });

  it('limits to MAX_EXPANDED_TERMS', () => {
    const many = Array.from({ length: 10 }, (_, i) => ({
      expanded: `term${i}`,
      similarity: 0.9,
      frequency: 1,
      relevanceScore: 0.8,
    }));
    const result = selectExpansions(many, []);
    expect(result.length).toBe(MAX_EXPANDED_TERMS);
  });
});

// ---------------------------------------------------------------------------
// buildExpandedFtsQuery
// ---------------------------------------------------------------------------

describe('expand-query: buildExpandedFtsQuery', () => {
  it('returns original query when no expansions', () => {
    expect(buildExpandedFtsQuery('crossover', [])).toBe('crossover');
  });

  it('appends OR clauses for single-word expansions', () => {
    const result = buildExpandedFtsQuery('crossover', [
      { expanded: 'recombination' },
      { expanded: 'gene' },
    ]);
    expect(result).toContain('OR recombination');
    expect(result).toContain('OR gene');
  });

  it('wraps multi-word expansions in phrase quotes', () => {
    const result = buildExpandedFtsQuery('crossover', [
      { expanded: 'gene swap' },
    ]);
    expect(result).toContain('OR "gene swap"');
  });
});

// ---------------------------------------------------------------------------
// computeExpandedEmbedding
// ---------------------------------------------------------------------------

describe('expand-query: computeExpandedEmbedding', () => {
  it('mean-pools and L2-normalizes embeddings', () => {
    const original = new Float32Array([1, 0, 0]);
    const expansions = [new Float32Array([0, 1, 0])];
    const result = computeExpandedEmbedding(original, expansions);
    // Mean of [1,0,0] and [0,1,0] = [0.5,0.5,0], L2-normalized = [0.707,0.707,0]
    expect(result[0]).toBeCloseTo(0.7071, 3);
    expect(result[1]).toBeCloseTo(0.7071, 3);
    expect(result[2]).toBeCloseTo(0, 3);
  });

  it('handles zero embeddings', () => {
    const original = new Float32Array([0, 0, 0]);
    const result = computeExpandedEmbedding(original, []);
    expect(result).toEqual(new Float32Array([0, 0, 0]));
  });

  it('returns L2-normalized for single embedding', () => {
    const original = new Float32Array([3, 0, 0]);
    const result = computeExpandedEmbedding(original, []);
    // Mean = [3,0,0], normalized = [1,0,0]
    expect(result[0]).toBeCloseTo(1, 5);
    expect(result[1]).toBeCloseTo(0, 5);
  });
});

// ---------------------------------------------------------------------------
// findNearestTermsServerSide
// ---------------------------------------------------------------------------

describe('expand-query: findNearestTermsServerSide', () => {
  it('falls back to brute-force when ANN fails', async () => {
    const mockClient = {
      async execute({ sql }) {
        if (sql.includes('vector_top_k')) {
          throw new Error('ANN not available');
        }
        // brute-force path
        return {
          rows: [
            { term: 'crossover', distance: 0.1, frequency: 5 },
            { term: 'mutation', distance: 0.3, frequency: 3 },
          ],
        };
      },
    };
    const result = await findNearestTermsServerSide(
      mockClient,
      Buffer.from(new Float32Array([0.1, 0.2, 0.3]).buffer),
      'test-model',
    );
    expect(result.length).toBeGreaterThan(0);
    expect(result[0].term).toBe('crossover');
    expect(result[0].similarity).toBeCloseTo(0.9, 1);
  });

  it('uses ANN when available', async () => {
    const mockClient = {
      async execute({ sql }) {
        if (sql.includes('vector_top_k')) {
          return {
            rows: [
              { term: 'crossover', distance: 0.05, frequency: 5 },
              { term: 'mutation', distance: 0.2, frequency: 3 },
            ],
          };
        }
        return { rows: [] };
      },
    };
    const result = await findNearestTermsServerSide(
      mockClient,
      Buffer.from(new Float32Array([0.1, 0.2, 0.3]).buffer),
      'test-model',
    );
    expect(result.length).toBe(2);
    expect(result[0].similarity).toBeGreaterThan(result[1].similarity);
  });

  it('filters by minSimilarity', async () => {
    const mockClient = {
      async execute({ sql }) {
        if (sql.includes('vector_top_k')) {
          throw new Error('ANN not available');
        }
        return {
          rows: [
            { term: 'good', distance: 0.1, frequency: 5 },
            { term: 'bad', distance: 0.9, frequency: 3 },
          ],
        };
      },
    };
    const result = await findNearestTermsServerSide(
      mockClient,
      Buffer.from(new Float32Array([0.1, 0.2, 0.3]).buffer),
      'test-model',
      { minSimilarity: 0.5 },
    );
    expect(result.length).toBe(1);
    expect(result[0].term).toBe('good');
  });
});

// ---------------------------------------------------------------------------
// expandQuery
// ---------------------------------------------------------------------------

describe('expand-query: expandQuery', () => {
  it('returns identity when expandQuery is false', async () => {
    const result = await expandQuery({
      query: 'NEAT crossover',
      expandQuery: false,
    });
    expect(result.expansion.applied).toBe(false);
    expect(result.expandedTerms).toEqual([]);
    expect(result.bm25Query).toBe(null);
    expect(result.expandedEmbedding).toBe(null);
  });

  it('returns identity when expandQuery is not set', async () => {
    const result = await expandQuery({ query: 'NEAT crossover' });
    expect(result.expansion.applied).toBe(false);
  });

  it('returns no expansion when query has no qualifying terms', async () => {
    const result = await expandQuery({ query: 'ab', expandQuery: true });
    expect(result.expansion.applied).toBe(false);
    expect(result.expansion.reason).toBe('No qualifying terms in query');
  });

  it('returns degraded when no database client and no term_embeddings table', async () => {
    // Use a mock client that has no term_embeddings table
    const mockClient = {
      async execute({ sql }) {
        if (sql.includes('sqlite_master')) {
          return { rows: [] }; // no term_embeddings table
        }
        return { rows: [] };
      },
    };
    const result = await expandQuery({
      query: 'crossover',
      expandQuery: true,
      client: mockClient,
    });
    expect(result.expansion.degraded).toBe(true);
  });

  it('returns degraded in domain-only mode when term_embeddings table is absent', async () => {
    invalidateDomainAssociationsCache();
    const mockClient = {
      async execute({ sql }) {
        if (sql.includes('sqlite_master')) {
          return { rows: [] };
        }
        return { rows: [] };
      },
    };
    const result = await expandQuery({
      query: 'crossover mutation',
      expandQuery: 'domain-only',
      client: mockClient,
    });
    expect(result.expansion.degraded).toBe(true);
  });

  it('returns no qualifying expansions when no candidates match', async () => {
    invalidateDomainAssociationsCache();
    const mockClient = {
      async execute({ sql }) {
        if (sql.includes('sqlite_master')) {
          return { rows: [{ name: 'term_embeddings' }] };
        }
        if (sql.includes('vector_top_k')) {
          throw new Error('ANN not available');
        }
        // brute-force — return terms with very low similarity (high distance)
        return {
          rows: [{ term: 'unrelated', distance: 1.5, frequency: 1 }],
        };
      },
    };
    const mockEmbed = async () => new Float32Array([0.1, 0.2, 0.3]);
    const result = await expandQuery({
      query: 'crossover',
      expandQuery: true,
      client: mockClient,
      embedText: mockEmbed,
    });
    expect(result.expansion.applied).toBe(false);
  });

  it('uses domain associations when available', async () => {
    invalidateDomainAssociationsCache();
    // Create a temp file for domain associations
    const { writeFile } = await import('node:fs/promises');
    const { mkdtemp } = await import('node:fs/promises');
    const { tmpdir } = await import('node:os');
    const pathMod = await import('node:path');
    const tmpDir = await mkdtemp(pathMod.join(tmpdir(), 'expand-test-'));
    const assocPath = pathMod.join(tmpDir, 'associations.json');
    await writeFile(
      assocPath,
      JSON.stringify({
        version: 1,
        associations: [
          {
            term: 'crossover',
            expansions: ['recombination', 'gene swap'],
            source: 'genetics',
            confidence: 0.95,
          },
        ],
      }),
    );
    try {
      const mockClient = {
        async execute({ sql }) {
          if (sql.includes('sqlite_master')) {
            return { rows: [] }; // no term_embeddings → degraded
          }
          return { rows: [] };
        },
      };
      const result = await expandQuery({
        query: 'crossover mutation',
        expandQuery: 'domain-only',
        client: mockClient,
        associationsPath: assocPath,
      });
      // Domain-only with no term_embeddings table → degraded
      expect(result.expansion.degraded).toBe(true);
    } finally {
      const { rm } = await import('node:fs/promises');
      await rm(tmpDir, { recursive: true, force: true });
    }
    invalidateDomainAssociationsCache();
  });
});
