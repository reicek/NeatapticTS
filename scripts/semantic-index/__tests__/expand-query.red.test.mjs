/**
 * @module expand-query.red.test
 * @description Red tests for the query expansion pipeline.
 *
 * Tests cover:
 * - Nearest-term discovery with cosine similarity
 * - Domain association lookup
 * - Expansion budget enforcement
 * - BM25 OR-expansion
 * - Dense mean-pool expansion
 * - Classification-aware expansion behavior
 * - Backward compatibility (expand_query=false produces identical results)
 * - Expansion relevance computation
 *
 * Pure .mjs test — runs directly via Jest ESM project (no ts-jest).
 */

// ---------------------------------------------------------------------------
// Domain associations tests
// ---------------------------------------------------------------------------

describe('expand-query: domain associations', () => {
  it('looks up known domain associations case-insensitively', async () => {
    const { lookupDomainAssociations } = await import('../expand-query.mjs');
    const dictionary = {
      version: 1,
      associations: [
        {
          term: 'NEAT',
          expansions: [
            'NeuroEvolution of Augmenting Topologies',
            'neuroevolution',
          ],
          source: 'academic',
          confidence: 1.0,
        },
        {
          term: 'slab',
          expansions: ['typed-array activation cache'],
          source: 'codebase',
          confidence: 0.9,
        },
      ],
    };
    const expansions = lookupDomainAssociations(
      ['NEAT', 'crossover'],
      dictionary,
    );
    expect(expansions.length).toBeGreaterThan(0);
    const neatExpansions = expansions.filter((e) => e.original === 'NEAT');
    expect(neatExpansions.length).toBe(2);
    expect(
      neatExpansions.some(
        (e) => e.expanded === 'NeuroEvolution of Augmenting Topologies',
      ),
    ).toBe(true);
  });

  it('returns empty array for terms with no associations', async () => {
    const { lookupDomainAssociations } = await import('../expand-query.mjs');
    const dictionary = {
      version: 1,
      associations: [
        {
          term: 'NEAT',
          expansions: ['neuroevolution'],
          source: 'academic',
          confidence: 1.0,
        },
      ],
    };
    const expansions = lookupDomainAssociations(['xyzunknown'], dictionary);
    expect(expansions.length).toBe(0);
  });

  it('computes relevance score with length bonus for multi-word expansions', async () => {
    const { lookupDomainAssociations } = await import('../expand-query.mjs');
    const dictionary = {
      version: 1,
      associations: [
        {
          term: 'NEAT',
          expansions: [
            'neuroevolution',
            'NeuroEvolution of Augmenting Topologies',
          ],
          source: 'academic',
          confidence: 1.0,
        },
      ],
    };
    const expansions = lookupDomainAssociations(['NEAT'], dictionary);
    const singleWord = expansions.find((e) => e.expanded === 'neuroevolution');
    const multiWord = expansions.find(
      (e) => e.expanded === 'NeuroEvolution of Augmenting Topologies',
    );
    // Multi-word expansion should have a higher relevance score due to length bonus
    expect(multiWord.relevanceScore).toBeGreaterThan(singleWord.relevanceScore);
  });
});

// ---------------------------------------------------------------------------
// Expansion budget enforcement tests
// ---------------------------------------------------------------------------

describe('expand-query: budget enforcement', () => {
  it('limits expansions to MAX_EXPANDED_TERMS (3)', async () => {
    const { selectExpansions, MAX_EXPANDED_TERMS } =
      await import('../expand-query.mjs');

    const embeddingExpansions = [
      {
        original: 'NEAT',
        expanded: 'neuroevolution',
        source: 'embedding-synonym',
        similarity: 0.8,
        confidence: null,
        relevanceScore: 0.78,
        type: 'embedding-synonym',
      },
      {
        original: 'NEAT',
        expanded: 'augment',
        source: 'embedding-synonym',
        similarity: 0.75,
        confidence: null,
        relevanceScore: 0.73,
        type: 'embedding-synonym',
      },
      {
        original: 'crossover',
        expanded: 'recombination',
        source: 'embedding-synonym',
        similarity: 0.7,
        confidence: null,
        relevanceScore: 0.68,
        type: 'embedding-synonym',
      },
      {
        original: 'crossover',
        expanded: 'offspring',
        source: 'embedding-synonym',
        similarity: 0.65,
        confidence: null,
        relevanceScore: 0.63,
        type: 'embedding-synonym',
      },
      {
        original: 'crossover',
        expanded: 'mating',
        source: 'embedding-synonym',
        similarity: 0.62,
        confidence: null,
        relevanceScore: 0.6,
        type: 'embedding-synonym',
      },
    ];
    const domainExpansions = [];

    const selected = selectExpansions(embeddingExpansions, domainExpansions);
    expect(selected.length).toBeLessThanOrEqual(MAX_EXPANDED_TERMS);
  });

  it('filters out candidates below MIN_EXPANSION_RELEVANCE (0.55)', async () => {
    const { selectExpansions, MIN_EXPANSION_RELEVANCE } =
      await import('../expand-query.mjs');

    const candidates = [
      {
        original: 'NEAT',
        expanded: 'neuroevolution',
        source: 'domain-association',
        confidence: 1.0,
        relevanceScore: 0.97,
        type: 'domain-association',
      },
      {
        original: 'NEAT',
        expanded: 'weak-term',
        source: 'embedding-synonym',
        similarity: 0.4,
        confidence: null,
        relevanceScore: 0.38,
        type: 'embedding-synonym',
      },
    ];

    const selected = selectExpansions(candidates, []);
    // All selected should meet minimum relevance
    expect(
      selected.every((s) => s.relevanceScore >= MIN_EXPANSION_RELEVANCE),
    ).toBe(true);
  });

  it('deduplicates expansions by expanded term (case-insensitive)', async () => {
    const { deduplicateExpansions } = await import('../expand-query.mjs');

    const candidates = [
      {
        original: 'NEAT',
        expanded: 'neuroevolution',
        source: 'embedding-synonym',
        relevanceScore: 0.72,
        type: 'embedding-synonym',
      },
      {
        original: 'NEAT',
        expanded: 'NeuroEvolution',
        source: 'domain-association',
        confidence: 1.0,
        relevanceScore: 0.97,
        type: 'domain-association',
      },
    ];

    const deduped = deduplicateExpansions(candidates);
    // "neuroevolution" and "NeuroEvolution" should deduplicate to 1 entry (higher score wins)
    expect(deduped.length).toBe(1);
    expect(deduped[0].relevanceScore).toBe(0.97);
  });
});

// ---------------------------------------------------------------------------
// BM25 OR-expansion tests
// ---------------------------------------------------------------------------

describe('expand-query: BM25 expansion', () => {
  it('builds OR-expanded FTS5 query with expanded terms', async () => {
    const { buildExpandedFtsQuery } = await import('../expand-query.mjs');
    const query = buildExpandedFtsQuery('NEAT crossover', [
      { expanded: 'neuroevolution' },
      { expanded: 'NeuroEvolution of Augmenting Topologies' },
    ]);
    expect(query).toContain('OR neuroevolution');
    expect(query).toContain('OR "NeuroEvolution of Augmenting Topologies"');
    expect(query).toContain('NEAT crossover');
  });

  it('returns original query when no expansions provided', async () => {
    const { buildExpandedFtsQuery } = await import('../expand-query.mjs');
    const query = buildExpandedFtsQuery('simple query', []);
    expect(query).toBe('simple query');
  });

  it('wraps multi-word expansions in FTS5 phrase quotes', async () => {
    const { buildExpandedFtsQuery } = await import('../expand-query.mjs');
    const query = buildExpandedFtsQuery('slab', [
      { expanded: 'typed-array activation cache' },
      { expanded: 'cache-friendly activation' },
    ]);
    expect(query).toContain('"typed-array activation cache"');
    expect(query).toContain('"cache-friendly activation"');
  });
});

// ---------------------------------------------------------------------------
// Dense mean-pool expansion tests
// ---------------------------------------------------------------------------

describe('expand-query: dense expansion', () => {
  it('computes L2-normalized mean-pooled embedding of original + expanded terms', async () => {
    const { computeExpandedEmbedding } = await import('../expand-query.mjs');

    const original = new Float32Array([0.5, 0.5, 0.5, 0.5]);
    const expansion = new Float32Array([0.5, -0.5, 0.5, -0.5]);

    const expanded = computeExpandedEmbedding(original, [expansion]);

    // Verify L2-normalization: magnitude should be ~1
    let magnitude = 0;
    for (let i = 0; i < expanded.length; i++)
      magnitude += expanded[i] * expanded[i];
    magnitude = Math.sqrt(magnitude);

    expect(Math.abs(magnitude - 1.0)).toBeLessThan(0.001);
    expect(expanded.length).toBe(4);
  });

  it('returns original embedding when no expansion embeddings provided', async () => {
    const { computeExpandedEmbedding } = await import('../expand-query.mjs');

    const original = new Float32Array([0.7071, 0.7071]);
    const expanded = computeExpandedEmbedding(original, []);

    expect(Math.abs(original[0] - expanded[0])).toBeLessThan(0.001);
    expect(Math.abs(original[1] - expanded[1])).toBeLessThan(0.001);
  });
});

// ---------------------------------------------------------------------------
// Classification-aware expansion behavior tests
// ---------------------------------------------------------------------------

describe('expand-query: classification-aware expansion', () => {
  it('disables expansion for simple_lookup queries', async () => {
    const { expansionBehaviorForClass } = await import('../expand-query.mjs');
    expect(expansionBehaviorForClass('simple_lookup')).toBe(false);
  });

  it('enables domain-only expansion for code_specific queries', async () => {
    const { expansionBehaviorForClass } = await import('../expand-query.mjs');
    expect(expansionBehaviorForClass('code_specific')).toBe('domain-only');
  });

  it('enables domain-only expansion for plan_specific queries', async () => {
    const { expansionBehaviorForClass } = await import('../expand-query.mjs');
    expect(expansionBehaviorForClass('plan_specific')).toBe('domain-only');
  });

  it('enables full expansion for cross_boundary queries', async () => {
    const { expansionBehaviorForClass } = await import('../expand-query.mjs');
    expect(expansionBehaviorForClass('cross_boundary')).toBe(true);
  });

  it('enables full expansion for multi_hop queries', async () => {
    const { expansionBehaviorForClass } = await import('../expand-query.mjs');
    expect(expansionBehaviorForClass('multi_hop')).toBe(true);
  });

  it('enables full expansion for exploratory queries', async () => {
    const { expansionBehaviorForClass } = await import('../expand-query.mjs');
    expect(expansionBehaviorForClass('exploratory')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Backward compatibility tests
// ---------------------------------------------------------------------------

describe('expand-query: backward compatibility', () => {
  it('returns identity result when expandQuery is false (default)', async () => {
    const { expandQuery } = await import('../expand-query.mjs');
    const result = await expandQuery({
      query: 'NEAT crossover',
      expandQuery: false,
    });
    expect(result.expansion.applied).toBe(false);
    expect(result.expandedTerms.length).toBe(0);
    expect(result.bm25Query).toBeNull();
  });

  it('returns identity result when expandQuery is undefined', async () => {
    const { expandQuery } = await import('../expand-query.mjs');
    const result = await expandQuery({ query: 'NEAT crossover' });
    expect(result.expansion.applied).toBe(false);
    expect(result.expandedTerms.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Expansion relevance computation tests
// ---------------------------------------------------------------------------

describe('expand-query: expansion relevance', () => {
  it('computes relevance with frequency penalty', async () => {
    const { expansionRelevance } = await import('../expand-query.mjs');
    const rare = expansionRelevance({
      similarity: 0.8,
      frequency: 5,
      expanded: 'term',
    });
    const common = expansionRelevance({
      similarity: 0.8,
      frequency: 1000,
      expanded: 'term',
    });
    // Rare terms should have higher relevance (less frequency penalty)
    expect(rare).toBeGreaterThan(common);
  });

  it('computes relevance with length bonus for multi-word expansions', async () => {
    const { expansionRelevance } = await import('../expand-query.mjs');
    const single = expansionRelevance({
      similarity: 0.8,
      frequency: 10,
      expanded: 'term',
    });
    const multi = expansionRelevance({
      similarity: 0.8,
      frequency: 10,
      expanded: 'multi word term',
    });
    // Multi-word expansions should have higher relevance (length bonus)
    expect(multi).toBeGreaterThan(single);
  });
});

// ---------------------------------------------------------------------------
// findNearestTerms tests — removed (old client-side cosine similarity)
// The server-side vector_top_k approach replaces findNearestTerms.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

describe('expand-query: exported constants', () => {
  it('exports correct budget and threshold values', async () => {
    const { MAX_EXPANDED_TERMS, MIN_EXPANSION_RELEVANCE, MIN_SIMILARITY } =
      await import('../expand-query.mjs');
    expect(MAX_EXPANDED_TERMS).toBe(3);
    expect(MIN_EXPANSION_RELEVANCE).toBe(0.55);
    expect(MIN_SIMILARITY).toBe(0.65);
  });
});
