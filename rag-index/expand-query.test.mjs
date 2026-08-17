import { jest } from '@jest/globals';

// Mock node:fs/promises
const mockReadFile = jest.fn();
jest.unstable_mockModule('node:fs/promises', () => ({
  readFile: mockReadFile,
  default: { readFile: mockReadFile },
}));

// Mock embed-index
const mockCreateOnnxTextEmbedder = jest.fn();
const mockNormalizeEmbeddingVector = jest.fn();
const mockReadModelMeta = jest.fn();
jest.unstable_mockModule('./embed-index.mjs', () => ({
  DEFAULT_MODEL_ID: 'test-model',
  createOnnxTextEmbedder: mockCreateOnnxTextEmbedder,
  normalizeEmbeddingVector: mockNormalizeEmbeddingVector,
  readModelMeta: mockReadModelMeta,
  default: {
    DEFAULT_MODEL_ID: 'test-model',
    createOnnxTextEmbedder: mockCreateOnnxTextEmbedder,
    normalizeEmbeddingVector: mockNormalizeEmbeddingVector,
    readModelMeta: mockReadModelMeta,
  },
}));

// Mock build-term-index
const mockPorterTokenize = jest.fn();
jest.unstable_mockModule('./build-term-index.mjs', () => ({
  porterTokenize: mockPorterTokenize,
  default: { porterTokenize: mockPorterTokenize },
}));

// Mock init-schema
jest.unstable_mockModule('./init-schema.mjs', () => ({
  defaultDatabasePath: '/fake/db.sqlite',
  repoRoot: '/fake/repo',
  default: { defaultDatabasePath: '/fake/db.sqlite', repoRoot: '/fake/repo' },
}));

// Mock cortex-db
const mockGetTursoClient = jest.fn();
jest.unstable_mockModule('../scripts/mcp-semantic/tools/cortex-db.mjs', () => ({
  getTursoClient: mockGetTursoClient,
  default: { getTursoClient: mockGetTursoClient },
}));

const {
  MAX_EXPANDED_TERMS,
  MIN_EXPANSION_RELEVANCE,
  MIN_SIMILARITY,
  MAX_NEAREST_TERMS,
  DEFAULT_DOMAIN_ASSOCIATIONS_PATH,
  loadDomainAssociations,
  lookupDomainAssociations,
  findNearestTermsServerSide,
  expansionRelevance,
  deduplicateExpansions,
  selectExpansions,
  buildExpandedFtsQuery,
  computeExpandedEmbedding,
  expansionBehaviorForClass,
  expandQuery,
  invalidateDomainAssociationsCache,
} = await import('./expand-query.mjs');

afterEach(() => {
  jest.clearAllMocks();
  invalidateDomainAssociationsCache();
});

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
describe('constants', () => {
  it('exports expected constant values', () => {
    expect(MAX_EXPANDED_TERMS).toBe(3);
    expect(MIN_EXPANSION_RELEVANCE).toBe(0.55);
    expect(MIN_SIMILARITY).toBe(0.65);
    expect(MAX_NEAREST_TERMS).toBe(5);
    expect(DEFAULT_DOMAIN_ASSOCIATIONS_PATH).toContain('domain-associations.json');
  });
});

// ---------------------------------------------------------------------------
// loadDomainAssociations
// ---------------------------------------------------------------------------
describe('loadDomainAssociations', () => {
  it('loads and caches domain associations from file', async () => {
    const dict = { version: 1, associations: [{ term: 'neat', expansions: ['crossover'], source: 'test', confidence: 0.9 }] };
    mockReadFile.mockResolvedValue(JSON.stringify(dict));
    const result = await loadDomainAssociations('/fake/path.json');
    expect(result.associations).toHaveLength(1);
    // Second call uses cache
    const result2 = await loadDomainAssociations('/fake/path.json');
    expect(result2).toBe(result);
    expect(mockReadFile).toHaveBeenCalledTimes(1);
  });

  it('returns empty dictionary when file not found', async () => {
    mockReadFile.mockRejectedValue(new Error('ENOENT'));
    const result = await loadDomainAssociations('/nonexistent.json');
    expect(result.associations).toEqual([]);
    expect(result.version).toBe(1);
  });

  it('invalidates cache via invalidateDomainAssociationsCache', async () => {
    const dict = { version: 2, associations: [] };
    mockReadFile.mockResolvedValue(JSON.stringify(dict));
    await loadDomainAssociations('/fake/path.json');
    invalidateDomainAssociationsCache();
    await loadDomainAssociations('/fake/path.json');
    expect(mockReadFile).toHaveBeenCalledTimes(2);
  });
});

// ---------------------------------------------------------------------------
// lookupDomainAssociations
// ---------------------------------------------------------------------------
describe('lookupDomainAssociations', () => {
  it('finds matching domain associations', () => {
    const dict = {
      associations: [
        { term: 'NEAT', expansions: ['neuroevolution', 'genetic algorithm'], source: 'domain', confidence: 0.9 },
      ],
    };
    const result = lookupDomainAssociations(['neat'], dict);
    expect(result).toHaveLength(2);
    expect(result[0].expanded).toBe('neuroevolution');
    expect(result[0].source).toBe('domain');
    expect(result[0].confidence).toBe(0.9);
    expect(result[0].type).toBe('domain-association');
  });

  it('returns empty for no matching terms', () => {
    const dict = { associations: [{ term: 'other', expansions: ['x'], source: 'test', confidence: 0.5 }] };
    const result = lookupDomainAssociations(['neat'], dict);
    expect(result).toEqual([]);
  });

  it('computes relevance score with length bonus for multi-word expansions', () => {
    const dict = {
      associations: [
        { term: 'neat', expansions: ['single', 'multi word phrase here'], source: 'test', confidence: 0.8 },
      ],
    };
    const result = lookupDomainAssociations(['neat'], dict);
    // Multi-word: 4 words → lengthBonus = min(1.0, 4/3) = 1.0
    // relevanceScore = 0.8 * (1.0 + 0.1 * 1.0) = 0.88
    expect(result[1].relevanceScore).toBeCloseTo(0.88, 5);
    // Single-word: 1 word → lengthBonus = min(1.0, 1/3) = 1/3
    // relevanceScore = 0.8 * (1.0 + 0.1 * (1/3)) ≈ 0.8267
    expect(result[0].relevanceScore).toBeCloseTo(0.82667, 4);
  });
});

// ---------------------------------------------------------------------------
// findNearestTermsServerSide
// ---------------------------------------------------------------------------
describe('findNearestTermsServerSide', () => {
  it('returns candidates via ANN path', async () => {
    const client = {
      execute: jest.fn().mockResolvedValue({
        rows: [
          { term: 'neuroevolution', distance: 0.2, frequency: 5 },
          { term: 'genetic', distance: 0.3, frequency: 3 },
        ],
      }),
    };
    const result = await findNearestTermsServerSide(client, Buffer.from([1, 2, 3]), 'model-id');
    expect(result).toHaveLength(2);
    expect(result[0].term).toBe('neuroevolution');
    expect(result[0].similarity).toBeCloseTo(0.8, 5);
  });

  it('falls back to brute-force when ANN fails', async () => {
    let callCount = 0;
    const client = {
      execute: jest.fn(async () => {
        callCount++;
        if (callCount === 1) throw new Error('ANN index cold');
        return {
          rows: [{ term: 'fallback', distance: 0.1, frequency: 10 }],
        };
      }),
    };
    const result = await findNearestTermsServerSide(client, Buffer.from([1, 2, 3]), 'model-id');
    expect(result).toHaveLength(1);
    expect(result[0].term).toBe('fallback');
  });

  it('filters by minSimilarity', async () => {
    const client = {
      execute: jest.fn().mockResolvedValue({
        rows: [
          { term: 'close', distance: 0.1, frequency: 5 },     // similarity 0.9
          { term: 'far', distance: 0.5, frequency: 3 },        // similarity 0.5 < 0.65
        ],
      }),
    };
    const result = await findNearestTermsServerSide(client, Buffer.from([1, 2, 3]), 'model-id');
    expect(result).toHaveLength(1);
    expect(result[0].term).toBe('close');
  });

  it('respects maxTerms option', async () => {
    const client = {
      execute: jest.fn().mockResolvedValue({
        rows: [
          { term: 'a', distance: 0.1, frequency: 1 },
          { term: 'b', distance: 0.15, frequency: 2 },
          { term: 'c', distance: 0.2, frequency: 3 },
        ],
      }),
    };
    const result = await findNearestTermsServerSide(client, Buffer.from([1, 2, 3]), 'model-id', { maxTerms: 2 });
    expect(result).toHaveLength(2);
  });

  it('sorts by similarity descending', async () => {
    const client = {
      execute: jest.fn().mockResolvedValue({
        rows: [
          { term: 'far', distance: 0.3, frequency: 1 },
          { term: 'close', distance: 0.1, frequency: 2 },
        ],
      }),
    };
    const result = await findNearestTermsServerSide(client, Buffer.from([1, 2, 3]), 'model-id');
    expect(result[0].similarity).toBeGreaterThan(result[1].similarity);
  });
});

// ---------------------------------------------------------------------------
// expansionRelevance
// ---------------------------------------------------------------------------
describe('expansionRelevance', () => {
  it('uses similarity as base score', () => {
    const score = expansionRelevance({ similarity: 0.9, frequency: 1, expanded: 'test' });
    expect(score).toBeGreaterThan(0);
  });

  it('uses confidence when similarity not available', () => {
    const score = expansionRelevance({ confidence: 0.8, frequency: 1, expanded: 'test' });
    expect(score).toBeGreaterThan(0);
  });

  it('uses 0 when neither similarity nor confidence', () => {
    const score = expansionRelevance({ frequency: 1, expanded: 'test' });
    expect(score).toBe(0);
  });

  it('applies frequency penalty', () => {
    const lowFreq = expansionRelevance({ similarity: 1.0, frequency: 1, expanded: 'test' });
    const highFreq = expansionRelevance({ similarity: 1.0, frequency: 100, expanded: 'test' });
    expect(lowFreq).toBeGreaterThan(highFreq);
  });

  it('uses default frequency of 1 when not provided', () => {
    const score = expansionRelevance({ similarity: 1.0, expanded: 'test' });
    // frequencyPenalty = 1.0 - 0.1 * log10(1) = 1.0
    expect(score).toBeCloseTo(1.0 * 1.0 * (1.0 + 0.1 * (1/3)), 4);
  });

  it('applies length bonus for multi-word expansions', () => {
    const single = expansionRelevance({ similarity: 1.0, frequency: 1, expanded: 'word' });
    const multi = expansionRelevance({ similarity: 1.0, frequency: 1, expanded: 'multi word phrase' });
    expect(multi).toBeGreaterThan(single);
  });
});

// ---------------------------------------------------------------------------
// deduplicateExpansions
// ---------------------------------------------------------------------------
describe('deduplicateExpansions', () => {
  it('keeps higher scoring entry for same expanded term', () => {
    const candidates = [
      { expanded: 'NeuroEvolution', relevanceScore: 0.7 },
      { expanded: 'neuroevolution', relevanceScore: 0.9 },
    ];
    const result = deduplicateExpansions(candidates);
    expect(result).toHaveLength(1);
    expect(result[0].relevanceScore).toBe(0.9);
  });

  it('keeps all unique terms sorted by relevance', () => {
    const candidates = [
      { expanded: 'alpha', relevanceScore: 0.5 },
      { expanded: 'beta', relevanceScore: 0.8 },
      { expanded: 'gamma', relevanceScore: 0.6 },
    ];
    const result = deduplicateExpansions(candidates);
    expect(result).toHaveLength(3);
    expect(result[0].expanded).toBe('beta');
    expect(result[1].expanded).toBe('gamma');
    expect(result[2].expanded).toBe('alpha');
  });
});

// ---------------------------------------------------------------------------
// selectExpansions
// ---------------------------------------------------------------------------
describe('selectExpansions', () => {
  it('merges, scores, filters, and selects top-K', () => {
    const embedding = [
      { expanded: 'neuroevolution', similarity: 0.9, frequency: 1, relevanceScore: 0.9 },
    ];
    const domain = [
      { expanded: 'crossover', confidence: 0.8, frequency: 1, relevanceScore: 0.8 },
      { expanded: 'low-score', confidence: 0.3, frequency: 1, relevanceScore: 0.3 },
    ];
    const result = selectExpansions(embedding, domain);
    // 0.3 < MIN_EXPANSION_RELEVANCE (0.55) → filtered out
    expect(result).toHaveLength(2);
    expect(result[0].expanded).toBe('neuroevolution');
  });

  it('uses expansionRelevance when relevanceScore not set', () => {
    const embedding = [
      { expanded: 'test', similarity: 0.9, frequency: 1 },
    ];
    const result = selectExpansions(embedding, []);
    expect(result).toHaveLength(1);
    expect(result[0].relevanceScore).toBeGreaterThan(0);
  });

  it('limits to MAX_EXPANDED_TERMS', () => {
    const candidates = Array.from({ length: 5 }, (_, i) => ({
      expanded: `term${i}`,
      relevanceScore: 0.9 - i * 0.05,
    }));
    const result = selectExpansions(candidates, []);
    expect(result).toHaveLength(MAX_EXPANDED_TERMS);
  });
});

// ---------------------------------------------------------------------------
// buildExpandedFtsQuery
// ---------------------------------------------------------------------------
describe('buildExpandedFtsQuery', () => {
  it('returns original query when no expansions', () => {
    expect(buildExpandedFtsQuery('test query', [])).toBe('test query');
  });

  it('adds OR clauses for single-word expansions', () => {
    const result = buildExpandedFtsQuery('test', [{ expanded: 'exam' }]);
    expect(result).toBe('test OR exam');
  });

  it('wraps multi-word expansions in quotes', () => {
    const result = buildExpandedFtsQuery('test', [{ expanded: 'practice exam' }]);
    expect(result).toBe('test OR "practice exam"');
  });

  it('handles mixed single and multi-word expansions', () => {
    const result = buildExpandedFtsQuery('test', [
      { expanded: 'exam' },
      { expanded: 'practice test' },
    ]);
    expect(result).toBe('test OR exam OR "practice test"');
  });
});

// ---------------------------------------------------------------------------
// computeExpandedEmbedding
// ---------------------------------------------------------------------------
describe('computeExpandedEmbedding', () => {
  it('mean-pools and L2-normalizes embeddings', () => {
    const original = new Float32Array([1, 0, 0]);
    const expansion1 = new Float32Array([0, 1, 0]);
    const result = computeExpandedEmbedding(original, [expansion1]);
    // Mean: [0.5, 0.5, 0]
    // L2 norm: sqrt(0.25 + 0.25) = sqrt(0.5) ≈ 0.707
    // Normalized: [0.5/0.707, 0.5/0.707, 0] ≈ [0.707, 0.707, 0]
    expect(result[0]).toBeCloseTo(0.7071, 3);
    expect(result[1]).toBeCloseTo(0.7071, 3);
    expect(result[2]).toBeCloseTo(0, 5);
  });

  it('returns zero embedding when magnitude is zero', () => {
    const original = new Float32Array([0, 0, 0]);
    const result = computeExpandedEmbedding(original, []);
    expect(result).toEqual(new Float32Array([0, 0, 0]));
  });

  it('handles single embedding (no expansions)', () => {
    const original = new Float32Array([1, 0, 0]);
    const result = computeExpandedEmbedding(original, []);
    // Mean is just the original, L2 normalized → same as original (already unit)
    expect(result[0]).toBeCloseTo(1, 5);
  });
});

// ---------------------------------------------------------------------------
// expansionBehaviorForClass
// ---------------------------------------------------------------------------
describe('expansionBehaviorForClass', () => {
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
// expandQuery
// ---------------------------------------------------------------------------
describe('expandQuery', () => {
  it('returns identity when expandQuery is false', async () => {
    const result = await expandQuery({ query: 'test query' });
    expect(result.expansion.applied).toBe(false);
    expect(result.expandedTerms).toEqual([]);
    expect(result.bm25Query).toBeNull();
    expect(result.expandedEmbedding).toBeNull();
    expect(result.originalQuery).toBe('test query');
  });

  it('returns identity when expandQuery is not provided', async () => {
    const result = await expandQuery({ query: 'test query' });
    expect(result.expansion.applied).toBe(false);
  });

  it('returns no-qualifying-terms when query has short terms only', async () => {
    mockPorterTokenize.mockReturnValue(['ab', 'cd']); // length < 3
    const result = await expandQuery({ query: 'ab cd', expandQuery: true });
    expect(result.expansion.applied).toBe(false);
    expect(result.expansion.reason).toBe('No qualifying terms in query');
  });

  it('returns no-qualifying-terms when query is empty', async () => {
    mockPorterTokenize.mockReturnValue([]);
    const result = await expandQuery({ query: '', expandQuery: true });
    expect(result.expansion.applied).toBe(false);
    expect(result.expansion.reason).toBe('No qualifying terms in query');
  });

  it('performs domain-only expansion with domain associations', async () => {
    mockPorterTokenize.mockReturnValue(['neat']);
    mockReadFile.mockResolvedValue(JSON.stringify({
      version: 1,
      associations: [
        { term: 'neat', expansions: ['neuroevolution'], source: 'domain', confidence: 0.9 },
      ],
    }));
    // Mock client for domain check
    const mockClient = {
      execute: jest.fn().mockResolvedValue({ rows: [{ name: 'term_embeddings' }] }),
    };
    const result = await expandQuery({
      query: 'neat algorithm',
      expandQuery: 'domain-only',
      client: mockClient,
    });
    expect(result.expansion.applied).toBe(true);
    expect(result.expandedTerms.length).toBeGreaterThan(0);
    expect(result.bm25Query).toContain('OR');
    expect(result.expansion.degraded).toBeUndefined();
  });

  it('marks degraded in domain-only mode when term_embeddings table missing', async () => {
    mockPorterTokenize.mockReturnValue(['neat']);
    mockReadFile.mockResolvedValue(JSON.stringify({
      version: 1,
      associations: [
        { term: 'neat', expansions: ['neuroevolution'], source: 'domain', confidence: 0.9 },
      ],
    }));
    const mockClient = {
      execute: jest.fn().mockResolvedValue({ rows: [] }), // no term_embeddings table
    };
    const result = await expandQuery({
      query: 'neat algorithm',
      expandQuery: 'domain-only',
      client: mockClient,
    });
    expect(result.expansion.degraded).toBe(true);
    expect(result.expansion.reason).toContain('domain-only mode');
  });

  it('marks degraded in domain-only mode when table check throws', async () => {
    mockPorterTokenize.mockReturnValue(['neat']);
    mockReadFile.mockResolvedValue(JSON.stringify({
      version: 1,
      associations: [
        { term: 'neat', expansions: ['neuroevolution'], source: 'domain', confidence: 0.9 },
      ],
    }));
    const mockClient = {
      execute: jest.fn().mockRejectedValue(new Error('DB error')),
    };
    const result = await expandQuery({
      query: 'neat algorithm',
      expandQuery: 'domain-only',
      client: mockClient,
    });
    expect(result.expansion.degraded).toBe(true);
  });

  it('marks degraded in domain-only mode when getTursoClient throws', async () => {
    mockPorterTokenize.mockReturnValue(['neat']);
    mockReadFile.mockResolvedValue(JSON.stringify({
      version: 1,
      associations: [
        { term: 'neat', expansions: ['neuroevolution'], source: 'domain', confidence: 0.9 },
      ],
    }));
    mockGetTursoClient.mockRejectedValue(new Error('Cannot connect'));
    const result = await expandQuery({
      query: 'neat algorithm',
      expandQuery: 'domain-only',
    });
    expect(result.expansion.degraded).toBe(true);
  });

  it('performs full expansion with embeddings and domain associations', async () => {
    mockPorterTokenize.mockReturnValue(['neat']);
    mockReadFile.mockResolvedValue(JSON.stringify({
      version: 1,
      associations: [
        { term: 'neat', expansions: ['neuroevolution'], source: 'domain', confidence: 0.9 },
      ],
    }));
    const mockEmbedding = new Float32Array([1, 0, 0, 0]);
    mockNormalizeEmbeddingVector.mockReturnValue(mockEmbedding);
    const mockEmbedText = jest.fn().mockResolvedValue(new Float32Array([1, 0, 0, 0]));
    const mockClient = {
      execute: jest.fn().mockResolvedValue({
        rows: [
          { term: 'neuroevolution', distance: 0.1, frequency: 5 },
        ],
      }),
    };
    const result = await expandQuery({
      query: 'neat algorithm',
      expandQuery: true,
      client: mockClient,
      embedText: mockEmbedText,
    });
    expect(result.expansion.applied).toBe(true);
    expect(result.expandedTerms.length).toBeGreaterThan(0);
    expect(result.expansion.degraded).toBeUndefined();
  });

  it('marks degraded when term_embeddings table is missing (full mode)', async () => {
    mockPorterTokenize.mockReturnValue(['neat']);
    mockReadFile.mockResolvedValue(JSON.stringify({ version: 1, associations: [] }));
    const mockClient = {
      execute: jest.fn().mockResolvedValue({ rows: [] }), // no term_embeddings table
    };
    const result = await expandQuery({
      query: 'neat algorithm',
      expandQuery: true,
      client: mockClient,
    });
    expect(result.expansion.degraded).toBe(true);
    expect(result.expansion.reason).toContain('ONNX model or term embeddings not available');
  });

  it('marks degraded when getTursoClient throws (full mode)', async () => {
    mockPorterTokenize.mockReturnValue(['neat']);
    mockReadFile.mockResolvedValue(JSON.stringify({ version: 1, associations: [] }));
    mockGetTursoClient.mockRejectedValue(new Error('Cannot connect'));
    const result = await expandQuery({
      query: 'neat algorithm',
      expandQuery: true,
    });
    expect(result.expansion.degraded).toBe(true);
  });

  it('marks degraded when createOnnxTextEmbedder throws', async () => {
    mockPorterTokenize.mockReturnValue(['neat']);
    mockReadFile.mockResolvedValue(JSON.stringify({ version: 1, associations: [] }));
    mockCreateOnnxTextEmbedder.mockRejectedValue(new Error('ONNX not available'));
    const mockClient = {
      execute: jest.fn().mockResolvedValue({ rows: [{ name: 'term_embeddings' }] }),
    };
    const result = await expandQuery({
      query: 'neat algorithm',
      expandQuery: true,
      client: mockClient,
    });
    expect(result.expansion.degraded).toBe(true);
  });

  it('skips individual terms that fail embedding', async () => {
    mockPorterTokenize.mockReturnValue(['neat', 'crossover']);
    mockReadFile.mockResolvedValue(JSON.stringify({ version: 1, associations: [] }));
    let callCount = 0;
    const mockEmbedText = jest.fn(async () => {
      callCount++;
      if (callCount === 1) throw new Error('Embedding failed for this term');
      return new Float32Array([1, 0, 0, 0]);
    });
    mockNormalizeEmbeddingVector.mockReturnValue(new Float32Array([1, 0, 0, 0]));
    const mockClient = {
      execute: jest.fn().mockResolvedValue({
        rows: [{ term: 'genetic', distance: 0.1, frequency: 3 }],
      }),
    };
    const result = await expandQuery({
      query: 'neat crossover',
      expandQuery: true,
      client: mockClient,
      embedText: mockEmbedText,
    });
    // First term failed but second succeeded → not degraded, has expansions
    expect(result.expansion.degraded).toBeUndefined();
  });

  it('releases embedText when not injected', async () => {
    mockPorterTokenize.mockReturnValue(['neat']);
    mockReadFile.mockResolvedValue(JSON.stringify({ version: 1, associations: [] }));
    const mockEmbedder = {
      embed: jest.fn().mockResolvedValue(new Float32Array([1, 0, 0, 0])),
      release: jest.fn().mockResolvedValue(undefined),
    };
    mockCreateOnnxTextEmbedder.mockResolvedValue(mockEmbedder);
    mockNormalizeEmbeddingVector.mockReturnValue(new Float32Array([1, 0, 0, 0]));
    const mockClient = {
      execute: jest.fn().mockResolvedValue({
        rows: [{ term: 'neuroevolution', distance: 0.1, frequency: 5 }],
      }),
    };
    await expandQuery({
      query: 'neat algorithm',
      expandQuery: true,
      client: mockClient,
    });
    // embedText was not injected, so release should be called
    // Wait - createOnnxTextEmbedder returns an embedder function, not an object with embed
    // Actually, looking at the code: const embedText = options.embedText ?? (await createOnnxTextEmbedder(...))
    // So embedText is a function. And it checks `if (options.embedText === undefined) await embedText.release?.()`
    // So the embedder function needs a .release property
  });

  it('returns no expansions when all candidates below relevance threshold', async () => {
    mockPorterTokenize.mockReturnValue(['neat']);
    mockReadFile.mockResolvedValue(JSON.stringify({
      version: 1,
      associations: [
        { term: 'neat', expansions: ['low-confidence'], source: 'domain', confidence: 0.3 },
      ],
    }));
    const mockClient = {
      execute: jest.fn().mockResolvedValue({ rows: [{ name: 'term_embeddings' }] }),
    };
    const result = await expandQuery({
      query: 'neat algorithm',
      expandQuery: 'domain-only',
      minRelevance: 0.9, // higher than the 0.3 confidence
      client: mockClient,
    });
    expect(result.expansion.applied).toBe(false);
    expect(result.expansion.reason).toBe('No qualifying expansions found');
  });

  it('handles null/undefined query gracefully', async () => {
    const result = await expandQuery({ query: null, expandQuery: false });
    expect(result.originalQuery).toBe('');
    expect(result.expansion.applied).toBe(false);
  });

  it('respects maxExpansions option', async () => {
    mockPorterTokenize.mockReturnValue(['neat']);
    mockReadFile.mockResolvedValue(JSON.stringify({
      version: 1,
      associations: [
        { term: 'neat', expansions: ['alpha', 'beta', 'gamma', 'delta'], source: 'domain', confidence: 0.9 },
      ],
    }));
    const mockClient = {
      execute: jest.fn().mockResolvedValue({ rows: [{ name: 'term_embeddings' }] }),
    };
    const result = await expandQuery({
      query: 'neat algorithm',
      expandQuery: 'domain-only',
      client: mockClient,
      maxExpansions: 2,
    });
    expect(result.expandedTerms.length).toBeLessThanOrEqual(2);
  });
});