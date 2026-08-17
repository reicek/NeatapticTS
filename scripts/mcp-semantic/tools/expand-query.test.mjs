/**
 * @module expand-query.test
 * @description Coverage tests for expand-query.mjs MCP tool handler.
 */
import { jest } from '@jest/globals';

// Mock the upstream expand-query module before importing the tool.
const mockExpandQuery = jest.fn();
const mockExpansionBehaviorForClass = jest.fn();

jest.unstable_mockModule('../../../rag-index/expand-query.mjs', () => ({
  expandQuery: mockExpandQuery,
  expansionBehaviorForClass: mockExpansionBehaviorForClass,
  __esModule: true,
}));

const { expandQueryHandler } = await import('./expand-query.mjs');

describe('expand-query handler', () => {
  beforeEach(() => {
    mockExpandQuery.mockReset();
    mockExpansionBehaviorForClass.mockReset();
  });

  it('returns empty result when query is empty string', async () => {
    const result = await expandQueryHandler({ query: '' });
    expect(result).toEqual({
      original_query: '',
      expanded_terms: [],
      bm25_query: null,
      expansion: {
        applied: false,
        reason: 'Query is empty',
      },
    });
    expect(mockExpandQuery).not.toHaveBeenCalled();
  });

  it('returns empty result when query is whitespace only', async () => {
    const result = await expandQueryHandler({ query: '   ' });
    expect(result.original_query).toBe('');
    expect(result.expansion.applied).toBe(false);
  });

  it('returns empty result when query is undefined', async () => {
    const result = await expandQueryHandler({});
    expect(result.original_query).toBe('');
  });

  it('returns empty result when no arguments provided', async () => {
    const result = await expandQueryHandler();
    expect(result.original_query).toBe('');
  });

  it('calls expandQuery with default expand_query=true', async () => {
    mockExpandQuery.mockResolvedValue({
      originalQuery: 'test query',
      expandedTerms: ['term1', 'term2'],
      bm25Query: 'test OR term1 OR term2',
      expansion: { applied: true, reason: 'ok' },
    });

    const result = await expandQueryHandler({ query: 'test query' });

    expect(mockExpandQuery).toHaveBeenCalledWith({
      query: 'test query',
      expandQuery: true,
    });
    expect(result.original_query).toBe('test query');
    expect(result.expanded_terms).toEqual(['term1', 'term2']);
    expect(result.bm25_query).toBe('test OR term1 OR term2');
    expect(result.expansion.applied).toBe(true);
  });

  it('respects expand_query=false', async () => {
    mockExpandQuery.mockResolvedValue({
      originalQuery: 'test',
      expandedTerms: [],
      bm25Query: null,
      expansion: { applied: false, reason: 'disabled' },
    });

    const result = await expandQueryHandler({
      query: 'test',
      expand_query: false,
    });

    expect(mockExpandQuery).toHaveBeenCalledWith({
      query: 'test',
      expandQuery: false,
    });
    expect(result.expansion.applied).toBe(false);
  });

  it('respects expand_query as string "domain-only"', async () => {
    mockExpandQuery.mockResolvedValue({
      originalQuery: 'test',
      expandedTerms: ['domain1'],
      bm25Query: 'test OR domain1',
      expansion: { applied: true, reason: 'domain-only' },
    });

    const result = await expandQueryHandler({
      query: 'test',
      expand_query: 'domain-only',
    });

    expect(mockExpandQuery).toHaveBeenCalledWith({
      query: 'test',
      expandQuery: 'domain-only',
    });
    expect(result.expansion.applied).toBe(true);
  });

  it('maps query_class to false behavior', async () => {
    mockExpansionBehaviorForClass.mockReturnValue(false);
    mockExpandQuery.mockResolvedValue({
      originalQuery: 'test',
      expandedTerms: [],
      bm25Query: null,
      expansion: { applied: false, reason: 'disabled by class' },
    });

    const result = await expandQueryHandler({
      query: 'test',
      query_class: 'simple_lookup',
    });

    expect(mockExpansionBehaviorForClass).toHaveBeenCalledWith(
      'simple_lookup',
    );
    expect(mockExpandQuery).toHaveBeenCalledWith({
      query: 'test',
      expandQuery: false,
    });
    expect(result.expansion.applied).toBe(false);
  });

  it('maps query_class to domain-only behavior', async () => {
    mockExpansionBehaviorForClass.mockReturnValue('domain-only');
    mockExpandQuery.mockResolvedValue({
      originalQuery: 'test',
      expandedTerms: ['domain1'],
      bm25Query: 'test OR domain1',
      expansion: { applied: true, reason: 'domain-only expansion' },
    });

    const result = await expandQueryHandler({
      query: 'test',
      query_class: 'code_specific',
    });

    expect(mockExpandQuery).toHaveBeenCalledWith({
      query: 'test',
      expandQuery: 'domain-only',
    });
    expect(result.expansion.applied).toBe(true);
  });

  it('maps query_class to true behavior (keeps expandQuery=true)', async () => {
    mockExpansionBehaviorForClass.mockReturnValue(true);
    mockExpandQuery.mockResolvedValue({
      originalQuery: 'test',
      expandedTerms: ['expanded'],
      bm25Query: 'test OR expanded',
      expansion: { applied: true, reason: 'full expansion' },
    });

    const result = await expandQueryHandler({
      query: 'test',
      query_class: 'cross_boundary',
    });

    expect(mockExpandQuery).toHaveBeenCalledWith({
      query: 'test',
      expandQuery: true,
    });
    expect(result.expansion.applied).toBe(true);
  });

  it('does NOT apply query_class mapping when expand_query is already false', async () => {
    mockExpandQuery.mockResolvedValue({
      originalQuery: 'test',
      expandedTerms: [],
      bm25Query: null,
      expansion: { applied: false, reason: 'disabled' },
    });

    await expandQueryHandler({
      query: 'test',
      expand_query: false,
      query_class: 'cross_boundary',
    });

    // expansionBehaviorForClass should NOT be called because expandQueryOption is false, not true
    expect(mockExpansionBehaviorForClass).not.toHaveBeenCalled();
    expect(mockExpandQuery).toHaveBeenCalledWith({
      query: 'test',
      expandQuery: false,
    });
  });

  it('does NOT apply query_class mapping when expand_query is string', async () => {
    mockExpandQuery.mockResolvedValue({
      originalQuery: 'test',
      expandedTerms: [],
      bm25Query: null,
      expansion: { applied: true, reason: 'ok' },
    });

    await expandQueryHandler({
      query: 'test',
      expand_query: 'domain-only',
      query_class: 'cross_boundary',
    });

    expect(mockExpansionBehaviorForClass).not.toHaveBeenCalled();
  });

  it('returns degraded result when expandQuery throws', async () => {
    mockExpandQuery.mockRejectedValue(new Error('ONNX model not available'));

    const result = await expandQueryHandler({ query: 'test query' });

    expect(result.original_query).toBe('test query');
    expect(result.expanded_terms).toEqual([]);
    expect(result.bm25_query).toBeNull();
    expect(result.expansion.applied).toBe(false);
    expect(result.expansion.degraded).toBe(true);
    expect(result.expansion.reason).toContain('Expansion failed');
    expect(result.expansion.reason).toContain('ONNX model not available');
  });

  it('forwards injected client and adds embeddings_source metadata (empty query)', async () => {
    const client = { execute: jest.fn() };
    const result = await expandQueryHandler({ query: '', client });

    expect(result.expansion.embeddings_source).toBe('injected-client');
  });

  it('forwards injected client to expandQuery and adds embeddings_source', async () => {
    const client = { execute: jest.fn() };
    mockExpandQuery.mockResolvedValue({
      originalQuery: 'test',
      expandedTerms: ['term1'],
      bm25Query: 'test OR term1',
      expansion: { applied: true, reason: 'ok' },
    });

    const result = await expandQueryHandler({ query: 'test', client });

    expect(mockExpandQuery).toHaveBeenCalledWith({
      query: 'test',
      expandQuery: true,
      client,
    });
    expect(result.expansion.embeddings_source).toBe('injected-client');
  });

  it('adds embeddings_source to degraded result with injected client', async () => {
    const client = { execute: jest.fn() };
    mockExpandQuery.mockRejectedValue(new Error('fail'));

    const result = await expandQueryHandler({ query: 'test', client });

    expect(result.expansion.embeddings_source).toBe('injected-client');
    expect(result.expansion.degraded).toBe(true);
  });

  it('passes through full expansion metadata from expandQuery result', async () => {
    mockExpandQuery.mockResolvedValue({
      originalQuery: 'test',
      expandedTerms: ['a', 'b'],
      bm25Query: 'test OR a OR b',
      expansion: {
        applied: true,
        degraded: false,
        reason: 'success',
        custom_field: 'value',
      },
    });

    const result = await expandQueryHandler({ query: 'test' });

    expect(result.expansion).toEqual({
      applied: true,
      degraded: false,
      reason: 'success',
      custom_field: 'value',
    });
  });
});