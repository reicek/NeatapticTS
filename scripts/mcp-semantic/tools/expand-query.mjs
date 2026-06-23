/**
 * @module expand-query-mcp-tool
 * @description MCP tool handler for the query expansion pipeline.
 *
 * Wraps the `expandQuery` function from `expand-query.mjs` as an MCP tool
 * that can be invoked directly by VS Code's MCP host. Handles parameter
 * coercion, error isolation, and result normalization.
 */
import { expandQuery } from '../../semantic-index/expand-query.mjs';

/**
 * Handle an expand_query MCP tool invocation.
 *
 * Accepts the standard MCP `argumentsObject` and delegates to
 * {@link expandQuery} from the expansion pipeline. When the caller
 * does not provide `expand_query`, defaults to `true` (since calling
 * this tool implies the caller wants expansion). When `query_class`
 * is provided, maps it to the appropriate expansion behavior using
 * `expansionBehaviorForClass`.
 *
 * @param {{ query?: string, expand_query?: boolean | string, query_class?: string }} argumentsObject - MCP tool arguments.
 * @returns {Promise<object>} Expansion result with `original_query`, `expanded_terms`, `bm25_query`, and `expansion` metadata.
 */
export async function expandQueryHandler(argumentsObject = {}) {
  const query = String(argumentsObject.query ?? '').trim();
  // When an injected async `client` is provided, mark the expansion so callers
  // know the client will be forwarded to downstream DB access (Step 05 wires
  // the actual client forwarding into semantic-index/expand-query.mjs).
  const injectedClient = argumentsObject.client;
  if (!query) {
    return {
      original_query: '',
      expanded_terms: [],
      bm25_query: null,
      expansion: {
        applied: false,
        reason: 'Query is empty',
        ...(injectedClient ? { embeddings_source: 'injected-client' } : {}),
      },
    };
  }

  // Default expand_query to true when this tool is called directly
  let expandQueryOption = argumentsObject.expand_query ?? true;

  // Map query_class to expansion behavior when provided
  if (argumentsObject.query_class && expandQueryOption === true) {
    const { expansionBehaviorForClass } =
      await import('../../semantic-index/expand-query.mjs');
    const behavior = expansionBehaviorForClass(argumentsObject.query_class);
    if (behavior === false) {
      expandQueryOption = false;
    } else if (behavior === 'domain-only') {
      expandQueryOption = 'domain-only';
    }
    // behavior === true → keep expandQueryOption = true
  }

  try {
    const result = await expandQuery({
      query,
      expandQuery: expandQueryOption,
      ...(injectedClient ? { client: injectedClient } : {}),
    });

    return {
      original_query: result.originalQuery,
      expanded_terms: result.expandedTerms,
      bm25_query: result.bm25Query,
      expansion: {
        ...result.expansion,
        ...(injectedClient ? { embeddings_source: 'injected-client' } : {}),
      },
    };
  } catch (error) {
    return {
      original_query: query,
      expanded_terms: [],
      bm25_query: null,
      expansion: {
        applied: false,
        degraded: true,
        reason: `Expansion failed: ${error.message}`,
        ...(injectedClient ? { embeddings_source: 'injected-client' } : {}),
      },
    };
  }
}
