# Repo Cortex MCP Server

`neataptic-cortex-mcp` exposes the Repo Cortex semantic index as read-only MCP tools. It is a local `stdio` server over the repository's SQLite corpus index, so agents can search, load, and validate indexed repository context without walking the raw filesystem for every question.

The server is intentionally bounded to repo-static and direct-MCP facts. It reads indexed files, chunk metadata, freshness proofs, and aggregate corpus counts from `data/semantic-index.sqlite`. It does not read live VS Code UI state, Copilot client state, selected agent state, tool-picker state, or model-selection state; those remain outside this direct MCP surface unless a future documented bridge supplies them with source and freshness metadata.

## Relationship to the MCP Server Set

The workspace registers four sibling MCP servers. Repo Cortex adds semantic corpus access without replacing the workflow, validation, or gate servers.

| Server | Boundary | What it answers |
|---|---|---|
| `neataptic-workflow-mcp` | Repo-static workflow context | Active plan and deterministic workflow inventory facts. |
| `neataptic-validation-mcp` | Direct validation gate execution | Exact allow-listed validation commands from the active step packet. |
| `neataptic-gate-mcp` | Release gate contracts | Gate metadata and contract-oriented release checks. |
| `neataptic-cortex-mcp` | Repo-static semantic corpus | BM25 search, chunk/document loading, index freshness, and corpus statistics. |

## Index Configuration

The server opens the semantic index in read-only mode. The MCP registration provides the default index path through `CORTEX_DB_PATH`:

```json
{
  "env": {
    "CORTEX_DB_PATH": "${workspaceFolder}/data/semantic-index.sqlite"
  }
}
```

For direct script runs, you can also pass `--databasePath=<path>` or set `CORTEX_DB_PATH` in the environment. If the index is missing or stale, rebuild it from repository sources:

```powershell
node scripts/semantic-index/build-index.mjs
```

## Tool Schemas and Examples

The examples below show representative `tools/call` arguments and shortened outputs. Exact counts, chunk IDs, scores, and timestamps depend on the current `data/semantic-index.sqlite` build.

### `search_corpus`

BM25 full-text search over indexed chunks with default-on hybrid dense reranking and compact, LLM-friendly output.

By default the MCP handler sets `compact: true`, so the response contains only the fields an agent typically needs for a relevance decision: `chunk_id`, `file_path`, `family`, `chunk_index`, `heading_path`, `context_header`, `symbol_name`, a truncated text snippet, the `score`, and any `ranking_explanation`. Set `compact: false` to receive the full per-result metadata. Every response also carries a `freshness` stanza that proves the index state (timestamp, stale flag, last indexed document, and an `mtime_ms`/`size`/`sha256` freshness proof).

Schema:

```json
{
  "type": "object",
  "properties": {
    "query": { "type": "string" },
    "limit": { "type": "number", "description": "Maximum result count (default 5, clamped to [1, 100])." },
    "family": { "type": "string", "description": "Optional document family filter, e.g. src, examples, scripts, plans." },
    "use_dense": { "type": "boolean", "default": true, "description": "Enable hybrid dense reranking when the index is warm." },
    "alpha": { "type": "number", "description": "BM25/dense blend weight (0 = BM25 only, 1 = dense only)." },
    "query_class": { "type": "string", "enum": ["simple_lookup", "cross_boundary", "multi_hop", "exploratory", "code_specific", "plan_specific"], "description": "Override query classification for routing." },
    "classification_hints": { "type": "object", "description": "Optional overrides for classification-derived alpha and family." },
    "expand_query": { "type": ["boolean", "string"], "description": "Enable query expansion: true for full expansion, 'domain-only' for domain associations only, false for no expansion." },
    "use_rerank": { "type": "boolean", "default": false, "description": "Enable cross-encoder re-ranking of the top hybrid candidates." },
    "rerank_candidates_count": { "type": "number", "description": "Number of hybrid candidates to feed the reranker (default 50)." },
    "compact": { "type": "boolean", "default": true, "description": "When true, return only essential fields and truncate text snippets." },
    "metadata": { "type": "object", "description": "Optional structured metadata filter predicate tree." }
  },
  "required": ["query"],
  "additionalProperties": false
}
```

Example call:

```json
{
  "name": "search_corpus",
  "arguments": {
    "query": "network.activate",
    "limit": 3
  }
}
```

Representative output:

```json
{
  "query": "network.activate",
  "limit": 3,
  "use_dense": true,
  "compact": true,
  "query_class": "code_specific",
  "confidence": 0.92,
  "family": "ts-source",
  "dense_state": "warm",
  "freshness": {
    "timestamp": 1779540000000,
    "stale": false,
    "last_update_source": "corpus_index",
    "last_indexed_at": 1779539000000,
    "freshness_proof": {
      "mtime_ms": 1779539000000,
      "size": 12345,
      "sha256": "..."
    }
  },
  "results": [
    {
      "chunk_id": 42,
      "file_path": "src/architecture/network/activate/network.activate.ts",
      "family": "ts-source",
      "chunk_index": 0,
      "heading_path": "activate",
      "context_header": "[src/architecture/network/activate/network.activate.ts > activate]",
      "symbol_name": "activate",
      "text": "export function activate(network, input) { ... }",
      "score": -8.31,
      "feedback_boost": 0.05
    }
  ]
}
```

Use `family` when the search should stay inside one indexed document family. When `query_class` is omitted, the server classifies the query automatically; `code_specific` queries boost the `ts-source` family by default so source chunks rank above generated READMEs.

### `search_context`

Hybrid search plus context assembly. The tool runs a search, then stitches the best matching chunks into a single token-bounded context window suitable for a language-model prompt. It is the fastest way to turn a question into a compact block of relevant source text.

The MCP handler defaults `compact` to `true` and `read_top_result` to `true`. The assembled `context` is always included; when `read_top_result` is true, `top_result` carries the full metadata for the best hit, and `follow_up_refs` lists related chunks and symbols discovered through graph traversal so the caller can dig deeper without a second search.

Schema:

```json
{
  "type": "object",
  "properties": {
    "query": { "type": "string" },
    "limit": { "type": "number", "description": "Maximum result count before context assembly (default 5)." },
    "budget": { "type": "number", "default": 1024, "description": "Approximate output-token budget for the assembled context." },
    "context_format": { "type": "string", "default": "text", "enum": ["text", "json"], "description": "Format of the returned context string." },
    "include_metadata": { "type": "boolean", "default": false, "description": "Return full v2 metadata for every chunk in the results array." },
    "dedup_strategy": { "type": "string", "default": "cosine", "enum": ["exact", "cosine"], "description": "Collapse near-duplicate chunks by hash or embedding similarity." },
    "expand_query": { "type": ["boolean", "string"], "description": "Enable query expansion before search." },
    "use_dense": { "type": "boolean", "default": true, "description": "Enable hybrid dense reranking." },
    "alpha": { "type": "number", "description": "BM25/dense blend weight." },
    "use_rerank": { "type": "boolean", "default": false, "description": "Enable cross-encoder re-ranking." },
    "family": { "type": "string", "description": "Optional family filter." },
    "compact": { "type": "boolean", "default": true, "description": "Return essential fields only for search results." },
    "read_top_result": { "type": "boolean", "default": true, "description": "Return full metadata and content for the highest-ranked result." },
    "follow_up_refs": { "type": "boolean", "default": true, "description": "Include graph-discovered related chunks and symbols." }
  },
  "required": ["query"],
  "additionalProperties": false
}
```

Example call:

```json
{
  "name": "search_context",
  "arguments": {
    "query": "network.activate",
    "budget": 2048,
    "read_top_result": true,
    "follow_up_refs": true
  }
}
```

Representative output:

```json
{
  "query": "network.activate",
  "budget": 2048,
  "context": "# [src/architecture/network/activate/network.activate.ts > activate]\n\nexport function activate(network, input) { ... }\n...",
  "context_tokens": 412,
  "top_result": {
    "chunk_id": 42,
    "file_path": "src/architecture/network/activate/network.activate.ts",
    "family": "ts-source",
    "symbol_name": "activate",
    "text": "export function activate(network, input) { ... }",
    "score": -8.31
  },
  "follow_up_refs": [
    { "chunk_id": 43, "relationship": "parent_chunk", "symbol_name": "propagate" },
    { "chunk_id": 44, "relationship": "references", "symbol_name": "Network" }
  ],
  "dense_state": "warm",
  "freshness": { "timestamp": 1779540000000, "stale": false, "last_indexed_at": 1779539000000 }
}
```

Use `search_context` when the next step is to pass evidence directly to a language model; use `search_corpus` when you only need a ranked result list.

### `search_advanced`

Full-pipeline search with classification-aware defaults, automatic fallback, and optional result explanations.

`auto_fallback` defaults to `true`: if the primary search returns no relevant results, the tool automatically rewrites the query (for example by removing code punctuation or falling back to BM25-only retrieval) and appends a second result set. `include_code_only` defaults to `true` for `code_specific` queries so only source-code chunks are returned. Set `explain_ranking: true` to receive a `ranking_explanation` field that describes why each result was selected.

Schema:

```json
{
  "type": "object",
  "properties": {
    "query": { "type": "string" },
    "limit": { "type": "number", "description": "Maximum result count (default 5)." },
    "context_budget": { "type": "number", "description": "Token budget when assembling a context window." },
    "query_class": { "type": "string", "enum": ["simple_lookup", "cross_boundary", "multi_hop", "exploratory", "code_specific", "plan_specific"], "description": "Override query classification." },
    "classification_hints": { "type": "object", "description": "Override classification-derived alpha and family." },
    "alpha": { "type": "number", "description": "BM25/dense blend weight." },
    "use_dense": { "type": "boolean", "default": true, "description": "Enable hybrid dense reranking." },
    "expand_query": { "type": ["boolean", "string"], "description": "Enable query expansion." },
    "use_rerank": { "type": "boolean", "default": false, "description": "Enable cross-encoder re-ranking." },
    "rerank_candidates_count": { "type": "number", "description": "Number of hybrid candidates for reranking." },
    "include_metadata": { "type": "boolean", "default": false, "description": "Return full v2 metadata per chunk." },
    "dedup_strategy": { "type": "string", "default": "cosine", "enum": ["exact", "cosine"] },
    "family": { "type": "string", "description": "Optional family filter." },
    "compact": { "type": "boolean", "default": true, "description": "Return essential fields only." },
    "read_top_result": { "type": "boolean", "default": true, "description": "Return full top-result metadata." },
    "follow_up_refs": { "type": "boolean", "default": true, "description": "Include graph-discovered related chunks and symbols." },
    "auto_fallback": { "type": "boolean", "default": true, "description": "Automatically retry with a fallback query if the primary search is empty." },
    "include_code_only": { "type": "boolean", "description": "For code_specific queries, return only source-code chunks. Defaults to true." },
    "explain_ranking": { "type": "boolean", "default": false, "description": "Return a human-readable ranking_explanation." },
    "metadata": { "type": "object", "description": "Structured metadata filter predicate tree." }
  },
  "required": ["query"],
  "additionalProperties": false
}
```

Example call:

```json
{
  "name": "search_advanced",
  "arguments": {
    "query": "How does network.activate handle recurrent connections?",
    "limit": 5,
    "auto_fallback": true,
    "explain_ranking": true
  }
}
```

Representative output:

```json
{
  "query": "How does network.activate handle recurrent connections?",
  "query_class": "code_specific",
  "confidence": 0.92,
  "auto_fallback": true,
  "results": [
    {
      "chunk_id": 42,
      "file_path": "src/architecture/network/activate/network.activate.ts",
      "family": "ts-source",
      "symbol_name": "activate",
      "text": "export function activate(network, input) { ... }",
      "score": -8.31
    }
  ],
  "fallback": {
    "triggered": false,
    "results": []
  },
  "ranking_explanation": "Selected because the chunk directly matches the identifier 'activate' and contains recurrent-connection handling in the same file.",
  "dense_state": "warm",
  "freshness": { "timestamp": 1779540000000, "stale": false, "last_indexed_at": 1779539000000 },
  "latency_ms": 120
}
```

### `load_chunk`

Load one indexed chunk by numeric chunk ID.

Schema:

```json
{
  "type": "object",
  "properties": {
    "chunk_id": { "type": "number" }
  },
  "required": ["chunk_id"],
  "additionalProperties": false
}
```

Example call:

```json
{
  "name": "load_chunk",
  "arguments": {
    "chunk_id": 42
  }
}
```

Representative output:

```json
{
  "chunk": {
    "chunk_id": 42,
    "file_path": "src/architecture/network/README.md",
    "family": "src",
    "chunk_index": 0,
    "heading_path": "Network",
    "text": "...full indexed chunk text...",
    "char_start": 0,
    "char_end": 1200
  }
}
```

### `load_document`

Load all ordered chunks for one indexed repository path.

Schema:

```json
{
  "type": "object",
  "properties": {
    "file_path": { "type": "string" }
  },
  "required": ["file_path"],
  "additionalProperties": false
}
```

Example call:

```json
{
  "name": "load_document",
  "arguments": {
    "file_path": "plans/Semantic_Knowledge_MCP_Tools.plans.md"
  }
}
```

Representative output:

```json
{
  "file_path": "plans/Semantic_Knowledge_MCP_Tools.plans.md",
  "chunks": [
    {
      "chunk_id": 9001,
      "file_path": "plans/Semantic_Knowledge_MCP_Tools.plans.md",
      "family": "plans",
      "chunk_index": 0,
      "heading_path": "Semantic Knowledge MCP Tools",
      "text": "...first chunk...",
      "char_start": 0,
      "char_end": 1500
    }
  ]
}
```

The `file_path` value must stay inside the repository and is normalized to POSIX-style separators.

### `freshness_check`

Compare indexed freshness metadata with current filesystem metadata for one document or for all indexed documents.

Schema:

```json
{
  "type": "object",
  "properties": {
    "file_path": { "type": "string" },
    "freshnessProof": { "type": "object" }
  },
  "additionalProperties": false
}
```

Example call:

```json
{
  "name": "freshness_check",
  "arguments": {
    "file_path": "README.md"
  }
}
```

Representative output:

```json
{
  "fresh": true,
  "stale": [],
  "documents": [
    {
      "file_path": "README.md",
      "fresh": true,
      "indexed": {
        "mtime_ms": 1779540000000,
        "file_size": 12345,
        "sha256": "..."
      },
      "current": {
        "mtime_ms": 1779540000000,
        "file_size": 12345,
        "sha256": "..."
      }
    }
  ]
}
```

When `file_path` is omitted, the tool checks every indexed document. A supplied `freshnessProof` is intended for deterministic tests or direct validation scenarios; normal calls let the tool compute the current proof from the filesystem.

### `index_stats`

Return corpus row counts and the last indexed timestamp.

Schema: no arguments.

Example call:

```json
{
  "name": "index_stats",
  "arguments": {}
}
```

Representative output:

```json
{
  "total_documents": 831,
  "total_chunks": 27703,
  "total_families": 9,
  "last_build_timestamp": "2026-05-23T00:00:00.000Z"
}
```

### `list_families`

List indexed document families with document and chunk counts.

Schema: no arguments.

Example call:

```json
{
  "name": "list_families",
  "arguments": {}
}
```

Representative output:

```json
{
  "families": [
    {
      "family": "plans",
      "documents": 25,
      "chunks": 740
    },
    {
      "family": "src",
      "documents": 320,
      "chunks": 16000
    }
  ]
}
```

## Query routing and tokenization

The search tools classify each query automatically unless `query_class` is supplied explicitly. The classifier chooses a default `alpha` (BM25/dense blend), a default `family`, and whether to expand the query or run cross-encoder reranking. For example, `code_specific` queries such as `network.activate` or `Network.prototype.activate` default to the `ts-source` family, keep expansion off, and bias toward BM25 so identifier matches carry more weight than semantic paraphrases.

The corpus index also applies identifier-aware tokenization to source text. `camelCase`, `snake_case`, and dotted identifiers are indexed as single terms in addition to being split, so a search for `network.activate` can match the exact dotted identifier rather than only the individual words. This is why short, symbol-like queries often return the most relevant source chunk first even without explicit family filters.

If the classifier is uncertain, set `query_class` and `classification_hints` to take control. For broad conceptual questions, `exploratory` enables expansion and reranking by default; for a narrow symbol lookup, `simple_lookup` or `code_specific` keeps the query tight.

## Local Checks

Use the server help output to confirm the registered tool list:

```powershell
node scripts/mcp-semantic/repo-cortex-mcp.mjs --help
```

Use the smoke gate to confirm the configured index exists, contains chunks, and can satisfy a representative search:

```powershell
node scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json
```

If the smoke gate reports a missing index, rebuild first:

```powershell
node scripts/semantic-index/build-index.mjs
```