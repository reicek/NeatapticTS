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

BM25 full-text search over indexed chunks.

Schema:

```json
{
  "type": "object",
  "properties": {
    "query": { "type": "string" },
    "limit": { "type": "number" },
    "family": { "type": "string" }
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
    "query": "NEAT activation",
    "limit": 3
  }
}
```

Representative output:

```json
{
  "query": "NEAT activation",
  "limit": 3,
  "results": [
    {
      "chunk_id": 42,
      "file_path": "src/architecture/network/README.md",
      "family": "src",
      "chunk_index": 0,
      "heading_path": "Network",
      "text": "...activation...",
      "char_start": 0,
      "char_end": 1200,
      "score": -8.31
    }
  ]
}
```

Use `family` when the search should stay inside one indexed document family, such as `src`, `examples`, `scripts`, or `plans` when present in the current index.

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