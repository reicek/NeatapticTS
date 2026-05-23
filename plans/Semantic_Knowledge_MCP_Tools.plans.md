# Semantic Knowledge MCP Tools (Repo Cortex — Layer 2)

**Status:** [PLANNED]

> Exposes the Repo Cortex SQLite corpus index as first-class MCP tools for AI agents
> helping with the full NeatapticTS library, all demos, and the agentic workflow itself.
> Depends on the archived [Semantic_Knowledge_Foundation.plans.md](completed/Semantic_Knowledge_Foundation.plans.md) baseline.

## Purpose

Create a fourth MCP server (`neataptic-cortex-mcp`) that exposes the semantic corpus index
as structured MCP tools. The tools enable AI agents to perform fast BM25 context search,
load specific corpus chunks, check index freshness, and inspect corpus statistics — all
without traversing the raw filesystem or re-reading large files.

**Scope of the corpus:** the whole NeatapticTS library, all demos (Flappy Bird, ASCII Maze,
NeatChat, and future demos), all scripts, and all agentic workflow artifacts. This MCP server
serves the entire repo, not any single demo.

**Relationship to existing MCP servers:**

| Server | Purpose |
|---|---|
| `neataptic-workflow-mcp` | Plan/flow context loading for agentic workflow |
| `neataptic-validation-mcp` | Gate and allow-list validation |
| `neataptic-gate-mcp` | Release gate contracts |
| `neataptic-cortex-mcp` | **Semantic search over the full corpus (this plan)** |

## Non-goals

- Browser snapshot or IndexedDB exposure (see `Semantic_Knowledge_Browser_Snapshot.plans.md`).
- Dense embedding search (see `Semantic_Knowledge_Embeddings.plans.md`).
- Changes to `src/` library code or existing MCP servers.
- NeatChat-internal conversational memory (separate plan: `NeatChat_Local_Retrieval_Memory.plans.md`).
- Replacing existing `loadActivePlanContext` or plan-binding tools; this is additive.

## Dependencies

- [Semantic_Knowledge_Foundation.plans.md](completed/Semantic_Knowledge_Foundation.plans.md) [DONE] —
  `data/semantic-index.sqlite` must exist and be queryable before this server is usable.
- Existing MCP server infrastructure in `scripts/agent-customization/mcp/`.
- `.vscode/mcp.json` registration (additive, does not replace existing servers).

## Scope

### MCP tools exposed by `neataptic-cortex-mcp`

| Tool name | Description | Key parameters |
|---|---|---|
| `search_corpus` | BM25 full-text search over all indexed chunks | `query: string`, `limit?: number`, `family?: string` |
| `load_chunk` | Load the full text of a specific chunk by ID | `chunk_id: number` |
| `load_document` | Load all chunks for a specific file path | `file_path: string` |
| `freshness_check` | Return freshness status for one or all documents | `file_path?: string` |
| `index_stats` | Return row counts, doc families, and last-build timestamp | — |
| `list_families` | List all indexed document families with counts | — |

### MCP resources (optional)

- `cortex://stats` — live JSON index statistics snapshot.
- `cortex://families` — enumeration of corpus document families.

### Artifacts

| Artifact | Path | Notes |
|---|---|---|
| MCP server entry | `scripts/mcp-semantic/repo-cortex-mcp.mjs` | stdio JSON-RPC MCP server |
| Tool: search | `scripts/mcp-semantic/tools/search-corpus.mjs` | BM25 query + rank |
| Tool: load chunk | `scripts/mcp-semantic/tools/load-chunk.mjs` | Chunk retrieval by ID |
| Tool: load document | `scripts/mcp-semantic/tools/load-document.mjs` | All chunks for a path |
| Tool: freshness | `scripts/mcp-semantic/tools/freshness-check.mjs` | Freshness triple comparison |
| Tool: stats | `scripts/mcp-semantic/tools/index-stats.mjs` | Row counts + timestamp |
| Tool: list families | `scripts/mcp-semantic/tools/list-families.mjs` | Family enumeration |
| Smoke gate | `scripts/agent-customization/gates/cortex-mcp-smoke.mjs` | Self-check validation gate |
| MCP registration | `.vscode/mcp.json` | Add `neataptic-cortex-mcp` server entry |

### MCP server registration (`.vscode/mcp.json` addition)

```json
"neataptic-cortex-mcp": {
  "type": "stdio",
  "command": "node",
  "args": ["scripts/mcp-semantic/repo-cortex-mcp.mjs"],
  "env": {
    "CORTEX_DB_PATH": "${workspaceFolder}/data/semantic-index.sqlite"
  }
}
```

### Self-check smoke gate contract

`cortex-mcp-smoke.mjs` must:
1. Confirm `data/semantic-index.sqlite` exists and is non-empty.
2. Call `index_stats` and assert `total_chunks > 0`.
3. Call `search_corpus` with `query="NEAT activation"` and assert `results.length > 0`.
4. Return `{ pass: true, evidence, fixHint, owner }` JSON.
5. Return `{ pass: false, fixHint: "Run: node scripts/semantic-index/build-index.mjs", ... }` when the index is missing.

## Implementation phases

### Phase 1 — Planning [PLANNED]

#### Step 01 — Author step packets (01-planning) [PLANNED]

```yaml
phase: 1
step: 1
agent: "01-planning"
agent_file: ".github/agents/01-planning.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_MCP_Tools.plans.md"
skills: "tracker-handoff, plan-sync-validation, mcp-local-server-workflow"
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Semantic_Knowledge_MCP_Tools.plans.md
gate: "semantic-foundation-exists"
gate_check: "node scripts/semantic-index/validate-index.mjs --json"
```

**Step objective:** Confirm `Semantic_Knowledge_Foundation` is [DONE] in `plans/completed/` and the index exists.
Read existing MCP server scripts in `scripts/agent-customization/mcp/` to identify reusable
JSON-RPC patterns. Author Step 02 through Step 07 packets with concrete file targets.

### Phase 2 — Research [PLANNED]

#### Step 02 — Recon existing MCP server patterns (02-researching) [PLANNED]

```yaml
phase: 2
step: 2
agent: "02-researching"
agent_file: ".github/agents/02-researching.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_MCP_Tools.plans.md"
skills: "mcp-local-server-workflow"
validation:
  - Read scripts/agent-customization/mcp/ for existing server patterns
  - Read .vscode/mcp.json for current server registrations
```

**Step objective:** Map the existing MCP server entry-point pattern (stdio framing, tool
registration, error handling) so the new server can reuse the same shape. Confirm the
`.vscode/mcp.json` addition is additive and does not break existing servers.

### Phase 3 — Red tests [PLANNED]

#### Step 03 — Red contracts for MCP tools and smoke gate (03-red-testing) [PLANNED]

```yaml
phase: 3
step: 3
agent: "03-red-testing"
agent_file: ".github/agents/03-red-testing.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_MCP_Tools.plans.md"
skills: "red-test-contracts, mcp-local-server-workflow"
validation:
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/mcp-semantic
```

**Step objective:** Write failing tests for:
- `search-corpus.mjs`: given a known query and seeded test DB, returns ranked chunk array.
- `freshness-check.mjs`: detects stale document when mtime has changed.
- `cortex-mcp-smoke.mjs`: returns `{ pass: false }` when DB is absent, `{ pass: true }` when seeded.

### Phase 4 — Implementation [PLANNED]

#### Step 04 — Implement all MCP tools and server entry (04-implementing) [PLANNED]

```yaml
phase: 4
step: 4
agent: "04-implementing"
agent_file: ".github/agents/04-implementing.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_MCP_Tools.plans.md"
skills: "mcp-local-server-workflow, agent-script-tooling"
validation:
  - node scripts/mcp-semantic/repo-cortex-mcp.mjs --help
  - node scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json
```

**Step objective:** Implement all artifacts listed in the Scope section:
1. `repo-cortex-mcp.mjs` — stdio JSON-RPC server with tool/resource registration.
2. All six tool handlers (search, load-chunk, load-document, freshness-check, stats, list-families).
3. `cortex-mcp-smoke.mjs` gate.
4. Add `neataptic-cortex-mcp` entry to `.vscode/mcp.json`.
All handlers must validate inputs, return typed JSON, and handle missing-index gracefully with a
fixHint pointing to `node scripts/semantic-index/build-index.mjs`.

### Phase 5 — Green validation [PLANNED]

#### Step 05 — Validate all MCP tools end-to-end (05-green-testing) [PLANNED]

```yaml
phase: 5
step: 5
agent: "05-green-testing"
agent_file: ".github/agents/05-green-testing.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_MCP_Tools.plans.md"
skills: "green-validation-gates, mcp-local-server-workflow"
validation:
  - node scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/mcp-semantic
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Semantic_Knowledge_MCP_Tools.plans.md
```

**Step objective:** Confirm the MCP server starts cleanly, all tools return correct results
against the real index, the smoke gate passes, all unit tests are green, and the `.vscode/mcp.json`
change does not break existing server registrations.

### Phase 6 — Docs [PLANNED]

#### Step 06 — Document all MCP tools with schema and examples (06-documenting) [PLANNED]

```yaml
phase: 6
step: 6
agent: "06-documenting"
agent_file: ".github/agents/06-documenting.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_MCP_Tools.plans.md"
skills: "educational-docs"
```

**Step objective:** Write `scripts/mcp-semantic/README.md` documenting each tool's schema,
example call, and expected output. Include a short section on how the server relates to the
other three MCP servers and how to rebuild the index when corpus sources change.

### Phase 7 — Logging [PLANNED]

#### Step 07 — Session log and handoff to Browser Snapshot plan (07-logging) [PLANNED]

```yaml
phase: 7
step: 7
agent: "07-logging"
agent_file: ".github/agents/07-logging.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_MCP_Tools.plans.md"
skills: "tracker-handoff, summarizing-session-log"
```

**Step objective:** Record compressed done-state entry and confirm that
`Semantic_Knowledge_Browser_Snapshot.plans.md` is ready to proceed as the next Repo Cortex layer.

## Acceptance criteria and validation gates

| Gate | Command | Expected |
|---|---|---|
| Smoke gate passes | `node scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json` | `{ pass: true }` |
| Server starts | `node scripts/mcp-semantic/repo-cortex-mcp.mjs --help` | Exit 0 |
| Search returns results | Via MCP call `search_corpus("NEAT activation")` | ≥ 3 ranked results |
| Freshness check works | Via MCP call `freshness_check()` | JSON with staleness summary |
| Stats non-empty | Via MCP call `index_stats()` | `total_chunks > 0` |
| Unit tests green | `npx jest --testPathPattern=scripts/mcp-semantic` | All pass |
| MCP registration valid | `.vscode/mcp.json` parses and starts existing servers | No regressions |

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Active plan: plans/Semantic_Knowledge_MCP_Tools.plans.md [PLANNED]

Prerequisite: plans/Semantic_Knowledge_Foundation.plans.md must be [DONE] and
data/semantic-index.sqlite must exist. Verify with:
  node scripts/semantic-index/validate-index.mjs --json

Goal: Implement the neataptic-cortex-mcp MCP server (Repo Cortex Layer 2).

Key artifacts:
  scripts/mcp-semantic/repo-cortex-mcp.mjs        (server entry)
  scripts/mcp-semantic/tools/search-corpus.mjs     (BM25 search tool)
  scripts/mcp-semantic/tools/load-chunk.mjs        (chunk loader)
  scripts/mcp-semantic/tools/freshness-check.mjs   (freshness tool)
  scripts/mcp-semantic/tools/index-stats.mjs       (stats tool)
  scripts/agent-customization/gates/cortex-mcp-smoke.mjs (smoke gate)
  .vscode/mcp.json                                 (add neataptic-cortex-mcp entry)

Start with Step 01 (01-planning): read scripts/agent-customization/mcp/ for existing
server patterns before authoring the remaining step packets.

Smoke gate:
  node scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json

Plan sync check:
  node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Semantic_Knowledge_MCP_Tools.plans.md
```
