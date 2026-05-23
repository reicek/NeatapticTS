# Semantic Knowledge Browser Snapshot (Repo Cortex — Layer 3)

**Status:** [PLANNED]

> Generates a browser-consumable JSON snapshot of the Repo Cortex corpus index and provides
> a lightweight IndexedDB-backed loader for all NeatapticTS demos.
> Depends on [Semantic_Knowledge_Foundation.plans.md](Semantic_Knowledge_Foundation.plans.md).

## Purpose

Expose the Repo Cortex semantic corpus as a prebuilt static JSON snapshot so browser-based
demos can load rich documentation context without requiring a live SQLite connection or a Node
server. The snapshot is a **generated artifact** produced by the docs pipeline, not a
hand-edited file.

The primary consumers are all demos under `examples/` that want to show contextual help,
concept explanations, or architecture documentation alongside their browser UI. The approach
mirrors the generated example publication pattern already established in this repo: source
input → docs pipeline → generated output under `docs/`.

## Non-goals

- Real-time BM25 search in the browser (the snapshot is a pre-ranked, pre-chunked JSON file,
  not a live SQLite instance).
- Replacing the server-side SQLite index (see `Semantic_Knowledge_Foundation.plans.md`).
- Replacing MCP tools (see `Semantic_Knowledge_MCP_Tools.plans.md`).
- Dense embedding search in the browser (see `Semantic_Knowledge_Embeddings.plans.md`).
- Changes to `src/` library code.
- NeatChat-internal conversational memory (separate plan: `NeatChat_Local_Retrieval_Memory.plans.md`).
- Editing `docs/assets/semantic-snapshot.json` directly — it is a generated artifact.

## Dependencies

- [Semantic_Knowledge_Foundation.plans.md](Semantic_Knowledge_Foundation.plans.md) [PLANNED] —
  `data/semantic-index.sqlite` must exist before the snapshot generator can run.
- `npm run docs` pipeline must call the snapshot generator as a step (additive, non-breaking).
- `scripts/copy-examples.ts` or a new docs pipeline hook for snapshot publication.

## Scope

### Snapshot format

The browser snapshot is a single JSON file containing a compact representation of the corpus:

```jsonc
{
  "generated_at": "<ISO timestamp>",
  "schema_version": "1",
  "families": ["readme", "skill", "agent", "plan", "demo"],
  "documents": [
    {
      "doc_id": 1,
      "file_path": "src/neat/README.md",
      "family": "readme",
      "chunks": [
        {
          "chunk_id": 1,
          "heading_path": "## Overview",
          "body_text": "...",
          "char_start": 0,
          "char_end": 2048
        }
      ]
    }
  ]
}
```

The snapshot is intentionally read-optimized (not a full FTS5 index). Browser-side search
uses a simple term-frequency scan over the pre-loaded snapshot rather than SQLite FTS5.

### IndexedDB cache

The demo-side loader caches the fetched snapshot in IndexedDB keyed by `generated_at`
timestamp. On reload, the loader compares the cached timestamp against the served snapshot's
`generated_at` before re-fetching. This avoids repeated full-file downloads across sessions.

### Artifacts

| Artifact | Path | Notes |
|---|---|---|
| Snapshot generator | `scripts/semantic-index/build-browser-snapshot.mjs` | Reads SQLite → emits JSON |
| Browser loader module | `examples/shared/semantic/semantic-snapshot-loader.ts` | Fetch + IndexedDB cache |
| Browser search utility | `examples/shared/semantic/semantic-snapshot-search.ts` | Term-frequency search over snapshot |
| Browser type definitions | `examples/shared/semantic/semantic-snapshot.types.ts` | SnapshotDocument, SnapshotChunk types |
| Generated snapshot | `docs/assets/semantic-snapshot.json` | **Generated artifact — do not edit directly** |
| Docs pipeline hook | `scripts/copy-examples.ts` or `scripts/generate-docs.ts` | Runs snapshot generator before publish |
| `package.json` script | `package.json` | `index:build-snapshot` (also runs during `npm run docs`) |

### Generated output handling

- `docs/assets/semantic-snapshot.json` is a **generated artifact** built by the docs pipeline.
- Edit source: rebuild `data/semantic-index.sqlite` via `node scripts/semantic-index/build-index.mjs`,
  then run `node scripts/semantic-index/build-browser-snapshot.mjs`.
- `npm run docs` automatically runs both steps so the published snapshot stays in sync.
- Never hand-edit `docs/assets/semantic-snapshot.json` — treat it as read-only.

### Demo integration guidance

Any demo under `examples/` that wants corpus context can import the shared loader:

```ts
import { loadSemanticSnapshot, searchSnapshot } from '../shared/semantic/semantic-snapshot-loader';

const snapshot = await loadSemanticSnapshot('/assets/semantic-snapshot.json');
const results = searchSnapshot(snapshot, 'NEAT activation', { limit: 5 });
```

The loader is opt-in. Demos that do not call it are not affected.

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
source_of_truth: "plans/Semantic_Knowledge_Browser_Snapshot.plans.md"
skills: "tracker-handoff, plan-sync-validation, browser-build"
gate: "semantic-foundation-exists"
gate_check: "node scripts/semantic-index/validate-index.mjs --json"
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Semantic_Knowledge_Browser_Snapshot.plans.md
```

**Step objective:** Confirm Foundation is [DONE]. Read `scripts/copy-examples.ts` and the
existing docs pipeline to identify the correct hook point for the snapshot generator. Confirm
`docs/assets/` is the right output directory. Author remaining step packets.

### Phase 2 — Research [PLANNED]

#### Step 02 — Recon docs pipeline and examples/shared surface (02-researching) [PLANNED]

```yaml
phase: 2
step: 2
agent: "02-researching"
agent_file: ".github/agents/02-researching.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Browser_Snapshot.plans.md"
skills: "browser-build, plan-alignment"
validation:
  - Read scripts/copy-examples.ts to identify pipeline hook point
  - Read examples/ root for existing shared/ pattern
  - Read docs/assets/ for existing snapshot artifacts
```

**Step objective:** Map the docs pipeline hook point, confirm `examples/shared/` exists or
should be created, identify any existing TypeScript compilation needed for shared demo
utilities, and hand off a compact brief to Step 04.

### Phase 3 — Red tests [PLANNED]

#### Step 03 — Red contracts for snapshot generator and browser loader (03-red-testing) [PLANNED]

```yaml
phase: 3
step: 3
agent: "03-red-testing"
agent_file: ".github/agents/03-red-testing.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Browser_Snapshot.plans.md"
skills: "red-test-contracts"
validation:
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/shared/semantic
```

**Step objective:** Write failing tests for:
- `build-browser-snapshot.mjs`: given a seeded test DB, emits valid JSON with expected fields.
- `semantic-snapshot-loader.ts`: given a mock fetch response, caches in IndexedDB and returns snapshot.
- `semantic-snapshot-search.ts`: given snapshot fixture, returns ranked results for a known query.

### Phase 4 — Implementation [PLANNED]

#### Step 04 — Build snapshot generator and browser loader (04-implementing) [PLANNED]

```yaml
phase: 4
step: 4
agent: "04-implementing"
agent_file: ".github/agents/04-implementing.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Browser_Snapshot.plans.md"
skills: "browser-build, agent-script-tooling"
validation:
  - node scripts/semantic-index/build-browser-snapshot.mjs --dry-run
  - node scripts/semantic-index/build-browser-snapshot.mjs
  - Test that docs/assets/semantic-snapshot.json exists and is valid JSON
```

**Step objective:** Implement all artifacts:
1. `build-browser-snapshot.mjs` — reads SQLite → emits `docs/assets/semantic-snapshot.json`.
2. `examples/shared/semantic/semantic-snapshot.types.ts` — SnapshotDocument, SnapshotChunk, SnapshotIndex types.
3. `examples/shared/semantic/semantic-snapshot-loader.ts` — fetch + IndexedDB cache + freshness check.
4. `examples/shared/semantic/semantic-snapshot-search.ts` — term-frequency search + heading-weighted ranking.
5. Docs pipeline hook: add `build-browser-snapshot` call to `scripts/copy-examples.ts` or `package.json` `docs` script.
6. Add `index:build-snapshot` npm script.

### Phase 5 — Green validation [PLANNED]

#### Step 05 — Validate snapshot generation and browser loader (05-green-testing) [PLANNED]

```yaml
phase: 5
step: 5
agent: "05-green-testing"
agent_file: ".github/agents/05-green-testing.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Browser_Snapshot.plans.md"
skills: "green-validation-gates, browser-build"
validation:
  - node scripts/semantic-index/build-browser-snapshot.mjs
  - node -e "JSON.parse(require('fs').readFileSync('docs/assets/semantic-snapshot.json','utf-8'))"
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/shared/semantic
  - npm run docs
  - Confirm docs/assets/semantic-snapshot.json is updated after npm run docs
```

**Step objective:** Confirm the snapshot generator runs cleanly, the output is valid JSON,
`npm run docs` updates the snapshot, and all unit tests are green.

### Phase 6 — Docs [PLANNED]

#### Step 06 — Document loader API and generated output contract (06-documenting) [PLANNED]

```yaml
phase: 6
step: 6
agent: "06-documenting"
agent_file: ".github/agents/06-documenting.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Browser_Snapshot.plans.md"
skills: "educational-docs"
```

**Step objective:** Add JSDoc to `semantic-snapshot-loader.ts` and `semantic-snapshot-search.ts`.
Write a short `examples/shared/semantic/README.md` that documents the snapshot format,
the generated output contract (do not edit `docs/assets/semantic-snapshot.json` directly),
and the IndexedDB cache behavior.

### Phase 7 — Logging [PLANNED]

#### Step 07 — Session log and handoff (07-logging) [PLANNED]

```yaml
phase: 7
step: 7
agent: "07-logging"
agent_file: ".github/agents/07-logging.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Browser_Snapshot.plans.md"
skills: "tracker-handoff, summarizing-session-log"
```

**Step objective:** Record compressed done-state entry. Confirm that
`Semantic_Knowledge_Embeddings.plans.md` is ready as the final advanced Repo Cortex layer.

## Acceptance criteria and validation gates

| Gate | Command | Expected |
|---|---|---|
| Snapshot builds | `node scripts/semantic-index/build-browser-snapshot.mjs` | Exit 0; JSON file emitted |
| Snapshot is valid JSON | `node -e "JSON.parse(...)"` on the output | Parses without error |
| Schema version present | `jq .schema_version docs/assets/semantic-snapshot.json` | `"1"` |
| Docs pipeline runs | `npm run docs` | Updates `docs/assets/semantic-snapshot.json` |
| Loader unit tests | `npx jest --testPathPattern=examples/shared/semantic` | All pass |
| IndexedDB cache hit | Manual browser test or unit mock | Second load uses cached snapshot |
| Search returns results | `searchSnapshot(snapshot, "NEAT")` in test | ≥ 1 result |

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Active plan: plans/Semantic_Knowledge_Browser_Snapshot.plans.md [PLANNED]

Prerequisites:
  plans/Semantic_Knowledge_Foundation.plans.md must be [DONE].
  data/semantic-index.sqlite must exist. Verify:
    node scripts/semantic-index/validate-index.mjs --json

Goal: Generate a browser-consumable JSON snapshot of the Repo Cortex corpus (Layer 3).

Key artifacts:
  scripts/semantic-index/build-browser-snapshot.mjs     (snapshot generator)
  docs/assets/semantic-snapshot.json                    (generated output — do not edit directly)
  examples/shared/semantic/semantic-snapshot-loader.ts  (fetch + IndexedDB cache)
  examples/shared/semantic/semantic-snapshot-search.ts  (term-frequency search)
  examples/shared/semantic/semantic-snapshot.types.ts   (type definitions)

The generated output is published by npm run docs — treat docs/assets/semantic-snapshot.json
as a generated artifact, just like docs/examples/**/index.html.

Start with Step 01 (01-planning): read scripts/copy-examples.ts to find the correct
docs-pipeline hook point before authoring remaining step packets.

Plan sync check:
  node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Semantic_Knowledge_Browser_Snapshot.plans.md
```
