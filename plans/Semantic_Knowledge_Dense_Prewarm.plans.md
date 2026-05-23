# Semantic Knowledge Dense Prewarm (Repo Cortex — Layer 6)

**Status:** [PLANNED]

> Direct follow-up to Repo Cortex Layer 5 (Semantic_Knowledge_Embeddings). Operationalizes
> full embedding prewarm so `use_dense: true` can become the MCP server default from day 1
> and dense scores are warm for the whole project corpus on any machine that runs the bootstrap.
>
> This plan does **not** invent dense retrieval — Layer 5 owns that. This plan defines the
> operational contract, readiness states, prewarm bootstrap, graceful degradation path, and
> gate that collectively make default-on dense safe and honest.
>
> **Dependency:** `Semantic_Knowledge_Embeddings.plans.md` must be [DONE] before this plan starts.

## Purpose

Layer 5 ships the ONNX embedding pipeline with `use_dense: false` as the default until MRR@5
evaluation proves consistent hybrid improvement. Once that evidence exists and Layer 5 is [DONE],
this plan flips the default, defines a strict bootstrap-first contract for fresh clones, adds a
readiness probe operators can call directly, and ensures the MCP server degrades gracefully when
embeddings have not yet been built.

**Why a separate plan?**
Flipping `use_dense` to `true` is not just a one-line config change. It requires:

- A verified prewarm script that is idempotent, JSON-capable, and CI-friendly.
- An explicit readiness probe so operators and agents can determine dense state without issuing
  a live query.
- A graceful degradation path in the MCP server so an unprimed machine does not return hard errors.
- Documentation of the bootstrap contract so new contributors know what to run after `git clone`.
- A gate script usable in CI and by numbered orchestrators.

## Fresh-clone honesty constraint

`data/embeddings.sqlite` and `scripts/semantic-index/models/` are **gitignored generated
artifacts**. They do not exist on a fresh `git clone` and cannot be committed.

This plan does **not** claim that `use_dense: true` works magically on a fresh clone. The
strict bootstrap-first contract is:

> After `git clone`, `npm install`, and `npm run build`, an operator must run
> `npm run index:prewarm` before `use_dense: true` is fully honored.
> Until prewarm completes, the MCP server falls back to BM25-only and warns in the response.

That fallback is safe and explicit. It is not a silent failure.

## Readiness states

| State        | Model present | `data/embeddings.sqlite` present | Count valid         | `use_dense` behavior                                       |
| ------------ | ------------- | -------------------------------- | ------------------- | ---------------------------------------------------------- |
| `cold`       | No            | No                               | —                   | MCP falls back to BM25; `dense_degraded: true` in response |
| `model-only` | Yes           | No                               | —                   | MCP falls back to BM25; `dense_degraded: true` in response |
| `warm`       | Yes           | Yes                              | Count = chunk count | MCP uses dense by default; no degradation                  |

The readiness probe (`dense-readiness.mjs`) returns one of these three states plus a human-readable
`reason`. The gate script wraps the probe into the standard `{pass, evidence, fixHint, owner}` contract.

## Default-on guarantee — precise definition

`use_dense: true` is **genuinely honored by default** if and only if all four of the following hold:

1. **Default parameter value**: The MCP `search_corpus` tool definition sets `use_dense` to `true`
   when the caller omits the argument.
2. **Warm-path positive proof**: Every warm-path response includes `dense_state: "warm"` explicitly.
   Absence of `dense_degraded` alone is **insufficient** — a silent BM25 fallback with no disclosure
   would also produce no `dense_degraded` field, making it indistinguishable from a genuine dense
   result to any caller that relies only on field absence.
3. **Cold-path honest degradation**: When embeddings are not warm, the response must include
   `dense_degraded: true`, `dense_state` set to `"cold"` or `"model-only"`, and a non-empty
   `dense_reason` string. A silent BM25-only result without these three fields is a contract
   violation.
4. **Schema provenance**: The tool output schema must declare `dense_state` with a literal union
   type (`"cold" | "model-only" | "warm"`), not a generic `string`. This allows callers and tests
   to assert the exact state without fragile string-contains matching.

> **Implementation bar**: A future implementation pass that omits `dense_state` from the warm
> response, uses a generic `string` type for `dense_state`, or fails to emit `dense_reason` on
> degraded paths does **not** satisfy this guarantee, even if all other tests pass.

## Layer 4 dependency note

This plan depends only on Layer 5 ([DONE]) and the existing Layer 2 MCP server
(`neataptic-cortex-mcp`, also [DONE]).

If Repo Cortex Layer 4 ([completed/Repo_Cortex_MCP_Reliability.plans.md](completed/Repo_Cortex_MCP_Reliability.plans.md)) is also [DONE] by the time
this plan executes, its lifecycle gate and MCP reliability hardening will help the cortex server
start warm; but Layer 4 is a **soft dependency** here — this plan can execute whether Layer 4
is [DONE] or still [PLANNED], since the graceful degradation path covers the cold-start case.

## Non-goals

- Re-implementing ONNX embeddings or the hybrid rank formula (owned by Layer 5).
- Changing the corpus scanner, BM25 index, or chunking strategy (owned by Layer 1).
- Browser-side dense search (out of scope per Layer 5 non-goals).
- Changes to `src/` library code.
- Training or fine-tuning the embedding model.
- Removing BM25-only mode — BM25 must always remain available as a fallback.

## Scope

### New scripts

| Script               | Path                                                         | Notes                                                                                            |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Prewarm bootstrap    | `scripts/semantic-index/prewarm-dense.mjs`                   | Runs download-model → embed-index → validate in one idempotent pass; `--json`, `--dry-run` flags |
| Readiness probe      | `scripts/semantic-index/dense-readiness.mjs`                 | Returns `{ready, state, reason, chunk_count, embedding_count}`; `--json` flag                    |
| Dense readiness gate | `scripts/agent-customization/gates/dense-readiness.gate.mjs` | Standard `{pass, evidence, fixHint, owner}` gate contract; exit 0/1                              |

### MCP server changes

| Change                       | File                                                 | Notes                                                                                                                                                                                                                                                                                                                                                                            |
| ---------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Default `use_dense` → `true` | `scripts/mcp-semantic/*.mjs` (search_corpus handler) | Flip default; add inline comment referencing bootstrap contract                                                                                                                                                                                                                                                                                                                  |
| Readiness guard              | Same handler                                         | Check `dense-readiness.mjs` probe; if state ≠ `warm`, degrade gracefully: serve BM25, add `dense_degraded: true`, `dense_state: "<state>"`, and `dense_reason: "<non-empty explanation>"` to response; all three fields are required on any degraded response                                                                                                                    |
| Response schema extension    | Tool definition                                      | Add `dense_degraded?: boolean`, `dense_state?: "cold" \| "model-only" \| "warm"` (literal union, not generic `string`), and `dense_reason?: string` to `search_corpus` output schema; `dense_state` must be emitted on **every** response where `use_dense` is true, including the warm path, so callers can assert positive provenance rather than inferring from field absence |

### `package.json` scripts

| Script                  | Command                                                  |
| ----------------------- | -------------------------------------------------------- |
| `index:prewarm`         | `node scripts/semantic-index/prewarm-dense.mjs`          |
| `index:dense-readiness` | `node scripts/semantic-index/dense-readiness.mjs --json` |

### Documentation updates

| Target                             | Change                                                                                                                |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `scripts/semantic-index/README.md` | Add "Bootstrap contract" section: what to run after clone, what the readiness states mean, how to force a full rewarm |
| `CLAUDE.md`                        | Add bootstrap note under "Commands": `npm run index:prewarm` as the post-clone dense bootstrap                        |
| MCP tool JSDoc / schema comments   | Note that `use_dense` defaults to `true` and falls back gracefully if embeddings are cold                             |

## Prewarm contract specification

```text
Strict bootstrap-first contract:
  1. Run: npm run index:download-model       (idempotent; skips if model already present)
  2. Run: npm run index:embed                (rebuilds if stale, skips unchanged chunks)
  3. Run: npm run index:validate-embeddings  (asserts count match)

OR equivalently:
  npm run index:prewarm                      (runs all three steps in order)

After prewarm:
  node scripts/semantic-index/dense-readiness.mjs --json
  → { ready: true, state: "warm", chunk_count: N, embedding_count: N, reason: "All N chunks have embeddings." }

MCP default behavior after prewarm:
  search_corpus({ query: "NEAT activation" })
  → dense re-ranking applied by default (use_dense implicitly true)
  → { results: [...], dense_state: "warm" }   ← dense_state REQUIRED; this is the authoritative proof
  → dense_degraded field ABSENT (but dense_state: "warm" is the signal, not this absence)

MCP fallback behavior without prewarm:
  search_corpus({ query: "NEAT activation" })
  → BM25 results only
  → { results: [...], dense_degraded: true, dense_state: "cold", dense_reason: "Embeddings not built." }
```

## Rewarm policy

The prewarm script is idempotent. Running it again after the corpus changes (new READMEs,
new plans, new examples) will only re-embed changed or new chunks. The rewarm cycle is:

```sh
npm run build-index      # rebuild corpus (existing Layer 1 script)
npm run index:prewarm    # re-embed changed chunks + validate
```

Operators are not required to rewarm on every commit — BM25 results remain available instantly.
Dense results reflect the last prewarm state, which is documented in `dense-readiness.mjs` output.

## Implementation phases

### Phase 1 — Planning [PLANNED]

#### Step 01 — Confirm prerequisites and author step packets (01-planning) [PLANNED]

```yaml
phase: 1
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Semantic_Knowledge_Dense_Prewarm.plans.md'
skills: 'tracker-handoff, plan-sync-validation'
gate: 'semantic-embeddings-done'
gate_check: |
  Verify plans/Semantic_Knowledge_Embeddings.plans.md status is [DONE].
  Verify data/embeddings.sqlite exists and validate-embeddings.mjs --json returns { pass: true }.
  Verify scripts/mcp-semantic/ contains the search_corpus handler with use_dense parameter.
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Semantic_Knowledge_Dense_Prewarm.plans.md
```

**Step objective:** Confirm Layer 5 artifacts exist:
`scripts/semantic-index/download-model.mjs`, `scripts/semantic-index/embed-index.mjs`,
`scripts/semantic-index/hybrid-rank.mjs`, `data/embeddings.sqlite` (or equivalent path),
and the `search_corpus` MCP tool with `use_dense` parameter.
Map the current default value of `use_dense` in the MCP handler.
Author remaining step packets. If Layer 5 is not yet [DONE], record a blocker and stop.

### Phase 2 — Research [PLANNED]

#### Step 02 — Map MCP handler and existing prewarm patterns (02-researching) [PLANNED]

```yaml
phase: 2
step: 2
agent: '02-researching'
agent_file: '.github/agents/02-researching.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Semantic_Knowledge_Dense_Prewarm.plans.md'
skills: 'plan-alignment, agent-script-tooling'
validation:
  - Read scripts/mcp-semantic/ to locate the search_corpus tool handler and current use_dense default
  - Read scripts/semantic-index/ to confirm download-model and embed-index script interfaces
  - Confirm validate-embeddings.mjs --json output shape (needed by readiness probe)
  - Check whether dense-readiness.gate.mjs would conflict with cortex-index.gate.mjs from Layer 4
```

**Step objective:** Produce a recon brief covering:
(1) exact file and line where `use_dense` default is set in the MCP handler,
(2) the output shape of `validate-embeddings.mjs --json` so the readiness probe can parse it,
(3) the gate output shape expected by `scripts/agent-customization/gates/` so the new gate
script is consistent with existing gates,
(4) any Layer 4 gate that already checks embeddings presence (avoid duplication).

### Phase 3 — Red tests [PLANNED]

#### Step 03 — Red contracts for prewarm script and readiness probe (03-red-testing) [PLANNED]

```yaml
phase: 3
step: 3
agent: '03-red-testing'
agent_file: '.github/agents/03-red-testing.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Semantic_Knowledge_Dense_Prewarm.plans.md'
skills: 'red-test-contracts, agent-script-tooling'
validation:
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/semantic-index/prewarm
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/semantic-index/dense-readiness
```

**Step objective:** Write failing tests for:

- `prewarm-dense.mjs`: given a mocked `download-model` and `embed-index`, orchestrates both in order and exits 0.
- `prewarm-dense.mjs --dry-run`: logs planned steps, exits 0 without calling scripts.
- `dense-readiness.mjs`: returns `{ready: false, state: "cold"}` when model dir and embeddings DB are absent.
- `dense-readiness.mjs`: returns `{ready: false, state: "model-only"}` when model present but DB absent.
- `dense-readiness.mjs`: returns `{ready: true, state: "warm"}` when model + DB + count match.
- MCP handler: when readiness state is `cold`, response includes `dense_degraded: true`, `dense_state: "cold"`, and a non-empty trimmed `dense_reason` string (`.trim().length >= 1`) — all three fields individually asserted; BM25 results only.
- MCP handler: when readiness state is `warm`, response must include `dense_state: "warm"` AND must NOT include a `dense_degraded` field — both assertions are required. The warm test must fail if `dense_state` is absent even when `dense_degraded` is also absent (positive proof, not negative inference).

Use filesystem mocks or temp directories — do not require a real ONNX model or a real corpus DB in tests.

### Phase 4 — Implementation [PLANNED]

#### Step 04 — Implement prewarm script, readiness probe, gate, and MCP default (04-implementing) [PLANNED]

```yaml
phase: 4
step: 4
agent: '04-implementing'
agent_file: '.github/agents/04-implementing.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Semantic_Knowledge_Dense_Prewarm.plans.md'
skills: 'agent-script-tooling, plan-alignment'
validation:
  - node scripts/semantic-index/dense-readiness.mjs --json
  - node scripts/semantic-index/prewarm-dense.mjs --dry-run
  - node scripts/agent-customization/gates/dense-readiness.gate.mjs --json
  - DENSE_FORCE_STATE=cold node scripts/semantic-index/dense-readiness.mjs --json  →  { ready: false, state: "cold" }
  - DENSE_FORCE_STATE=model-only node scripts/semantic-index/dense-readiness.mjs --json  →  { ready: false, state: "model-only" }
```

**Step objective:** Implement all new artifacts:

1. `scripts/semantic-index/prewarm-dense.mjs` —
   Idempotent bootstrap: (a) invoke download-model.mjs if model missing, (b) invoke embed-index.mjs
   (which already skips unchanged chunks), (c) invoke validate-embeddings.mjs --json, (d) exit 0 on
   success. Supports `--dry-run` (log steps, no execution) and `--json` (machine-readable output).

2. `scripts/semantic-index/dense-readiness.mjs` —
   Probe script: check model directory presence, check embeddings DB existence, compare embedding
   count vs chunk count (reuse validate-embeddings output), return
   `{ready, state, reason, chunk_count, embedding_count}` as JSON.

3. `scripts/agent-customization/gates/dense-readiness.gate.mjs` —
   Gate wrapper: call dense-readiness.mjs, map to `{pass, evidence, fixHint, owner}` contract.
   `fixHint` must say: "Run `npm run index:prewarm` to build the embedding index."
   `owner` is `"01-planning"`.

4. MCP server `search_corpus` handler —
   (a) Change default `use_dense` from `false` to `true`.
   (b) On each call where `use_dense` is true (explicitly or by default), invoke the readiness
   probe (or a cached check valid for the server process lifetime). If state ≠ `warm`, degrade:
   serve BM25 results, append `dense_degraded: true` and `dense_state: <state>` to response.
   (c) Extend the tool output schema with `dense_degraded?: boolean`,
   `dense_state?: "cold" | "model-only" | "warm"` (literal union type, **not** generic `string`),
   and `dense_reason?: string`. `dense_state` must be emitted on every response where `use_dense`
   is true — including the warm path. Warm path: `{ results: [...], dense_state: "warm" }`.
   Degraded paths: `{ results: [...], dense_degraded: true, dense_state: "<state>", dense_reason: "<non-empty trimmed explanation>" }`. `dense_reason` must satisfy `.trim().length >= 1` — an empty string or whitespace-only string is a contract violation.
   (d) Support the `DENSE_FORCE_STATE` environment variable for deterministic testing. **This is mandatory, not optional.** Allowed values: `cold`, `model-only`. When set, the readiness probe must return the forced state regardless of actual filesystem state. This override is required for validation in Step 05 and for CI-safe deterministic cold-state assertions without renaming the database. Document allowed values in `dense-readiness.mjs` JSDoc.

5. `package.json` scripts —
   Add `"index:prewarm"` and `"index:dense-readiness"` entries.

**Constraint:** The prewarm script must not make network calls unless model is absent.
If model is already present at `scripts/semantic-index/models/`, skip download silently.

### Phase 5 — Green validation [PLANNED]

#### Step 05 — Run prewarm end-to-end and confirm readiness probe passes (05-green-testing) [PLANNED]

```yaml
phase: 5
step: 5
agent: '05-green-testing'
agent_file: '.github/agents/05-green-testing.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Semantic_Knowledge_Dense_Prewarm.plans.md'
skills: 'green-validation-gates'
validation:
  - node scripts/semantic-index/prewarm-dense.mjs --json
  - node scripts/semantic-index/dense-readiness.mjs --json
  - node scripts/agent-customization/gates/dense-readiness.gate.mjs --json
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/semantic-index/prewarm
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/semantic-index/dense-readiness
  - Verify MCP search_corpus with no use_dense arg returns response with `dense_state: "warm"` present AND `dense_degraded` absent — both assertions required; `dense_state: "warm"` is the authoritative proof
  - Verify MCP cold-state response (DENSE_FORCE_STATE=cold env var — mandatory override, not optional) contains `dense_degraded: true`, `dense_state: "cold"`, and a non-empty trimmed `dense_reason` string (`.trim().length >= 1`) — all three fields individually asserted
```

**Step objective:** Run `npm run index:prewarm` against the real corpus. Confirm:

- Prewarm exits 0 and `--json` output shows all steps passed.
- `dense-readiness.mjs --json` returns `{ready: true, state: "warm"}`.
- Gate script returns `{pass: true}`.
- All unit tests green.
- Live MCP call without `use_dense` flag returns `{ results: [...], dense_state: "warm" }` with `dense_degraded` absent — both assertions required; `dense_state: "warm"` presence is the authoritative proof.
- Simulated cold-state MCP call (set env var `DENSE_FORCE_STATE=cold` — this is the mandatory deterministic override; do not rely on renaming the DB) returns `{ results: [...], dense_degraded: true, dense_state: "cold", dense_reason: "<non-empty trimmed>" }` — all four fields must be individually asserted; `dense_reason.trim().length >= 1` is a hard requirement.

### Phase 6 — Docs [PLANNED]

#### Step 06 — Document bootstrap contract and readiness probe (06-documenting) [PLANNED]

```yaml
phase: 6
step: 6
agent: '06-documenting'
agent_file: '.github/agents/06-documenting.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Semantic_Knowledge_Dense_Prewarm.plans.md'
skills: 'educational-docs'
```

**Step objective:**

1. Update `scripts/semantic-index/README.md` — add a "Bootstrap contract" section covering:
   - Fresh-clone instructions: run `npm run index:prewarm` after install.
   - Readiness states (`cold`, `model-only`, `warm`) with example probe output.
   - Rewarm cycle: when and how to rewarm after corpus changes.
   - Graceful degradation behavior when cold.
   - Reference to `dense-readiness.gate.mjs` for CI use.

2. Update `CLAUDE.md` — add `npm run index:prewarm` under Commands as the post-clone dense
   bootstrap step, with a one-line note about when to rerun it.

3. Update MCP tool JSDoc and schema comments — note that `use_dense` defaults to `true` and
   the server degrades gracefully when embeddings are cold; point operators to
   `npm run index:prewarm`.

### Phase 7 — Logging [PLANNED]

#### Step 07 — Session log and mark [DONE] (07-logging) [PLANNED]

```yaml
phase: 7
step: 7
agent: '07-logging'
agent_file: '.github/agents/07-logging.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/Semantic_Knowledge_Dense_Prewarm.plans.md'
skills: 'tracker-handoff, summarizing-session-log'
```

**Step objective:** Record evidence: prewarm run output, readiness probe JSON, gate result, MCP
degradation test result, and all test counts. Compress this tracker. Create a matching
`Semantic_Knowledge_Dense_Prewarm.logs.md` audit record. Move both to `plans/completed/`.
Update `plans/README.md` and `plans/Roadmap.md` to mark this plan [DONE] and archived.

## Acceptance criteria and validation gates

| Gate                  | Command                                                                            | Expected                                                                                                                                                                                              |
| --------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prewarm bootstrap     | `npm run index:prewarm`                                                            | Exit 0; model present; embeddings count = chunk count                                                                                                                                                 |
| Readiness probe warm  | `node scripts/semantic-index/dense-readiness.mjs --json`                           | `{ready: true, state: "warm"}`                                                                                                                                                                        |
| Readiness gate        | `node scripts/agent-customization/gates/dense-readiness.gate.mjs --json`           | `{pass: true}`                                                                                                                                                                                        |
| Unit tests            | `npx jest --testPathPattern=scripts/semantic-index/prewarm\|dense-readiness`       | All pass                                                                                                                                                                                              |
| MCP default-on        | `search_corpus({query: "NEAT"})` (no use_dense arg)                                | `{ results: [...], dense_state: "warm" }`; `dense_degraded` absent — `dense_state: "warm"` is the authoritative proof, not field absence                                                              |
| MCP cold degradation  | Same call with `DENSE_FORCE_STATE=cold` env var (mandatory deterministic override) | `{ results: [...], dense_degraded: true, dense_state: "cold", dense_reason: "<non-empty trimmed>" }` — all four fields required and individually asserted; `dense_reason.trim().length >= 1` enforced |
| MCP schema provenance | Inspect `search_corpus` tool output schema definition                              | `dense_state` typed as literal union `"cold" \| "model-only" \| "warm"` (not generic `string`); `dense_reason` present as `string`; `dense_state` always emitted when `use_dense` is true             |
| Dry-run safe          | `node scripts/semantic-index/prewarm-dense.mjs --dry-run`                          | Exit 0; no network calls; logs planned steps                                                                                                                                                          |

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Active plan: plans/Semantic_Knowledge_Dense_Prewarm.plans.md [PLANNED]

Hard prerequisite (must be [DONE] before this plan starts):
  - plans/Semantic_Knowledge_Embeddings.plans.md [DONE]
  Verify: node scripts/semantic-index/validate-embeddings.mjs --json  →  { pass: true }

Soft prerequisite (helps but not a blocker):
  - plans/completed/Repo_Cortex_MCP_Reliability.plans.md (Layer 4 cortex reliability hardening)

Goal: operationalize full embedding prewarm so use_dense: true is the MCP default and dense
scores are warm for the whole corpus from day 1 on any machine that runs the bootstrap.

Key design decisions:
  - Strict bootstrap-first contract: fresh clone requires npm run index:prewarm before dense is safe.
  - Three readiness states: cold / model-only / warm; MCP degrades gracefully in cold and model-only.
  - No changes to src/ library code.
  - Rewarm cycle: npm run build-index && npm run index:prewarm after corpus changes.

Key artifacts to create:
  scripts/semantic-index/prewarm-dense.mjs              (idempotent bootstrap: download + embed + validate)
  scripts/semantic-index/dense-readiness.mjs            (readiness probe: {ready, state, reason, counts})
  scripts/agent-customization/gates/dense-readiness.gate.mjs   (standard {pass, evidence, fixHint, owner})
  MCP search_corpus handler: default use_dense true + readiness guard + graceful degradation
  package.json scripts: index:prewarm, index:dense-readiness
  CLAUDE.md and scripts/semantic-index/README.md: bootstrap contract docs

Start with Step 01 (01-planning): confirm Semantic_Knowledge_Embeddings.plans.md is [DONE],
confirm data/embeddings.sqlite exists, map current use_dense default in MCP handler.

Plan sync check:
  node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Semantic_Knowledge_Dense_Prewarm.plans.md
```
