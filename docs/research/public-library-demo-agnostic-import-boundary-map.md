# Demo-agnostic refactor — import boundary map for the racing-named GPU surface

## Question

What is the current import boundary for `src/architecture/network/gpu/network.gpu.racing.ts` and its exported symbols, and is the WebGPU dependency quiet enough that the demo-agnostic rename can proceed safely?

## Evidence

### Methodology

1. Ran Cortex MCP `search_corpus` for `network.gpu.racing` and the exported symbols; results only surfaced the module itself and its generated README, confirming no dense-link consumer surface.
2. Ran `git status --short` on the racing module and its co-located test to detect in-flight edits.
3. Ran a fixed-string `git grep` across `src/` and `examples/` for the module path and every exported racing-named symbol.
4. Cross-referenced the WebGPU plan (`plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md`) to confirm its active slice is paused.

### WebGPU dependency state

- The WebGPU plan is **paused by user directive** so the demo-leak refactor can take priority.
- Its active slice `02-05b-buffer-parallel` is functionally green but intentionally stopped; the remaining work is deferred.
- `git status` shows the racing files are modified, but the diff is the paused slice's implementation, not new active work:

```text
M src/architecture/network/gpu/network.gpu.racing.test.ts
 M src/architecture/network/gpu/network.gpu.racing.ts
```

- Diff summary (`git diff --stat`):

```text
 src/architecture/network/gpu/network.gpu.racing.test.ts | 13 +++++++++-
 src/architecture/network/gpu/network.gpu.racing.ts      | 28 +++++++++++++++-------
 2 files changed, 31 insertions(+), 10 deletions(-)
```

- The implementation changes add a working `evaluateConcurrentRacingAgents` dispatch and an empty-request test; they do **not** introduce new external consumers of the racing-named path.

### Import / reference footprint

Query:

```bash
git grep -n "network.gpu.racing\|RacingBatchOptions\|evaluateRacingGeneration\|RacingAgentRequest\|evaluateConcurrentRacingAgents" -- src/ examples/
```

| File | Line | Reference | Type |
|---|---|---|---|
| `src/architecture/network/gpu/README.md` | 1066 | `evaluateRacingGeneration` | Generated README (will be regenerated) |
| `src/architecture/network/gpu/README.md` | 1222 | `network.gpu.racing.ts` heading | Generated README (will be regenerated) |
| `src/architecture/network/gpu/README.md` | 1246 | `evaluateRacingGeneration` | Generated README (will be regenerated) |
| `src/architecture/network/gpu/README.md` | 1253 | `RacingBatchOptions` | Generated README (will be regenerated) |
| `src/architecture/network/gpu/README.md` | 1295 | `network.gpu.racing.ts` section | Generated README (will be regenerated) |
| `src/architecture/network/gpu/docs.order.json` | 11 | `network.gpu.racing.ts` | Docs metadata — must be updated |
| `src/architecture/network/gpu/network.gpu.batched.ts` | 12 | `evaluateRacingGeneration` (JSDoc only, no import) | Source JSDoc — must be updated |
| `src/architecture/network/gpu/network.gpu.racing.test.ts` | 3–5 | imports the four exported symbols | Co-located test — will be renamed |
| `src/architecture/network/gpu/network.gpu.racing.test.ts` | 10–200 | repeated uses of `evaluateRacingGeneration` / `evaluateConcurrentRacingAgents` | Co-located test — will be renamed |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 10 | `RacingBatchOptions` export | Source module — will be renamed |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 24 | `RacingBatchOptions` JSDoc mention | Source module — will be renamed |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 141 | `evaluateRacingGeneration` export | Source module — will be renamed |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 145 | `RacingBatchOptions` parameter | Source module — will be renamed |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 160 | `RacingAgentRequest` export | Source module — will be renamed |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 180 | `evaluateConcurrentRacingAgents` export | Source module — will be renamed |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 182–183 | `RacingAgentRequest` / `RacingBatchOptions` parameters | Source module — will be renamed |

### Boundary summary

- **External `src/` importers of the module:** `NONE`
- **External `src/` callers of `RacingBatchOptions`:** `NONE`
- **External `src/` callers of `evaluateRacingGeneration`:** `NONE` (only a JSDoc mention in `network.gpu.batched.ts`)
- **External `src/` callers of `RacingAgentRequest`:** `NONE`
- **External `src/` callers of `evaluateConcurrentRacingAgents`:** `NONE`
- **`examples/` consumers:** `NONE`

The racing-named GPU surface is a **leaf node** in the `src/` import graph. The only references are the module itself, its co-located test, a single JSDoc mention in `network.gpu.batched.ts`, generated READMEs, and `docs.order.json`.

## Decision

The dependency verdict is **CLEAR**: the WebGPU plan is paused, and no `src/` file outside the module imports from `network.gpu.racing.ts` or calls the racing-named exported symbols. The rename can proceed without colliding with in-flight library work.

The rename will not break any external `src/` consumer because there are none. Required follow-ups are:

1. Rename the module and its test file.
2. Update the single JSDoc reference in `src/architecture/network/gpu/network.gpu.batched.ts`.
3. Update `src/architecture/network/gpu/docs.order.json`.
4. Regenerate `src/**/README.md` and `docs/` outputs with `npm run docs` after the source JSDoc is sanitized.

## Risks

| Risk | Mitigation |
|---|---|
| Uncommitted WebGPU slice changes could conflict with the rename if `git mv` is not used. | Rename via `git mv` so the paused slice diff tracks across the new path; then run the co-rename of symbols and tests in the same commit. |
| The single JSDoc mention in `network.gpu.batched.ts` is easy to miss. | Include it in slice 03-02 acceptance criteria (`AC-03-02-002`). |
| Generated READMEs still expose the old names until `npm run docs` runs. | Schedule docs regeneration in slice 03-03 or as a final 06-documenting step. |
| `examples/` or `docs/browser-tests/` may import the old symbols (allowed demo-specific consumers). | The grep across `src/ examples/` returned zero `examples/` matches; verify again during green testing. |
