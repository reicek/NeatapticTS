# Evolution–Training Interoperability Contracts Plan

**Status:** [WIP]

## Purpose

Make hybrid workflows (evolution + gradient fine-tuning) predictable and well-typed by defining **clear contracts** between:

- evolved network representations
- trainable parameter vectors
- optimizer/training loops

This prevents “special-case glue” and improves correctness, reproducibility, and docs.

## Roadmap alignment note

- In the agreed serial pre-NGE sequence, this lane starts after the archived ONNX baseline in `plans/completed/ONNX_EXPORT_PLAN.md` and before `plans/NEATchat.plans.md` is allowed to move from dependency-gated planning into implementation.
- Do not use this file to skip ahead into NEATchat or Phase 7 / NGE work; it is the last non-chat foundation lane before the gated conversational-system follow-up.

## Current status in context

- `plans/Roadmap.md` now places this lane after the archived `plans/completed/ONNX_EXPORT_PLAN.md` baseline and before `plans/NEATchat.plans.md`.
- This lane is now active because the hybrid-interoperability workstream has been explicitly opened; the archived ONNX closure remains the handoff boundary, not a signal to skip directly into later hybrid phases or NEATchat work.
- The current active frontier is now Phase 5 Step 07 closure after a completed Step 05 rerun from current repo state. Phase 4 is [DONE] from current repo state: Step 02 locked the `src/neat/hybrid/` owner boundary, standalone helper, dataset compatibility, worker-ordering note, and conditional blocked note; Step 03 added owner-local tests for `never`, fitness-only `always`, Lamarckian opt-in `always`, the explicit `conditional` blocker, and the missing-`fineTuneOptions` guard; Step 04 implemented `evaluateCandidate` plus the policy or result types in `src/neat/hybrid/` on top of Phase 3 `fineTuneVector(...)`; Step 05 revalidated the hybrid-policy slice with `1` suite and `5` tests green, confirmed `src/neat/hybrid/neat.hybrid.ts` at `100/100/100/100`, kept `src/neat/hybrid/neat.hybrid.types.ts` as a type-only non-runtime coverage surface, and recorded the unchanged external ONNX baseline blocker at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135`; Step 06 improved source-first JSDoc, ran `npm run docs`, and generated `src/neat/hybrid/README.md` without hand-editing generated output; and Phase 5 Step 04 added the bounded root-facade re-exports in `src/neataptic.ts`, expanded the hybrid module overview and policy guidance, and refreshed generated docs for the cohesive public workflow example. Phase 5 Step 05 first confirmed the generated hybrid README carries the Lamarckian explanation, Baldwin-effect distinction with Wikipedia grounding, the three-rung determinism ladder, the recommended defaults progression, the explicit `conditional` blocker, and a cohesive public API example aligned with `src/neataptic.ts`; re-ran the focused hybrid slice with `1` suite and `5` tests green; re-ran `npm run docs`; kept `src/neat/hybrid/neat.hybrid.ts` at `100/100/100/100`; and confirmed `npx tsc --noEmit -p tsconfig.json` still fails only on the unchanged external ONNX `TS2345` baseline at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135`. The prior root-facade coverage-only miss then closed in the owner-local Step 04 follow-up: `src/neataptic.test.ts` touches `toParameterVector`, `fromParameterVector`, `fineTuneVector`, and `evaluateCandidate`; the focused owner-local facade slice passes with `1` suite and `15` tests green; and focused coverage for `src/neataptic.ts` returns `100/100/100/100`. The final Step 05 rerun then re-ran both focused test slices, `npm run docs`, `npx tsc --noEmit -p tsconfig.json`, and the focused coverage gates for `src/neataptic.ts` plus `src/neat/hybrid/neat.hybrid.ts`; all Phase 5 gates are now green except for the unchanged external ONNX `TS2345` baseline. The next safe step is Phase 5 Step 07 closure and archive work; do not reopen Step 04 or Step 05 unless one of these gates regresses.
- The execution frontier is now expressed as numbered phases that each begin with a Step 01 planning packet; future sessions should elaborate a phase only when that phase is explicitly opened.
- The first implementation pass should stay narrow: make deterministic parameter layout plus vector export/import real before adding fine-tuning or NEAT-loop policy hooks.
- Existing repo seams already relevant to this lane:
  - network and visualization surfaces already sort nodes by stable node gene id for deterministic ordering,
  - connection surfaces already expose explicit innovation identifiers plus `Connection.innovationID(...)` as a deterministic fallback,
  - checkpoint and export docs already define replay language that this lane should reuse rather than rephrase independently.

## Goals

- G1: Define a stable way to map a network to/from a parameter vector (“weights + biases”).
- G2: Define training hooks that can be used during evaluation (optional fine-tune) without mutating shared state accidentally.
- G3: Provide deterministic behavior when seed/state handling is enabled.

## Non-goals

- Implementing a full deep learning training framework.
- Supporting automatic differentiation for arbitrary graph operations.

## Owner boundary and likely source surfaces

- The generic parameter-vector contract should be owned by a network surface under `src/architecture/network/`, not by ONNX export, checkpointing, or one downstream application.
- NEAT controller surfaces should own hybrid evaluation policy integration only after the network-owned vector contract is stable.
- Checkpointing, workers, ONNX, and NEATchat should consume the resulting contract rather than defining competing private vector formats.
- Prefer a dedicated network-owned module boundary for this work if the first pass grows beyond a small helper addition.

## Reproducibility contract for this lane

### Replay boundary A — parameter export/import roundtrip

- Target rung: ordered deterministic on the same runtime.
- Required tuple: ordered network state, explicit layout version, serialized parameter values, and runtime environment.
- Phase 1 and Phase 2 should not claim exact replay across Node, browser, and worker paths yet.
- Acceptance proof: repeated exports of the same network yield the same layout and value order, and import restores inference outputs on the same runtime.

### Replay boundary B — isolated fine-tune pass

- Target rung: ordered deterministic on the same runtime when seed, dataset order, training settings, and RNG state are explicit.
- Required tuple extends to `seed`, current RNG state, dataset ordering, optimizer or training-loop settings, serialized parameter vector, runtime environment, and input stream.
- Without explicit seed or RNG capture, the correct claim is best-effort reproducible rather than replay exact.

## Design decisions to lock before the first red test

- What counts as a v1 trainable parameter: the minimum viable scope should be static weights plus biases only unless an additional parameter already participates in the current public training surface.
- Whether disabled connections occupy vector slots: decide once, document it in layout v1, and reject implicit switching between export modes.
- Whether recurrent runtime buffers or carried state are part of the vector: default answer should be no; the vector maps trainable parameters, not transient activation state.
- Precision boundary: keep the public contract in `Float64Array` for fidelity, while downstream export surfaces remain free to quantize separately.
- Compatibility rule: layout version plus exact ordered entry descriptors should determine whether import is allowed; “close enough topology” is not sufficient.

## Key contracts

### Contract A — Parameter vector mapping

A network must be able to:

- export its parameters in a deterministic order
- import parameters back

```ts
export interface ParameterVector {
  values: Float64Array;
  layout: {
    version: 1;
    entries: Array<{
      kind: 'weight' | 'bias';
      from?: number;
      to?: number;
      nodeId?: number;
    }>;
  };
}

export function toParameterVector(network: Network): ParameterVector;
export function fromParameterVector(
  network: Network,
  vector: ParameterVector,
): void;
```

Notes:

- Layout is explicit and versioned.
- Deterministic ordering is required (stable node/edge order).

### Contract B — Trainable view / evaluation isolation

Training during evaluation must not accidentally leak into other candidates.

Proposed model:

- training operates on a cloned network, or
- training operates on a parameter vector and returns a new vector

```ts
export interface FineTuneOptions {
  steps: number;
  learningRate: number;
  seed?: number;
}

export interface FineTuneResult {
  trainedVector: ParameterVector;
  metrics?: Record<string, number>;
}

export async function fineTuneVector(
  vector: ParameterVector,
  dataset: TrainingDataset,
  options: FineTuneOptions,
): Promise<FineTuneResult>;
```

This is compatible with worker evaluation and deterministic checkpointing.

### Contract C — Fitness evaluation semantics

Define an explicit policy for when fine-tuning happens:

- never
- always
- conditional (e.g., only top-K or above threshold)

And whether the trained parameters:

- are thrown away (fitness-only)
- are persisted back into the genome/network (Lamarckian)

## Proposed public API (high-level)

```ts
export interface HybridEvaluationPolicy {
  fineTune: 'never' | 'always' | 'conditional';
  persistTrainedWeights: boolean;
}

export interface EvaluateCandidateOptions {
  policy: HybridEvaluationPolicy;
  fineTuneOptions?: FineTuneOptions;
}

export async function evaluateCandidate(
  network: Network,
  dataset: TrainingDataset,
  options: EvaluateCandidateOptions,
): Promise<{ fitness: number; trainedNetwork?: Network }>;
```

Exact naming should align with existing code; the important part is to make policies explicit.

## Downstream unlock — NEATchat follow-up

This plan is one of the hard blockers for `plans/NEATchat.plans.md`.
NEATchat should not start comparing, fine-tuning, or promoting personalized
candidates until the repo has an explicit and deterministic training-isolation
contract.

For NEATchat, this plan is considered ready only when all of the following are
true:

1. Phase 1 and Phase 2 are complete, so deterministic parameter layout plus
   vector export/import are real and roundtrip inference correctly.
2. Phase 3 is complete, so fine-tuning cannot mutate shared candidate state by
   accident.
3. Phase 4 is complete, so the policy for fitness-only versus persistent
   training is explicit rather than hidden inside downstream application code.
4. Phase 5 is complete, so the public contract is documented clearly enough that NEATchat can compare a frozen base, a personalized variant, and candidate deltas without inventing a parallel private vector format.

This plan does not replace checkpointing or worker execution. It provides the
promotion and isolation seam that those other NEATchat blockers rely on.

## Recommended agent + skill combo by phase

- Phase 1 — `Hybrid Interop Scout` + `hybrid-training-interop`
- Phase 2 — `Hybrid Interop Scout` + `hybrid-training-interop`
- Phase 3 — `Hybrid Interop Scout` + `hybrid-training-interop`
- Phase 4 — `Hybrid Interop Scout` + `hybrid-training-interop`
- Phase 5 — `Docs Scout` + `educational-docs`

## Phased implementation flow

Phase ordering and stop-line guidance:

- Phase 1 and Phase 2 still form the first implementation tranche and should land together unless a red-phase test proves that the ordering can be isolated safely on its own.
- Phase 3 should introduce one minimal isolation primitive first; avoid opening both clone-based and vector-based ergonomics before one deterministic base path is stable.
- Phase 4 should not start until Phase 3 proves that fine-tuning can happen without shared candidate mutation.
- Phase 5 closes the lane only after public naming, determinism wording, and the example surface are stable enough for downstream consumers.

### Phase 1 — Stabilize deterministic parameter ordering [DONE]

**Phase objective:** Lock the layout-owned ordering contract so parameter export is deterministic, versionable, and owned by the network boundary rather than by adjacent export or checkpoint surfaces.

**Phase progression rule:** Start with only Step 01. Step 01 must author the remaining numbered step packets, or explicit skipped-step packets, before the phase can advance.

**Preserved implementation contract:**

- Require stable node and edge ordering.
- Define source-owned `ParameterLayoutV1` and `ParameterLayoutEntry` types rather than deriving layout from ONNX export structures.
- Lock a canonical ordering rule for v1:
  1. bias entries ordered by stable node gene id,
  2. weight entries ordered by explicit stable edge identity, preferring connection innovation when present and an explicit deterministic fallback when not,
  3. one documented fold order for entry kinds, kept stable under version `1`.
- Reject ambiguous ordering cases rather than relying on incidental container order.
- Add a narrow owner-local test slice covering repeated export stability and equivalent reconstructed network ordering.

**Acceptance when this phase closes:**

- Same network always produces the same layout ordering.

**Focused validation to preserve:**

- Repeated same-runtime export yields identical ordered entry descriptors.
- Equivalent reconstructed networks yield the same layout ordering.

#### Step 01: Plan the deterministic ordering tranche [DONE]

```yaml
phase: 1
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning-architect.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 02 — Research deterministic ordering boundary'
skills:
  - 'hybrid-training-interop'
specialists:
  - 'Hybrid Interop Scout'
validation:
  - 'Manual tracker check that the canonical ordering rule, acceptance gate, and focused validation are preserved before Step 02-07 packets are authored.'
```

**User instruction:** Start a fresh session, select `01-planning`, and paste this full step packet.

**Step objective:** Turn Phase 1 into an executable deterministic-ordering workset by authoring Phase 1 Step 02-07 packets, deciding which value-gate steps are real versus explicit skips, and preserving the existing ordering contract without widening scope into policy or fine-tuning work.

**Context the agent must know:**

- This lane remains `[PLANNED]` until the user explicitly opens hybrid interoperability work; planning must not start implementation early.
- The network boundary owns `ParameterLayoutV1` and deterministic ordering; ONNX, checkpointing, workers, and NEATchat are downstream consumers.
- The first implementation tranche is still coupled with Phase 2 unless a red-phase test proves the ordering can be isolated honestly.
- Same-runtime ordered determinism is the current replay claim; cross-runtime exact replay is not in scope for this phase.

**Execution steps:**

1. Reconfirm the preserved Phase 1 contract, acceptance rule, and focused validation against the current plan text and roadmap.
2. Decide whether Research, Red Testing, Green Validation, Documentation, and Session Logging each add independent value for this phase or should be recorded as explicit skipped-step packets.
3. Author Phase 1 Step 02-07 packets or explicit skipped-step packets with the narrowest honest boundary-local scope.
4. Record any Phase 1 to Phase 2 coupling constraints in the plan so the first implementation tranche stays aligned.

**Stop conditions:**

- Done: Phase 1 has a paste-ready next active step and all remaining step slots are authored or explicitly skipped.
- Blocked: owner-boundary ambiguity or roadmap tension prevents a safe deterministic-ordering packet.
- Route back: if planning reveals that Phase 1 cannot be separated from a prerequisite not already documented in this plan, update the tracker and route back to planning rather than widening implementation.

**Required validation:** Manual diff review confirming that every former Step 1 detail now lives in Phase 1 and that the first-tranche coupling note with Phase 2 remains explicit.

**Plan update requirement:** Update this plan with the new Step 02-07 packets, any skipped-step rationale, validation evidence, and the next active step before ending the session.

**Planning decisions and validation evidence:** Step 01 preserved the Phase 1 contract, roadmap placement, same-runtime determinism wording, and Phase 1 to Phase 2 coupling. `plans/README.md` and `plans/Roadmap.md` confirm this lane is the next non-chat foundation handoff after the archived ONNX baseline and before gated NEATchat follow-up. Plan Scout found no roadmap or owner-boundary blocker. Research, Red Testing, Implementation, Green Validation, Documentation, and Session Logging all add independent value because this phase changes a network-owned deterministic contract; no Step 02-07 slot is skipped.

#### Step 02: Research deterministic ordering boundary [DONE]

```yaml
phase: 1
step: 2
agent: '02-researching'
agent_file: '.github/agents/02-research-coordinator.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 03 — Design deterministic ordering red tests'
skills:
  - 'hybrid-training-interop'
  - 'plan-alignment'
specialists:
  - 'Hybrid Interop Scout'
validation:
  - 'Read-only research brief names the network owner boundary, stable node and edge identities, likely owner-local test file, and whether Phase 1 ordering can be red-tested without implementing Phase 2 import/export.'
```

**User instruction:** Start a fresh session, select `02-researching`, and paste this full step packet.

**Step objective:** Gather the smallest read-only evidence needed to design a deterministic ordering red test without widening into vector import/export, fine-tuning, hybrid policy, or downstream consumer formats.

**Context the agent must know:**

- Phase 1 owns only deterministic `ParameterLayoutV1` ordering under the network boundary.
- ONNX, checkpointing, workers, and NEATchat must stay downstream consumers rather than sources of truth.
- Phase 1 and Phase 2 remain a coupled first implementation tranche unless Step 03 proves the ordering test can stand honestly on its own.
- Same-runtime ordered determinism is the replay claim; cross-runtime exact replay is out of scope.

**Execution steps:**

1. Read `plans/README.md`, `plans/Roadmap.md`, this Phase 1 section, and the nearest relevant `src/architecture/network/**/README.md` surfaces before individual source files.
2. Delegate a focused read-only packet to `Hybrid Interop Scout` to map stable node identity, stable connection identity, deterministic fallback risks, and existing tests near the network boundary.
3. Identify the narrowest owner-local test location for repeated layout ordering and equivalent reconstructed network ordering.
4. Record whether Phase 1 ordering can be observed before Phase 2 vector import/export, and preserve any coupling constraint directly in this plan.

**Stop conditions:**

- Done: research evidence identifies the owner boundary, source/test files, stable ordering inputs, ambiguity risks, and the next red-test target.
- Blocked: stable node or edge identity cannot be identified from current source evidence.
- Route back: roadmap or owner-boundary evidence conflicts with the current Phase 1 contract; update the plan instead of designing tests.

**Required validation:** Manual evidence review confirming the research remains read-only, boundary-local, and aligned with the Phase 1 acceptance rule.

**Plan update requirement:** Update this plan with the research brief, any Phase 1 to Phase 2 coupling note, and the Step 03 handoff before ending the session.

**Research brief and validation evidence:** `plans/README.md` and `plans/Roadmap.md` still align this lane as the next non-chat foundation handoff after the archived ONNX baseline, with no roadmap conflict for a network-owned ordering contract. The nearest relevant README surfaces were `src/architecture/network/README.md`, `src/architecture/network/serialize/README.md`, and `src/architecture/network/genetic/README.md`; together they keep ordering under the public `Network` boundary while treating genetic and restore code as adjacent identity evidence rather than as new owners.

- Owner boundary: the best existing owner-local seam for `ParameterLayoutV1` ordering is the network serialize chapter, with shared historical-identity types already present in `src/architecture/network/network.types.ts` and the likely implementation surface adjacent to `src/architecture/network/serialize/network.serialize.utils.ts`.
- Stable node identity: `Node.geneId` is the stable node identifier, assigned in `src/architecture/node/node.ts` and kept monotonic through `Node.syncGeneIdCounter(...)`.
- Stable connection identity: `Connection.innovation` is the preferred stable edge identity, assigned in `src/architecture/connection/connection.ts` and preserved by the network-owned `ConnectionHistoricalIdentity` fields (`innovation`, `fromGeneId`, `toGeneId`, `gaterGeneId`) plus restore helpers in the serialize seam.
- Deterministic fallback risks: runtime node indices are not honest ordering inputs because the genetic seam strips index hints before materialization; endpoint-pair fallback via `Connection.innovationID(...)` is deterministic but weaker than true historical innovation; if the explicit fallback still leaves ambiguity, Phase 1 should reject that case instead of inheriting incidental container order.
- Narrowest owner-local red-test location: `src/architecture/network/serialize/network.serialize.test.ts` is the best Step 03 seam because it already verifies historical identity preservation across `toJSON()`, `clone()`, and compact `serialize()` or `deserialize()` rebuild flows.
- Phase 1-only observability: repeated ordered-entry descriptors can be tested honestly before Phase 2 import or export write-back by comparing one live network against repeated reads from the same instance and equivalent reconstructed networks from `Network.fromJSON(network.toJSON())` and `Network.deserialize(network.serialize(), input, output)`. This supports a real Step 03 red test for descriptor ordering only.

**Phase 1 to Phase 2 coupling note:** Step 02 confirms that Step 03 can isolate a truthful red test for ordered entry descriptors before `toParameterVector` or `fromParameterVector` exist, but that does not decouple the first implementation tranche. Phase 1 and Phase 2 still land together unless later red or green evidence shows the implementation itself can remain honestly separated.

**Step 03 handoff:** Add the smallest failing owner-local test in `src/architecture/network/serialize/network.serialize.test.ts` for a pure layout-descriptor seam. Cover two observables only: repeated same-runtime descriptor ordering on one live network, and matching descriptor ordering for equivalent reconstructed networks. Bias descriptors should order by stable `node.geneId`; weight descriptors should prefer `connection.innovation`, use one explicit non-index fallback when innovation is absent, and fail on unresolved ambiguity instead of inheriting array order. Keep Step 03 out of vector values, import validation, roundtrip inference, fine-tuning, and hybrid policy behavior.

#### Step 03: Design deterministic ordering red tests [DONE]

```yaml
phase: 1
step: 3
agent: '03-red-testing'
agent_file: '.github/agents/03-red-test-architect.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 04 — Implement ParameterLayoutV1 ordering'
skills:
  - 'hybrid-training-interop'
  - 'red-test-contracts'
  - 'reproducibility-contracts'
specialists:
  - 'Determinism Scout'
validation:
  - 'A focused owner-local red test or explicit blocked record proves whether repeated same-runtime layout ordering and equivalent reconstructed network ordering can be tested before Phase 2.'
```

**User instruction:** Start a fresh session, select `03-red-testing`, and paste this full step packet.

**Step objective:** Add or define the smallest failing test contract for deterministic layout ordering before implementation, while keeping vector values, import validation, and roundtrip inference in Phase 2.

**Context the agent must know:**

- The red contract should cover ordered entry descriptors, not optimizer behavior or fine-tuning.
- The required assertions are repeated export stability and equivalent reconstructed network ordering on the same runtime.
- Tests must follow repo conventions, including owner-local placement and one top-level `expect(...)` per test.
- If no honest Phase 1-only observable exists, record that proof and keep Phase 1 coupled to Phase 2 rather than inventing a fake seam.

**Execution steps:**

1. Read Step 02 evidence, the target source boundary, and the nearest existing owner-local tests.
2. Add the smallest failing test or test-design record for `ParameterLayoutV1` descriptor ordering.
3. Ensure the test does not require `fromParameterVector`, value import, fine-tuning, or hybrid evaluation policy.
4. Run the narrowest practical Jest command for the touched test file or record why execution is blocked.
5. Update this plan with the failing command, expected failure, and exact Step 04 green condition.

**Stop conditions:**

- Done: a focused red test fails for the missing deterministic layout contract or a precise blocked record proves Phase 1 cannot be isolated before Phase 2.
- Blocked: the research evidence leaves the observable behavior or owner-local test boundary ambiguous.
- Route back: the red design requires Phase 2 import/export or policy work; update the coupling note and return to planning.

**Required validation:** Focused red-test command evidence, or a manual blocked record that explains why no honest Phase 1-only red test exists.

**Plan update requirement:** Update this plan with changed test files or blocked rationale, red evidence, and the Step 04 implementation handoff before ending the session.

**Step 03 red contract and validation evidence:** Added owner-local red tests in `src/architecture/network/serialize/network.serialize.test.ts` that import the planned `createParameterLayoutV1` helper from the serialize boundary and lock two Phase 1 observables only: repeated same-runtime descriptor ordering on one live runtime, and matching descriptor ordering across equivalent `Network.fromJSON(...)` and `Network.deserialize(...)` rebuilds. The JSON rebuild fixture perturbs hidden-node order and connection-row order while preserving input/output role order so the contract cannot pass on incidental container order alone.

- Changed test file: `src/architecture/network/serialize/network.serialize.test.ts`
- Focused command: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --testNamePattern="ParameterLayoutV1"`
- Observed red failure: `TS2305: Module "./network.serialize.utils" has no exported member "createParameterLayoutV1".`
- Scope guard: Step 03 stays descriptor-only; it does not introduce parameter values, `fromParameterVector`, roundtrip inference, fine-tuning, or hybrid evaluation policy behavior.

**Step 04 green condition:** Export the smallest network-owned `createParameterLayoutV1` helper from `src/architecture/network/serialize/network.serialize.utils.ts` so the focused Step 03 Jest slice goes green with version `1` descriptors, bias entries folded before weight entries, stable per-kind ordering, and matching summaries for the live runtime, the reordered `fromJSON()` rebuild, and the compact `deserialize()` rebuild.

#### Step 04: Implement ParameterLayoutV1 ordering [DONE]

```yaml
phase: 1
step: 4
agent: '04-implementing'
agent_file: '.github/agents/04-implementation-architect.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 05 — Validate deterministic ordering gates'
skills:
  - 'hybrid-training-interop'
  - 'reproducibility-contracts'
specialists:
  - 'Hybrid Interop Scout'
validation:
  - 'The Step 03 red test goes green with a network-owned layout ordering implementation and no Phase 2 vector import/export behavior.'
```

**User instruction:** Start a fresh session, select `04-implementing`, and paste this full step packet.

**Step objective:** Implement the smallest network-owned `ParameterLayoutV1` and `ParameterLayoutEntry` ordering surface that satisfies Step 03 without adding value import, roundtrip inference, fine-tuning, or hybrid policy behavior.

**Context the agent must know:**

- Bias entries are ordered by stable node gene id.
- Weight entries are ordered by explicit stable edge identity, preferring connection innovation when present and an explicit deterministic fallback when not.
- The fold order for entry kinds must be documented and stable under layout version `1`.
- Ambiguous ordering cases should fail explicitly rather than rely on incidental object or array order.

**Execution steps:**

1. Read the Phase 1 plan text, Step 03 red evidence, the nearest folder README, and the source/test files named by Step 02 and Step 03.
2. Add the minimal network-owned layout types and ordering helper or module, using a dedicated folder boundary only if the change is larger than a small helper addition.
3. Keep implementation scoped to descriptor ordering and ambiguity rejection; do not implement `fromParameterVector`, inference roundtrip, training isolation, or policy hooks.
4. Add concise JSDoc for exported layout types and any public helper introduced by this step.
5. Rerun the focused Step 03 command and update this plan with changed files, remaining risks, and expected Step 05 validation gates.

**Stop conditions:**

- Done: the focused red test is green and the implementation remains network-owned and Phase 1 scoped.
- Blocked: stable edge identity or fallback semantics cannot be implemented without changing upstream contracts.
- Route back: implementation reveals the ordering seam cannot be separated from Phase 2 vector export/import; update the plan and return to planning.

**Required validation:** Focused Step 03 test command passes after the implementation change.

**Plan update requirement:** Update this plan with implementation files, focused validation evidence, any ambiguity decisions, and the Step 05 validation handoff before ending the session.

**Step 04 implementation and validation evidence:** Added the smallest network-owned `ParameterLayoutV1` surface at the serialize boundary by exporting `ParameterLayoutEntry` and `ParameterLayoutV1` from `src/architecture/network/serialize/network.serialize.utils.types.ts` and `createParameterLayoutV1(network)` from `src/architecture/network/serialize/network.serialize.utils.ts`. The helper stays Phase 1 scoped: it returns version `1` descriptor metadata only, folds all bias entries before all weight entries, sorts biases by stable node gene id, sorts weights by connection innovation when present and by endpoint gene-id fallback when innovation is absent, and does not introduce parameter values, import behavior, roundtrip inference, fine-tuning, or hybrid policy hooks.

- Changed implementation files: `src/architecture/network/serialize/network.serialize.utils.ts`, `src/architecture/network/serialize/network.serialize.utils.types.ts`
- Focused command: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --testNamePattern="ParameterLayoutV1"`
- Observed green result: the focused `ParameterLayoutV1 ordering` slice passed for repeated same-runtime reads and for equivalent `Network.fromJSON(...)` and `Network.deserialize(...)` rebuilds.
- Ambiguity decisions: the helper fails explicitly when a node lacks a stable gene id, when two bias descriptors would share the same node id, or when two weight descriptors would share the same stable ordering identity. Weight identity prefers `innovation` when present and rejects duplicate innovations as ambiguous; when innovation is absent it falls back to stable endpoint gene ids and rejects duplicate fallback pairs as ambiguous.
- Remaining risks: Step 04 proves only the descriptor-ordering seam. Step 05 still needs to rerun the focused slice, run `npx tsc --noEmit -p tsconfig.json`, and enforce coverage guard for the touched `src/` files.

**Step 05 validation handoff:** Treat `src/architecture/network/serialize/network.serialize.utils.ts` and `src/architecture/network/serialize/network.serialize.utils.types.ts` as the touched source boundary. Re-run the focused `ParameterLayoutV1` Jest slice first, then run `npx tsc --noEmit -p tsconfig.json`, then run coverage guard on each touched `src/` file. If validation fails, route back to Step 04 and keep fixes inside the layout-ordering seam only.

#### Step 05: Validate deterministic ordering gates [DONE]

```yaml
phase: 1
step: 5
agent: '05-green-testing'
agent_file: '.github/agents/05-green-validation-runner.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 07 — Log Phase 1 closure or reroute'
skills:
  - 'green-validation-gates'
  - 'coverage-guard'
  - 'hybrid-training-interop'
specialists:
  - 'Coverage Guard'
  - 'Determinism Scout'
validation:
  - 'Focused deterministic-ordering tests and owner-local coverage pass for every touched serialize src file, and repo-wide TypeScript introduces no new failure into the seam beyond the recorded ONNX baseline blocker.'
```

**User instruction:** Start a fresh session, select `05-green-testing`, and paste this full step packet.

**Step objective:** Prove the Phase 1 implementation satisfies the deterministic-ordering contract with focused validation before documentation or phase closure.

**Context the agent must know:**

- Validate only the Phase 1 layout-ordering surface and directly touched boundaries.
- Do not run ahead into Phase 2 roundtrip, mismatch rejection, fine-tuning, or policy semantics.
- If `src/` files changed, coverage guard is required for each touched source file.
- Failed validation routes back to the smallest prior step that can fix it.

**Execution steps:**

1. Read Step 04 changed-file summary and expected validation commands.
2. Rerun the focused deterministic-ordering Jest slice.
3. Run `npx tsc --noEmit -p tsconfig.json` after the focused slice is green.
4. Run coverage guard for every touched `src/` file, or record why no `src/` coverage gate applies.
5. Update this plan with pass/fail evidence and route failures to Step 03 or Step 04 as appropriate.

**Stop conditions:**

- Done: focused tests and required coverage gates pass, and any remaining TypeScript failure is explicitly confirmed as the unchanged external baseline rather than a new serialize-seam regression.
- Blocked: a validation command cannot run or produces an environmental failure that needs user intervention.
- Route back: test failures return to Step 04; missing or incorrect red coverage returns to Step 03.

**Required validation:** Command evidence for the focused test slice, `npx tsc --noEmit -p tsconfig.json`, and coverage guard on touched `src/` files when applicable.

**Plan update requirement:** Update this plan with validation evidence, reroute decisions if any, and the Step 06 documentation handoff before ending the session.

**Step 05 validation evidence and reroute:** Re-ran the focused deterministic-ordering slice with `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --testNamePattern="ParameterLayoutV1"`. The Phase 1 behavior gate stayed green: `1` suite passed, `2` targeted `ParameterLayoutV1 ordering` tests passed, and no Step 04 implementation reroute was needed.

Repo-wide TypeScript validation did not clear. `npx tsc --noEmit -p tsconfig.json` failed in the unrelated ONNX validation boundary at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` with `TS2345`, because `standardDomainImports[0]!.version` can be `undefined`. That error sits outside the Phase 1 layout-ordering seam, so Step 05 cannot close until the repo TypeScript baseline is restored or the user narrows the required gate.

Coverage guard on the touched `src/` files used the owner-local serialize suite: `npx jest --config=jest.config.mjs --no-cache --coverage --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --collectCoverageFrom='src/architecture/network/serialize/network.serialize.utils.ts' --collectCoverageFrom='src/architecture/network/serialize/network.serialize.utils.types.ts' --coverageReporters=text --coverageReporters=json-summary`. `src/architecture/network/serialize/network.serialize.utils.types.ts` stayed at `100/100/100/100`, but `src/architecture/network/serialize/network.serialize.utils.ts` failed the guard at `95.40` statements, `77.94` branches, `100` functions, and `95.23` lines with uncovered lines `257-261`, `273`, and `289`. Those gaps are in the new layout-ordering guard-rail branches, so the validation reroute goes back to Step 03 for the smallest owner-local red cases that cover the missing ambiguity and missing-stable-id paths before Step 05 is rerun.

**Step 05 rerun after the Step 03 coverage follow-up:** Re-ran the focused deterministic-ordering slice with `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --testNamePattern="ParameterLayoutV1 ordering"`. The behavior gate stayed green again: `1` suite passed and all `5` targeted `ParameterLayoutV1 ordering` tests passed, including the repeated-read, equivalent-rebuild, finite-innovation-before-fallback, duplicate-fallback-identity, and missing-stable-weight-endpoint cases.

Repo-wide TypeScript validation still does not clear. `npx tsc --noEmit -p tsconfig.json` failed again in the unrelated ONNX validation boundary at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` with the same `TS2345`, because `standardDomainImports[0]!.version` can be `undefined`. That baseline remains outside the Phase 1 layout-ordering seam, so this rerun confirms the external blocker still prevents TypeScript closure.

The owner-local coverage guard improved but still does not clear the touched seam. Re-ran `npx jest --config=jest.config.mjs --no-cache --coverage --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --collectCoverageFrom='src/architecture/network/serialize/network.serialize.utils.ts' --collectCoverageFrom='src/architecture/network/serialize/network.serialize.utils.types.ts' --coverageReporters=text --coverageReporters=json-summary`. `src/architecture/network/serialize/network.serialize.utils.types.ts` stayed at `100/100/100/100`, and `src/architecture/network/serialize/network.serialize.utils.ts` improved to `98.97` statements, `94.11` branches, `100` functions, and `98.94` lines. The previously uncovered lines `257-261`, `273`, and `289` are now closed, but the guard still reports uncovered lines `158` and `239`, which correspond to the duplicate-bias-node ambiguity branch and the same-innovation secondary weight ordering branch when source gene ids differ. The validation reroute stays with Step 03 for the smallest owner-local red cases that cover those two remaining layout-ordering paths before Step 05 is rerun again.

Tracker sync validation still passes for the active workstream after this Step 05 rerun update with `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json`: `PASS plan sync: 0 errors, 0 warnings`.

**Step 03 owner-local red follow-up after the second Step 05 reroute:** Added two more owner-local `ParameterLayoutV1 ordering` guard-rail tests in `src/architecture/network/serialize/network.serialize.test.ts`. The new cases stay inside the layout-ordering seam only: one forces duplicate bias node ids so `assertDistinctBiasOrdering(...)` rejects an ambiguous bias fold, and the other forces two connections with the same finite innovation but different source gene ids so the secondary `fromGeneId` sort branch is exercised before duplicate innovation identity is rejected.

- Changed test file: `src/architecture/network/serialize/network.serialize.test.ts`
- Focused command: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --testNamePattern="ParameterLayoutV1 ordering"`
- Observed follow-up result: the focused owner-local slice stayed green with `1` suite passed and all `7` targeted `ParameterLayoutV1 ordering` tests passed. The layout-ordering seam is now covered for repeated same-runtime reads, equivalent rebuild ordering, finite-innovation-before-fallback ordering, duplicate fallback identity ambiguity, duplicate bias node-id ambiguity, same-innovation ambiguity across distinct source nodes, and missing stable weight-endpoint gene ids.

**Updated Step 05 rerun handoff:** Call `05-green-testing` next. Re-run `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --testNamePattern="ParameterLayoutV1 ordering"`, then `npx tsc --noEmit -p tsconfig.json`, then the owner-local coverage guard command for `src/architecture/network/serialize/network.serialize.utils.ts` and `src/architecture/network/serialize/network.serialize.utils.types.ts`. Treat the existing `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` as the unchanged repo baseline blocker unless a new failure appears inside the serialize seam. If the focused slice or coverage guard fails inside layout ordering, route back to Step 03 for missing guard-rail tests or Step 04 for implementation defects. Do not activate Phase 2.

**Exact next orchestrator prompt:** Re-run Phase 1 Step 05 for the layout-ordering seam only. Use `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --testNamePattern="ParameterLayoutV1 ordering"` first, then `npx tsc --noEmit -p tsconfig.json` while treating the existing `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` as the unchanged repo baseline blocker unless the failure surface moves into serialize ordering, then run the owner-local coverage guard command for `src/architecture/network/serialize/network.serialize.utils.ts` and `src/architecture/network/serialize/network.serialize.utils.types.ts`. If the focused slice or coverage guard fails inside the layout-ordering seam, route to Step 03 for missing guard-rail coverage or Step 04 for implementation defects. Do not activate Phase 2.

**Step 06 documentation gate note:** A source-first documentation pass can keep the new layout surface teachable and regenerate the serialize README, but Phase 1 still cannot close until the Step 03 coverage follow-up lands and the repo-wide TypeScript baseline is green again or the user narrows that required gate.

#### Step 06: Curate layout-ordering docs [DONE]

```yaml
phase: 1
step: 6
agent: '06-documenting'
agent_file: '.github/agents/06-educational-docs-curator.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 07 — Log Phase 1 closure or reroute'
skills:
  - 'educational-docs'
  - 'docs-academic-citation-audit'
  - 'hybrid-training-interop'
specialists:
  - 'Docs Scout'
validation:
  - 'Source-first docs for any exported layout symbols are present, generated docs are refreshed if JSDoc inputs changed, and no Phase 2 or fine-tuning docs are introduced.'
```

**User instruction:** Start a fresh session, select `06-documenting`, and paste this full step packet.

**Step objective:** Keep the deterministic layout-ordering surface teachable and synchronized with generated docs without turning Phase 1 into the broader Phase 5 documentation closure.

**Context the agent must know:**

- Documentation scope is limited to `ParameterLayoutV1`, `ParameterLayoutEntry`, ordering rules, ambiguity rejection, and same-runtime determinism wording.
- Phase 2 vector roundtrip, Phase 3 isolation, Phase 4 policy, and Phase 5 examples remain out of scope.
- Generated `src/**/README.md` files and `docs/examples/**` outputs are read-only; update source JSDoc and run generation when needed.
- Public docs should not mention tracker phases, roadmap chronology, or chat-only context.

**Execution steps:**

1. Read Step 04 changed public surfaces and Step 05 validation evidence.
2. Improve only source-owned JSDoc or nearby handwritten docs required by the new layout-ordering surface.
3. Run `npm run docs` if JSDoc or generated-doc inputs changed; otherwise record a manual no-docs-needed rationale.
4. Verify generated docs were not hand-edited as a shortcut.
5. Update this plan with documentation evidence and any residual docs risk.

**Stop conditions:**

- Done: required source docs are aligned and docs generation either passes or is explicitly not needed.
- Blocked: public naming or determinism wording is unstable enough that docs would mislead downstream consumers.
- Route back: documentation reveals implementation drift in the layout contract; route to Step 04 or Step 05 rather than widening docs.

**Required validation:** `npm run docs` when source JSDoc or generated-doc inputs changed, or a manual no-docs-needed record when implementation introduced no doc-facing surface.

**Step 06 documentation changes and validation evidence:** Tightened the source-owned JSDoc for `ParameterLayoutEntry` and `ParameterLayoutV1` in `src/architecture/network/serialize/network.serialize.utils.types.ts` and for `createParameterLayoutV1` in `src/architecture/network/serialize/network.serialize.utils.ts`. The doc pass stayed Phase 1 scoped: it teaches only the bias-before-weight fold order, innovation-preferred fallback ordering, explicit ambiguity rejection, and the same-runtime ordered-determinism boundary for a fixed topology with stable historical ids.

- Changed source docs: `src/architecture/network/serialize/network.serialize.utils.types.ts`, `src/architecture/network/serialize/network.serialize.utils.ts`
- Validation commands: `npm run docs` twice. The first refresh proved the generated README was stale and surfaced that the docs generator did not render the new determinism wording while it still lived under `@remarks`; the second refresh, after moving that wording into the main descriptions, completed successfully.
- Generated output refreshed: `src/architecture/network/serialize/README.md` now includes `ParameterLayoutEntry`, `ParameterLayoutV1`, and `createParameterLayoutV1` with the intended ordering, ambiguity-rejection, and same-runtime determinism wording.
- Source-of-truth guard: no generated `src/**/README.md` file or `docs/examples/**` output was hand-edited as a shortcut. The only manual edits in this step were to the two source JSDoc files, and the README refresh came from `npm run docs`.
- Scope guard: no Phase 2 vector roundtrip, Phase 3 isolation, Phase 4 policy, or Phase 5 example documentation was introduced.
- Citation note: no external citation was added because this narrow layout-ordering contract is repo-owned documentation rather than an external algorithm summary.
- Residual docs risk: none within the Phase 1 ordering seam; Step 07 should decide phase closure against the unchanged external ONNX TypeScript baseline and preserve the Phase 1 to Phase 2 coupling note.

**Plan update requirement:** Update this plan with documentation changes, validation evidence, residual gaps, and the Step 07 closure handoff before ending the session.

#### Step 07: Log Phase 1 closure or reroute [DONE]

```yaml
phase: 1
step: 7
agent: '07-logging'
agent_file: '.github/agents/07-session-log-archivist.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Phase 2 Step 03 — Design vector roundtrip red tests'
skills:
  - 'tracker-handoff'
  - 'plan-sync-validation'
  - 'hybrid-training-interop'
specialists:
  - 'Plan Registration Auditor'
validation:
  - 'Manual tracker review confirms Phase 1 acceptance evidence, Phase 2 coupling state, and the refreshed Handoff query before the next phase opens.'
```

**User instruction:** Start a fresh session, select `07-logging`, and paste this full step packet.

**Step objective:** Close or reroute Phase 1 based on validated evidence, preserve the Phase 1 to Phase 2 coupling decision, and leave the tracker ready for the next narrow step.

**Context the agent must know:**

- Phase 1 closes only when same-network layout ordering is stable and validated.
- The first implementation tranche remains coupled with Phase 2 unless the red and green evidence explicitly proves Phase 1 was safely isolated.
- The overall lane remains active/planned; do not archive this tracker unless the entire hybrid-interoperability workstream is done.
- Chat is not the source of truth; the tracker and handoff query must carry the next session.

**Execution steps:**

1. Read Phase 1 Step 02-06 evidence and verify no required validation is missing.
2. Mark completed Phase 1 steps `[DONE]` or record the smallest reroute if evidence is incomplete.
3. Record whether Phase 2 remains coupled to the opening implementation tranche or whether Phase 1 red/green evidence justified isolation.
4. Refresh the `Handoff query` to point to the next safe step, which is Phase 2 Step 03 when Phase 1 closes because Phase 2 Step 01 and Step 02 are already prepared.
5. Run or record focused tracker validation appropriate to the changed plan surface.

**Stop conditions:**

- Done: Phase 1 evidence is compactly recorded, next step is explicit, and the handoff query is current.
- Blocked: validation, documentation, or ownership evidence is missing and cannot be reconstructed from the tracker.
- Route back: missing tests route to Step 03, implementation drift routes to Step 04, failed validation routes to Step 05, and doc drift routes to Step 06.

**Required validation:** Manual tracker diff review confirming Phase 1 acceptance evidence, no premature Phase 2 behavior, and a current next-step handoff.

**Plan update requirement:** Update this plan with Phase 1 closure or reroute state, refreshed handoff query, and the next active step before ending the session.

**Step 07 closure review and handoff evidence:** Step 07 reviewed the recorded Phase 1 evidence from Step 02 through Step 06 together with the latest Step 05 rerun from current repo state only. Phase 1 can now close. The acceptance criterion and focused validation for deterministic layout ordering are satisfied inside the owner-local serialize boundary, and the only remaining repo-wide TypeScript failure is the unchanged external ONNX baseline blocker.

- Phase 1 closure decision: close Phase 1. The focused `ParameterLayoutV1 ordering` slice passed with `1` suite and all `8` targeted ordering tests, and the owner-local coverage guard now reports `100/100/100/100` for both `src/architecture/network/serialize/network.serialize.utils.ts` and `src/architecture/network/serialize/network.serialize.utils.types.ts`.
- External baseline decision: do not keep Phase 1 open solely because `npx tsc --noEmit -p tsconfig.json` still fails at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` with `TS2345` because `standardDomainImports[0]!.version` can be `undefined`. No new failure moved into the serialize seam, so that error remains a recorded repo baseline blocker rather than a Phase 1 reroute.
- Coupling decision: preserve the Phase 1 to Phase 2 coupling note. Phase 1 and Phase 2 still form the opening implementation tranche, but the coupling no longer blocks activation because the Phase 1 ordering seam is now closed.
- Tracker state decision: keep the top-level tracker active in `plans/` as `[WIP]`; mark Phase 1 complete; move the active frontier to Phase 2; do not archive the plan or create a same-boundary `.logs.md` record because the overall hybrid-interoperability lane still has real next steps.
- Next safe step: activate `03-red-testing` for Phase 2 `Step 03 — Design vector roundtrip red tests`. Do not skip ahead to Phase 2 Step 04 or later packets.
- Tracker validation: rerun `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` after this tracker refresh.

### Phase 2 — Implement vector export/import [WIP]

**Phase objective:** Add the versioned network-owned export and import surface so deterministic layout metadata and parameter values can roundtrip safely on the same runtime.

**Phase progression rule:** Start with only Step 01. Step 01 must author the remaining numbered step packets, or explicit skipped-step packets, before the phase can advance.

**Preserved implementation contract:**

- Implement `toParameterVector` and `fromParameterVector`.
- Export values plus layout together from the network-owned boundary.
- Validate version compatibility, entry-count compatibility, and descriptor compatibility before mutating the target network.
- Keep recurrent and advanced topology support honest: when a parameter family cannot be mapped cleanly under layout v1, fail explicitly instead of widening the promise silently.
- Add negative tests for version mismatch, length mismatch, and layout-descriptor mismatch.

**Acceptance when this phase closes:**

- Roundtrip preserves inference outputs.

**Focused validation to preserve:**

- Export plus import preserves outputs for the same runtime and topology.
- Invalid vector or layout payloads fail explicitly.

#### Step 01: Plan the vector roundtrip tranche [DONE]

```yaml
phase: 2
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning-architect.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 02 — Research vector roundtrip owner boundary'
skills:
  - 'hybrid-training-interop'
specialists:
  - 'Hybrid Interop Scout'
validation:
  - 'Manual tracker check that roundtrip, mismatch-rejection, and network-owned contract details are preserved before Step 02-07 packets are authored.'
```

**User instruction:** Start a fresh session, select `01-planning`, and paste this full step packet.

**Step objective:** Turn Phase 2 into an executable vector export/import workset by authoring Phase 2 Step 02-07 packets, preserving the roundtrip and mismatch contracts, and keeping the phase aligned with the coupled Phase 1 opening tranche.

**Context the agent must know:**

- Phase 2 depends on the deterministic ordering contract from Phase 1 and should still be planned as part of the same initial implementation tranche unless a red-phase test proves otherwise.
- The layout version, entry descriptors, and values must travel together; downstream consumers must not infer compatibility from topology similarity alone.
- Unsupported or deferred parameter families must fail honestly instead of broadening the v1 promise silently.
- This phase remains same-runtime deterministic only; it must not claim broader replay guarantees than the plan already allows.

**Execution steps:**

1. Reconfirm the preserved Phase 2 contract, acceptance rule, and focused validation against the current plan text and the Phase 1 coupling note.
2. Decide which phase-agent steps are real value gates for this roundtrip surface and which should be explicit skipped-step packets.
3. Author Phase 2 Step 02-07 packets or explicit skipped-step packets with the narrowest honest network-owned scope.
4. Record any dependency on unresolved Phase 1 ordering decisions directly in the plan before naming the next active step.

**Stop conditions:**

- Done: Phase 2 has a paste-ready next active step and all remaining step slots are authored or explicitly skipped.
- Blocked: Phase 1 ordering or owner-boundary ambiguity prevents an honest vector roundtrip packet.
- Route back: if planning exposes a missing prerequisite or roadmap mismatch, update the tracker and return to planning instead of widening into fine-tuning helpers.

**Required validation:** Manual diff review confirming that every former Step 2 detail now lives in Phase 2 and that the first-tranche coupling note with Phase 1 remains explicit.

**Plan update requirement:** Update this plan with the new Step 02-07 packets, any skipped-step rationale, validation evidence, and the next active step before ending the session.

**Planning decisions and validation evidence:** Step 01 preserved the Phase 2 contract, the same-runtime roundtrip wording, the negative mismatch requirements, and the rule that layout metadata and values must travel together from the network-owned boundary. `plans/README.md` and `plans/Roadmap.md` still place this lane after the archived ONNX baseline and before the dependency-gated NEATchat follow-up, while Plan Scout confirmed that Phase 2 packetization remains safe and that, after Phase 1 Step 07 closure, the next active frontier is now Phase 2 Step 03 while the unchanged ONNX TypeScript failure remains a recorded external baseline outside this seam.

- Value-gate decision: Research, Red Testing, Implementation, Green Validation, Documentation, and Session Logging all add independent value for this roundtrip surface because Phase 2 introduces a new network-owned export/import contract, explicit mismatch rejection, and public docs that should not be folded into Phase 1 or Phase 3.
- Skipped-step decision: no Phase 2 Step 02-07 slot is skipped.
- Coupling note: Phase 2 remains coupled to the opening implementation tranche with Phase 1, and Phase 1 Step 07 now clears that tranche to proceed. The unchanged ONNX TypeScript failure stays recorded as external baseline context, not as a Phase 2 planning blocker unless the failure surface moves into the vector roundtrip seam.
- Roadmap note: the roadmap sequencing still aligns, but the roadmap summary text is slightly stale about this lane's completion state; that wording does not block Phase 2 planning.
- Tracker validation: `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` should be rerun after this tracker edit.

#### Step 02: Research vector roundtrip owner boundary [DONE]

```yaml
phase: 2
step: 2
agent: '02-researching'
agent_file: '.github/agents/02-research-coordinator.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 03 — Design vector roundtrip red tests'
skills:
  - 'hybrid-training-interop'
  - 'plan-alignment'
specialists:
  - 'Hybrid Interop Scout'
validation:
  - 'Read-only research brief names the network owner boundary, the v1 parameter-family inclusion rule, the required mismatch checks, the likely owner-local test file, and any unsupported-family honesty failures needed before red tests.'
```

**User instruction:** Start a fresh session, select `02-researching`, and paste this full step packet.

**Step objective:** Gather the smallest read-only evidence needed to lock the Phase 2 owner boundary, v1 parameter-family scope, and red-test seam for vector export/import without widening into isolation helpers, persistence policy, or downstream consumer formats.

**Context the agent must know:**

- This packet now sits on the active frontier after Phase 1 Step 07 closure; carry the unchanged ONNX TypeScript failure as external baseline context unless the failure surface moves into this seam.
- Phase 2 must reuse the deterministic layout contract from Phase 1 rather than invent a parallel descriptor format.
- Layout version, ordered descriptors, and parameter values must travel together; compatibility must not be inferred from topology similarity alone.
- The minimum honest v1 scope should stay as narrow as possible, typically weights and biases only unless the current public training surface already proves another trainable family belongs in scope.
- Same-runtime ordered determinism remains the replay claim; broader cross-runtime replay language is out of scope.

**Execution steps:**

1. Read `plans/README.md`, `plans/Roadmap.md`, this Phase 2 section, and the nearest relevant `src/architecture/network/**/README.md` surfaces before individual source files.
2. Identify the smallest honest owner boundary for `ParameterVector`, `toParameterVector`, and `fromParameterVector`, reusing the Phase 1 layout types where possible and naming any dedicated module need explicitly.
3. Lock the v1 inclusion rule for parameter slots: which trainable families are included, whether disabled connections occupy slots, whether transient runtime state is excluded, and which unsupported or deferred families must fail explicitly.
4. Name the narrowest owner-local test seam for roundtrip inference equality and mismatch rejection without opening Phase 3 isolation or Phase 4 policy behavior.
5. Record any remaining dependency on unresolved Phase 1 ordering work directly in this plan before handing off to Step 03.

**Stop conditions:**

- Done: research identifies the owner boundary, v1 inclusion or exclusion rule, mismatch surfaces, source or test files, and any activation dependency on unresolved Phase 1 work.
- Blocked: the owner boundary or parameter-family inclusion rule cannot be named honestly from current repo evidence.
- Route back: research shows the vector roundtrip seam cannot remain network-owned or requires Phase 3 or Phase 4 policy work to be meaningful.

**Required validation:** Manual evidence review confirming the research remains read-only, boundary-local, and aligned with the Phase 2 acceptance rule.

**Plan update requirement:** Update this plan with the research brief, any Phase 1 dependency note, the named red-test seam, and the Step 03 handoff before ending the session.

**Research brief and validation evidence:** `plans/README.md` and `plans/Roadmap.md` still align this lane as the next non-chat foundation handoff after the archived ONNX baseline, with no roadmap conflict for a serialize-owned vector roundtrip seam. The nearest relevant README surfaces were `src/architecture/network/README.md`, `src/architecture/network/serialize/README.md`, and `src/architecture/network/training/README.md`; together they keep the public owner at the `Network` boundary, place versioned portability contracts under the serialize shelf, and show the current learnable runtime surface without widening into policy or worker concerns. A focused `Hybrid Interop Scout` read-only packet plus bounded source reads confirmed the Phase 2 boundary below.

- Owner boundary: the smallest honest owner seam is still the network serialize chapter. Define `ParameterVector` beside `ParameterLayoutV1` in `src/architecture/network/serialize/network.serialize.utils.types.ts`, implement `toParameterVector` and `fromParameterVector` beside `createParameterLayoutV1(...)` in `src/architecture/network/serialize/network.serialize.utils.ts`, and expose only thin `Network` delegates adjacent to the existing serialize delegates in `src/architecture/network/network.utils.ts` and `src/architecture/network/network.ts`. That keeps ownership out of training, checkpointing, workers, ONNX, and downstream application code.
- v1 inclusion rule: include exactly one bias slot for every live runtime node and one weight slot for every live runtime forward connection plus self-connection, reusing the existing `ParameterLayoutV1` fold order and identity rules. Because `createParameterLayoutV1(...)` currently iterates `networkInternals.nodes`, `networkInternals.connections`, and `networkInternals.selfconns` without filtering on `enabled`, disabled connections should still occupy vector slots whenever they still exist in the runtime graph. This keeps the vector aligned with the runtime-owned layout contract rather than the topology-only view that excludes disabled edges.
- v1 exclusions: exclude the enabled flag itself, connection gain, node response, gater assignment, innovation or gene-id metadata as parameter values, and all transient runtime or training state such as activation values, recurrent state, traces, dropout or noise masks, optimizer accumulators, mixed-precision state, RNG state, and checkpoint-only state. The current training utilities still accumulate and clip only `totalDeltaWeight` and `totalDeltaBias`, so weights plus biases are the narrowest honest first-pass trainable families.
- Required mismatch checks before mutation: `fromParameterVector` should reject a layout version mismatch, a layout entry-count mismatch, a values-length mismatch, and any ordered descriptor mismatch against a freshly rebuilt target-network `ParameterLayoutV1`. Descriptor compatibility should be full-entry equality on `kind` plus the stable identity fields already defined in `ParameterLayoutEntry`, not a loose topology-shape comparison. Import should also fail before mutation when the target layout cannot be synthesized because a node lacks a stable gene id or the weight identities are ambiguous; those remain Phase 1 invariants rather than partial-apply cases.
- Owner-local red-test seam: `src/architecture/network/serialize/network.serialize.test.ts` is still the narrowest honest file for Step 03. Phase 1 ordering tests already live there, and nearby serialize-owned tests already verify preservation of enabled flags, node responses, and connection gain across clone or restore paths, so the same file can add same-runtime roundtrip inference equality and atomic mismatch rejection without widening into Phase 3 or Phase 4.
- Unsupported-family honesty failures: the main honesty risk is silently dropping non-neutral `node.response` or `connection.gain` even though the strict genome archive path currently treats them as first-class capture families. If Phase 2 v1 omits them, the contract and docs should say so explicitly and the red phase should prefer rejecting non-neutral response or gain over silently flattening them away. A second honesty risk is disabled connections: topology utilities intentionally exclude disabled edges from architecture summaries, but serialize and genetic surfaces still preserve their identity and weight, so the vector contract must not silently fork into an expressed-topology-only slot policy. A smaller but real risk is public expectation drift around input-node biases, because the current layout includes all node biases, not only hidden or output-node biases.

**Phase 1 dependency note:** owner-boundary ambiguity is resolved, and Phase 1 Step 07 has now closed the layout-ordering seam. Phase 2 Step 03 can become the lane's active next session while carrying forward the recorded repo-wide TypeScript blocker as an unchanged external ONNX validation baseline rather than as a Phase 2 owner-boundary dependency.

**Step 03 handoff:** Add the smallest failing owner-local tests in `src/architecture/network/serialize/network.serialize.test.ts` for the serialize-owned vector seam only. Cover two behavior groups: same-runtime export plus import roundtrip preserves inference outputs for the same topology, and invalid payloads fail atomically before mutation for version mismatch, layout-entry-count or values-length mismatch, and ordered descriptor mismatch. Keep the red phase network-owned and Phase 2 scoped: reuse `ParameterLayoutV1`, do not open Phase 3 isolation or Phase 4 policy, and if v1 continues to omit `node.response` or `connection.gain`, prefer an explicit non-neutral rejection test or a blocked honesty note instead of silently broadening the promise.

#### Step 03: Design vector roundtrip red tests [DONE]

```yaml
phase: 2
step: 3
agent: '03-red-testing'
agent_file: '.github/agents/03-red-test-architect.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 04 — Implement parameter-vector roundtrip boundary'
skills:
  - 'hybrid-training-interop'
  - 'red-test-contracts'
  - 'reproducibility-contracts'
specialists:
  - 'Determinism Scout'
validation:
  - 'A focused owner-local red test or explicit blocked record proves whether same-runtime export or import roundtrip and mismatch rejection can fail honestly before implementation.'
```

**User instruction:** Start a fresh session, select `03-red-testing`, and paste this full step packet.

**Step objective:** Add or define the smallest failing test contract for Phase 2 vector export/import roundtrip and mismatch rejection before implementation, while keeping training isolation and persistence policy work out of scope.

**Context the agent must know:**

- The red surface is `toParameterVector` plus `fromParameterVector`, not Phase 3 fine-tuning or Phase 4 policy behavior.
- Required negative cases include version mismatch, entry-count or value-length mismatch, and descriptor mismatch before any target-network mutation.
- Roundtrip acceptance is same-runtime inference equality for the same topology and layout contract, not broad replay across runtimes.
- Unsupported or deferred parameter-family failures should be red-tested only when Step 02 identifies an honest current failure path.
- Tests must follow repo conventions, including owner-local placement and one top-level `expect(...)` per test.

**Execution steps:**

1. Read the Step 02 research brief, the target source boundary, and the nearest existing owner-local tests.
2. Add the smallest failing tests for vector roundtrip inference equality and the required mismatch-rejection cases.
3. Keep the tests network-owned and Phase 2 scoped; do not pull in Phase 1-only descriptor assertions, Phase 3 isolation helpers, or Phase 4 persistence policy paths.
4. Run the narrowest practical Jest command for the touched owner-local test file, or record precisely why the red command is blocked.
5. Update this plan with the failing command, expected failure, and the exact Step 04 green condition before ending the session.

**Stop conditions:**

- Done: focused red tests fail for the missing vector roundtrip contract, or a precise blocked record proves that Phase 2 cannot be isolated honestly yet.
- Blocked: Step 02 leaves the owner-local test boundary or observable behavior ambiguous.
- Route back: the red design requires Phase 3 isolation, Phase 4 policy semantics, or unresolved Phase 1 behavior beyond the recorded coupling note.

**Required validation:** Focused red-test command evidence, or a manual blocked record that explains why no honest Phase 2-only red test exists yet.

**Plan update requirement:** Update this plan with changed test files or blocked rationale, red evidence, and the Step 04 implementation handoff before ending the session.

**Step 03 red contract and validation evidence:** Added owner-local red tests in `src/architecture/network/serialize/network.serialize.test.ts` that import the planned `toParameterVector` and `fromParameterVector` helpers from the serialize boundary and lock two Phase 2 behavior groups only: same-runtime export or import roundtrip restores inference outputs for a compatible same-topology clone, and invalid payloads fail atomically before mutation for version mismatch, layout entry-count mismatch, values-length mismatch, and ordered descriptor mismatch. The roundtrip fixture stays inside the current bias-and-weight v1 scope by using neutral `node.response` and `connection.gain`; Step 04 must still preserve the honesty rule that any omitted non-neutral response or gain path is rejected explicitly rather than flattened silently if implementation reaches it.

- Changed test file: `src/architecture/network/serialize/network.serialize.test.ts`
- Focused command: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --testNamePattern="ParameterVector"`
- Observed red failure: `TS2305: Module "./network.serialize.utils" has no exported member "fromParameterVector".` and `TS2305: Module "./network.serialize.utils" has no exported member "toParameterVector".`
- Scope guard: Step 03 stays network-owned and Phase 2 scoped; it does not reopen Phase 1 ordering work or introduce Phase 3 isolation or Phase 4 policy behavior.

**Step 04 green condition:** Export the smallest network-owned `ParameterVector`, `toParameterVector`, and `fromParameterVector` surface from the serialize boundary so the focused `ParameterVector` Jest slice goes green. Reuse `ParameterLayoutV1`, keep ordered layout metadata and values together under one versioned payload, preserve same-runtime roundtrip inference for compatible same-topology clones, reject version mismatch, layout entry-count mismatch, values-length mismatch, and ordered descriptor mismatch before mutation, and do not silently flatten non-neutral `node.response` or `connection.gain` if that unsupported-family path is reached.

#### Step 04: Implement parameter-vector roundtrip boundary [DONE]

```yaml
phase: 2
step: 4
agent: '04-implementing'
agent_file: '.github/agents/04-implementation-architect.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 05 — Validate vector roundtrip gates'
skills:
  - 'hybrid-training-interop'
  - 'reproducibility-contracts'
specialists:
  - 'Hybrid Interop Scout'
validation:
  - 'The Step 03 red test goes green with a network-owned parameter-vector export or import implementation and no Phase 3 or Phase 4 behavior.'
```

**User instruction:** Start a fresh session, select `04-implementing`, and paste this full step packet.

**Step objective:** Implement the smallest network-owned `ParameterVector` export/import surface that satisfies the Phase 2 red tests while keeping the contract honest about supported v1 parameter families and same-runtime determinism only.

**Context the agent must know:**

- Exported payloads must keep ordered layout metadata and values together under one versioned network-owned surface.
- `fromParameterVector` must validate version compatibility, entry-count compatibility, and descriptor compatibility before mutating the target network.
- The v1 contract must fail explicitly for unsupported or deferred parameter families instead of silently widening the promise.
- Mutation should be atomic for the active target boundary: do not partially apply a vector before all compatibility checks pass.
- This step must not introduce Phase 3 isolation helpers, Phase 4 persistence policy hooks, or broader replay guarantees.

**Execution steps:**

1. Read the Phase 2 plan text, the Step 03 red evidence, the nearest folder README, and the named source or test files.
2. Add the minimal network-owned `ParameterVector` surface plus `toParameterVector` and `fromParameterVector`, reusing the Phase 1 layout descriptors instead of cloning a competing metadata format.
3. Implement all compatibility checks ahead of mutation and make the v1 inclusion rule explicit in code and types.
4. Reject unsupported or deferred parameter families explicitly instead of dropping them silently or broadening the contract.
5. Add concise JSDoc for exported vector types and public helpers introduced by this step.
6. Rerun the focused Step 03 command and update this plan with changed files, compatibility decisions, remaining risks, and the Step 05 validation handoff.

**Stop conditions:**

- Done: the focused red tests are green and the implementation remains network-owned and Phase 2 scoped.
- Blocked: stable parameter-family ownership or compatibility semantics cannot be implemented without changing upstream contracts first.
- Route back: implementation reveals the roundtrip seam cannot remain honest without reopening Step 02 research or Step 03 red design.

**Required validation:** The focused Step 03 red-test command passes after the implementation change.

**Plan update requirement:** Update this plan with implementation files, focused validation evidence, any v1 inclusion or rejection decisions, and the Step 05 handoff before ending the session.

**Step 04 implementation and validation evidence:** Added the smallest network-owned `ParameterVector` surface at the serialize boundary by exporting `ParameterVector` from `src/architecture/network/serialize/network.serialize.utils.types.ts` and `toParameterVector(network)` plus `fromParameterVector(network, parameterVector)` from `src/architecture/network/serialize/network.serialize.utils.ts`. The implementation reuses `ParameterLayoutV1` instead of inventing a parallel descriptor format, keeps ordered layout metadata and aligned `Float64Array` values together under one payload, rebuilds the target layout fresh during import, and performs version, layout entry-count, values-length, and ordered descriptor checks before any bias or weight mutation occurs.

- Changed implementation files: `src/architecture/network/serialize/network.serialize.utils.ts`, `src/architecture/network/serialize/network.serialize.utils.types.ts`
- Focused command: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --testNamePattern="ParameterVector"`
- Observed green result: `PASS` for `src/architecture/network/serialize/network.serialize.test.ts`; `1` suite passed and all `5` targeted `ParameterVector v1 roundtrip` tests passed for same-topology inference restoration plus version, entry-count, values-length, and ordered descriptor mismatch rejection before mutation.
- v1 inclusion decision: the payload carries exactly the existing layout-owned bias slots for live nodes and weight slots for live forward connections plus self-connections, including disabled connections when they still exist in the runtime graph, because the contract reuses `ParameterLayoutV1` directly.
- v1 rejection decision: export and import now reject any unsupported non-neutral `node.response` or `connection.gain` path explicitly before emitting or applying a vector instead of flattening those omitted families silently.
- Remaining risks: Step 04 proves only the focused serialize seam. Step 05 still needs to rerun the focused `ParameterVector` Jest slice, run `npx tsc --noEmit -p tsconfig.json` while treating the unchanged ONNX `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` as external unless the failure moves into this seam, and run coverage guard for the touched `src/architecture/network/serialize/network.serialize.utils.ts` and `src/architecture/network/serialize/network.serialize.utils.types.ts` files.

**Step 05 validation handoff:** Treat `src/architecture/network/serialize/network.serialize.utils.ts` and `src/architecture/network/serialize/network.serialize.utils.types.ts` as the touched source boundary. Re-run the focused `ParameterVector` Jest slice first, then run `npx tsc --noEmit -p tsconfig.json`, then run coverage guard for both touched `src/` files. If the focused slice fails inside same-runtime roundtrip or mismatch rejection, route back to Step 04. If coverage is missing for the new export/import guard rails, route back to Step 03 only for the smallest owner-local test additions. Keep the ONNX `TS2345` baseline recorded as external unless a new failure appears inside the parameter-vector seam.

#### Step 05: Validate vector roundtrip gates [DONE]

```yaml
phase: 2
step: 5
agent: '05-green-testing'
agent_file: '.github/agents/05-green-validation-runner.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 06 — Curate vector-contract docs'
skills:
  - 'green-validation-gates'
  - 'coverage-guard'
  - 'hybrid-training-interop'
specialists:
  - 'Coverage Guard'
  - 'Determinism Scout'
validation:
  - 'Focused vector roundtrip and mismatch tests pass, TypeScript validation passes or a pre-existing blocked baseline is recorded explicitly, and coverage guard passes for every touched src file.'
```

**User instruction:** Start a fresh session, select `05-green-testing`, and paste this full step packet.

**Step objective:** Prove the Phase 2 implementation satisfies the same-runtime roundtrip and mismatch-rejection contract with focused validation before documentation or phase closure.

**Context the agent must know:**

- Validate only the Phase 2 vector export/import boundary and directly touched files.
- The primary behavior gates are roundtrip inference equality and explicit mismatch rejection before mutation.
- If `src/` files changed, coverage guard is required for every touched source file.
- If repo-wide TypeScript still fails because of an unrelated baseline outside the Phase 2 seam, record that blocker precisely instead of widening this step into unrelated fixes.
- Failed validation routes back to the smallest prior step that can repair the active seam.

**Execution steps:**

1. Read the Step 04 changed-file summary and expected validation commands.
2. Rerun the focused vector roundtrip or mismatch Jest slice.
3. Run `npx tsc --noEmit -p tsconfig.json` after the focused slice is green.
4. Run coverage guard for every touched `src/` file, or record why no `src/` coverage gate applies.
5. Update this plan with pass or fail evidence and reroute failures to Step 03 or Step 04 as appropriate.

**Stop conditions:**

- Done: focused tests, TypeScript validation, and required coverage gates pass.
- Blocked: a validation command cannot run, or a pre-existing unrelated TypeScript baseline still blocks repo-wide validation.
- Route back: behavior failures return to Step 04; missing or insufficient owner-local coverage returns to Step 03.

**Required validation:** Command evidence for the focused test slice, `npx tsc --noEmit -p tsconfig.json`, and coverage guard on touched `src/` files when applicable.

**Plan update requirement:** Update this plan with validation evidence, reroute decisions if any, and the Step 06 documentation handoff before ending the session.

**Step 05 validation evidence and follow-up:** Re-ran the focused vector roundtrip slice with `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --testNamePattern="ParameterVector"`. The Phase 2 behavior gate stayed green: `PASS` for `src/architecture/network/serialize/network.serialize.test.ts`, `1` suite passed, `5` targeted `ParameterVector v1 roundtrip` tests passed, and same-runtime roundtrip plus atomic mismatch rejection did not reroute to Step 04 for a behavior defect.

Repo-wide TypeScript still does not clear. `npx tsc --noEmit -p tsconfig.json` fails only at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` with the unchanged external `TS2345` because `standardDomainImports[0]!.version` can be `undefined`. No new failure moved into the serialize-owned parameter-vector seam, so that ONNX result remains a recorded baseline blocker rather than a Phase 2 reroute.

A first coverage probe with `--testNamePattern="ParameterVector"` was too narrow to serve as the real guard because it suppressed owner-local coverage from the rest of `src/architecture/network/serialize/network.serialize.test.ts`. The actual owner-local guard rerun used `npx jest --config=jest.config.mjs --no-cache --coverage --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --collectCoverageFrom="src/architecture/network/serialize/network.serialize.utils.ts" --collectCoverageFrom="src/architecture/network/serialize/network.serialize.utils.types.ts"`. `src/architecture/network/serialize/network.serialize.utils.types.ts` stayed at `100/100/100/100`, but `src/architecture/network/serialize/network.serialize.utils.ts` remains below the gate at `98.43` statements, `94.56` branches, `100` functions, and `98.39` lines with uncovered lines `397`, `406`, `468`, and `486`.

Those remaining misses were mixed rather than a pure Step 03 test gap. Lines `397` and `406` were the explicit unsupported-family rejections for non-neutral `node.response` and `connection.gain`, while lines `468` and `486` were the missing bias-binding and missing weight-binding throws that appeared unreachable because `createParameterRuntimeContext(network)` rebuilds both the layout and the binding maps from the same live runtime before export or import.

**Step 05 completion from current repo state:** The Step 04 follow-up removed the unreachable runtime-binding throws in `src/architecture/network/serialize/network.serialize.utils.ts`, and the Step 03 follow-up added the smallest owner-local tests in `src/architecture/network/serialize/network.serialize.test.ts` for non-neutral `node.response`, non-neutral `connection.gain`, and fallback descriptor export when a weight lacks an innovation id.

- Changed files during the reroute closure: `src/architecture/network/serialize/network.serialize.utils.ts`, `src/architecture/network/serialize/network.serialize.test.ts`
- Focused coverage guard rerun: `npx jest --config=jest.config.mjs --no-cache --coverage --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --collectCoverageFrom="src/architecture/network/serialize/network.serialize.utils.ts" --collectCoverageFrom="src/architecture/network/serialize/network.serialize.utils.types.ts"`
- Coverage result: `PASS`; `1` suite passed, `122` tests passed, and both `src/architecture/network/serialize/network.serialize.utils.ts` and `src/architecture/network/serialize/network.serialize.utils.types.ts` are now at `100/100/100/100`.
- Focused behavior rerun: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --testNamePattern="ParameterVector"`
- Focused behavior result: `PASS`; `1` suite passed and all `8` targeted `ParameterVector` tests are green, including the new unsupported-family and fallback-descriptor cases.
- Repo TypeScript rerun: `npx tsc --noEmit -p tsconfig.json`
- TypeScript result: still fails only at the unchanged external ONNX baseline `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` with `TS2345` because `standardDomainImports[0]!.version` can be `undefined`.

**Updated Step 06 handoff:** Call `06-documenting` next. Keep the next pass source-doc-only and scoped to the serialize-owned vector surface: review `ParameterVector`, `toParameterVector`, and `fromParameterVector` JSDoc plus the generated `src/architecture/network/serialize/README.md` output for alignment with the now-final v1 contract, including the weights-and-biases inclusion rule, explicit unsupported-family rejection for non-neutral `node.response` and `connection.gain`, disabled-connection slot behavior, fallback descriptor semantics when an innovation id is absent, and same-runtime determinism wording. Run `npm run docs` only if source JSDoc changes are required. Do not widen into Phase 3 isolation, Phase 4 policy, or unrelated ONNX work.

#### Step 06: Curate vector-contract docs [DONE]

```yaml
phase: 2
step: 6
agent: '06-documenting'
agent_file: '.github/agents/06-educational-docs-curator.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 07 — Log Phase 2 closure or reroute'
skills:
  - 'educational-docs'
  - 'docs-academic-citation-audit'
  - 'hybrid-training-interop'
specialists:
  - 'Docs Scout'
validation:
  - 'Source-first docs for any exported vector symbols are present, generated docs are refreshed if JSDoc inputs changed, and no Phase 3 or Phase 4 docs are introduced accidentally.'
```

**User instruction:** Start a fresh session, select `06-documenting`, and paste this full step packet.

**Step objective:** Keep the new vector export/import surface teachable and synchronized with generated docs without turning Phase 2 into the broader Phase 5 documentation closure.

**Context the agent must know:**

- Documentation scope is limited to `ParameterVector`, `toParameterVector`, `fromParameterVector`, the v1 inclusion rule, mismatch rejection, and same-runtime determinism wording.
- Phase 3 isolation, Phase 4 policy, and Phase 5 examples remain out of scope.
- Generated `src/**/README.md` files and `docs/examples/**` outputs are read-only; update source JSDoc and run generation when needed.
- Public docs should not mention tracker phases, roadmap chronology, or chat-only context.

**Execution steps:**

1. Read the Step 04 changed public surfaces and the Step 05 validation evidence.
2. Improve only the source-owned JSDoc or nearby handwritten docs required by the new vector export/import surface.
3. Run `npm run docs` if JSDoc or generated-doc inputs changed; otherwise record a manual no-docs-needed rationale.
4. Verify that no generated README or published example output was hand-edited as a shortcut.
5. Update this plan with documentation evidence and any residual docs risk before ending the session.

**Stop conditions:**

- Done: required source docs are aligned and docs generation either passes or is explicitly not needed.
- Blocked: public naming or determinism wording is unstable enough that docs would mislead downstream consumers.
- Route back: documentation reveals implementation drift in the vector contract; route to Step 04 or Step 05 rather than widening docs.

**Required validation:** `npm run docs` when source JSDoc or generated-doc inputs changed, or a manual no-docs-needed record when implementation introduced no doc-facing surface.

**Plan update requirement:** Update this plan with documentation changes, validation evidence, residual gaps, and the Step 07 closure handoff before ending the session.

**Step 06 documentation changes and validation evidence:** Reviewed the source-owned JSDoc for `ParameterVector`, `toParameterVector`, and `fromParameterVector` together with the generated `src/architecture/network/serialize/README.md` output. The source docs needed one narrow Phase 2-only clarification pass, so this step updated only the serialize source-of-truth JSDoc and then regenerated docs. The refreshed README now includes all three vector-contract sections and matches the final v1 contract without widening into Phase 3 or Phase 4.

- Changed source docs: `src/architecture/network/serialize/network.serialize.utils.types.ts`, `src/architecture/network/serialize/network.serialize.utils.ts`
- Validation command: `npm run docs`
- Observed result: the docs build completed successfully, refreshed `src/architecture/network/serialize/README.md`, and now exposes `ParameterVector`, `toParameterVector`, and `fromParameterVector` with the expected vector-contract wording.
- Contract coverage confirmed: the docs now state the weights-and-biases inclusion rule, explicit rejection of non-neutral `node.response` and `connection.gain`, disabled-connection slot behavior, fallback descriptor semantics when an innovation id is absent, and same-runtime deterministic wording without claiming cross-runtime exact replay.
- Tracker validation: `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` passed with `0` errors and `0` warnings after this Step 06 tracker refresh.
- Packet-validator note: `node scripts/agent-customization/validate-plan-phase-packets.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` still reports `No implementation phase packets found.` and `Expected exactly one [WIP] phase, found 0.` This broader tracker-format mismatch was already failing from current repo state before this source-doc-only pass, so Step 06 did not widen into repairing it.
- Source-of-truth guard: no generated `src/**/README.md` file or `docs/examples/**` output was hand-edited as a shortcut.
- Citation note: no external citation or license update was required because this pass only documents a repo-owned serialize contract.
- Residual docs risk: none within the Phase 2 vector seam; Step 07 should decide whether Phase 2 can close against the unchanged external ONNX TypeScript baseline.
- Exact next orchestrator: `07-logging`
- Exact handoff prompt: see `## Handoff query` below.

#### Step 07: Log Phase 2 closure or reroute [DONE]

```yaml
phase: 2
step: 7
agent: '07-logging'
agent_file: '.github/agents/07-session-log-archivist.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Phase 3 Step 01 — Plan the isolation-helper tranche'
skills:
  - 'tracker-handoff'
  - 'plan-sync-validation'
  - 'hybrid-training-interop'
specialists:
  - 'Plan Registration Auditor'
validation:
  - 'Manual tracker review confirms Phase 2 acceptance evidence, the Phase 1 coupling state, and the refreshed Handoff query before the next phase opens.'
```

**User instruction:** Start a fresh session, select `07-logging`, and paste this full step packet.

**Step objective:** Close or reroute Phase 2 based on validated evidence, preserve the Phase 1 coupling state honestly, and leave the tracker ready either for Phase 3 planning or for the smallest blocking reroute.

**Context the agent must know:**

- Phase 2 closes only when roundtrip preserves inference outputs on the same runtime and invalid vector or layout payloads fail explicitly before mutation.
- Phase 3 should not open until both Phase 1 and Phase 2 are green enough to support a real parameter-vector contract.
- The overall lane remains active; do not archive this tracker unless the full hybrid-interoperability workstream is done.
- Chat is not the source of truth; the tracker and the `Handoff query` must carry the next session.

**Execution steps:**

1. Read the Phase 2 Step 02-06 evidence and verify that no required validation is missing.
2. Mark completed Phase 2 steps `[DONE]` or record the smallest reroute if evidence is incomplete.
3. Record that the Phase 1 coupling gate is already cleared, and whether Phase 3 Step 01 can become the next safe step.
4. Refresh the `Handoff query` to point either to Phase 3 Step 01 or to the smallest blocking reroute step instead of implying Phase 2 is complete when it is not.
5. Run or record focused tracker validation appropriate to the changed plan surface.

**Stop conditions:**

- Done: Phase 2 evidence is compactly recorded, the next safe step is explicit, and the handoff query is current.
- Blocked: validation, documentation, or ownership evidence is missing and cannot be reconstructed from the tracker.
- Route back: missing or inadequate red coverage routes to Step 03, implementation drift routes to Step 04, failed validation routes to Step 05, and doc drift routes to Step 06.

**Required validation:** Manual tracker diff review confirming Phase 2 acceptance evidence, no premature Phase 3 behavior, and a current next-step handoff.

**Plan update requirement:** Update this plan with Phase 2 closure or reroute state, refreshed handoff query, and the next active step before ending the session.

**Step 07 closure review and handoff evidence:** Step 07 reviewed the recorded Phase 2 evidence from Step 02 through Step 06 from current repo state only. Phase 2 can now close. The serialize-owned vector contract satisfies the phase acceptance gate: same-runtime export plus import preserves inference outputs for compatible same-topology networks, invalid vector or layout payloads fail explicitly before mutation, owner-local coverage for the touched serialize source files is back at `100/100/100/100`, and the source-doc-only pass refreshed the generated serialize README to match the final v1 contract.

- Phase 2 closure decision: close Phase 2. The focused `ParameterVector` behavior slice passed with `1` suite and all `8` targeted vector-contract tests green, and the owner-local coverage guard reports `100/100/100/100` for both `src/architecture/network/serialize/network.serialize.utils.ts` and `src/architecture/network/serialize/network.serialize.utils.types.ts`.
- External baseline decision: do not keep Phase 2 open solely because `npx tsc --noEmit -p tsconfig.json` still fails at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` with `TS2345` because `standardDomainImports[0]!.version` can be `undefined`. No new TypeScript failure moved into the serialize-owned vector seam, so that result remains a recorded external ONNX baseline rather than a Phase 2 reroute.
- Coupling decision: record that the Phase 1 coupling gate is cleared. Phase 1 and Phase 2 now stand as the completed opening implementation tranche, which is sufficient to open Phase 3 planning without widening into implementation.
- Tracker state decision: keep the top-level tracker active in `plans/` as `[WIP]`; mark Phase 2 complete; move the active frontier to Phase 3 Step 01; do not archive this tracker or create a same-boundary `.logs.md` file because the hybrid-interoperability lane still has real next phases.
- Next safe step: activate `01-planning` for Phase 3 `Step 01 — Plan the isolation-helper tranche`. Do not widen into Phase 3 implementation, Phase 4 policy work, or unrelated ONNX fixes.
- Tracker validation: rerun `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` after this tracker refresh.

### Phase 3 — Add training isolation helpers [DONE]

**Phase objective:** Define one minimal isolated fine-tuning seam that can improve candidates without mutating shared population state by accident.

**Phase progression rule:** Start with only Step 01. Step 01 must author the remaining numbered step packets, or explicit skipped-step packets, before the phase can advance.

**Preserved implementation contract:**

- Provide one minimal isolation primitive first.
- Prefer a low-level vector-in and trained-vector-out helper because it is easier to keep deterministic across worker, checkpoint, and evaluation surfaces.
- A clone-based ergonomic wrapper can follow as a thin convenience layer once the vector-first path is stable.
- Require dataset ordering and training settings to be explicit inputs to the helper.
- Any deterministic claim at this step must require explicit seed handling and a documented RNG owner.
- Avoid helpers that silently mutate the supplied network or vector unless the mutating behavior is stated directly in the API name.

**Acceptance when this phase closes:**

- Fine-tuning cannot mutate shared population state accidentally.

**Focused validation to preserve:**

- Fine-tuning leaves the original candidate unchanged.
- Same seed plus same dataset order yields stable fine-tuned output on the same runtime.

#### Step 01: Plan the isolation-helper tranche [DONE]

```yaml
phase: 3
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning-architect.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 02 — Research isolation-helper boundary'
skills:
  - 'hybrid-training-interop'
specialists:
  - 'Hybrid Interop Scout'
validation:
  - 'Manual tracker check that the isolation rule, vector-first preference, and deterministic-claim guardrails are preserved before Step 02-07 packets are authored.'
```

**User instruction:** Start a fresh session, select `01-planning`, and paste this full step packet.

**Step objective:** Turn Phase 3 into an executable isolation-helper workset by authoring Phase 3 Step 02-07 packets, keeping the first pass vector-first, and preserving the no-shared-mutation contract.

**Context the agent must know:**

- Phase 3 should not open until Phase 1 and Phase 2 are green enough to support a real parameter-vector contract.
- The first isolation primitive should be low-level and explicit about whether it mutates or returns a detached result.
- Deterministic claims at this phase require explicit seed handling, dataset order, and a named RNG owner.
- Worker, checkpoint, and evaluation consumers are downstream of this contract, so the phase must stay detached from application-specific glue.

**Execution steps:**

1. Reconfirm the preserved Phase 3 contract, acceptance rule, and focused validation against the current plan text.
2. Decide which phase-agent steps are real value gates for proving isolation and which should be explicit skipped-step packets.
3. Author Phase 3 Step 02-07 packets or explicit skipped-step packets with a narrow vector-first isolation boundary.
4. Record any dependency on unresolved Phase 1 or Phase 2 work directly in the plan before naming the next active step.

**Stop conditions:**

- Done: Phase 3 has a paste-ready next active step and all remaining step slots are authored or explicitly skipped.
- Blocked: unresolved parameter-vector or determinism ownership prevents an honest isolation-helper packet.
- Route back: if planning shows the isolation boundary is still underspecified, update the tracker and return to planning instead of widening into hybrid policy integration.

**Required validation:** Manual diff review confirming that every former Step 3 detail now lives in Phase 3 and that the vector-first, no-shared-mutation contract remains explicit.

**Plan update requirement:** Update this plan with the new Step 02-07 packets, any skipped-step rationale, validation evidence, and the next active step before ending the session.

**Planning decisions and validation evidence:** Step 01 preserved the Phase 3 contract, acceptance rule, focused validation, and the no-shared-mutation contract from the current plan text. Phase 1 and Phase 2 are both [DONE] and supply the real parameter-vector contract that Phase 3 isolation requires; the Phase 1-to-Phase 2 coupling gate is already cleared. The unchanged ONNX TypeScript failure at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` (`TS2345`) is still the recorded external baseline and is not a Phase 3 planning blocker. A brief source scan confirmed that the training seam already exposes `TrainingSample`, `trainSetCore`, and a rich `src/architecture/network/training/` boundary that Phase 3 research should map before red tests begin. Research, Red Testing, Implementation, Green Validation, Documentation, and Session Logging all add independent value because Phase 3 introduces a new behavioral isolation contract; no Step 02-07 slot is skipped.

**Next active step:** Phase 3 Step 02 — Research isolation-helper boundary.

#### Step 02: Research isolation-helper boundary [DONE]

```yaml
phase: 3
step: 2
agent: '02-researching'
agent_file: '.github/agents/02-research-coordinator.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 03 — Design isolation-helper red tests'
skills:
  - 'hybrid-training-interop'
  - 'plan-alignment'
specialists:
  - 'Hybrid Interop Scout'
validation:
  - 'Read-only research brief names the training-seam owner boundary, the isolation helper signature, the dataset and seed type, the likely owner-local test file, and any clone-path or training-loop dependency that Step 03 must be aware of.'
```

**User instruction:** Start a fresh session, select `02-researching`, and paste this full step packet.

**Step objective:** Gather the smallest read-only evidence needed to design an honest isolation-helper red test, specifically the training-seam owner boundary, the proposed `fineTuneVector` signature, the existing `TrainingSample` and training-loop surface, and the narrowest owner-local test seam.

**Context the agent must know:**

- Phase 3 must provide a low-level vector-in and trained-vector-out helper first; a clone-based wrapper can follow only after the vector-first path is stable.
- The helper must not mutate the supplied `ParameterVector` or any `Network` that is part of shared population state; the isolation contract is the entire point of this phase.
- Any deterministic claim requires explicit seed handling, dataset order, and a named RNG owner; do not widen the determinism language beyond same-runtime ordered determinism on the same topology.
- The training seam (`src/architecture/network/training/`) is the preferred owner boundary because this is a training operation, not a portability or checkpoint operation.
- Carry the unchanged ONNX `TS2345` baseline as external context unless the failure surface moves into this seam.

**Execution steps:**

1. Read `plans/README.md`, `plans/Roadmap.md`, this Phase 3 section, and the nearest relevant `src/architecture/network/training/README.md` and `src/architecture/network/serialize/README.md` before individual source files.
2. Delegate a focused read-only packet to `Hybrid Interop Scout` to map the existing training-loop surface (`TrainingSample`, `trainSetCore`, clone paths, seed or RNG exposure) and identify whether a `fineTuneVector` helper can reuse them without widening the isolation boundary.
3. Lock the proposed helper signature: whether it requires a base `Network` for topology in addition to the `ParameterVector`, what the `TrainingDataset` or `TrainingSample[]` input shape is, and how `FineTuneOptions` maps to training-loop knobs.
4. Name the narrowest owner-local test seam for the isolation contract (original vector unchanged) and the determinism contract (same seed plus same dataset order yields same fine-tuned output).
5. Record any dependency on unresolved Phase 2 vector semantics directly in this plan before handing off to Step 03.

**Stop conditions:**

- Done: research identifies the training-seam owner boundary, proposed signature, dataset shape, test seam, and any unresolved clone-path or RNG dependency.
- Blocked: the training-loop surface does not expose enough for an honest isolation helper without changing upstream contracts.
- Route back: research shows the isolation seam cannot remain in the training boundary or requires Phase 4 policy work to be meaningful.

**Required validation:** Manual evidence review confirming the research remains read-only, boundary-local, and aligned with the Phase 3 acceptance rule.

**Plan update requirement:** Update this plan with the research brief, the proposed signature, the named test seam, and the Step 03 handoff before ending the session.

**Research brief and validation evidence:** `plans/README.md` and `plans/Roadmap.md` still align this lane as the last non-chat foundation handoff before the dependency-gated NEATchat follow-up. Phase 2 is fully closed with owner-local coverage at `100/100/100/100` for the serialize-seam files. The nearest relevant README surfaces were `src/architecture/network/training/README.md` and `src/architecture/network/serialize/README.md`; together they confirm the training seam is the correct new owner and the serialize seam stays as the vector-portability supplier consumed by the isolation helper.

- **Owner boundary:** `src/architecture/network/training/` — new implementation file `network.training.isolate.utils.ts`, following the existing training naming convention (alongside `network.training.loop.utils.ts`, `network.training.finalize.utils.ts`, etc.). Types that grow beyond two or three entries should go in a matching `network.training.isolate.utils.types.ts`.
- **Proposed `fineTuneVector` signature:**

  ```ts
  export interface FineTuneOptions {
    steps: number; // Maps to TrainingOptions.iterations
    learningRate: number; // Maps to TrainingOptions.rate
    seed?: number; // Optional: called via workingCopy.setSeed(seed) before training
  }

  export interface FineTuneResult {
    trainedVector: ParameterVector;
    metrics?: Record<string, number>;
  }

  export function fineTuneVector(
    baseNetwork: Network, // Required: topology source for clone; never mutated
    vector: ParameterVector, // Required: applied to clone; original never mutated
    dataset: TrainingSample[], // Required: ordered explicitly by the caller
    options: FineTuneOptions, // Required: explicit training settings
  ): FineTuneResult;
  ```

  `baseNetwork` is required because a `ParameterVector` alone does not carry topology; the isolation path must call `baseNetwork.clone()` to get a temporary working copy before applying the input vector.

- **`TrainingSample` type:** `{ input: number[]; output: number[] }`. It is defined in `src/architecture/network/training/network.training.utils.types.ts` (training-local alias) and matches the public shape in `src/architecture/network/network.types.ts`. The helper should import from the training-local types file to stay in-boundary.
- **Training-loop reuse:** The internal clone path is: `const workingCopy = baseNetwork.clone(); fromParameterVector(workingCopy, vector); if (options.seed !== undefined) workingCopy.setSeed(options.seed); trainImpl(workingCopy, dataset, { iterations: options.steps, rate: options.learningRate }); return { trainedVector: toParameterVector(workingCopy) }`. `fromParameterVector` and `toParameterVector` are imported from the serialize seam; `trainImpl` is imported from `network.training.utils.ts`; `Network.clone()` is already available on the `Network` class.
- **Seed and RNG owner:** `Network.setSeed(seed: number)` delegates to `src/architecture/network/deterministic/network.deterministic.utils.ts`, which installs a reproducible random stream. This is the named RNG owner. Without `options.seed`, the training pass is best-effort reproducible (same-runtime ordered determinism still holds for non-stochastic topology when dropout is absent).
- **Shuffle or random in training loop:** `trainFinalizeCore` (which backs `trainImpl`) does **not** shuffle the dataset. The training pass iterates samples in the order given, so same-runtime ordered determinism is achievable without custom shuffling when the caller controls dataset order explicitly.
- **Owner-local test seam:** New file `src/architecture/network/training/network.training.isolate.utils.test.ts`, following the same naming pattern as `network.training.advanced.test.ts`, `network.training.basic.test.ts`, and `network.training.dropout.test.ts`. Two test groups cover the two Phase 3 observables: the isolation contract (original `ParameterVector` and `baseNetwork` unchanged after the helper runs) and the same-seed determinism contract (two `fineTuneVector` calls with the same seed plus same dataset order yield the same `trainedVector` values).
- **Phase 2 dependency:** None unresolved. `ParameterVector`, `toParameterVector`, and `fromParameterVector` are stable in the serialize seam. `Network.clone()` is already available. No upstream contract changes are required.
- **ONNX `TS2345` baseline:** Unchanged — still at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135`. Not a Phase 3 blocker.

**Step 03 handoff:** Add the smallest failing owner-local tests in `src/architecture/network/training/network.training.isolate.utils.test.ts` for the isolation-helper seam only. Cover two test groups: (1) the isolation contract — the original `ParameterVector` array contents and the `baseNetwork` weights are each bitwise unchanged after `fineTuneVector` returns, and (2) the same-seed determinism contract — two `fineTuneVector` calls with the same `baseNetwork`, same `vector`, same `dataset` order, same `steps`, same `learningRate`, and same explicit `seed` yield `trainedVector` values that are element-wise equal. Keep Step 03 out of Phase 4 policy hooks, Lamarckian persistence, and Phase 5 example constructs.

#### Step 03: Design isolation-helper red tests [DONE]

```yaml
phase: 3
step: 3
agent: '03-red-testing'
agent_file: '.github/agents/03-red-test-architect.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 04 — Implement fineTuneVector isolation helper'
skills:
  - 'hybrid-training-interop'
  - 'red-test-contracts'
  - 'reproducibility-contracts'
specialists:
  - 'Determinism Scout'
validation:
  - 'A focused owner-local red test or explicit blocked record proves that the isolation contract (original unchanged) and the same-seed determinism contract can be tested before implementation.'
```

**User instruction:** Start a fresh session, select `03-red-testing`, and paste this full step packet.

**Step objective:** Add the smallest failing test contract for Phase 3 isolation before implementation: one test that proves the original `ParameterVector` is unchanged after `fineTuneVector` runs, and one test that proves same seed plus same dataset order yields stable fine-tuned output on the same runtime.

**Context the agent must know:**

- The red surface is the isolation and determinism contract of `fineTuneVector` only; do not pull in Phase 4 policy behavior or Phase 5 examples.
- The original `ParameterVector` (and any supplied `Network`) must remain unchanged after the helper runs.
- Determinism tests require explicit seed and dataset-order inputs; do not claim exact cross-runtime replay.
- Tests must follow repo conventions: owner-local placement, one top-level `expect(...)` per test, AAA structure, nested `describe` blocks.
- Carry the Step 02 research brief as the primary evidence for owner boundary, signature, and test seam.

**Execution steps:**

1. Read the Step 02 research brief, the named test seam, and the nearest existing owner-local training tests.
2. Add the smallest failing tests for the isolation contract (original unchanged) and the same-seed determinism contract (stable fine-tuned output).
3. Ensure the tests do not require Phase 4 policy hooks, Lamarckian persistence, or Phase 5 example constructs.
4. Run the narrowest practical Jest command for the touched owner-local test file, or record precisely why the red command is blocked.
5. Update this plan with the failing command, expected failure, and the exact Step 04 green condition before ending the session.

**Stop conditions:**

- Done: focused red tests fail for the missing isolation-helper contract, or a precise blocked record proves Phase 3 cannot be isolated honestly yet.
- Blocked: Step 02 leaves the owner-local test boundary or observable behavior ambiguous.
- Route back: the red design requires Phase 4 policy semantics or unresolved Phase 2 vector behavior.

**Required validation:** Focused red-test command evidence, or a manual blocked record that explains why no honest Phase 3-only red test exists yet.

**Plan update requirement:** Update this plan with changed test files or blocked rationale, red evidence, and the Step 04 implementation handoff before ending the session.

**Step 03 red contract and validation evidence:** Added owner-local red tests in `src/architecture/network/training/network.training.isolate.utils.test.ts` that import the planned `fineTuneVector` helper from the training boundary and lock two Phase 3 behavior groups only: the isolation contract (the supplied `ParameterVector` snapshot and the base-network snapshot stay unchanged after the helper returns) and the same-seed determinism contract (two calls with the same seed plus the same dataset order yield identical trained-vector snapshots on the same runtime). The fixture stays Phase 3-scoped by using a minimal `Network(1, 1, { seed })` topology, ordered `TrainingSample[]` input, and explicit `{ steps, learningRate, seed }` options without widening into Lamarckian persistence, policy hooks, or Phase 5 examples.

- Changed test file: `src/architecture/network/training/network.training.isolate.utils.test.ts`
- Focused command: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/architecture/network/training/network.training.isolate.utils.test.ts`
- Observed red failure: `TS2307: Cannot find module "./network.training.isolate.utils" or its corresponding type declarations.`
- Scope guard: Step 03 stays training-owned and Phase 3 scoped; it does not introduce Phase 4 policy behavior, Lamarckian persistence, or example-facing integration.

**Step 04 green condition:** Add the smallest training-owned `src/architecture/network/training/network.training.isolate.utils.ts` surface that exports `fineTuneVector` together with `FineTuneOptions` and `FineTuneResult` so the focused Jest slice goes green. Use the Step 02 clone path (`baseNetwork.clone()` -> `fromParameterVector(...)` -> optional `workingCopy.setSeed(seed)` -> `trainImpl(...)` -> `toParameterVector(...)`), preserve same-runtime ordered determinism for identical seed and dataset-order inputs, and keep the supplied `ParameterVector` and `baseNetwork` snapshots unchanged without introducing Phase 4 policy hooks.

#### Step 04: Implement fineTuneVector isolation helper [DONE]

```yaml
phase: 3
step: 4
agent: '04-implementing'
agent_file: '.github/agents/04-implementation-architect.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 05 — Validate isolation-helper gates'
skills:
  - 'hybrid-training-interop'
  - 'reproducibility-contracts'
specialists:
  - 'Hybrid Interop Scout'
validation:
  - 'The Step 03 red tests go green with a training-owned isolation helper and no Phase 4 policy behavior or silent mutation of the supplied network or vector.'
```

**User instruction:** Start a fresh session, select `04-implementing`, and paste this full step packet.

**Step objective:** Implement the smallest training-owned `fineTuneVector` helper (and supporting types `FineTuneOptions`, `FineTuneResult`) that satisfies the Phase 3 red tests while keeping the isolation and determinism contracts honest.

**Context the agent must know:**

- The helper must not mutate the supplied `ParameterVector` or any `Network` that is part of shared population state; use an internal clone or temporary working copy only.
- Dataset ordering and training settings must be explicit inputs, not inferred defaults.
- Deterministic behavior is available only when seed and RNG owner are explicit; document the boundary clearly in code and types.
- The API name must make the mutation or non-mutation contract obvious; `fineTuneVector` signals that it returns a new vector without touching the input.
- Keep recurrent buffer state, optimizer accumulators, and other transient training fields out of the returned result; the result must be a `ParameterVector` only.
- Do not introduce Phase 4 policy hooks (eligibility, persistence, Lamarckian opt-in) in this step.

**Execution steps:**

1. Read Phase 3 plan text, the Step 02 research brief, the Step 03 red evidence, and the named source or test files.
2. Add the minimal training-owned `fineTuneVector` helper, `FineTuneOptions`, and `FineTuneResult` types at the owner boundary confirmed by Step 02.
3. Implement the clone-or-working-copy approach so that the supplied `ParameterVector` (and any base `Network`) is never mutated.
4. Keep training-loop settings explicit and add concise JSDoc for all exported types and helpers.
5. Rerun the focused Step 03 command and update this plan with changed files, isolation decisions, remaining risks, and the Step 05 validation handoff.

**Stop conditions:**

- Done: the focused red tests are green and the implementation remains training-owned and Phase 3 scoped.
- Blocked: the existing clone path or training-loop surface does not support honest isolation without upstream contract changes.
- Route back: implementation reveals the isolation seam cannot remain honest without reopening Step 02 research or Step 03 red design.

**Required validation:** The focused Step 03 red-test command passes after the implementation change.

**Plan update requirement:** Update this plan with implementation files, focused validation evidence, isolation decisions, and the Step 05 handoff before ending the session.

**Step 04 implementation and validation evidence:** Added the smallest training-owned isolation helper at `src/architecture/network/training/network.training.isolate.utils.ts`. The new file defines `FineTuneOptions`, `FineTuneResult`, and `fineTuneVector(baseNetwork, vector, dataset, options)` in one boundary-local surface because the Phase 3 type contract is still trivial. The helper stays inside the Step 02 clone path: it clones `baseNetwork`, applies the supplied `ParameterVector` to the working copy, optionally installs `options.seed` through `workingCopy.setSeed(...)`, runs `trainImpl(...)` against the caller-provided ordered dataset, and returns `toParameterVector(workingCopy)` together with numeric training metrics from the training summary.

- Changed implementation file: `src/architecture/network/training/network.training.isolate.utils.ts`
- Focused command: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/architecture/network/training/network.training.isolate.utils.test.ts`
- Observed green result: `PASS` for `src/architecture/network/training/network.training.isolate.utils.test.ts`; `1` suite passed and both targeted `fineTuneVector isolation helper` tests passed for unchanged input snapshots and same-seed deterministic trained-vector output on the same runtime.
- Isolation decision: `fineTuneVector` treats `baseNetwork` and the supplied `ParameterVector` as read-only inputs. All mutation is confined to `baseNetwork.clone()`, so the helper returns a detached `trainedVector` without introducing Lamarckian persistence or any Phase 4 policy hook.
- Determinism decision: the helper documents only same-runtime ordered determinism when topology, dataset order, training settings, and explicit `seed` all match. It does not claim cross-runtime exact replay, and it does not return transient optimizer, activation, or recurrent runtime state.
- Scope guard: Step 04 remains training-owned and Phase 3 scoped; it does not add clone-based convenience wrappers, eligibility or persistence policy behavior, or example-facing integration.
- Remaining risks: Step 05 still needs to rerun the focused Jest slice, run `npx tsc --noEmit -p tsconfig.json` while treating the unchanged ONNX `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` as external unless the failure surface moves into this seam, and run coverage guard on the touched `src/architecture/network/training/network.training.isolate.utils.ts` file.

**Step 05 validation handoff:** Treat `src/architecture/network/training/network.training.isolate.utils.ts` as the touched source boundary and `src/architecture/network/training/network.training.isolate.utils.test.ts` as the owner-local behavior slice. Re-run the focused isolation-helper Jest command first, then run `npx tsc --noEmit -p tsconfig.json`, then run coverage guard on the touched `src/` file. If the focused slice fails inside the clone or vector-export path, route back to Step 04. If coverage is missing for the new helper or its metrics helpers, route back to Step 03 only for the smallest owner-local test additions.

#### Step 05: Validate isolation-helper gates [DONE]

```yaml
phase: 3
step: 5
agent: '05-green-testing'
agent_file: '.github/agents/05-green-validation-runner.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 06 — Curate isolation-helper docs'
skills:
  - 'green-validation-gates'
  - 'coverage-guard'
  - 'hybrid-training-interop'
specialists:
  - 'Coverage Guard'
  - 'Determinism Scout'
validation:
  - 'Focused isolation and determinism tests pass, TypeScript validation passes or the pre-existing ONNX baseline is explicitly confirmed unchanged, and coverage guard passes at 100/100/100/100 for every touched src file.'
```

**User instruction:** Start a fresh session, select `05-green-testing`, and paste this full step packet.

**Step objective:** Prove the Phase 3 implementation satisfies the isolation and determinism contracts with focused validation before documentation or phase closure.

**Context the agent must know:**

- Validate only the Phase 3 isolation-helper boundary and directly touched files.
- Primary behavior gates are: original `ParameterVector` unchanged after `fineTuneVector`, and same-seed plus same-dataset-order yields stable fine-tuned output.
- If `src/` files changed, coverage guard at `100/100/100/100` is required for every touched source file.
- The unchanged ONNX `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` remains the recorded external baseline unless the failure surface moves into this seam.
- Failed validation routes back to the smallest prior step that can repair the active seam.

**Execution steps:**

1. Read the Step 04 changed-file summary and expected validation commands.
2. Rerun the focused isolation-and-determinism Jest slice.
3. Run `npx tsc --noEmit -p tsconfig.json` after the focused slice is green.
4. Run coverage guard for every touched `src/` file.
5. Update this plan with pass or fail evidence and reroute failures to Step 03 or Step 04 as appropriate.

**Stop conditions:**

- Done: focused tests, TypeScript validation, and required coverage gates pass.
- Blocked: a validation command cannot run or a pre-existing TypeScript baseline still blocks repo-wide validation.
- Route back: behavior failures return to Step 04; missing or insufficient owner-local coverage returns to Step 03.

**Required validation:** Command evidence for the focused test slice, `npx tsc --noEmit -p tsconfig.json`, and coverage guard on touched `src/` files.

**Plan update requirement:** Update this plan with validation evidence, reroute decisions if any, and the Step 06 documentation handoff before ending the session.

**Step 05 completion from current repo state:** Re-ran the focused isolation-helper slice with `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/architecture/network/training/network.training.isolate.utils.test.ts`. The Phase 3 behavior gate stayed green again: `PASS` for `src/architecture/network/training/network.training.isolate.utils.test.ts`, `1` suite passed, and all `3` targeted `fineTuneVector isolation helper` tests passed for unchanged input snapshots, same-seed deterministic trained-vector output, and the no-seed isolation case.

Repo-wide TypeScript validation still does not clear, but the failure surface remains unchanged. `npx tsc --noEmit -p tsconfig.json` failed only at the recorded external ONNX baseline `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` with `TS2345`, because `standardDomainImports[0]!.version` can be `undefined`. No new TypeScript failure appeared inside the Phase 3 training-isolation seam, so this rerun treats the ONNX error as external baseline context rather than a Phase 3 blocker.

Coverage guard on the touched `src/` file used the owner-local training suite: `npx jest --config=jest.config.mjs --no-cache --coverage --runTestsByPath src/architecture/network/training/network.training.isolate.utils.test.ts --collectCoverageFrom=src/architecture/network/training/network.training.isolate.utils.ts --coverageReporters=text-summary`. The guard passed, and `src/architecture/network/training/network.training.isolate.utils.ts` is now at `100/100/100/100` with `11/11` statements, `2/2` branches, `3/3` functions, and `11/11` lines covered.

- Step 05 gate decision: close Step 05 and hand off to Step 06 documentation.
- Tracker validation: `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` passed with `0` errors and `0` warnings after this Step 05 tracker refresh.
- Scope guard: this rerun stayed inside the Phase 3 isolation-helper seam only; no production code, Phase 4 policy behavior, Lamarckian persistence, or unrelated ONNX work changed.
- Exact next orchestrator: `06-documenting`
- Exact handoff prompt: see `## Handoff query` below.

#### Step 06: Curate isolation-helper docs [DONE]

```yaml
phase: 3
step: 6
agent: '06-documenting'
agent_file: '.github/agents/06-educational-docs-curator.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 07 — Log Phase 3 closure or reroute'
skills:
  - 'educational-docs'
  - 'docs-academic-citation-audit'
  - 'hybrid-training-interop'
specialists:
  - 'Docs Scout'
validation:
  - 'Source-first docs for all exported isolation-helper symbols are present, generated docs are refreshed if JSDoc inputs changed, and no Phase 4 policy or Lamarckian persistence docs are introduced.'
```

**User instruction:** Start a fresh session, select `06-documenting`, and paste this full step packet.

**Step objective:** Keep the isolation-helper surface teachable and synchronized with generated docs without turning Phase 3 into the broader Phase 5 documentation closure.

**Context the agent must know:**

- Documentation scope is limited to `fineTuneVector`, `FineTuneOptions`, `FineTuneResult`, the isolation contract (original unchanged), same-runtime determinism wording, and explicit seed or dataset-order semantics.
- Phase 4 policy (Lamarckian, fitness-only, conditional), Phase 5 examples, and Phase 5 Lamarckian explanation remain out of scope.
- Generated `src/**/README.md` files and `docs/examples/**` outputs are read-only; update source JSDoc and run generation when needed.
- Public docs should be atemporal: no tracker phases, roadmap chronology, or chat-only context.

**Execution steps:**

1. Read Step 04 changed public surfaces and Step 05 validation evidence.
2. Improve only source-owned JSDoc or nearby handwritten docs required by the new isolation-helper surface.
3. Run `npm run docs` if JSDoc or generated-doc inputs changed; otherwise record a manual no-docs-needed rationale.
4. Verify that no generated README or published example output was hand-edited as a shortcut.
5. Update this plan with documentation evidence and any residual docs risk before ending the session.

**Stop conditions:**

- Done: required source docs are aligned and docs generation either passes or is explicitly not needed.
- Blocked: public naming or determinism wording is unstable enough that docs would mislead downstream consumers.
- Route back: documentation reveals implementation drift in the isolation contract; route to Step 04 or Step 05 rather than widening docs.

**Required validation:** `npm run docs` when source JSDoc or generated-doc inputs changed, or a manual no-docs-needed record when implementation introduced no doc-facing surface.

**Plan update requirement:** Update this plan with documentation changes, validation evidence, residual gaps, and the Step 07 closure handoff before ending the session.

**Step 06 documentation changes and validation evidence:** Reviewed source JSDoc for `FineTuneOptions`, `FineTuneResult`, and `fineTuneVector` in `src/architecture/network/training/network.training.isolate.utils.ts`. The generated `src/architecture/network/training/README.md` did not yet include any of the three new isolation-helper symbols (the file had been added in Step 04 but `npm run docs` had not been run for this boundary). Source JSDoc was improved as follows, keeping all changes Phase 3 scoped and atemporal:

- `FineTuneOptions`: added a closing paragraph explaining that without `seed` isolation is still guaranteed (original network and vector never mutated) but repeated calls may diverge when the training loop contains stochastic behaviour.
- `FineTuneResult`: added a closing sentence listing downstream use patterns — compare against original vector, forward to a worker, persist as a checkpoint delta, or discard when only the fitness score matters — without introducing Lamarckian persistence or Phase 4 policy language.
- `fineTuneVector`: added an `@example` fenced code block showing: export the current parameter vector, call `fineTuneVector` with explicit `steps`, `learningRate`, and `seed`, log the training error metric, and confirm the original candidate and vector are unchanged after the call. No Phase 4 or Phase 5 language was added.

- Changed source file: `src/architecture/network/training/network.training.isolate.utils.ts`
- Validation command: `npm run docs` — completed successfully with `HTML docs generated.`
- Generated output verified: `src/architecture/network/training/README.md` now includes `FineTuneOptions`, `FineTuneResult`, and `fineTuneVector` with isolation contract, stochastic-dropout note, downstream-use note, and the full `@example` block rendering correctly.
- Source-of-truth guard: no generated `src/**/README.md` file or `docs/examples/**` output was hand-edited as a shortcut. The only manual edits were to the source JSDoc file, and the README refresh came from `npm run docs`.
- Scope guard: no Phase 4 policy, Lamarckian persistence, conditional fine-tuning, or Phase 5 example language was introduced.
- Citation note: no external citation required; the isolation contract is repo-owned.
- Tracker validation: `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` passed with `0` errors and `0` warnings before this tracker update.
- Residual docs risk: none within the Phase 3 isolation-helper seam. Step 07 should decide Phase 3 closure against the preserved Phase 3 acceptance evidence and the unchanged external ONNX TypeScript baseline.

#### Step 07: Log Phase 3 closure or reroute [DONE]

```yaml
phase: 3
step: 7
agent: '07-logging'
agent_file: '.github/agents/07-session-log-archivist.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Phase 4 Step 01 — Plan the hybrid-policy tranche'
skills:
  - 'tracker-handoff'
  - 'plan-sync-validation'
  - 'hybrid-training-interop'
specialists:
  - 'Plan Registration Auditor'
validation:
  - 'Manual tracker review confirms Phase 3 acceptance evidence, the isolation contract and determinism claims are documented, and the refreshed Handoff query points to Phase 4 Step 01 before the next phase opens.'
```

**User instruction:** Start a fresh session, select `07-logging`, and paste this full step packet.

**Step objective:** Close or reroute Phase 3 based on validated evidence, preserve the isolation and determinism contract decisions, and leave the tracker ready either for Phase 4 planning or for the smallest blocking reroute.

**Context the agent must know:**

- Phase 3 closes only when fine-tuning provably cannot mutate shared population state and same-seed plus same-dataset-order yields stable fine-tuned output.
- Phase 4 should not open until Phase 3 isolation is green and coverage guard passes for all touched `src/` files.
- The overall lane remains active; do not archive this tracker unless the full hybrid-interoperability workstream is done.
- Chat is not the source of truth; the tracker and the `Handoff query` must carry the next session.

**Execution steps:**

1. Read the Phase 3 Step 02-06 evidence and verify that no required validation is missing.
2. Mark completed Phase 3 steps `[DONE]` or record the smallest reroute if evidence is incomplete.
3. Record that the isolation contract is now real and that Phase 4 policy integration can begin.
4. Refresh the `Handoff query` to point either to Phase 4 Step 01 or to the smallest blocking reroute step.
5. Run `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` after the tracker refresh.

**Stop conditions:**

- Done: Phase 3 evidence is compactly recorded, the next safe step is explicit, and the handoff query is current.
- Blocked: validation, documentation, or isolation evidence is missing and cannot be reconstructed from the tracker.
- Route back: missing isolation coverage routes to Step 03, implementation mutation drift routes to Step 04, failed validation routes to Step 05, and doc drift routes to Step 06.

**Required validation:** Manual tracker diff review confirming Phase 3 acceptance evidence, no premature Phase 4 behavior, and a current next-step handoff.

**Plan update requirement:** Update this plan with Phase 3 closure or reroute state, refreshed handoff query, and the next active step before ending the session.

**Step 07 closure review and handoff evidence:** Step 07 reviewed the recorded Phase 3 evidence from Step 02 through Step 06 from current repo state only. Phase 3 can now close. The training-owned isolation seam satisfies the phase acceptance gate: Step 02 locked the training owner boundary, helper signature, dataset shape, clone path, and RNG owner; Step 03 added owner-local tests for isolation and same-seed determinism and expanded the slice with a no-seed isolation follow-up; Step 04 implemented `fineTuneVector`, `FineTuneOptions`, and `FineTuneResult` in `src/architecture/network/training/network.training.isolate.utils.ts`; Step 05 revalidated the focused behavior slice with `1` suite and `3` tests green, confirmed the unchanged external ONNX `TS2345` as the only repo-wide TypeScript baseline failure, and passed coverage guard at `100/100/100/100` for the touched training source file; and Step 06 improved source-first JSDoc and refreshed `src/architecture/network/training/README.md` via `npm run docs` without hand-editing generated output.

- Phase 3 closure decision: close Phase 3. No reroute is required because the isolation contract, same-runtime determinism wording, focused validation, coverage guard, and generated-doc refresh are all durably recorded.
- Isolation contract decision: `fineTuneVector` now makes the isolation rule real by cloning `baseNetwork`, applying the supplied `ParameterVector` only to the working copy, and returning a detached `trainedVector`; the supplied network and vector remain unchanged. Determinism remains intentionally scoped to same-runtime ordered runs when topology, dataset order, training settings, and explicit `seed` match.
- Tracker state decision: keep the top-level tracker active in `plans/` as `[WIP]`; mark Phase 3 complete; keep Phase 4 unopened in planning state; do not archive this tracker or create a same-boundary `.logs.md` file because the hybrid-interoperability lane still has Phase 4 and Phase 5 work.
- Next safe step: activate `01-planning` for Phase 4 `Step 01 — Plan the hybrid-policy tranche`. Keep the next pass planning-only; do not implement policy hooks, Lamarckian persistence behavior, or unrelated ONNX fixes in this closure step.
- Tracker validation: `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` passed after this tracker refresh with `0` errors and `0` warnings.

### Phase 4 — Define hybrid evaluation policy integration [DONE]

**Phase objective:** Make hybrid evaluation policy explicit at the NEAT surface so users can choose when fine-tuning occurs and whether trained weights are discarded or persisted.

**Phase progression rule:** Start with only Step 01. Step 01 must author the remaining numbered step packets, or explicit skipped-step packets, before the phase can advance.

**Preserved implementation contract:**

- Integrate policy hooks into the NEAT evaluation surface only after Phase 1 through Phase 3 are green.
- Keep three decisions separate:
  - which candidates are allowed to fine-tune,
  - how the trained variant is scored,
  - whether trained weights persist back into the canonical candidate.
- Treat `never`, `always`, and `conditional` as policy choices, but make the conditional path define deterministic ranking or tie-break rules rather than inheriting current iteration order.
- Require worker and single-thread paths to agree on semantic result order before Lamarckian persistence is applied.

**Acceptance when this phase closes:**

- Users can choose fitness-only fine-tuning or Lamarckian persistence.

**Focused validation to preserve:**

- The no-fine-tune policy leaves candidates untouched.
- Fitness-only fine-tuning returns a trained result without persistence.
- Lamarckian persistence happens only on explicit opt-in.

#### Step 01: Plan the hybrid-policy tranche [DONE]

```yaml
phase: 4
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning-architect.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 02 — Research hybrid-policy boundary'
skills:
  - 'hybrid-training-interop'
specialists:
  - 'Hybrid Interop Scout'
validation:
  - 'Manual tracker check that the policy split, Lamarckian opt-in rule, and deterministic ranking guardrails are preserved before Step 02-07 packets are authored.'
```

**User instruction:** Start a fresh session, select `01-planning`, and paste this full step packet.

**Step objective:** Turn Phase 4 into an executable hybrid-policy workset by authoring Phase 4 Step 02-07 packets, preserving the explicit persistence-policy split, and keeping the phase gated on successful completion of the earlier parameter-vector and isolation phases.

**Context the agent must know:**

- Phase 4 depends on Phase 1 through Phase 3; it must not introduce policy hooks before the lower-level vector and isolation seams are stable.
- The policy surface must keep eligibility, scoring, and persistence as separate decisions rather than collapsing them into one boolean or application-specific shortcut.
- Conditional fine-tuning requires deterministic ranking or tie-break rules.
- Lamarckian persistence must remain explicit opt-in and must not become the default side effect of evaluation.

**Execution steps:**

1. Reconfirm the preserved Phase 4 contract, acceptance rule, and focused validation against the current plan text.
2. Decide which phase-agent steps are real value gates for policy integration and which should be explicit skipped-step packets.
3. Author Phase 4 Step 02-07 packets or explicit skipped-step packets with the narrowest honest NEAT-evaluation scope.
4. Record any dependency on unresolved worker-ordering or isolation semantics directly in the plan before naming the next active step.

**Stop conditions:**

- Done: Phase 4 has a paste-ready next active step and all remaining step slots are authored or explicitly skipped.
- Blocked: unresolved Phase 1 through Phase 3 contracts prevent an honest hybrid-policy packet.
- Route back: if planning reveals that persistence or ordering semantics are still ambiguous, update the tracker and return to planning instead of widening into docs or examples.

**Required validation:** Manual diff review confirming that every former Step 4 detail now lives in Phase 4 and that Lamarckian persistence remains explicit opt-in only.

**Plan update requirement:** Update this plan with the new Step 02-07 packets, any skipped-step rationale, validation evidence, and the next active step before ending the session.

**Planning decisions and validation evidence:** Step 01 preserved the Phase 4 contract, acceptance rule, focused validation, and three-decision split from the current plan text. Phase 1, Phase 2, and Phase 3 are each [DONE] and supply the real parameter-vector layout, roundtrip, and isolation seams that Phase 4 policy integration requires. The unchanged ONNX TypeScript failure at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` (`TS2345`) remains the recorded external baseline and is not a Phase 4 planning blocker. `plans/README.md` and `plans/Roadmap.md` still align this lane as the last non-chat foundation handoff before the dependency-gated NEATchat follow-up, with no roadmap conflict for a NEAT-evaluation-surface policy hook. Research, Red Testing, Implementation, Green Validation, Documentation, and Session Logging all add independent value because Phase 4 introduces a new behavioral policy surface; no Step 02-07 slot is skipped.

**Worker-ordering dependency note:** The Phase 4 contract requires worker and single-thread evaluation paths to agree on semantic result order before Lamarckian persistence is applied. Step 02 must map the NEAT evaluation surface and worker path before Step 03 can design honest tests for Lamarckian persistence. If Step 02 cannot confirm the ordering constraint from current repo evidence, Step 03 should record an explicit conditional-policy blocked note and scope tests to `never` and `always` paths only.

**Next active step:** Phase 4 Step 02 — Research hybrid-policy boundary.

#### Step 02: Research hybrid-policy boundary [DONE]

```yaml
phase: 4
step: 2
agent: '02-researching'
agent_file: '.github/agents/02-research-coordinator.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 03 — Design hybrid-policy red tests'
skills:
  - 'hybrid-training-interop'
  - 'plan-alignment'
specialists:
  - 'Hybrid Interop Scout'
validation:
  - 'Read-only research brief names the NEAT evaluation owner boundary, the natural policy-hook location, worker vs single-thread ordering constraints, and the narrowest owner-local test seam before red tests begin.'
```

**User instruction:** Start a fresh session, select `02-researching`, and paste this full step packet.

**Step objective:** Gather the smallest read-only evidence needed to design honest red tests for the three policy choices (`never`, `always` fitness-only, `always` Lamarckian) without opening the NEAT evaluation loop, worker threading implementation, or Phase 5 documentation.

**Context the agent must know:**

- Phase 3 supplies `fineTuneVector(baseNetwork, vector, dataset, options): FineTuneResult` from `src/architecture/network/training/network.training.isolate.utils.ts`. Phase 4 wraps that seam with an explicit policy layer.
- The three decisions are: which candidates fine-tune (eligibility), how the trained variant is scored, whether trained weights persist back (Lamarckian opt-in).
- Worker and single-thread evaluation paths must agree on semantic result order before Lamarckian persistence is applied; map this ordering constraint before Step 03 designs tests.
- Carry the unchanged ONNX `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` as external baseline context unless the failure surface moves into this seam.

**Execution steps:**

1. Read `plans/README.md`, `plans/Roadmap.md`, this Phase 4 section, and the nearest relevant `src/neat/evaluate/README.md` plus `src/multithreading/README.md` before individual source files.
2. Delegate a focused read-only packet to `Hybrid Interop Scout` to map the NEAT evaluation surface: where per-candidate evaluation is called, what the evaluation function signature looks like, and how the worker path differs from the single-thread path.
3. Identify the smallest honest owner boundary for `HybridEvaluationPolicy` and the policy evaluation helper — whether it belongs in `src/neat/evaluate/`, a dedicated `src/neat/hybrid/`, or alongside the Phase 3 training boundary.
4. Confirm that the `TrainingSample[]` shape consumed by `fineTuneVector` is compatible with the dataset the NEAT evaluation already carries, or note any shape gap.
5. Record the worker-vs-single-thread ordering constraint: whether result ordering is safe enough for Lamarckian persistence, and whether a test fixture can simulate both paths honestly without full worker threads.

**Stop conditions:**

- Done: research identifies the owner boundary, proposed policy-hook location, ordering constraints, and the narrowest owner-local test seam.
- Blocked: the NEAT evaluation surface lacks a clear hook point or the worker ordering constraint cannot be confirmed from read-only evidence.
- Route back: research shows policy integration requires upstream changes to the worker or evaluation loop before Phase 4 can be honest.

**Required validation:** Manual evidence review confirming the research remains read-only, boundary-local, and aligned with the Phase 4 acceptance rule.

**Plan update requirement:** Update this plan with the research brief, policy hook location, ordering constraint decision, and the Step 03 handoff before ending the session.

**Research brief and validation evidence (Step 02 — [DONE]):**

- **Owner boundary:** New module `src/neat/hybrid/` following the CLAUDE.md folder-based naming convention. Files: `neat.hybrid.ts` (orchestration, public `evaluateCandidate` helper), `neat.hybrid.types.ts` (`HybridEvaluationPolicy`, `EvaluateCandidateOptions`, `HybridEvaluationResult`). This is preferred over injecting into `src/neat/evaluate/` because the policy helper is a **standalone function** the user calls inside their fitness delegate, not a hook into the NEAT evaluation loop itself.

- **Policy hook location:** The per-candidate evaluation loop in `src/neat/evaluate/fitness/evaluate.fitness.ts` iterates `controller.population` sequentially and assigns `genome.score = await controller.fitness(genome)`. The hook is **not** placed inside this loop. Instead, the Phase 4 helper is a standalone `evaluateCandidate(network, dataset, options)` async function the user calls inside their fitness delegate. The NEAT evaluation loop (`runFitnessEvaluation`) stays unchanged.

- **Dataset compatibility:** CONFIRMED. `TrainingSample = { input: number[]; output: number[] }` from `src/architecture/network/training/network.training.utils.types.ts` is identical to what `fineTuneVector` expects. NEAT's fitness delegate receives a `GenomeForEvaluation` object, NOT a dataset; the user always supplies the dataset explicitly to the policy helper. There is no shape gap.

- **Worker vs. single-thread ordering:**
  - Single-thread: sequential per-genome iteration — deterministic by construction.
  - Worker path (`evaluateInWorkers`): returns `BatchEvaluationResult<TResult>` with `results: TResult[]` **ordered and index-aligned** to the original population array. `createNeatParallelPopulationEvaluator` applies results via `population.forEach` with `genomeIndex` read from `batchResult.results[genomeIndex]`.
  - **Lamarckian constraint on worker path:** Workers operate on serialized copies and return scalar scores (`TResult = number`). Applying trained weights back to the live genome in the host process is NOT supported by the default `createNeatParallelPopulationEvaluator` result-assignment path. If Lamarckian persistence on the worker path is needed, the caller must encode trained weights in `TResult` and apply them in `assignResult` — this is a design extension beyond Phase 4 scope.
  - **Test fixture implication:** Phase 4 Lamarckian tests can be scoped to the single-thread path only. A mock fitness delegate simulates both paths without full worker threads.

- **Conditional policy status:** BLOCKED. The NEAT evaluation loop carries no stable top-K ranking or tie-break surface during per-genome fitness evaluation; ranking happens in `evolve()` after evaluation. Step 03 must record an explicit conditional-policy blocked note and scope tests to `never` and `always` paths only.

- **External baseline unchanged:** ONNX `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` remains the only repo-wide TypeScript blocker; it is unrelated to this lane.

**Step 03 handoff:**

Add the smallest failing owner-local tests in a new file `src/neat/hybrid/neat.hybrid.test.ts` covering three behavior groups:

1. `never` policy — `evaluateCandidate` returns the raw fitness score without calling `fineTuneVector` and leaves the candidate weights bitwise unchanged.
2. `always` fitness-only policy — the helper calls `fineTuneVector`, uses the trained result for scoring, but does NOT apply trained weights back to the original genome.
3. `always` Lamarckian policy — the helper calls `fineTuneVector` and applies trained weights to the candidate only when `persistTrainedWeights === true`.
4. Record an explicit conditional-policy blocked note — no stable deterministic ranking surface is confirmed at the current evaluate boundary.

Tests must stay out of full NEAT population wiring, worker threads, and Phase 5 example constructs. One top-level `expect(...)` per test, AAA structure, nested `describe` blocks per repo conventions.

#### Step 03: Design hybrid-policy red tests [DONE]

```yaml
phase: 4
step: 3
agent: '03-red-testing'
agent_file: '.github/agents/03-red-test-architect.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 04 — Implement hybrid evaluation policy'
skills:
  - 'hybrid-training-interop'
  - 'red-test-contracts'
  - 'reproducibility-contracts'
specialists:
  - 'Determinism Scout'
validation:
  - 'Focused owner-local red tests or an explicit blocked record prove that the three policy choices (never, fitness-only, Lamarckian opt-in) can be tested before implementation.'
```

**User instruction:** Start a fresh session, select `03-red-testing`, and paste this full step packet.

**Step objective:** Add the smallest failing test contract for the three Phase 4 policy behaviors: no-fine-tune leaves candidates untouched, fitness-only fine-tuning returns a trained result without persistence, and Lamarckian persistence applies trained weights back only on explicit `persistTrainedWeights: true`.

**Context the agent must know:**

- The three test groups correspond to three separate behavioral contracts: the `never` policy (no mutation, no extra calls to `fineTuneVector`), the `always` fitness-only policy (trained fitness returned, original candidate weights bitwise unchanged), and the `always` Lamarckian policy (trained weights applied to candidate on explicit `persistTrainedWeights: true`).
- Conditional fine-tuning (`conditional` policy) requires deterministic ranking or tie-break rules; if Step 02 research cannot identify a stable ranking surface, record an explicit conditional-policy blocked note and keep Step 03 scoped to `never` and `always` paths only.
- Tests must follow repo conventions: owner-local placement, one top-level `expect(...)` per test, AAA structure, nested `describe` blocks.
- Carry the Step 02 research brief as the primary evidence for owner boundary and test seam.

**Execution steps:**

1. Read the Step 02 research brief, the named test seam, and the nearest existing owner-local tests.
2. Add the smallest failing tests for the three policy behaviors, or record an explicit blocked note for conditional policy if deterministic ranking is not yet clear.
3. Ensure tests do not require Phase 5 example constructs or full NEAT population wiring.
4. Run the narrowest practical Jest command for the touched owner-local test file, or record precisely why the red command is blocked.
5. Update this plan with the failing command, expected failure, and the exact Step 04 green condition.

**Stop conditions:**

- Done: focused red tests fail for the missing policy surface, or a precise blocked record proves the policy cannot be tested honestly before Step 02 research is extended.
- Blocked: Step 02 leaves the policy hook location or observable behavior ambiguous.
- Route back: the red design requires Phase 5 documentation or upstream evaluation-loop changes that expose new worker ordering constraints.

**Required validation:** Focused red-test command evidence, or a manual blocked record that explains why no honest Phase 4-only red test exists yet.

**Plan update requirement:** Update this plan with changed test files or blocked rationale, red evidence, and the Step 04 implementation handoff before ending the session.

**Step 03 red contract and validation evidence:** Added the smallest owner-local Phase 4 red contract in `src/neat/hybrid/neat.hybrid.test.ts`. The new slice keeps Phase 4 scoped to a standalone helper and avoids full NEAT population wiring, worker threads, and Phase 5 example constructs. The tests assume one owner-local `evaluateCandidate(network, dataset, options)` helper at the new `src/neat/hybrid/` seam and lock three behavior groups only:

1. `never` policy returns the base evaluation, leaves candidate weights bitwise unchanged, and returns no trained result.
2. `always` fitness-only policy scores the trained variant, returns that trained result, and leaves the original candidate unchanged.
3. `always` Lamarckian policy scores the trained variant and applies the trained weights back only when `persistTrainedWeights: true`.

The red fixture stays deterministic and boundary-local by reusing the existing Phase 3 `fineTuneVector(...)` helper only to precompute baseline-vs-trained network signatures for a `scoreNetwork` callback. That callback returns distinct numeric scores for the untouched baseline versus the trained variant, so the red assertions prove which network state the future helper used for scoring without widening into worker-ordering or full evaluation-loop integration.

- Changed test file: `src/neat/hybrid/neat.hybrid.test.ts`
- Focused command: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts`
- Observed red failure: `TS2307: Cannot find module './neat.hybrid' or its corresponding type declarations.`
- Conditional-policy blocked note: `conditional` remains explicitly blocked in Step 03. Step 02 confirmed there is no stable deterministic top-K or tie-break surface at the current per-genome evaluate boundary, so this red contract stays scoped to `never` and `always` paths only.
- Scope guard: Step 03 remains Phase 4 scoped and owner-local to the new hybrid seam. It does not alter `runFitnessEvaluation`, worker result assignment, Lamarckian-on-workers behavior, or Phase 5 docs or example surfaces.

**Step 04 green condition:** Add the smallest `src/neat/hybrid/` production surface so the focused red slice goes green: `src/neat/hybrid/neat.hybrid.ts` exporting `evaluateCandidate`, plus `src/neat/hybrid/neat.hybrid.types.ts` exporting the policy/result types. The helper must accept the Step 03 standalone orchestration shape (`network`, explicit `dataset`, explicit `policy`, `fineTuneOptions`, and a network-scoring callback), skip `fineTuneVector` entirely for `fineTune: 'never'`, score a detached trained variant for `persistTrainedWeights: false`, and apply trained weights back to the candidate only when `persistTrainedWeights === true`. Keep `conditional` blocked for now rather than inventing an unstable ranking rule.

#### Step 04: Implement hybrid evaluation policy [DONE]

```yaml
phase: 4
step: 4
agent: '04-implementing'
agent_file: '.github/agents/04-implementation-architect.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 05 — Validate hybrid-policy gates'
skills:
  - 'hybrid-training-interop'
  - 'reproducibility-contracts'
specialists:
  - 'Hybrid Interop Scout'
validation:
  - 'The Step 03 red tests go green with a policy-aware evaluation helper and no default Lamarckian behavior or implicit candidate mutation.'
```

**User instruction:** Start a fresh session, select `04-implementing`, and paste this full step packet.

**Step objective:** Implement the smallest `HybridEvaluationPolicy` surface and the policy evaluation helper that separates the three policy decisions (eligibility, scoring, persistence) so that Lamarckian persistence cannot happen without `persistTrainedWeights: true`.

**Context the agent must know:**

- The three decisions must remain explicit in the API: `fineTune` (eligibility and trigger: `never` | `always` | `conditional`), `persistTrainedWeights` (persistence opt-in boolean), and implicit scoring (always use trained result when `fineTune !== 'never'`).
- Lamarckian persistence must not be the default; `persistTrainedWeights: false` must be the safe default.
- The implementation should consume `fineTuneVector` from Phase 3 (`src/architecture/network/training/network.training.isolate.utils.ts`) rather than re-implementing training logic.
- Worker and single-thread paths must agree on semantic result order for Lamarckian persistence; if this constraint is not yet resolvable, record the blocker and gate Lamarckian persistence on explicit single-thread evidence only.
- Do not widen into full NEAT population integration or Phase 5 example constructs in this step unless Step 02 research proved the hook point is within the population evaluator itself.

**Execution steps:**

1. Read Phase 4 plan text, the Step 02 research brief, the Step 03 red evidence, and the named source or test files.
2. Add the minimal `HybridEvaluationPolicy` type (and any supporting types) and the policy evaluation helper at the owner boundary confirmed by Step 02.
3. Implement the three policy paths with explicit guard-rails: `never` skips `fineTuneVector`, `always` fitness-only calls `fineTuneVector` without applying trained weights back, `always` Lamarckian calls `fineTuneVector` and applies trained weights back only when `persistTrainedWeights === true`.
4. Add concise JSDoc for all exported types and helpers, emphasizing that `persistTrainedWeights: false` is the safe default.
5. Rerun the focused Step 03 command and update this plan with changed files, policy decisions, remaining risks, and the Step 05 validation handoff.

**Stop conditions:**

- Done: the focused red tests are green and the implementation remains Phase 4 scoped with no implicit Lamarckian default.
- Blocked: the worker ordering constraint from Step 02 prevents an honest Lamarckian path from being implemented safely.
- Route back: implementation reveals the policy hook cannot be isolated without reopening Step 02 research.

**Required validation:** The focused Step 03 red-test command passes after the implementation change.

**Plan update requirement:** Update this plan with implementation files, focused validation evidence, policy decisions, and the Step 05 handoff before ending the session.

**Step 04 implementation and validation evidence:** Added the smallest owner-local hybrid policy surface under `src/neat/hybrid/`. The new `src/neat/hybrid/neat.hybrid.types.ts` file defines `HybridFineTuneMode`, `HybridEvaluationPolicy`, `HybridScoreNetwork`, `EvaluateCandidateOptions`, and `HybridEvaluationResult`, with JSDoc that keeps `persistTrainedWeights: false` as the safe default. The new `src/neat/hybrid/neat.hybrid.ts` file exports `evaluateCandidate(network, dataset, options)` and keeps the three policy decisions explicit: `fineTune: 'never'` scores the live candidate without touching `fineTuneVector`, `fineTune: 'always'` delegates training to Phase 3's `fineTuneVector(...)` and scores a detached trained clone, and Lamarckian persistence applies the trained vector back to the original candidate only when `persistTrainedWeights === true`. The unresolved `conditional` mode now fails explicitly with a deterministic-ranking blocker instead of inventing an unstable policy path.

- Changed implementation files: `src/neat/hybrid/neat.hybrid.ts`, `src/neat/hybrid/neat.hybrid.types.ts`
- Focused command: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts`
- Observed green result: `PASS` for `src/neat/hybrid/neat.hybrid.test.ts`; `1` suite passed and all `3` targeted `evaluateCandidate` policy tests passed for `never`, fitness-only `always`, and Lamarckian `always` behavior.
- Policy decision: scoring is detached from persistence. Whenever `fineTune !== 'never'`, `evaluateCandidate` scores a trained clone built from the `trainedVector`; candidate mutation happens only after successful scoring and only when `persistTrainedWeights === true`.
- Determinism decision: this step reuses the same-runtime ordered determinism already documented by `fineTuneVector(...)`; it does not add a new cross-runtime replay claim.
- Conditional-policy decision: `conditional` remains intentionally blocked because Step 02 did not find a stable deterministic ranking or tie-break surface at the current per-genome evaluate boundary.
- Scope guard: Step 04 stays Phase 4 scoped and owner-local to `src/neat/hybrid/`. It does not change `runFitnessEvaluation`, worker result assignment, Lamarckian persistence on the worker path, or Phase 5 docs and examples.
- Remaining risks: Step 05 still needs to rerun the focused hybrid Jest slice, run `npx tsc --noEmit -p tsconfig.json` while treating the unchanged ONNX `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` as external unless the failure surface moves into this seam, and run coverage guard for `src/neat/hybrid/neat.hybrid.ts` and `src/neat/hybrid/neat.hybrid.types.ts`.

**Step 05 validation handoff:** Treat `src/neat/hybrid/neat.hybrid.ts` and `src/neat/hybrid/neat.hybrid.types.ts` as the touched source boundary and `src/neat/hybrid/neat.hybrid.test.ts` as the owner-local behavior slice. Re-run the focused hybrid-policy Jest command first, then run `npx tsc --noEmit -p tsconfig.json`, then run coverage guard on both touched `src/` files. If the focused slice fails inside the policy-routing or Lamarckian-persistence path, route back to Step 04. If coverage is missing for the new helper or types file, route back to Step 03 only for the smallest owner-local test additions.

#### Step 05: Validate hybrid-policy gates [DONE]

```yaml
phase: 4
step: 5
agent: '05-green-testing'
agent_file: '.github/agents/05-green-validation-runner.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 06 — Curate hybrid-policy docs'
skills:
  - 'green-validation-gates'
  - 'coverage-guard'
  - 'hybrid-training-interop'
specialists:
  - 'Coverage Guard'
  - 'Determinism Scout'
validation:
  - 'Focused policy-behavior tests pass, TypeScript validation passes or the pre-existing ONNX baseline is explicitly confirmed unchanged, and coverage guard passes at 100/100/100/100 for every touched src file.'
```

**User instruction:** Start a fresh session, select `05-green-testing`, and paste this full step packet.

**Step objective:** Prove the Phase 4 implementation satisfies the three policy contracts with focused validation before documentation or phase closure.

**Context the agent must know:**

- Validate only the Phase 4 policy surface and directly touched files.
- Primary behavior gates: `never` leaves candidates untouched, `always` fitness-only returns trained fitness without persistence, and Lamarckian persistence applies trained weights only on `persistTrainedWeights: true`.
- If `src/` files changed, coverage guard at `100/100/100/100` is required for every touched source file.
- The unchanged ONNX `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` remains the recorded external baseline unless the failure surface moves into this seam.
- Failed validation routes back to the smallest prior step that can repair the active seam.

**Execution steps:**

1. Read the Step 04 changed-file summary and expected validation commands.
2. Rerun the focused policy-behavior Jest slice.
3. Run `npx tsc --noEmit -p tsconfig.json` after the focused slice is green.
4. Run coverage guard for every touched `src/` file.
5. Update this plan with pass or fail evidence and reroute failures to Step 03 or Step 04 as appropriate.

**Stop conditions:**

- Done: focused tests, TypeScript validation, and required coverage gates pass.
- Blocked: a validation command cannot run or a pre-existing TypeScript baseline still blocks repo-wide validation.
- Route back: behavior failures return to Step 04; missing or insufficient owner-local coverage returns to Step 03.

**Required validation:** Command evidence for the focused test slice, `npx tsc --noEmit -p tsconfig.json`, and coverage guard on touched `src/` files.

**Plan update requirement:** Update this plan with validation evidence, reroute decisions if any, and the Step 06 documentation handoff before ending the session.

**Step 05 validation evidence and reroute:** Re-ran the focused hybrid-policy slice with `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts`. The Phase 4 behavior gate stayed green: `PASS` for `src/neat/hybrid/neat.hybrid.test.ts`, `1` suite passed, and all `3` targeted `evaluateCandidate` policy tests passed for `never`, fitness-only `always`, and Lamarckian `always` behavior.

Repo-wide TypeScript validation still does not clear, but the failure surface remains unchanged. `npx tsc --noEmit -p tsconfig.json` failed only at the recorded external ONNX baseline `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` with `TS2345`, because `standardDomainImports[0]!.version` can be `undefined`. No new TypeScript failure appeared inside the Phase 4 hybrid-policy seam, so this rerun treats the ONNX error as external baseline context rather than a Phase 4 blocker.

Coverage guard on the touched `src/` files used the owner-local hybrid suite: `npx jest --config=jest.config.mjs --no-cache --coverage --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts --collectCoverageFrom='src/neat/hybrid/neat.hybrid.ts' --collectCoverageFrom='src/neat/hybrid/neat.hybrid.types.ts' --coverageReporters=text --coverageReporters=json-summary`. `src/neat/hybrid/neat.hybrid.types.ts` was not counted by Jest coverage because it is a type-only file with no coverable runtime statements. `src/neat/hybrid/neat.hybrid.ts` failed the guard at `91.3` statements, `75` branches, `100` functions, and `91.3` lines with uncovered lines `50` and `94`. Those gaps map to the explicit `conditional` rejection path and the missing-`fineTuneOptions` guard-rail path, so the validation reroute goes back to Step 03 for the smallest owner-local test additions before Step 05 is rerun.

- Step 05 gate decision: do not advance to Step 06 yet.
- Exact next orchestrator: `03-red-testing`
- Updated Step 05 rerun handoff: add the smallest owner-local follow-up cases in `src/neat/hybrid/neat.hybrid.test.ts` for (1) `fineTune: 'conditional'` rejecting with the deterministic-ranking blocker and (2) `fineTune !== 'never'` without `fineTuneOptions` rejecting before training runs. After that Step 03 follow-up, rerun the same Step 05 sequence: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts`, `npx tsc --noEmit -p tsconfig.json`, and the focused coverage command above.
- Tracker validation: `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` passed with `0` errors and `0` warnings after this Step 05 tracker refresh.
- Scope guard: this validation pass stayed inside the Phase 4 hybrid-policy seam only; no production code, worker-path behavior, Phase 5 docs, or ONNX fixes changed.

**Step 03 owner-local red follow-up after the Step 05 coverage reroute:** Added the two smallest guard-rail tests to `src/neat/hybrid/neat.hybrid.test.ts` and kept the follow-up test-only. One case requests `fineTune: 'conditional'` with valid `fineTuneOptions` and locks the explicit deterministic-ranking blocker. The other requests `fineTune: 'always'` without `fineTuneOptions` and locks the missing-options rejection before training runs. No production code, worker-path behavior, Phase 5 docs, or ONNX fixes changed.

- Changed test file: `src/neat/hybrid/neat.hybrid.test.ts`
- Focused behavior follow-up: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts` passed with `1` suite and all `5` targeted `evaluateCandidate` tests green.
- Focused coverage follow-up: `npx jest --config=jest.config.mjs --no-cache --coverage --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts --collectCoverageFrom='src/neat/hybrid/neat.hybrid.ts' --coverageReporters=text --coverageReporters=json-summary` passed with `src/neat/hybrid/neat.hybrid.ts` at `100/100/100/100`.
- Updated Step 05 rerun handoff: call `05-green-testing` next. Re-run `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts`, then `npx tsc --noEmit -p tsconfig.json` while treating the unchanged ONNX `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` as the external baseline unless the failure surface moves into `src/neat/hybrid/`. The owner-local hybrid coverage rerun is already green; reopen Step 03 only if a new hybrid coverage gap appears.
- Scope guard: this follow-up stayed inside the Phase 4 hybrid-policy seam only; no production code, worker-path behavior, Phase 5 docs, or ONNX fixes changed.

**Step 05 rerun after the Step 03 coverage follow-up from current repo state:** Re-ran the required focused hybrid-policy slice with `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts`. The Phase 4 behavior gate stayed green again: `PASS` for `src/neat/hybrid/neat.hybrid.test.ts`, `1` suite passed, and all `5` targeted `evaluateCandidate` tests passed for `never`, the explicit `conditional` blocker, the missing-`fineTuneOptions` guard, fitness-only `always`, and Lamarckian `always` behavior.

Re-ran `npx tsc --noEmit -p tsconfig.json`. Repo-wide TypeScript still does not clear, but the failure surface remains unchanged: the only error is the recorded external ONNX baseline at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` with `TS2345`, because `standardDomainImports[0]!.version` can be `undefined`. No new TypeScript failure appeared inside `src/neat/hybrid/`, so this rerun treats the ONNX error as external baseline context rather than a Phase 4 blocker.

Re-ran the focused coverage guard with `npx jest --config=jest.config.mjs --no-cache --coverage --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts --collectCoverageFrom='src/neat/hybrid/neat.hybrid.ts' --collectCoverageFrom='src/neat/hybrid/neat.hybrid.types.ts' --coverageReporters=text --coverageReporters=json-summary`. The text report and `coverage/coverage-summary.json` count only the runtime-bearing `src/neat/hybrid/neat.hybrid.ts` file, which now passes at `100/100/100/100` with `23/23` lines, `23/23` statements, `4/4` functions, and `8/8` branches covered. `src/neat/hybrid/neat.hybrid.types.ts` remains a type-only file with no runtime statements, so Jest does not count it in coverage totals.

- Tracker validation: `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` passed with `0` errors and `0` warnings after this Step 05 rerun tracker refresh.
- Step 05 gate decision: close Step 05 and hand off to Step 06 documentation.
- Exact next orchestrator: `06-documenting`
- Scope guard: this rerun stayed inside the Phase 4 hybrid-policy seam only; no production code, worker-path behavior, Phase 5 docs, or ONNX fixes changed.

#### Step 06: Curate hybrid-policy docs [DONE]

```yaml
phase: 4
step: 6
agent: '06-documenting'
agent_file: '.github/agents/06-educational-docs-curator.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 07 — Log Phase 4 closure or reroute'
skills:
  - 'educational-docs'
  - 'docs-academic-citation-audit'
  - 'hybrid-training-interop'
specialists:
  - 'Docs Scout'
validation:
  - 'Source-first docs for all exported policy symbols are present, generated docs are refreshed if JSDoc inputs changed, and no Phase 5 example constructs or Lamarckian-is-default language appears.'
```

**User instruction:** Start a fresh session, select `06-documenting`, and paste this full step packet.

**Step objective:** Keep the policy surface teachable and synchronized with generated docs without turning Phase 4 into the broader Phase 5 documentation closure.

**Context the agent must know:**

- Documentation scope is limited to `HybridEvaluationPolicy`, the policy evaluation helper, the three policy choices (`never`, `always`, `conditional`), the `persistTrainedWeights: false` safe-default note, and the same-runtime determinism wording for the conditional path.
- Phase 5 content (Lamarckian deep-dive, reproducibility ladder, toy-dataset example) remains out of scope.
- Generated `src/**/README.md` files and `docs/examples/**` outputs are read-only; update source JSDoc and run generation when needed.
- Public docs should be atemporal: no tracker phases, roadmap chronology, or chat-only context.

**Execution steps:**

1. Read Step 04 changed public surfaces and Step 05 validation evidence.
2. Improve only the source-owned JSDoc or nearby handwritten docs required by the new policy surface.
3. Run `npm run docs` if JSDoc or generated-doc inputs changed; otherwise record a manual no-docs-needed rationale.
4. Verify that no generated README or published example output was hand-edited as a shortcut.
5. Update this plan with documentation evidence and any residual docs risk before ending the session.

**Stop conditions:**

- Done: required source docs are aligned and docs generation either passes or is explicitly not needed.
- Blocked: public naming or determinism wording is unstable enough that docs would mislead downstream consumers.
- Route back: documentation reveals implementation drift in the policy contract; route to Step 04 or Step 05 rather than widening docs.

**Required validation:** `npm run docs` when source JSDoc or generated-doc inputs changed, or a manual no-docs-needed record when implementation introduced no doc-facing surface.

**Plan update requirement:** Update this plan with documentation changes, validation evidence, residual gaps, and the Step 07 closure handoff before ending the session.

**Step 06 documentation changes and validation evidence:** Reviewed source-owned JSDoc for `HybridFineTuneMode`, `HybridEvaluationPolicy`, `HybridScoreNetwork`, `EvaluateCandidateOptions`, `HybridEvaluationResult` in `src/neat/hybrid/neat.hybrid.types.ts` and for `evaluateCandidate` in `src/neat/hybrid/neat.hybrid.ts`. The generated `src/neat/hybrid/README.md` did not yet exist because this was the first `npm run docs` pass for the new hybrid module. Source JSDoc was improved as follows, keeping all changes Phase 4 scoped and atemporal:

- `HybridFineTuneMode`: expanded description to name all three modes explicitly and clarify that `conditional` requires a deterministic ranking or tie-break contract at the evaluation boundary and throws at runtime until that surface exists.
- `HybridEvaluationResult`: added a downstream-use note listing typical caller patterns — forwarding fitness to the NEAT population score, comparing trained weights against the original candidate to measure fine-tune delta, checkpointing the trained snapshot, or discarding the result when only fitness matters.
- `evaluateCandidate`: added an `@example` fenced code block showing fitness-only (persistTrainedWeights: false) and Lamarckian (persistTrainedWeights: true) usage patterns with a minimal XOR-shaped dataset and an explicit seed.

- Changed source files: `src/neat/hybrid/neat.hybrid.types.ts`, `src/neat/hybrid/neat.hybrid.ts`
- Validation command: `npm run docs` — completed successfully with `HTML docs generated.`
- Generated output created: `src/neat/hybrid/README.md` now includes `HybridFineTuneMode`, `HybridEvaluationPolicy`, `HybridScoreNetwork`, `EvaluateCandidateOptions`, `HybridEvaluationResult`, and `evaluateCandidate` with the correct policy wording, `persistTrainedWeights: false` safe-default note, `conditional` ranking-requirement note, downstream-use note, and the full `@example` block.
- Focused behavior rerun: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts` passed with `1` suite and all `5` targeted tests green.
- Source-of-truth guard: no generated `src/**/README.md` file or `docs/examples/**` output was hand-edited as a shortcut. The only manual edits were to the two source JSDoc files, and the README was created by `npm run docs`.
- Scope guard: no Phase 5 example constructs, Lamarckian deep-dive, reproducibility ladder, or tracker/roadmap language was introduced in public docs.
- Citation note: no external citation required; the policy surface is repo-owned.
- Tracker validation: `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` passed with `0` errors and `0` warnings after this Step 06 tracker update.
- Residual docs risk: none within the Phase 4 hybrid-policy seam. Step 07 should decide Phase 4 closure against the preserved Phase 4 acceptance evidence and the unchanged external ONNX TypeScript baseline.

#### Step 07: Log Phase 4 closure or reroute [DONE]

```yaml
phase: 4
step: 7
agent: '07-logging'
agent_file: '.github/agents/07-session-log-archivist.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Phase 5 Step 01 — Plan the docs-and-example closure'
skills:
  - 'tracker-handoff'
  - 'plan-sync-validation'
  - 'hybrid-training-interop'
specialists:
  - 'Plan Registration Auditor'
validation:
  - 'Manual tracker review confirms Phase 4 acceptance evidence, Lamarckian persistence is documented as explicit opt-in, and the refreshed Handoff query points to Phase 5 Step 01 before the next phase opens.'
```

**User instruction:** Start a fresh session, select `07-logging`, and paste this full step packet.

**Step objective:** Close or reroute Phase 4 based on validated evidence, preserve the three-policy-decision split with Lamarckian as explicit opt-in only, and leave the tracker ready for Phase 5 docs-and-examples planning.

**Context the agent must know:**

- Phase 4 closes only when users can demonstrably choose fitness-only fine-tuning or Lamarckian persistence, and Lamarckian persistence is provably gated behind `persistTrainedWeights: true`.
- Phase 5 should not open until the policy surface is stable, tested, and covered.
- The overall lane remains active; do not archive this tracker unless the full hybrid-interoperability workstream is done.
- Chat is not the source of truth; the tracker and the `Handoff query` must carry the next session.

**Execution steps:**

1. Read the Phase 4 Step 02-06 evidence and verify that no required validation is missing.
2. Mark completed Phase 4 steps `[DONE]` or record the smallest reroute if evidence is incomplete.
3. Record that the policy surface is now real and that Phase 5 docs closure can begin.
4. Refresh the `Handoff query` to point either to Phase 5 Step 01 or to the smallest blocking reroute step.
5. Run `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` after the tracker refresh.

**Stop conditions:**

- Done: Phase 4 evidence is compactly recorded, the next safe step is explicit, and the handoff query is current.
- Blocked: validation, documentation, or policy ownership evidence is missing and cannot be reconstructed from the tracker.
- Route back: missing red coverage routes to Step 03, implementation drift routes to Step 04, failed validation routes to Step 05, and doc drift routes to Step 06.

**Required validation:** Manual tracker diff review confirming Phase 4 acceptance evidence, no implicit Lamarckian default, and a current next-step handoff.

**Plan update requirement:** Update this plan with Phase 4 closure or reroute state, refreshed handoff query, and the next active step before ending the session.

**Step 07 closure review and handoff evidence:** Step 07 reviewed the recorded Phase 4 evidence from Step 02 through Step 06 from current repo state only. Phase 4 can now close. The hybrid-policy seam satisfies the phase acceptance gate: Step 02 confirmed the `src/neat/hybrid/` owner boundary, the standalone `evaluateCandidate(...)` hook, dataset compatibility with `TrainingSample[]`, the worker-ordering constraint, and the explicit conditional-policy blocked note; Step 03 added owner-local tests for `never`, fitness-only `always`, Lamarckian opt-in `always`, and later the explicit `conditional` blocker plus missing-`fineTuneOptions` guard; Step 04 implemented `evaluateCandidate`, `HybridFineTuneMode`, `HybridEvaluationPolicy`, `EvaluateCandidateOptions`, and `HybridEvaluationResult` in `src/neat/hybrid/` while consuming Phase 3 `fineTuneVector(...)`; Step 05 revalidated the focused behavior slice with `1` suite and `5` tests green, confirmed the unchanged external ONNX `TS2345` as the only repo-wide TypeScript baseline failure, and passed coverage guard at `100/100/100/100` for `src/neat/hybrid/neat.hybrid.ts` while keeping `src/neat/hybrid/neat.hybrid.types.ts` as a type-only non-runtime coverage surface; and Step 06 improved source-first JSDoc and refreshed `src/neat/hybrid/README.md` via `npm run docs` without hand-editing generated output.

- Phase 4 closure decision: close Phase 4. No reroute is required because the owner boundary, red coverage, implementation, focused validation, and generated-doc refresh are all durably recorded.
- Policy decision: users can now choose three explicit policy states at the helper surface: no fine-tune with `fineTune: 'never'`, fitness-only fine-tune with `fineTune: 'always'` plus `persistTrainedWeights: false`, and Lamarckian persistence only on explicit opt-in with `fineTune: 'always'` plus `persistTrainedWeights: true`.
- Conditional-policy decision: `fineTune: 'conditional'` remains intentionally blocked pending a deterministic ranking or tie-break surface at the evaluation boundary; Step 04 and Step 05 keep that blocker explicit instead of inventing unstable policy semantics.
- Tracker state decision: keep the top-level tracker active in `plans/` as `[WIP]`; mark Phase 4 complete; activate Phase 5 as the single `[WIP]` phase; do not archive this tracker or create a same-boundary `.logs.md` file because the docs-and-examples phase is still real work.
- Next safe step: activate `01-planning` for Phase 5 `Step 01 — Plan the docs-and-example closure`. Keep the next pass planning-only; do not widen into Phase 5 implementation, unrelated ONNX fixes, or worker-path policy extensions in this closure step.

### Phase 5 — Docs and examples [WIP]

**Phase objective:** Close the lane with source-first documentation and one small example so downstream consumers can understand the determinism scope, isolation semantics, and persistence policy without inventing private conventions.

**Phase progression rule:** Start with only Step 01. Step 01 must author the remaining numbered step packets, or explicit skipped-step packets, before the phase can advance.

**Preserved implementation contract:**

- Document:
  - what “Lamarckian” means in this library,
  - reproducibility considerations,
  - recommended defaults.
- Explain the determinism ladder used by this lane: same-runtime ordered deterministic versus replay exact versus best-effort reproducible.
- Include one tiny example on a toy dataset that exercises vector export/import and isolated fine-tuning without introducing a large training framework.
- Refresh generated docs after public JSDoc changes so downstream README surfaces stay synchronized.

**Acceptance when this phase closes:**

- Docs contain a small example and clear warnings.

**Focused validation to preserve:**

- The example stays aligned with the real public API.
- `npm run docs` succeeds after the doc-facing pass.

#### Step 01: Plan the docs-and-example closure [PLANNED]

```yaml
phase: 5
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning-architect.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 02 — Research doc-and-example scope'
skills:
  - 'educational-docs'
specialists:
  - 'Docs Scout'
validation:
  - 'Manual tracker check that the determinism wording, Lamarckian explanation, and example-validation expectations are preserved before Step 02-07 packets are authored.'
```

**User instruction:** Start a fresh session, select `01-planning`, and paste this full step packet.

**Step objective:** Turn Phase 5 into an executable documentation closure workset by authoring Phase 5 Step 02-07 packets, preserving the determinism ladder and example expectations, and keeping generated docs aligned with source-owned public contracts.

**Context the agent must know:**

- Phase 5 closes the lane only after the public naming, determinism wording, and example surface are stable enough for downstream consumers.
- Documentation must stay source-first and should refresh generated docs rather than hand-edit generated README surfaces.
- The example should stay tiny, use the real public API, and teach vector export/import plus isolated fine-tuning without implying a full training framework.
- The docs pass should teach the distinction between fitness-only and Lamarckian persistence clearly enough for downstream consumers such as NEATchat.

**Execution steps:**

1. Reconfirm the preserved Phase 5 contract, acceptance rule, and focused validation against the current plan text.
2. Decide which phase-agent steps are real value gates for the documentation closure and which should be explicit skipped-step packets.
3. Author Phase 5 Step 02-07 packets or explicit skipped-step packets with the narrowest honest doc-and-example scope.
4. Record the doc refresh and example-validation expectations directly in the plan before naming the next active step.

**Stop conditions:**

- Done: Phase 5 has a paste-ready next active step and all remaining step slots are authored or explicitly skipped.
- Blocked: unresolved public contract naming or determinism wording prevents an honest docs closure packet.
- Route back: if planning reveals unresolved upstream contract drift, update the tracker and return to planning instead of pretending the docs phase can close the lane early.

**Required validation:** Manual diff review confirming that every former Step 5 detail now lives in Phase 5 and that the generated-doc refresh expectation remains explicit.

**Plan update requirement:** Update this plan with the new Step 02-07 packets, any skipped-step rationale, validation evidence, and the next active step before ending the session.

**Planning decisions and validation evidence:** Step 01 preserved the Phase 5 contract, acceptance rule, and focused validation from the current plan text. Phases 1 through 4 are all [DONE] and supply the stable public API surfaces that Phase 5 must document clearly for downstream consumers. The unchanged ONNX `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` remains the recorded external baseline and is not a Phase 5 blocker. `plans/README.md` and `plans/Roadmap.md` confirm this lane is the last non-chat foundation handoff before the dependency-gated NEATchat follow-up; no roadmap conflict exists for a documentation-and-example closure pass.

- Value-gate decision: Research (Step 02), Authoring (Step 04), and Green Validation (Step 05) are real value gates. Session Logging and Lane Closure (Step 07) is a real gate because it must compress, archive, and update the plans index.
- Skipped-step decision: Step 03 (Red Testing) and Step 06 (Documentation) are explicit skips for Phase 5.
  - Step 03 skip: Phase 5 introduces no behavioral change. The acceptance gate — example aligns with the real public API and `npm run docs` succeeds — is a green-validation concern confirmed in Step 05. No failing test is an honest gate for a documentation-only closure pass.
  - Step 06 skip: the entire phase is a documentation pass. Step 04 is the authoring step and Step 05 is the validation gate. A separate documentation step would be a redundant no-op in this phase structure.
- Conditional policy carry-forward: `fineTune: 'conditional'` remains intentionally blocked pending deterministic ranking semantics as decided in Phase 4. Phase 5 must document that blocked state honestly without implementing it or pretending it is resolved.
- Example placement: Step 02 research should confirm whether the cohesive workflow example belongs in a JSDoc `@example` block on an existing public-facing function or in a small standalone `examples/` file following the existing examples naming convention. Given that Phase 3 and Phase 4 already have `@example` blocks on individual helper functions, Phase 5 likely needs a cohesive end-to-end snippet (export → fine-tune → compare or import) either as a module-level overview comment on the hybrid boundary or as a small examples file if one proves more readable.
- Next active step: Phase 5 Step 02 — Research doc-and-example scope.

#### Step 02: Research doc-and-example scope [DONE]

```yaml
phase: 5
step: 2
agent: '02-researching'
agent_file: '.github/agents/02-research-coordinator.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 04 — Author doc improvements and example'
skills:
  - 'educational-docs'
  - 'docs-academic-citation-audit'
specialists:
  - 'Docs Scout'
validation:
  - 'Read-only research brief names which source JSDoc surfaces need the Lamarckian explanation, determinism ladder, and recommended-defaults language, confirms example placement, and identifies any citation gap.'
```

**User instruction:** Start a fresh session, select `02-researching`, and paste this full step packet.

**Step objective:** Gather the smallest read-only evidence needed to scope the Phase 5 doc-authoring pass without widening into new API surfaces, behavioral changes, or unrelated ONNX fixes.

**Context the agent must know:**

- Phase 5 introduces no behavioral change. The only work is improved source JSDoc plus one cohesive toy-dataset workflow example.
- The three main public API boundaries from this lane are: `src/architecture/network/serialize/` (parameter vector), `src/architecture/network/training/` (isolation helper), and `src/neat/hybrid/` (evaluation policy).
- The conditional policy blocker (`fineTune: 'conditional'`) must appear in documentation honestly, not as an undocumented omission.
- Carry the unchanged ONNX `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` as external baseline context.

**Execution steps:**

1. Read `plans/README.md`, `plans/Roadmap.md`, this Phase 5 section, and the current generated `src/neat/hybrid/README.md`, `src/architecture/network/serialize/README.md`, and `src/architecture/network/training/README.md` for documentation coverage gaps.
2. Delegate a focused read-only packet to `Docs Scout` to identify: which source JSDoc surfaces need the Lamarckian explanation, determinism ladder wording, and recommended-defaults language, and whether any of those explanations are missing or misleading in the current generated READMEs.
3. Confirm the example placement decision: whether the cohesive workflow example belongs in a new `@example` JSDoc block on an existing public-facing function, in a synthesized module-level overview comment, or in a standalone `examples/` file following the existing examples naming convention.
4. Record any citation or attribution gaps for Lamarckian evolution semantics that Phase 5 should address.
5. Record the exact scope for Step 04 before ending the session.

**Stop conditions:**

- Done: research brief names the JSDoc surfaces to improve, confirms example placement, and records any citation expectations.
- Blocked: source JSDoc surfaces or generated README outputs are inaccessible or structurally inconsistent.
- Route back: research shows a named Phase 5 documentation surface requires an upstream API change; update the tracker and return to planning instead of widening implementation.

**Required validation:** Manual evidence review confirming the research remains read-only and boundary-local.

**Plan update requirement:** Update this plan with the research brief, example placement decision, citation notes, and the Step 04 authoring handoff before ending the session.

**Research brief and validation evidence (Step 02 — [DONE]):**

- **Roadmap and README alignment:** `plans/README.md` and `plans/Roadmap.md` still align this lane as the next non-chat foundation handoff after the archived ONNX baseline, with no new roadmap conflict for a documentation-only closure pass. Phases 1–4 are each [DONE] and supply the stable public API surfaces that Phase 5 must document.

- **Generated README current state:**
  - `src/neat/hybrid/README.md` — Uses "Lamarckian" in passing (safe-default note, example comment) but has no dedicated explanation of what Lamarckian means in this library, no determinism ladder, and no recommended-defaults progression. The `evaluateCandidate` JSDoc has a combined fitness-only-plus-Lamarckian `@example` block but not a cohesive end-to-end workflow example (export → fine-tune → compare → decide).
  - `src/architecture/network/serialize/README.md` — Contains "does not claim cross-runtime exact replay" on `ParameterLayoutV1` and `ParameterVector` but has no Lamarckian mention, no determinism ladder, and no recommended defaults. That boundary is well-documented for its own contracts.
  - `src/architecture/network/training/README.md` — Contains "does not claim cross-runtime exact replay" on `fineTuneVector` but has no Lamarckian mention, no determinism ladder, and no recommended defaults. That boundary is well-documented for its own contracts.

- **Doc gap location decision:** The three Phase 5 required items — Lamarckian explanation, determinism ladder, and recommended defaults — are most naturally cohesive at the **hybrid boundary** (`src/neat/hybrid/neat.hybrid.ts`) because the hybrid policy is the highest-level surface that unifies all three lane concerns. The correct vehicle is a **module-level JSDoc overview paragraph** at the top of `neat.hybrid.ts`, above the `evaluateCandidate` function, that teaches the conceptual model before the API surface. `HybridEvaluationPolicy` in `neat.hybrid.types.ts` should also receive a tighter recommended-defaults progression note in its existing JSDoc.

- **Phase 5 item placement:**
  1. **Lamarckian explanation** — In the `neat.hybrid.ts` module-level JSDoc overview: define Lamarckian as trained weights propagating back to the parent candidate (`persistTrainedWeights: true`). Note why the default is fitness-only (safer, preserves evaluation–population separation) and distinguish it from the Baldwin effect (phenotypic improvement without genotypic change).
  2. **Determinism ladder** — In the same `neat.hybrid.ts` module-level overview: three rungs — (a) same-runtime ordered deterministic when topology, dataset order, training settings, and explicit `seed` all match; (b) best-effort reproducible when no explicit `seed` is provided; (c) cross-runtime exact replay is out of scope for this lane.
  3. **Recommended defaults** — In both the `neat.hybrid.ts` module-level overview and the `HybridEvaluationPolicy` JSDoc in `neat.hybrid.types.ts`: start with `fineTune: 'never'`, switch to `fineTune: 'always'` with `persistTrainedWeights: false` for exploratory scoring, and opt into `persistTrainedWeights: true` only when explicit Lamarckian persistence is intentional.
  4. **Conditional policy blocker** — Already in existing JSDoc, but should be surfaced in the module-level overview as well so it is visible before the API table.

- **Example placement decision:** The cohesive toy-dataset workflow example (export parameter vector → fine-tune → compare trained vs. original → decide whether to import) belongs as a **module-level `@example` block in `src/neat/hybrid/neat.hybrid.ts`** rather than as a standalone `examples/` folder entry. Reason: none of the three new lane surfaces (`toParameterVector`, `fromParameterVector`, `fineTuneVector`, `evaluateCandidate`, and their types) are currently re-exported from `src/neataptic.ts`. A standalone `examples/hybridTraining/` file that must use internal import paths would not satisfy the "real public API" criterion from the Phase 5 contract, and adding re-exports to `neataptic.ts` is an API surface extension that needs a coverage guard pass and is outside the documentation-only scope. A JSDoc `@example` block is honest about its module path, feeds the generated README, and stays source-owned.

- **Public API re-export gap (decision: note and defer):** `toParameterVector`, `fromParameterVector`, `createParameterLayoutV1`, `ParameterVector`, `ParameterLayoutV1`, `ParameterLayoutEntry`, `fineTuneVector`, `FineTuneOptions`, `FineTuneResult`, `evaluateCandidate`, `HybridFineTuneMode`, `HybridEvaluationPolicy`, `HybridScoreNetwork`, `EvaluateCandidateOptions`, and `HybridEvaluationResult` are not re-exported from `src/neataptic.ts`. Phase 5 Step 04 should include adding these re-exports as a bounded scope extension — it is safe for a documentation closure phase because the underlying implementations already exist and are tested. The re-export addition must be followed by coverage guard on `src/neataptic.ts` after the change.

- **Citation gap:** A Wikipedia link to [Lamarckism](https://en.wikipedia.org/wiki/Lamarckism) is recommended in the module-level overview alongside the Baldwin effect distinction. No peer-reviewed citation is required because "Lamarckian fine-tuning" in neuroevolution is well-established vocabulary and the implementation semantics are repo-owned. The `educational-docs` skill prefers external links for grounding; a short Wikipedia attribution satisfies that requirement here.

- **ONNX `TS2345` baseline:** Unchanged at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135`. Not a Phase 5 blocker.

**Step 04 authoring handoff:** Improve source JSDoc for the three Phase 5 required items at the hybrid boundary and add bounded re-exports to `src/neataptic.ts`. Specific scope:

1. Add a module-level JSDoc overview paragraph in `src/neat/hybrid/neat.hybrid.ts` covering: (a) Lamarckian definition plus Baldwin-effect distinction plus Wikipedia link, (b) three-rung determinism ladder, (c) recommended-defaults progression (`never` → fitness-only `always` → Lamarckian `always`), (d) conditional blocker note. Include a cohesive `@example` block covering the full workflow: export vector, fine-tune, compare, and decide whether to import back.
2. Add a recommended-defaults progression note to the `HybridEvaluationPolicy` JSDoc in `src/neat/hybrid/neat.hybrid.types.ts`.
3. Add re-exports for all three lane boundaries to `src/neataptic.ts`.
4. Run `npm run docs` after all source JSDoc changes and confirm generated READMEs are updated.
5. Run coverage guard on `src/neataptic.ts` if touched.
   Do not introduce behavioral changes, new training helpers, or Phase 7/NGE content.

#### Step 03: SKIP — No behavioral change in Phase 5 [DONE]

```yaml
phase: 5
step: 3
status: '[DONE]'
skip_reason: 'Phase 5 introduces no behavioral change. The acceptance gate — example aligns with the real public API and npm run docs succeeds — is a green-validation concern confirmed in Step 05. No failing test is an honest gate for a documentation-only closure pass.'
```

No red test is authored in Phase 5. The green acceptance gate is Step 05.

#### Step 04: Author doc improvements and example [DONE]

```yaml
phase: 5
step: 4
agent: '04-implementing'
agent_file: '.github/agents/04-implementation-architect.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 05 — Validate doc-and-example gates'
skills:
  - 'educational-docs'
  - 'docs-academic-citation-audit'
  - 'hybrid-training-interop'
specialists:
  - 'Docs Scout'
validation:
  - 'All three Phase 5 contract items are present: Lamarckian explanation, determinism ladder, recommended defaults, and one cohesive toy-dataset example. Generated docs are refreshed. No behavioral change is introduced.'
```

**User instruction:** Start a fresh session, select `04-implementing`, and paste this full step packet.

**Step objective:** Author the three required Phase 5 documentation items across the relevant source JSDoc surfaces — Lamarckian explanation, determinism ladder, and recommended defaults — plus one cohesive toy-dataset workflow example at the location confirmed by Step 02.

**Context the agent must know:**

- This step introduces no behavioral change. All edits are source-owned JSDoc improvements or `@example` blocks.
- The three Phase 5 items to deliver:
  1. **Lamarckian explanation**: what "Lamarckian" means here — trained weights propagating back to the parent candidate — and why the default is fitness-only (no persistence) rather than Lamarckian.
  2. **Determinism ladder**: the three rungs this lane uses — (a) same-runtime ordered deterministic when topology, dataset order, training settings, and seed match, (b) best-effort reproducible when no explicit seed is provided, and (c) cross-runtime exact replay is out of scope.
  3. **Recommended defaults**: start with `fineTune: 'never'`, switch to `fineTune: 'always'` with `persistTrainedWeights: false` for exploratory scoring, and opt in to `persistTrainedWeights: true` only for explicit Lamarckian persistence.
- The toy-dataset example must use real public API, cover the full workflow (export parameter vector, call `fineTuneVector`, compare or import), and use a tiny XOR-shaped or similar dataset without importing a large training framework.
- The conditional policy blocker (`fineTune: 'conditional'`) must appear in documentation as an honest note, not as an undocumented omission.
- After any source JSDoc change, run `npm run docs` to refresh generated READMEs. Do not hand-edit generated README files.

**Execution steps:**

1. Read the Step 02 research brief, the Phase 5 contract, and the current source JSDoc for the three main boundaries.
2. Add or improve JSDoc for the Lamarckian explanation, determinism ladder, and recommended defaults at the source surfaces confirmed by Step 02.
3. Add the cohesive toy-dataset workflow example at the location confirmed by Step 02.
4. Add any missing academic citation for Lamarckian evolution semantics if Step 02 identified a citation gap.
5. Run `npm run docs` after all source JSDoc changes and confirm the generated READMEs are updated without hand-editing.
6. Update this plan with changed files, validation evidence, and the Step 05 handoff before ending the session.

**Stop conditions:**

- Done: all three Phase 5 contract items are documented, the example is present at the confirmed location, and `npm run docs` passes.
- Blocked: a required source surface cannot be improved without a behavioral API change.
- Route back: example placement requires a new `examples/` file structure that needs a prior architecture decision; update the plan and return to Step 02.

**Required validation:** `npm run docs` passes after all JSDoc changes, and no generated README or published example page is hand-edited as a shortcut.

**Plan update requirement:** Update this plan with changed source files, doc items confirmed, `npm run docs` result, and the Step 05 handoff before ending the session.

**Step 04 implementation and validation evidence:** Added the Phase 5 closure docs at the hybrid boundary and the bounded public facade gap needed to make the example honest. `src/neat/hybrid/neat.hybrid.ts` now opens with a module-level overview that explains Lamarckian persistence versus the Baldwin effect with Wikipedia grounding, states the three-rung determinism ladder, recommends the default progression `fineTune: 'never'` -> `fineTune: 'always'` fitness-only -> explicit Lamarckian opt-in, surfaces the `conditional` blocker before the API table, and includes one cohesive XOR-shaped public API example that uses `toParameterVector`, `fineTuneVector`, `fromParameterVector`, and `evaluateCandidate` from `neataptic`. `src/neat/hybrid/neat.hybrid.types.ts` now adds the same recommended-defaults progression to `HybridEvaluationPolicy`.

`src/neataptic.ts` now re-exports the example-facing lane surfaces and types: `toParameterVector`, `fromParameterVector`, `ParameterLayoutEntry`, `ParameterLayoutV1`, `ParameterVector`, `fineTuneVector`, `FineTuneOptions`, `FineTuneResult`, `evaluateCandidate`, `HybridFineTuneMode`, `HybridEvaluationPolicy`, `HybridScoreNetwork`, `EvaluateCandidateOptions`, and `HybridEvaluationResult`. `createParameterLayoutV1` remains on the serialize boundary because the public workflow example does not require it and Step 04 stayed bounded to the docs-and-example closure surface rather than widening the root facade beyond the honest example path.

- Changed source files: `src/neataptic.ts`, `src/neat/hybrid/neat.hybrid.ts`, `src/neat/hybrid/neat.hybrid.types.ts`
- Generated output refreshed by `npm run docs`: `src/neat/hybrid/README.md`
- Validation command: `npm run docs` completed successfully; the docs pass regenerated per-folder READMEs and progressed into the HTML render stage without reporting a Phase 5 failure.
- Narrow diagnostics: `get_errors` reported no errors in `src/neataptic.ts`, `src/neat/hybrid/neat.hybrid.ts`, or `src/neat/hybrid/neat.hybrid.types.ts` after the edit.
- Source-of-truth guard: no generated README or published example page was hand-edited as a shortcut.
- Residual risks: Step 05 still needs to confirm the generated hybrid README carries all three Phase 5 contract items plus the `conditional` blocker note, that the example-facing root exports remain aligned with `src/neataptic.ts`, and that the existing root-facade coverage slice still covers the touched entrypoint. The unchanged ONNX `TS2345` baseline at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` remains external context.

**Step 05 validation handoff:** Treat `src/neataptic.ts`, `src/neat/hybrid/neat.hybrid.ts`, and `src/neat/hybrid/neat.hybrid.types.ts` as the touched source boundary and `src/neat/hybrid/README.md` as the generated output to inspect. Confirm the example imports are real from `src/neataptic.ts`, rerun `npm run docs` only if output drift is suspected, inspect the generated README for the Lamarckian explanation, the determinism ladder, the recommended defaults, and the `conditional` blocker note, and run the existing root-facade coverage slice against `src/neataptic.ts` because Step 04 added public re-exports there: `npx jest --config=jest.config.mjs --no-cache --coverage --runTestsByPath src/neataptic.test.ts --collectCoverageFrom='src/neataptic.ts' --coverageReporters=text --coverageReporters=json-summary`. If the example references a missing symbol or the generated README drops one of the Phase 5 items, route back to Step 04. If the root-facade coverage slice misses the new export lines, route back to Step 04 only for the smallest root export alignment or test-slice change.

#### Step 05: Validate doc-and-example gates [DONE]

```yaml
phase: 5
step: 5
agent: '05-green-testing'
agent_file: '.github/agents/05-green-validation-runner.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Step 07 — Log Phase 5 closure and archive lane'
skills:
  - 'green-validation-gates'
  - 'coverage-guard'
  - 'educational-docs'
  - 'hybrid-training-interop'
specialists:
  - 'Docs Scout'
validation:
  - 'The generated hybrid README contains the Lamarckian explanation, Baldwin-effect distinction with Wikipedia grounding, the determinism ladder, recommended defaults, the conditional blocker, and a cohesive public API example; the focused hybrid Jest slice passes; npm run docs passes; TypeScript shows only the unchanged ONNX baseline; and focused coverage stays at 100 for src/neat/hybrid/neat.hybrid.ts and src/neataptic.ts.'
```

**User instruction:** Start a fresh session, select `05-green-testing`, and paste this full step packet.

**Step objective:** Confirm the Phase 5 documentation closure satisfies its acceptance gates without widening scope: the example uses real public API, the generated hybrid README contains every required Phase 5 item, docs regeneration still succeeds, the touched runtime file under `src/neat/hybrid/` keeps full coverage, and the touched root facade under `src/neataptic.ts` keeps full coverage after the new re-exports.

**Context the agent must know:**

- Phase 5 introduces no intended behavioral change, but Step 04 touched `src/neataptic.ts`, `src/neat/hybrid/neat.hybrid.ts`, and `src/neat/hybrid/neat.hybrid.types.ts`, so Step 05 still needs one focused hybrid behavior rerun plus owner-local coverage gates on the touched runtime files.
- The acceptance gates are: (1) example aligns with the real public API — imports exist, function signatures match, no imaginary helpers; (2) `src/neat/hybrid/README.md` includes the Lamarckian explanation, Baldwin-effect distinction with Wikipedia grounding, the determinism ladder, recommended defaults, the conditional blocker or ranking requirement, and the cohesive public API toy example; (3) `npm run docs` passes and no generated README or published example page is hand-edited as a shortcut; (4) `src/neat/hybrid/neat.hybrid.ts` stays at `100/100/100/100`; and (5) `src/neataptic.ts` stays at `100/100/100/100` after the new re-exports.
- Treat `src/neat/hybrid/neat.hybrid.types.ts` as a type-only non-runtime coverage surface, consistent with the earlier Phase 4 Step 05 ruling.
- Carry the unchanged ONNX `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` as external baseline context unless a new failure appears inside the Phase 5 surfaces.

**Execution steps:**

1. Read the Step 04 changed-file summary and confirm the required Phase 5 contract items.
2. Verify the example imports and function calls against the real root facade by checking that the referenced symbols are exported from `src/neataptic.ts`.
3. Re-run the focused hybrid behavior slice: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts`.
4. Re-run `npm run docs` to confirm generated output stays synchronized after the source-owned JSDoc pass.
5. Run `npx tsc --noEmit -p tsconfig.json`, treating the unchanged ONNX `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` as external only if no new failure appears in the Phase 5 surfaces.
6. Run focused coverage on `src/neataptic.ts` via `src/neataptic.test.ts` and on `src/neat/hybrid/neat.hybrid.ts` via `src/neat/hybrid/neat.hybrid.test.ts`.
7. Update this plan with pass or fail evidence and route failures to Step 04 or Step 02 as appropriate before ending the session.

**Stop conditions:**

- Done: all acceptance gates pass and the plan is updated with validation evidence.
- Blocked: `npm run docs` fails for a reason unrelated to Phase 5 source changes.
- Route back: missing or incorrect exports, missing README closure items, failed hybrid behavior rerun, or focused coverage loss on `src/neataptic.ts` or `src/neat/hybrid/neat.hybrid.ts` returns to Step 04 for the smallest bounded follow-up. Return to Step 02 only if the Phase 5 requirement itself proves to need a different documented placement or public-contract framing than Step 04 assumed.

**Required validation:** README content check, focused hybrid Jest rerun, `npm run docs`, `npx tsc --noEmit -p tsconfig.json`, API alignment check, source-of-truth guard for generated files, and focused coverage evidence for `src/neataptic.ts` plus `src/neat/hybrid/neat.hybrid.ts`.

**Plan update requirement:** Update this plan with validation evidence, any reroute decision, and either the Step 07 lane-closure handoff or the smallest Step 04 or Step 02 recovery handoff before ending the session.

**Step 05 validation evidence and reroute (current repo state):** Inspected the generated `src/neat/hybrid/README.md` and confirmed the Phase 5 closure items are present in generated output: the library-specific Lamarckian explanation at lines `8-10`, the Baldwin-effect distinction with Wikipedia grounding at lines `11-16`, the determinism ladder at lines `19-26`, the recommended-defaults progression at lines `29-35` and again in `HybridEvaluationPolicy` at lines `104-114`, the explicit `conditional` blocker and ranking requirement at lines `38-39`, `138-140`, and `176-178`, and a cohesive public API toy example using `Network`, `toParameterVector`, `fineTuneVector`, `fromParameterVector`, and `evaluateCandidate` at lines `44-87`. The root facade exports those example-facing symbols from `src/neataptic.ts` at lines `53-72`, so the example no longer depends on private import paths.

- Focused behavior gate: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts` passed with `1` suite and all `5` targeted tests green.
- Docs gate: `npm run docs` passed again, regenerated per-folder README output, validated Mermaid charts through the HTML render stage, and did not require any hand-edited generated file.
- TypeScript gate: `npx tsc --noEmit -p tsconfig.json` still fails only on the unchanged external ONNX baseline at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` with `TS2345`; no new failure appeared in `src/neataptic.ts`, `src/neat/hybrid/neat.hybrid.ts`, or `src/neat/hybrid/neat.hybrid.types.ts`.
- Hybrid coverage gate: `npx jest --config=jest.config.mjs --no-cache --coverage --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts --collectCoverageFrom='src/neat/hybrid/neat.hybrid.ts' --coverageReporters=text --coverageReporters=json-summary` passed and kept `src/neat/hybrid/neat.hybrid.ts` at `100/100/100/100`.
- Root-facade coverage gate: `npx jest --config=jest.config.mjs --no-cache --coverage --runTestsByPath src/neataptic.test.ts --collectCoverageFrom='src/neataptic.ts' --coverageReporters=text --coverageReporters=json-summary` failed the coverage-guard threshold for `src/neataptic.ts`: statements `100`, branches `100`, functions `88.23`, lines `100`. The local cause is visible in the owner-local test file: `src/neataptic.test.ts` does not import or touch the new vector and hybrid re-exports, so the Step 04 root-facade addition is not fully covered yet.
- Tracker sync validation: `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` passed with `PASS plan sync: 0 errors, 0 warnings`.
- Reroute decision: return to Step 04 only for the smallest root-facade test-slice or export-alignment follow-up needed to cover the newly added `toParameterVector`, `fromParameterVector`, `fineTuneVector`, `evaluateCandidate`, and related hybrid or vector type re-exports in `src/neataptic.ts`. Do not reopen the hybrid README wording, the hybrid behavior seam, or the unrelated ONNX baseline.

**Step 04 owner-local follow-up after the Step 05 root-facade coverage reroute:** Added the smallest owner-local root-facade follow-up in `src/neataptic.test.ts` only. The new assertion imports and touches `toParameterVector`, `fromParameterVector`, `fineTuneVector`, and `evaluateCandidate` as runtime functions so the root facade covers the newly added public re-exports without changing production code or reopening the hybrid seam.

- Changed test file: `src/neataptic.test.ts`
- Focused behavior follow-up: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/neataptic.test.ts` passed with `PASS`, `1` suite, and all `15` root-facade tests green.
- Focused coverage follow-up: `npx jest --config=jest.config.mjs --no-cache --coverage --runTestsByPath src/neataptic.test.ts --collectCoverageFrom='src/neataptic.ts' --coverageReporters=text --coverageReporters=json-summary` passed with `src/neataptic.ts` at `100/100/100/100`.
- Tracker sync validation: `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` passed with `PASS plan sync: 0 errors, 0 warnings` after this Step 04 follow-up update.
- Scope guard: this follow-up stayed owner-local to the root-facade test slice only. No production code, hybrid docs, generated outputs, or ONNX surfaces changed.

**Step 05 rerun completion from current repo state:** Re-ran the Phase 5 validation sequence after the owner-local root-facade coverage follow-up and all acceptance gates now pass inside the Phase 5 surface. The generated `src/neat/hybrid/README.md` still contains the required closure items in current generated output: the Lamarckian explanation, the Baldwin-effect distinction with Wikipedia grounding, the determinism ladder, the recommended progression, the explicit `conditional` blocker, and the cohesive public API example. The example-facing root exports also remain present in `src/neataptic.ts`: `toParameterVector`, `fromParameterVector`, `ParameterLayoutEntry`, `ParameterLayoutV1`, `ParameterVector`, `fineTuneVector`, `FineTuneOptions`, `FineTuneResult`, `evaluateCandidate`, `EvaluateCandidateOptions`, `HybridEvaluationPolicy`, `HybridEvaluationResult`, `HybridFineTuneMode`, and `HybridScoreNetwork`.

- Focused hybrid behavior rerun: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts` passed with `1` suite and all `5` tests green.
- Focused root-facade rerun: `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/neataptic.test.ts` passed with `1` suite and all `15` tests green.
- Docs gate: `npm run docs` passed again from current repo state and regenerated docs output without any hand-edited generated README or published example file.
- TypeScript gate: `npx tsc --noEmit -p tsconfig.json` still fails only on the unchanged external ONNX baseline at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` with `TS2345`; no new failure appeared in `src/neataptic.ts`, `src/neat/hybrid/neat.hybrid.ts`, or `src/neat/hybrid/neat.hybrid.types.ts`.
- Root-facade coverage gate: `npx jest --config=jest.config.mjs --no-cache --coverage --runTestsByPath src/neataptic.test.ts --collectCoverageFrom='src/neataptic.ts' --coverageReporters=text --coverageReporters=json-summary` passed and kept `src/neataptic.ts` at `100/100/100/100`.
- Hybrid coverage gate: `npx jest --config=jest.config.mjs --no-cache --coverage --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts --collectCoverageFrom='src/neat/hybrid/neat.hybrid.ts' --coverageReporters=text --coverageReporters=json-summary` passed and kept `src/neat/hybrid/neat.hybrid.ts` at `100/100/100/100`.
- Tracker sync validation: `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` passed with `PASS plan sync: 0 errors, 0 warnings` after this Step 05 rerun update.
- Reroute decision: none inside the Phase 5 boundary. The only remaining blocker is the unchanged external ONNX `TS2345` baseline, which stays outside this lane.

**Updated Step 07 closure handoff:** Call `07-logging` next. Phase 5 Step 05 is now green from current repo state, so Step 07 may close the lane, compress the tracker, create the same-boundary log, update `plans/README.md` plus `plans/Roadmap.md`, and archive into `plans/completed/` while preserving the unchanged ONNX `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` as external baseline context rather than reopening Phase 5 code or docs.

#### Step 06: SKIP — Documentation pass is the entire phase [DONE]

```yaml
phase: 5
step: 6
status: '[DONE]'
skip_reason: 'Phase 5 is a documentation-only closure pass. Step 04 is the authoring step and Step 05 is the green-validation gate. A separate Step 06 documentation step would be a redundant no-op in this phase structure.'
```

No separate Step 06 documentation pass is needed. Step 04 authors and Step 05 validates.

#### Step 07: Log Phase 5 closure and archive lane [PLANNED]

```yaml
phase: 5
step: 7
agent: '07-logging'
agent_file: '.github/agents/07-session-log-archivist.agent.md'
status: '[PLANNED]'
mode: 'fresh-session'
source_of_truth: 'plans/Evolution_Training_Interoperability_Contracts.md'
copy_paste: true
next_step: 'Lane complete — archive to plans/completed/'
skills:
  - 'tracker-handoff'
  - 'plan-sync-validation'
specialists:
  - 'Plan Registration Auditor'
validation:
  - 'Compressed closed plan and same-boundary log are in plans/completed/, plans/README.md and plans/Roadmap.md are updated to [DONE], and validate-plan-sync passes.'
```

**User instruction:** Start a fresh session, select `07-logging`, and paste this full step packet.

**Step objective:** Close the full hybrid-interoperability lane by compressing this tracker to a closed baseline, creating the same-boundary `Evolution_Training_Interoperability_Contracts.logs.md` audit record, moving both files to `plans/completed/`, updating `plans/README.md` and `plans/Roadmap.md` to mark this lane `[DONE]`, and running tracker sync validation before ending the session.

**Context the agent must know:**

- This is the terminal closure step for the entire `Evolution_Training_Interoperability_Contracts` lane. Do not keep the tracker open after this step closes.
- A compressed closed plan records: what the lane delivered, which files changed in each phase, the external ONNX TypeScript baseline decision, the conditional-policy blocked decision, and the NEATchat unlock conditions that are now met.
- The same-boundary log must be named `Evolution_Training_Interoperability_Contracts.logs.md` and must contain a concise pass history, files changed, validation evidence, and residual risks.
- After archiving, update `plans/README.md` to change the entry from active `[WIP]` to `[DONE]` and move the trigger phrases to the completed section.
- Update `plans/Roadmap.md` to mark this lane complete and unblock the next dependency-gated lane (NEATchat follow-up).
- Run `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/completed/Evolution_Training_Interoperability_Contracts.md --json` after the archive move.

**Execution steps:**

1. Read Phase 5 Step 04-05 evidence and confirm all lane acceptance gates are satisfied.
2. Compress the current detailed tracker to a short closed baseline record.
3. Create `plans/completed/Evolution_Training_Interoperability_Contracts.logs.md` with a concise pass history.
4. Move `plans/Evolution_Training_Interoperability_Contracts.md` to `plans/completed/`.
5. Update `plans/README.md` to reflect [DONE] status and move the trigger phrase to the completed section.
6. Update `plans/Roadmap.md` to mark this lane complete.
7. Run tracker sync validation and confirm pass.

**Stop conditions:**

- Done: archive files are in `plans/completed/`, plans index and roadmap are updated, and sync validation passes.
- Blocked: one or more Phase 5 acceptance gates from Step 04-05 are not confirmed; do not archive until those gates are satisfied.
- Route back: if validation, documentation, or example evidence is incomplete, return to the appropriate Phase 5 step rather than archiving prematurely.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/completed/Evolution_Training_Interoperability_Contracts.md --json` passes after archive.

**Plan update requirement:** Archive this plan, create the log, update plans index and roadmap, and confirm sync validation before ending the session.

## Recommended first implementation frontier after the ONNX stop line

1. Open Phase 1 and Phase 2 as the coupled first tranche; the first planner pass should author the smallest failing tests for deterministic layout ordering and parameter roundtrip.
2. Implement a network-owned `ParameterVector` plus `ParameterLayoutV1` surface.
3. Add explicit mismatch validation before opening any Phase 3 isolation helper work.
4. Run the narrow test slice first, then TypeScript validation, then coverage guard on touched `src/` files.

## Testing strategy

- Parameter mapping tests:
  - repeated export equality on the same runtime
  - roundtrip equality on outputs
  - version, length, and descriptor mismatch rejection
  - honesty tests for unsupported or deferred parameter families
- Hybrid evaluation tests:
  - fine-tuning improves loss for a tiny dataset (separate tests)
  - the original candidate stays unchanged after isolated fine-tuning
  - persistence policy changes whether weights are retained
  - conditional policy uses explicit deterministic ordering or tie-break rules
- Validation cadence:
  - start with the narrowest failing test slice,
  - rerun the same slice immediately after the first substantive edit,
  - run `npx tsc --noEmit -p tsconfig.json` after the active slice is green,
  - if `src/` changes, run coverage guard on every touched source file before closing the tranche.

## Risks and mitigations

- Risk: API complexity.
  - Mitigation: keep high-level defaults; expose advanced policies separately.
- Risk: determinism is hard with floating-point + parallelism.
  - Mitigation: clearly document determinism scope; keep ordering fixed.
- Risk: ordering drift leaks in from implicit object or array order.
  - Mitigation: derive v1 ordering from explicit stable node and connection identity only.
- Risk: ONNX or checkpoint code becomes the accidental source of truth for parameter vectors.
  - Mitigation: keep the contract network-owned and let adjacent surfaces consume it.
- Risk: disabled-connection or deferred-parameter ambiguity widens the surface too early.
  - Mitigation: lock the v1 inclusion rule before implementation and reject unsupported cases explicitly.

## Success criteria

- Hybrid workflows are well-defined and reproducible.
- Training integration does not require ad-hoc per-project glue.
- Parameter mapping becomes a later unification seam for export- and worker-adjacent tooling once those Phase 4 foundations exist.
- A network-owned parameter-vector contract exists and is versioned.
- The lane states its determinism promise precisely instead of equating “same seed” with full replay.
- NEATchat can later consume this surface without inventing a private vector format or hidden mutation path.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
The active tracker is plans/Evolution_Training_Interoperability_Contracts.md and the lane is [WIP]. Phase 1 is [DONE]. Phase 2 is [DONE]. Phase 3 is [DONE]. Phase 4 is [DONE]. Phase 5 is [WIP]. The active frontier is Phase 5 Step 07 closure after a completed Step 05 rerun from current repo state.

Phase 5 Step 05 now confirms from the current repo state:
- `src/neat/hybrid/README.md` includes the library-specific Lamarckian explanation, the Baldwin-effect distinction with Wikipedia grounding, the three-rung determinism ladder, the recommended defaults progression, the explicit `conditional` blocker or ranking requirement, and a cohesive public API toy example using `Network`, `toParameterVector`, `fineTuneVector`, `fromParameterVector`, and `evaluateCandidate`.
- `src/neataptic.ts` exports the example-facing lane symbols at the root facade: `toParameterVector`, `fromParameterVector`, `ParameterLayoutEntry`, `ParameterLayoutV1`, `ParameterVector`, `fineTuneVector`, `FineTuneOptions`, `FineTuneResult`, `evaluateCandidate`, `HybridFineTuneMode`, `HybridEvaluationPolicy`, `HybridScoreNetwork`, `EvaluateCandidateOptions`, and `HybridEvaluationResult`.
- `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/neat/hybrid/neat.hybrid.test.ts` passed with `1` suite and all `5` targeted tests green.
- `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/neataptic.test.ts` passed with `1` suite and all `15` root-facade tests green.
- `npm run docs` passed again, regenerated per-folder README output, and completed the HTML render stage. No generated README or published example page was hand-edited.
- `npx tsc --noEmit -p tsconfig.json` still fails only on the unchanged external ONNX `TS2345` baseline at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135`.
- Focused root-facade coverage passed: `src/neataptic.ts` stayed at `100/100/100/100`.
- Focused hybrid coverage passed: `src/neat/hybrid/neat.hybrid.ts` stayed at `100/100/100/100`.
- `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` passed with `PASS plan sync: 0 errors, 0 warnings`.

Next work must stay narrow inside Step 07 closure only:
1. Read the recorded Phase 5 Step 04-05 evidence and the existing Step 07 packet.
2. Close the lane with the planned Step 07 archive flow: compress the tracker, create `Evolution_Training_Interoperability_Contracts.logs.md`, move both files into `plans/completed/`, update `plans/README.md` and `plans/Roadmap.md`, and rerun plan-sync validation.
3. Preserve the unchanged ONNX `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` as external baseline context rather than reopening Phase 5 implementation, docs, or tests.

Next orchestrator: 07-logging for Phase 5 Step 07 closure and archive lane.
```
