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
- The current active frontier is now Phase 3 Step 01. Phase 2 is [DONE]: the serialize-owned `ParameterVector` export/import seam is green, owner-local coverage for the touched serialize source files is back at `100/100/100/100`, the source-doc-only pass refreshed `src/architecture/network/serialize/README.md`, and repo-wide TypeScript still reproduces only the unchanged external ONNX baseline blocker at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135`.
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
agent: "01 Planning Architect"
agent_file: ".github/agents/01-planning-architect.agent.md"
status: "[DONE]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Step 02 — Research deterministic ordering boundary"
skills:
  - "hybrid-training-interop"
specialists:
  - "Hybrid Interop Scout"
validation:
  - "Manual tracker check that the canonical ordering rule, acceptance gate, and focused validation are preserved before Step 02-07 packets are authored."
```

**User instruction:** Start a fresh session, select `01 Planning Architect`, and paste this full step packet.

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
agent: "02 Research Coordinator"
agent_file: ".github/agents/02-research-coordinator.agent.md"
status: "[DONE]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Step 03 — Design deterministic ordering red tests"
skills:
  - "hybrid-training-interop"
  - "plan-alignment"
specialists:
  - "Hybrid Interop Scout"
validation:
  - "Read-only research brief names the network owner boundary, stable node and edge identities, likely owner-local test file, and whether Phase 1 ordering can be red-tested without implementing Phase 2 import/export."
```

**User instruction:** Start a fresh session, select `02 Research Coordinator`, and paste this full step packet.

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
agent: "03 Red Test Architect"
agent_file: ".github/agents/03-red-test-architect.agent.md"
status: "[DONE]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Step 04 — Implement ParameterLayoutV1 ordering"
skills:
  - "hybrid-training-interop"
  - "red-test-contracts"
  - "reproducibility-contracts"
specialists:
  - "Determinism Scout"
validation:
  - "A focused owner-local red test or explicit blocked record proves whether repeated same-runtime layout ordering and equivalent reconstructed network ordering can be tested before Phase 2."
```

**User instruction:** Start a fresh session, select `03 Red Test Architect`, and paste this full step packet.

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
agent: "04 Implementation Architect"
agent_file: ".github/agents/04-implementation-architect.agent.md"
status: "[DONE]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Step 05 — Validate deterministic ordering gates"
skills:
  - "hybrid-training-interop"
  - "reproducibility-contracts"
specialists:
  - "Hybrid Interop Scout"
validation:
  - "The Step 03 red test goes green with a network-owned layout ordering implementation and no Phase 2 vector import/export behavior."
```

**User instruction:** Start a fresh session, select `04 Implementation Architect`, and paste this full step packet.

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
agent: "05 Green Validation Runner"
agent_file: ".github/agents/05-green-validation-runner.agent.md"
status: "[DONE]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Step 07 — Log Phase 1 closure or reroute"
skills:
  - "green-validation-gates"
  - "coverage-guard"
  - "hybrid-training-interop"
specialists:
  - "Coverage Guard"
  - "Determinism Scout"
validation:
  - "Focused deterministic-ordering tests and owner-local coverage pass for every touched serialize src file, and repo-wide TypeScript introduces no new failure into the seam beyond the recorded ONNX baseline blocker."
```

**User instruction:** Start a fresh session, select `05 Green Validation Runner`, and paste this full step packet.

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

**Updated Step 05 rerun handoff:** Call `05 Green Validation Runner` next. Re-run `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --testNamePattern="ParameterLayoutV1 ordering"`, then `npx tsc --noEmit -p tsconfig.json`, then the owner-local coverage guard command for `src/architecture/network/serialize/network.serialize.utils.ts` and `src/architecture/network/serialize/network.serialize.utils.types.ts`. Treat the existing `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` as the unchanged repo baseline blocker unless a new failure appears inside the serialize seam. If the focused slice or coverage guard fails inside layout ordering, route back to Step 03 for missing guard-rail tests or Step 04 for implementation defects. Do not activate Phase 2.

**Exact next orchestrator prompt:** Re-run Phase 1 Step 05 for the layout-ordering seam only. Use `npx jest --config=jest.config.mjs --no-cache --runTestsByPath src/architecture/network/serialize/network.serialize.test.ts --testNamePattern="ParameterLayoutV1 ordering"` first, then `npx tsc --noEmit -p tsconfig.json` while treating the existing `TS2345` at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` as the unchanged repo baseline blocker unless the failure surface moves into serialize ordering, then run the owner-local coverage guard command for `src/architecture/network/serialize/network.serialize.utils.ts` and `src/architecture/network/serialize/network.serialize.utils.types.ts`. If the focused slice or coverage guard fails inside the layout-ordering seam, route to Step 03 for missing guard-rail coverage or Step 04 for implementation defects. Do not activate Phase 2.

**Step 06 documentation gate note:** A source-first documentation pass can keep the new layout surface teachable and regenerate the serialize README, but Phase 1 still cannot close until the Step 03 coverage follow-up lands and the repo-wide TypeScript baseline is green again or the user narrows that required gate.

#### Step 06: Curate layout-ordering docs [DONE]

```yaml
phase: 1
step: 6
agent: "06 Educational Docs Curator"
agent_file: ".github/agents/06-educational-docs-curator.agent.md"
status: "[DONE]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Step 07 — Log Phase 1 closure or reroute"
skills:
  - "educational-docs"
  - "docs-academic-citation-audit"
  - "hybrid-training-interop"
specialists:
  - "Docs Scout"
validation:
  - "Source-first docs for any exported layout symbols are present, generated docs are refreshed if JSDoc inputs changed, and no Phase 2 or fine-tuning docs are introduced."
```

**User instruction:** Start a fresh session, select `06 Educational Docs Curator`, and paste this full step packet.

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
agent: "07 Session Log Archivist"
agent_file: ".github/agents/07-session-log-archivist.agent.md"
status: "[DONE]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Phase 2 Step 03 — Design vector roundtrip red tests"
skills:
  - "tracker-handoff"
  - "plan-sync-validation"
  - "hybrid-training-interop"
specialists:
  - "Plan Registration Auditor"
validation:
  - "Manual tracker review confirms Phase 1 acceptance evidence, Phase 2 coupling state, and the refreshed Handoff query before the next phase opens."
```

**User instruction:** Start a fresh session, select `07 Session Log Archivist`, and paste this full step packet.

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
- Next safe step: activate `03 Red Test Architect` for Phase 2 `Step 03 — Design vector roundtrip red tests`. Do not skip ahead to Phase 2 Step 04 or later packets.
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
agent: "01 Planning Architect"
agent_file: ".github/agents/01-planning-architect.agent.md"
status: "[DONE]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Step 02 — Research vector roundtrip owner boundary"
skills:
  - "hybrid-training-interop"
specialists:
  - "Hybrid Interop Scout"
validation:
  - "Manual tracker check that roundtrip, mismatch-rejection, and network-owned contract details are preserved before Step 02-07 packets are authored."
```

**User instruction:** Start a fresh session, select `01 Planning Architect`, and paste this full step packet.

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
agent: "02 Research Coordinator"
agent_file: ".github/agents/02-research-coordinator.agent.md"
status: "[DONE]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Step 03 — Design vector roundtrip red tests"
skills:
  - "hybrid-training-interop"
  - "plan-alignment"
specialists:
  - "Hybrid Interop Scout"
validation:
  - "Read-only research brief names the network owner boundary, the v1 parameter-family inclusion rule, the required mismatch checks, the likely owner-local test file, and any unsupported-family honesty failures needed before red tests."
```

**User instruction:** Start a fresh session, select `02 Research Coordinator`, and paste this full step packet.

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
agent: "03 Red Test Architect"
agent_file: ".github/agents/03-red-test-architect.agent.md"
status: "[DONE]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Step 04 — Implement parameter-vector roundtrip boundary"
skills:
  - "hybrid-training-interop"
  - "red-test-contracts"
  - "reproducibility-contracts"
specialists:
  - "Determinism Scout"
validation:
  - "A focused owner-local red test or explicit blocked record proves whether same-runtime export or import roundtrip and mismatch rejection can fail honestly before implementation."
```

**User instruction:** Start a fresh session, select `03 Red Test Architect`, and paste this full step packet.

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
agent: "04 Implementation Architect"
agent_file: ".github/agents/04-implementation-architect.agent.md"
status: "[DONE]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Step 05 — Validate vector roundtrip gates"
skills:
  - "hybrid-training-interop"
  - "reproducibility-contracts"
specialists:
  - "Hybrid Interop Scout"
validation:
  - "The Step 03 red test goes green with a network-owned parameter-vector export or import implementation and no Phase 3 or Phase 4 behavior."
```

**User instruction:** Start a fresh session, select `04 Implementation Architect`, and paste this full step packet.

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
agent: "05 Green Validation Runner"
agent_file: ".github/agents/05-green-validation-runner.agent.md"
status: "[DONE]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Step 06 — Curate vector-contract docs"
skills:
  - "green-validation-gates"
  - "coverage-guard"
  - "hybrid-training-interop"
specialists:
  - "Coverage Guard"
  - "Determinism Scout"
validation:
  - "Focused vector roundtrip and mismatch tests pass, TypeScript validation passes or a pre-existing blocked baseline is recorded explicitly, and coverage guard passes for every touched src file."
```

**User instruction:** Start a fresh session, select `05 Green Validation Runner`, and paste this full step packet.

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

**Updated Step 06 handoff:** Call `06 Educational Docs Curator` next. Keep the next pass source-doc-only and scoped to the serialize-owned vector surface: review `ParameterVector`, `toParameterVector`, and `fromParameterVector` JSDoc plus the generated `src/architecture/network/serialize/README.md` output for alignment with the now-final v1 contract, including the weights-and-biases inclusion rule, explicit unsupported-family rejection for non-neutral `node.response` and `connection.gain`, disabled-connection slot behavior, fallback descriptor semantics when an innovation id is absent, and same-runtime determinism wording. Run `npm run docs` only if source JSDoc changes are required. Do not widen into Phase 3 isolation, Phase 4 policy, or unrelated ONNX work.

#### Step 06: Curate vector-contract docs [DONE]

```yaml
phase: 2
step: 6
agent: "06 Educational Docs Curator"
agent_file: ".github/agents/06-educational-docs-curator.agent.md"
status: "[DONE]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Step 07 — Log Phase 2 closure or reroute"
skills:
  - "educational-docs"
  - "docs-academic-citation-audit"
  - "hybrid-training-interop"
specialists:
  - "Docs Scout"
validation:
  - "Source-first docs for any exported vector symbols are present, generated docs are refreshed if JSDoc inputs changed, and no Phase 3 or Phase 4 docs are introduced accidentally."
```

**User instruction:** Start a fresh session, select `06 Educational Docs Curator`, and paste this full step packet.

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
- Exact next orchestrator: `07 Session Log Archivist`
- Exact handoff prompt: see `## Handoff query` below.

#### Step 07: Log Phase 2 closure or reroute [DONE]

```yaml
phase: 2
step: 7
agent: "07 Session Log Archivist"
agent_file: ".github/agents/07-session-log-archivist.agent.md"
status: "[DONE]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Phase 3 Step 01 — Plan the isolation-helper tranche"
skills:
  - "tracker-handoff"
  - "plan-sync-validation"
  - "hybrid-training-interop"
specialists:
  - "Plan Registration Auditor"
validation:
  - "Manual tracker review confirms Phase 2 acceptance evidence, the Phase 1 coupling state, and the refreshed Handoff query before the next phase opens."
```

**User instruction:** Start a fresh session, select `07 Session Log Archivist`, and paste this full step packet.

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
- Next safe step: activate `01 Planning Architect` for Phase 3 `Step 01 — Plan the isolation-helper tranche`. Do not widen into Phase 3 implementation, Phase 4 policy work, or unrelated ONNX fixes.
- Tracker validation: rerun `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` after this tracker refresh.

### Phase 3 — Add training isolation helpers [WIP]

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

#### Step 01: Plan the isolation-helper tranche [WIP]

```yaml
phase: 3
step: 1
agent: "01 Planning Architect"
agent_file: ".github/agents/01-planning-architect.agent.md"
status: "[WIP]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Step 02 — Planner-defined by this step"
skills:
  - "hybrid-training-interop"
specialists:
  - "Hybrid Interop Scout"
validation:
  - "Manual tracker check that the isolation rule, vector-first preference, and deterministic-claim guardrails are preserved before Step 02-07 packets are authored."
```

**User instruction:** Start a fresh session, select `01 Planning Architect`, and paste this full step packet.

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

### Phase 4 — Define hybrid evaluation policy integration [PLANNED]

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

#### Step 01: Plan the hybrid-policy tranche [PLANNED]

```yaml
phase: 4
step: 1
agent: "01 Planning Architect"
agent_file: ".github/agents/01-planning-architect.agent.md"
status: "[PLANNED]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Step 02 — Planner-defined by this step"
skills:
  - "hybrid-training-interop"
specialists:
  - "Hybrid Interop Scout"
validation:
  - "Manual tracker check that the policy split, Lamarckian opt-in rule, and deterministic ranking guardrails are preserved before Step 02-07 packets are authored."
```

**User instruction:** Start a fresh session, select `01 Planning Architect`, and paste this full step packet.

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

### Phase 5 — Docs and examples [PLANNED]

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
agent: "01 Planning Architect"
agent_file: ".github/agents/01-planning-architect.agent.md"
status: "[PLANNED]"
mode: "fresh-session"
source_of_truth: "plans/Evolution_Training_Interoperability_Contracts.md"
copy_paste: true
next_step: "Step 02 — Planner-defined by this step"
skills:
  - "educational-docs"
specialists:
  - "Docs Scout"
validation:
  - "Manual tracker check that the determinism wording, Lamarckian explanation, and example-validation expectations are preserved before Step 02-07 packets are authored."
```

**User instruction:** Start a fresh session, select `01 Planning Architect`, and paste this full step packet.

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
The active tracker is plans/Evolution_Training_Interoperability_Contracts.md and the lane is still [WIP]. Phase 1 is [DONE]. Phase 2 is now [DONE] from current repo state: the serialize-owned `ParameterVector` export/import seam is green, owner-local coverage for `src/architecture/network/serialize/network.serialize.utils.ts` and `src/architecture/network/serialize/network.serialize.utils.types.ts` is `100/100/100/100`, and the source-doc-only pass refreshed `src/architecture/network/serialize/README.md` so the generated output now matches the final v1 vector contract.
The reviewed vector docs now cover the weights-and-biases inclusion rule, explicit rejection of non-neutral `node.response` and `connection.gain`, disabled-connection slot behavior, fallback descriptor semantics when an innovation id is absent, and same-runtime deterministic wording without claiming cross-runtime exact replay. Repo-wide TypeScript still fails only at the unchanged external ONNX baseline `src/architecture/network/onnx/validate/network.onnx.validate.ts:135` with `TS2345` because `standardDomainImports[0]!.version` can be `undefined`; treat that as recorded external baseline context unless the failure surface moves into this lane.
The next safe step is `01 Planning Architect` for Phase 3 Step 01 only. Turn Phase 3 into an executable isolation-helper workset: author Phase 3 Step 02-07 packets or explicit skips, keep the first helper vector-first, preserve the no-shared-mutation contract, and keep any deterministic claim gated on explicit seed handling, dataset order, and a named RNG owner. Run `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/Evolution_Training_Interoperability_Contracts.md --json` after the tracker update. Do not widen into Phase 3 implementation, Phase 4 policy work, or unrelated ONNX fixes.
```
