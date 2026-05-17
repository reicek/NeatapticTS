# Evolution–Training Interoperability Contracts Plan

**Status:** [PLANNED]

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
- This file should stay planning-only until the user explicitly opens the hybrid-interoperability workstream; the archived ONNX closure is the handoff boundary, not a signal to start hybrid work early by default.
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
- Step 1 and Step 2 should not claim exact replay across Node, browser, and worker paths yet.
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

1. Step 1 and Step 2 are complete, so deterministic parameter layout plus
   vector export/import are real and roundtrip inference correctly.
2. Step 3 is complete, so fine-tuning cannot mutate shared candidate state by
   accident.
3. Step 4 is complete, so the policy for fitness-only versus persistent
   training is explicit rather than hidden inside downstream application code.
4. The public contract is documented clearly enough that NEATchat can compare a
   frozen base, a personalized variant, and candidate deltas without inventing
   a parallel private vector format.

This plan does not replace checkpointing or worker execution. It provides the
promotion and isolation seam that those other NEATchat blockers rely on.

## Recommended agent + skill combo by step

- Step 1 — `Hybrid Interop Scout` + `hybrid-training-interop`
- Step 2 — `Hybrid Interop Scout` + `hybrid-training-interop`
- Step 3 — `Hybrid Interop Scout` + `hybrid-training-interop`
- Step 4 — `Hybrid Interop Scout` + `hybrid-training-interop`
- Step 5 — `Docs Scout` + `educational-docs`

## Implementation steps

Implementation order and stop-line guidance:

- Step 1 and Step 2 form the first implementation tranche and should land together unless a failing test proves that ordering can be isolated safely on its own.
- Step 3 should introduce one minimal isolation primitive first; avoid opening both clone-based and vector-based ergonomics before one deterministic base path is stable.
- Step 4 should not start until Step 3 proves that fine-tuning can happen without shared candidate mutation.
- Step 5 closes the lane only after public naming, determinism wording, and the example surface are stable enough for downstream consumers.

### Step 1 — Stabilize deterministic parameter ordering

- Require stable node and edge ordering.
- Define source-owned `ParameterLayoutV1` and `ParameterLayoutEntry` types rather than deriving layout from ONNX export structures.
- Lock a canonical ordering rule for v1:
  1. bias entries ordered by stable node gene id,
  2. weight entries ordered by explicit stable edge identity, preferring connection innovation when present and an explicit deterministic fallback when not,
  3. one documented fold order for entry kinds, kept stable under version `1`.
- Reject ambiguous ordering cases rather than relying on incidental container order.
- Add a narrow owner-local test slice covering repeated export stability and equivalent reconstructed network ordering.

Acceptance:

- Same network always produces the same layout ordering.

Focused validation:

- Repeated same-runtime export yields identical ordered entry descriptors.
- Equivalent reconstructed networks yield the same layout ordering.

### Step 2 — Implement vector export/import

- Implement `toParameterVector` and `fromParameterVector`.
- Export values plus layout together from the network-owned boundary.
- Validate version compatibility, entry-count compatibility, and descriptor compatibility before mutating the target network.
- Keep recurrent and advanced topology support honest: when a parameter family cannot be mapped cleanly under layout v1, fail explicitly instead of widening the promise silently.
- Add negative tests for version mismatch, length mismatch, and layout-descriptor mismatch.

Acceptance:

- Roundtrip preserves inference outputs.

Focused validation:

- Export plus import preserves outputs for the same runtime and topology.
- Invalid vector or layout payloads fail explicitly.

### Step 3 — Add training isolation helpers

- Provide one minimal isolation primitive first.
- Prefer a low-level vector-in and trained-vector-out helper because it is easier to keep deterministic across worker, checkpoint, and evaluation surfaces.
- A clone-based ergonomic wrapper can follow as a thin convenience layer once the vector-first path is stable.
- Require dataset ordering and training settings to be explicit inputs to the helper.
- Any deterministic claim at this step must require explicit seed handling and a documented RNG owner.
- Avoid helpers that silently mutate the supplied network or vector unless the mutating behavior is stated directly in the API name.

Acceptance:

- Fine-tuning cannot mutate shared population state accidentally.

Focused validation:

- Fine-tuning leaves the original candidate unchanged.
- Same seed plus same dataset order yields stable fine-tuned output on the same runtime.

### Step 4 — Define hybrid evaluation policy integration

- Integrate policy hooks into the NEAT evaluation surface only after Steps 1 through 3 are green.
- Keep three decisions separate:
  - which candidates are allowed to fine-tune,
  - how the trained variant is scored,
  - whether trained weights persist back into the canonical candidate.
- Treat `never`, `always`, and `conditional` as policy choices, but make the conditional path define deterministic ranking or tie-break rules rather than inheriting current iteration order.
- Require worker and single-thread paths to agree on semantic result order before Lamarckian persistence is applied.

Acceptance:

- Users can choose fitness-only fine-tuning or Lamarckian persistence.

Focused validation:

- The no-fine-tune policy leaves candidates untouched.
- Fitness-only fine-tuning returns a trained result without persistence.
- Lamarckian persistence happens only on explicit opt-in.

### Step 5 — Docs + examples

- Document:
  - what “Lamarckian” means in this library
  - reproducibility considerations
  - recommended defaults
- Explain the determinism ladder used by this lane: same-runtime ordered deterministic versus replay exact versus best-effort reproducible.
- Include one tiny example on a toy dataset that exercises vector export/import and isolated fine-tuning without introducing a large training framework.
- Refresh generated docs after public JSDoc changes so downstream README surfaces stay synchronized.

Acceptance:

- Docs contain a small example and clear warnings.

Focused validation:

- The example stays aligned with the real public API.
- `npm run docs` succeeds after the doc-facing pass.

## Recommended first implementation tranche after the ONNX stop line

1. Add the smallest failing tests for deterministic layout ordering and parameter roundtrip.
2. Implement a network-owned `ParameterVector` plus `ParameterLayoutV1` surface.
3. Add explicit mismatch validation before opening any fine-tuning helper work.
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
