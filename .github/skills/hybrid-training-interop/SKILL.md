---
name: hybrid-training-interop
description: 'Design, implement, or validate deterministic parameter-vector and fine-tuning contracts in NeatapticTS. Use when working on toParameterVector, fromParameterVector, layout versioning, isolated fine-tuning, Lamarckian persistence policy, optimizer interoperability, cloned-vs-vector training, or hybrid evolution plus gradient workflows.'
argument-hint: 'Describe the parameter-vector or hybrid-training target, current step in Evolution_Training_Interoperability_Contracts.md, the mutation-isolation requirement, and whether the pass is design, implementation, or validation.'
user-invocable: true
disable-model-invocation: false
---

# Hybrid Training Interoperability Playbook

Use this skill when NeatapticTS work touches the parameter-vector and isolated
training seam defined in
`plans/Evolution_Training_Interoperability_Contracts.md`.

This skill owns the durable workflow for deterministic parameter layout,
vector export and import, isolated fine-tuning, explicit persistence policy, and
training interoperability that does not mutate shared candidate state by
accident.

When tracker files need updating, `tracker-handoff` owns plan and log shape.
When sequencing or dependency tension is unclear, use `plan-alignment`.

See [hybrid training sources](./references/hybrid-training-sources.md) for
reference notes on parameter flattening, gradient updates, and source
attribution.

## Scope Boundary

- In scope: `ParameterVector`, layout versioning, deterministic parameter
  ordering, export or import roundtrips, cloned-network versus vector-based
  fine-tuning isolation, policy surfaces such as fitness-only versus Lamarckian
  persistence, dataset or optimizer handoff semantics, small validation helpers
  for hybrid evaluation.
- Out of scope: full training framework design, arbitrary autograd over every
  graph shape, checkpoint format ownership (owned by
  `checkpointing-persistence`), worker pool scheduling (owned by
  `multithread-evaluation`), and ONNX import or export (owned by `onnx-work`).

## When to Use

- Deterministic parameter ordering is being introduced or repaired.
- A network needs to export weights and biases into a stable vector.
- Fine-tuning is mutating shared state accidentally.
- The repo needs a clear hybrid policy for training during evaluation.
- A downstream system such as NEATchat needs to compare frozen, personalized,
  and candidate parameter deltas without inventing its own vector format.

## Core Contracts

### Parameter layout contract

- Export order must be deterministic and documented.
- Layout metadata must be versioned and self-describing.
- Import must validate vector compatibility before mutating a network.

### Isolation contract

- Fine-tuning must operate on a clone or on a detached parameter vector.
- Shared population state must not mutate unless the chosen policy explicitly
  persists the trained weights.

### Policy contract

- The public API must make persistence policy explicit:
  - fitness-only training,
  - persistent or Lamarckian training,
  - never train.
- Do not hide policy inside application-specific glue code.

## Gradient Reminder

Backpropagation computes gradients, not the whole learning policy. A basic
update step is:

$$
\Delta w_{ij} = -\eta \frac{\partial E}{\partial w_{ij}}
$$

where $\eta$ is the learning rate and $E$ is the loss. In this repo, that means
the gradient mechanism and the persistence decision must stay separate.

## Task Packet

Pass a compact packet that includes:

- active plan step,
- parameter-layout or fine-tuning target,
- isolation requirement,
- persistence policy in scope,
- determinism requirement,
- validation target: vector roundtrip, loss-improvement test, policy test, or
  all three.

Compact example:

```text
Use hybrid-training-interop for Step 2 vector export/import.
Plan: plans/Evolution_Training_Interoperability_Contracts.md.
Target: deterministic ParameterVector layout v1.
Isolation rule: import must not mutate unrelated candidates.
Policy: fitness-only for now; no Lamarckian persistence in this pass.
Validate with: vector roundtrip inference equality, layout compatibility negative tests, and npm run test:silent.
```

## Required Workflow

1. Read `plans/README.md`, then
   `plans/Evolution_Training_Interoperability_Contracts.md`.
2. Read the nearest relevant README and owner-local tests before deep source
   edits.
3. Decide the canonical parameter order explicitly.
4. Decide whether the work uses clone-based or vector-based isolation.
5. Add or update the smallest focused red-phase test for the contract in scope.
6. Implement the smallest boundary-local change.
7. Immediately rerun the same focused validation after the first substantive
   edit.
8. Run `coverage-guard` on every touched `src/` file.
9. Document the persistence policy and determinism scope clearly.
10. Update the plan only after the code and validation are green.

## Layout Rules

- Use a versioned layout object, not only a raw value vector.
- Bias and weight entries must be distinguishable in metadata.
- Use explicit node or connection identifiers where possible.
- Do not rely on incidental object iteration order when the contract is meant to
  survive refactors.

## Isolation Rules

### Clone-based isolation

- Use when training logic naturally mutates network objects.
- Prefer when the training surface is still exploratory and correctness matters
  more than throughput.

### Vector-based isolation

- Use when training can stay detached from live candidate objects.
- Prefer when worker compatibility or checkpoint compatibility matters.
- Treat returned vectors as immutable results until they are deliberately applied.

## Policy Rules

- `fineTune: 'never'` means the candidate is evaluated without training side
  effects.
- `fineTune: 'always'` or `'conditional'` must also state whether trained
  weights are discarded or persisted.
- Lamarckian persistence means the trained weights become part of the candidate
  state after evaluation; this must never be implicit.

## Validation Cadence

- Parameter vector roundtrip test: export, import, and compare inference output.
- Negative compatibility tests for layout mismatch or length mismatch.
- Isolation tests proving that one candidate's fine-tune pass does not mutate
  another candidate.
- Policy tests proving fitness-only versus persistent training behavior differ in
  the intended way.
- `npm run test:silent` after the focused tranche is green.

## Guardrails

- Do not conflate parameter layout order with current in-memory incidental
  ordering.
- Do not let fine-tuning mutate shared state by default.
- Do not expose a parameter vector without layout metadata if the format is
  meant to survive future graph changes.
- Do not hide Lamarckian persistence behind a boolean with vague naming.
- Do not claim deterministic training if random seeds, data order, or floating
  point caveats are uncontrolled.

## Expected Final Output

A strong hybrid-training pass should report:

- the parameter-vector or policy surface targeted,
- the deterministic layout or isolation rule added,
- whether persistence is fitness-only or Lamarckian,
- focused validation results,
- coverage result for touched `src/` files,
- the updated plan step.
