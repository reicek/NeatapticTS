---
name: nge-core-algorithm
description: 'Design, implement, or validate the NEAT Genesis EvoDevo core algorithm in NeatapticTS. Use when working on computation motifs, NGE_DNA, deterministic development, lifecycle state machines, local growth or prune policy, memory tiers, neuromodulation, reproduction modes, collective-intelligence primitives, or core invariants that benchmark demos will later consume.'
argument-hint: 'Describe the NGE phase or primitive in scope, the archived plan section in plans/completed/NEAT_Genesis_EvoDevo.md, whether the pass is architecture, implementation, or validation, and which invariants must remain deterministic and opt-in.'
user-invocable: true
disable-model-invocation: false
---

# NGE Core Algorithm Playbook

Use this skill when the work is about the algorithmic core of NGE rather than a
demo-specific benchmark or visualization.

This skill owns the durable workflow for:

- computation motifs,
- `NGE_DNA`,
- deterministic development,
- lifecycle state transitions,
- local growth and pruning,
- memory tiers,
- neuromodulation,
- reproduction modes,
- core collective-intelligence primitives such as stigmergic field semantics.

Follow-on benchmark plans and demo harnesses stay out of scope here; those belong
to a separate benchmark owner.

See [NGE core sources](./references/nge-core-sources.md) for paraphrased notes on
evo-devo, stigmergy, and neuromodulation.

## Scope Boundary

- In scope: `computationType` catalogue design, `ResidualStream`,
  `WeightSharedCohort`, coordinate injection, `NGE_DNA` schema and versioning,
  deterministic build pipeline, lifecycle stages, focus scoring, morphogenesis
  policies, memory tiers, neuromodulator zones, reproduction policy semantics,
  role-differentiation primitives, typed-array stigmergy field primitives, and
  core invariants.
- Out of scope: final benchmark curricula, browser UX polish, demo progression,
  scoring dashboards, and environment-specific performance tuning unless needed
  to defend a core invariant.

## Core Framing

The plan treats DNA as a compact program rather than an explicit graph. That means
the primary question is not "what graph do we save?" but "what deterministic
program and governance knobs produce the graph?"

## When to Use

- Adding or refining an NGE computation motif.
- Defining `NGE_DNA` or its compatibility rules.
- Implementing deterministic development from DNA plus seed.
- Designing juvenile, adult, or equilibrium stage behavior.
- Adding neuromodulator broadcasts or memory-tier primitives.
- Implementing parthenogenesis, polyandric, or standard sexual reproduction
  modes.
- Introducing typed-array stigmergy or shared-field semantics as a primitive
  later used by demos.

## Phase Ownership

This skill primarily owns the algorithm side of these plan phases:

- Phase 0: computation motifs,
- Phase A: DNA plus deterministic development,
- Phase B: juvenile focus and local growth or prune,
- Phase C: adult optimization and equilibrium detection,
- Phase D: assimilation,
- Phase E: evolution integration and reproduction modes,
- core primitives inside Phase G when they are algorithmic rather than
  demo-specific.

## Design Pillars

- Opt-in and isolated: classic NEAT must remain unchanged when NGE is disabled.
- Deterministic by default: same DNA, seed, and experience stream must reproduce
  the same lifecycle checkpoints when the plan says it should.
- Budgeted growth: every build or morph action obeys explicit caps.
- Motif-first runtime: modules are not generic nodes; each archetype declares a
  `computationType`.
- Assimilation writes back structural priors, not inherited weights.
- Core before benchmark: shared primitives should stabilize before benchmark code
  starts compensating for missing library contracts.

## Task Packet

Pass a compact packet that includes:

- the NGE phase or primitive,
- the exact invariant under review,
- whether the pass is design, implementation, or validation,
- memory-plan dependency if relevant,
- focused validation target.

Compact example:

```text
Use nge-core-algorithm for Phase A deterministic development.
Plan: plans/completed/NEAT_Genesis_EvoDevo.md.
Target: canonical NGE_DNA module ordering and deterministic materialization.
Invariant: same DNA plus seed plus experience stream yields identical module IDs and edge hashes.
Validation: repeated-build hash tests, ordering tests, and opt-in verification that classic NEAT is unchanged.
```

## Required Workflow

1. Read `plans/Memory_Optimization.md` for authoritative gating when relevant.
2. Read `plans/completed/NEAT_Genesis_EvoDevo.md` and isolate the smallest archived phase or
   primitive.
3. Decide whether the current work belongs in core algorithm or benchmark/demo
   ownership.
4. Add the smallest focused test that should fail for the target invariant.
5. Implement the smallest owner-local change.
6. Immediately rerun the same focused validation after the first substantive
   edit.
7. Run `coverage-guard` on every touched `src/` file.
8. Report which invariant is now guaranteed and which remain future-phase work.

## Core Architecture Rules

### Computation motifs

- `computationType` is mandatory for every module archetype.
- Prefer explicit motif contracts over generic node behavior with hidden flags.
- Keep motif semantics stable enough that DNA versioning can reason about them.

### Deterministic development

- Stable ordering must be defined for modules, rule passes, edges, and memory
  slots.
- Build pipeline steps should remain explicit: substrate, rule passes, indirect
  connectivity, materialize.

### Memory tiers

- Short-term, medium-term, and long-term memory must stay conceptually separate.
- Capacity growth and pruning belong to lifecycle policy, not ad hoc runtime
  mutation.

### Neuromodulation

- Model fast mode switching separately from slow structural adaptation.
- Broadcast semantics should be low-dimensional, zone-aware, and cheap relative
  to structural edits.

### Reproduction

- Keep parthenogenesis, polyandric, and standard sexual paths explicit.
- Queen-template priority and patch-region assignment must be deterministic.
- Optional epigenetic priors must remain weak and decaying, not hidden weight
  inheritance.

### Collective-intelligence primitives

- The core owns shared-field semantics and stigmergic trace rules.
- Benchmark or demo owners decide environment-specific tasks, maps, and scoring.

## Focus Scoring Reminder

The plan already proposes a core heuristic:

$$
focus(m) = w_u\,\hat{util}(m) + w_r\,\hat{rewardDelta}(m) + w_n\,\hat{novelty}(m) + w_s\,\hat{stabilityAge}(m) - w_c\,\hat{wiringCost}(m)
$$

Treat this as a policy surface that should remain inspectable and testable, not
as hidden tuning noise.

## Validation Cadence

- Deterministic repeated-build hash tests.
- Lifecycle stage transition tests with measurable guards.
- Budget enforcement and rollback tests.
- Reproduction-mode correctness tests.
- Memory-tier correctness tests.
- Opt-in isolation tests proving classic NEAT is unchanged.

## Guardrails

- Do not let demo-specific assumptions leak into `NGE_DNA` or motif semantics.
- Do not encode explicit phenotype details into DNA when a generator-level prior
  would suffice.
- Do not mix fast behavioral switching with slow structural morphogenesis.
- Do not hide nondeterministic tie-breaks inside lifecycle or reproduction logic.
- Do not let NGE features mutate classic NEAT behavior when disabled.

## Expected Final Output

A strong NGE core pass should report:

- the phase or primitive targeted,
- the invariant or contract added,
- whether the work is algorithm-core or benchmark-adjacent,
- focused validation results,
- coverage results for touched `src/` files,
- any remaining phase dependencies from the memory plan.
