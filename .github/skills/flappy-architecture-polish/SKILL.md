---
name: flappy-architecture-polish
description: 'Tune, stabilize, and instrument one Flappy Bird architecture profile across browser runtime, warm-start, worker evaluation, and durable long-run probes so the same polish loop can be reused for LSTM, GRU, NARX, MLP, or Sparse profiles.'
argument-hint: 'Describe the Flappy architecture profile, the current symptom or polish target, whether a durable probe already exists, and whether this pass is reconnaissance, implementation, or rerun validation.'
user-invocable: true
disable-model-invocation: false
---

# Flappy Architecture Polish Playbook

Use this skill when a Flappy Bird architecture profile needs a disciplined
polish pass that improves the live browser-worker path and leaves behind a
durable way to rerun the same evaluation later.

This skill is the canonical workflow for architecture-specific Flappy Bird
polish in NeatapticTS. It owns the repeatable tuning loop for browser runtime
budget, generation-zero warm-start, worker-side evaluation fairness, durable
progress probes, and the fast regressions that should remain after the pass.

When the workstream needs a durable tracker, `tracker-handoff` owns the plan
and log format. When the polish work touches roadmap-sensitive architecture or
runtime direction, `plan-alignment` owns plan selection and terminology.

## When to Use

- The user wants to polish or retune one Flappy architecture profile such as
  `mlp`, `random-sparse`, `narx`, `gru`, or `lstm`.
- Manual browser testing improved, but the result is not yet backed by a
  durable empirical rerun surface.
- A recurrent profile regresses after generation-zero warm-start or stalls on
  one lucky rollout lane.
- The browser demo needs architecture-specific budget or fairness tuning
  without drifting into demo-only hacks.
- A prior LSTM-style polish loop should be repeated for another profile.
- A long-running Jest probe was useful for investigation, but the durable end
  state should be a CLI or scriptable probe rather than a multi-minute test.

## Scope Boundary

This skill owns the repeatable Flappy architecture-polish workflow for:

- browser runtime population and elitism budgets,
- worker-side shared-seed evaluation policy,
- generation-zero warm-start rollout and teacher strategy,
- durable long-run progress probe design,
- fast regression coverage for the tuning boundaries,
- validation cadence and close-out expectations.

This skill does not replace `plan-alignment` for roadmap selection, and it does
not replace `tracker-handoff` for tracker structure.

## Task Packet

Pass a compact packet that includes:

- architecture profile id,
- current symptom or polish goal,
- whether the pass is reconnaissance, implementation, or rerun validation,
- existing durable probe command or path if one already exists,
- whether the issue looks like browser stutter, first-pipe discovery,
  post-warm-start regression, multi-seed instability, or docs/tooling drift,
- whether a durable tracker already exists,
- required final validations.

Compact example:

```text
Use flappy-architecture-polish for the GRU profile in examples/flappy_bird.
Goal: reduce post-warm-start regression and leave behind a reusable progress probe.
Mode: implementation plus rerun validation.
Current durable probe: none yet.
Likely issue: worker selection fairness across generations.
Final validation: targeted Jest, npm run build, focused eslint, then strict probe rerun.
```

## README-First Discovery Order

Before deep code reads, prefer this order:

1. `examples/flappy_bird/README.md`
2. `examples/flappy_bird/browser-entry/README.md` when runtime or playback is involved
3. `examples/flappy_bird/flappy-evolution-worker/README.md` when worker,
   warm-start, or probe work is involved
4. `examples/flappy_bird/simulation-shared/README.md` when observation or
   teacher logic is involved
5. only then the smallest relevant source files

Use the README pass to decide whether the issue is really in runtime budget,
warm-start, worker scoring, observation contract, or docs/probe drift.

## Current Durable Boundaries

For the current LSTM polish lane, the durable boundaries are:

- `examples/flappy_bird/browser-entry/runtime/runtime.population-budget.ts`
- `examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.runtime.service.ts`
- `examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.warm-start.service.ts`
- `examples/flappy_bird/probeFlappyArchitectureProgress.ts`
- `package.json` probe scripts

Future architecture passes should start by deciding whether those same
boundaries apply or whether the problem is actually in a different owner-local
surface.

## Required Workflow

1. Confirm the architecture profile and the success signal.
   - Typical signals: stable first-pipe discovery, stronger late-phase frames,
     stronger late-phase pipe pressure, and multi-seed stability.

2. Decide whether the existing probe should be reused, generalized, or cloned.
   - If the metrics contract is the same, prefer one shared runner over several
     unrelated probes.
   - If the current runner is intentionally architecture-local, keep the output
     contract aligned with the existing polish lane.

3. Establish a baseline before changing behavior.
   - Prefer a durable CLI or scriptable probe over a multi-minute Jest test.
  - The current generic runner is `npm run flappy:architecture:progress`.
  - The LSTM alias remains `npm run flappy:lstm:progress`.
   - For npm-run named flags, preserve `npm_config_*` fallbacks because npm 11
     warns on unknown flags and still forwards them through config env vars.

4. Keep example-specific probes inside the example boundary.
   - If the probe imports Flappy example code, place it under
     `examples/flappy_bird/`, not under `scripts/`.
   - Expose it through `package.json` with a prebuild step when needed.
   - This avoids dragging example-only imports into the docs-script TypeScript
     compile surface.

5. Tune the smallest owner-local boundary first.
   - Typical order: runtime budget, warm-start rollout plan, worker shared-seed
     policy, then probe/reporting contract.
   - Only change the observation contract when it is the real root cause.

6. Treat demo symptoms as shared-runtime evidence first.
   - If the issue is really in worker selection, warm-start scoring, or shared
     defaults, fix it there instead of papering over it in a demo-local path.

7. Add or refresh fast regressions as each tuning boundary lands.
   - Good targets: runtime budget resolver, worker seed rotation or seed-count
     logic, warm-start scoring plans, probe helper logic, and pure summary
     computations.
   - Do not keep multi-minute progress probes in the normal Jest suite once the
     durable CLI or equivalent runner exists.
   - If any `src/` files were modified during tuning, run `coverage-guard` on
     each changed file to enforce 100% coverage in all four categories before
     closing the step.

8. Preserve the durable progress-check contract unless there is a documented
   reason to revise it.
   - Reuse these check names when the same success story applies:
     - `clearsFirstPipe`
     - `improvesOverOwnEarlyFrames`
     - `improvesOverOwnEarlyPipePressure`
     - `finishesWithStableMultiSeedImprovement`
   - Preserve the midpoint-based early/late phase split unless the change is
     intentional and called out explicitly.

9. Keep strict pass/fail mode opt-in.
   - Default observational runs should still print the checks and summary.
   - Strict mode should be an explicit CLI flag so failed checks can become a
     nonzero exit code only when the user wants a gate.

10. Close with the full validation cadence.
   - Targeted Jest or boundary-local tests first
   - `npm run build`
   - focused `eslint` on changed files
   - `npm run docs` when package scripts, docs tooling, or JSDoc surfaces moved
   - final empirical rerun in non-strict or strict mode as requested

## Probe Output Contract

A durable polish probe should normally emit:

- one header line describing architecture, generation count, and seed counts,
- compact periodic generation snapshots,
- a `checks=` JSON line,
- a `status=` line with stable failed-check names,
- a final summary JSON line.

If the probe supports strict mode, the strict flag should only affect exit code
behavior, not hide the summary.

## Guardrails

- Do not accept manual demo improvement as sufficient evidence when the task is
  explicitly about polish or repeatability.
- Do not place example-specific probes under `scripts/` when they import
  example code.
- Do not leave a multi-minute Jest probe in the normal suite after the durable
  probe contract has been ported into a CLI.
- Do not silently invent a different success metric story for one profile when
  the existing check contract still applies.
- Do not trust a single display seed as proof of multi-seed stability.
- Do not overfit recurrent worker evaluation to one eternal rollout lane;
  consider per-generation rotation when the profile stalls or regresses.
- Do not skip `npm run build` after changing package scripts or probe entrypoints.
- Do not forget `npm run docs` when package scripts or docs/tooling surfaces are
  affected.
- Do not improvise tracker structure; use `tracker-handoff` when the pass needs
  durable continuation.
- Do not bypass `plan-alignment` when the tuning work touches roadmap-sensitive
  runtime semantics or architectural intent.

## Validation Rules

Preferred cadence:

- During implementation:
  - narrow tests for the changed boundary,
  - cheap probe smoke runs such as one-generation runs,
  - file-level lint when the surface is isolated.
- After the boundary is green:
  - `npm run build`,
  - focused `eslint`,
  - `npm run docs` when package or docs-tooling surfaces changed.
- For close-out:
  - observational rerun when the user wants data,
  - strict rerun when the user wants a go/no-go gate.

Current example commands:

```text
npm run flappy:architecture:progress -- --profile=gru --generations=30 --report-every=5 --validation-seeds=3
npm run flappy:architecture:progress -- --profile=lstm --generations=30 --report-every=5 --validation-seeds=3 --require-pass=true
```

## Expected Final Output

A strong run should report:

- which architecture profile was polished,
- which owner-local boundaries changed,
- whether the durable probe was reused, generalized, or newly added,
- which fast regressions now cover the tuning logic,
- the validation results,
- whether the final empirical probe was observational or strict,
- any remaining risks or next-step polish targets.