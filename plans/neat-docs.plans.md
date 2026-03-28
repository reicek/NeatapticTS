# NEAT Root Docs Plan

**Status:** [WIP]

## Scope

This plan tracks the educational-docs pass for the root NEAT public surface.
The source of truth is the generated documentation fed by [src/neat.ts](../src/neat.ts),
which renders into [src/README.md](../src/README.md) after `npm run docs`.

Primary reader:

- first-time NeatapticTS readers who want to understand how the top-level NEAT controller works before diving into chaptered internals.

Primary surfaces:

- [src/neat.ts](../src/neat.ts)
- [src/README.md](../src/README.md)

Out of scope for this pass:

- hand-editing generated READMEs under `src/`
- adding external images or citations unless a concrete teaching gap requires them
- broad rewrites of every `src/neat/**` chapter README

## Assumptions

- The generated root NEAT story is expected to live in [src/README.md](../src/README.md), not in [src/neat/README.md](../src/neat/README.md).
- The highest-leverage fix is richer JSDoc in [src/neat.ts](../src/neat.ts), especially module-level and class-level introductions.
- External sources and Wikipedia media are optional, not mandatory. If none are needed, the safest compliant outcome is to use no external media and no external citations.

## Session Log

### Foundation

- Strengthened [src/neat.ts](../src/neat.ts) so the generated root story in [src/README.md](../src/README.md) reads as a clearer controller overview.

### Generator follow-up

- The docs generator work is complete enough for now: [scripts/generate-docs/generate-docs.ts](../scripts/generate-docs/generate-docs.ts) supports per-folder ordering controls through `docs.order.json`, and the shared-speciation pilot is already live in [src/neat/speciation/shared/docs.order.json](../src/neat/speciation/shared/docs.order.json).

### Latest completed chapter work

- Recent source-first passes strengthened [src/neat/shared/neat.shared.types.ts](../src/neat/shared/neat.shared.types.ts), [src/neat/neat.constants.ts](../src/neat/neat.constants.ts), [src/neat/cache/cache.ts](../src/neat/cache/cache.ts), [src/neat/compat/compat.ts](../src/neat/compat/compat.ts), [src/neat/harness/neat.harness.types.ts](../src/neat/harness/neat.harness.types.ts), [src/neat/init/neat.init.ts](../src/neat/init/neat.init.ts), [src/neat/helpers/neat.helpers.ts](../src/neat/helpers/neat.helpers.ts), and [src/neat/rng/rng.ts](../src/neat/rng/rng.ts); the RNG root now distinguishes caller-owned randomness from controller-owned replay state more explicitly, and the root [src/README.md](../src/README.md) plus [src/neat.ts](../src/neat.ts) story still holds up against the current pedagogical standard. Docs regeneration and TypeScript validation should be kept on every new pass.
- A bounded root-surface polish split then moved the root compatibility alias shelf into [src/neat/neat.types.ts](../src/neat/neat.types.ts), moved the public defaults shelf into [src/neat/neat.defaults.constants.ts](../src/neat/neat.defaults.constants.ts), and centralized the empty diversity fallback in [src/neat/diversity/diversity.ts](../src/neat/diversity/diversity.ts); [src/README.md](../src/README.md) still presents the stable public root story while [src/neat/README.md](../src/neat/README.md) now carries the deeper root-types and root-defaults chapter docs.

Remaining gaps:

- Most remaining weaknesses are now small chapter-quality revisits, not untouched surfaces.
- Generator limits still show up in some larger or re-export-heavy chapters, so only use `docs.order.json` when ordering is the real blocker.
- Keep prioritizing compact runtime, facade, or policy boundaries that still read more like shelves than chapters.

Next step:

- Re-read [src/neat/lineage/README.md](../src/neat/lineage/README.md) and [src/neat/lineage/lineage.ts](../src/neat/lineage/lineage.ts) to decide whether the lineage root is now the next weakest compact controller-facing boundary after the RNG pass.

## Handoff Prompt

```text
Continue the educational-docs pass for the next adjacent NEAT chapter using plans/neat-docs.plans.md as the source of truth.

Current state:
- The generator pause is complete enough; use [plans/pedagogical-docs.plans.md](../plans/pedagogical-docs.plans.md) only if a new generator gap appears.
- Recent local passes already strengthened the shared-contracts, root-constants, cache-root, compatibility-root, harness-root, init-root, helpers-root, and RNG-root chapters, and the root [src/README.md](../src/README.md) story still holds up against the newer pedagogical standard.

What to verify next:
- Re-read [src/neat/lineage/README.md](../src/neat/lineage/README.md) and [src/neat/lineage/lineage.ts](../src/neat/lineage/lineage.ts).
- If the lineage root still feels flatter than nearby compatibility and telemetry-facing chapters, improve the source JSDoc in [src/neat/lineage/lineage.ts](../src/neat/lineage/lineage.ts) and add a chart only if it materially clarifies ancestry evidence versus structural-distance evidence faster than the current prose alone.
- Otherwise move to the next smallest bridge, facade, or policy boundary that still reads like an API shelf, but keep the selection root-first instead of jumping straight into a subfolder.
- Keep using `docs.order.json` only where compiled order is the real obstacle.

If more work is needed:
- Make source-first JSDoc improvements in the next weakest NEAT chapter, then run `npm run docs` and `npx tsc --noEmit -p tsconfig.json`.
- Do not hand-edit generated `src/**/README.md` files.
- Update [plans/neat-docs.plans.md](../plans/neat-docs.plans.md) with only the latest minimal pass summary.
```
