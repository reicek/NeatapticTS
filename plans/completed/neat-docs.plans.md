# NEAT Root Docs Plan

**Status:** [DONE]

## Scope

- Educational-docs pass for root-facing NEAT surfaces and closely related
  generated chapters.
- Strengthen chapter openings without changing NEAT runtime behavior.

## Final state

- The NEAT root documentation surfaces now present a clearer controller and
  chapter-map story instead of a compatibility-first opening.
- Root-facing chapters such as RNG and lineage now open with stronger
  educational framing, diagrams, and examples where they mattered.
- No active backlog remains; this plan is now a reopen point for future
  root-facing NEAT documentation drift.

## Audit summary

- Documentation updates were validated with `npm run docs` and
  `npx tsc --noEmit -p tsconfig.json`.
- The final state is stable enough to stop being a documentation blocker.

## Reopen conditions

- Root-facing NEAT chapters drift below the current educational-docs bar.
- Generated opening ownership regresses after later structural work.
- A major NEAT public-surface change needs renewed root chapter framing.

## Audit log

- Durable completion notes now live in [neat-docs.logs.md](neat-docs.logs.md).
