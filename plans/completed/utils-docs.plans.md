# Utils Docs Plan

**Status:** [DONE]

## Scope

- Educational-docs pass for the shared utility chapters under `src/utils/`.
- Raise the generated utility openings without changing runtime behavior.

## Final state

- The utility chapter openings now explain memory heuristics and snapshot
  layering more clearly instead of reading as implementation shelves.
- Mermaid-supported diagrams are part of the generated utility documentation
  path where they materially improve comprehension.
- No active backlog remains; this plan is now a reopen point for future utility
  documentation drift.

## Audit summary

- Documentation updates were validated with `npm run docs` and
  `npx tsc --noEmit -p tsconfig.json`.
- A pre-existing Mermaid rendering issue in another docs surface was also fixed
  during this lane to keep global docs generation healthy.

## Reopen conditions

- New utility modules are added under `src/utils/`.
- Utility README openings drift below the current documentation bar.
- Mermaid or generated-doc ownership regressions reappear.

## Audit log

- Durable completion notes now live in [utils-docs.logs.md](utils-docs.logs.md).
