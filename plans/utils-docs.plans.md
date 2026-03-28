# Utils Docs Plan

**Status:** [DONE]

## Scope

This plan tracks educational-docs passes for the small shared utility surfaces
under [src/utils/README.md](../src/utils/README.md).

Primary reader:

- readers who want to understand support utilities as teaching tools rather
  than as unframed implementation leftovers.

Primary surfaces:

- [src/utils/memory.ts](../src/utils/memory.ts)
- [src/utils/memory.utils.ts](../src/utils/memory.utils.ts)
- [src/utils/README.md](../src/utils/README.md)

## Session Log

### Latest completed chapter work

- Strengthened the root [src/utils/memory.ts](../src/utils/memory.ts) opening so
  the generated [src/utils/README.md](../src/utils/README.md) now explains why
  heuristic memory instrumentation exists, which questions it answers, and how
  to read the snapshot in layers.
- Added a real module introduction to
  [src/utils/memory.utils.ts](../src/utils/memory.utils.ts) so the helper file
  reads as aggregation and snapshot mechanics rather than a flat export shelf.
- Added Mermaid teaching diagrams to both module openings.
- Regenerated docs and re-ran `npx tsc --noEmit -p tsconfig.json` successfully.
- Fixed an existing Mermaid rendering failure in
  [src/neat/init/neat.init.ts](../src/neat/init/neat.init.ts) that had been
  blocking global docs generation.

Remaining gaps:

- The `src/utils` folder is now small and reasonably aligned; future work here
  is more likely to be refinement than rescue.
- If additional utility modules are added later, keep this plan focused on
  chapter quality instead of growing a long historical log.

Next step:

- Move to the next small cross-cutting folder that still reads more like a
  helper shelf than a chapter, using [src/utils/README.md](../src/utils/README.md)
  as the quality bar for support modules.
