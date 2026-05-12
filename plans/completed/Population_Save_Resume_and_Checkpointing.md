# Population Save/Resume + Checkpointing Plan

**Status:** [DONE]

## Scope

- Freeze a versioned, orchestration-owned checkpoint contract for long-running evolution runs.
- Keep the distinction between exact-resume full checkpoints and best-effort light checkpoints explicit in schema, restore behavior, and public docs.
- Reuse the existing network serialization boundary for graph identity and network parameters instead of inventing a second network wire format inside checkpointing.

## Final state

- The library now ships a closed persistence ladder across population-only snapshots, light checkpoints, and full checkpoints.
- `Neat.exportState()` and `Neat.importState()` keep the strict-versus-best-effort full-checkpoint split explicit for replay-critical runtime and speciation state.
- `Neat.exportLightState()` and `Neat.importLightState()` provide an honest restart contract that preserves the saved restart-scale `popsize` while only restoring a curated elite subset.
- Both full and light checkpoint bundle types now reserve a top-level `extensions?: Record<string, unknown>` bag for downstream metadata without weakening the core resume contract.
- Public docs now teach the persistence decision ladder through the `src/neat/export/` and root `Neat` surfaces; the short-lived standalone checkpoint walkthrough was later retired after proving redundant beside those maintained teaching surfaces.

## Audit summary

- Step 0 and Step 1 closed the owner map, exactness matrix, and replay-critical state inventory around the existing `src/neat/export/` seam.
- Step 2 and Step 3 hardened the full-checkpoint contract with focused tests around adaptive runtime state, strict-versus-best-effort restore policy, and structural failure rules.
- Step 4 added and validated the light-checkpoint contract, including elite-subset export/import, restart-scale `popsize` preservation, and forward evolution from a light-restored controller.
- Step 5 closed with source-first docs, the reserved downstream `extensions` surface, and the original standalone checkpoint example that contrasted population-only, light, and strict full checkpoint flows before its later retirement.

## Reopen conditions

- A future persistence lane needs checkpoint schema versioning beyond the archived v1 contract, such as migrations or compression tiers that materially change restore semantics.
- A downstream system needs stronger checkpoint-owned metadata contracts than the reserved `extensions` bag can safely carry.
- Exact-resume claims need to widen beyond the archived strict full-checkpoint contract.

## Audit log

- See [Population_Save_Resume_and_Checkpointing.logs.md](Population_Save_Resume_and_Checkpointing.logs.md).
