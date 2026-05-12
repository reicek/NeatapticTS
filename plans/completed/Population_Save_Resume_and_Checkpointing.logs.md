# Population Save/Resume + Checkpointing Log

**Status:** [DONE]

## Audit scope

- Objective: close the Phase 4 checkpointing lane after turning the shipped full-checkpoint seam into a documented, test-backed persistence ladder with population-only snapshots, light checkpoints, and strict full checkpoints.
- Coverage included owner mapping, replay-critical state inventory, strict-versus-best-effort full restore semantics, the light-checkpoint contract, the reserved downstream `extensions` bag, and the source-first docs/example closeout.

## Durable milestones

### [DONE] Owner map and replay-critical state inventory

- Froze the checkpoint owner boundary around `src/neat/export/` and the existing network serialization surface.
- Documented which state is required for strict full resume, which fields are light-only, and which metadata stays outside the core replay contract.

### [DONE] Full-checkpoint restore hardening

- Added focused tests and implementation support for replay-critical runtime state, adaptive controller data, and speciation resume policy.
- Made the strict-versus-best-effort restore split explicit without allowing structural bundle damage to silently downgrade.

### [DONE] Light-checkpoint contract and docs closeout

- Added the light-checkpoint API pair and validated elite-subset restore, saved generation continuity, and restart-scale `popsize` preservation on the next evolve step.
- Reserved a top-level `extensions?: Record<string, unknown>` bag on both full and light checkpoint bundles.
- Refreshed source-first docs and briefly shipped `examples/checkpointResume` as a standalone persistence walkthrough before that example was later retired.

## Controls and evidence

- Focused Jest coverage for `src/neat/export/neat.export.test.ts`.
- Focused Jest coverage for the original `examples/checkpointResume/checkpointResume.test.ts` during the closeout pass.
- Documentation regeneration via `npm run docs`.
- TypeScript validation via `npm run build:ts`.

## Reopen triggers

- A future checkpoint consumer needs a stronger schema versioning or metadata story than the archived v1 contract provides.
- Restore semantics for full or light checkpoints need to change in a way that would invalidate the archived persistence decision ladder.
