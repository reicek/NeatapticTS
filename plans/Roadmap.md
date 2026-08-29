# Roadmap

Active workstreams and their plan trackers.

## Active demo workstreams

- Neon_Shooter_NGE_Demo: `Neon_Shooter_NGE_Demo.plans.md` - Phase 4 [DONE] | Phase 5 [DONE] (archived in `plans/completed/`)
- Neatenstein Auto / NEAT Mode Wiring: `plans/completed/neatenstein-auto-neat-mode.plans.md` — Phases 0-9 [DONE] (archived in `plans/completed/`) — wiring evolution harness into live game loop
- Neatenstein Vision System Wiring Fixes: `neatenstein-vision-wiring-fixes.plans.md` — Phase 1 [DONE] (archived in `plans/completed/`) — pragmatic fix plan for vision-aware fallback, wave-clear bug, fire-gate threading, dead shooter deletion, PBRS/sensorHistory cleanup
- Neatenstein Firing / Sensor System: `neatenstein-firing-sensor-system.plans.md` — Phases 1-5 [DONE] (archived in `plans/completed/`) — fix NEAT player firing: shared constants, fitness replacement, sensor expansion 12→15, reward redesign, fire gating
- Neatenstein Hunter Bugfix: `plans/completed/neatenstein-hunter-bugfix.plans.md` — Phase 1 [DONE] (archived in `plans/completed/`) — fix three live hunter bugs: center spin, wall vision, wave spawn at 14 kills
- Neatenstein Scroll/Ammo: `plans/completed/neatenstein-scroll-ammo.plans.md` — Phases 1-2 [DONE] (archived in `plans/completed/`) — smooth AI scroll/turn with accel/decel and add ammo-pickup path awareness + bonus
- Neatenstein Ultimate Quality Upgrade: `plans/completed/neatenstein-ultimate-quality-upgrade.plans.md` — Phases A-C [DONE] (archived in `plans/completed/`) — comprehensive multi-perspective upgrade: maze generation overhaul, per-frame allocation elimination, NGE hero evolution, enemy NEAT evolution, raycasting bug fixes, enemy parallelism, code quality, shader raycaster, MAP-Elites/CMA-ES, rendering polish

## Standalone Meta-Workflow Lane

- Docs Quality Metrics Gap: `plans/docs-quality-metrics-gap.plans.md` — Phase 1 [DONE] (active tracker in `plans/`) — close the gaps in `npm run docs:quality:metrics` so the pipeline passes clean (coverage scope alignment, expanded scanner dimensions, real timestamp, coverage gating, symbol deduplication, manifest cleanup)
- Root Folder Cleanup: `plans/Root_Folder_Cleanup.plans.md` — Phase 1 [DONE] (closed tracker in `plans/`) — audit and remove temporal/debug/snapshot files from repo root and relocate root scripts/docs to proper subdirectories
- RAG Index Freshness Strategy: `plans/completed/RAG_Index_Freshness_Strategy.plans.md` — Phases 0-3 [DONE] (archived to `plans/completed/`) — replace the content-blind 24h age gate in `rag-index/validate-index.mjs` with a per-family, content-hash-only freshness model and synchronous post-save plan reindex hook

Reserved for agentic workflow infrastructure and cross-cutting tooling plans.
