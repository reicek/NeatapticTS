# Roadmap

Completed workstreams and their archived plan trackers. No active or pending plans remain.

## Completed demo workstreams

- Neon_Shooter_NGE_Demo: `plans/completed/Neon_Shooter_NGE_Demo.plans.md` — Phase 4 [DONE] | Phase 5 [DONE] — NGE demo wiring and polish
- Neatenstein Auto / NEAT Mode Wiring: `plans/completed/neatenstein-auto-neat-mode.plans.md` — Phases 0-9 [DONE] — wiring evolution harness into live game loop
- Neatenstein Vision System Wiring Fixes: `plans/completed/neatenstein-vision-wiring-fixes.plans.md` — Phase 1 [DONE] — pragmatic fix plan for vision-aware fallback, wave-clear bug, fire-gate threading, dead shooter deletion, PBRS/sensorHistory cleanup
- Neatenstein Firing / Sensor System: `plans/completed/neatenstein-firing-sensor-system.plans.md` — Phases 1-5 [DONE] — fix NEAT player firing: shared constants, fitness replacement, sensor expansion 12→15, reward redesign, fire gating
- Neatenstein Hunter Bugfix: `plans/completed/neatenstein-hunter-bugfix.plans.md` — Phase 1 [DONE] — fix three live hunter bugs: center spin, wall vision, wave spawn at 14 kills
- Neatenstein Scroll/Ammo: `plans/completed/neatenstein-scroll-ammo.plans.md` — Phases 1-2 [DONE] — smooth AI scroll/turn with accel/decel and add ammo-pickup path awareness + bonus
- Neatenstein Ultimate Quality Upgrade: `plans/completed/neatenstein-ultimate-quality-upgrade.plans.md` — Phases A-C [DONE] — comprehensive multi-perspective upgrade: maze generation overhaul, per-frame allocation elimination, NGE hero evolution, enemy NEAT evolution, raycasting bug fixes, enemy parallelism, code quality, shader raycaster, MAP-Elites/CMA-ES, rendering polish

## Standalone Meta-Workflow Lane

All entries in this lane are completed and archived.

- RAG / Cortex Self-Healing Hooks: `plans/completed/rag-self-heal-hooks.plans.md` — Phases 1-5 [DONE] — automatic call-time detection-and-repair safety net for Cortex/RAG degradation (empty dense embeddings, BM25-only fallback, server spawn failure) with cooldown/backoff state files, background repair reusing existing index automation, and model-facing pause-and-ask guidance
- Docs Quality Metrics Gap: `plans/completed/Docs_Quality_Metrics_Contract_and_Parity.plans.md` — Phase 1 [DONE] — close the gaps in `npm run docs:quality:metrics` so the pipeline passes clean (coverage scope alignment, expanded scanner dimensions, real timestamp, coverage gating, symbol deduplication, manifest cleanup). The active tracker name `plans/docs-quality-metrics-gap.plans.md` was retired; the closed archive is `Docs_Quality_Metrics_Contract_and_Parity.plans.md`.
- Root Folder Cleanup: `plans/completed/Root_Folder_Cleanup.plans.md` — Phase 1 [DONE] — audit and remove temporal/debug/snapshot files from repo root and relocate root scripts/docs to proper subdirectories
- RAG Index Freshness Strategy: `plans/completed/RAG_Index_Freshness_Strategy.plans.md` — Phases 0-3 [DONE] — replace the content-blind 24h age gate in `rag-index/validate-index.mjs` with a per-family, content-hash-only freshness model and synchronous post-save plan reindex hook

Reserved for future agentic workflow infrastructure and cross-cutting tooling plans.
