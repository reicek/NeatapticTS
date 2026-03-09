# flappy_bird SOLID Split

Purpose

- Keep `test/examples/flappy_bird` as the reference demo for SOLID, modular example work.
- Keep this document resumable across sessions.
- Keep this document high level only.
- Keep implementation details, temporary file maps, and short-lived decisions in the chat session rather than in this file.

Progress rules

- Use `[]` for a step that is not yet completed.
- Use `[DONE]` immediately after finishing a step.
- Update this document whenever the step order changes, a step is added, a step is removed, or the overall plan shifts.
- Keep completed steps in place so progress is visible when resuming later.
- Do not expand this file with temporary implementation detail; only record durable high-level progress.

Split standard

- Large or multi-responsibility demo files should not grow into a wider pile of sibling files at the same folder level.
- When an area becomes a real subsystem, move it into a dedicated folder that keeps orchestration, contracts, helpers, constants, errors, and services together.
- Keep the main exported module file as the easiest entry point and keep it orchestration-first.
- Prefer dedicated subfolders once a subsystem has clear internal categories.
- Preserve domain-scoped constants instead of recentralizing them.

Naming pattern

- Main module pattern:
- `module/module.ts`
- `module/module.utils.ts`
- `module/module.types.ts`
- `module/module.errors.ts`
- `module/module.services.ts`
- `module/module.constants.ts`

- Nested sub-module pattern:
- `module/sub-module/module.sub-module.ts`
- `module/sub-module/module.sub-module.utils.ts`
- `module/sub-module/module.sub-module.types.ts`
- `module/sub-module/module.sub-module.errors.ts`
- `module/sub-module/module.sub-module.services.ts`
- `module/sub-module/module.sub-module.constants.ts`

Execution steps

- [DONE] Step 1: Audit `browser-entry` utility surfaces and confirm which broad files still need dedicated subfolder splits.
- [DONE] Step 2: Continue the `browser-entry/playback/` split so playback orchestration, rendering support, and snapshot interpretation stay in focused module-owned files.
- [DONE] Step 3: Continue the `browser-entry/visualization/` split so visualization orchestration, layout, rendering, and interpretation responsibilities stay separate.
- [DONE] Step 4: Continue the `browser-entry/network-view/` split so layout, color semantics, labeling, and draw-time interpretation do not collapse into one broad utility surface.
- [DONE] Step 5: Continue the `browser-entry/host/` split so DOM assembly stays compositional and host setup does not keep accumulating in one utility file.
- [DONE] Step 6: Keep `trainer/trainer.ts` orchestration-first by moving reporting, policy, planning, and stop-control details behind trainer-owned helpers and services.
- [DONE] Step 7: Keep the worker entry surface thin by preserving `flappy-evolution-worker` as a protocol-first boundary with playback, simulation, evolution, and runtime details behind worker-owned modules.
- [DONE] Step 8: Continue splitting `simulation-shared` observation assembly so shared observation logic does not reform into a broad policy hub.
- [DONE] Step 9: Recheck the browser visualization and network-view path for responsibility drift after the earlier splits, and realign any overlap back into explicit module boundaries.
- [DONE] Step 10: Validate the final shape by checking naming consistency, folder ownership, generated-doc expectations, and TypeScript/build health.
- [DONE] Step 11: Extract playback background composition into a dedicated background submodule so sky, horizon, and future ground parallax can evolve independently.
- [DONE] Step 12: Extract host resize orchestration, layout math, and DOM mutation helpers into a dedicated `host/resize/` submodule so responsive layout policies evolve without bloating the host root.
- [DONE] Step 13: Extract playback starfield contracts and tile-render mechanics into dedicated starfield-owned files so cache orchestration stays thin and reusable.
- [DONE] Step 14: Extract evaluation rollout orchestration into a dedicated `evaluation/rollout/` submodule so runtime state, policy, shaping helpers, and rollout-local contracts evolve behind one stable public entry.

Done criteria

- Browser runtime files read as orchestration over focused services rather than large utility buckets.
- Trainer and worker entrypoints stay thin, with domain logic living behind explicit service boundaries.
- Visualization, observation, playback, and host setup each evolve independently without overlapping responsibility.
- Dedicated folders, file names, and subfolder names continue following one predictable pattern instead of drifting into ad hoc layouts.
- The demo remains the stronger reference shape that other examples can copy and resume cleanly.
