# Plans Index

This file is a lightweight index for the `plans/` folder.

Purpose:

- help agents and contributors choose the right plan document quickly,
- keep architectural work aligned with roadmap intent,
- avoid loading the entire `plans/` directory into context.

Status convention:

- Every roadmap-tracked plan file should expose a top-level status line immediately under the title.
- Use the exact format `**Status:** [DONE]`, `**Status:** [WIP]`, or `**Status:** [PLANNED]`.
- Keep that top-level status aligned with [plans/Roadmap.md](Roadmap.md), which is the authoritative source for cross-plan sequencing and active priority.
- Completed plans should prefer a short closed tracker plus a same-boundary
  `.logs.md` companion file for audit history and durable milestone evidence.
- Detailed phase-local status notes can still appear deeper in a plan when they add useful implementation detail.

Recommended reading order:

1. Read `plans/README.md` for the map.
2. Read `plans/Roadmap.md` when the task depends on sequencing, current phase status, or cross-plan priority.
3. Read only the single most relevant detailed plan.
4. Read one additional related plan only when the task clearly spans two initiatives.
5. Read `plans/neat.plans.md` for the core NEAT direction and current engineering baseline when the task touches architecture or evolutionary correctness.

Selection guide:

- `plans/analyze-trace-solid-split.plans.md`: completed trace-analyzer tooling split and reopen point for future script-boundary work.
- `plans/architecture-solid-split.plans.md`: completed architecture folderization and docs follow-through baseline, kept as the reopen point for future facade-removal or ownership audits in `src/architecture`.
- `plans/neat.plans.md`: core NEAT correctness, innovation tracking, crossover alignment, speciation invariants.
- `plans/Roadmap.md`: dependency-aware execution order across all initiatives.
- `plans/Architecture_Primitives_Node_Group_Layer.md`: first-class architecture-building primitives such as nodes, groups, and layers.
- `plans/asciiMaze-typescript-repair.plans.md`: completed `asciiMaze` TypeScript repair baseline and reopen point for future diagnostics.
- `plans/Browser_Build_and_CDN_Distribution.md`: browser packaging, CDN usage, and distribution ergonomics.
- `plans/Construct_From_Parts_Graph_Assembly.md`: deterministic graph assembly and validated network construction from parts.
- `plans/ES2023 migration`: repository-wide ES2023 syntax and modernization lane.
- `plans/Evolution_Training_Interoperability_Contracts.md`: contracts between evolution workflows and gradient-based training.
- `plans/Flappy_Bird_Folder_Documentation_Pass.md`: completed Flappy Bird folder documentation baseline and reopen point for example-docs follow-up.
- `plans/generate-docs-solid-split.plans.md`: completed docs-generator tooling split and reopen point for future `scripts/generate-docs/` work.
- `plans/HyperEvoDevoMorphoNEAT.md`: evo-devo and morphology-oriented research direction.
- `plans/HyperEvoDevo_Racing_Curriculum_and_Behavioral_Drives.md`: Hyper follow-on racing benchmark plan with a rich sensorium, early optimal-line guidance, behavioral drives, and competitive self-play racecraft.
- `plans/Interactive_Examples_and_Learning_Path.md`: runnable examples, onboarding flow, and learning-path improvements.
- `plans/Memory_Optimization.md`: scaling, memory layout, and strategies for very large networks.
- `plans/methods-docs.plans.md`: completed educational-docs lane for `src/methods` and reopen point for methods documentation drift.
- `plans/methods-solid-split.plans.md`: completed structural split lane for `src/methods` and reopen point for later refactors.
- `plans/neat-docs.plans.md`: completed educational-docs baseline for root-facing NEAT surfaces, kept as the reopen point for future README-opening drift or generator-ordering regressions.
- `plans/readme-first-section-pass.plans.md`: completed repo-wide README-opening baseline across `src/` and `test/examples/`; use the matching `.logs.md` file for audit history and reopen the plan only if a changed boundary or new README surface reopens the lane.
- `plans/neat-test-surface-repair.plans.md`: completed NEAT public test-surface repair baseline and reopen point for future compatibility regressions.
- `plans/Network_Visualization_Export_Schema.md`: stable export schema for visualization and inspection tooling.
- `plans/ONNX_EXPORT_PLAN.md`: ONNX export/import architecture and rollout phases.
- `plans/Population_Save_Resume_and_Checkpointing.md`: checkpointing, persistence, save/resume workflows.
- `plans/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md`: prebuilt architecture constructors and sequence-oriented builders.
- `plans/render-docs-html-solid-split.plans.md`: completed HTML docs renderer split and reopen point for future docs-site tooling work.
- `plans/src-no-explicit-any-cleanup.plans.md`: completed `src/` strict-typing cleanup baseline for `@typescript-eslint/no-explicit-any`, kept as the reopen point if later refactors reintroduce debt.
- `plans/Stable_Activation_Ordering_and_Explicit_IO_Roles.md`: deterministic execution ordering and explicit input/output roles.
- `plans/Standalone_Inference_Export.md`: dependency-free exported inference runtime.
- `plans/Turnkey_Multithread_Evaluation_API.md`: parallel evaluation API for Node and browser workers.
- `plans/utils-docs.plans.md`: completed educational-docs lane for `src/utils` and reopen point for later documentation drift.
- `plans/Worker_Friendly_Network_Serialization_Fastpath.md`: fast serialization path for worker-based evaluation.

Task-to-plan trigger phrases:

- trace analyzer, Chrome trace, Perfetto report script, `trace:analyze`, tooling split reopen: `plans/analyze-trace-solid-split.plans.md`
- architecture split, folderization, orchestration-first cleanup in `src/architecture`: `plans/architecture-solid-split.plans.md`
- NEAT correctness, innovation IDs, crossover, compatibility distance, speciation: `plans/neat.plans.md`
- roadmap, sequence, dependency order, what comes first: `plans/Roadmap.md`
- architecture builder, layer API, node/group primitives: `plans/Architecture_Primitives_Node_Group_Layer.md`
- asciiMaze TypeScript diagnostics, example test compile failures, reopen asciiMaze repair: `plans/asciiMaze-typescript-repair.plans.md`
- construct from parts, graph assembly, deterministic builder: `plans/Construct_From_Parts_Graph_Assembly.md`
- browser bundle, CDN, browser-first usage: `plans/Browser_Build_and_CDN_Distribution.md`
- ES2023, immutable array methods, modernization pass, syntax cleanup, migration sequencing: `plans/ES2023 migration`
- Flappy Bird docs pass, example folder documentation, generated README quality for Flappy: `plans/Flappy_Bird_Folder_Documentation_Pass.md`
- generate-docs, folder README generation, docs.order, docs generator split: `plans/generate-docs-solid-split.plans.md`
- ONNX, import/export interoperability: `plans/ONNX_EXPORT_PLAN.md`
- methods docs, methods README quality, educational-docs for methods: `plans/methods-docs.plans.md`
- methods split, `src/methods` refactor, methods folderization: `plans/methods-solid-split.plans.md`
- NEAT docs, generated README quality, NEAT documentation lane: `plans/neat-docs.plans.md`
- README opening, first section, chapter intro quality, repo-wide README openings, `src` plus `test/examples` docs pass: `plans/readme-first-section-pass.plans.md`
- NEAT test surface, public type compatibility, root facade repair: `plans/neat-test-surface-repair.plans.md`
- explicit any, no-explicit-any, strict typing, src lint cleanup follow-up, reopen completed cleanup lane: `plans/src-no-explicit-any-cleanup.plans.md`
- HTML docs renderer, sidebar, Mermaid validation, docs site tooling split: `plans/render-docs-html-solid-split.plans.md`
- visualization, schema, inspect network shape: `plans/Network_Visualization_Export_Schema.md`
- checkpoint, resume, save population: `plans/Population_Save_Resume_and_Checkpointing.md`
- workers, threads, parallel evaluation: `plans/Turnkey_Multithread_Evaluation_API.md`
- worker serialization, transfer cost, fastpath: `plans/Worker_Friendly_Network_Serialization_Fastpath.md`
- standalone runtime, exported inference file: `plans/Standalone_Inference_Export.md`
- activation order, IO roles, deterministic execution: `plans/Stable_Activation_Ordering_and_Explicit_IO_Roles.md`
- memory pressure, large networks, compact storage: `plans/Memory_Optimization.md`
- hybrid evolution plus training, optimizer handoff: `plans/Evolution_Training_Interoperability_Contracts.md`
- preconfigured models, MLP, LSTM, GRU, NARX builders: `plans/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md`
- examples, tutorials, learning path, onboarding: `plans/Interactive_Examples_and_Learning_Path.md`
- evo-devo, morphology, research-heavy extensions: `plans/HyperEvoDevoMorphoNEAT.md`
- racing benchmark, optimal line, behavioral drives, overtaking, self-play racecraft, Hyper follow-on: `plans/HyperEvoDevo_Racing_Curriculum_and_Behavioral_Drives.md`
- utils docs, utility README quality, educational-docs for `src/utils`: `plans/utils-docs.plans.md`

Working rule:
When a task changes code in a way that could conflict with one of these plans, mention the relevant plan in the working notes or final summary and call out any mismatch instead of silently diverging from the roadmap.

Authority rule:
For cross-plan sequencing and active priority, `plans/Roadmap.md` is authoritative. Detailed plans may contain future-state inventory, local sub-phases, or stale status notes; if a detailed plan appears to pull later-phase work earlier, follow `plans/Roadmap.md` unless both files are updated together.
