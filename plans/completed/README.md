# Completed Plans Archive

This folder holds terminally closed plan baselines and their matching `.logs.md` audit files.

Use archived plans as reopen points, not as the default starting point for new work.

How to use this archive:

1. Start with [plans/README.md](../README.md) and [plans/Roadmap.md](../Roadmap.md).
2. Open the archived plan that matches the boundary you are reopening.
3. Read the same-boundary `.logs.md` file when you need milestone detail, validation history, or final-state evidence.

Archive selection guide:

- [neat.plans.md](neat.plans.md): Phase 1 proper-NEAT correctness baseline.
- [Stable_Activation_Ordering_and_Explicit_IO_Roles.md](Stable_Activation_Ordering_and_Explicit_IO_Roles.md): deterministic execution ordering and explicit input/output role baseline.
- [Architecture_Primitives_Node_Group_Layer.md](Architecture_Primitives_Node_Group_Layer.md): closed primitive DX baseline for `Node`, `Group`, and `Layer`.
- [Construct_From_Parts_Graph_Assembly.md](Construct_From_Parts_Graph_Assembly.md): closed Phase 2 whole-graph assembly and construct diagnostics baseline.
- [Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md](Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md): closed Phase 2 preconfigured architecture builders baseline (MLP, RandomSparse, NARX, GRU, LSTM) including shared demo profile contract, Flappy Bird and ASCII Maze integration, cross-demo e2e matrix, and profile-aware trainer/worker recurrent evolution settings.
- [Standalone_Inference_Export.md](Standalone_Inference_Export.md): closed Phase 4 standalone export architecture baseline.
- [ES2023 migration](ES2023%20migration): Phase 0 modernization baseline for ES2023 syntax, named errors, and lint enforcement.
- [Flappy_Bird_Folder_Documentation_Pass.md](Flappy_Bird_Folder_Documentation_Pass.md): completed Flappy Bird documentation pass.
- [architecture-solid-split.plans.md](architecture-solid-split.plans.md): completed `src/architecture` split baseline.
- [methods-solid-split.plans.md](methods-solid-split.plans.md): completed `src/methods` split baseline.
- [methods-docs.plans.md](methods-docs.plans.md): completed `src/methods` educational-docs baseline.
- [neat-docs.plans.md](neat-docs.plans.md): completed root-facing NEAT documentation baseline.
- [readme-first-section-pass.plans.md](readme-first-section-pass.plans.md): completed repo-wide README opening pass.
- [utils-docs.plans.md](utils-docs.plans.md): completed `src/utils` educational-docs baseline.
- [neat-test-surface-repair.plans.md](neat-test-surface-repair.plans.md): completed NEAT public test-surface repair baseline.
- [asciiMaze-typescript-repair.plans.md](asciiMaze-typescript-repair.plans.md): completed `asciiMaze` TypeScript repair baseline.
- [generate-docs-solid-split.plans.md](generate-docs-solid-split.plans.md): completed docs-generator split baseline.
- [render-docs-html-solid-split.plans.md](render-docs-html-solid-split.plans.md): completed HTML docs renderer split baseline.
- [analyze-trace-solid-split.plans.md](analyze-trace-solid-split.plans.md): completed trace-analyzer tooling split baseline.
- [src-no-explicit-any-cleanup.plans.md](src-no-explicit-any-cleanup.plans.md): completed `src/` no-explicit-any cleanup baseline.
- [flappy-startup-loading-preview.plans.md](flappy-startup-loading-preview.plans.md): completed Flappy startup loading preview lane.
- [test-colocation-and-root-examples.plans.md](test-colocation-and-root-examples.plans.md): completed test colocation and root examples lane.

Trigger phrases:

- proper NEAT, innovation IDs, crossover, compatibility distance, speciation: `plans/completed/neat.plans.md`
- activation order, explicit IO roles, deterministic scheduling: `plans/completed/Stable_Activation_Ordering_and_Explicit_IO_Roles.md`
- architecture primitives, node/group/layer DX, primitive descriptors: `plans/completed/Architecture_Primitives_Node_Group_Layer.md`
- construct from parts, graph assembly, deterministic builder reopen: `plans/completed/Construct_From_Parts_Graph_Assembly.md`
- preconfigured builders, MLP/NARX/GRU/LSTM reopen, shared profile contract, demo profile integration: `plans/completed/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md`
- standalone runtime, exported inference file, Phase 4 export reopen: `plans/completed/Standalone_Inference_Export.md`
- ES2023, immutable array methods, named errors, no CommonJS in new work: `plans/completed/ES2023 migration`
- methods docs or split reopen: `plans/completed/methods-docs.plans.md`, `plans/completed/methods-solid-split.plans.md`
- README opening drift or docs-generator reopen: `plans/completed/readme-first-section-pass.plans.md`, `plans/completed/generate-docs-solid-split.plans.md`, `plans/completed/render-docs-html-solid-split.plans.md`
- trace analyzer split reopen: `plans/completed/analyze-trace-solid-split.plans.md`
- Flappy docs or startup preview reopen: `plans/completed/Flappy_Bird_Folder_Documentation_Pass.md`, `plans/completed/flappy-startup-loading-preview.plans.md`

Archive rule:

When a new workstream reaches terminal `[DONE]`, keep the compressed plan and the matching `.logs.md` file together in this folder before beginning the next workstream.
