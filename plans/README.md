# Plans Index

This file is a lightweight index for the `plans/` folder.

Purpose:

- help agents and contributors choose the right plan document quickly,
- keep architectural work aligned with roadmap intent,
- avoid loading the entire `plans/` directory into context.

Recommended reading order:

1. Read `plans/README.md` for the map.
2. Read `plans/Roadmap.md` when the task depends on sequencing, current phase status, or cross-plan priority.
3. Read only the single most relevant detailed plan.
4. Read one additional related plan only when the task clearly spans two initiatives.
5. Read `plans/neat.plans.md` for the core NEAT direction and current engineering baseline when the task touches architecture or evolutionary correctness.

Selection guide:

- `plans/neat.plans.md`: core NEAT correctness, innovation tracking, crossover alignment, speciation invariants.
- `plans/Roadmap.md`: dependency-aware execution order across all initiatives.
- `plans/Architecture_Primitives_Node_Group_Layer.md`: first-class architecture-building primitives such as nodes, groups, and layers.
- `plans/Browser_Build_and_CDN_Distribution.md`: browser packaging, CDN usage, and distribution ergonomics.
- `plans/Construct_From_Parts_Graph_Assembly.md`: deterministic graph assembly and validated network construction from parts.
- `plans/Evolution_Training_Interoperability_Contracts.md`: contracts between evolution workflows and gradient-based training.
- `plans/HyperEvoDevoMorphoNEAT.md`: evo-devo and morphology-oriented research direction.
- `plans/Interactive_Examples_and_Learning_Path.md`: runnable examples, onboarding flow, and learning-path improvements.
- `plans/Memory_Optimization.md`: scaling, memory layout, and strategies for very large networks.
- `plans/Network_Visualization_Export_Schema.md`: stable export schema for visualization and inspection tooling.
- `plans/ONNX_EXPORT_PLAN.md`: ONNX export/import architecture and rollout phases.
- `plans/Population_Save_Resume_and_Checkpointing.md`: checkpointing, persistence, save/resume workflows.
- `plans/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md`: prebuilt architecture constructors and sequence-oriented builders.
- `plans/Stable_Activation_Ordering_and_Explicit_IO_Roles.md`: deterministic execution ordering and explicit input/output roles.
- `plans/Standalone_Inference_Export.md`: dependency-free exported inference runtime.
- `plans/Turnkey_Multithread_Evaluation_API.md`: parallel evaluation API for Node and browser workers.
- `plans/Worker_Friendly_Network_Serialization_Fastpath.md`: fast serialization path for worker-based evaluation.

Task-to-plan trigger phrases:

- NEAT correctness, innovation IDs, crossover, compatibility distance, speciation: `plans/neat.plans.md`
- roadmap, sequence, dependency order, what comes first: `plans/Roadmap.md`
- architecture builder, layer API, node/group primitives: `plans/Architecture_Primitives_Node_Group_Layer.md`
- construct from parts, graph assembly, deterministic builder: `plans/Construct_From_Parts_Graph_Assembly.md`
- browser bundle, CDN, browser-first usage: `plans/Browser_Build_and_CDN_Distribution.md`
- ONNX, import/export interoperability: `plans/ONNX_EXPORT_PLAN.md`
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

Working rule:
When a task changes code in a way that could conflict with one of these plans, mention the relevant plan in the working notes or final summary and call out any mismatch instead of silently diverging from the roadmap.

Authority rule:
For cross-plan sequencing and active priority, `plans/Roadmap.md` is authoritative. Detailed plans may contain future-state inventory, local sub-phases, or stale status notes; if a detailed plan appears to pull later-phase work earlier, follow `plans/Roadmap.md` unless both files are updated together.
