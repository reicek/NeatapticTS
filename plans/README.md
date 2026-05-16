# Plans Index

This file is a lightweight map for the active `plans/` surface.

Purpose:

- help agents and contributors choose the right active plan quickly,
- keep architectural work aligned with roadmap intent,
- keep the root `plans/` folder focused now that terminally closed trackers live in `plans/completed/`.

Status convention:

- Every roadmap-tracked plan file should expose a top-level status line immediately under the title.
- Use the exact format `**Status:** [DONE]`, `**Status:** [WIP]`, or `**Status:** [PLANNED]`.
- Keep that top-level status aligned with [plans/Roadmap.md](Roadmap.md), which is the authoritative source for cross-plan sequencing and active priority.
- Terminally closed plans should be compressed and archived under [plans/completed/](completed/) alongside their same-boundary `.logs.md` files.

Folder split:

- `plans/` holds the active roadmap surface: `[WIP]` and `[PLANNED]` trackers plus [plans/Roadmap.md](Roadmap.md) and this index.
- `plans/completed/` holds reopen-only baselines and their matching `.logs.md` audit records.
- When a task reopens finished work, start with [plans/completed/README.md](completed/README.md).

Recommended reading order:

1. Read `plans/README.md` for the active map.
2. Read `plans/Roadmap.md` when the task depends on sequencing, current phase status, or cross-plan priority.
3. Read only the single most relevant active detailed plan.
4. If the task reopens a closed lane, read [plans/completed/README.md](completed/README.md) and then the archived plan.
5. Read [plans/completed/neat.plans.md](completed/neat.plans.md) when the task touches core NEAT architecture or evolutionary correctness.

Active selection guide:

- [plans/Roadmap.md](Roadmap.md): dependency-aware execution order across all initiatives.
- [plans/completed/Browser_Build_and_CDN_Distribution.md](completed/Browser_Build_and_CDN_Distribution.md): browser packaging, CDN usage, and distribution ergonomics. [DONE]
- [plans/completed/Memory_Optimization.md](completed/Memory_Optimization.md): archived pre-NGE memory foundation baseline through Track 1 / Phase 10. [DONE]
- [plans/completed/Network_Visualization_Export_Schema.plans.md](completed/Network_Visualization_Export_Schema.plans.md): archived visualization export schema baseline for inspection tooling and DOT/schema output. [DONE]
- [plans/Evolution_Training_Interoperability_Contracts.md](Evolution_Training_Interoperability_Contracts.md): planned hybrid-interoperability lane for deterministic parameter vectors, training isolation, and explicit persistence policy.
- [plans/ONNX_EXPORT_PLAN.md](ONNX_EXPORT_PLAN.md): active ONNX export/import architecture and rollout lane, with Phases 3-6 closed for the current conservative subset, Phase 7A closed, the first 7B storage-FP16 lane landed, the 7C calibration contract landed, and the current 7D same-family dense qlinear slice landed with explicit bias bridges plus preserved unary activations including focused Softplus, Softsign, Selu, opset-18 Mish, and opset-20 Gelu parity proofs, with Mish below opset 18 now also proven to keep static-8bit dense lowering while the activation itself stays on the honest implicit-identity fallback path. The remaining plan is now explicitly compliance-oriented: broader dense, spatial, and dynamic quantized lowering remain open inside Phase 7, and plan closure now also requires later binary `ModelProto` serialization, checker-backed validation, real runtime parity, and a named external import subset before the ONNX lane can move to `[DONE]`. [WIP]
- [plans/NEATchat.plans.md](NEATchat.plans.md): planned NEATchat follow-up lane for turning the closed toy demo into a stronger conversational system, gated on Phase 4 substrate readiness plus recurrent ONNX hardening before implementation starts.
- [plans/completed/Population_Save_Resume_and_Checkpointing.md](completed/Population_Save_Resume_and_Checkpointing.md): archived checkpointing, persistence, and save/resume baseline covering population-only snapshots, light checkpoints, strict full checkpoints, and the standalone persistence walkthrough. [DONE]
- [plans/completed/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md](completed/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md): prebuilt architecture constructors and sequence-oriented builders. [DONE]
- [plans/completed/Turnkey_Multithread_Evaluation_API.md](completed/Turnkey_Multithread_Evaluation_API.md): archived low-friction parallel evaluation API extraction baseline for Node and browser workers. [DONE]
- [plans/completed/Worker_Friendly_Network_Serialization_Fastpath.md](completed/Worker_Friendly_Network_Serialization_Fastpath.md): archived worker transport substrate baseline covering `PortableInferencePayload`, `TransferableInferencePayload`, `InferenceChannel`, and `SharedInferenceWorker`. [DONE]
- [plans/NEAT_Genesis_EvoDevo.md](NEAT_Genesis_EvoDevo.md): NEAT Genesis EvoDevo (NGE) — evo-devo algorithm with computation motifs, memory architecture, neuromodulation, and reproduction system.
- [plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md](NEAT_Genesis_EvoDevo_Racing_Curriculum.md): NGE racing benchmark — single-agent sensory specialization and behavioral drives.
- [plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md](NEAT_Genesis_EvoDevo_AntHive_Demo.md): NGE ant hive ecosystem — multi-agent stigmergy, role differentiation, and collective intelligence.
- [plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md): NGE predator/prey co-evolution — sensory arms race and reproduction mode dynamics.
- [plans/completed/README.md](completed/README.md): archive map for reopen-only baselines and their logs.

Task-to-plan trigger phrases:

- roadmap, sequence, dependency order, what comes first: `plans/Roadmap.md`
- construct from parts, graph assembly, deterministic builder reopen: `plans/completed/Construct_From_Parts_Graph_Assembly.md`
- browser bundle, CDN, browser-first usage: `plans/completed/Browser_Build_and_CDN_Distribution.md`
- worker serialization, transfer cost, fastpath reopen: `plans/completed/Worker_Friendly_Network_Serialization_Fastpath.md`
- workers, threads, parallel evaluation reopen: `plans/completed/Turnkey_Multithread_Evaluation_API.md`
- checkpoint, resume, save population, strict restore, checkpoint state inventory: `plans/completed/Population_Save_Resume_and_Checkpointing.md`
- hybrid evolution plus training, optimizer handoff: `plans/Evolution_Training_Interoperability_Contracts.md`
- preconfigured models, MLP, LSTM, GRU, NARX builders: `plans/completed/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md`
- ONNX, import/export interoperability, recurrent import hardening, supported subset, seed import honesty: `plans/ONNX_EXPORT_PLAN.md`
- memory pressure, large networks, compact storage: `plans/completed/Memory_Optimization.md`
- visualization, schema, inspect network shape: `plans/completed/Network_Visualization_Export_Schema.plans.md`
- examples, tutorials, learning path, onboarding: `plans/completed/Interactive_Examples_and_Learning_Path.plans.md`
- neatchat, chatbot, conversational system, pretrained seed import, retrieval memory, online language learning, tiny conversation bot: `plans/NEATchat.plans.md`
- evo-devo, NGE, morphology, research-heavy extensions, computation motifs, DNA program, neuromodulation, reproduction: `plans/NEAT_Genesis_EvoDevo.md`
- racing benchmark, optimal line, behavioral drives, overtaking, self-play racecraft, NGE racing: `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md`
- ant hive, stigmergy, pheromone, role differentiation, colony, collective intelligence, multi-agent: `plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md`
- predator prey, co-evolution, arms race, camouflage, evasion, pursuit, two populations: `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md`
- archived reopen triggers: see `plans/completed/README.md`

Working rule:

When a task changes code in a way that could conflict with one of these plans, mention the relevant plan in the working notes or final summary and call out any mismatch instead of silently diverging from the roadmap.

Authority rule:

For cross-plan sequencing and active priority, `plans/Roadmap.md` is authoritative. Detailed plans may contain future-state inventory, local sub-phases, or stale status notes; if a detailed plan appears to pull later-phase work earlier, follow `plans/Roadmap.md` unless both files are updated together.
