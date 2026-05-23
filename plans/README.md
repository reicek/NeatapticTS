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
- [plans/completed/workspace-mcp-registration.plans.md](completed/workspace-mcp-registration.plans.md): archived workspace MCP registration baseline for `.vscode/mcp.json`, direct MCP discovery, fixed plan binding, plan-sync alignment, and the shared stdio transport closeout. [DONE]
- [plans/completed/Agentic_Workflow_Architecture.plans.md](completed/Agentic_Workflow_Architecture.plans.md): archived standalone meta-workflow baseline for the numbered agent architecture, hidden specialist delegation, skill-first customization, model routing, validation scripts, skill evals, and MCP runtime-visibility ownership boundaries. [DONE]
- [plans/completed/Agentic_Flows_and_Gates_Upgrade.plans.md](completed/Agentic_Flows_and_Gates_Upgrade.plans.md): archived standalone meta-workflow amendment for named agent flows, deterministic exit gates, flow-aware MCP resources/prompts/tools, gate exceptions, and universal `00-helping` escalation. [DONE]
- [plans/completed/Browser_Build_and_CDN_Distribution.md](completed/Browser_Build_and_CDN_Distribution.md): browser packaging, CDN usage, and distribution ergonomics. [DONE]
- [plans/completed/Memory_Optimization.md](completed/Memory_Optimization.md): archived pre-NGE memory foundation baseline through Track 1 / Phase 10. [DONE]
- [plans/completed/Network_Visualization_Export_Schema.plans.md](completed/Network_Visualization_Export_Schema.plans.md): archived visualization export schema baseline for inspection tooling and DOT/schema output. [DONE]
- [plans/completed/ONNX_EXPORT_PLAN.md](completed/ONNX_EXPORT_PLAN.md): archived ONNX export/import baseline through the current Phase 9 compliance target, including the closed recurrent, spatial, advanced-graph, optimization, precision, binary, runtime-parity, and first external-import stop lines for the declared lower-opset same-family subset. [DONE]
- [plans/completed/Evolution_Training_Interoperability_Contracts.md](completed/Evolution_Training_Interoperability_Contracts.md): archived hybrid-interoperability baseline for deterministic parameter vectors, training isolation, explicit persistence policy, root-facade re-exports, and the docs surface that now feeds the NEATchat follow-up. [DONE]
- [plans/completed/NEATchat_Followup.plans.md](completed/NEATchat_Followup.plans.md): archived [DONE] NEATchat follow-up lane — persistent session identity, stronger pretrained seed, episodic memory and retrieval, background adaptation, hybrid routing, attribution-aware regression harness, and safety gate. All six workstreams (W1–W6) closed 2026-05-22. Unrelated broad-run heap OOM remains outside NEATchat ownership.
- [plans/completed/neatChat-live-safety-red.plans.md](completed/neatChat-live-safety-red.plans.md): archived [DONE] NEATchat live exchange safety baseline for reconnecting `checkSafety`, bounded local fallback, and the shipped-browser four-turn quality bar. Reopen only for stricter extended-turn freshness. [DONE]
- [plans/completed/Population_Save_Resume_and_Checkpointing.md](completed/Population_Save_Resume_and_Checkpointing.md): archived checkpointing, persistence, and save/resume baseline covering population-only snapshots, light checkpoints, strict full checkpoints, and the standalone persistence walkthrough. [DONE]
- [plans/completed/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md](completed/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md): prebuilt architecture constructors and sequence-oriented builders. [DONE]
- [plans/completed/Turnkey_Multithread_Evaluation_API.md](completed/Turnkey_Multithread_Evaluation_API.md): archived low-friction parallel evaluation API extraction baseline for Node and browser workers. [DONE]
- [plans/completed/Worker_Friendly_Network_Serialization_Fastpath.md](completed/Worker_Friendly_Network_Serialization_Fastpath.md): archived worker transport substrate baseline covering `PortableInferencePayload`, `TransferableInferencePayload`, `InferenceChannel`, and `SharedInferenceWorker`. [DONE]
- [plans/completed/MCP_Server_Validation_and_Hardening.plans.md](completed/MCP_Server_Validation_and_Hardening.plans.md): archived MCP server validation and hardening baseline for post-restart stdio behavior, JSON-RPC framing, tool/resource/prompt lists, representative tool calls, gate MCP behavior, and allow-list contracts across all three configured servers. [DONE]
- [plans/mcp-active-binding.plans.md](mcp-active-binding.plans.md): permanent perpetual MCP server binding — provides a stable `[WIP]` phase for `neataptic-workflow-mcp` and `neataptic-validation-mcp` so they can start cleanly regardless of which workstream plans are archived. Never archive this file. [WIP]
- [plans/NEAT_Genesis_EvoDevo.md](NEAT_Genesis_EvoDevo.md): NEAT Genesis EvoDevo (NGE) — evo-devo algorithm with computation motifs, memory architecture, neuromodulation, and reproduction system.
- [plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md](NEAT_Genesis_EvoDevo_Racing_Curriculum.md): NGE racing benchmark — single-agent sensory specialization and behavioral drives.
- [plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md](NEAT_Genesis_EvoDevo_AntHive_Demo.md): NGE ant hive ecosystem — multi-agent stigmergy, role differentiation, and collective intelligence.
- [plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md): NGE predator/prey co-evolution — sensory arms race and reproduction mode dynamics.
- [plans/completed/Semantic_Knowledge_Foundation.plans.md](completed/Semantic_Knowledge_Foundation.plans.md): archived Repo Cortex Layer 1 baseline — SQLite corpus index, BM25 full-text search, freshness proofs (mtime + size + SHA-256), corpus scanner over generated READMEs / skills / agents / plans / demos, and the `validate-index` gate. [DONE]
- [plans/completed/Semantic_Knowledge_MCP_Tools.plans.md](completed/Semantic_Knowledge_MCP_Tools.plans.md): archived Repo Cortex Layer 2 baseline — `neataptic-cortex-mcp` MCP server exposing `search_corpus`, `load_chunk`, `load_document`, `freshness_check`, `index_stats`, and `list_families` tools for the full repo/library/demos surface. [DONE]
- [plans/completed/Semantic_Knowledge_Browser_Snapshot.plans.md](completed/Semantic_Knowledge_Browser_Snapshot.plans.md): archived Repo Cortex Layer 3 baseline — generated browser JSON snapshot, docs pipeline integration, IndexedDB cache/loader for all demos, and generated-output contract clarification. [DONE]
- [plans/Delegation_Tier_Enforcement.plans.md](Delegation_Tier_Enforcement.plans.md): Agentic Workflow Enforcement Prerequisite — 5-layer agent delegation tier graph, frontmatter tier inventory, `validate-agent-graph.mjs` extension, MCP gate tool, and audit/escalation policy. Must complete before Repo Cortex Layer 4. [PLANNED]
- [plans/Repo_Cortex_MCP_Reliability.plans.md](Repo_Cortex_MCP_Reliability.plans.md): Repo Cortex Layer 4 — Cortex MCP reliability hardening: Repo Cortex Scout agent, Cortex Embeddings Scout agent, `repo-cortex-workflow` skill, unified `cortex-index.gate.mjs`, `plan-session-redirect.mjs`, `validate-tsconfig-docs.mjs`, `validate-index.mjs` fixHint enrichment, `build-index.mjs --json-health`, and `neataptic-workflow-mcp` per-call `plan_path` override. [PLANNED]
- [plans/Semantic_Knowledge_Embeddings.plans.md](Semantic_Knowledge_Embeddings.plans.md): Repo Cortex Layer 5 — ONNX local embedding model, vector storage, hybrid BM25 + dense ranking, eval query set, MRR@5 evaluation, and opt-in policy. Final advanced Repo Cortex step. [PLANNED]
- [plans/NeatChat_Local_Retrieval_Memory.plans.md](NeatChat_Local_Retrieval_Memory.plans.md): NeatChat-internal local DB / retrieval / memory for conversation quality — separate from Repo Cortex; owns its own SQLite (Node) and IndexedDB (browser) contracts; may reuse vector/search patterns but must not depend on the corpus index. [PLANNED]
- [plans/completed/README.md](completed/README.md): archive map for reopen-only baselines and their logs.

Task-to-plan trigger phrases:

- MCP server validation, post-restart stdio, JSON-RPC framing, gate MCP, workflow MCP, validation MCP, allow-list contracts, neataptic-gate-mcp, neataptic-workflow-mcp, neataptic-validation-mcp: `plans/completed/MCP_Server_Validation_and_Hardening.plans.md`
- MCP active binding, perpetual MCP binding, stable plan binding, mcp-active-binding: `plans/mcp-active-binding.plans.md`
- roadmap, sequence, dependency order, what comes first: `plans/Roadmap.md`
- workspace MCP, `.vscode/mcp.json`, direct MCP registration, workflow MCP, validation MCP, fixed plan binding: `plans/completed/workspace-mcp-registration.plans.md`
- agent architecture, custom agents, user-invocable, subagent delegation, model routing, skill evals, agent skills, workflow agents, numbered SDLC orchestrators: `plans/completed/Agentic_Workflow_Architecture.plans.md`
- agentic flows and gates upgrade, workflow gates, validation gates, flow routing, gate exceptions, universal helper escape, flow-aware MCP: `plans/completed/Agentic_Flows_and_Gates_Upgrade.plans.md`
- construct from parts, graph assembly, deterministic builder reopen: `plans/completed/Construct_From_Parts_Graph_Assembly.md`
- browser bundle, CDN, browser-first usage: `plans/completed/Browser_Build_and_CDN_Distribution.md`
- worker serialization, transfer cost, fastpath reopen: `plans/completed/Worker_Friendly_Network_Serialization_Fastpath.md`
- workers, threads, parallel evaluation reopen: `plans/completed/Turnkey_Multithread_Evaluation_API.md`
- checkpoint, resume, save population, strict restore, checkpoint state inventory: `plans/completed/Population_Save_Resume_and_Checkpointing.md`
- hybrid evolution plus training, optimizer handoff, parameter vector reopen, fineTuneVector, Lamarckian policy reopen: `plans/completed/Evolution_Training_Interoperability_Contracts.md`
- preconfigured models, MLP, LSTM, GRU, NARX builders: `plans/completed/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md`
- ONNX, import/export interoperability, recurrent import hardening, supported subset, runtime parity, external import reopen, seed import honesty: `plans/completed/ONNX_EXPORT_PLAN.md`
- memory pressure, large networks, compact storage: `plans/completed/Memory_Optimization.md`
- visualization, schema, inspect network shape: `plans/completed/Network_Visualization_Export_Schema.plans.md`
- examples, tutorials, learning path, onboarding: `plans/completed/Interactive_Examples_and_Learning_Path.plans.md`
- neatchat, chatbot, conversational system, pretrained seed import, retrieval memory, online language learning, tiny conversation bot: `plans/completed/NEATchat_Followup.plans.md`
- neatchat live safety, live exchange safety gate, checkSafety, live flow safety, live-flow safety, neatChat-live-safety-red: `plans/completed/neatChat-live-safety-red.plans.md`
- repo cortex, semantic index, corpus scanner, BM25, SQLite index, freshness proof, build-index, validate-index, semantic knowledge foundation: `plans/completed/Semantic_Knowledge_Foundation.plans.md`
- repo cortex MCP, neataptic-cortex-mcp, search corpus, load chunk, cortex MCP tools, cortex mcp smoke: `plans/completed/Semantic_Knowledge_MCP_Tools.plans.md`
- browser snapshot, semantic snapshot, IndexedDB cache, docs snapshot, semantic-snapshot-loader, build-browser-snapshot: `plans/completed/Semantic_Knowledge_Browser_Snapshot.plans.md`
- cortex reliability, cortex MCP reliability, cortex lifecycle, cortex lifecycle gate, cortex-index gate, cortex scout, repo cortex scout, cortex embeddings scout, embeddings scout, plan session redirect, validate-tsconfig-docs, build-index json-health, validate-index fixhint, neataptic-workflow-mcp plan_path override, repo-cortex-workflow skill: `plans/Repo_Cortex_MCP_Reliability.plans.md`
- delegation tier, tier graph, validate-agent-graph, tier inventory, tier enforcement, 5-layer delegation: `plans/Delegation_Tier_Enforcement.plans.md`
- semantic embeddings, ONNX embeddings, dense retrieval, hybrid BM25 dense, vector storage, embed-index, eval-embeddings, MRR: `plans/Semantic_Knowledge_Embeddings.plans.md`
- neatchat memory, neatchat retrieval, local memory DB, conversation memory, memory services, memory IDB, neatChat local retrieval: `plans/NeatChat_Local_Retrieval_Memory.plans.md`
- evo-devo, NGE, morphology, research-heavy extensions, computation motifs, DNA program, neuromodulation, reproduction: `plans/NEAT_Genesis_EvoDevo.md`
- racing benchmark, optimal line, behavioral drives, overtaking, self-play racecraft, NGE racing: `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md`
- ant hive, stigmergy, pheromone, role differentiation, colony, collective intelligence, multi-agent: `plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md`
- predator prey, co-evolution, arms race, camouflage, evasion, pursuit, two populations: `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md`
- archived reopen triggers: see `plans/completed/README.md`

Working rule:

When a task changes code in a way that could conflict with one of these plans, mention the relevant plan in the working notes or final summary and call out any mismatch instead of silently diverging from the roadmap.

Authority rule:

For cross-plan sequencing and active priority, `plans/Roadmap.md` is authoritative. Detailed plans may contain future-state inventory, local sub-phases, or stale status notes; if a detailed plan appears to pull later-phase work earlier, follow `plans/Roadmap.md` unless both files are updated together.
