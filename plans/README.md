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
- [plans/Agent_Dispatch_MCP_Server.plans.md](Agent_Dispatch_MCP_Server.plans.md): standalone meta-workflow lane — add a repo-owned direct MCP server that resolves a target agent name against `.github/agents/*.agent.md` and returns a validated dispatch packet without spawning subagents. [WIP]
- [plans/completed/docs-quality-complexity-cleanup.plans.md](completed/docs-quality-complexity-cleanup.plans.md): archived docs-quality cleanup lane for reducing `npm run docs:quality:metrics` high-complexity findings in `src/` to zero through orchestration-first helper extraction, with owned node coverage closure and the unrelated NEATchat live-flow failure kept out of scope. [DONE]
- [plans/completed/Docs_Quality_Metrics_Contract_and_Parity.plans.md](completed/Docs_Quality_Metrics_Contract_and_Parity.plans.md): archived permanent fix lane for weak-doc metric inconsistency — canonical versioned metric contract, deterministic runner artifacts, strict comparator guards, MCP and CLI parity, CI gate checks, and migration runbook closure. [DONE]
- [plans/completed/coverage-metrics-and-gap-closure.plans.md](completed/coverage-metrics-and-gap-closure.plans.md): archived docs-quality coverage lane for additive `summary.coverage` reporting in `npm run docs:quality:metrics`, deterministic per-file deficit rows, and final `src` coverage closure at `100/100/100/100`. [DONE]
- [plans/completed/NEAT_Genesis_EvoDevo.md](completed/NEAT_Genesis_EvoDevo.md): NEAT Genesis EvoDevo (NGE) — archived evo-devo core baseline covering computation motifs, memory architecture, neuromodulation, reproduction, lifecycle stages, and collective intelligence. [DONE]
- [plans/completed/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md](completed/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md): NGE core readiness audit — primitive-by-primitive readiness matrix, gap classification by owner boundary, and first implementation tranche selection. Core-first: demos are downstream e2e tests; racing is the first e2e proving ground. [DONE]
- [plans/completed/Repo_Cortex_Advanced_RAG_Architecture.plans.md](completed/Repo_Cortex_Advanced_RAG_Architecture.plans.md): Repo Cortex Layer 7+ — advanced RAG architecture: semantic chunking, query classification, cross-encoder re-ranking, context window assembly, entity/relationship graphs, query expansion, relevance feedback, structured metadata filtering, multi-hop retrieval, ANN indexing, and comprehensive eval suite. [DONE]
- [plans/completed/Cortex_RAG_Premium_Primary_Search.plans.md](completed/Cortex_RAG_Premium_Primary_Search.plans.md): Repo Cortex premium primary search — smart freshness hooks, quality gap closure, LLM-friendly defaults (compact mode, single-call search-and-read, follow-up refs), and tiered high-signal results. Depends on completed advanced RAG plan. [DONE]
- [plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md](NEAT_Genesis_EvoDevo_Racing_Curriculum.md): NGE racing benchmark — Team A/B curriculum refactor, worker-streamed runtime authority, deterministic race packs, and MCP-tracked reference completion. MCP validator path: `plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md`. [PLANNED]
- [plans/completed/Folder_Quality_Gate_and_Racing_Hotfix.plans.md](completed/Folder_Quality_Gate_and_Racing_Hotfix.plans.md): archived standalone lane for the racing curriculum browser hotfix (`NetworkActivateInputSizeMismatchError` Float32Array rejection) and the new folder-quality-metrics static gate script. [DONE]
- [plans/completed/Racing_Pathtracking_Debug_and_Quality_Followup.plans.md](completed/Racing_Pathtracking_Debug_and_Quality_Followup.plans.md): archived standalone follow-up lane for the racing curriculum Catmull-Rom path-tracking fix and the bounded deferred folder-quality cleanup, closing with only the accepted `browser-entry.ts` sibling-test debt. [DONE]
- [plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md](NEAT_Genesis_EvoDevo_AntHive_Demo.md): NGE ant hive ecosystem — multi-agent stigmergy, role differentiation, and collective intelligence. [PLANNED]
- [plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md): NGE predator/prey co-evolution — sensory arms race and reproduction mode dynamics. [PLANNED]
- [plans/completed/Semantic_Knowledge_Foundation.plans.md](completed/Semantic_Knowledge_Foundation.plans.md): archived Repo Cortex Layer 1 baseline — SQLite corpus index, BM25 full-text search, freshness proofs (mtime + size + SHA-256), corpus scanner over generated READMEs / skills / agents / plans / demos, and the `validate-index` gate. [DONE]
- [plans/Orchestration_System_Optimization.plans.md](Orchestration_System_Optimization.plans.md): standalone meta-workflow optimization — Tier 1→Tier 2/3/4 delegation enforcement, skill extraction (`implementation-standards`, `research-methodology`, `routing-optimization-policy`), specialist creation (`implementation-executor`, `research-synthesis-specialist`, `code-quality-auditor`), and flow integration. Phase 4 Step 03 [DONE] (all gates PASS). [WIP]
- [plans/Step_Packet_Goal_Redesign.plans.md](Step_Packet_Goal_Redesign.plans.md): standalone meta-workflow redesign — replace step packet `agent`/`agent_file` fields with `goal` field, add `tdd_sequence` for multi-phase dispatch, update orchestrator routing table, migrate all plan files, and remove backward compatibility. [WIP]
- [plans/completed/Chrome_DevTools_MCP_Integration.plans.md](completed/Chrome_DevTools_MCP_Integration.plans.md): archived standalone meta-workflow lane — integrated Chrome DevTools MCP server with three new Tier 3 specialists (performance-trace, browser-ui, browser-memory), new `execute` skill for mandatory delegation enforcement, new `chrome-devtools-mcp` skill, strict sliced RED→IMPLEMENT→GREEN loop, updated testing agents, two new validation gates, and trace analysis infrastructure. [DONE]
- [plans/completed/Semantic_Knowledge_MCP_Tools.plans.md](completed/Semantic_Knowledge_MCP_Tools.plans.md): archived Repo Cortex Layer 2 baseline — `neataptic-cortex-mcp` MCP server exposing `search_corpus`, `load_chunk`, `load_document`, `freshness_check`, `index_stats`, and `list_families` tools for the full repo/library/demos surface. [DONE]
- [plans/completed/Semantic_Knowledge_Browser_Snapshot.plans.md](completed/Semantic_Knowledge_Browser_Snapshot.plans.md): archived Repo Cortex Layer 3 baseline — generated browser JSON snapshot, docs pipeline integration, IndexedDB cache/loader for all demos, and generated-output contract clarification. [DONE]
- [plans/completed/Delegation_Tier_Enforcement.plans.md](completed/Delegation_Tier_Enforcement.plans.md): archived Agentic Workflow Enforcement Prerequisite baseline for the 5-layer agent delegation tier graph, all 55 `tier:` frontmatter assignments, `validate-agent-graph.mjs` enforcement, `tier-enforcement-gate.mjs`, the human-readable audit report, and the `query_tier_graph` MCP tool. [DONE]
- [plans/completed/Repo_Cortex_MCP_Reliability.plans.md](completed/Repo_Cortex_MCP_Reliability.plans.md): Repo Cortex Layer 4 — Cortex MCP reliability hardening: Repo Cortex Scout agent, Cortex Embeddings Scout agent, `repo-cortex-workflow` skill, unified `cortex-index.gate.mjs`, `plan-session-redirect.mjs`, `validate-tsconfig-docs.mjs`, `validate-index.mjs` fixHint enrichment, `build-index.mjs --json-health`, and `neataptic-workflow-mcp` per-call `plan_path` override. [DONE]
- [plans/completed/Semantic_Knowledge_Embeddings.plans.md](completed/Semantic_Knowledge_Embeddings.plans.md): archived Repo Cortex Layer 5 baseline — ONNX local embedding model, vector storage, hybrid BM25 + dense ranking, eval query set, MRR@5 evaluation, and conservative opt-in default policy. [DONE]
- [plans/completed/Semantic_Knowledge_Dense_Prewarm.plans.md](completed/Semantic_Knowledge_Dense_Prewarm.plans.md): archived Repo Cortex Layer 6 baseline — operationalized full embedding prewarm so `use_dense: true` is the MCP default when readiness is warm, defined the strict bootstrap-first contract, shipped the three readiness states (cold/model-only/warm), the idempotent prewarm script, the readiness probe, the dense-readiness gate, and graceful MCP degradation when embeddings are cold. [DONE]
- [plans/completed/NeatChat_Local_Retrieval_Memory.plans.md](completed/NeatChat_Local_Retrieval_Memory.plans.md): closed NeatChat-internal local DB / retrieval / memory baseline for conversation quality — separate from Repo Cortex; owns its own SQLite (Node) and IndexedDB (browser) contracts; may reuse vector/search patterns but must not depend on the corpus index. [DONE]
- [plans/completed/turso-rag-migration.plans.md](completed/turso-rag-migration.plans.md): Turso RAG migration — migrate the Repo Cortex RAG system from the legacy synchronous local SQLite driver + brute-force vector search to Turso (libSQL cloud database with native vector search, DiskANN, FTS5, embedded replicas, and Platform API). Replaces the database driver, vector index, hybrid ranking strategy, deployment topology, and connection model. 8 phases, ~40 steps. [DONE] (Phase 8: evaluation, optimization, and rollout complete; plan archived to completed/)
- [plans/holistic-agent-skill-optimization.plans.md](holistic-agent-skill-optimization.plans.md): holistic agent & skill optimization — grade, fix, and regrade ALL 65 agents (8 Tier 1, 11 Tier 2, 42 Tier 3, 4 Tier 4) and ALL 58 skills to 100/100/100 across Orchestration, Tools & Skills, and Role Knowledge dimensions; executor never assesses itself. Supersedes the archived tier1-delegation-remediation plan with expanded scope. Addresses P0 (output contract + gate), P1 (Mission mandate), P2 (slice delegate_to field), P3 (Tier 1→specialist table), P4 (routing table refs), P5 (ordered flow dispatch). [WIP]
- [plans/completed/README.md](completed/README.md): archive map for reopen-only baselines and their logs.

Task-to-plan trigger phrases:

- MCP server validation, post-restart stdio, JSON-RPC framing, gate MCP, workflow MCP, validation MCP, allow-list contracts, neataptic-gate-mcp, neataptic-workflow-mcp, neataptic-validation-mcp: `plans/completed/MCP_Server_Validation_and_Hardening.plans.md`
- MCP active binding, perpetual MCP binding, stable plan binding, mcp-active-binding: `plans/mcp-active-binding.plans.md`
- docs quality complexity cleanup, highComplexity cleanup, complexity cleanup, docs-quality refactor, docs:quality:metrics highComplexity zero: `plans/completed/docs-quality-complexity-cleanup.plans.md`
- weak-doc metric inconsistency, docs quality metrics, docs:quality:metrics, docs:quality:compare, comparator mismatch reason code, docs quality run-id artifact contract, MCP CLI parity for docs quality: `plans/completed/Docs_Quality_Metrics_Contract_and_Parity.plans.md`
- add coverage metrics to docs:quality:metrics, docs quality coverage metrics, coverage gap closure, raise testing to 100%, filesBelow100Detail reopen: `plans/completed/coverage-metrics-and-gap-closure.plans.md`
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
- cortex reliability, cortex MCP reliability, cortex lifecycle, cortex lifecycle gate, cortex-index gate, cortex scout, repo cortex scout, cortex embeddings scout, embeddings scout, plan session redirect, validate-tsconfig-docs, build-index json-health, validate-index fixhint, neataptic-workflow-mcp plan_path override, repo-cortex-workflow skill: `plans/completed/Repo_Cortex_MCP_Reliability.plans.md`
- delegation tier, tier graph, validate-agent-graph, tier inventory, tier enforcement, 5-layer delegation: `plans/completed/Delegation_Tier_Enforcement.plans.md`
- semantic embeddings, ONNX embeddings, dense retrieval, hybrid BM25 dense, vector storage, embed-index, eval-embeddings, MRR: `plans/completed/Semantic_Knowledge_Embeddings.plans.md`
- dense prewarm, embedding prewarm, use_dense default, default-on dense, dense readiness, prewarm bootstrap, cold state degradation, dense-readiness gate, prewarm-dense, warm embeddings, dense default contract: `plans/completed/Semantic_Knowledge_Dense_Prewarm.plans.md`
- neatchat memory, neatchat retrieval, local memory DB, conversation memory, memory services, memory IDB, neatChat local retrieval: `plans/completed/NeatChat_Local_Retrieval_Memory.plans.md`
- NGE core readiness, NGE primitive audit, NGE readiness matrix, core-first NGE, NGE gap classification, NGE first tranche, nge-adult readiness, team-level fitness gap, generation barriers gap, deterministic evaluation packs gap, NGE public API exposure: `plans/completed/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md` (`plans\completed\NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`)
- advanced RAG, semantic chunking, query classification, cross-encoder re-ranking, context assembly, entity graph, multi-hop retrieval, query expansion, relevance feedback, metadata filtering, ANN index, RAG eval, Repo Cortex Layer 7: `plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md` (`plans\Repo_Cortex_Advanced_RAG_Architecture.plans.md`)
- premium primary search, smart freshness hooks, single-call search, search-and-read, compact search results, code-source boost, exact symbol lookup, include_code_only, explain ranking, follow-up refs, auto fallback, native search fallback, freshness transparency, search budget: `plans/completed/Cortex_RAG_Premium_Primary_Search.plans.md` (`plans\completed\Cortex_RAG_Premium_Primary_Search.plans.md`)
- evo-devo, NGE, morphology, research-heavy extensions, computation motifs, DNA program, neuromodulation, reproduction: `plans/completed/NEAT_Genesis_EvoDevo.md`
- racing benchmark, optimal line, behavioral drives, overtaking, self-play racecraft, NGE racing, Team A/B curriculum, worker-streamed racing: `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md` (`plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md`)
- racing curriculum browser bug, NetworkActivateInputSizeMismatchError, Float32Array mismatch, activate typed array, folder quality metrics, folder quality gate, quality:folder, folder-quality-metrics script, post-edit quality check: `plans/completed/Folder_Quality_Gate_and_Racing_Hotfix.plans.md`
- racing path tracking, rounded racing lane, Catmull-Rom steering, spline path tracking, lane drift, browser-entry test debt: `plans/completed/Racing_Pathtracking_Debug_and_Quality_Followup.plans.md`
- ant hive, stigmergy, pheromone, role differentiation, colony, collective intelligence, multi-agent: `plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md`
- predator prey, co-evolution, arms race, camouflage, evasion, pursuit, two populations: `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md`
- orchestration optimization, mini-agent, Tier 1 delegation, skill extraction, specialist creation, agent frontmatter, routing table, agent-graph gate, tier-enforcement: `plans/completed/Orchestration_System_Optimization.plans.md`
- step packet goal, goal-based dispatch, tdd_sequence, agent to goal migration, orchestrator routing table, step-packet gate update: `plans/completed/Step_Packet_Goal_Redesign.plans.md`
- Chrome DevTools MCP, browser testing, performance trace specialist, UI manipulation specialist, browser memory specialist, execute skill, sliced implementation loop, RED IMPLEMENT GREEN loop, chrome-devtools-mcp skill, trace compression, trace summarization, browser validation: `plans/completed/Chrome_DevTools_MCP_Integration.plans.md`
- Turso RAG migration, libSQL, legacy sync SQLite driver replacement, Turso vector search, DiskANN, F8_BLOB, vector8, embedded replicas, RRF hybrid search, server-side ranking, parallel search, multi-hop search, database branching, point-in-time recovery, Turso Platform API, async DB driver migration: `plans/completed/turso-rag-migration.plans.md`
- holistic agent optimization, agent skill optimization, agent grading, skill grading, 100/100/100, Orchestration Tools Role Knowledge, delegation mandate, SUB_ORCHESTRATORS_USED NONE, monolithic agent, output contract delegation, delegate_to slice field, Tier 1→specialist table, flow dispatch sequences, delegate-skill-coverage gate enhancement: `plans/holistic-agent-skill-optimization.plans.md`
- agent dispatch MCP, neataptic-dispatch-mcp, dispatch packet, resolve agent target, build dispatch packet, delegation lookup, agent target resolution: `plans/completed/Agent_Dispatch_MCP_Server.plans.md`
- archived reopen triggers: see `plans/completed/README.md`

Working rule:

When a task changes code in a way that could conflict with one of these plans, mention the relevant plan in the working notes or final summary and call out any mismatch instead of silently diverging from the roadmap.

Authority rule:

For cross-plan sequencing and active priority, `plans/Roadmap.md` is authoritative. Detailed plans may contain future-state inventory, local sub-phases, or stale status notes; if a detailed plan appears to pull later-phase work earlier, follow `plans/Roadmap.md` unless both files are updated together.
