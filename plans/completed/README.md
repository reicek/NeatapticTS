# Completed Plans Archive

This folder holds terminally closed plan baselines and their matching `.logs.md` audit files.

Use archived plans as reopen points, not as the default starting point for new work.

How to use this archive:

1. Start with [plans/README.md](../README.md) and [plans/Roadmap.md](../Roadmap.md).
2. Open the archived plan that matches the boundary you are reopening.
3. Read the same-boundary `.logs.md` file when you need milestone detail, validation history, or final-state evidence.

Archive selection guide:

- [Agentic_Workflow_Architecture.plans.md](Agentic_Workflow_Architecture.plans.md): closed standalone meta-workflow baseline for numbered phase agents, hidden specialist delegation, skill-first customization, validation scripts and evals, and the MCP runtime-visibility ownership model.
- [Agentic_Flows_and_Gates_Upgrade.plans.md](Agentic_Flows_and_Gates_Upgrade.plans.md): closed standalone meta-workflow amendment for named agent flows, deterministic exit gates, flow-aware MCP resources and prompts, gate exceptions, and universal `00-helping` escalation.
- [workspace-mcp-registration.plans.md](workspace-mcp-registration.plans.md): closed workspace MCP registration baseline for `.vscode/mcp.json`, direct MCP discovery, fixed plan binding, plan-sync alignment, and the shared stdio transport closeout.
- [MCP_Server_Validation_and_Hardening.plans.md](MCP_Server_Validation_and_Hardening.plans.md): closed MCP server hardening baseline for post-restart stdio validation, JSON-RPC method coverage, allow-list extraction, and the stable `mcp-active-binding` handoff.
- [Semantic_Knowledge_MCP_Tools.plans.md](Semantic_Knowledge_MCP_Tools.plans.md): closed Repo Cortex Layer 2 baseline for the `neataptic-cortex-mcp` server, semantic corpus MCP tools, smoke gate, registration, and documentation.
- [Semantic_Knowledge_Browser_Snapshot.plans.md](Semantic_Knowledge_Browser_Snapshot.plans.md): closed Repo Cortex Layer 3 baseline for the generated browser JSON snapshot, docs pipeline integration, IndexedDB-backed shared loader/search utilities, and generated-output contract.
- [Semantic_Knowledge_Embeddings.plans.md](Semantic_Knowledge_Embeddings.plans.md): closed Repo Cortex Layer 5 baseline for local ONNX embeddings, BLOB vector storage, `ts-source` corpus chunks, hybrid BM25 + dense ranking, MRR@5 evaluation, and opt-in dense policy.
- [Semantic_Knowledge_Dense_Prewarm.plans.md](Semantic_Knowledge_Dense_Prewarm.plans.md): closed Repo Cortex Layer 6 baseline for the default-on dense bootstrap contract, idempotent prewarm, readiness probe and gate, warm-state `dense_state` provenance, and graceful cold/model-only degradation.
- [Docs_Quality_Metrics_Contract_and_Parity.plans.md](Docs_Quality_Metrics_Contract_and_Parity.plans.md): closed docs-quality metrics contract baseline for canonical versioned manifests, deterministic run artifacts, strict comparator reason-code guards, MCP and CLI parity, docs-quality CI gate wiring, and migration runbook closure.
- [docs-quality-complexity-cleanup.plans.md](docs-quality-complexity-cleanup.plans.md): closed docs-quality cleanup baseline for driving `npm run docs:quality:metrics` high-complexity findings in `src/` to zero through orchestration-first helper extraction, restoring the owned node mutation guard, and closing `src/architecture/node/node.ts` coverage at `100/100/100/100` while keeping the unrelated NEATchat live-flow failure out of scope.
- [coverage-metrics-and-gap-closure.plans.md](coverage-metrics-and-gap-closure.plans.md): closed docs-quality coverage baseline for additive `summary.coverage` reporting, deterministic `filesBelow100Detail` rows, and final repo-wide `src` coverage closure at `100/100/100/100`.
- [Delegation_Tier_Enforcement.plans.md](Delegation_Tier_Enforcement.plans.md): closed Agentic Workflow Enforcement Prerequisite baseline for the 5-layer delegation tier graph, the enforced 55-agent `tier:` frontmatter map, `validate-agent-graph.mjs`, `tier-enforcement-gate.mjs`, the human-readable audit report, and the `query_tier_graph` MCP tool.
- [neat.plans.md](neat.plans.md): Phase 1 proper-NEAT correctness baseline.
- [Stable_Activation_Ordering_and_Explicit_IO_Roles.md](Stable_Activation_Ordering_and_Explicit_IO_Roles.md): deterministic execution ordering and explicit input/output role baseline.
- [Architecture_Primitives_Node_Group_Layer.md](Architecture_Primitives_Node_Group_Layer.md): closed primitive DX baseline for `Node`, `Group`, and `Layer`.
- [Construct_From_Parts_Graph_Assembly.md](Construct_From_Parts_Graph_Assembly.md): closed Phase 2 whole-graph assembly and construct diagnostics baseline.
- [Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md](Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md): closed Phase 2 preconfigured architecture builders baseline (MLP, RandomSparse, NARX, GRU, LSTM) including shared demo profile contract, Flappy Bird and ASCII Maze integration, cross-demo e2e matrix, and profile-aware trainer/worker recurrent evolution settings.
- [Standalone_Inference_Export.md](Standalone_Inference_Export.md): closed Phase 4 standalone export architecture baseline.
- [Population_Save_Resume_and_Checkpointing.md](Population_Save_Resume_and_Checkpointing.md): closed Phase 4 checkpointing baseline for population-only snapshots, light checkpoints, strict full checkpoints, and the persistence decision ladder.
- [Evolution_Training_Interoperability_Contracts.md](Evolution_Training_Interoperability_Contracts.md): closed hybrid-interoperability baseline for deterministic parameter vectors, isolated fine-tuning, explicit hybrid persistence policy, root-facade exports, and the docs surface now consumed by the NEATchat follow-up.
- [Memory_Optimization.md](Memory_Optimization.md): closed pre-NGE memory foundation baseline through Track 1 / Phase 10.
- [ONNX_EXPORT_PLAN.md](ONNX_EXPORT_PLAN.md): closed ONNX export/import baseline through the current Phase 9 compliance target, including binary-first runtime parity for the approved five-lane subset and the first named external binary import subset.
- [Turnkey_Multithread_Evaluation_API.md](Turnkey_Multithread_Evaluation_API.md): closed Phase 4 ergonomic extraction baseline for capability probes, transport auto-selection, browser worker delivery helpers, reusable pools, ordered batch evaluation, and the NEAT population helper.
- [Worker_Friendly_Network_Serialization_Fastpath.md](Worker_Friendly_Network_Serialization_Fastpath.md): closed Phase 4 transport substrate baseline for the shared inference IR, portable and transferable payloads, persistent channels, and shared-memory workers.
- [neatChat-live-safety-red.plans.md](neatChat-live-safety-red.plans.md): closed NEATchat live exchange safety baseline for reconnecting `checkSafety`, bounded vocabulary-aware fallback selection, and the shipped-browser four-turn quality bar.
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
- [Interactive_Examples_and_Learning_Path.plans.md](Interactive_Examples_and_Learning_Path.plans.md): completed Phase 3 starter-examples lane (helloNetwork, evolveXor, sequenceReset, browser quickstart, learning-path docs, smoke validation).

Trigger phrases:

- agent architecture, custom agents, user-invocable, subagent delegation, model routing, skill evals, MCP runtime visibility reopen: `plans/completed/Agentic_Workflow_Architecture.plans.md`
- agentic flows and gates upgrade, workflow gates, validation gates, flow routing, gate exceptions, universal helper escape, flow-aware MCP reopen: `plans/completed/Agentic_Flows_and_Gates_Upgrade.plans.md`
- workspace MCP, `.vscode/mcp.json`, direct MCP registration, workflow MCP, validation MCP, fixed plan binding reopen: `plans/completed/workspace-mcp-registration.plans.md`
- MCP server validation, post-restart stdio, JSON-RPC framing, gate MCP, workflow MCP, validation MCP, allow-list contracts reopen: `plans/completed/MCP_Server_Validation_and_Hardening.plans.md`
- repo cortex MCP, neataptic-cortex-mcp, search corpus, load chunk, cortex MCP tools, cortex mcp smoke reopen: `plans/completed/Semantic_Knowledge_MCP_Tools.plans.md`
- browser snapshot, semantic snapshot, IndexedDB cache, docs snapshot, semantic-snapshot-loader, build-browser-snapshot reopen: `plans/completed/Semantic_Knowledge_Browser_Snapshot.plans.md`
- semantic embeddings, ONNX embeddings, dense retrieval, hybrid BM25 dense, vector storage, embed-index, eval-embeddings, MRR reopen: `plans/completed/Semantic_Knowledge_Embeddings.plans.md`
- dense prewarm, embedding prewarm, default-on dense, dense readiness, prewarm bootstrap, graceful degradation, warm dense provenance reopen: `plans/completed/Semantic_Knowledge_Dense_Prewarm.plans.md`
- weak-doc metric inconsistency reopen, docs quality metrics reopen, docs:quality:metrics, docs:quality:compare, docs-quality comparator reason codes, docs-quality run artifacts, docs-quality MCP CLI parity: `plans/completed/Docs_Quality_Metrics_Contract_and_Parity.plans.md`
- docs quality complexity cleanup, highComplexity cleanup, complexity cleanup, docs-quality refactor, docs:quality:metrics highComplexity zero reopen: `plans/completed/docs-quality-complexity-cleanup.plans.md`
- coverage metrics detail reopen, coverage gap closure reopen, filesBelow100Detail, docs quality coverage detail, raise testing to 100 reopen: `plans/completed/coverage-metrics-and-gap-closure.plans.md`
- delegation tier, tier graph, validate-agent-graph, tier inventory, tier enforcement, 5-layer delegation reopen: `plans/completed/Delegation_Tier_Enforcement.plans.md`
- proper NEAT, innovation IDs, crossover, compatibility distance, speciation: `plans/completed/neat.plans.md`
- activation order, explicit IO roles, deterministic scheduling: `plans/completed/Stable_Activation_Ordering_and_Explicit_IO_Roles.md`
- architecture primitives, node/group/layer DX, primitive descriptors: `plans/completed/Architecture_Primitives_Node_Group_Layer.md`
- construct from parts, graph assembly, deterministic builder reopen: `plans/completed/Construct_From_Parts_Graph_Assembly.md`
- preconfigured builders, MLP/NARX/GRU/LSTM reopen, shared profile contract, demo profile integration: `plans/completed/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md`
- standalone runtime, exported inference file, Phase 4 export reopen: `plans/completed/Standalone_Inference_Export.md`
- checkpoint, resume, save population, light checkpoint, strict restore reopen: `plans/completed/Population_Save_Resume_and_Checkpointing.md`
- hybrid evolution plus training reopen, parameter vectors, fineTuneVector, Lamarckian persistence policy: `plans/completed/Evolution_Training_Interoperability_Contracts.md`
- memory optimization reopen, release gates, Track 1 memory baseline: `plans/completed/Memory_Optimization.md`
- ONNX export/import reopen, runtime parity, external import subset, supported-subset honesty: `plans/completed/ONNX_EXPORT_PLAN.md`
- turnkey worker evaluation reopen, transport auto-selection, ordered worker batches, reusable NEAT helper: `plans/completed/Turnkey_Multithread_Evaluation_API.md`
- worker transport substrate reopen, payload fastpath, channel workers, shared-memory workers: `plans/completed/Worker_Friendly_Network_Serialization_Fastpath.md`
- ES2023, immutable array methods, named errors, no CommonJS in new work: `plans/completed/ES2023 migration`
- methods docs or split reopen: `plans/completed/methods-docs.plans.md`, `plans/completed/methods-solid-split.plans.md`
- README opening drift or docs-generator reopen: `plans/completed/readme-first-section-pass.plans.md`, `plans/completed/generate-docs-solid-split.plans.md`, `plans/completed/render-docs-html-solid-split.plans.md`
- trace analyzer split reopen: `plans/completed/analyze-trace-solid-split.plans.md`
- Flappy docs or startup preview reopen: `plans/completed/Flappy_Bird_Folder_Documentation_Pass.md`, `plans/completed/flappy-startup-loading-preview.plans.md`
- neatchat live safety, live exchange safety gate, checkSafety, live flow safety, live-flow safety, neatChat-live-safety-red reopen: `plans/completed/neatChat-live-safety-red.plans.md`
- starter examples, learning path, helloNetwork, evolveXor, sequenceReset, browser quickstart, smoke validation reopen: `plans/completed/Interactive_Examples_and_Learning_Path.plans.md`

Archive rule:

When a new workstream reaches terminal `[DONE]`, keep the compressed plan and the matching `.logs.md` file together in this folder before beginning the next workstream.
