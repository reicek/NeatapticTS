# NEATchat Follow-up — Compressed Archive

**Status:** [DONE]
**Closed:**

## Scope

Post-toy conversational-systems lane that built on the closed Phase 3 NEATchat
toy demo (`plans/completed/NEATchat.plans.md`) and moved NEATchat closer to a
modern, transparent, persistently owned conversational system without claiming
transformer parity. Delivered over six workstreams .

## Dependency gates satisfied at open

1. Checkpointing — `Neat.exportState()` / `importState()` / light variants
   stable and covered (`plans/completed/Population_Save_Resume_and_Checkpointing.md`).
2. Worker payload + multithread eval — `PortableInferencePayload`,
   `evaluateInWorkers`, `createNeatParallelPopulationEvaluator`, and
   `resolveAutoInferenceTransport` all exported
   (`plans/completed/Worker_Friendly_Network_Serialization_Fastpath.md`,
   `plans/completed/Turnkey_Multithread_Evaluation_API.md`).
3. Evolution-training interop — `toParameterVector`, `fromParameterVector`,
   `fineTuneVector`, and `HybridEvaluationPolicy` all exported
   (`plans/completed/Evolution_Training_Interoperability_Contracts.md`).
4. ONNX recurrent import — conditionally open; the non-ONNX parameter-vector
   conversion path was formally adopted as the chosen approach
   (`plans/completed/ONNX_EXPORT_PLAN.md`).

## Workstream closure summary

### [DONE] W1 — Durable infrastructure substrate

Session snapshot v2, checkpoint-aware import/export, worker-friendly replay
hooks, and parameter-vector-safe resume behavior. Touched
`examples/neatChat/core/` snapshot and session services, plus one narrow
replay guard in `src/architecture/network/serialize/network.serialize.utils.ts`.
Coverage: `100/100/100/100` on all touched boundaries. Both TypeScript checks,
owner-local suite, `npm run docs`, and plan sync PASS.

### [DONE] W2 — Stronger pretrained seed import and distillation

Honest external-seed contract (single-layer GRU/LSTM, vocab 300-3000, hidden
8-128, sigmoid+tanh, one-hot); promoted default seed improved
`heldOutNextTokenAccuracy` from `9.77` to `12.29` with `repetitionRate = 0`
and `responseLengthStability = 1`. Stayed inside `examples/neatChat/`.
Coverage, TypeScript, regression pack, docs, and plan sync PASS.

### [DONE] W3 — Multi-tier memory and retrieval

Episodic memory bank, token-overlap retrieval, snapshot-v2 persistence. Memory
survives checkpoint round-trips; retrieval leaves network weights byte-identical.
Defect fix: snapshot normalization strips raw `memoryBank` payloads before
spread. Stayed inside `examples/neatChat/core/`. Coverage, TypeScript, suite,
docs, plan sync PASS.

### [DONE] W4 — Background adaptation and candidate search

`scheduleNeatChatAdaptation` defers fine-tuning via `queueMicrotask`;
candidates persist in `pendingCandidates`; `candidateLog` survives snapshot
v2. Design: frozen base vector, explicit promote/reject only, no auto-promote,
transient ready candidates. Stayed inside `examples/neatChat/`. Coverage,
TypeScript, suite, docs, plan sync PASS.

### [DONE] W5 — Hybrid routing and specialist submodels

Base, personalized, and retrieval-grounded paths produce candidates before
selection; deterministic base-path tie-breaking. `routingLog` records selected
path, candidate count, scores, and retrieval usage — observability only, no
weight promotion. New files: `neatChat.routing.types.ts`,
`neatChat.routing.services.ts`, `neatChat.routing.services.test.ts`. Updated:
`neatChat.types.ts`, `neatChat.snapshot.v2.services.ts`,
`neatChat.session.services.ts`. Coverage, TypeScript, suite, docs, plan sync
PASS.

### [DONE] W6 — Evaluation, safety, and publishable product shape

Steps W6-01 through W6-08 delivered:

- `neatChat.evaluation.types.ts`: five evaluation/scoring/attribution types.
- `neatChat.evaluation.services.ts`: `runNeatChatRegressionSuite`,
  `attributeToFailureBucket`, and five score helpers; 39 tests,
  `100/100/100/100` coverage.
- `neatChat.safety.types.ts` + `neatChat.safety.services.ts`: `checkSafety`,
  `isUnknownToken`, `isRepetitionCollapse`, `isDegenerateResponse`; 32 tests,
  `100/100/100/100` coverage.
- `examples/neatChat/index.html`: feature-lane labels (live stable /
  experimental routing / experimental adaptation) in neon-retro CSS.
- `examples/neatChat/index.ts`: re-exports for evaluation, safety, and routing
  surfaces; `createNeatChatExampleContract` JSDoc updated with limitations and
  baseline scores.
- `examples/neatChat/README.md`: hybrid routing, evaluation harness, safety
  gate, and scale-limitations sections added.
- W6 full-suite (10 suites, 306 tests) and targeted coverage gates all PASS.

## Residual notes

- Broad-run default-heap OOM and unrelated non-NEATchat failures
  (`telemetry.facade.buffer.test.ts`, `evolveXor.test.ts`) remain outside
  NEATchat ownership throughout all workstreams.
- Routing selection remains observability-only and does not promote weights.
- Safety gate is additive, not blocking; it returns structured
  `SafetyCheckResult` rather than throwing.

## Validation at closure

- `npx tsc --noEmit -p tsconfig.json` PASS
- `npx tsc --noEmit -p tsconfig.test.json` PASS
- Focused NEATchat Jest slice: `10 suites, 306 tests passed`
- `npm run docs` PASS (176 Mermaid diagrams valid)
- Plan sync PASS `0 errors, 0 warnings` (validated against archived path)

## Archive notes

- Phase 3 toy-demo baseline: `plans/completed/NEATchat.plans.md` (preserved)
- Phase 3 audit log: `plans/completed/NEATchat.logs.md` (preserved)
- This follow-up plan: `plans/completed/NEATchat_Followup.plans.md`
- This follow-up log: `plans/completed/NEATchat_Followup.logs.md`
