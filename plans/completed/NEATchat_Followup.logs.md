# NEATchat Follow-up — Audit Log

**Status:** [DONE]
**Closed:**

## Pass history

| Pass | Workstream / Step                               | Outcome                                                                                                                                                                                                                                     |
| ---- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | W1 — Durable infrastructure substrate           | Session snapshot v2, checkpoint-aware import/export, worker-friendly replay hooks, parameter-vector-safe resume. Coverage `100/100/100/100` on all touched files.                                                                           |
| 2    | W2 — Stronger pretrained seed import            | Honest external-seed contract documented. Default seed promoted: `heldOutNextTokenAccuracy` 9.77 → 12.29, `repetitionRate = 0`, `responseLengthStability = 1`.                                                                              |
| 3    | W3 — Multi-tier memory and retrieval            | Episodic memory bank, token-overlap retrieval, snapshot-v2 persistence. Snapshot normalization defect fixed.                                                                                                                                |
| 4    | W4 — Background adaptation and candidate search | `scheduleNeatChatAdaptation` via `queueMicrotask`. Explicit promote/reject lifecycle. No auto-promotion or weight mutation outside deliberate promote path.                                                                                 |
| 5    | W5 — Hybrid routing and specialist submodels    | Three-path routing (base, personalized, retrieval-grounded) with deterministic tie-breaking. `routingLog` observability contract; no weight promotion.                                                                                      |
| 6    | W6-01                                           | W6 step sequence authored: W6-02 through W6-08 planned. Plan sync PASS.                                                                                                                                                                     |
| 7    | W6-02                                           | `neatChat.evaluation.types.ts` created with 5 type contracts. Types-only; no runtime code. Both TypeScript checks PASS.                                                                                                                     |
| 8    | W6-03                                           | `neatChat.evaluation.services.test.ts` created: 32 red tests across 9 describe blocks. Plan sync PASS.                                                                                                                                      |
| 9    | W6-04                                           | `neatChat.evaluation.services.ts` implemented: `runNeatChatRegressionSuite`, `attributeToFailureBucket`, 5 score helpers. Expanded to 39 tests. Coverage `100/100/100/100`.                                                                 |
| 10   | W6-05                                           | Green validation: 39 tests PASS, coverage `100/100/100/100`, TypeScript PASS, `npm run docs` PASS, plan sync PASS.                                                                                                                          |
| 11   | W6-06                                           | Safety gate: `neatChat.safety.types.ts` + `neatChat.safety.services.ts` + 32 tests. `checkSafety` priority order (degenerate → repetition-collapse → unknown-token). Full NEATchat suite: 10 suites, 306 tests. Coverage `100/100/100/100`. |
| 12   | W6-07                                           | Publishable product shape: feature-lane labels in demo HTML, evaluation/safety/routing re-exports in `index.ts`, README sections for routing, evaluation harness, safety gate, and scale limitations. `npm run docs` PASS.                  |
| 13   | W6-08                                           | Closure: tracker compressed, archived as `NEATchat_Followup.plans.md` + `NEATchat_Followup.logs.md`, README.md and Roadmap.md updated, plan sync PASS.                                                                                      |

## Acceptance summary

All six workstreams and all W6 steps accepted. The NEATchat follow-up lane:

- Delivered a persistent, checkpointable session identity with memory and
  routing observability.
- Shipped a stronger pretrained default seed with a documented honest-subset
  import contract.
- Added episodic memory, async-deferred adaptation, three-path routing, an
  attribution-aware regression harness, and a structured safety gate.
- Published the browser demo with explicit live/experimental/background-job
  lane labels.
- Maintained `100/100/100/100` coverage on every touched NEATchat boundary.
- Kept all W1-W5 invariants stable: routing is observability-only, no
  auto-promotion, safety gate is non-throwing, memory retrieval is weight-safe.

## Residual open items

- Broad-run default-heap OOM and two unrelated suite failures
  (`telemetry.facade.buffer.test.ts`, `evolveXor.test.ts`) remain outside
  NEATchat ownership and were not introduced by any NEATchat workstream.
- The non-ONNX parameter-vector conversion path for external seeds is the
  formally adopted approach; a recurrent ONNX reopen is a potential future
  amendment, not a current obligation.
