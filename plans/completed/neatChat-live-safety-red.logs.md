# NEATchat Live Safety Red Contract - Audit Log

**Status:** [DONE]
**Closed:** 2026-05-23

## Pass history

| Pass | Focus                            | Outcome                                                                                                                                                                                                                         |
| ---- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Live safety reconnection         | `checkSafety` was reattached to `runNeatChatExchange`; the live path now prefers the next safe local candidate instead of committing the top unsafe reply.                                                                      |
| 2    | Fragment and duplicate rejection | Placeholder punctuation, incomplete fragments such as `how was your` / `i will`, and recent duplicate replies were rejected before surface delivery.                                                                            |
| 3    | Snapshot and runtime loop        | The snapshot generator/runtime bottlenecks were isolated, the capped-source training path was corrected, and the rerun envelope was reduced enough to continue the quality loop without widening beyond `examples/neatChat/**`. |
| 4    | Browser fallback polish          | The no-safe-candidate palette became vocabulary-aware and deterministic, which removed the shipped-snapshot four-turn fallback collapse.                                                                                        |
| 5    | Closeout                         | Focused Jest slice PASS, `npm run build:neat-chat` PASS, shipped-browser four-turn probe PASS, doc follow-up already landed, and the tracker was compressed for archive.                                                        |

## Acceptance summary

- Closed the requested live safety boundary without widening into unrelated library or demo work.
- Preserved the shipped-browser practical bar: no placeholder leakage, no obviously broken fragments, and no repetition collapse across the accepted four-turn probe.
- Left one explicit reopen condition only: stricter extended-turn freshness beyond the accepted four-turn bar.

## Residual open items

- The fifth shipped-browser probe turn reused `i see`; treat that as a reopen-only freshness risk, not a blocker to this closeout.
- If reopened, keep the first move local to `examples/neatChat/core/neatChat.session.services.ts` rather than defaulting back to snapshot-generation work.
