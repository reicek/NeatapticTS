# Interactive Examples + Learning Path — Audit Log

**Status:** [DONE]

## Pass history

| Step                               | Outcome                                                                                                                                                                                                                                                          |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Freeze host and boundary decisions | Kept `examples/` as canonical root; reused browser/docs plumbing; scoped NEATchat out.                                                                                                                                                                           |
| Step 1 — Starter tranche shape     | Froze folder-based pattern: `examples/<name>/index.ts`, `README.md`, `index.html`, `browser-entry.ts`. Defined starter set: helloNetwork, evolveXor, sequenceReset, browser quickstart.                                                                          |
| Step 2 — Hello Network             | `examples/helloNetwork/` with deterministic inference walkthrough, Jest test, `run.ts`, browser entry, and README.                                                                                                                                               |
| Step 3 — Evolve XOR                | `examples/evolveXor/` with seeded feed-forward NEAT loop, bounded behavioral test, `run.ts`, browser entry, README. Repaired add-node acyclic split and duplicate-innovation bug in the library.                                                                 |
| Step 4 — Sequence Reset            | `examples/sequenceReset/` with fixed-weight LSTM, three-run comparison, behavioral test, `run.ts`, browser entry, README, Mermaid diagram. Library fix: `Node.clear()` was resetting gated connection gains to `0` instead of `1`.                               |
| Step 5 — Browser quickstart        | Covered by the three starter browser-entry pages already published through `docs/examples/`. No separate quickstart surface needed.                                                                                                                              |
| Step 6 — Learning path docs        | Rewrote `examples/README.md` as a guided syllabus with starter-path table; demoted flagship examples to second half; added NEATchat note.                                                                                                                        |
| Step 7 — Smoke validation          | Added `examples/starter-examples.smoke.test.ts` (8 checks: export surface + format-title). Added `starter-examples` Jest project in `jest.config.mjs`. Added `test:smoke:starters` npm script. Excluded smoke file from `default` project to prevent double-run. |

## Closed boundary

All deliverables shipped and validated. NEATchat remains a separate follow-up in `plans/NEATchat.plans.md`.
