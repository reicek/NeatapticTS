# NEATchat Live Safety Red Contract

**Status:** [DONE]
**Closed:** 2026-05-23

## Scope

- Reopened the live NEATchat exchange path inside `examples/neatChat/core/` so `runNeatChatExchange` enforces `checkSafety`, rejects broken fragments, and keeps all-unsafe turns on a bounded local fallback floor.
- Stayed local to `examples/neatChat/**`, owner-local validation, shipped-browser probing, and the narrow documentation follow-up already applied by `06-documenting`.

## Final state

- `runNeatChatExchange` now safety-screens routing candidates before commit, strips placeholder punctuation tokens, rejects incomplete prompt-echo fragments, and skips recent duplicate replies from the last three exchanges.
- When every surfaced candidate is unsafe, the live path now uses a vocabulary-aware deterministic fallback palette instead of resurfacing the top unsafe fragment or collapsing to the same reply across the accepted four-turn probe.
- The only requested documentation follow-up is already in place at `examples/neatChat/core/neatChat.session.services.ts` and `examples/neatChat/README.md`.

## Audit summary

- Final owner-local Jest gate stayed green: `npx jest --config=jest.config.mjs --no-cache --runInBand --runTestsByPath examples/neatChat/core/neatChat.live-flow.safety.test.ts --testNamePattern "does not loop the shipped-snapshot fallback floor when rephrase tell and plans are unavailable"` -> PASS (`1 passed, 11 skipped`).
- Browser build stayed green: `npm run build:neat-chat` -> PASS.
- Final shipped-browser probe cleared the requested four-turn bar on the shipped snapshot: `hello agent` -> `i see`, `how are you today?` -> `sounds great`, `what was that?` -> `i am good`, `looping?` -> `i see`.
- Tracker sync passed before archive: `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/neatChat-live-safety-red.plans.md` -> PASS (`0 errors, 0 warnings`).

## Reopen conditions

- Reopen only if the user wants a stricter extended-turn freshness bar than the accepted four-turn browser probe.
- Keep any reopen local to `examples/neatChat/core/neatChat.session.services.ts` unless a future request explicitly widens scope back to snapshot generation or browser-entry behavior.
- Current residual risk to preserve: the fifth probe turn reused `i see`, so mild extended-turn freshness drift still exists beyond the accepted bar.

## Audit log

- See `plans/completed/neatChat-live-safety-red.logs.md`.