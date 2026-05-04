# Methods Docs Plan

**Status:** [DONE]

## Scope

- Educational-docs pass for the shared vocabulary and chapter openings under
  `src/methods/`.
- Improve the generated first-contact reading path without changing runtime
  behavior.

## Final state

- The methods root and the key family chapters now open with clearer
  educational framing around what choices the families represent.
- Activation, gating, connection, rate, and mutation documentation now better
  answer the reader's boundary-level questions instead of acting like symbol
  shelves.
- No active backlog remains; this plan is now a reopen point for documentation
  drift in `src/methods/`.

## Audit summary

- Documentation-affecting passes regenerated docs with `npm run docs`.
- Type safety was preserved with `npx tsc --noEmit -p tsconfig.json`.

## Reopen conditions

- A methods chapter opening regresses below the current educational-docs bar.
- A new methods family needs the same chapter-framing treatment.
- Generator ordering causes a methods README to open from the wrong owner.

## Audit log

- Durable completion notes now live in
  [methods-docs.logs.md](methods-docs.logs.md).
