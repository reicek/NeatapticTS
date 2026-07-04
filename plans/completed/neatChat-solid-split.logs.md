# NEATchat SOLID Split Log

**Status:** [DONE]

## Audit scope

- Objective: split examples/neatChat/index.ts into small chapter-owned core
  boundaries while preserving the stable facade import path.

## Durable milestones

### [DONE] Folder-first chapter split

- Extracted core ownership chapters for constants, types, tokenization,
  sessions, A/B comparison, snapshots, and typed errors.
- Preserved examples/neatChat/index.ts as an orchestration-first stable facade.

### [DONE] Named error ownership and migration

- Added boundary-owned typed error classes in
  examples/neatChat/core/neatChat.errors.ts.
- Migrated snapshot and integer validation throws to NEATchat-specific typed
  errors without changing behavior.

### [DONE] Educational-docs follow-up closure

- Tightened JSDoc on the Step 8 touched boundary to clarify error semantics and
  failure modes.
- Added focused examples and explicit throws coverage where readers inspect
  import and validation behavior.

## Controls and evidence

- Validation used:
- npx tsc --noEmit -p tsconfig.test.json
- npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatChat/neatChat.test.ts
- npm run docs

## Reopen triggers

- Facade drift returns orchestration and validation sinks to one file.
- Core chapter docs become inconsistent with typed error behavior.
- Another responsibility seam in NEATchat needs a dedicated chapter boundary.
