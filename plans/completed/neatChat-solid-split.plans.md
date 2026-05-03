# NEATchat SOLID Split

**Status:** [DONE]

## Scope

- Apply a folder-first SOLID split to the overloaded NEATchat boundary while
  keeping examples/neatChat/index.ts as the stable public facade.
- Move responsibility clusters into small core chapters under
  examples/neatChat/core/.
- Complete the mandatory educational-docs follow-up pass on touched chapters.

## Final state

- The NEATchat facade in examples/neatChat/index.ts remains thin and stable.
- Core responsibilities now live in dedicated chapter files: constants, types,
  tokenization utilities, session services, A/B services, snapshot services,
  and typed error classes.
- Step 9 educational-docs follow-up is complete for the Step 8 touched
  boundary, including clearer typed-error semantics and examples in the core
  docs surface.
- No in-plan split backlog remains for this workstream.

## Audit summary

- Step 9 docs pass updated source JSDoc in:
  - examples/neatChat/core/neatChat.errors.ts
  - examples/neatChat/core/neatChat.snapshot.services.ts
  - examples/neatChat/core/neatChat.tokenization.utils.ts
- Validation completed with:
  - npx tsc --noEmit -p tsconfig.test.json
  - npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatChat/neatChat.test.ts
  - npm run docs

## Reopen conditions

- NEATchat core chapters drift back into facade-heavy orchestration.
- Generated docs readability regresses on the NEATchat split boundary.
- A new responsibility cluster appears that should become its own chapter.

## Audit log

- Durable completion notes live in
  [neatChat-solid-split.logs.md](neatChat-solid-split.logs.md).
