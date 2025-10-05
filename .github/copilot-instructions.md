# Repo Copilot Instructions — NeatapticTS

Purpose
-------
When generating, modifying, or suggesting code that touches files under `src/` or `test/`, follow the project `STYLEGUIDE.md` rules and perform the quick validations listed below before returning suggestions.

ES2023-first policy (strict)
----------------------------
For educational clarity and a modern look, always prefer idiomatic ES2023 syntax when it improves readability or safety without changing behavior. This repo is intentionally opinionated: use the immutable array methods and modern language constructs by default.

Prefer (non-exhaustive):
- Arrays: `toSorted`, `toReversed`, `toSpliced`, `with`, `.at(-1)`, `findLast`, `findLastIndex`.
- Objects: spread/rest over `Object.assign` for shallow copies and merges.
- Optional chaining `?.` and nullish coalescing `??` (avoid `||` for defaulting unless you mean falsy semantics).
- Deep clone: `structuredClone` (or the project helper `safeStructuredClone` when cross-env safety is needed).
- Errors: `Error` with `{ cause }` (e.g., `new Error(msg, { cause })`).
- Numerics: numeric separators for long literals (for readability only, not to change values).
- Modules: ES modules `import`/`export` over CommonJS `require`/`module.exports` (follow the repo’s phase plan; new code should be ESM). 

Avoid (legacy/less clear):
- In-place `sort`, `reverse`, `splice` in code paths that expect immutability; use the ES2023 immutable variants above.
- `Object.assign({}, obj)` or `Object.assign([], arr)` for cloning; use object/array spread.
- `JSON.parse(JSON.stringify(x))` for deep clone; use `structuredClone`/`safeStructuredClone`.
- Index math like `arr[arr.length - 1]`; prefer `arr.at(-1)` when readability benefits.
- CommonJS `require()` in new or refactored modules; prefer ESM.

How to use these instructions
-----------------------------
- Always prefer to produce code that already satisfies the style guide.
- If you cannot fully transform a file (large refactor), return a patch with clear TODO comments, an explicit list of remaining violations, and small, safe automated fixes where possible.
- If you propose changes that alter public behavior, include tests and TypeScript typechecks.

Strict rules to enforce (apply to any suggestion touching `src/` or `test/`)
---------------------------------------------------------------------
1. Naming: avoid short local identifiers. Do not use these short names for non-trivial locals: `dx`, `dy`, `d`, `i`, `a`, `b`, `c`, `p`, `o`, `cand`, `tries`, `idx`.
   - If the original code uses a short name in a tiny loop (1–3 lines) and it is clearly idiomatic, allow `i`, `j` only.
   - Prefer descriptive names: `candidateDirection`, `bestDistance`, `currentPosition`.

2. JSDoc: exported classes/functions/constants and public methods must have JSDoc with `@param` and `@returns` where appropriate. Add short `@example` when behavior is non-obvious.

3. Tests: follow the single-expect rule. Each `it()` (or `test()`) must have exactly one top-level `expect(...)` statement. If multiple assertions are needed, split into multiple `it()` cases or use helper assertions.

4. Constants: replace magic numbers with named `export const` or class-private `static #` constants with a short JSDoc.

5. Comments: methods should have step-level inline comments explaining intent (not every line). Use numbered steps where helpful.

6. Lookup tables and enums: prefer a single table/enum for small fixed mappings (for example direction deltas) and helper methods like `#opposite(direction)` rather than scattered arithmetic.

7. Types: avoid `any` and `unknown` in `src/` and `test/`. Use precise types or `// eslint-disable-next-line @typescript-eslint/no-explicit-any` with a short justification comment.

Automated validations to run before finalizing a suggestion
-------------------------------------------------------
When you modify or create files under `src/` or `test/`, run (or advise running) these quick validations. If you cannot run them, still make sure your suggestion would pass them.

1) TypeScript diagnostics

   # Copilot instructions — STYLEGUIDE light checks

   Purpose
   -------
   Give brief, actionable guidance so suggestions touching `src/` or `test/` prioritize compliance with `STYLEGUIDE.md`.

   Keep it light: prefer small, automated checks and a short validation summary with every patch.

   Quick checks to run (recommended)
   --------------------------------
   - TypeScript: run `npm run build` and report pass/fail.
   - Tests heuristic: flag test files that contain more than one `expect(` occurrence (these should be split into multiple `it()` blocks).
   - JSDoc: for new exported symbols, ensure a JSDoc block with `@param`/`@returns` exists (or flag if missing).
   - ES2023 modernization: flag legacy patterns and suggest modern equivalents (see below one-liners).

   PowerShell examples (local validation)
   -------------------------------------
   Typecheck:
   ```powershell
   npx tsc --noEmit -p tsconfig.json
   ```

   What to include with a suggestion
   --------------------------------
   - A short validation summary (TypeScript: pass/fail, short-id matches: list or 0, test-expect heuristic: list or 0, JSDoc missing: list or 0).
   - ES2023: list any flagged legacy patterns and the intended replacements (e.g., `Object.assign` -> spread, `arr[arr.length-1]` -> `arr.at(-1)`, `JSON.parse(JSON.stringify())` -> `structuredClone`).
   - If any issue can't be safely fixed automatically, include a TODO comment at the top of the changed file and a one-line explanation in the patch.

