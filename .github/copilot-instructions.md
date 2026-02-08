# Repo Copilot Instructions — NeatapticTS

Purpose
-------
When generating, modifying, or suggesting code that touches files under `src/` or `test/`, follow the project `STYLEGUIDE.md` rules and perform the quick validations listed below before returning suggestions.

**CRITICAL: Fix-first strategy for test failures**
---------------------------------------------------
When asked to fix multiple test failures:

1. **FOCUS on fixing ALL issues systematically BEFORE running tests**
   - Create or update a comprehensive fix plan (e.g., `plans/TestsFix.md`)
   - Categorize errors by type (TypeScript compilation, async/await, runtime logic)
   - Prioritize: HIGH (compilation blockers) → MEDIUM (easy fixes) → LOW (investigation needed)
   - Mark completed items with ✅ as you progress

2. **DO NOT run tests or check test output during the fix phase**
   - Avoid running `npm test`, `npm run test:silent`, or any test execution commands
   - Do NOT run partial test checks like `Select-String -Pattern "●"` to see failures
   - Do NOT attempt to run individual tests to "verify" fixes
   - TypeScript compilation checks (`npx tsc --noEmit`) are acceptable to validate type fixes
   - **REMEMBER**: You already have the test failure output - use it to guide your fixes

3. **ONLY run full test suite AFTER all fixes are applied**
   - Apply ALL planned fixes first (optimistically)
   - Use `npm test` or `npm run test:silent` for final validation ONLY
   - Analyze remaining failures and update the plan accordingly
   - If you feel tempted to run a test, STOP and apply more fixes instead

4. **Follow the plan strictly**
   - Do not improvise or skip ahead
   - Execute fixes in the documented priority order
   - Update the plan with ✅ checkmarks and status notes as you complete each item
   - Mark ALL items completed before running tests

**Why this matters:**
- Running tests interrupts the systematic fix workflow
- Test output during fixes causes distraction and context switching
- You already have all the error information needed to fix issues
- Optimistic fixing is faster than iterative test-fix-test cycles
- This strategy ensures thorough, complete fixes before validation

This strategy prevents distraction, maintains focus, and ensures systematic completion of all fixes before validation.


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

   JSDoc-for-constants rule: All exported or shared default constants in `src/` and `test/` must include a concise educational JSDoc explaining what the value controls (e.g., decay factor meaning, floor rates). Keep descriptions short and clarifying.

3. Tests: follow the single-expect rule. Each `it()` (or `test()`) must have exactly one top-level `expect(...)` statement. If multiple assertions are needed, split into multiple `it()` cases or use helper assertions.

4. Constants: replace magic numbers with named `export const` or class-private `static #` constants with a short JSDoc.

5. Comments: methods should have step-level inline comments explaining intent (not every line). Use numbered steps where helpful.

6. Lookup tables and enums: prefer a single table/enum for small fixed mappings (for example direction deltas) and helper methods like `#opposite(direction)` rather than scattered arithmetic.

7. Types: avoid `any` and `unknown` in `src/` and `test/`. Use precise types or `// eslint-disable-next-line @typescript-eslint/no-explicit-any` with a short justification comment.

8. Local helper structure preference:
   - For new or refactored functions that introduce internal helpers, order the function as:
       1) Local variables/constants at top
       2) Declarative logic (calls to helpers)
       3) Return (fold)
       4) Internal helper function declarations at the end of the parent function
    - Helpers should be small and pure where practical, with step-level inline comments and JSDoc.

9. Mandatory implementation pattern (always; keep cognitive complexity low):
   - Applies to all new code and any modified/refactored code in `src/` and `test/`.
   - Prefer a *declarative top-level flow* ("collect → transform → fold/return") over deeply nested control flow.
   - Avoid ternary chains (especially nested) for multi-branch fallback logic; use named resolver helpers with early returns instead.
   - When normalizing legacy/loose data, isolate type assertions/casting into a single helper and keep the rest strongly typed.
   - Keep helpers after the fold, and give each helper a single responsibility (SOLID: SRP). If the logic reads like a decision tree, it likely wants 2–4 small helpers.

Example (ideal structure)
-------------------------
```ts
export function exampleMethod(input: Input) {
   const constants = /* ... */;
   const locals = /* ... */;

   if (/* guard */) return /* fold */;

   const stepOne = helperOne(input, locals, constants);
   const stepTwo = helperTwo(stepOne, locals, constants);
   return helperThree(stepTwo, locals, constants);

   /** @param value - Input. @returns Intermediate. */
   function helperOne(value: Input, _locals: unknown, _constants: unknown): Intermediate {
      // Step 1: ...
      return /* ... */;
   }

   /** @param value - Intermediate. @returns Intermediate. */
   function helperTwo(value: Intermediate, _locals: unknown, _constants: unknown): Intermediate {
      // Step 1: ...
      return /* ... */;
   }

   /** @param value - Intermediate. @returns Output. */
   function helperThree(value: Intermediate, _locals: unknown, _constants: unknown): Output {
      // Step 1: Fold/return.
      return /* ... */;
   }
}

type Input = unknown;
type Intermediate = unknown;
type Output = unknown;
```

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

   Test failure workflow
   ---------------------
   When fixing multiple test failures:
   1. Create/update a comprehensive plan document (e.g., `plans/TestsFix.md`)
   2. Apply ALL fixes systematically without running tests
   3. Validate TypeScript compilation with `npx tsc --noEmit -p tsconfig.test.json`
   4. ONLY run `npm test` after all planned fixes are complete
   5. Analyze results and iterate on remaining issues

