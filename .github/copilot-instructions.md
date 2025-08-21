# Repo Copilot Instructions — NeatapticTS

Purpose
-------
When generating, modifying, or suggesting code that touches files under `src/` or `test/`, follow the project `STYLEGUIDE.md` rules and perform the quick validations listed below before returning suggestions.

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
   - Short-id scan: flag uses of the short identifiers regex `\b(dx|dy|d|i|a|b|c|p|o|idx|cand|tries)\b` in changed files.
   - Tests heuristic: flag test files that contain more than one `expect(` occurrence (these should be split into multiple `it()` blocks).
   - JSDoc: for new exported symbols, ensure a JSDoc block with `@param`/`@returns` exists (or flag if missing).

   PowerShell examples (local validation)
   -------------------------------------
   Typecheck:
   ```powershell
   npx tsc --noEmit -p tsconfig.json
   ```

   Short-id scan:
   ```powershell
   Get-ChildItem -Path src,test -Recurse -Include *.ts,*.tsx | Select-String -Pattern '\b(dx|dy|d|i|a|b|c|p|o|idx|cand|tries)\b' -NotMatch '\b(i|j)\b' -List
   ```

   Test heuristic:
   ```powershell
   Get-ChildItem -Path test -Recurse -Include *.ts,*.tsx | ForEach-Object {
      $count = (Get-Content $_.FullName | Select-String 'expect\(' -AllMatches).Matches.Count
      if ($count -gt 1) { Write-Output "$($_.FullName): $count expects" }
   }
   ```

   What to include with a suggestion
   --------------------------------
   - A short validation summary (TypeScript: pass/fail, short-id matches: list or 0, test-expect heuristic: list or 0, JSDoc missing: list or 0).
   - If any issue can't be safely fixed automatically, include a TODO comment at the top of the changed file and a one-line explanation in the patch.

- A runnable patch or new file content that follows the rules above.

