# NeatapticTS Style Guide

This repository is an educational neural-network library. Clarity, pedagogy, and reproducibility matter as much as correctness. This style guide enforces strict naming, documentation, and testing practices to make the codebase approachable and maintainable.

---

## Goals

- Be explicit and self-documenting: prefer long, descriptive local variable names.
- Make code educational: thorough JSDoc, in-method comments, and examples.
- Keep behavior stable: no semantic changes without tests and/or benchmarks.
- Use ES2023 idioms where they improve clarity (private `#` fields, readonly types).
- Protect performance-sensitive code (especially `src/neat/**`) behind micro-benchmarks before changing algorithms.

---

## Naming rules (very strict)

- Always use descriptive names for local variables inside functions and methods. Replace 1–3 letter local identifiers.
  - Good examples: `candidateDirection`, `bestDistance`, `currentPosition`, `stepsSinceImprovement`, `entropySum`.
  - Avoid: `dx`, `dy`, `d`, `i`, `a`, `b`, `c`, `p`, `o`, `cand`, `tries`, `idx`.
- Exceptions:
  - `x` and `y` are allowed only as coordinate parameter names in public APIs where it improves readability.
  - Very short, trivial loop indices (`i`, `j`) are allowed in tiny loops (1–3 lines) but prefer `stepIndex`, `rowIndex`, etc.
--

## Control flow preference

- Prefer `switch`/`case` for multi-branch conditionals that test the same variable or map discrete patterns to outcomes. This improves readability for dispatch-style logic (for example mapping direction deltas to indices).
- For ordered predicates (where branch order matters but there's no single matched value), prefer the explicit pattern:

```ts
switch (true) {
  case predicateA:
    // ...
    break;
  case predicateB:
    // ...
    break;
}
```

  This makes evaluation order intentional and avoids long `else if` chains.
- Do not use `switch` where behaviour depends on complex short-circuiting across unrelated predicates; in those cases prefer `if/else`.
- When converting `else if` chains to `switch`/`switch(true)`, ensure coverage by tests and keep behavior identical.
- Use camelCase for local variables, parameters, and non-exported functions.
- Reducer callbacks: prefer `accumulator` and `value` (or `acc`, `val`) over `a`, `b`.

Consistent naming makes intent obvious and reduces cognitive load when reading learning algorithms.

---

### Enums and lookup tables for small fixed mappings

- For small, fixed mappings (for example mapping cardinal directions to coordinate deltas), prefer a single, named table or enum instead of repeated switch blocks or scattered arithmetic like `(dir + 2) % 4`.
- Benefits:
  - Centralizes intent and values (`#DIRECTION_DELTAS` or `enum Direction { North = 0, East = 1, ... }`).
  - Makes it trivial to provide helper utilities (for example `#opposite(direction)` or `#delta(direction)`) and to change the action dimensionality in one place.
  - Reduces off-by-one and magic-number mistakes across the file.
- Example pattern (class-private):

```ts
static #DIRECTION_DELTAS = [[0,-1],[1,0],[0,1],[-1,0]];
static #opposite(d:number) { return (d + MazeMovement.#DIRECTION_DELTAS.length/2) % MazeMovement.#DIRECTION_DELTAS.length }
```

Use such helpers instead of ad-hoc arithmetic sprinkled throughout the code.

## Constants and magic numbers

- Replace magic numbers with private class constants (`static #MY_CONST = ...`) when they are implementation details.
- For public constants (exported), use `export const descriptiveName = ...` with a leading JSDoc comment.
- Place a short JSDoc above each constant. Example:

```ts
/** Timeout (in ms) used for network training demos */
export const trainingDemoTimeout = 5000;
```

In classes prefer `static #PRIVATE_CONSTANT` and name them in CLEAR_DESCRIPTIVE_STYLE but in this repo we use `#CamelCase` for private statics to emphasize readability.

---

### Typed-array pooling and scratch buffers (strong preference)

- For performance-sensitive, hot-path code (for example vision/preprocessing, inner-loop simulators, and small per-step helpers) prefer reuse of typed-array scratch buffers instead of allocating new typed arrays on every call.
- Benefits:
  - Reduces GC pressure and short-lived allocations in tight loops.
  - Often yields measurable throughput improvements in microbenchmarks for per-step code.

- Rules and caveats:
  - Keep the pooled buffers private to the module or class (use `static #` fields) so ownership is clear.
  - Document the non-reentrant nature of pooled buffers in the JSDoc for the method (for example: "This method reuses internal scratch buffers and is not reentrant/should not be called concurrently").
  - Initialize (`fill`) the scratch buffers at the start of each invocation to avoid leaking state between calls.
  - Do not expose pooled buffers directly to callers — copy or return plain numbers/objects when returning results.

- Example (class-private pooling pattern):

```ts
// class-private pooled buffers
static #SCRATCH_X = new Int32Array(4);
static #SCRATCH_Y = new Int32Array(4);

// In the hot method:
const xs = MazeVision.#SCRATCH_X;
const ys = MazeVision.#SCRATCH_Y;
xs.fill(0); ys.fill(0);
// populate & use xs/ys for local work; do not return them directly
```

- When to avoid pooling:
  - Large, rarely-used data structures (pooling adds complexity without payoff).
  - When your code must be safely reentrant or run concurrently — prefer per-call allocations or explicit pool checkout semantics.

Add an inline `@remarks` note to methods that rely on pooled buffers so readers of generated docs see the constraint.


### Vision inputs and grouped indices

- For structured input vectors (for example the maze "vision" arrays), avoid scattered numeric index literals across the codebase. Instead:
  - Centralize group starts and lengths as private class constants (for example `static #VISION_LOS_START`, `static #VISION_GROUP_LEN`).
  - Provide a small private helper (placed inside the same class when it needs access to private `#` fields) to read or sum groups: this clarifies intent and reduces duplicated indices.
  - Prefer named helpers (`#sumVisionGroup`, `#hasNearbyGradient`) over inline index arithmetic.

- When helper logic needs access to private `#` constants, place the helper inside the class so it can use the private fields directly (this keeps the API surface small and avoids exporting implementation details).

## JSDoc, inline comments, and educational docs

This is an educative library. All public API items and most internal behaviors should be documented with clear JSDoc and inline explanatory comments.

- Add JSDoc to all:
  - exported classes, interfaces, functions, and constants
  - public methods and constructors
  - any non-trivial internal helper function you want included in generated docs
- JSDoc requirements:
  - Use `@param` for each parameter with a short description and example when appropriate.
  - Use `@returns` with the return shape and an example value.
  - Use `@example` for short usage snippets when the behavior is non-obvious.
  - Use `@remarks` for caveats, algorithmic notes, and differences to standard approaches.
  - Do not include secrets or sensitive data — generated docs are public.

- Inline comment requirements inside methods:
  - Add `//` comments that explain each conceptual step of the method ("// Step 1: calculate vision inputs"), not line-by-line low-level noise.
  - Comment intent, not just what the language expresses; justify unusual choices.

Example JSDoc for a method and a constant (educational style):

```ts
/**
 * Reward scale applied to most shaping rewards.
 * Smaller values reduce selection pressure in demos—use for gentle gradients.
 * @example 0.5
 */
static #REWARD_SCALE = 0.5;

/**
 * Simulate an agent walking the maze.
 *
 * This simulation is intentionally minimal to demonstrate how networks learn
 * to solve a navigation task. It logs intermediate signals that are useful
 * for visualization in tutorials.
 *
 * @param network - The network controlling the agent (see `INetwork`).
 * @param encodedMaze - 2D grid where `-1` is a wall and `0+` is free space.
 * @param startPos - Start coordinates: `[x, y]`.
 * @param exitPos - Goal coordinates: `[x, y]`.
 * @returns An object with `success`, `steps`, `path`, and `fitness`.
 * @example
 * const result = MazeMovement.simulateAgent(net, maze, [1,1], [10,10]);
 */
static simulateAgent(...) { /* method */ }
```

---

## Testing requirements (strict)

All tests in the repository must follow these rules to keep examples, tutorials, and regressions easy to reason about and to ensure the learner-friendly structure.

- Single expectation per `it()` block:
  - Each `it('should ...', () => { ... })` should have exactly one top-level expectation (`expect(...)`).
  - If a scenario requires verifying multiple outcomes, split them into multiple `it()` tests.
  - Use helper functions or shared `beforeEach`/`describe`-scoped data to avoid repeating setup.

- Follow the AAA pattern (Arrange, Act, Assert):
  - Arrange: build inputs, stub dependencies, create the system under test.
  - Act: perform the operation under test.
  - Assert: make a single expectation that clearly states the outcome.

- Group tests into scenarios with `describe()`. Nest scenarios as needed — there is no limit on nesting depth.
  - Define common test data at the `describe()` level using `const` or `let` with `beforeEach` if mutability is required.
  - Each nested `describe()` represents a more specific case; override or narrow data in deeper scopes.

- When possible, define common testing data directly on the `describe()` scope and then write assertions for it. This makes nested scenarios read like a decision tree and produces highly focused single-expectation tests.

- Aim for 100% coverage on logic you change or add. For legacy modules where 100% isn't feasible in one pass, document missing coverage areas and add tests incrementally.

- Check existing tests and folder patterns before creating a new test file:
  - Use existing folders (e.g. `test/neat/`) and file naming conventions as a guide.
  - Reuse and extend existing fixtures/helpers instead of adding new duplicate files.

Example test layout:

```ts
describe('MazeMovement.simulateAgent', () => {
  const start = [1, 1];
  const exit = [3, 1];
  let maze: number[][];

  beforeEach(() => {
    // Arrange: small 3x3 corridor
    maze = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]
    ];
  });

  describe('when network always chooses greedy path', () => {
    it('succeeds in reaching the exit', () => {
      // Act
      const result = MazeMovement.simulateAgent(fakeGreedyNetwork, maze, start, exit);
      // Assert (single expectation)
      expect(result.success).toBe(true);
    });
  });
});
```

Notes on the single-expectation rule:
- Use helper assertions when you need to check multiple derived values by creating separate `it()` blocks. For example, one `it()` for success boolean and another `it()` for fitness threshold.

---

## Code comments and method-level detail

- For each method, add step-level inline comments. Use numbered or named steps when helpful:

```ts
// Step 1: Mark position visited
// Step 2: Compute vision inputs (LOS + gradients)
// Step 3: Activate network and compute action stats
// Step 4: Apply move and update rewards
```

- Comments should be educational: explain "why" and "what effect this has on learning" when appropriate.

---

## Documentation build notes

- JSDoc is compiled to HTML docs. Keep public JSDoc examples concise and redact private details.
- Use `@example` that can be copy-pasted into a small REPL; examples should be short and runnable when possible.

---

## Applying these rules in practice

- When modernizing a file:
  1. Read the existing file and nearby tests.
 2. Make a small, focused change (naming, JSDoc, constant extraction).
 3. Run per-file TypeScript diagnostics.
 4. Run tests for the affected behavior (or the full suite if safe).
 5. Commit with a focused message: `style(test): enforce AAA + single-expect tests for mazeMovement`.

---

## Quick grep to find short identifiers to fix

Use this regex to find likely candidates to rename (review results manually):

```
\b(dx|dy|d|i|a|b|c|p|o|idx|cand|tries)\b
```

---

If you want, I can run a repo-wide scan and propose a patchset that renames short locals automatically (I will not apply bulk renames without your approval). If you'd like that, say which folders to prioritize (suggestion: `test/examples/asciiMaze/` then `src/`).

Thank you — I'll keep the guide updated as we iterate on modernization and educational docs. <3
