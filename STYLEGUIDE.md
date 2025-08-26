# NeatapticTS Style Guide

This repository is an educational neural-network library. Clarity, pedagogy, and reproducibility matter as much as correctness. This style guide enforces strict naming, documentation, and testing practices to make the codebase approachable and maintainable.

> Modernization note (2025 refresh): This guide now embeds ES2023+ language features and deeper performance + memory best practices for browser + Node targets. The neural components should be both *educative* and *fast*: predictable object shapes, pooled typed arrays, cache‑friendly data layouts, and off‑main‑thread scheduling where appropriate. Sections added/expanded: ES2023 Modern Features, Memory & Layout, Typed-Array Pooling (deep dive), Caching Strategy, Parallelism & Scheduling, Microbenchmarking, and Determinism.

---

## Goals

- Be explicit and self-documenting: prefer long, descriptive local variable names.
- Make code educational: thorough JSDoc, in-method comments, and examples.
- Keep behavior stable: no semantic changes without tests and/or benchmarks.
- Use ES2023 idioms where they improve clarity (private `#` fields, readonly types).
- Protect performance-sensitive code (especially `src/neat/**`) behind micro-benchmarks before changing algorithms.
- Optimize for *steady-state throughput* of large populations in browsers (GC pressure minimization, cache locality, minimal hidden class churn).
- Maintain deterministic simulation modes (seeded RNG + stable iteration ordering) for reproducible research demos.
- Encourage ergonomic profiling: embed lightweight instrumentation hooks guarded by feature flags (no permanent perf tax).
- Provide clear extension seams for WASM / WebGPU acceleration without forcing them.

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

### ES2023+ Modern Language Features (Use Intentionally)

Adopt modern features when they *increase clarity or safety* — not just novelty.

- Private fields & methods: `#privateField` for internal state, especially pooled buffers and scratch counters. Prefer these over closures that allocate per-instance.
- Static initialization blocks: use sparingly to precompute lookup tables (e.g. activation function dispatch maps) once per class.
- New Array helpers: `toSorted()`, `toReversed()`, `with()`, `findLast()`, `findLastIndex()` — prefer non-mutating forms in pure utility paths; retain in-place ops in hot loops when mutation avoids allocations.
- `structuredClone` over manual deep copies for simple graph-free objects (avoid for large typed arrays—prefer explicit `slice()` or shared views).
- Optional chaining / nullish coalescing for concise guard code (avoid stacking in hot inner loops where explicit branching improves JIT clarity).
- Temporal / Records-Tuples: **Do not use** until fully standardized & broadly shipped—keep portability.
- Top-level await: avoid in library code (increases module graph latency); prefer explicit async init functions.

Performance caveat: Non-mutating array helpers allocate new arrays. In hot loops use preallocated buffers or in-place mutation, adding comments clarifying the decision.

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

When mapping operation kinds (activation, mutation operators), prefer a single frozen `const` dispatch object or a `switch` — measure both. Enumerations should avoid sparse numeric assignments (dense 0..N improves branch prediction & potential table lookups).

## Constants and magic numbers

- Replace magic numbers with private class constants (`static #MY_CONST = ...`) when they are implementation details.
- For public constants (exported), use `export const descriptiveName = ...` with a leading JSDoc comment.
- Place a short JSDoc above each constant. Example:

```ts
/** Timeout (in ms) used for network training demos */
export const trainingDemoTimeout = 5000;
```

In classes prefer `static #PRIVATE_CONSTANT` and name them in CLEAR_DESCRIPTIVE_STYLE but in this repo we use `#CamelCase` for private statics to emphasize readability.

Prefer *numeric* constants over string literals inside tight loops. Convert human-readable string configuration to numeric codes **during setup** (one-time) then operate on numeric codes during simulation/evolution.

Freeze large shared configuration objects (`Object.freeze`) after construction to lock hidden class shape early and help engines optimize property access.

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

#### Deep Dive: Patterns & Anti‑Patterns

| Scenario | Recommended Pattern | Rationale |
|----------|--------------------|-----------|
| Repeated forward pass over many genomes | Single contiguous weight `Float32Array` slice per network, views for layers | Improves cache line utilization & enables vectorized/WebGPU future path |
| Short-lived intermediate activations | Class-static pooled `Float32Array` sized to max layer width | Avoid per-pass allocation & GC | 
| Variable-sized temporary (depends on layer count) | Size bucket pools (e.g. powers of two) + checkout function | Amortizes large reallocation spikes |
| Rare debug path (export JSON) | Allocate ad-hoc arrays normally | Keeps pooling surface minimal |

##### Pool Implementation Guidelines

1. Centralize pools per *concern* (activation scratch, mutation temp weights, maze vision) — avoid one generic mega-pool.
2. Provide a small internal helper: `checkoutActivationBuffer(requiredLength)` that returns a view large enough (grows underlying array if needed) — **never** shrink synchronously (let caller reuse).
3. Return *views* (`subarray`) instead of copying when safe; document aliasing semantics.
4. Always clear or `fill(0)` buffers on paths where stale values could impact logic or determinism; skip clearing when the algorithm overwrites every index deterministically (document the invariant).
5. NEVER expose pooled buffers directly through public API return values; copy if external mutation risk exists.
6. For cross-call reentrancy (e.g., parallel evaluation in a Worker pool) either:
   - Use per-Worker pools (simplest), or
   - Implement a lock-free ring of buffers (Array of N typed arrays) and an atomic index (with `Atomics.add` on a `SharedArrayBuffer`).

##### Choosing Element Types

- Default to `Float32Array` for neural weights & activations (browser-friendly, GPU-aligned, halved bandwidth vs `Float64Array`).
- Use `Int32Array` or `Uint16Array` for indices, innovation numbers, or categorical encodings.
- Avoid `Float64Array` unless a precision error demonstrably harms learning (document the experiment showing necessity).

##### Layout: Structure of Arrays (SoA) vs Array of Structures (AoS)

- Prefer SoA for evolutionary metadata (separate typed arrays: `fitness`, `species`, `age`) to enable vector-style scans & SIMD-friendly future paths.
- Use AoS (objects) only for surfaces consumed directly by end users or for sparse optional metadata.

##### Determinism Considerations

- Reusing buffers can leak nondeterministic content if not cleared and if algorithms read before write; enforce ordering or zero-fill.
- Provide a `DETERMINISTIC` build flag (env or constant) to force zeroing + stable sorts (avoid `Array.prototype.sort` without comparator—spec allows tie reordering).

##### Example Pattern (Expanded)

```ts
class ActivationRunner {
  static #ActivationScratch = new Float32Array(1024); // grows dynamically
  static #MaxSize = 1024;

  /** Acquire a scratch buffer (view) of at least the requested length. */
  static #acquire(length: number): Float32Array {
    if (length > ActivationRunner.#MaxSize) {
      // Grow by 2x strategy to reduce realloc churn
      let newSize = ActivationRunner.#MaxSize;
      while (newSize < length) newSize *= 2;
      ActivationRunner.#ActivationScratch = new Float32Array(newSize);
      ActivationRunner.#MaxSize = newSize;
    }
    return ActivationRunner.#ActivationScratch.subarray(0, length);
  }

  /**
   * Compute layer activations.
   * @remarks Non-reentrant (shared scratch). Zero-fills only in deterministic mode.
   */
  static runLayer(weights: Float32Array, inputs: Float32Array, length: number, deterministic = false): Float32Array {
    const out = ActivationRunner.#acquire(length);
    if (deterministic) out.fill(0);
    // Hot loop: overwrite every index (safe to skip fill in non-deterministic mode)
    for (let index = 0; index < length; index++) {
      out[index] = Math.tanh(weights[index] * inputs[index]);
    }
    return out;
  }
}
```

##### Anti-Patterns

- Creating new typed arrays inside *nested* loops.
- Using `Array<number>` for dense numeric vectors in hot paths (boxed numbers, poorer locality).
- Returning pooled buffers to user code that might store them long-term.
- Over-pooling (managing pools for objects created a handful of times per second—added complexity with no benefit).

##### Memory Pressure Monitoring

Instrument occasionally with `performance.measure()` around evolutionary generations and record allocated bytes deltas (in Node via `process.memoryUsage().heapUsed`). Document any significant regressions in PR descriptions.

---

### Memory & Data Layout Optimization

1. Flatten network weight matrices into a single `Float32Array` per genome; maintain an index map for layer offsets. Avoid nested arrays.
2. Precompute activation function dispatch indices (e.g., 0 = tanh, 1 = sigmoid) and store in a compact `Uint8Array` aligned with neuron ordering.
3. Keep frequently accessed scalar properties (e.g., `fitness`, `age`) on the root genome object (first allocation) to reduce property lookups through nested objects.
4. Avoid polymorphism in hot loops: ensure objects iterated in large batches share the same hidden class (construct them via a single factory function in consistent property order).
5. Use bit packs (`Uint32` flags) for boolean feature toggles when they are evaluated in tight evolutionary scoring loops.
6. Reuse `TextEncoder` / `TextDecoder` instances (class-static) for serialization tasks.
7. For large immutable lookup data (e.g., activation LUTs), place them in module scope and freeze.

#### Contiguous Genome Buffer (Illustrative)

```ts
// Single buffer layout (example): [ meta (4 ints) | layer1 weights | layer2 weights | biases | ... ]
// meta: [inputCount, hiddenCount, outputCount, activationTableOffset]
```

Provide helper offsets so code never hardcodes numeric positions; document structure in JSDoc.

---

### Caching Strategy & Invalidations

Use caching where recomputation cost dominates and inputs are stable; *never* silently cache mutable objects without versioning.

- Derive cache keys from structural hashes (counts + configuration) not from object identity.
- Maintain a simple `generationTag` (increment per topology mutation) — invalidate any topology-derived caches when it changes.
- For computed innovation maps, store a `Map<string, number>` and reuse across genome mutations within the same generation.
- Provide explicit `clearCaches()` dev helper for debugging memory leaks.

Document each cache with: purpose, key composition, invalidation triggers.

---

### Parallelism, Workers & Scheduling

Browser main thread must remain responsive:

1. Offload bulk fitness evaluation to a Worker pool (size = `navigator.hardwareConcurrency - 1` capped at 4 by default).
2. Use transferable objects (`ArrayBuffer`) rather than structured clone for large numeric buffers between main and workers.
3. Batch messages (evaluate multiple genomes per postMessage) to amortize overhead.
4. Optionally explore `Atomics.wait` + shared ring buffer for streaming tasks (advanced, document thoroughly if added).
5. Use microtask checkpoints: yield control (`await 0` pattern via `queueMicrotask`) after configurable work units to keep UI fluid when running on main thread (educational demos).

#### Worker Pool Skeleton (Pseudo-code)

```ts
// main thread
pool.runGenomes(weightBuffer, genomeDescriptors, (results) => render(results));

// worker
self.onmessage = ({ data }) => {
  const { weightBuffer, tasks } = data;
  const weights = new Float32Array(weightBuffer); // zero copy
  const results = tasks.map(runFitness);
  self.postMessage(results, []);
};
```

---

### Microbenchmarking & Profiling

Include minimal benchmarks for *changed hot paths* before merging.

- Use high-resolution timers: `performance.now()` in browser, `perf_hooks.performance` in Node.
- Warm up: execute the function ~200–500 times before measuring to stabilize JIT optimizations.
- Measure variance: run multiple iterations; log median not mean (robust against outliers).
- Track allocation rate: optionally wrap code with `const before = performance.memory?.usedJSHeapSize` (when available) for manual inspection.

Keep benchmark scripts deterministic (seed RNG) and exclude I/O.

---

### Determinism & Reproducibility

Provide a deterministic mode for educational reproduction:

- Central seeded PRNG (e.g., Mulberry32) stored as a module-scoped object with methods `nextFloat()`, `nextInt(max)`.
- Avoid `Math.random()` in core code when deterministic mode is enabled.
- Stable iteration: avoid relying on property enumeration order of plain objects — iterate arrays or sorted keys.
- Document any intentional nondeterminism (e.g., tie-breaking random mutations) with a JSDoc `@remarks`.

---

### Error Handling & Assertions

- Use lightweight internal invariant checks (throwing) in debug/dev builds only; guard them behind `if (process.env.NODE_ENV !== 'production')` or a compiled constant to allow dead code elimination.
- Prefer early throws with detailed messages over silent NaN propagation.
- Validate external user inputs at public API boundaries; internal performance-sensitive functions may assume validated invariants.

---

### Serialization & Export (ONNX / Custom)

- Perform shape validation & canonical ordering once before export; reuse cached shape metadata.
- Avoid constructing large intermediate JS objects; stream write directly into typed-array buffers or incremental JSON strings where feasible.
- Reuse encoder instances; avoid repeated `JSON.stringify` of large graphs — serialize directly.

---

### Documentation of Performance-Sensitive Methods

Every performance-sensitive method should add to its JSDoc:

- Complexity: `O(n)` with `n = neurons` (for example).
- Mutability notes: which parameters are read-only, which buffers are reused.
- Determinism notes: whether output depends on random state.
- Reentrancy: explicit statement if using shared scratch.

Example snippet:

```ts
/**
 * Mutate weights in-place.
 * @param weights - Contiguous weight buffer (modified in-place).
 * @param rate - Mutation probability per weight.
 * @remarks Non-reentrant (shared RNG). Uses pooled temp buffer for Gaussian noise. O(W) time.
 */
```

---


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
