---
name: implementation-standards
description: 'Owns durable code standards for NeatapticTS: ES2023-first syntax, folder-based module architecture, JSDoc requirements, naming conventions, cognitive complexity rules, and validation gates. Use when implementing, reviewing, or refactoring src/ code to ensure consistency with repo patterns and quality gates.'
argument-hint: 'Provide the target surface (src/ file or folder), the type of work (implement, refactor, review), and any validation constraints such as build, quality:folder, or coverage requirements.'
user-invocable: false
disable-model-invocation: false
---

# Implementation Standards Playbook

Use this skill to apply and enforce the durable code standards for NeatapticTS.
These standards define how production code in `src/`, tests in `testing/`, and
examples in `examples/` should be written, structured, and validated.

This skill owns the canonical interpretation of:

- ES2023-first syntax policies,
- folder-based module architecture patterns,
- JSDoc requirements for exported symbols,
- naming conventions and identifier rules,
- cognitive complexity and helper structure guidelines,
- validation gates and quality checks.

When a companion agent or numbered orchestrator performs implementation work,
this skill defines the quality bar that the output must meet.

## Core Promise

`implementation-standards` exists to keep the codebase consistent, readable,
and maintainable as it grows. The standards are not arbitrary rules; they are
distilled patterns that make the repo easier to:

- navigate for first-time contributors,
- reason about during refactors,
- test to 100% coverage,
- document with generated READMEs,
- evolve without accumulating technical debt.

## When to Use

Invoke `implementation-standards` when:

- implementing new functionality in `src/` and need to follow repo patterns,
- refactoring an existing module and want to ensure architectural consistency,
- reviewing code and need to check against canonical standards,
- onboarding and need to understand the repo's code style and quality gates,
- a code change needs validation with `npm run quality:folder` or similar gates.

Do **not** use this skill for:

- documentation-only changes (use `educational-docs`),
- test-only changes without production code (use `coverage-tranche` or `test-fix-workflow`),
- planning or roadmap alignment (use `plan-alignment`).

## ES2023-First Policy

This repo uses modern JavaScript/TypeScript features to improve safety,
readability, and correctness. Prefer these constructs:

### Immutable Array Methods

```ts
// ✅ Prefer
const sorted = arr.toSorted((a, b) => a - b);
const reversed = arr.toReversed();
const last = arr.at(-1);

// ❌ Avoid
const sorted = arr.sort((a, b) => a - b); // mutates original
const reversed = arr.reverse(); // mutates original
const last = arr[arr.length - 1]; // index math
```

### Structured Cloning

```ts
// ✅ Prefer
const clone = structuredClone(obj);

// ❌ Avoid
const clone = JSON.parse(JSON.stringify(obj)); // loses types, dates, undefined
const clone = Object.assign({}, obj); // shallow, misses nested
```

### Nullish Coalescing and Optional Chaining

```ts
// ✅ Prefer
const value = config?.timeout ?? 5000;
const nested = data?.items?.at(0)?.name;

// ❌ Avoid
const value = config && config.timeout ? config.timeout : 5000;
const nested = data && data.items && data.items[0] ? data.items[0].name : undefined;
```

### Numeric Separators

```ts
// ✅ Prefer
const timeout = 30_000;
const bytes = 1_048_576;

// ❌ Avoid
const timeout = 30000;
const bytes = 1048576;
```

### ES Modules

```ts
// ✅ Prefer
import { Network } from './network.js';
export { Neat } from './neat.js';

// ❌ Avoid
const { Network } = require('./network'); // CommonJS
module.exports = { Neat }; // CommonJS
```

### Error with Cause

```ts
// ✅ Prefer
throw new Error('Failed to load', { cause: originalError });

// ❌ Avoid
throw new Error('Failed to load'); // loses context
```

## Module Architecture

Non-trivial modules use a folder-based layout that separates concerns and
makes the public surface legible.

### Standard Layout

For a module `foo` inside parent `bar`:

```
bar/foo/
  bar.foo.ts           ← orchestration (public surface, exports)
  bar.foo.utils.ts     ← helper functions
  bar.foo.types.ts     ← interfaces, types, result objects
  bar.foo.errors.ts    ← error classes
  bar.foo.constants.ts ← named constants
```

### Sub-module Pattern

Sub-modules follow the same naming convention:

```
bar/foo/sub/
  bar.foo.sub.ts       ← orchestration for sub-module
  bar.foo.sub.types.ts ← sub-module types
```

### Orchestration-First

The main `.ts` file (orchestration) should:

- export the public API surface,
- define top-level functions as declarative steps,
- call small single-responsibility helpers defined below the fold,
- avoid embedding complex logic inline.

```ts
// ✅ Prefer
export function process(data: Data): Result {
  const validated = validateInput(data);
  const transformed = transform(validated);
  return fold(transformed);
}

function validateInput(data: Data): ValidatedData { /* ... */ }
function transform(data: ValidatedData): Transformed { /* ... */ }
function fold(data: Transformed): Result { /* ... */ }
```

## JSDoc Requirements

All exported symbols must have JSDoc comments that explain **what**, **why**,
and **when** — not just restate the signature.

### Required Tags

```ts
/**
 * Brief one-line summary of what this does.
 *
 * Longer explanation of why this exists, what problem it solves,
 * and any important tradeoffs or invariants.
 *
 * @param name - Description of parameter including valid range or constraints
 * @param options - Configuration options with defaults explained
 * @returns Description of return value and what it represents
 * @throws ErrorType when condition X occurs
 *
 * @example
 * ```ts
 * const result = myFunction(input, { option: true });
 * console.log(result.value);
 * ```
 */
export function myFunction(name: string, options?: Options): Result {
  // ...
}
```

### Constants and Types

```ts
/**
 * Maximum number of generations before forced termination.
 * Used when no fitness threshold is configured.
 */
export const MAX_GENERATIONS = 1000;

/**
 * Result of a mutation operation.
 * @property success - true if mutation was applied
 * @property genome - the modified genome (same reference as input)
 * @property reason - failure reason if success is false
 */
export interface MutationResult {
  success: boolean;
  genome: Genome;
  reason?: string;
}
```

## Code Style Rules

### Identifier Naming

```ts
// ✅ Prefer
for (let i = 0; i < items.length; i++) { /* trivial loop */ }
for (let index = 0; index < items.length; index++) { /* complex body */ }
const userInput = getConfig();
const connectionCount = connections.length;

// ❌ Avoid
const x = getConfig(); // what is x?
const cnt = connections.length; // abbreviation without context
for (let i = 0; i < items.length; i++) { /* 50-line body */ } // too long for short name
```

### Single-Expect Rule

Each `it()` block must have exactly one top-level `expect()`:

```ts
// ✅ Prefer
it('returns sorted array', () => {
  const result = toSorted(input);
  expect(result).toEqual(expected);
});

// ❌ Avoid
it('returns sorted array', () => {
  const result = toSorted(input);
  expect(result).toEqual(expected);
  expect(result).not.toBe(input);
  expect(result.length).toBe(input.length);
});
```

### Named Constants Over Magic Numbers

```ts
// ✅ Prefer
const DEFAULT_MUTATION_RATE = 0.03;
if (rate > DEFAULT_MUTATION_RATE) { /* ... */ }

// ❌ Avoid
if (rate > 0.03) { /* ... */ }
```

### Step-Level Inline Comments

```ts
// ✅ Prefer
export function evolve(population: Population): Population {
  // Step 1: Evaluate fitness for all genomes
  const evaluated = evaluateAll(population);
  
  // Step 2: Select parents based on fitness
  const parents = selectParents(evaluated);
  
  // Step 3: Apply crossover and mutation
  const offspring = reproduce(parents);
  
  // Step 4: Form next generation with elitism
  return formNextGeneration(population, offspring);
}
```

### Fixed Mappings: Single Table or Enum

```ts
// ✅ Prefer
export const ACTIVATION_FUNCTIONS = {
  sigmoid: sigmoidFn,
  tanh: tanhFn,
  relu: reluFn,
} as const;

// ❌ Avoid
if (name === 'sigmoid') { return sigmoidFn; }
else if (name === 'tanh') { return tanhFn; }
else if (name === 'relu') { return reluFn; }
```

### Avoid `any` and `unknown`

```ts
// ✅ Prefer
function process(data: unknown): asserts data is ValidData {
  if (!isValid(data)) {
    throw new Error('Invalid data');
  }
}

// With eslint-disable when truly necessary
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function legacyInterop(value: any): LegacyResult { /* ... */ }

// ❌ Avoid
function process(data: any): any { /* ... */ } // no justification
```

## Cognitive Complexity Guidelines

### Helper Structure Order

Order local helpers as: locals → calls → return → helpers at end.

```ts
export function complexOperation(input: Input): Result {
  // 1. Local variables and constants
  const threshold = getThreshold();
  const cache = new Map();
  
  // 2. Main logic as calls to small helpers
  const validated = validate(input, threshold);
  const transformed = transform(validated, cache);
  
  // 3. Return
  return finalize(transformed);
  
  // 4. Helpers below the fold
  function validate(inp: Input, thresh: number): Validated { /* ... */ }
  function transform(v: Validated, c: Map): Transformed { /* ... */ }
  function finalize(t: Transformed): Result { /* ... */ }
}
```

### Declarative Flow

Prefer `collect → transform → fold` over nested control flow:

```ts
// ✅ Prefer
export function analyze(genomes: Genome[]): Analysis {
  const valid = genomes.filter(isValid);
  const scores = valid.map(computeScore);
  return scores.reduce(aggregate, emptyAnalysis);
}

// ❌ Avoid
export function analyze(genomes: Genome[]): Analysis {
  const result = emptyAnalysis;
  for (let i = 0; i < genomes.length; i++) {
    if (isValid(genomes[i])) {
      const score = computeScore(genomes[i]);
      for (const key of Object.keys(score)) {
        if (!result[key]) {
          result[key] = 0;
        }
        result[key] += score[key];
      }
    }
  }
  return result;
}
```

### Multi-Pass Decomposition

When logic is complex:

1. Stabilize seams: identify natural boundaries in the logic.
2. Extract helpers: pull each seam into a named function.
3. Typed context: give each helper explicit input/output types.
4. Orchestration top level: keep the main function as a declarative pipeline.

## Validation Checklist

After implementation work, validate with these gates:

### Type Checking

```bash
npx tsc --noEmit -p tsconfig.json
npx tsc --noEmit -p tsconfig.test.json
```

### Folder Quality Gate

```bash
npm run quality:folder -- --folder=src/touched/folder
```

### Build Verification

```bash
npm run build
```

### Dependency Installation

After manifest or lockfile changes:

```bash
npm ci
```

### Test File Review

Flag any test files with multiple top-level `expect()` per `it()`:

```bash
# Manual review or add to CI lint
grep -r "expect(" testing/**/*.test.ts | grep -B5 "it("
```

### JSDoc Verification

Ensure all new exported symbols have JSDoc:

```bash
# Use docs-academic-citation-audit or educational-docs for verification
npm run docs
```

### CI Compatibility

Validate for Linux/Chromium runners:

- Do not assume local Windows success is sufficient.
- Account for Chromium sandbox restrictions in CI.
- Run `npm run docs` for docs/tooling changes.

### Legacy Pattern Audit

List any flagged legacy patterns and intended replacements:

```ts
// In code review or PR description:
// - arr.sort() → arr.toSorted()
// - JSON.parse(JSON.stringify()) → structuredClone()
// - index math → .at(-1)
```

## Coordination with Other Skills

| Skill | Handoff Condition |
|-------|-------------------|
| `educational-docs` | When JSDoc improvements or README generation is needed |
| `coverage-guard` | After any `src/` change to verify 100% coverage |
| `coverage-tranche` | When expanding coverage on passing code |
| `test-fix-workflow` | When tests fail during implementation |
| `solid-split` | When module boundaries need refactoring |
| `docs-academic-citation-audit` | When algorithms need academic citations |

## Guardrails

- Do not accept `any` or `unknown` types without explicit justification.
- Do not write tests with multiple top-level `expect()` calls.
- Do not use in-place array mutation methods when immutable alternatives exist.
- Do not edit generated `src/**/README.md` files directly; improve JSDoc and run `npm run docs`.
- Do not skip `npm run quality:folder` after changes to `src/`.
- Do not leave magic numbers in code; extract to named constants.
- Do not write nested control flow when declarative pipelines are possible.
- Do not create new test files when owner-local test files exist.

## Expected Final Output

A strong implementation pass should report:

- which files were changed and why,
- which ES2023 features were applied,
- how the module architecture follows the folder pattern,
- confirmation that all exports have JSDoc with examples,
- validation results (build, quality:folder, type-check),
- any legacy patterns flagged for future replacement,
- coverage verification status via `coverage-guard`.
