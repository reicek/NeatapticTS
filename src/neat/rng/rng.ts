/**
 * Deterministic RNG support for the NEAT controller.
 *
 * This chapter answers the controller-facing randomness question: how does a
 * NEAT run stay reproducible when mutation, selection, crossover, and other
 * probabilistic choices all depend on a live random stream? The answer is not
 * "freeze randomness" but "make randomness inspectable and restorable." The
 * exported root surface exists so callers can create a deterministic stream,
 * snapshot its current state, restore that state later, and sample it during
 * tests or diagnostics without coupling every caller to the full `Neat`
 * runtime.
 *
 * Read this chapter as a replay-oriented map. The root README should help a
 * reader answer three practical questions quickly:
 *
 * 1. where the RNG comes from,
 * 2. how to capture and restore it,
 * 3. which constants and helper layers define the deterministic contract.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   seed[Option seed or restored state]:::base --> create[Create or reuse RNG]:::accent
 *   create --> stream[Random stream for controller decisions]:::base
 *   stream --> snapshot[Snapshot or export state]:::base
 *   snapshot --> restore[Restore later for replay]:::base
 *   stream --> diagnostics[Test and diagnostics sampling]:::base
 * ```
 *
 * The root RNG entrypoint stays small on purpose because the educational story
 * has two focused layers underneath it.
 *
 * - `core/` explains the xorshift-based random stream and seed lifecycle.
 * - `facade/` explains the stable `Neat` methods used by tests, replay, and
 *   diagnostics.
 *
 * Practical reading order:
 *
 * 1. Start with `getOrCreateRng()` to see how the controller resolves its live
 *    random stream.
 * 2. Read `snapshotRngState()`, `exportRngState()`, and `restoreRngState()` as
 *    the replay boundary.
 * 3. Read `RngHost` to understand the intentionally small state contract.
 * 4. Use the exported constants when you want to understand the fixed xorshift
 *    and seed-guarding choices rather than just treat them as opaque numbers.
 * 5. Continue into `facade/` when you want the stable `Neat` wrapper methods.
 *
 * @example
 * ```ts
 * const rng = getOrCreateRng(neat);
 * const beforeMutation = exportRngState(neat);
 * const sample = rng();
 *
 * restoreRngState(neat, beforeMutation);
 * const replayedSample = getOrCreateRng(neat)();
 * ```
 */
export * from './core/rng.constants';
export * from './core/rng.types';
export * from './core/rng.utils';
