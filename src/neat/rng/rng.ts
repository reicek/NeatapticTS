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
 * The key idea is ownership. Sometimes the caller owns randomness by injecting
 * an RNG directly. Sometimes the controller owns randomness by advancing and
 * checkpointing its internal deterministic stream. Replay only makes sense when
 * that ownership boundary stays explicit. Otherwise a restored run can look
 * deterministic on paper while silently drawing from a different source of
 * randomness.
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
 * A second way to read the chapter is as an ownership split:
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   caller[Caller intent]:::accent --> injected[Injected RNG<br/>caller owns randomness]:::base
 *   caller --> internal[Internal xorshift stream<br/>controller owns replay state]:::base
 *   internal --> snapshot[Snapshot export restore]:::base
 *   injected --> diagnostics[Tests or custom experiments]:::base
 *   snapshot --> replay[Deterministic replay path]:::base
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
 * Historically, reproducible evolutionary runs become much easier to debug once
 * randomness can be treated as state instead of mystery. This chapter is the
 * controller-side expression of that shift: not less randomness, but randomness
 * that can be inspected, exported, restored, and explained.
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
