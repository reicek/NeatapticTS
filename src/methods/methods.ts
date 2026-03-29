/**
 * Shared method families for signal shaping, search pressure, and structural policy.
 *
 * This folder is the library's reusable vocabulary shelf. The heavier
 * controller chapters in `neat/` decide when a policy should be applied. The
 * `methods/` folder defines what those policy choices actually are. That split
 * keeps the rest of the repo readable: examples, architecture helpers, and
 * evolutionary controllers can reuse the same small method objects without each
 * subsystem inventing its own private dialect.
 *
 * The quickest way to understand the chapter is to split it into three reader
 * questions. How should a unit transform signal? That is `Activation`. How
 * should error or learning tempo be interpreted? That is `Cost` and `Rate`.
 * How should evolutionary pressure and wiring structure be adjusted? That is
 * `selection`, `mutation`, `crossover`, `gating`, and `groupConnection`.
 *
 * That boundary matters because it lets you change policy without changing
 * orchestration. A `Neat` controller can switch from gentle to aggressive
 * selection, or an architecture builder can swap activation families, without
 * rewriting evaluation loops or graph code. The method objects stay small on
 * purpose so experiments can compose them instead of hiding them in conditionals.
 *
 * The structural pair deserves special attention because the names sound close
 * while the responsibilities are different. `groupConnection` answers how two
 * groups should be wired before any runtime signal exists. `gating` answers how
 * an already-existing connection should be modulated once the network is
 * running. One is about topology layout. The other is about runtime control.
 *
 * Two compact background bridges help here. See Wikipedia contributors,
 * [Activation function](https://en.wikipedia.org/wiki/Activation_function), for
 * the signal-shaping side of the shelf, and Wikipedia contributors,
 * [Selection (genetic algorithm)](https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)),
 * for the search-pressure side. Together they frame the two big forces this
 * folder keeps in play: how nodes respond to signal and how search decides
 * which traits survive.
 *
 * Read the chapter in three passes:
 *
 * 1. start with `Activation`, `Cost`, and `Rate` when you are thinking like a
 *    trainer tuning signal flow, error shape, and optimization tempo,
 * 2. continue to `selection`, `mutation`, and `crossover` when you are
 *    thinking like an evolutionary controller tuning search pressure,
 * 3. finish with `gating` and `groupConnection` when you need lower-level
 *    structural vocabulary and want to distinguish routing control from raw
 *    wiring layout.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Methods[methods chapter]:::accent --> Training[Training and optimization vocabulary]:::base
 *   Methods --> Evolution[Evolutionary search vocabulary]:::base
 *   Methods --> Structure[Structural control vocabulary]:::base
 *   Training --> Activation[Activation]:::base
 *   Training --> Cost[Cost]:::base
 *   Training --> Rate[Rate]:::base
 *   Evolution --> Selection[selection]:::base
 *   Evolution --> Mutation[mutation]:::base
 *   Evolution --> Crossover[crossover]:::base
 *   Structure --> Gating[gating]:::base
 *   Structure --> Connection[groupConnection]:::base
 * ```
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Signals["How should signal behave?"]:::accent --> SignalShelf["Activation + Cost + Rate"]:::base
 *   Search["How should search behave?"]:::accent --> SearchShelf["selection + mutation + crossover"]:::base
 *   Structure["How should wiring be controlled?"]:::accent --> StructureShelf["gating + groupConnection"]:::base
 * ```
 *
 * Example: assemble one compact training vocabulary for signal shape, loss, and
 * tempo.
 *
 * ```ts
 * const trainingPolicy = {
 *   activation: Activation.relu,
 *   loss: Cost.mse,
 *   schedule: Rate.step(0.9, 100),
 * };
 * ```
 *
 * Example: assemble a stronger search-pressure and routing vocabulary without
 * changing the surrounding controller code.
 *
 * ```ts
 * const evolutionaryPolicy = {
 *   parentSelection: { ...selection.POWER, power: 6 },
 *   gatePlacement: gating.SELF,
 *   denseBridge: groupConnection.ALL_TO_ALL,
 * };
 * ```
 */
export { default as Cost } from './cost/cost';
export { default as Rate } from './rate/rate';
export { default as Activation } from './activation/activation';
export { gating } from './gating/gating';
export { mutation } from './mutation/mutation';
export { selection } from './selection/selection';
export { crossover } from './crossover/crossover';
export { default as groupConnection } from './connection/connection';
