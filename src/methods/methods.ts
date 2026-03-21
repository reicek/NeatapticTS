/**
 * Shared method families for learning, mutation, and structural policy.
 *
 * This folder is the library's reusable policy shelf. The heavier controller
 * chapters in `neat/` decide when to evaluate, mutate, select, or schedule a
 * learning rate change. The `methods/` folder defines the small vocabulary of
 * choices those higher-level chapters reuse.
 *
 * That boundary matters because these exports are intentionally broader than any
 * one subsystem:
 *
 * - `Activation` shapes how nodes transform signals,
 * - `Cost` defines what prediction error means,
 * - `Rate` defines how aggressively learning rates should change over time,
 * - `selection`, `mutation`, and `crossover` define how evolutionary search
 *   applies pressure and creates variation,
 * - `gating` and `groupConnection` describe smaller structural control choices.
 *
 * Read the chapter in three passes:
 *
 * 1. start with `Activation`, `Cost`, and `Rate` when you are thinking like a
 *    trainer tuning signal flow, error shape, and optimization tempo,
 * 2. continue to `selection`, `mutation`, and `crossover` when you are
 *    thinking like an evolutionary controller tuning search pressure,
 * 3. finish with `gating` and `groupConnection` when you need lower-level
 *    structural wiring vocabulary.
 */
/**
 * Shared method families for learning, mutation, and structural policy.
 *
 * This folder is the library's reusable policy shelf. The heavier controller
 * chapters in `neat/` decide when to evaluate, mutate, select, or schedule a
 * learning rate change. The `methods/` folder defines the small vocabulary of
 * choices those higher-level chapters reuse.
 *
 * That boundary matters because these exports are intentionally broader than any
 * one subsystem:
 *
 * - `Activation` shapes how nodes transform signals,
 * - `Cost` defines what prediction error means,
 * - `Rate` defines how aggressively learning rates should change over time,
 * - `selection`, `mutation`, and `crossover` define how evolutionary search
 *   applies pressure and creates variation,
 * - `gating` and `groupConnection` describe smaller structural control choices.
 *
 * Read the chapter in three passes:
 *
 * 1. start with `Activation`, `Cost`, and `Rate` when you are thinking like a
 *    trainer tuning signal flow, error shape, and optimization tempo,
 * 2. continue to `selection`, `mutation`, and `crossover` when you are
 *    thinking like an evolutionary controller tuning search pressure,
 * 3. finish with `gating` and `groupConnection` when you need lower-level
 *    structural wiring vocabulary.
 *
 * ```mermaid
 * flowchart TD
 *   Methods[methods chapter] --> Training[Training and optimization vocabulary]
 *   Methods --> Evolution[Evolutionary search vocabulary]
 *   Methods --> Structure[Structural control vocabulary]
 *   Training --> Activation[Activation]
 *   Training --> Cost[Cost]
 *   Training --> Rate[Rate]
 *   Evolution --> Selection[selection]
 *   Evolution --> Mutation[mutation]
 *   Evolution --> Crossover[crossover]
 *   Structure --> Gating[gating]
 *   Structure --> Connection[groupConnection]
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
