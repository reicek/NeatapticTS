/**
 * Stable public entry point for the `Layer` model-stage primitive.
 *
 * A `Layer` packages a set of nodes and their wiring into a recognizable model
 * stage with standard entrypoints: `activate()`, `propagate()`, `connect()`,
 * and named factory constructors (`Layer.Dense(n)`, `Layer.LSTM(n)`, etc.).
 *
 * If `Node` is the single-neuron chapter and `Network` is the whole-graph
 * chapter, `Layer` is the middle shelf: dense stages, recurrent cells, and
 * normalization passes expressed as self-contained, reusable blocks.
 *
 * The full implementation lives in `layer/layer.ts`. This root file is a
 * stable facade.
 *
 * @example
 * ```ts
 * import Layer from './layer';
 *
 * const hidden = Layer.Dense(16);
 * const output = Layer.Dense(2, 'output');
 * hidden.connect(output);
 * ```
 */
export { default } from './layer/layer';
