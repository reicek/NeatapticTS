/**
 * Stable public entry point for the `Architect` preset builder.
 *
 * `Architect` is the point where raw graph primitives (Node, Group, Layer)
 * stop being building blocks and become named neural network recipes.
 * It assembles those primitives into complete graphs and normalizes the
 * final `Network` so callers can activate, train, serialize, or evolve
 * the result immediately.
 *
 * Choose a builder based on the kind of memory your task needs:
 *
 * - `Architect.perceptron(...sizes)` — dense feed-forward baseline.
 * - `Architect.randomSparse(i, h, o, opts)` — sparse random topology for
 *   evolution-friendly starting graphs.
 * - `Architect.narx(i, h, o, inputDelay, outputDelay)` — short explicit
 *   time-window with past input and output delay lines.
 * - `Architect.gru(i, ...hidden, o)` — gated recurrent units with learned
 *   state inside each cell.
 * - `Architect.lstm(i, ...hidden, o)` — long short-term memory cells with
 *   separate forget, input, and output gates.
 *
 * The full implementation lives in `architect/architect.ts`. This root file
 * is a stable facade that keeps the architecture import path short.
 *
 * @example
 * ```ts
 * import Architect from './architect';
 *
 * // Dense feed-forward: 2 inputs → 4 hidden → 1 output
 * const mlp = Architect.perceptron(2, 4, 1);
 * mlp.activate([0, 1]); // => [number]
 *
 * // LSTM for sequence modeling
 * const rnn = Architect.lstm(1, 8, 1);
 * rnn.activate([0.5]); // carries recurrent state between calls
 * ```
 */
export { default } from './architect/architect';
