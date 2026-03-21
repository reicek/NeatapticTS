/**
 * Defines the small routing shelf that decides where a gater applies control.
 *
 * Gating is one of the lightest structural policies in the library: the graph
 * stays the same, but another neuron or group gets to modulate how strongly a
 * connection participates in the current computation. That makes gating useful
 * when a network needs context-sensitive routing, soft memory behavior, or a
 * way to expose only part of an otherwise valid intermediate result.
 *
 * Read this file as an answer to one placement question: which part of the
 * connection should the gater influence?
 *
 * - `INPUT` modulates the signal as it enters the target,
 * - `OUTPUT` modulates what the target passes onward,
 * - `SELF` modulates the connection strength itself.
 *
 * Those choices matter because they create different control surfaces. Some
 * experiments need a gate that behaves like an evidence filter, some need a
 * gate that behaves like an output valve, and some need the weight itself to
 * become state-dependent instead of fixed.
 *
 * A practical chooser for first experiments:
 *
 * - start with `INPUT` when the main question is how much incoming evidence
 *   should reach the target at all,
 * - use `OUTPUT` when the target should still integrate normally but reveal
 *   only part of its result to the next layer,
 * - choose `SELF` when the connection should act more like a dynamic coupling
 *   whose strength changes with context.
 *
 * ```mermaid
 * flowchart LR
 *   Source[Source neuron] --> Connection[Connection weight]
 *   Connection --> Target[Target neuron]
 *   Gater[Gater]
 *   Gater -. INPUT .-> Target
 *   Gater -. OUTPUT .-> Target
 *   Gater -. SELF .-> Connection
 * ```
 *
 * Minimal workflow:
 *
 * ```ts
 * const routingShelf = {
 *   incomingGate: gating.INPUT,
 *   outgoingGate: gating.OUTPUT,
 *   adaptiveWeightGate: gating.SELF,
 * };
 * ```
 *
 * @see {@link https://en.wikipedia.org/wiki/Artificial_neural_network#Gating_mechanisms}
 */
export const gating = {
  /**
   * Output gating lets the target compute normally and then places the gate on
   * what the rest of the network is allowed to observe.
   *
   * Use this when the target neuron should still integrate its inputs normally,
   * but the network needs a separate decision about how much of that result is
   * allowed to propagate onward.
   *
   * @example
   * ```ts
   * const visibleSignalGate = gating.OUTPUT;
   * ```
   * @property {string} name - Identifier for the output gating method.
   */
  OUTPUT: {
    name: 'OUTPUT',
  },

  /**
   * Input gating places the gate before the target updates, so the gater acts
   * like a filter on incoming evidence.
   *
   * This is the most natural choice when you want the gate to behave like an
   * evidence filter before the target updates its own state.
   *
   * @example
   * ```ts
   * const incomingEvidenceGate = gating.INPUT;
   * ```
   * @property {string} name - Identifier for the input gating method.
   */
  INPUT: {
    name: 'INPUT',
  },

  /**
   * Self gating places the gate on the connection strength itself, turning a
   * fixed weight into a context-sensitive coupling.
   *
   * Use this when the connection should stop behaving like a fixed cable and
   * start behaving like a context-sensitive coupling whose strength changes at
   * runtime.
   *
   * @example
   * ```ts
   * const adaptiveCouplingGate = gating.SELF;
   * ```
   * @property {string} name - Identifier for the self-gating method.
   */
  SELF: {
    name: 'SELF',
  },
};
