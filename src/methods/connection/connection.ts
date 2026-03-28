/**
 * Defines the small wiring-policy shelf for connecting one node group to another.
 *
 * Read this file as a topology chooser rather than a bag of connection names.
 * These policies do not decide weights, learning, or mutation pressure; they
 * answer a narrower structural question first: what edge pattern should exist
 * between the source group and the target group before later optimization
 * details matter?
 *
 * The three built-ins answer three different wiring intents:
 *
 * - `ALL_TO_ALL` asks for the densest possible bridge between the groups,
 * - `ALL_TO_ELSE` keeps that dense bridge but avoids trivial self-links when
 *   the source and target are the same group,
 * - `ONE_TO_ONE` preserves positional pairing instead of creating a dense mesh.
 *
 * Those choices matter because they create very different starting biases. A
 * dense bridge maximizes routing freedom, a dense-without-self-links bridge is
 * often the cleanest way to describe intra-group recurrence, and one-to-one
 * wiring preserves explicit alignment instead of encouraging cross-talk.
 *
 * A practical chooser for first experiments:
 *
 * - start with `ALL_TO_ALL` when every source feature should be allowed to
 *   influence every target unit,
 * - use `ALL_TO_ELSE` when you want dense recurrent-style reuse inside one
 *   group without creating direct self-connections,
 * - choose `ONE_TO_ONE` when index alignment matters and each source unit
 *   should feed exactly one partner.
 *
 * ```mermaid
 * flowchart LR
 *   Dense[Dense mesh] --> AllToAll[ALL_TO_ALL]
 *   Dense --> AllToElse[ALL_TO_ELSE]
 *   Paired[Positional pairing] --> OneToOne[ONE_TO_ONE]
 * ```
 *
 * Minimal workflow:
 *
 * ```ts
 * const wiringShelf = {
 *   denseBridge: groupConnection.ALL_TO_ALL,
 *   denseWithoutSelfLoops: groupConnection.ALL_TO_ELSE,
 *   alignedBridge: groupConnection.ONE_TO_ONE,
 * };
 * ```
 */
export const groupConnection = Object.freeze({
  // Renamed export
  /**
   * Connects every source node to every target node.
   *
   * This is the default dense pattern: maximum routing freedom at the cost of
   * more edges, more parameters, and less built-in structural restraint.
   *
   * @example
   * ```ts
   * const denseBridge = groupConnection.ALL_TO_ALL;
   * ```
   */
  ALL_TO_ALL: Object.freeze({
    name: 'ALL_TO_ALL', // Renamed name
  }),

  /**
   * Connects every source node to every target node except direct self-links
   * when source and target refer to the same group.
   *
   * Use this when you want near-dense recurrence or intra-group communication
   * without letting a node connect directly back into itself.
   *
   * @example
   * ```ts
   * const denseWithoutSelfLoops = groupConnection.ALL_TO_ELSE;
   * ```
   */
  ALL_TO_ELSE: Object.freeze({
    name: 'ALL_TO_ELSE', // Renamed name
  }),

  /**
   * Connects each source node to the target node at the same index.
   *
   * This is the file's most structured pattern. It keeps positional alignment
   * intact and requires both groups to have matching size.
   *
   * @example
   * ```ts
   * const alignedBridge = groupConnection.ONE_TO_ONE;
   * ```
   */
  ONE_TO_ONE: Object.freeze({
    name: 'ONE_TO_ONE', // Renamed name
  }),
});

/**
 * Default export for the group-connection policy shelf.
 */
export default groupConnection; // Export renamed object
