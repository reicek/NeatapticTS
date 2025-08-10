/**
 * Represents a connection between two nodes in a neural network.
 *
 * Connections transfer activation values from one node to another, with an associated weight
 * that determines the strength of the connection. Connections can also be gated by other nodes.
 */
import Node from './node'; // Import Node type

export default class Connection {
  from: Node; // The source node of the connection
  to: Node; // The target node of the connection
  gain: number; // Gain applied to the connection
  weight: number; // Weight of the connection
  gater: Node | null; // Node that gates this connection, if any
  eligibility: number; // Eligibility trace for backpropagation
  previousDeltaWeight: number; // Previous weight change for momentum
  totalDeltaWeight: number; // Accumulated weight change for batch training
  xtrace: { nodes: Node[]; values: number[] }; // Extended trace for eligibility propagation
  // --- Optimizer moment states ---
  opt_m?: number; // First moment (Adam)
  opt_v?: number; // Second moment (Adam)
  opt_cache?: number; // Accumulator (RMSProp/Adagrad)
  // Additional optimizer states
  opt_vhat?: number; // AMSGrad max second moment
  opt_u?: number; // Adamax infinity norm
  opt_m2?: number; // Lion second momentum like term
  _la_shadowWeight?: number; // Lookahead shadow param
  dcMask?: number; // DropConnect mask (1 active, 0 dropped)

  /**
   * Creates a new connection between two nodes.
   *
   * @param {Node} from - The source node of the connection.
   * @param {Node} to - The target node of the connection.
   * @param {number} [weight] - The weight of the connection. Defaults to a random value between -0.1 and 0.1.
   */
  constructor(from: Node, to: Node, weight?: number) {
    this.from = from;
    this.to = to;
    this.gain = 1;
    this.weight = weight ?? Math.random() * 0.2 - 0.1;
    this.gater = null;
    this.eligibility = 0;

    // For tracking momentum
    this.previousDeltaWeight = 0;

    // Batch training
    this.totalDeltaWeight = 0;

    this.xtrace = {
      nodes: [],
      values: [],
    };

    // Initialize optimizer moments
    this.opt_m = 0;
    this.opt_v = 0;
    this.opt_cache = 0;
    this.opt_vhat = 0;
    this.opt_u = 0;
    this.opt_m2 = 0;
    // Initialize dropconnect mask
    this.dcMask = 1;
  }

  /**
   * Converts the connection to a JSON object for serialization.
   *
   * @returns {{ from: number | undefined, to: number | undefined, weight: number, gain: number, gater: number | null }} A JSON representation of the connection.
   */
  toJSON() {
    const json: any = {
      from: this.from.index ?? undefined,
      to: this.to.index ?? undefined,
      weight: this.weight,
      gain: this.gain
    };
    if (this.gater && typeof this.gater.index !== 'undefined') {
      json.gater = this.gater.index;
    }
    return json;
  }

  /**
   * Generates a unique innovation ID for the connection.
   *
   * The innovation ID is calculated using the Cantor pairing function, which maps two integers
   * (representing the source and target nodes) to a unique integer.
   *
   * @param {number} a - The ID of the source node.
   * @param {number} b - The ID of the target node.
   * @returns {number} The innovation ID based on the Cantor pairing function.
   * @see {@link https://en.wikipedia.org/wiki/Pairing_function Cantor pairing function}
   */
  static innovationID(a: number, b: number): number {
    return (1 / 2) * (a + b) * (a + b + 1) + b;
  }
}
