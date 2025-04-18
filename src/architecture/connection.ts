/**
 * Represents a connection between two nodes in a neural network.
 */
export default class Connection {
  from: any;
  to: any;
  gain: number;
  weight: number;
  gater: any | null;
  elegibility: number;
  previousDeltaWeight: number;
  totalDeltaWeight: number;
  xtrace: { nodes: any[]; values: number[] };

  /**
   * Creates a new connection between two nodes.
   * @param {any} from - The source node of the connection.
   * @param {any} to - The target node of the connection.
   * @param {number} [weight] - The weight of the connection. Defaults to a random value between -0.1 and 0.1.
   */
  constructor(from: any, to: any, weight?: number) {
    this.from = from;
    this.to = to;
    this.gain = 1;
    this.weight = weight ?? Math.random() * 0.2 - 0.1;
    this.gater = null;
    this.elegibility = 0;

    /** For tracking momentum */
    this.previousDeltaWeight = 0;

    /** Batch training */
    this.totalDeltaWeight = 0;

    this.xtrace = {
      nodes: [],
      values: [],
    };
  }

  /**
   * Converts the connection to a JSON object.
   * @returns {{ weight: number }} A JSON representation of the connection.
   */
  toJSON(): { weight: number } {
    const json = {
      weight: this.weight,
    };

    return json;
  }

  /**
   * Returns an innovation ID for the connection.
   * @param {number} a - The ID of the source node.
   * @param {number} b - The ID of the target node.
   * @returns {number} The innovation ID based on the Cantor pairing function.
   * @see {@link https://en.wikipedia.org/wiki/Pairing_function | Cantor pairing function}
   */
  static innovationID(a: number, b: number): number {
    return (1 / 2) * (a + b) * (a + b + 1) + b;
  }
}
