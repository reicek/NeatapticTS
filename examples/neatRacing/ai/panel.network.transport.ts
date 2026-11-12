/**
 * The panel network transport codec for the Neat Racing AI-mode network
 * panel (Step 24C, slice P4-S2C-transport).
 *
 * Two pure laws over the sim worker's `panel-network` reply envelope:
 *
 * 1. The idempotent restore: `restoreNeatRacingPanelNetworkFromReply`
 *    validates the envelope's self-describing shape (the worker's
 *    `NEATRACING_PANEL_NETWORK_ENVELOPE_KIND` discriminator, an integer
 *    rival car index, positive channel/output counts and the EXISTING
 *    serialized-network hex law) and rebuilds a REAL `Network` whose
 *    positional connection weights byte-match the encoded spawn mind.
 *    The rebuild is IDEMPOTENT: the same envelope always rebuilds a
 *    structurally identical network (the constructor is a deterministic
 *    input-to-output layer build — same sizes, same connection order), so
 *    the weights ride positional assignment over
 *    `network.connections[i].weight`.
 *
 * 2. The carrier match: `resolveNeatRacingPanelFeedNetwork` hands the
 *    elected panel candidate ONLY a resolution whose car index equals the
 *    elected candidate's agent id; a pending (missing) resolution or a
 *    foreign car's resolution feeds the candidate's OWN network field (the
 *    F3 network-less-winner fallback law — the idle path stays).
 *
 * The hex law is the established serialized-network envelope law
 * (promotion.rules.ts / rival.propagation.ts: lowercase hex, two
 * characters per byte, no separators, native byte order, one or more whole
 * float32 values) mirrored locally with the one-serialization-path
 * per-payload precedent. Same-runtime byte parity only, never
 * cross-environment exact.
 */
import Network from '../../../src/architecture/network';
import type { NeatRacingNetworkPanelCandidate } from '../visualizer/network-panel.adapter';
import { NEATRACING_PANEL_NETWORK_ENVELOPE_KIND } from '../workers/simulation-worker/simulation-worker';
import type { NeatRacingSimWorkerPanelNetworkReplyPayload } from '../workers/simulation-worker/simulation-worker';

/**
 * The error voice of the reply guards (the named-rejection law).
 *
 * Educational note — WHY A NAMED PREFIX: a corrupted envelope is a LOUD
 * rejection, never a silent fallback. The prefix names the lane so a
 * consumer can tell a transport rejection from any other failure without
 * matching on the message body.
 */
const PANEL_NETWORK_REPLY_ERROR_PREFIX =
  'NeatRacingPanelNetworkReplyRejectedError';

/**
 * The byte-exact hex law of the serialized weights (the mirrored
 * promotion-lane envelope encoding: lowercase, two characters per byte, no
 * separators).
 */
const PANEL_NETWORK_HEX_WEIGHTS_PATTERN = /^[0-9a-f]+$/;

/** The hex length of ONE whole float32 value (four bytes, two per byte). */
const PANEL_NETWORK_HEX_F32_LENGTH = 8;

/** The restored elected-winner resolution bound by its carrier car index. */
export interface NeatRacingPanelNetworkResolution {
  /** The rival car index the resolution was restored for. */
  carIndex: number;
  /** The REAL network rebuilt from the reply's spawn-mind weights. */
  network: Network;
}

/** The guard view of the validated envelope (the payload's own shape). */
type PanelNetworkReplyGuardView = NeatRacingSimWorkerPanelNetworkReplyPayload;

/**
 * Decodes a byte-exact hex string back into its `Float32Array`.
 *
 * The caller's hex law guard (lowercase, non-empty, whole float32 values)
 * rejects every malformed string BEFORE the decode, so the decode itself
 * stays total on guarded input.
 *
 * @param hex - The guarded hex string (lowercase, a multiple of eight
 *   characters).
 * @returns The decoded weights (native byte order).
 */
function decodeHexAsFloat32Array(hex: string): Float32Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let byteIndex = 0; byteIndex < bytes.length; byteIndex += 1) {
    bytes[byteIndex] = parseInt(
      hex.slice(byteIndex * 2, byteIndex * 2 + 2),
      16,
    );
  }
  return new Float32Array(bytes.buffer);
}

/**
 * Restores the sim worker's panel-network reply envelope into a REAL
 * network (P4-S2C, Step 24C — the docked-right panel network transport).
 *
 * Educational note — WHY THE REBUILD IS IDEMPOTENT: the rebuilt network's
 * STRUCTURE comes from the envelope's own dimensions (the channel count and
 * the output count), never from a random seed, so the same envelope always
 * yields the same connection count and the same connection order; the
 * weights then ride positional assignment over `network.connections[i]
 * .weight`. The rebuild therefore reproduces the worker's serialized mind
 * byte-for-byte — same-runtime parity, never cross-environment exact.
 *
 * @param payload - The reply envelope (the worker's `panel-network`
 *   payload); every malformed shape is a loud rejection.
 * @returns The restored resolution (the carrier car index and the rebuilt
 *   network).
 * @throws {RangeError} When the envelope does not carry the panel-network
 *   kind discriminator, a rival car index, positive channel/output counts,
 *   hex weights of one or more whole float32 values, or when the decoded
 *   weights do not match the rebuilt network's connection count.
 *
 * @example
 * ```ts
 * const resolution = restoreNeatRacingPanelNetworkFromReply(reply.payload);
 * console.log(resolution.carIndex, resolution.network.connections.length);
 * ```
 */
export function restoreNeatRacingPanelNetworkFromReply(
  payload: unknown,
): NeatRacingPanelNetworkResolution {
  // Step 1: The envelope must be an object (the transport's own boundary).
  if (payload === null || typeof payload !== 'object') {
    throw new RangeError(
      `${PANEL_NETWORK_REPLY_ERROR_PREFIX}: the reply envelope must be an object`,
    );
  }
  const envelope = payload as Partial<PanelNetworkReplyGuardView>;

  // Step 2: The self-describing kind discriminator (the envelope law).
  if (envelope.kind !== NEATRACING_PANEL_NETWORK_ENVELOPE_KIND) {
    throw new RangeError(
      `${PANEL_NETWORK_REPLY_ERROR_PREFIX}: the reply envelope must carry the panel-network kind discriminator`,
    );
  }

  // Step 3: The carrier car index (a rival slot; car 0 is never carried).
  const carIndex = envelope.carIndex;
  if (
    typeof carIndex !== 'number' ||
    !Number.isInteger(carIndex) ||
    carIndex < 1
  ) {
    throw new RangeError(
      `${PANEL_NETWORK_REPLY_ERROR_PREFIX}: the reply envelope must carry a rival car index`,
    );
  }

  // Step 4: The network's input width (the rung's observation channels).
  const channelCount = envelope.channelCount;
  if (
    typeof channelCount !== 'number' ||
    !Number.isInteger(channelCount) ||
    channelCount < 1
  ) {
    throw new RangeError(
      `${PANEL_NETWORK_REPLY_ERROR_PREFIX}: the reply envelope must carry a positive channel count`,
    );
  }

  // Step 5: The network's output width (the controller's control axes).
  const outputCount = envelope.outputCount;
  if (
    typeof outputCount !== 'number' ||
    !Number.isInteger(outputCount) ||
    outputCount < 1
  ) {
    throw new RangeError(
      `${PANEL_NETWORK_REPLY_ERROR_PREFIX}: the reply envelope must carry a positive output count`,
    );
  }

  // Step 6: The serialized weights (the mirrored hex law: lowercase, no
  //         separators, whole float32 values only).
  const weights = envelope.weights;
  if (
    typeof weights !== 'string' ||
    weights.length === 0 ||
    weights.length % PANEL_NETWORK_HEX_F32_LENGTH !== 0 ||
    !PANEL_NETWORK_HEX_WEIGHTS_PATTERN.test(weights)
  ) {
    throw new RangeError(
      `${PANEL_NETWORK_REPLY_ERROR_PREFIX}: the reply envelope must carry the serialized-network hex weights of one or more float32 values`,
    );
  }

  // Step 7: Rebuild the deterministic structure (the envelope's own
  //         dimensions — never a random seed).
  const network = new Network(channelCount, outputCount);
  const decodedWeights = decodeHexAsFloat32Array(weights);

  // Step 8: The decoded plane must cover the rebuilt network exactly.
  if (decodedWeights.length !== network.connections.length) {
    throw new RangeError(
      `${PANEL_NETWORK_REPLY_ERROR_PREFIX}: the rebuilt network's connection count must match the decoded weights`,
    );
  }

  // Step 9: The weights ride positional assignment (the ledger law).
  for (
    let weightIndex = 0;
    weightIndex < network.connections.length;
    weightIndex += 1
  ) {
    network.connections[weightIndex].weight = decodedWeights[weightIndex];
  }

  return { carIndex, network };
}

/**
 * Resolves the network the elected panel candidate is fed with (P4-S2C,
 * Step 24C — the docked-right panel network transport).
 *
 * Educational note — THE CARRIER MATCH: only a resolution whose car index
 * equals the elected candidate's agent id feeds that candidate. A pending
 * resolution (undefined) and a foreign car's resolution both fall back to
 * the candidate's OWN network field — the F3 law: a network-less winner is
 * never replaced by a networked runner-up (AC-034).
 *
 * @param elected - The elected panel candidate (the most-forward rival).
 * @param resolution - The restored resolution (undefined while the
 *   transport is pending).
 * @returns The network the candidate is fed with, or undefined when the
 *   candidate owns none.
 *
 * @example
 * ```ts
 * const feed = resolveNeatRacingPanelFeedNetwork(winner, resolution);
 * ```
 */
export function resolveNeatRacingPanelFeedNetwork(
  elected: NeatRacingNetworkPanelCandidate,
  resolution: NeatRacingPanelNetworkResolution | undefined,
): Network | undefined {
  if (resolution !== undefined && resolution.carIndex === elected.agentId) {
    return resolution.network;
  }
  return elected.network;
}
