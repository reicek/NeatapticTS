/** Supported external seed families for direct NEATchat import. */
export type NeatChatSupportedSeedFamily = 'gru' | 'lstm';

/** Error codes raised by the NEATchat external-seed import boundary. */
export type NeatChatSeedImportErrorCode =
  | 'UNSUPPORTED_OPERATOR'
  | 'DIMENSION_MISMATCH'
  | 'ACTIVATION_INCOMPATIBLE';

/** One recurrent-layer weight bundle exported from a compatible teacher checkpoint. */
export interface NeatChatExternalSeedLayerWeights {
  /** Input-to-gate matrix in PyTorch gate-row order. */
  readonly weightIh: readonly (readonly number[])[];
  /** Hidden-to-gate matrix in PyTorch gate-row order. */
  readonly weightHh: readonly (readonly number[])[];
  /** Input-side gate biases in PyTorch gate-row order. */
  readonly biasIh: readonly number[];
  /** Hidden-side gate biases in PyTorch gate-row order. */
  readonly biasHh: readonly number[];
}

/**
 * JSON descriptor accepted by the NEATchat external-seed direct-import bridge.
 *
 * This shape mirrors the PyTorch `nn.GRU` / `nn.LSTM` `state_dict()` export convention
 * for single-layer models. Weight matrices are stored in gate-row order:
 * - GRU: reset `[0..H-1]`, update `[H..2H-1]`, new `[2H..3H-1]`
 * - LSTM: input `[0..H-1]`, forget `[H..2H-1]`, cell `[2H..3H-1]`, output `[3H..4H-1]`
 *
 * A minimal offline Python export script produces this descriptor from `nn.GRU.state_dict()`
 * by reading `weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0`, and `bias_hh_l0`.
 *
 * @see {@link https://arxiv.org/abs/1406.1078} Cho et al. (2014), Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation — canonical GRU gate formulation.
 * @see {@link https://en.wikipedia.org/wiki/Gated_recurrent_unit} Gated recurrent unit, Wikipedia.
 * @example
 * ```json
 * {
 *   "family": "gru",
 *   "vocabSize": 512,
 *   "hiddenSize": 24,
 *   "layers": [{
 *     "weightIh": [[...]],
 *     "weightHh": [[...]],
 *     "biasIh":   [...],
 *     "biasHh":   [...]
 *   }]
 * }
 * ```
 */
export interface NeatChatExternalSeedDescriptor {
  /** External recurrent family to map into a native NEATchat builder. */
  readonly family: string;
  /** Total one-hot input and output width used by the external model. */
  readonly vocabSize: number;
  /** Hidden-block width in the external recurrent layer. */
  readonly hiddenSize: number;
  /** Recurrent layer state exported in PyTorch gate-row order. */
  readonly layers: readonly NeatChatExternalSeedLayerWeights[];
  /** Optional external readout weights when the teacher exports a linear head. */
  readonly linearWeight?: readonly (readonly number[])[];
  /** Optional external readout biases when the teacher exports a linear head. */
  readonly linearBias?: readonly number[];
}

/**
 * Metadata written into the v2 snapshot `extensions.neatchat.seedMetadata` bag
 * after a direct parameter-vector conversion or supervised distillation pass.
 *
 * When `conversionSource` is `'external-parameter-vector'`, the native network was built
 * by mapping a compatible PyTorch-format checkpoint through the offline conversion bridge.
 * When `conversionSource` is `'distillation'`, a native builder network of the target
 * dimensions was trained to match a teacher checkpoint's token distribution on a
 * representative corpus.
 */
export interface NeatChatSeedMetadata {
  /** Native recurrent family used by the converted seed snapshot. */
  readonly family: NeatChatSupportedSeedFamily;
  /** Total one-hot IO width preserved by the converted snapshot. */
  readonly vocabSize: number;
  /** Hidden-block width preserved by the converted snapshot. */
  readonly hiddenSize: number;
  /** Conversion path used to build the native NEATchat artifact. */
  readonly conversionSource: 'external-parameter-vector' | 'distillation';
}

/** Summary of one completed external-seed import pass. */
export interface NeatChatSeedImportResult {
  /** Import-ready v2 snapshot containing the converted native seed network. */
  readonly snapshot: import('./neatChat.types').NeatChatSessionSnapshotV2;
  /** Metadata mirrored into `snapshot.extensions.neatchat.seedMetadata`. */
  readonly seedMetadata: NeatChatSeedMetadata;
  /** Scalar parameter count written into the converted snapshot. */
  readonly parameterCount: number;
}