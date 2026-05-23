import { NeatChatError } from './neatChat.errors';
import type { NeatChatSeedImportErrorCode } from './neatChat.seed-import.types';

/**
 * Thrown when an external recurrent seed falls outside the direct-import contract.
 *
 * Three distinct failure codes cover the error space:
 * - `'UNSUPPORTED_OPERATOR'`: the descriptor uses a family, layer count, or operator
 *   that the direct-import bridge cannot map mechanically (e.g. bidirectional, attention,
 *   multi-layer stacked RNN). When this code is raised, `distillationSuggestion` is always
 *   populated with guidance directing callers to the supervised-distillation path.
 * - `'DIMENSION_MISMATCH'`: vocabulary size or hidden size falls outside the supported range,
 *   a weight matrix shape does not match the descriptor header, or a non-finite value appears.
 * - `'ACTIVATION_INCOMPATIBLE'`: the descriptor activation family is not sigmoid or tanh and
 *   cannot be mapped to the native gate topology.
 *
 * @example
 * ```ts
 * import { validateNeatChatSeedFamily, NeatChatSeedImportError } from './index';
 *
 * try {
 *   validateNeatChatSeedFamily({
 *     family: 'bidirectional-gru',
 *     vocabSize: 512,
 *     hiddenSize: 24,
 *     layers: [],
 *   });
 * } catch (err) {
 *   if (err instanceof NeatChatSeedImportError) {
 *     console.log(err.code);                  // 'UNSUPPORTED_OPERATOR'
 *     console.log(err.distillationSuggestion); // distillation guidance string
 *   }
 * }
 * ```
 */
export class NeatChatSeedImportError extends NeatChatError {
  /** Stable machine-readable import failure code. */
  public readonly code: NeatChatSeedImportErrorCode;

  /** Optional guidance for callers that should use supervised distillation instead. */
  public readonly distillationSuggestion?: string;

  public constructor(
    /** Human-readable import failure summary. */
    message: string,
    /** Stable machine-readable import failure code. */
    code: NeatChatSeedImportErrorCode,
    /** Optional fallback guidance directing callers to the supervised-distillation path. */
    distillationSuggestion?: string,
  ) {
    super(message);
    this.code = code;
    this.distillationSuggestion = distillationSuggestion;
  }
}