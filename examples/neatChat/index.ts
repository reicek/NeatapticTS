import { Architect, Network } from '../../src/browser-entry.ts';
import {
  NEATCHAT_AB_VARIANTS,
  NEATCHAT_DEFAULT_ARCHITECTURE_FAMILY,
  NEATCHAT_DEFAULT_CHUNK_TOKEN_COUNT,
  NEATCHAT_DEFAULT_CONTEXT_WINDOW_TOKEN_COUNT,
  NEATCHAT_DEFAULT_NARX_INPUT_MEMORY,
  NEATCHAT_DEFAULT_NARX_OUTPUT_MEMORY,
  NEATCHAT_DEFAULT_RECURRENT_BLOCK_SIZE,
  NEATCHAT_DEFAULT_TOP_WORD_LIMIT,
  NEATCHAT_DEFAULT_UNKNOWN_TOKEN,
  NEATCHAT_DEFAULT_VALIDATION_LINE_COUNT,
  NEATCHAT_DELIVERY_PROGRESS,
  NEATCHAT_LIGHTWEIGHT_METRICS,
  NEATCHAT_PUBLISHED_DOCS_EXAMPLE_PATH,
  NEATCHAT_PUBLISHED_EXAMPLES_CATEGORY,
  NEATCHAT_PUBLIC_BROWSER_ENTRYPOINT_MODULE_PATH,
  NEATCHAT_PUBLIC_BROWSER_HOST_PAGE_PATH,
  NEATCHAT_RECOMMENDED_TOP_WORD_LIMIT_RANGE,
  NEATCHAT_SPECIAL_TOKENS,
  NEATCHAT_SUPPORTED_ARCHITECTURE_FAMILIES,
  NEATCHAT_CORPUS_REPORT_FIELDS,
  NEATCHAT_VISUALIZER_DRAW_MODULE_PATH,
  NEATCHAT_VISUALIZER_EXAMPLE_ID,
  NEATCHAT_VISUALIZER_FRAME_RESOLVER_MODULE_PATH,
  NEATCHAT_VISUALIZER_OWNER_MODULE_PATH,
} from './core/neatChat.constants';
import {
  estimateNeatChatRuntime,
  resolvePositiveInteger,
} from './core/neatChat.tokenization.utils';
import { NEATCHAT_SAMPLE_CONVERSATION_LINES } from './sample-conversation';
import type {
  CreateNeatChatSeedNetworkOptions,
  NeatChatArchitectureFamily,
  NeatChatExampleContract,
  NeatChatSeedNetworkResult,
  NeatChatSeedNetworkSummary,
  NeatChatSeedValidationSplit,
} from './core/neatChat.types';

export type {
  NeatChatAdaptationCandidate,
  NeatChatAdaptationManager,
  NeatChatCandidateLogEntry,
  ScheduleNeatChatAdaptationOptions,
} from './core/neatChat.adaptation.types';
export type {
  CreateNeatChatAbComparisonOptions,
  CreateNeatChatPretrainingPreviewOptions,
  CreateNeatChatSeedNetworkOptions,
  CreateNeatChatSessionOptions,
  EstimateNeatChatRuntimeOptions,
  NeatChatAbComparisonContract,
  NeatChatAbComparisonResult,
  NeatChatAbVariant,
  NeatChatAbVariantResult,
  NeatChatArchitectureFamily,
  NeatChatCorpusReport,
  NeatChatCorpusReportField,
  NeatChatCorpusReportFieldKey,
  NeatChatDeliveryStatus,
  NeatChatDeliveryStep,
  NeatChatExchangeRecord,
  NeatChatExchangeResult,
  NeatChatExampleContract,
  NeatChatLightweightMetrics,
  NeatChatMetricName,
  NeatChatPretrainingContract,
  NeatChatPretrainingPreview,
  NeatChatRuntimeDurationBucket,
  NeatChatRuntimeEstimate,
  NeatChatSeedNetworkResult,
  NeatChatSeedNetworkSummary,
  NeatChatSeedValidationSplit,
  NeatChatSession,
  NeatChatSessionSnapshot,
  NeatChatSessionSnapshotV2,
  NeatChatVisualizationContract,
  NeatChatVocabulary,
} from './core/neatChat.types';
export type {
  NeatChatMemoryRecord,
  NeatChatEpisodicMemoryBank,
  NeatChatMemoryRetrievalResult,
  CreateNeatChatEpisodicMemoryBankOptions,
  RetrieveNeatChatMemoriesOptions,
} from './core/neatChat.memory.types';
export type {
  NeatChatExternalSeedDescriptor,
  NeatChatSeedImportResult,
  NeatChatSeedMetadata,
  NeatChatSupportedSeedFamily,
} from './core/neatChat.seed-import.types';
export {
  createNeatChatPretrainingPreview,
  estimateNeatChatRuntime,
  extractNeatChatConversationLines,
  tokenizeNeatChatText,
} from './core/neatChat.tokenization.utils';
export {
  createNeatChatAdaptationManager,
  promoteNeatChatAdaptationCandidate,
  rejectNeatChatAdaptationCandidate,
  scheduleNeatChatAdaptation,
} from './core/neatChat.adaptation.services.ts';
export {
  buildNeatChatVocabulary,
  createNeatChatSession,
  pretrainNeatChatSessionWithConversationLines,
  runNeatChatExchange,
  updateNeatChatSessionContextWindowTokenCount,
} from './core/neatChat.session.services';
export { createNeatChatAbComparison } from './core/neatChat.ab.services.ts';
export {
  exportNeatChatSession,
  importNeatChatSession,
} from './core/neatChat.snapshot.services';
export {
  exportNeatChatPortablePayload,
  exportNeatChatSessionV2,
  importNeatChatSessionV2,
} from './core/neatChat.snapshot.v2.services';
export {
  NEATCHAT_DEFAULT_MEMORY_BANK_MAX_RECORDS,
  createNeatChatEpisodicMemoryBank,
  addNeatChatMemoryRecord,
  retrieveNeatChatMemories,
  pruneNeatChatMemoryBank,
} from './core/neatChat.memory.services';
export { NeatChatSeedImportError } from './core/neatChat.seed-import.errors.ts';
export {
  buildSeedSnapshotFromExternalWeights,
  mapExternalRecurrentWeightsToParameterVector,
  validateNeatChatSeedFamily,
} from './core/neatChat.seed-import.services.ts';
export type {
  EvaluationMetric,
  FailureBucket,
  RegressionEntry,
  RegressionSuiteResult,
  EvaluationHarnessInput,
} from './core/neatChat.evaluation.types';
export {
  runNeatChatRegressionSuite,
  attributeToFailureBucket,
  scoreNextTokenAccuracy,
  scoreRepetitionRate,
  scoreResponseLengthStability,
  scoreUnknownHandling,
  scoreFactualConsistency,
} from './core/neatChat.evaluation.services';
export type {
  SafetyViolation,
  SafetyCheckResult,
} from './core/neatChat.safety.types';
export {
  checkSafety,
  isUnknownToken,
  isRepetitionCollapse,
  isDegenerateResponse,
} from './core/neatChat.safety.services';
export type {
  NeatChatRoutingCandidate,
  NeatChatRoutingDecisionLogEntry,
  NeatChatRoutingPath,
} from './core/neatChat.routing.types';
export {
  generateNeatChatCandidates,
  selectNeatChatCandidate,
  appendNeatChatRoutingDecision,
} from './core/neatChat.routing.services';

/**
 * Returns the initial public contract for the NEATchat example.
 *
 * The contract is intentionally narrow: one tiny recurrent seed-network path,
 * an optional pretraining surface with bounded vocabulary growth, a strict UNK
 * policy, one-session A or B comparison semantics, and an explicit promise to
 * reuse the Flappy visualizer baseline before adding any NEATchat-specific UI.
 *
 * ## Scale and limitations
 *
 * NEATchat is a toy sequence-learning demo with explicit boundaries:
 * - **One-hot vocabulary**: tokens are discrete one-hot vectors, not learned embeddings.
 *   Vocabulary is capped to the top-N most frequent terms; all other terms map to `UNK`.
 * - **Context window cap**: each exchange sees a short fixed-length token window.
 *   Long histories are truncated — there is no sliding attention or memory compression.
 * - **No transformer parity**: the architecture is a local LSTM, GRU, or NARX builder.
 *   It does not replicate transformer self-attention, positional encoding, or
 *   multi-head behavior and should not be compared against transformer-scale outputs.
 * - **Seed-import eligibility**: only single-layer GRU or LSTM models with one-hot IO,
 *   vocabulary between 300 and 3000, and sigmoid/tanh activations are eligible for
 *   direct parameter-vector import. Incompatible shapes route to supervised distillation.
 *
 * ## Feature lanes
 *
 * **Live (stable)**: tokenization, bounded pretraining, online exchange updates,
 * session export/import (`exportNeatChatSessionV2` / `importNeatChatSessionV2`),
 * episodic memory bank, and A/B metric comparison.
 *
 * **Experimental — observability only**: multi-path routing
 * (`generateNeatChatCandidates`, `selectNeatChatCandidate`) compares candidates
 * across base, personalized, and retrieval-grounded paths. The routing log is
 * an observability record only — routing does not promote weights or mutate
 * session state. Do not rely on routing decisions as durable training signal.
 *
 * **Experimental — explicit background job**: background adaptation
 * (`scheduleNeatChatAdaptation`) defers fine-tuning to the microtask queue.
 * The base session network is frozen during the job. Candidates must be
 * explicitly promoted via `promoteNeatChatAdaptationCandidate` or rejected via
 * `rejectNeatChatAdaptationCandidate`. Adaptation currently runs on the main
 * thread; a real worker-thread backend is not yet present.
 *
 * **Stable (W6 addition)**: the regression harness (`runNeatChatRegressionSuite`,
 * `attributeToFailureBucket`, and score helpers) runs held-out corpus evaluation
 * across five metrics. Baseline scores from the shipped default seed:
 * `heldOutNextTokenAccuracy = 12.29`, `repetitionRate = 0`,
 * `responseLengthStability = 1`. The safety gate (`checkSafety`,
 * `isUnknownToken`, `isRepetitionCollapse`, `isDegenerateResponse`) classifies
 * three failure modes — unknown-token, repetition-collapse, and degenerate
 * response — without throwing. All current baseline outputs pass `ok: true`.
 *
 * **External seed import**: the module also exports `validateNeatChatSeedFamily`,
 * `mapExternalRecurrentWeightsToParameterVector`, and `buildSeedSnapshotFromExternalWeights`
 * for the offline non-ONNX parameter-vector conversion flow. Compatible external
 * checkpoints are single-layer GRU or LSTM models with one-hot IO, exported in PyTorch
 * gate-row order. Incompatible models should use the supervised-distillation path instead.
 *
 * **Episodic memory bank**: the module also exports `addNeatChatMemoryRecord`,
 * `retrieveNeatChatMemories`, and `exportNeatChatSessionV2` for short-term
 * persistent memory of user-specific facts. Create a bank with
 * `createNeatChatEpisodicMemoryBank`, store facts with `addNeatChatMemoryRecord`,
 * retrieve relevant records by prompt-token overlap with `retrieveNeatChatMemories`
 * before generating each reply, and checkpoint the full bank state via
 * `exportNeatChatSessionV2` for durable session persistence.
 *
 * **Background adaptation (experimental)**: the module also exports
 * `createNeatChatAdaptationManager`, `scheduleNeatChatAdaptation`,
 * `promoteNeatChatAdaptationCandidate`, and `rejectNeatChatAdaptationCandidate`
 * for explicit, non-destructive personalization. `scheduleNeatChatAdaptation`
 * is async and non-blocking: it defers the fine-tune pass to the microtask queue
 * and returns a new manager view with the ready candidate appended to
 * `pendingCandidates`. The base session network is frozen throughout — it is
 * never mutated during a running job. `promoteNeatChatAdaptationCandidate`
 * applies explicit promotion by cloning the session network and writing the
 * trained weights via `fromParameterVector`; the original network is unaffected.
 * Rejected candidates are discarded via `rejectNeatChatAdaptationCandidate`.
 * Every promote or reject decision is appended to `session.candidateLog`, which
 * is persisted across `exportNeatChatSessionV2` / `importNeatChatSessionV2`
 * round-trips for a durable audit trail of personalization decisions.
 *
 * @returns Public contract summary for the first NEATchat slice.
 *
 * @example
 * ```ts
 * import { createNeatChatExampleContract } from './index';
 *
 * const contract = createNeatChatExampleContract();
 * console.log(contract.defaultArchitectureFamily); // lstm
 * console.log(contract.pretraining.defaultTopWordLimit); // 3000
 * ```
 */
export function createNeatChatExampleContract(): NeatChatExampleContract {
  const defaultRuntimeEstimate = estimateNeatChatRuntime();

  return {
    exampleId: 'NEATchat',
    publishedExamplesCategory: NEATCHAT_PUBLISHED_EXAMPLES_CATEGORY,
    publicEntrypointModulePath: 'examples/neatChat/index.ts',
    publicRunModulePath: 'examples/neatChat/run.ts',
    publicBrowserEntrypointModulePath:
      NEATCHAT_PUBLIC_BROWSER_ENTRYPOINT_MODULE_PATH,
    publicBrowserHostPagePath: NEATCHAT_PUBLIC_BROWSER_HOST_PAGE_PATH,
    publishedDocsExamplePath: NEATCHAT_PUBLISHED_DOCS_EXAMPLE_PATH,
    defaultArchitectureFamily: NEATCHAT_DEFAULT_ARCHITECTURE_FAMILY,
    supportedArchitectureFamilies: NEATCHAT_SUPPORTED_ARCHITECTURE_FAMILIES,
    inputFormat:
      'Encode each teaching exchange as BOS, user tokens, TURN_BREAK, assistant tokens, then EOS. Map any term outside the retained vocabulary to UNK so blank-start and preseeded runs share one stable token stream.',
    resetBehavior:
      'Create fresh blank-start and preseeded sessions for each A or B comparison, call clear before replaying a prompt, and only carry recurrent state forward inside one response-generation or one short online update window.',
    expectedOutput:
      'Blank-start runs should begin terse and unstable, while preseeded runs should reuse retained vocabulary sooner, show lower early UNK churn, and remain directly comparable through the same lightweight metrics loop.',
    sequenceBuilderRationale:
      'Start with a single small LSTM block because it demonstrates recurrent state clearly, stays close to the existing sequenceReset teaching path, and avoids overclaiming scale or architectural complexity.',
    pretraining: {
      optional: true,
      defaultTopWordLimit: NEATCHAT_DEFAULT_TOP_WORD_LIMIT,
      recommendedTopWordLimitRange: NEATCHAT_RECOMMENDED_TOP_WORD_LIMIT_RANGE,
      defaultContextWindowTokenCount:
        NEATCHAT_DEFAULT_CONTEXT_WINDOW_TOKEN_COUNT,
      defaultChunkTokenCount: NEATCHAT_DEFAULT_CHUNK_TOKEN_COUNT,
      unknownToken: NEATCHAT_DEFAULT_UNKNOWN_TOKEN,
      tokenizationRule:
        "Lowercase text, normalize common contractions (for example, can't to can not), collapse punctuation into compact class tokens, bucket numeric terms by size, truncate each prompt or reply slice to the short context window, and map out-of-vocabulary terms to UNK once the retained vocabulary is known.",
      defaultRuntimeEstimate,
      corpusReportFields: NEATCHAT_CORPUS_REPORT_FIELDS,
    },
    abComparison: {
      variants: NEATCHAT_AB_VARIANTS,
      lightweightMetrics: NEATCHAT_LIGHTWEIGHT_METRICS,
    },
    deliveryProgress: NEATCHAT_DELIVERY_PROGRESS,
    visualization: {
      exampleId: NEATCHAT_VISUALIZER_EXAMPLE_ID,
      ownerModulePath: NEATCHAT_VISUALIZER_OWNER_MODULE_PATH,
      frameResolverModulePath: NEATCHAT_VISUALIZER_FRAME_RESOLVER_MODULE_PATH,
      drawModulePath: NEATCHAT_VISUALIZER_DRAW_MODULE_PATH,
      reuseRule:
        'Reuse the Flappy host, network-view, and draw boundaries as the baseline network renderer before any NEATchat-specific visualizer polish is introduced.',
    },
  };
}

/**
 * Returns the built-in scripted sample conversation corpus.
 *
 * This corpus is shared by the browser one-click pretraining action and
 * NEATchat tests so both paths evaluate the same conversation flow baseline.
 *
 * @returns Defensive copy of sample conversation lines.
 */
export function getNeatChatSampleConversationLines(): string[] {
  return [...NEATCHAT_SAMPLE_CONVERSATION_LINES];
}

/**
 * Builds a small seed network from one of the public sequence builders.
 *
 * The builder stays intentionally compact because the first NEATchat slice is a
 * teaching example, not a large-scale language-model stack. The returned
 * network is the seed object that later tokenizer, online-update, and browser
 * layers can reuse without inventing a second network-construction path.
 *
 * @param options - Vocabulary size and builder-shape options.
 * @returns Built seed network plus a stable public summary.
 *
 * @example
 * ```ts
 * import { createNeatChatSeedNetwork } from './index';
 *
 * const seedNetwork = createNeatChatSeedNetwork({ vocabularySize: 64 });
 * console.log(seedNetwork.summary.effectiveVocabularySize); // 68
 * ```
 */
export function createNeatChatSeedNetwork(
  options: CreateNeatChatSeedNetworkOptions,
): NeatChatSeedNetworkResult {
  // Step 1: Validate the requested vocabulary and resolve default builder knobs.
  const vocabularySize = resolvePositiveInteger(
    options.vocabularySize,
    'vocabularySize',
  );
  const architectureFamily =
    options.architectureFamily ?? NEATCHAT_DEFAULT_ARCHITECTURE_FAMILY;
  const recurrentBlockSize = resolvePositiveInteger(
    options.recurrentBlockSize ?? NEATCHAT_DEFAULT_RECURRENT_BLOCK_SIZE,
    'recurrentBlockSize',
  );
  const narxInputMemory = resolvePositiveInteger(
    options.narxInputMemory ?? NEATCHAT_DEFAULT_NARX_INPUT_MEMORY,
    'narxInputMemory',
  );
  const narxOutputMemory = resolvePositiveInteger(
    options.narxOutputMemory ?? NEATCHAT_DEFAULT_NARX_OUTPUT_MEMORY,
    'narxOutputMemory',
  );

  // Step 2: Reserve special tokens before constructing the builder-backed graph.
  const effectiveVocabularySize =
    vocabularySize + NEATCHAT_SPECIAL_TOKENS.length;
  const network = buildSeedNetwork(
    architectureFamily,
    effectiveVocabularySize,
    recurrentBlockSize,
    narxInputMemory,
    narxOutputMemory,
  );

  // Step 3: Return both the network and the public summary used by docs or CLI.
  return {
    network,
    summary: {
      architectureFamily,
      effectiveVocabularySize,
      inputCount: network.inputNodeIds.length,
      outputCount: network.outputNodeIds.length,
      includesUnknownToken: NEATCHAT_SPECIAL_TOKENS.includes(
        NEATCHAT_DEFAULT_UNKNOWN_TOKEN,
      ),
      specialTokens: NEATCHAT_SPECIAL_TOKENS,
      topologyIntent: network.getTopologyIntent(),
    },
  };

  /**
   * Builds the requested network family from the public Architect surface.
   *
   * @param family - Selected builder family.
   * @param ioSize - Shared input and output vocabulary size.
   * @param hiddenBlockSize - Width of the recurrent hidden block.
   * @param inputMemory - NARX input delay depth.
   * @param outputMemory - NARX output delay depth.
   * @returns Builder-backed network.
   */
  function buildSeedNetwork(
    family: NeatChatArchitectureFamily,
    ioSize: number,
    hiddenBlockSize: number,
    inputMemory: number,
    outputMemory: number,
  ): Network {
    if (family === 'gru') {
      return Architect.gru(ioSize, hiddenBlockSize, ioSize, {
        inputToOutput: true,
      });
    }

    if (family === 'narx') {
      return Architect.narx(
        ioSize,
        [hiddenBlockSize],
        ioSize,
        inputMemory,
        outputMemory,
      );
    }

    return Architect.lstm(ioSize, hiddenBlockSize, ioSize, {
      inputToOutput: true,
    });
  }
}

/**
 * Splits conversation lines into seed and held-out validation slices.
 *
 * The split keeps the final `validationLineCount` lines as held-out data and
 * uses the remaining prefix for optional preseed training. For very small
 * inputs, the function gracefully returns an empty validation slice.
 *
 * @param conversationLines - Ordered conversation lines from corpus input.
 * @param validationLineCount - Number of trailing lines to reserve for validation.
 * @returns Seed and held-out validation slices.
 */
export function splitNeatChatSeedAndValidationLines(
  conversationLines: readonly string[],
  validationLineCount = NEATCHAT_DEFAULT_VALIDATION_LINE_COUNT,
): NeatChatSeedValidationSplit {
  const resolvedValidationLineCount = resolvePositiveInteger(
    validationLineCount,
    'validationLineCount',
  );
  const safeConversationLines = [...conversationLines]
    .map((conversationLine) => conversationLine.trim())
    .filter((conversationLine) => conversationLine.length > 0);

  if (safeConversationLines.length <= resolvedValidationLineCount) {
    return {
      seedConversationLines: [],
      validationConversationLines: safeConversationLines,
    };
  }

  const splitIndex = safeConversationLines.length - resolvedValidationLineCount;

  return {
    seedConversationLines: safeConversationLines.slice(0, splitIndex),
    validationConversationLines: safeConversationLines.slice(splitIndex),
  };
}

/**
 * Formats the NEATchat contract into a short human-readable preview.
 *
 * The output is intended for a tiny Node runner and for quick review in chat.
 * It explains the input format, reset behavior, and expected output in a few
 * compact paragraphs while also surfacing the visualizer reuse boundary.
 *
 * @param exampleContract - Contract returned by `createNeatChatExampleContract`.
 * @param seedNetworkSummary - Summary returned by `createNeatChatSeedNetwork`.
 * @returns Multi-paragraph preview of the current NEATchat example boundary.
 */
export function formatNeatChatExampleContract(
  exampleContract: NeatChatExampleContract,
  seedNetworkSummary: NeatChatSeedNetworkSummary,
): string {
  const corpusReportFieldLabels = exampleContract.pretraining.corpusReportFields
    .map((corpusReportField) => corpusReportField.label)
    .join(', ');
  const progressPreview = exampleContract.deliveryProgress
    .map((deliveryStep) => `${deliveryStep.status}: ${deliveryStep.label}`)
    .join(' | ');

  return [
    `NEATchat is a tiny online sequence-learning chatbot contract built around a ${seedNetworkSummary.architectureFamily.toUpperCase()} seed network with ${seedNetworkSummary.inputCount} input tokens and ${seedNetworkSummary.outputCount} output tokens. It is intentionally small enough to demonstrate sequence builders and online adaptation without pretending to be a transformer-scale trainer.`,
    `Published browser surface: ${exampleContract.publicBrowserHostPagePath} ships through ${exampleContract.publishedDocsExamplePath} in the ${exampleContract.publishedExamplesCategory} examples section, so progress stays visible without depending on Node.`,
    `Input format: ${exampleContract.inputFormat}`,
    `Reset behavior: ${exampleContract.resetBehavior}`,
    `Expected output: ${exampleContract.expectedOutput}`,
    `Optional pretraining keeps a top-word cap of ${exampleContract.pretraining.defaultTopWordLimit} by default, recommends a range of ${exampleContract.pretraining.recommendedTopWordLimitRange[0]}-${exampleContract.pretraining.recommendedTopWordLimitRange[1]}, uses a short context window of ${exampleContract.pretraining.defaultContextWindowTokenCount} tokens, processes pasted text in chunks of ${exampleContract.pretraining.defaultChunkTokenCount} tokens, and reports ${corpusReportFieldLabels}.`,
    `Runtime estimate: ${exampleContract.pretraining.defaultRuntimeEstimate.summary}`,
    `Delivery progress: ${progressPreview}.`,
    `The first visualizer path stays aligned with ${exampleContract.visualization.exampleId}: reuse ${exampleContract.visualization.ownerModulePath}, ${exampleContract.visualization.frameResolverModulePath}, and ${exampleContract.visualization.drawModulePath} before adding NEATchat-specific rendering polish.`,
  ].join('\n\n');
}
