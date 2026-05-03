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
  createNeatChatPretrainingPreview,
  estimateNeatChatRuntime,
  extractNeatChatConversationLines,
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
  NeatChatVisualizationContract,
  NeatChatVocabulary,
} from './core/neatChat.types';
export {
  createNeatChatPretrainingPreview,
  estimateNeatChatRuntime,
  extractNeatChatConversationLines,
  tokenizeNeatChatText,
} from './core/neatChat.tokenization.utils';
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

/**
 * Returns the initial public contract for the NEATchat example.
 *
 * The contract is intentionally narrow: one tiny recurrent seed-network path,
 * an optional pretraining surface with bounded vocabulary growth, a strict UNK
 * policy, one-session A or B comparison semantics, and an explicit promise to
 * reuse the Flappy visualizer baseline before adding any NEATchat-specific UI.
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
        'Lowercase the text, keep Unicode letter and number runs plus apostrophes, truncate each prompt or reply slice to the short context window, and map out-of-vocabulary terms to UNK once the retained vocabulary is known.',
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
