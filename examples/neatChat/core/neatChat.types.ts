import type { Network } from '../../../src/browser-entry.ts';
import type {
  ParameterLayoutEntry,
  ParameterVector,
} from '../../../src/neataptic.ts';
import type {
  NeatChatAdaptationCandidate,
  NeatChatCandidateLogEntry,
} from './neatChat.adaptation.types';
import type { NeatChatSeedMetadata } from './neatChat.seed-import.types';
import type { NeatChatEpisodicMemoryBank } from './neatChat.memory.types';
import type { NeatChatRoutingDecisionLogEntry } from './neatChat.routing.types';
import {
  NEATCHAT_PUBLISHED_DOCS_EXAMPLE_PATH,
  NEATCHAT_PUBLISHED_EXAMPLES_CATEGORY,
  NEATCHAT_PUBLIC_BROWSER_ENTRYPOINT_MODULE_PATH,
  NEATCHAT_PUBLIC_BROWSER_HOST_PAGE_PATH,
  NEATCHAT_VISUALIZER_EXAMPLE_ID,
} from './neatChat.constants';

/** Supported sequence-builder families for the NEATchat example. */
export type NeatChatArchitectureFamily = 'lstm' | 'gru' | 'narx';

/** Supported side-by-side comparison variants for one prompt. */
export type NeatChatAbVariant = 'blank-start' | 'preseeded';

/** Delivery status shown on the published browser contract page. */
export type NeatChatDeliveryStatus = 'available' | 'planned';

/** Lightweight metrics shown after each A or B comparison pass. */
export type NeatChatMetricName =
  | 'held-out-next-token-accuracy'
  | 'repetition-rate'
  | 'response-length-stability';

/** Lightweight runtime buckets used before pretraining starts. */
export type NeatChatRuntimeDurationBucket =
  | 'very-short'
  | 'short'
  | 'moderate'
  | 'heavy';

/** Stable field keys used in the pretraining corpus report. */
export type NeatChatCorpusReportFieldKey =
  | 'characterCount'
  | 'tokenCount'
  | 'uniqueTermCount'
  | 'retainedTermCount'
  | 'retainedTokenCoveragePercent';

type NeatChatSpecialToken = 'UNK' | 'BOS' | 'EOS' | 'TURN_BREAK';

/** Human-readable metadata for one corpus report field. */
export interface NeatChatCorpusReportField {
  /** Stable field key used in UI and logs. */
  readonly key: NeatChatCorpusReportFieldKey;
  /** Short label suitable for compact reports. */
  readonly label: string;
  /** Concise explanation of what the field means. */
  readonly description: string;
}

/** Concrete corpus-report values shown before pretraining begins. */
export interface NeatChatCorpusReport {
  /** Total pasted characters before tokenization. */
  readonly characterCount: number;
  /** Token count before the top-word cap is applied. */
  readonly tokenCount: number;
  /** Distinct normalized terms observed in the pasted corpus. */
  readonly uniqueTermCount: number;
  /** Terms kept after the top-word cap is applied. */
  readonly retainedTermCount: number;
  /** Share of tokens covered by the retained vocabulary. */
  readonly retainedTokenCoveragePercent: number;
}

/** Visualization reuse boundary for the first NEATchat slice. */
export interface NeatChatVisualizationContract {
  /** Existing example that owns the baseline visualizer implementation. */
  readonly exampleId: typeof NEATCHAT_VISUALIZER_EXAMPLE_ID;
  /** Host-side owner for canvas lifetime and pointer state. */
  readonly ownerModulePath: string;
  /** Frame resolver that caches topology and layout for redraws. */
  readonly frameResolverModulePath: string;
  /** Draw layer that keeps connection and node rendering semantics. */
  readonly drawModulePath: string;
  /** Rule that prevents an early NEATchat-specific visualization fork. */
  readonly reuseRule: string;
}

/** Optional copy-paste pretraining contract for the first public slice. */
export interface NeatChatPretrainingContract {
  /** Whether corpus pretraining is optional rather than required. */
  readonly optional: true;
  /** Default retained top-word cap used for vocabulary pruning. */
  readonly defaultTopWordLimit: number;
  /** Recommended experiment range for the top-word cap. */
  readonly recommendedTopWordLimitRange: readonly [number, number];
  /** Short token window kept for one prompt or reply slice. */
  readonly defaultContextWindowTokenCount: number;
  /** Internal chunk size used while processing large pasted corpora. */
  readonly defaultChunkTokenCount: number;
  /** Stable token used when a term falls outside the retained vocabulary. */
  readonly unknownToken: NeatChatSpecialToken;
  /** Human-readable description of the tokenizer used by the example. */
  readonly tokenizationRule: string;
  /** Default runtime estimate shown before pretraining starts. */
  readonly defaultRuntimeEstimate: NeatChatRuntimeEstimate;
  /** Ordered corpus report fields shown before training begins. */
  readonly corpusReportFields: readonly NeatChatCorpusReportField[];
}

/** Lightweight estimate shown before bounded pretraining starts. */
export interface NeatChatRuntimeEstimate {
  /** Requested top-word cap used for the estimate. */
  readonly topWordLimit: number;
  /** Reserved special tokens included in the retained vocabulary. */
  readonly specialTokenCount: number;
  /** Estimated retained vocabulary size including special tokens. */
  readonly estimatedRetainedVocabularySize: number;
  /** Short token window kept per prompt or reply slice. */
  readonly contextWindowTokenCount: number;
  /** Relative pretraining-time bucket for the chosen budget. */
  readonly expectedPretrainingDurationBucket: NeatChatRuntimeDurationBucket;
  /** Concise explanation suitable for CLI and browser previews. */
  readonly summary: string;
}

/** Options for building a lightweight pretraining runtime estimate. */
export interface EstimateNeatChatRuntimeOptions {
  /** Requested top-word cap for the estimate. */
  readonly topWordLimit?: number;
  /** Token window kept for one prompt or reply slice. */
  readonly contextWindowTokenCount?: number;
}

/** Options for building an optional pretraining preview from pasted text. */
export interface CreateNeatChatPretrainingPreviewOptions {
  /** User-pasted corpus text used for the optional pretraining preview. */
  readonly corpusText: string;
  /** Requested top-word cap for retained vocabulary terms. */
  readonly topWordLimit?: number;
  /** Number of tokens processed per bounded counting chunk. */
  readonly chunkTokenCount?: number;
  /** Number of retained terms surfaced in the preview summary. */
  readonly previewTermCount?: number;
}

/** Shared optional pretraining preview built from a pasted corpus. */
export interface NeatChatPretrainingPreview {
  /** Whether the user provided any corpus text at all. */
  readonly hasCorpus: boolean;
  /** Total pasted characters before tokenization. */
  readonly characterCount: number;
  /** Requested top-word cap used for retention. */
  readonly topWordLimit: number;
  /** Bounded token chunk size used while counting term frequency. */
  readonly chunkTokenCount: number;
  /** Total normalized tokens observed in the pasted corpus. */
  readonly totalTokenCount: number;
  /** Number of distinct normalized terms observed in the corpus. */
  readonly uniqueTermCount: number;
  /** Number of retained terms after the top-word cap is applied. */
  readonly retainedTermCount: number;
  /** Number of tokens covered by the retained vocabulary. */
  readonly retainedTokenCount: number;
  /** Retained-token coverage percentage rounded to two decimals. */
  readonly retainedTokenCoveragePercent: number;
  /** Full Step 4 corpus report used by browser and CLI surfaces. */
  readonly corpusReport: NeatChatCorpusReport;
  /** Number of bounded counting chunks processed from the corpus. */
  readonly processedChunkCount: number;
  /** Small preview list of retained terms for UI copy. */
  readonly previewTerms: readonly string[];
  /** Full ordered list of all retained terms available for vocabulary building. */
  readonly retainedTerms: readonly string[];
}

/** Token-to-index mapping built from the retained corpus vocabulary plus special tokens. */
export interface NeatChatVocabulary {
  /** Total vocabulary size including special tokens. */
  readonly size: number;
  /** Maps each normalized token string to its one-hot index. */
  readonly termToIndex: ReadonlyMap<string, number>;
  /** Maps each one-hot index back to its normalized token string. */
  readonly indexToTerm: readonly string[];
}

/** Options for creating a live chat session from an optional corpus vocabulary. */
export interface CreateNeatChatSessionOptions {
  /** Retained corpus terms used to build the session vocabulary. */
  readonly corpusRetainedTerms?: readonly string[];
  /** Optional normalized dialogue lines used for initial supervised seeding. */
  readonly seedConversationLines?: readonly string[];
  /** Builder family for the seed network. Defaults to LSTM. */
  readonly architectureFamily?: NeatChatArchitectureFamily;
  /** Width of the recurrent hidden block. Defaults to 24 units. */
  readonly recurrentBlockSize?: number;
  /** Maximum retained terms used in the live-session vocabulary. */
  readonly liveChatVocabLimit?: number;
  /** Context-window token count used to tokenize each live user message. */
  readonly contextWindowTokenCount?: number;
}

/** Record of one completed user-bot exchange including learning metadata. */
export interface NeatChatExchangeRecord {
  /** Original user message text. */
  readonly userMessage: string;
  /** Bot response as a decoded token string. */
  readonly response: string;
  /** Number of consecutive token pairs the network trained on for this exchange. */
  readonly trainedTokenPairCount: number;
  /** Normalized user tokens that fit within the context window. */
  readonly userTokens: readonly string[];
  /** Decoded response tokens generated by the network. */
  readonly responseTokens: readonly string[];
}

/** Stateful live-chat session holding the network, vocabulary, and exchange history. */
export interface NeatChatSession {
  /** Token-to-index vocabulary used by this session. */
  readonly vocabulary: NeatChatVocabulary;
  /**
   * LSTM seed network - mutated in-place by each exchange.
   *
   * The reference is stable across exchanges; only the internal weights change.
   */
  readonly network: Network;
  /** Ordered list of completed exchanges in this session. */
  readonly exchanges: readonly NeatChatExchangeRecord[];
  /** Total number of exchanges completed so far. */
  readonly learnedExchangeCount: number;
  /** Cumulative token pairs trained across all exchanges. */
  readonly learnedTokenPairCount: number;
  /** Token pairs used during initial session seeding before live exchanges. */
  readonly seededTokenPairCount: number;
  /** Context-window token count used for live-message tokenization. */
  readonly contextWindowTokenCount: number;
  /** Number of recent exchanges currently available to replay each update. */
  readonly replayBufferExchangeCount: number;
  /** Detached candidates that have not yet been promoted or rejected. */
  readonly pendingCandidates: readonly NeatChatAdaptationCandidate[];
  /** Durable record of promoted and rejected adaptation candidates. */
  readonly candidateLog: readonly NeatChatCandidateLogEntry[];
  /** Episodic memory bank storing retrievable user-specific facts and examples. */
  readonly memoryBank: NeatChatEpisodicMemoryBank;
  /**
   * Durable ordered log of response-path routing decisions for debugging and regression replay.
   *
   * Each entry records which candidate paths participated in the comparison, their scalar
   * scores, the winning path, and which retrieved memories (if any) contributed to a
   * retrieval-grounded candidate. The log survives snapshot export and import via
   * `exportNeatChatSessionV2` and `importNeatChatSessionV2` so routing history is
   * preserved across sessions.
   *
   * Routing selection never promotes weights; the log is a pure observability artifact.
   * Pre-W5 snapshots restore with an empty log (`[]`) as the backward-compatible default.
   */
  readonly routingLog: readonly NeatChatRoutingDecisionLogEntry[];
}

/** Serializable snapshot of a live NEATchat session. */
export interface NeatChatSessionSnapshot {
  /** Snapshot schema version for future compatibility guards. */
  readonly formatVersion: 1;
  /** Retained non-special vocabulary terms in index order after the control tokens. */
  readonly retainedTerms: readonly string[];
  /** Serialized network graph and weights. */
  readonly networkJson: Record<string, unknown>;
  /** Completed exchanges retained for continued online learning. */
  readonly exchanges: readonly NeatChatExchangeRecord[];
  /** Total completed exchange count at export time. */
  readonly learnedExchangeCount: number;
  /** Total learned token-pair count at export time. */
  readonly learnedTokenPairCount: number;
  /** Total seed token-pair count at export time. */
  readonly seededTokenPairCount: number;
  /** Active context window preserved with the snapshot. */
  readonly contextWindowTokenCount: number;
}

/**
 * Serializable v2 snapshot of a live NEATchat session.
 *
 * Version 2 extends the v1 topology JSON with a full parameter-vector payload
 * so the restored network rebuilds its graph shell from `networkJson` first and
 * then replays exact scalar weights — including recurrent self-connection weights
 * for LSTM, GRU, and NARX gates and memory cells — through
 * `fromParameterVector(...)`. This two-phase restore guarantees that the
 * in-memory session matches the original at the weight level, not merely at the
 * topology level, which is the key invariant for deterministic continued training
 * and candidate comparison.
 *
 * The `extensions` bag is a forward-compatible namespace for NEATchat-owned
 * snapshot metadata. Downstream validation code can read
 * `extensions.neatchat.vocabularySize` as a sanity check without rebuilding
 * the full token table from `retainedTerms`.
 *
 * @example
 * ```ts
 * // Persist a live session and restore it in a later run.
 * const snapshot = exportNeatChatSessionV2(session);
 * const json = JSON.stringify(snapshot);
 * // — later, in a new session —
 * const restoredSession = importNeatChatSessionV2(JSON.parse(json));
 * ```
 */
export interface NeatChatSessionSnapshotV2 {
  /** Snapshot schema version for future compatibility guards. */
  readonly formatVersion: 2;
  /** Retained non-special vocabulary terms in index order after the control tokens. */
  readonly retainedTerms: readonly string[];
  /** Serialized network graph and topology state. */
  readonly networkJson: Record<string, unknown>;
  /**
   * Exact network parameter payload used to restore weights after `networkJson` rebuild.
   *
   * The vector includes every trainable weight in the network: feed-forward
   * connection weights, bias values, and — critically for recurrent
   * architectures — the recurrent self-connection weights that encode LSTM gate
   * states, GRU reset and update logic, and NARX memory-tap coupling. Without
   * this vector, a topology-only restore from `networkJson` would lose the
   * learned recurrent dynamics accumulated during previous exchanges.
   *
   * The v2 exporter writes the summarized JSON-safe branch (plain `values`
   * array + `layoutEntries` descriptors). The runtime import also accepts a raw
   * `ParameterVector` so tests and transitional callers can round-trip through
   * the same `fromParameterVector(...)` seam without an extra serialization step.
   */
  readonly parameterVector:
    | ParameterVector
    | {
        /** Scalar parameter values aligned with `layoutEntries`. */
        readonly values: readonly number[];
        /** Version of the parameter-layout contract used by the snapshot. */
        readonly layoutVersion: number;
        /** Stable descriptor hash for the serialized layout entries. */
        readonly descriptorHash: string;
        /** Ordered layout descriptors aligned with `values`. */
        readonly layoutEntries: readonly ParameterLayoutEntry[];
      };
  /** Completed exchanges retained for continued online learning. */
  readonly exchanges: readonly NeatChatExchangeRecord[];
  /** Total completed exchange count at export time. */
  readonly learnedExchangeCount: number;
  /** Total learned token-pair count at export time. */
  readonly learnedTokenPairCount: number;
  /** Total seed token-pair count at export time. */
  readonly seededTokenPairCount: number;
  /** Active context window preserved with the snapshot. */
  readonly contextWindowTokenCount: number;
  /**
   * Forward-compatible metadata bag for NEATchat-owned snapshot extensions.
   *
   * Each key under `extensions` is a named namespace owned by one subsystem.
   * The `neatchat` branch is the canonical NEATchat namespace; callers must
   * not write to it outside the snapshot service. Additional namespaces can be
   * added by future workstreams without breaking existing importers.
   */
  readonly extensions: {
    /**
     * NEATchat-owned metadata branch.
     *
     * Values here are denormalized hints that let importers run fast sanity
     * checks — such as vocabulary-size consistency — without needing to
     * deserialize and rebuild the full token table from `retainedTerms` first.
     */
    readonly neatchat: {
      /**
       * Total vocabulary size (retained terms + special tokens) at export time.
       *
       * This is a denormalized hint: its value equals
       * `retainedTerms.length + NEATCHAT_SPECIAL_TOKENS.length`. The importer
       * validates this field against the rebuilt vocabulary to catch
       * truncated or mismatched snapshot payloads early, before attempting
       * network activation.
       */
      readonly vocabularySize: number;
      /** Optional metadata describing the external teacher seed that produced this snapshot. */
      readonly seedMetadata?: NeatChatSeedMetadata;
      /** Optional episodic memory bank persisted alongside the session snapshot. */
      readonly memoryBank?: NeatChatEpisodicMemoryBank;
      /** Optional durable record of promoted and rejected candidates. */
      readonly candidateLog?: readonly NeatChatCandidateLogEntry[];
      /** Optional durable record of response-path routing decisions. */
      readonly routingLog?: readonly NeatChatRoutingDecisionLogEntry[];
      readonly [key: string]: unknown;
    };
    readonly [key: string]: unknown;
  };
}

/** Result returned after one user-bot exchange and its supervised update. */
export interface NeatChatExchangeResult {
  /** Bot response as a decoded space-joined string. */
  readonly response: string;
  /** Decoded response tokens generated by the network. */
  readonly responseTokens: readonly string[];
  /** Normalized user tokens that fit within the context window. */
  readonly userTokens: readonly string[];
  /** Number of consecutive token pairs trained for this exchange. */
  readonly trainedTokenPairCount: number;
  /** Updated session with the new exchange appended and learning counts incremented. */
  readonly updatedSession: NeatChatSession;
}

/** One-session comparison contract between blank-start and preseeded runs. */
export interface NeatChatAbComparisonContract {
  /** Variants shown for the same prompt in one session. */
  readonly variants: readonly NeatChatAbVariant[];
  /** Small metrics loop that stays cheap enough for the example scope. */
  readonly lightweightMetrics: readonly NeatChatMetricName[];
}

/** One held-out split used for pretraining and tiny validation checks. */
export interface NeatChatSeedValidationSplit {
  /** Conversation lines used to preseed the pretraining variant. */
  readonly seedConversationLines: readonly string[];
  /** Held-out lines used for Step 6 next-token validation checks. */
  readonly validationConversationLines: readonly string[];
}

/** Lightweight quality metrics tracked for one A/B variant result. */
export interface NeatChatLightweightMetrics {
  /** Next-token accuracy on a tiny held-out sequence, in percent. */
  readonly heldOutNextTokenAccuracy: number;
  /** Share of adjacent response tokens that repeat consecutively, in [0, 1]. */
  readonly repetitionRate: number;
  /** Inverse mean absolute deviation of response lengths, in [0, 1]. */
  readonly responseLengthStability: number;
}

/** One variant result for a single prompt in the Step 6 A/B pass. */
export interface NeatChatAbVariantResult {
  /** Variant identity for this response and metric row. */
  readonly variant: NeatChatAbVariant;
  /** Prompt used for this variant run. */
  readonly prompt: string;
  /** Decoded response string generated from this variant's session. */
  readonly response: string;
  /** Response tokens generated by this variant run. */
  readonly responseTokens: readonly string[];
  /** Token-pair count used in the post-exchange supervised update. */
  readonly trainedTokenPairCount: number;
  /** Lightweight metrics tracked for this variant run. */
  readonly metrics: NeatChatLightweightMetrics;
}

/** Options for running one session-local blank-start vs preseeded comparison. */
export interface CreateNeatChatAbComparisonOptions {
  /** Retained corpus terms used to build the session vocabulary. */
  readonly corpusRetainedTerms?: readonly string[];
  /** Conversation lines used to preseed the pretraining variant. */
  readonly seedConversationLines?: readonly string[];
  /** Held-out lines used for tiny next-token validation checks. */
  readonly validationConversationLines?: readonly string[];
  /** Maximum retained terms used in both A/B session vocabularies. */
  readonly liveChatVocabLimit?: number;
  /** Builder family for both A/B sessions. Defaults to LSTM. */
  readonly architectureFamily?: NeatChatArchitectureFamily;
  /** Width of the recurrent hidden block for both A/B sessions. */
  readonly recurrentBlockSize?: number;
}

/** Combined result for one session-local A/B prompt comparison. */
export interface NeatChatAbComparisonResult {
  /** Prompt evaluated by both variants in this one-session pass. */
  readonly prompt: string;
  /** Ordered variant rows for blank-start and preseeded runs. */
  readonly variants: readonly NeatChatAbVariantResult[];
}

/** Progress row shown on the browser-hosted contract preview. */
export interface NeatChatDeliveryStep {
  /** Human-readable step label. */
  readonly label: string;
  /** Whether the step is already visible or still queued. */
  readonly status: NeatChatDeliveryStatus;
  /** Concise summary of what the step owns. */
  readonly summary: string;
}

/** Public contract summary for the first NEATchat example boundary. */
export interface NeatChatExampleContract {
  /** User-facing example id. */
  readonly exampleId: 'NEATchat';
  /** Category used when the browser page is published under docs/examples. */
  readonly publishedExamplesCategory: typeof NEATCHAT_PUBLISHED_EXAMPLES_CATEGORY;
  /** Smallest usable public module entrypoint for the example. */
  readonly publicEntrypointModulePath: 'examples/neatChat/index.ts';
  /** Default runnable preview path for Node-based inspection. */
  readonly publicRunModulePath: 'examples/neatChat/run.ts';
  /** Browser entrypoint bundled for the published docs example page. */
  readonly publicBrowserEntrypointModulePath: typeof NEATCHAT_PUBLIC_BROWSER_ENTRYPOINT_MODULE_PATH;
  /** Browser host page that loads the bundled example preview. */
  readonly publicBrowserHostPagePath: typeof NEATCHAT_PUBLIC_BROWSER_HOST_PAGE_PATH;
  /** Docs publication target for the browser-hosted example page. */
  readonly publishedDocsExamplePath: typeof NEATCHAT_PUBLISHED_DOCS_EXAMPLE_PATH;
  /** Sequence-builder family used for the first seed-network path. */
  readonly defaultArchitectureFamily: NeatChatArchitectureFamily;
  /** Builder families intentionally kept in scope for the tiny demo. */
  readonly supportedArchitectureFamilies: readonly NeatChatArchitectureFamily[];
  /** Compact explanation of the token-stream input format. */
  readonly inputFormat: string;
  /** Reset policy for A or B comparisons and short teaching loops. */
  readonly resetBehavior: string;
  /** High-level expectation for blank-start versus preseeded output. */
  readonly expectedOutput: string;
  /** Why the first slice defaults to an LSTM rather than a larger stack. */
  readonly sequenceBuilderRationale: string;
  /** Optional corpus-pretraining contract. */
  readonly pretraining: NeatChatPretrainingContract;
  /** One-session blank-start versus preseeded comparison contract. */
  readonly abComparison: NeatChatAbComparisonContract;
  /** Visible implementation-progress summary for the browser preview page. */
  readonly deliveryProgress: readonly NeatChatDeliveryStep[];
  /** Reused Flappy visualizer boundary for the first slice. */
  readonly visualization: NeatChatVisualizationContract;
}

/** Options for building a small NEATchat seed network from a public builder. */
export interface CreateNeatChatSeedNetworkOptions {
  /** Number of retained top-frequency vocabulary terms before special tokens. */
  readonly vocabularySize: number;
  /** Builder family used for the network. Defaults to LSTM. */
  readonly architectureFamily?: NeatChatArchitectureFamily;
  /** Width of the recurrent hidden block. Defaults to 24 units. */
  readonly recurrentBlockSize?: number;
  /** Explicit input-memory shelf depth when using NARX. */
  readonly narxInputMemory?: number;
  /** Explicit output-memory shelf depth when using NARX. */
  readonly narxOutputMemory?: number;
}

/** Stable summary of the built seed network. */
export interface NeatChatSeedNetworkSummary {
  /** Builder family used to construct the network. */
  readonly architectureFamily: NeatChatArchitectureFamily;
  /** Retained vocabulary size plus special tokens. */
  readonly effectiveVocabularySize: number;
  /** Input-node count for the built network. */
  readonly inputCount: number;
  /** Output-node count for the built network. */
  readonly outputCount: number;
  /** Whether the stable unknown-token path is present in the vocabulary. */
  readonly includesUnknownToken: boolean;
  /** Special tokens reserved before any user corpus is ingested. */
  readonly specialTokens: readonly NeatChatSpecialToken[];
  /** Public topology intent string reported by the built network. */
  readonly topologyIntent: string;
}

/** Built network plus a compact public summary of its shape. */
export interface NeatChatSeedNetworkResult {
  /** Newly constructed seed network for the example. */
  readonly network: Network;
  /** Public summary suitable for docs, logs, or CLI previews. */
  readonly summary: NeatChatSeedNetworkSummary;
}
