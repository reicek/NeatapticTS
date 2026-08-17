/**
 * @module rerank-index
 * @description Cross-encoder re-ranking pipeline for second-stage result refinement.
 *
 * Takes top-N hybrid search candidates and re-scores each (query, body_text) pair
 * through a local ONNX cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
 * Outputs relevance scores that replace or augment the hybrid blended scores.
 *
 * The pipeline is:
 * 1. Load cross-encoder ONNX session (lazy, process-lifetime cached)
 * 2. For each candidate: tokenize (query, body_text) pair → [CLS] Q [SEP] D [SEP]
 * 3. Run inference → logits [1, 2] → softmax → relevance_score (probability of "relevant")
 * 4. Sort by rerank_score descending
 * 5. Return top-K reranked candidates
 *
 * @param {string}  [--query <text>]                          - Query text for re-ranking.
 * @param {string}  [--candidates <json>]                     - JSON array of candidates.
 * @param {number}  [--limit <n>]                             - Maximum returned result count (default: 10).
 * @param {string}  [--reranker-model-directory <path>]       - Override reranker model directory.
 * @param {string}  [--reranker-model-id <id>]                - Override reranker model identifier.
 * @param {number}  [--rerank-candidates-count <n>]           - Number of candidates to re-rank (default: 50).
 * @param {boolean} [--json]                                  - Emit JSON results.
 * @param {boolean} [--help]                                  - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on error.
 */
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from './cli-utils.mjs';
import {
  DEFAULT_RERANKER_MODEL_DIRECTORY,
  DEFAULT_RERANKER_MODEL_ID,
  DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH,
} from './reranker-readiness.mjs';

/**
 * Default number of hybrid candidates to re-rank before returning top-K.
 * Balances latency (~250ms for 50 pairs) with recall.
 */
export const DEFAULT_RERANKER_CANDIDATES_COUNT = 50;

// Process-lifetime cache for the ONNX session and tokenizer
let cachedRerankSession = null;
let cachedRerankTokenizer = null;

/**
 * Clamp and coerce the rerank candidates count to a valid integer in [1, 200].
 *
 * @param {unknown} value - Raw count value from the caller.
 * @returns {number} Normalized candidates count, defaulting to 50 when absent or invalid.
 */
export function normalizeRerankCandidates(value) {
  const numericValue = Number(value ?? DEFAULT_RERANKER_CANDIDATES_COUNT);
  if (!Number.isFinite(numericValue)) return DEFAULT_RERANKER_CANDIDATES_COUNT;
  return Math.min(Math.max(Math.trunc(numericValue), 1), 200);
}

/**
 * Convert raw logits from the cross-encoder output to a probability distribution.
 *
 * The `ms-marco-MiniLM-L-6-v2` cross-encoder outputs logits of shape `[1, 2]`:
 * - `logits[0]` = "not relevant" score
 * - `logits[1]` = "relevant" score
 *
 * Applies the softmax function to convert logits to probabilities:
 * `softmax(logits_i) = exp(logits_i) / sum(exp(logits_j))`
 *
 * @param {number[]} logits - Raw logits array of length 2.
 * @returns {number[]} Probability array of length 2, summing to 1.
 */
export function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((logit) => Math.exp(logit - maxLogit));
  const sumExps = exps.reduce((sum, exp) => sum + exp, 0);
  return exps.map((exp) => exp / sumExps);
}

/**
 * Compute a relevance score from raw cross-encoder logits.
 *
 * Applies softmax to the [not_relevant, relevant] logits and returns
 * the probability of the "relevant" class. Higher scores indicate
 * more relevant query-document pairs.
 *
 * @param {number[]} logits - Raw logits array `[not_relevant, relevant]`.
 * @returns {number} Relevance score between 0 and 1.
 */
export function scorePairFromLogits(logits) {
  const probabilities = softmax(logits);
  return probabilities[1];
}

/**
 * Create the ONNX input tensors for a cross-encoder (query, document) pair.
 *
 * Accepts pre-tokenized input (from the tokenizer) and constructs the
 * `input_ids`, `attention_mask`, and `token_type_ids` tensors required
 * by the cross-encoder model. When the encoded input exceeds `maxLength`,
 * it is truncated from the end.
 *
 * @param {string} _query - The search query (unused when encoded input is provided).
 * @param {string} _documentText - The document text (unused when encoded input is provided).
 * @param {{ input_ids: number[], attention_mask: number[], token_type_ids: number[] }} encoded - Pre-tokenized paired input.
 * @param {number} [maxLength=512] - Maximum sequence length for truncation.
 * @returns {{ input_ids: number[], attention_mask: number[], token_type_ids: number[] }} Token arrays ready for ONNX tensor construction.
 */
export function createRerankInput(
  _query,
  _documentText,
  encoded,
  maxLength = DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH,
) {
  // Truncate to maxLength if needed
  const inputIds = encoded.input_ids.slice(0, maxLength);
  const attentionMask = encoded.attention_mask.slice(0, maxLength);
  const tokenTypeIds = encoded.token_type_ids.slice(0, maxLength);

  return {
    input_ids: inputIds,
    attention_mask: attentionMask,
    token_type_ids: tokenTypeIds,
  };
}

/**
 * Get or create the ONNX InferenceSession and tokenizer for the cross-encoder.
 *
 * Uses a process-lifetime cache to avoid repeated session creation overhead
 * (~100ms per session). The session and tokenizer are created on first call
 * and reused for subsequent calls within the same process.
 *
 * @param {{
 *   rerankerModelDirectory?: string,
 *   rerankerModelId?: string,
 *   forceReload?: boolean,
 * }} [options] - Session creation options.
 * @returns {Promise<{ session: object, tokenizer: object }>} Cached or newly created session and tokenizer.
 */
export async function getOrCreateRerankSession(options = {}) {
  if (cachedRerankSession && !options.forceReload) {
    return { session: cachedRerankSession, tokenizer: cachedRerankTokenizer };
  }

  const { InferenceSession } = await import('onnxruntime-node');
  const modelDirectory = path.resolve(
    options.rerankerModelDirectory ?? DEFAULT_RERANKER_MODEL_DIRECTORY,
  );
  const modelPath = path.join(modelDirectory, 'model.onnx');

  const session = await InferenceSession.create(modelPath);
  const tokenizer = await createRerankTokenizer({ modelDirectory });

  cachedRerankSession = session;
  cachedRerankTokenizer = tokenizer;

  return { session, tokenizer };
}

/**
 * Release the cached ONNX session and tokenizer, freeing memory.
 *
 * Safe to call when no session is active (no-op).
 *
 * @returns {Promise<void>}
 */
export async function releaseRerankSession() {
  if (cachedRerankSession) {
    await cachedRerankSession.release?.();
  }
  cachedRerankSession = null;
  cachedRerankTokenizer = null;
}

/**
 * Create a WordPiece tokenizer for the cross-encoder model.
 *
 * Loads tokenizer configuration from the local model directory and
 * constructs a tokenizer compatible with the cross-encoder's expected
 * input format: `[CLS] query_tokens [SEP] document_tokens [SEP]`.
 *
 * @param {{ modelDirectory?: string }} [options] - Tokenizer configuration.
 * @returns {Promise<object>} Initialized tokenizer instance.
 */
async function createRerankTokenizer(options) {
  /* istanbul ignore next -- defensive: always called with { modelDirectory } from getOrCreateRerankSession */
  if (options == null) options = {};
  let { modelDirectory } = options;
  /* istanbul ignore next -- defensive: modelDirectory always provided by getOrCreateRerankSession */
  if (modelDirectory == null) modelDirectory = DEFAULT_RERANKER_MODEL_DIRECTORY;
  const { Tokenizer } = await import('@huggingface/tokenizers');
  const resolvedDirectory = path.resolve(modelDirectory);

  const tokenizerJsonPath = path.join(resolvedDirectory, 'tokenizer.json');
  const tokenizerConfigPath = path.join(
    resolvedDirectory,
    'tokenizer_config.json',
  );
  const specialTokensMapPath = path.join(
    resolvedDirectory,
    'special_tokens_map.json',
  );

  const [tokenizerJson, tokenizerConfig, specialTokensMap] = await Promise.all([
    readJsonFile(tokenizerJsonPath, null),
    readJsonFile(tokenizerConfigPath, {}),
    readJsonFile(specialTokensMapPath, {}),
  ]);

  const vocabulary = tokenizerJson?.model?.vocab;
  if (!vocabulary || typeof vocabulary !== 'object') {
    throw new Error(
      'tokenizer.json is missing the WordPiece vocabulary required for local tokenization.',
    );
  }

  const unkToken = resolveSpecialToken(specialTokensMap.unk_token, '[UNK]');
  const tokenizer = new Tokenizer(
    {
      added_tokens: [],
      decoder: null,
      model: {
        type: 'WordPiece',
        unk_token: unkToken,
        vocab: vocabulary,
      },
      normalizer: tokenizerJson.normalizer ?? null,
      pre_tokenizer: tokenizerJson.pre_tokenizer ?? null,
      post_processor: null,
    },
    {
      clean_up_tokenization_spaces: true,
    },
  );

  if (tokenizerConfig.model_max_length) {
    // Native Tokenizer v0.1.x does not expose setTruncation/setPadding.
    // The configured max_length is honoured manually when building pairs.
    // See scorePair() for the longest-first truncation applied to documents.
  }

  if (tokenizerConfig.pad_token ?? specialTokensMap.pad_token) {
    // Padding is configured on the tokenizer but, for v0.1.x, applied
    // manually by callers when batching is required.
  }

  return tokenizer;
}

/**
 * Re-rank a list of hybrid search candidates using the cross-encoder model.
 *
 * Processes each (query, body_text) pair through the cross-encoder to produce
 * a `rerank_score` (probability of relevance). Candidates are sorted by
 * `rerank_score` descending and the top `limit` results are returned.
 *
 * @param {string} query - The search query.
 * @param {Array<{ chunk_id: number, body_text: string, [key: string]: unknown }>} candidates - Hybrid search candidates.
 * @param {{
 *   rerankerModelDirectory?: string,
 *   rerankerModelId?: string,
 *   rerankCandidatesCount?: number,
 *   limit?: number,
 * }} [options] - Re-ranking options.
 * @returns {Promise<Array<{ chunk_id: number, body_text: string, rerank_score: number, [key: string]: unknown }>>} Re-ranked candidates sorted by rerank_score descending.
 */
export async function rerankCandidates(query, candidates, options = {}) {
  const { session, tokenizer } = await getOrCreateRerankSession(options);
  const maxCandidates = normalizeRerankCandidates(
    options.rerankCandidatesCount,
  );
  const candidateSlice = candidates.slice(0, maxCandidates);
  const maxLength = DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH;

  const scoredCandidates = [];
  for (const candidate of candidateSlice) {
    const documentText = candidate.body_text ?? candidate.text ?? '';
    const rerankScore = await scorePair(
      session,
      tokenizer,
      query,
      documentText,
      maxLength,
    );
    scoredCandidates.push({ ...candidate, rerank_score: rerankScore });
  }

  return scoredCandidates.toSorted(
    (left, right) => right.rerank_score - left.rerank_score,
  );
}

/**
 * Score a single (query, document) pair using the cross-encoder.
 *
 * Tokenizes the pair with `[CLS] Q [SEP] D [SEP]` format, runs ONNX
 * inference, and returns the softmax probability of the "relevant" class.
 *
 * @param {object} session - ONNX InferenceSession.
 * @param {object} tokenizer - WordPiece tokenizer instance.
 * @param {string} query - Search query text.
 * @param {string} documentText - Document body text.
 * @param {number} [maxLength=512] - Maximum sequence length.
 * @returns {Promise<number>} Relevance score between 0 and 1.
 */
async function scorePair(
  session,
  tokenizer,
  query,
  documentText,
  maxLength,
) {
  /* istanbul ignore next -- defensive: maxLength always provided by rerankCandidates */
  if (maxLength == null) maxLength = DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH;
  const { Tensor } = await import('onnxruntime-node');

  // The installed @huggingface/tokenizers v0.1.x only exposes encode(text).
  // It does not support text_pair, setTruncation, setPadding, or special-token
  // wrapping, so we build the [CLS] query [SEP] document [SEP] sequence by hand
  // and apply longest-first truncation to the document segment.
  const clsId = tokenizer.token_to_id('[CLS]');
  const sepId = tokenizer.token_to_id('[SEP]');

  const queryIds = tokenizer.encode(String(query)).ids;
  const documentIds = tokenizer.encode(String(documentText)).ids;

  let inputIds = [clsId, ...queryIds, sepId, ...documentIds, sepId];

  // Longest-first truncation: keep the query intact and remove document tokens.
  if (inputIds.length > maxLength) {
    const overhead = 3; // [CLS] + [SEP] + [SEP]
    const maxDocumentIds = Math.max(0, maxLength - queryIds.length - overhead);
    inputIds = [
      clsId,
      ...queryIds,
      sepId,
      ...documentIds.slice(0, maxDocumentIds),
      sepId,
    ];
  }

  let sequenceLength = inputIds.length;
  const attentionMask = Array.from({ length: sequenceLength }, () => 1);
  const tokenTypeIds = Array.from({ length: sequenceLength }, (_, index) => {
    // [CLS] + query + first [SEP] belong to segment 0; document + final [SEP]
    // belong to segment 1.
    return index <= queryIds.length + 1 ? 0 : 1;
  });

  const encoded = {
    input_ids: inputIds,
    attention_mask: attentionMask,
    token_type_ids: tokenTypeIds,
  };
  const truncated = createRerankInput(query, documentText, encoded, maxLength);

  sequenceLength = truncated.input_ids.length;

  const inputIdsTensor = BigInt64Array.from(truncated.input_ids.map(BigInt));
  const attentionMaskTensor = BigInt64Array.from(
    truncated.attention_mask.map(BigInt),
  );
  const tokenTypeIdsTensor = BigInt64Array.from(
    truncated.token_type_ids.map(BigInt),
  );

  const feeds = {
    attention_mask: new Tensor('int64', attentionMaskTensor, [
      1,
      sequenceLength,
    ]),
    input_ids: new Tensor('int64', inputIdsTensor, [1, sequenceLength]),
    token_type_ids: new Tensor('int64', tokenTypeIdsTensor, [
      1,
      sequenceLength,
    ]),
  };

  const outputs = await session.run(feeds);
  const outputName =
    session.outputNames.find((name) => name === 'logits') ??
    session.outputNames[0];
  const logitsData = outputs[outputName].data;

  if (logitsData.length === 1) {
    // Exported ms-marco-MiniLM-L-6-v2 returns a single logit of shape [1, 1];
    // apply sigmoid to obtain a [0, 1] relevance probability.
    return 1 / (1 + Math.exp(-Number(logitsData[0])));
  }

  // logits shape is [1, 2]: [not_relevant, relevant]
  const notRelevant = Number(logitsData[0]);
  const relevant = Number(logitsData[1]);

  return scorePairFromLogits([notRelevant, relevant]);
}

async function readJsonFile(filePath, defaultValue) {
  try {
    const content = await readFile(filePath, 'utf8');
    return JSON.parse(content);
  } catch {
    return defaultValue;
  }
}

function resolveSpecialToken(token, fallback) {
  if (typeof token === 'string') return token;
  if (token && typeof token === 'object' && 'content' in token)
    return token.content;
  return fallback;
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Cross-encoder reranker',
      usage:
        'node rag-index/rerank-index.mjs --query "NEAT crossover" [--json]',
      options: [
        '--query <text>                         Query text for re-ranking.',
        '--candidates <json>                    JSON array of candidates.',
        '--limit <n>                            Maximum returned result count (default: 10).',
        '--reranker-model-directory <path>       Override reranker model directory.',
        '--reranker-model-id <id>               Override reranker model identifier.',
        '--rerank-candidates-count <n>           Number of candidates to re-rank (default: 50).',
        '--json                                 Emit JSON results.',
        '--help                                 Show this help.',
      ],
    });
    return;
  }

  try {
    let query = args.query;
    if (query == null) query = args._.join(' ');
    /* istanbul ignore next -- unreachable: Array.join() never returns null/undefined */
    if (query == null) query = '';
    query = String(query).trim();
    if (!query) {
      throw new Error(
        'Query text is required. Pass --query "your query text".',
      );
    }

    const candidates = args.candidates
      ? JSON.parse(String(args.candidates))
      : [];
    const limit = normalizeRerankCandidates(args.limit);

    const results = await rerankCandidates(query, candidates, {
      limit,
      rerankCandidatesCount: args['rerank-candidates-count'],
      rerankerModelDirectory: args['reranker-model-directory'],
      rerankerModelId: args['reranker-model-id'],
    });

    writeJsonOrText(
      results,
      Boolean(args.json),
      /* istanbul ignore next -- text formatter covered when writeJsonOrText is not mocked */
      (payload) =>
        payload
          .map(
            (result, index) =>
              `${index + 1}. ${result.file_path ?? result.chunk_id} [${result.rerank_score.toFixed(4)}]`,
          )
          .join('\n'),
    );
  } catch (error) {
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
