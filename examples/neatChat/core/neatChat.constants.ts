/** Supported sequence-builder families for the NEATchat example. */
export const NEATCHAT_DEFAULT_ARCHITECTURE_FAMILY = 'lstm' as const;

/** Families intentionally in-scope for this teaching boundary. */
export const NEATCHAT_SUPPORTED_ARCHITECTURE_FAMILIES = [
  'lstm',
  'gru',
  'narx',
] as const;

/** Default recurrent width used by the live chat seed network. */
export const NEATCHAT_DEFAULT_RECURRENT_BLOCK_SIZE = 24;

/** Default NARX input-memory shelf depth. */
export const NEATCHAT_DEFAULT_NARX_INPUT_MEMORY = 3;

/** Default NARX output-memory shelf depth. */
export const NEATCHAT_DEFAULT_NARX_OUTPUT_MEMORY = 2;

/** Default retained-vocabulary cap used in pretraining previews. */
export const NEATCHAT_DEFAULT_TOP_WORD_LIMIT = 3000;

/** Recommended top-word cap experiment range. */
export const NEATCHAT_RECOMMENDED_TOP_WORD_LIMIT_RANGE = [300, 5000] as const;

/** Default token window for prompt or reply slices. */
export const NEATCHAT_DEFAULT_CONTEXT_WINDOW_TOKEN_COUNT = 24;

/** Default chunk size used while counting pasted-corpus terms. */
export const NEATCHAT_DEFAULT_CHUNK_TOKEN_COUNT = 256;

/** Default bootstrap vocabulary cap for empty-session warm starts. */
export const NEATCHAT_DEFAULT_SESSION_BOOTSTRAP_TOP_WORD_LIMIT = 512;

/** Stable unknown-token marker. */
export const NEATCHAT_DEFAULT_UNKNOWN_TOKEN = 'UNK' as const;

/** Reserved control tokens at stable vocabulary indices. */
export const NEATCHAT_SPECIAL_TOKENS = [
  NEATCHAT_DEFAULT_UNKNOWN_TOKEN,
  'BOS',
  'EOS',
  'TURN_BREAK',
] as const;

/** A/B variants exposed by the Step 6 comparison surface. */
export const NEATCHAT_AB_VARIANTS = ['blank-start', 'preseeded'] as const;

/** Lightweight metrics exposed in the Step 6 comparison surface. */
export const NEATCHAT_LIGHTWEIGHT_METRICS = [
  'held-out-next-token-accuracy',
  'repetition-rate',
  'response-length-stability',
] as const;

/** Existing visualizer example reused by NEATchat. */
export const NEATCHAT_VISUALIZER_EXAMPLE_ID = 'flappy_bird' as const;

/** Flappy host ownership module reused by NEATchat. */
export const NEATCHAT_VISUALIZER_OWNER_MODULE_PATH =
  'examples/flappy_bird/browser-entry/host/host.ts';

/** Flappy frame resolver module reused by NEATchat. */
export const NEATCHAT_VISUALIZER_FRAME_RESOLVER_MODULE_PATH =
  'examples/flappy_bird/browser-entry/network-view/network-view.ts';

/** Flappy draw module reused by NEATchat. */
export const NEATCHAT_VISUALIZER_DRAW_MODULE_PATH =
  'examples/flappy_bird/browser-entry/visualization/visualization.draw.service.ts';

/** Docs/examples publication category used by NEATchat. */
export const NEATCHAT_PUBLISHED_EXAMPLES_CATEGORY = 'flagship' as const;

/** Public browser entrypoint path for NEATchat. */
export const NEATCHAT_PUBLIC_BROWSER_ENTRYPOINT_MODULE_PATH =
  'examples/neatChat/browser-entry.ts' as const;

/** Public browser host page path for NEATchat. */
export const NEATCHAT_PUBLIC_BROWSER_HOST_PAGE_PATH =
  'examples/neatChat/index.html' as const;

/** Docs publication output path for the browser page. */
export const NEATCHAT_PUBLISHED_DOCS_EXAMPLE_PATH =
  'docs/examples/neatChat/index.html' as const;

/** Max retained live-session vocabulary terms for chat interaction. */
export const NEATCHAT_LIVE_CHAT_MAX_VOCAB_TERMS =
  NEATCHAT_DEFAULT_TOP_WORD_LIMIT;

/** Online-learning SGD iterations per live exchange. */
export const NEATCHAT_ONLINE_LEARNING_ITERATIONS = 4;

/** Online-learning SGD rate per live exchange. */
export const NEATCHAT_ONLINE_LEARNING_RATE = 0.06;

/** Online-learning SGD momentum per live exchange. */
export const NEATCHAT_ONLINE_LEARNING_MOMENTUM = 0.12;

/** Corpus-stream warm-up iteration count during seeding. */
export const NEATCHAT_SEED_STREAM_TRAINING_ITERATIONS = 2;

/** Line-pair reinforcement iteration count during seeding. */
export const NEATCHAT_SEED_LINE_TRAINING_ITERATIONS = 1;

/** Trigram sequence reinforcement iteration count during seeding. */
export const NEATCHAT_SEED_TRIGRAM_TRAINING_ITERATIONS = 1;

/** Corpus-seeding training rate. */
export const NEATCHAT_SEED_TRAINING_RATE = 0.035;

/** Corpus-seeding momentum. */
export const NEATCHAT_SEED_TRAINING_MOMENTUM = 0.1;

/** Safety cap for conversation lines consumed by session seeding. */
export const NEATCHAT_MAX_SEED_CONVERSATION_LINES = 400;

/** Default held-out validation line count. */
export const NEATCHAT_DEFAULT_VALIDATION_LINE_COUNT = 6;

/** Default retained-term cap used by A/B sessions. */
export const NEATCHAT_AB_DEFAULT_VOCAB_LIMIT = 180;

/** Default recurrent width used by A/B sessions. */
export const NEATCHAT_AB_DEFAULT_RECURRENT_BLOCK_SIZE = 12;

/** Maximum seed lines consumed in one A/B pass. */
export const NEATCHAT_AB_MAX_SEED_CONVERSATION_LINES = 48;

/** Maximum warm-start lines consumed in one A/B pass. */
export const NEATCHAT_AB_MAX_WARM_START_LINES = 16;

/** Maximum warm-start training cases consumed in one A/B pass. */
export const NEATCHAT_AB_MAX_WARM_START_CASES = 96;

/** Maximum held-out validation lines consumed in one A/B pass. */
export const NEATCHAT_AB_MAX_VALIDATION_CONVERSATION_LINES = 12;

/** Maximum generated response tokens per exchange. */
export const NEATCHAT_MAX_RESPONSE_TOKENS = 10;

/** Score-division factor used by repetition-penalty decoding. */
export const NEATCHAT_REPETITION_PENALTY_FACTOR = 1.8;

/** Recent-token window length for repetition penalty checks. */
export const NEATCHAT_REPETITION_WINDOW_SIZE = 5;

/** Strong penalty factor for single-word responses to encourage diversity. */
export const NEATCHAT_SINGLE_WORD_PENALTY_FACTOR = 3.5;

/** Minimum response length (tokens) before allowing termination. Discourages stub replies. */
export const NEATCHAT_MINIMUM_RESPONSE_LENGTH = 2;

/** Stable special-token vocabulary indices. */
export const NEATCHAT_SPECIAL_TOKEN_INDICES = {
  UNK: 0,
  BOS: 1,
  EOS: 2,
  TURN_BREAK: 3,
} as const;

/** Ordered corpus-report fields surfaced before pretraining. */
export const NEATCHAT_CORPUS_REPORT_FIELDS = [
  {
    key: 'characterCount',
    label: 'Character count',
    description: 'Total pasted characters before tokenization.',
  },
  {
    key: 'tokenCount',
    label: 'Total tokens',
    description: 'Token count before the top-word cap is applied.',
  },
  {
    key: 'uniqueTermCount',
    label: 'Unique terms',
    description: 'Distinct normalized terms observed in the pasted corpus.',
  },
  {
    key: 'retainedTermCount',
    label: 'Retained terms',
    description: 'Terms kept after the top-word cap and special-token reserve.',
  },
  {
    key: 'retainedTokenCoveragePercent',
    label: 'Retained token coverage',
    description: 'Share of pasted tokens covered by the retained vocabulary.',
  },
] as const;

/** Delivery progression rows shown on the browser contract preview. */
export const NEATCHAT_DELIVERY_PROGRESS = [
  {
    label: 'Step 1 - Define the example contract',
    status: 'available',
    summary:
      'The example now exposes a Node contract preview and a browser-hosted flagship page that shows the same contract, chat preview shell, and publication seam.',
  },
  {
    label: 'Step 2 - Constrain vocabulary and sequence length',
    status: 'available',
    summary:
      'Keep the token window tiny, make topWordLimit user-tunable, surface a shared runtime estimate, and preserve one stable UNK path across Node and browser previews.',
  },
  {
    label: 'Step 3 - Optional copy-paste pretraining',
    status: 'available',
    summary:
      'Accept pasted corpora, process them in bounded chunks, and preseed one retained-vocabulary slice without making pretraining mandatory.',
  },
  {
    label: 'Step 4 - Add corpus report and controls',
    status: 'available',
    summary:
      'Show character, token, unique-term, retained-term, and retained-coverage numbers on the browser and CLI surfaces before pretraining begins.',
  },
  {
    label: 'Step 5 - Add online-learning loop',
    status: 'available',
    summary:
      'Apply one narrow supervised update after each exchange so users can observe measurable adaptation in short conversations.',
  },
  {
    label: 'Step 6 - Add A/B interaction and lightweight evaluation',
    status: 'available',
    summary:
      'Run one-session blank-start versus preseeded prompt comparisons while tracking held-out next-token accuracy, repetition rate, and response-length stability.',
  },
  {
    label: 'Step 7 - Document boundaries honestly',
    status: 'planned',
    summary:
      'Keep the docs explicit about toy-scale sequence learning, token-stream limits, and the runtime cost of larger vocabulary caps.',
  },
  {
    label: 'Step 8 - Plan later follow-ons separately',
    status: 'planned',
    summary:
      'Leave persistence, worker execution, hybrid training, and NEATchat-specific visualizer polish to their own follow-on plans.',
  },
] as const;
