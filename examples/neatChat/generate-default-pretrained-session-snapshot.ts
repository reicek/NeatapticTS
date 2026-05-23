import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { NEATCHAT_MAX_SEED_CONVERSATION_LINES } from './core/neatChat.constants.ts';
import {
  createNeatChatPretrainingPreview,
  createNeatChatSession,
  exportNeatChatSession,
  pretrainNeatChatSessionWithConversationLines,
  splitNeatChatSeedAndValidationLines,
  tokenizeNeatChatText,
} from './index.ts';
import { DEFAULT_NEATCHAT_PRETRAINING_CORPUS_LINES } from './default-pretraining-corpus.ts';
import { NEATCHAT_SAMPLE_CONVERSATION_LINES } from './sample-conversation.ts';

const DEFAULT_SNAPSHOT_TOP_WORD_LIMIT = 512;
const DEFAULT_SNAPSHOT_CONTEXT_WINDOW_TOKEN_COUNT = 64;
const DEFAULT_SNAPSHOT_EXTRA_REINFORCEMENT_PASSES = 3;
const DEFAULT_SNAPSHOT_MAX_SOURCE_LINES = 0;
const DEFAULT_SNAPSHOT_MAX_CASES_PER_PHASE = 0;
const DEFAULT_SNAPSHOT_VALIDATION_LINE_COUNT = 16;
const DEFAULT_SNAPSHOT_OUTPUT_FILE_NAME =
  'default-pretrained-session-snapshot.ts';
const DEFAULT_SNAPSHOT_EXPORT_NAME =
  'DEFAULT_NEATCHAT_PRETRAINED_SESSION_SNAPSHOT';
const DEFAULT_SNAPSHOT_GENERATION_SEED = 20_260_521;

type SnapshotGenerationConfig = {
  readonly topWordLimit: number;
  readonly contextWindowTokenCount: number;
  readonly extraReinforcementPasses: number;
  readonly maxSourceLines: number;
  readonly maxCasesPerPhase: number;
  readonly outputFileName: string;
  readonly exportName: string;
  readonly validationLineCount: number;
  readonly generationSeed: number;
  readonly progressOutputEnabled: boolean;
};

type SnapshotGenerationProgressEvent =
  | {
      readonly event: 'start';
      readonly sourceLineCount: number;
      readonly chunkCount: number;
      readonly extraReinforcementPasses: number;
      readonly topWordLimit: number;
      readonly maxCasesPerPhase: number;
    }
  | {
      readonly event: 'chunkDone';
      readonly pass: number;
      readonly chunkIndex: number;
      readonly chunkLineCount: number;
      readonly originalChunkLineCount: number;
      readonly phaseCaseCount: number;
      readonly processedLineCount: number;
      readonly totalLineCount: number;
      readonly elapsedMs: number;
    }
  | {
      readonly event: 'passDone';
      readonly pass: number;
      readonly elapsedMs: number;
    }
  | {
      readonly event: 'generationDone';
      readonly elapsedMs: number;
    };

type SeedTrainingPhasePlan = {
  readonly selectedConversationLines: readonly string[];
  readonly originalChunkLineCount: number;
  readonly phaseCaseCount: number;
};

/**
 * Generates the shipped default NEATchat session snapshot from a richer offline corpus.
 *
 * The snapshot keeps the browser demo responsive while avoiding a cold start.
 * It combines the original sample conversation with a larger topical corpus,
 * builds a bounded retained vocabulary, runs several deterministic reinforcement
 * passes, and emits a typed TypeScript module for direct browser bundling.
 */
async function main(): Promise<void> {
  const generationConfig = resolveGenerationConfig(process.argv.slice(2));
  const sourceConversationLines =
    collectSourceConversationLines(generationConfig);
  const progressStartedAtMs = Date.now();
  const corpusText = sourceConversationLines.join('\n');
  const preview = createNeatChatPretrainingPreview({
    corpusText,
    topWordLimit: generationConfig.topWordLimit,
  });

  const session = withSeededMathRandom(generationConfig.generationSeed, () => {
    const sourceConversationLineChunks = chunkConversationLines(
      sourceConversationLines,
      NEATCHAT_MAX_SEED_CONVERSATION_LINES,
    );
    emitProgressLine(generationConfig, {
      event: 'start',
      sourceLineCount: sourceConversationLines.length,
      chunkCount: sourceConversationLineChunks.length,
      extraReinforcementPasses: generationConfig.extraReinforcementPasses,
      topWordLimit: generationConfig.topWordLimit,
      maxCasesPerPhase: generationConfig.maxCasesPerPhase,
    });
    const initialSeedConversationPhasePlan = resolveSeedTrainingPhasePlan(
      sourceConversationLineChunks[0] ?? [],
      generationConfig.maxCasesPerPhase,
    );
    let generatedSession = createNeatChatSession({
      corpusRetainedTerms: preview.retainedTerms,
      seedConversationLines:
        initialSeedConversationPhasePlan.selectedConversationLines,
      liveChatVocabLimit: generationConfig.topWordLimit,
      contextWindowTokenCount: generationConfig.contextWindowTokenCount,
    });
    let initialPassProcessedLineCount =
      initialSeedConversationPhasePlan.selectedConversationLines.length;
    emitProgressLine(
      generationConfig,
      createChunkDoneProgressEvent({
        passIndex: 0,
        chunkIndex: 0,
        chunkLineCount:
          initialSeedConversationPhasePlan.selectedConversationLines.length,
        originalChunkLineCount:
          initialSeedConversationPhasePlan.originalChunkLineCount,
        phaseCaseCount: initialSeedConversationPhasePlan.phaseCaseCount,
        processedLineCount: initialPassProcessedLineCount,
        totalLineCount: sourceConversationLines.length,
        progressStartedAtMs,
      }),
    );

    for (const [
      additionalChunkOffset,
      additionalSeedChunk,
    ] of sourceConversationLineChunks.slice(1).entries()) {
      const additionalSeedConversationPhasePlan = resolveSeedTrainingPhasePlan(
        additionalSeedChunk,
        generationConfig.maxCasesPerPhase,
      );
      generatedSession = pretrainNeatChatSessionWithConversationLines(
        generatedSession,
        additionalSeedConversationPhasePlan.selectedConversationLines,
      );
      initialPassProcessedLineCount +=
        additionalSeedConversationPhasePlan.selectedConversationLines.length;
      emitProgressLine(
        generationConfig,
        createChunkDoneProgressEvent({
          passIndex: 0,
          chunkIndex: additionalChunkOffset + 1,
          chunkLineCount:
            additionalSeedConversationPhasePlan.selectedConversationLines
              .length,
          originalChunkLineCount:
            additionalSeedConversationPhasePlan.originalChunkLineCount,
          phaseCaseCount: additionalSeedConversationPhasePlan.phaseCaseCount,
          processedLineCount: initialPassProcessedLineCount,
          totalLineCount: sourceConversationLines.length,
          progressStartedAtMs,
        }),
      );
    }

    for (
      let reinforcementPassIndex = 0;
      reinforcementPassIndex < generationConfig.extraReinforcementPasses;
      reinforcementPassIndex++
    ) {
      const progressPassIndex = reinforcementPassIndex + 1;
      let passProcessedLineCount = 0;

      for (const [
        sourceConversationChunkIndex,
        sourceConversationLineChunk,
      ] of sourceConversationLineChunks.entries()) {
        const reinforcementPhasePlan = resolveSeedTrainingPhasePlan(
          sourceConversationLineChunk,
          generationConfig.maxCasesPerPhase,
        );
        generatedSession = pretrainNeatChatSessionWithConversationLines(
          generatedSession,
          reinforcementPhasePlan.selectedConversationLines,
        );
        passProcessedLineCount +=
          reinforcementPhasePlan.selectedConversationLines.length;
        emitProgressLine(
          generationConfig,
          createChunkDoneProgressEvent({
            passIndex: progressPassIndex,
            chunkIndex: sourceConversationChunkIndex,
            chunkLineCount:
              reinforcementPhasePlan.selectedConversationLines.length,
            originalChunkLineCount:
              reinforcementPhasePlan.originalChunkLineCount,
            phaseCaseCount: reinforcementPhasePlan.phaseCaseCount,
            processedLineCount: passProcessedLineCount,
            totalLineCount: sourceConversationLines.length,
            progressStartedAtMs,
          }),
        );
      }

      emitProgressLine(generationConfig, {
        event: 'passDone',
        pass: progressPassIndex,
        elapsedMs: elapsedMsSince(progressStartedAtMs),
      });
    }

    return generatedSession;
  });

  emitProgressLine(generationConfig, {
    event: 'generationDone',
    elapsedMs: elapsedMsSince(progressStartedAtMs),
  });

  const exportedSnapshot = exportNeatChatSession(session);
  const outputPath = resolve(
    process.cwd(),
    'examples',
    'neatChat',
    generationConfig.outputFileName,
  );
  const serializedSnapshotJson = JSON.stringify(exportedSnapshot);
  const generatedModuleSource = [
    "import type { NeatChatSessionSnapshot } from './index';",
    '',
    '/**',
    ' * Generated default NEATchat session snapshot for the browser demo.',
    ' *',
    ' * Regenerate with:',
    ' * `npm run generate:neat-chat-default-snapshot`',
    ' */',
    `const ${generationConfig.exportName}_JSON = ${JSON.stringify(serializedSnapshotJson)};`,
    '',
    `export const ${generationConfig.exportName}: NeatChatSessionSnapshot = JSON.parse(`,
    `  ${generationConfig.exportName}_JSON,`,
    ') as NeatChatSessionSnapshot;',
    '',
  ].join('\n');

  await writeFile(outputPath, generatedModuleSource, 'utf8');

  console.log(
    JSON.stringify(
      {
        outputPath,
        sourceLineCount: sourceConversationLines.length,
        validationLineCount: generationConfig.validationLineCount,
        retainedTermCount: preview.retainedTermCount,
        seededTokenPairCount: session.seededTokenPairCount,
        contextWindowTokenCount: session.contextWindowTokenCount,
        topWordLimit: generationConfig.topWordLimit,
        extraReinforcementPasses: generationConfig.extraReinforcementPasses,
        maxCasesPerPhase: generationConfig.maxCasesPerPhase,
        exportName: generationConfig.exportName,
        generationSeed: generationConfig.generationSeed,
      },
      null,
      2,
    ),
  );
}

function collectSourceConversationLines(
  generationConfig: SnapshotGenerationConfig,
): readonly string[] {
  const combinedConversationLines = [
    ...NEATCHAT_SAMPLE_CONVERSATION_LINES,
    ...DEFAULT_NEATCHAT_PRETRAINING_CORPUS_LINES,
  ];
  const cappedConversationLines =
    generationConfig.maxSourceLines <= 0
      ? combinedConversationLines
      : combinedConversationLines.slice(0, generationConfig.maxSourceLines);

  if (generationConfig.validationLineCount > 0) {
    return splitNeatChatSeedAndValidationLines(
      cappedConversationLines,
      generationConfig.validationLineCount,
    ).seedConversationLines;
  }

  return cappedConversationLines;
}

function chunkConversationLines(
  conversationLines: readonly string[],
  chunkSize: number,
): readonly (readonly string[])[] {
  if (conversationLines.length === 0) {
    return [[]];
  }

  const conversationLineChunks: string[][] = [];

  for (
    let conversationStartIndex = 0;
    conversationStartIndex < conversationLines.length;
    conversationStartIndex += chunkSize
  ) {
    conversationLineChunks.push(
      conversationLines.slice(
        conversationStartIndex,
        conversationStartIndex + chunkSize,
      ),
    );
  }

  return conversationLineChunks;
}

function resolveGenerationConfig(
  commandLineArguments: readonly string[],
): SnapshotGenerationConfig {
  const argumentMap = new Map(
    commandLineArguments.map((commandLineArgument) => {
      const trimmedArgument = commandLineArgument.trim();

      if (!trimmedArgument.startsWith('--')) {
        throw new Error(
          `Unsupported argument format: ${trimmedArgument}. Use --key=value.`,
        );
      }

      const separatorIndex = trimmedArgument.indexOf('=');

      if (separatorIndex < 0) {
        throw new Error(
          `Missing "=" in argument: ${trimmedArgument}. Use --key=value.`,
        );
      }

      return [
        trimmedArgument.slice(2, separatorIndex),
        trimmedArgument.slice(separatorIndex + 1),
      ] as const;
    }),
  );

  const exportName =
    argumentMap.get('exportName') ?? DEFAULT_SNAPSHOT_EXPORT_NAME;

  if (!/^[A-Z][A-Z0-9_]*$/.test(exportName)) {
    throw new Error(
      `Invalid exportName: ${exportName}. Use SCREAMING_SNAKE_CASE.`,
    );
  }

  return {
    topWordLimit: resolveRequiredPositiveInteger(
      argumentMap.get('topWordLimit'),
      'topWordLimit',
      DEFAULT_SNAPSHOT_TOP_WORD_LIMIT,
    ),
    contextWindowTokenCount: resolveRequiredPositiveInteger(
      argumentMap.get('contextWindowTokenCount'),
      'contextWindowTokenCount',
      DEFAULT_SNAPSHOT_CONTEXT_WINDOW_TOKEN_COUNT,
    ),
    extraReinforcementPasses: resolveRequiredNonNegativeInteger(
      argumentMap.get('extraReinforcementPasses'),
      'extraReinforcementPasses',
      DEFAULT_SNAPSHOT_EXTRA_REINFORCEMENT_PASSES,
    ),
    maxSourceLines: resolveRequiredNonNegativeInteger(
      argumentMap.get('maxSourceLines'),
      'maxSourceLines',
      DEFAULT_SNAPSHOT_MAX_SOURCE_LINES,
    ),
    maxCasesPerPhase: resolveRequiredNonNegativeInteger(
      argumentMap.get('maxCasesPerPhase'),
      'maxCasesPerPhase',
      DEFAULT_SNAPSHOT_MAX_CASES_PER_PHASE,
    ),
    outputFileName:
      argumentMap.get('outputFileName') ?? DEFAULT_SNAPSHOT_OUTPUT_FILE_NAME,
    exportName,
    validationLineCount: resolveRequiredNonNegativeInteger(
      argumentMap.get('validationLineCount'),
      'validationLineCount',
      DEFAULT_SNAPSHOT_VALIDATION_LINE_COUNT,
    ),
    generationSeed: resolveRequiredPositiveInteger(
      argumentMap.get('generationSeed'),
      'generationSeed',
      DEFAULT_SNAPSHOT_GENERATION_SEED,
    ),
    progressOutputEnabled: resolveOptionalBoolean(
      argumentMap.get('progress'),
      'progress',
      commandLineArguments.length > 0,
    ),
  };
}

function createChunkDoneProgressEvent(input: {
  readonly passIndex: number;
  readonly chunkIndex: number;
  readonly chunkLineCount: number;
  readonly originalChunkLineCount: number;
  readonly phaseCaseCount: number;
  readonly processedLineCount: number;
  readonly totalLineCount: number;
  readonly progressStartedAtMs: number;
}): SnapshotGenerationProgressEvent {
  return {
    event: 'chunkDone',
    pass: input.passIndex,
    chunkIndex: input.chunkIndex,
    chunkLineCount: input.chunkLineCount,
    originalChunkLineCount: input.originalChunkLineCount,
    phaseCaseCount: input.phaseCaseCount,
    processedLineCount: input.processedLineCount,
    totalLineCount: input.totalLineCount,
    elapsedMs: elapsedMsSince(input.progressStartedAtMs),
  };
}

function resolveSeedTrainingPhasePlan(
  seedConversationLines: readonly string[],
  maxCasesPerPhase: number,
): SeedTrainingPhasePlan {
  if (seedConversationLines.length < 2) {
    return {
      selectedConversationLines: seedConversationLines,
      originalChunkLineCount: seedConversationLines.length,
      phaseCaseCount: 0,
    };
  }

  const seedLineTokenCounts = seedConversationLines.map(
    (seedConversationLine) => tokenizeNeatChatText(seedConversationLine).length,
  );
  let cumulativeTokenCount = 0;
  let cumulativeNonEmptyLineCount = 0;
  let cumulativeAdjacentPairCaseCount = 0;
  let previousLineTokenCount = 0;
  let selectedLineCount = 0;
  let selectedCaseCount = 0;
  let minimumEligibleCaseCount = 0;

  for (const [
    lineIndex,
    currentLineTokenCount,
  ] of seedLineTokenCounts.entries()) {
    cumulativeTokenCount += currentLineTokenCount;

    if (currentLineTokenCount > 0) {
      cumulativeNonEmptyLineCount += 1;
    }

    if (lineIndex > 0) {
      cumulativeAdjacentPairCaseCount +=
        previousLineTokenCount + currentLineTokenCount + 2;
      const fullStreamCaseCount =
        1 + cumulativeTokenCount + cumulativeNonEmptyLineCount;
      const phaseCaseCount =
        fullStreamCaseCount + cumulativeAdjacentPairCaseCount * 2;

      if (lineIndex === 1) {
        minimumEligibleCaseCount = phaseCaseCount;
      }

      if (maxCasesPerPhase > 0 && phaseCaseCount > maxCasesPerPhase) {
        break;
      }

      selectedLineCount = lineIndex + 1;
      selectedCaseCount = phaseCaseCount;
    }

    previousLineTokenCount = currentLineTokenCount;
  }

  if (selectedLineCount === 0) {
    throw new Error(
      `maxCasesPerPhase=${maxCasesPerPhase} is too small for deterministic seed training. The smallest two-line prefix for this phase requires ${minimumEligibleCaseCount} cases.`,
    );
  }

  return {
    selectedConversationLines: seedConversationLines.slice(
      0,
      selectedLineCount,
    ),
    originalChunkLineCount: seedConversationLines.length,
    phaseCaseCount: selectedCaseCount,
  };
}

function emitProgressLine(
  generationConfig: SnapshotGenerationConfig,
  progressEvent: SnapshotGenerationProgressEvent,
): void {
  if (!generationConfig.progressOutputEnabled) {
    return;
  }

  process.stderr.write(`${JSON.stringify(progressEvent)}\n`);
}

function elapsedMsSince(startedAtMs: number): number {
  return Date.now() - startedAtMs;
}

function resolveRequiredPositiveInteger(
  rawValue: string | undefined,
  fieldName: string,
  fallbackValue: number,
): number {
  const resolvedValue =
    rawValue === undefined ? fallbackValue : Number(rawValue);

  if (!Number.isInteger(resolvedValue) || resolvedValue <= 0) {
    throw new Error(`${fieldName} must be a positive integer.`);
  }

  return resolvedValue;
}

function resolveRequiredNonNegativeInteger(
  rawValue: string | undefined,
  fieldName: string,
  fallbackValue: number,
): number {
  const resolvedValue =
    rawValue === undefined ? fallbackValue : Number(rawValue);

  if (!Number.isInteger(resolvedValue) || resolvedValue < 0) {
    throw new Error(`${fieldName} must be a non-negative integer.`);
  }

  return resolvedValue;
}

function resolveOptionalBoolean(
  rawValue: string | undefined,
  fieldName: string,
  fallbackValue: boolean,
): boolean {
  if (rawValue === undefined) {
    return fallbackValue;
  }

  switch (rawValue) {
    case 'true':
      return true;
    case 'false':
      return false;
    default:
      throw new Error(`${fieldName} must be true or false.`);
  }
}

function withSeededMathRandom<T>(generationSeed: number, callback: () => T): T {
  const originalRandom = Math.random;
  const seededRandom = createSeededRandom(generationSeed);

  Math.random = () => seededRandom();

  try {
    return callback();
  } finally {
    Math.random = originalRandom;
  }
}

function createSeededRandom(generationSeed: number): () => number {
  let currentState = generationSeed >>> 0;

  if (currentState === 0) {
    currentState = 1;
  }

  return () => {
    currentState = (1_664_525 * currentState + 1_013_904_223) >>> 0;
    return currentState / 4_294_967_296;
  };
}

void main();
