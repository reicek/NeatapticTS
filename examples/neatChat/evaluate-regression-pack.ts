import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

import {
  importNeatChatSession,
  importNeatChatSessionV2,
  splitNeatChatSeedAndValidationLines,
} from './index.ts';
import { evaluateNeatChatSessionMetrics } from './core/neatChat.ab.services.ts';
import { DEFAULT_NEATCHAT_PRETRAINING_CORPUS_LINES } from './default-pretraining-corpus.ts';
import { NEATCHAT_SAMPLE_CONVERSATION_LINES } from './sample-conversation.ts';

const DEFAULT_REGRESSION_PACK_HELD_OUT_LINE_COUNT = 16;
const DEFAULT_SNAPSHOT_MODULE_PATH =
  'examples/neatChat/default-pretrained-session-snapshot.ts';
const DEFAULT_SNAPSHOT_EXPORT_NAME =
  'DEFAULT_NEATCHAT_PRETRAINED_SESSION_SNAPSHOT';

type RegressionPackEvaluationConfig = {
  readonly snapshotModulePath: string;
  readonly snapshotExportName: string;
  readonly snapshotVersion: 1 | 2;
  readonly prompt: string;
  readonly heldOutLineCount: number;
};

async function main(): Promise<void> {
  const evaluationConfig = resolveEvaluationConfig(process.argv.slice(2));
  const combinedConversationLines = [
    ...NEATCHAT_SAMPLE_CONVERSATION_LINES,
    ...DEFAULT_NEATCHAT_PRETRAINING_CORPUS_LINES,
  ];
  const regressionPack = splitNeatChatSeedAndValidationLines(
    combinedConversationLines,
    evaluationConfig.heldOutLineCount,
  );
  const snapshotModule = await import(
    pathToFileURL(resolve(process.cwd(), evaluationConfig.snapshotModulePath))
      .href
  );
  const exportedSnapshot = snapshotModule[evaluationConfig.snapshotExportName];

  if (exportedSnapshot === undefined) {
    throw new Error(
      `Snapshot export ${evaluationConfig.snapshotExportName} was not found in ${evaluationConfig.snapshotModulePath}.`,
    );
  }

  const session =
    evaluationConfig.snapshotVersion === 2
      ? importNeatChatSessionV2(exportedSnapshot)
      : importNeatChatSession(exportedSnapshot);
  const evaluation = evaluateNeatChatSessionMetrics(
    session,
    evaluationConfig.prompt,
    regressionPack.validationConversationLines,
  );

  console.log(
    JSON.stringify(
      {
        snapshotModulePath: evaluationConfig.snapshotModulePath,
        snapshotExportName: evaluationConfig.snapshotExportName,
        snapshotVersion: evaluationConfig.snapshotVersion,
        prompt: evaluationConfig.prompt,
        response: evaluation.response,
        responseTokens: evaluation.responseTokens,
        metrics: evaluation.metrics,
        vocabularySize: session.vocabulary.size,
        retainedTermCount: session.vocabulary.size - 4,
        seedConversationLineCount: regressionPack.seedConversationLines.length,
        validationConversationLineCount:
          regressionPack.validationConversationLines.length,
      },
      null,
      2,
    ),
  );
}

function resolveEvaluationConfig(
  commandLineArguments: readonly string[],
): RegressionPackEvaluationConfig {
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
  const rawSnapshotVersion = argumentMap.get('snapshotVersion') ?? '1';
  const snapshotVersion = Number(rawSnapshotVersion);

  if (snapshotVersion !== 1 && snapshotVersion !== 2) {
    throw new Error('snapshotVersion must be 1 or 2.');
  }

  return {
    snapshotModulePath:
      argumentMap.get('snapshotModulePath') ?? DEFAULT_SNAPSHOT_MODULE_PATH,
    snapshotExportName:
      argumentMap.get('snapshotExportName') ?? DEFAULT_SNAPSHOT_EXPORT_NAME,
    snapshotVersion,
    prompt: argumentMap.get('prompt') ?? NEATCHAT_SAMPLE_CONVERSATION_LINES[0]!,
    heldOutLineCount: resolvePositiveInteger(
      argumentMap.get('heldOutLineCount'),
      'heldOutLineCount',
      DEFAULT_REGRESSION_PACK_HELD_OUT_LINE_COUNT,
    ),
  };
}

function resolvePositiveInteger(
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

void main();
