import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import {
  createNeatChatPretrainingPreview,
  createNeatChatSession,
  exportNeatChatSession,
  pretrainNeatChatSessionWithConversationLines,
} from './index.ts';
import { DEFAULT_NEATCHAT_PRETRAINING_CORPUS_LINES } from './default-pretraining-corpus.ts';
import { NEATCHAT_SAMPLE_CONVERSATION_LINES } from './sample-conversation.ts';

const DEFAULT_SNAPSHOT_TOP_WORD_LIMIT = 384;
const DEFAULT_SNAPSHOT_CONTEXT_WINDOW_TOKEN_COUNT = 64;
const DEFAULT_SNAPSHOT_EXTRA_REINFORCEMENT_PASSES = 1;
const DEFAULT_SNAPSHOT_MAX_SOURCE_LINES = 240;
const DEFAULT_SNAPSHOT_OUTPUT_FILE_NAME =
  'default-pretrained-session-snapshot.ts';

/**
 * Generates the shipped default NEATchat session snapshot from a richer offline corpus.
 *
 * The snapshot keeps the browser demo responsive while avoiding a cold start.
 * It combines the original sample conversation with a larger topical corpus,
 * builds a bounded retained vocabulary, runs several deterministic reinforcement
 * passes, and emits a typed TypeScript module for direct browser bundling.
 */
async function main(): Promise<void> {
  const sourceConversationLines = [
    ...NEATCHAT_SAMPLE_CONVERSATION_LINES,
    ...DEFAULT_NEATCHAT_PRETRAINING_CORPUS_LINES,
  ].slice(0, DEFAULT_SNAPSHOT_MAX_SOURCE_LINES);
  const corpusText = sourceConversationLines.join('\n');
  const preview = createNeatChatPretrainingPreview({
    corpusText,
    topWordLimit: DEFAULT_SNAPSHOT_TOP_WORD_LIMIT,
  });

  let session = createNeatChatSession({
    corpusRetainedTerms: preview.retainedTerms,
    seedConversationLines: sourceConversationLines,
    liveChatVocabLimit: DEFAULT_SNAPSHOT_TOP_WORD_LIMIT,
    contextWindowTokenCount: DEFAULT_SNAPSHOT_CONTEXT_WINDOW_TOKEN_COUNT,
  });

  for (
    let reinforcementPassIndex = 0;
    reinforcementPassIndex < DEFAULT_SNAPSHOT_EXTRA_REINFORCEMENT_PASSES;
    reinforcementPassIndex++
  ) {
    session = pretrainNeatChatSessionWithConversationLines(
      session,
      sourceConversationLines,
    );
  }

  const exportedSnapshot = exportNeatChatSession(session);
  const outputPath = resolve(
    process.cwd(),
    'examples',
    'neatChat',
    DEFAULT_SNAPSHOT_OUTPUT_FILE_NAME,
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
    `const DEFAULT_NEATCHAT_PRETRAINED_SESSION_SNAPSHOT_JSON = ${JSON.stringify(serializedSnapshotJson)};`,
    '',
    'export const DEFAULT_NEATCHAT_PRETRAINED_SESSION_SNAPSHOT: NeatChatSessionSnapshot = JSON.parse(',
    '  DEFAULT_NEATCHAT_PRETRAINED_SESSION_SNAPSHOT_JSON,',
    ') as NeatChatSessionSnapshot;',
    '',
  ].join('\n');

  await writeFile(outputPath, generatedModuleSource, 'utf8');

  console.log(
    JSON.stringify(
      {
        outputPath,
        sourceLineCount: sourceConversationLines.length,
        retainedTermCount: preview.retainedTermCount,
        seededTokenPairCount: session.seededTokenPairCount,
        contextWindowTokenCount: session.contextWindowTokenCount,
      },
      null,
      2,
    ),
  );
}

void main();
