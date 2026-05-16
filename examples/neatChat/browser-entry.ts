import {
  createNeatChatAbComparison,
  createNeatChatExampleContract,
  extractNeatChatConversationLines,
  exportNeatChatSession,
  getNeatChatSampleConversationLines,
  importNeatChatSession,
  createNeatChatPretrainingPreview,
  createNeatChatSession,
  pretrainNeatChatSessionWithConversationLines,
  updateNeatChatSessionContextWindowTokenCount,
  runNeatChatExchange,
  estimateNeatChatRuntime,
  type NeatChatAbComparisonResult,
  type NeatChatExampleContract,
  type NeatChatPretrainingPreview,
  type NeatChatRuntimeEstimate,
  type NeatChatSession,
  splitNeatChatSeedAndValidationLines,
} from './index';
import { DEFAULT_NEATCHAT_PRETRAINED_SESSION_SNAPSHOT } from './default-pretrained-session-snapshot';

type BrowserHostContainer = string | HTMLElement;
type NeatChatStart = (container?: BrowserHostContainer) => Promise<void>;
const SAMPLE_PRETRAIN_CHUNK_SIZE = 4;
const CONTEXT_WINDOW_OPTIONS = [
  12, 24, 32, 50, 75, 100, 150, 200, 300,
] as const;
const CONTEXT_WINDOW_EXPLANATION =
  'Controls how many tokens from your message the model processes. Larger windows let the model see more of your input, but may slow inference slightly.';
const DEFAULT_BROWSER_WARM_START_LINE_COUNT = SAMPLE_PRETRAIN_CHUNK_SIZE;
const DEFAULT_SNAPSHOT_FILENAME = 'neatchat-session.snapshot.json';
const NEATCHAT_SPECIAL_TOKEN_COUNT = 4;

declare global {
  interface Window {
    neatChat?: {
      start: NeatChatStart;
    };
    neatChatStart?: NeatChatStart;
  }
}

/** Module-level live-chat session — rebuilt when the user updates the preview. */
let currentSession: NeatChatSession | null = null;
let sampleConversationCursor = 0;

/**
 * Starts the browser-hosted NEATchat contract preview.
 *
 * @param container - Host element or element id.
 * @returns Promise resolved after the preview UI renders.
 */
export async function start(
  container: BrowserHostContainer = 'neat-chat-output',
): Promise<void> {
  const hostElement = resolveHostElement(container);

  hostElement.innerHTML = buildLoadingMarkup();
  await waitForAnimationFrame();

  try {
    const exampleContract = createNeatChatExampleContract();

    renderContractPreview(hostElement, exampleContract);
  } catch (error) {
    hostElement.innerHTML = buildErrorMarkup(
      error instanceof Error ? error.message : String(error),
    );
  }
}

function renderContractPreview(
  hostElement: HTMLElement,
  exampleContract: NeatChatExampleContract,
  topWordLimit = exampleContract.pretraining.defaultTopWordLimit,
  contextWindowTokenCount?: number,
  corpusText = '',
): void {
  sampleConversationCursor = 0;

  const resolvedContextWindowTokenCount =
    contextWindowTokenCount ??
    exampleContract.pretraining.defaultContextWindowTokenCount;

  const runtimeEstimate = estimateNeatChatRuntime({
    topWordLimit,
    contextWindowTokenCount: resolvedContextWindowTokenCount,
  });
  const pretrainingPreview = createNeatChatPretrainingPreview({
    corpusText,
    topWordLimit: runtimeEstimate.topWordLimit,
    chunkTokenCount: exampleContract.pretraining.defaultChunkTokenCount,
  });
  const sampleConversationLines = getNeatChatSampleConversationLines();
  const defaultWarmStartLines = sampleConversationLines.slice(
    0,
    DEFAULT_BROWSER_WARM_START_LINE_COUNT,
  );
  const initialConversationLines = extractNeatChatConversationLines(corpusText);
  const conversationLineSplit = splitNeatChatSeedAndValidationLines(
    initialConversationLines,
  );
  const activeCorpusLines = [...initialConversationLines];
  const activeSeedConversationLines =
    conversationLineSplit.seedConversationLines.length > 0
      ? conversationLineSplit.seedConversationLines
      : defaultWarmStartLines;

  sampleConversationCursor =
    initialConversationLines.length > 0
      ? 0
      : activeSeedConversationLines.length;

  hostElement.innerHTML = buildNeatChatMarkup(
    exampleContract,
    corpusText,
    runtimeEstimate,
    pretrainingPreview,
  );

  // Step 5: Build a new live-chat session from either a shipped base snapshot
  // or from the currently provided corpus text.
  let usedShippedBaseSnapshot = false;
  if (initialConversationLines.length === 0) {
    try {
      currentSession = importNeatChatSession(
        DEFAULT_NEATCHAT_PRETRAINED_SESSION_SNAPSHOT,
      );
      usedShippedBaseSnapshot = true;
    } catch {
      currentSession = createNeatChatSession({
        corpusRetainedTerms: pretrainingPreview.retainedTerms,
        seedConversationLines: activeSeedConversationLines,
        liveChatVocabLimit: runtimeEstimate.topWordLimit,
        contextWindowTokenCount: runtimeEstimate.contextWindowTokenCount,
      });
    }
  } else {
    currentSession = createNeatChatSession({
      corpusRetainedTerms: pretrainingPreview.retainedTerms,
      seedConversationLines: activeSeedConversationLines,
      liveChatVocabLimit: runtimeEstimate.topWordLimit,
      contextWindowTokenCount: runtimeEstimate.contextWindowTokenCount,
    });
  }
  synchronizeSessionSurfaceState(hostElement, currentSession);
  updateSessionStatsDisplay(hostElement, currentSession);

  if (
    initialConversationLines.length === 0 &&
    activeSeedConversationLines.length > 0
  ) {
    if (usedShippedBaseSnapshot) {
      appendSystemMessageToHistory(
        hostElement,
        'Started from the shipped pretrained base snapshot.',
      );
      updateSessionSnapshotStatus(
        hostElement,
        'Shipped pretrained basepoint active. Live chat now uses the bundled snapshot; the corpus preview report only changes when you paste or sample-train lines.',
      );
    } else {
      appendSystemMessageToHistory(
        hostElement,
        `Started from a bundled warm start using ${activeSeedConversationLines.length} sample lines.`,
      );
      updateSessionSnapshotStatus(
        hostElement,
        'Bundled warm start active. Live chat is using the rebuilt sample session; export it if the updated checkpoint performs better.',
      );
    }
  }

  const rerunButton = hostElement.querySelector<HTMLButtonElement>(
    '[data-neat-chat-rerender]',
  );
  rerunButton?.addEventListener('click', () => {
    const topWordLimitField = hostElement.querySelector<HTMLInputElement>(
      '[data-neat-chat-top-word-limit]',
    );
    const contextWindowField = hostElement.querySelector<HTMLSelectElement>(
      '[data-neat-chat-context-window-token-count]',
    );
    const corpusField = hostElement.querySelector<HTMLTextAreaElement>(
      '[data-neat-chat-corpus]',
    );

    renderContractPreview(
      hostElement,
      exampleContract,
      resolveTopWordLimit(topWordLimitField, exampleContract),
      resolveContextWindowTokenCount(contextWindowField, exampleContract),
      corpusField?.value ?? '',
    );
  });

  const contextWindowField = hostElement.querySelector<HTMLSelectElement>(
    '[data-neat-chat-context-window-token-count]',
  );
  contextWindowField?.addEventListener('change', () => {
    if (!currentSession) return;

    const resolvedContextWindowTokenCount = resolveContextWindowTokenCount(
      contextWindowField,
      exampleContract,
    );
    currentSession = updateNeatChatSessionContextWindowTokenCount(
      currentSession,
      resolvedContextWindowTokenCount,
    );
    appendSystemMessageToHistory(
      hostElement,
      `Updated live context window to ${resolvedContextWindowTokenCount} tokens without resetting learned weights.`,
    );
    updateSessionStatsDisplay(hostElement, currentSession);
  });

  // Step 5: Wire the live chat form — submit runs an exchange and supervised update.
  const liveChatForm = hostElement.querySelector<HTMLFormElement>(
    '[data-neat-chat-live-form]',
  );
  liveChatForm?.addEventListener('submit', (event) => {
    event.preventDefault();

    if (!currentSession) return;

    const liveInputField = hostElement.querySelector<HTMLInputElement>(
      '[data-neat-chat-live-input]',
    );
    const userMessage = liveInputField?.value.trim() ?? '';

    if (!userMessage) return;

    const result = runNeatChatExchange(currentSession, userMessage);
    currentSession = result.updatedSession;

    appendExchangeToHistory(
      hostElement,
      userMessage,
      result.response,
      result.trainedTokenPairCount,
      result.updatedSession,
    );
    updateSessionStatsDisplay(hostElement, currentSession);

    if (liveInputField) {
      liveInputField.value = '';
    }
  });

  const pretrainWithSampleButton = hostElement.querySelector<HTMLButtonElement>(
    '[data-neat-chat-load-sample]',
  );
  pretrainWithSampleButton?.addEventListener('click', () => {
    if (!currentSession) return;

    const sampleConversationLines = getNeatChatSampleConversationLines();

    if (sampleConversationCursor >= sampleConversationLines.length) {
      sampleConversationCursor = 0;
    }

    const chunkStartIndex = sampleConversationCursor;
    const chunkEndIndex = Math.min(
      chunkStartIndex + SAMPLE_PRETRAIN_CHUNK_SIZE,
      sampleConversationLines.length,
    );
    const seedChunkLines = sampleConversationLines.slice(
      chunkStartIndex,
      chunkEndIndex,
    );

    if (seedChunkLines.length === 0) {
      return;
    }

    currentSession = pretrainNeatChatSessionWithConversationLines(
      currentSession,
      seedChunkLines,
    );
    sampleConversationCursor = chunkEndIndex;
    activeCorpusLines.push(...seedChunkLines);

    const corpusField = hostElement.querySelector<HTMLTextAreaElement>(
      '[data-neat-chat-corpus]',
    );
    if (corpusField) {
      corpusField.value = activeCorpusLines.join('\n');
    }

    const updatedPretrainingPreview = createNeatChatPretrainingPreview({
      corpusText: activeCorpusLines.join('\n'),
      topWordLimit: runtimeEstimate.topWordLimit,
      chunkTokenCount: exampleContract.pretraining.defaultChunkTokenCount,
    });
    updateOptionalPretrainingSummaryDisplay(
      hostElement,
      updatedPretrainingPreview,
      runtimeEstimate.topWordLimit,
      exampleContract.pretraining.unknownToken,
    );

    appendSystemMessageToHistory(
      hostElement,
      `Pre-trained lines ${chunkStartIndex + 1}-${chunkEndIndex} of ${sampleConversationLines.length}.`,
    );
    updateSamplePretrainProgressDisplay(
      hostElement,
      chunkStartIndex,
      chunkEndIndex,
      sampleConversationLines.length,
    );
    updateSessionStatsDisplay(hostElement, currentSession);
  });

  const exportSessionButton = hostElement.querySelector<HTMLButtonElement>(
    '[data-neat-chat-export-session]',
  );
  exportSessionButton?.addEventListener('click', () => {
    if (!currentSession) return;

    const sessionSnapshot = exportNeatChatSession(currentSession);
    downloadTextFile(
      DEFAULT_SNAPSHOT_FILENAME,
      JSON.stringify(sessionSnapshot, null, 2),
    );
    updateSessionSnapshotStatus(
      hostElement,
      'Exported the current session snapshot.',
    );
  });

  const importSessionButton = hostElement.querySelector<HTMLButtonElement>(
    '[data-neat-chat-import-session-button]',
  );
  const importSessionInput = hostElement.querySelector<HTMLInputElement>(
    '[data-neat-chat-import-session-input]',
  );

  importSessionButton?.addEventListener('click', () => {
    importSessionInput?.click();
  });

  importSessionInput?.addEventListener('change', async () => {
    const selectedFile = importSessionInput.files?.[0];

    if (!selectedFile) {
      return;
    }

    try {
      const snapshotText = await selectedFile.text();
      currentSession = importNeatChatSession(JSON.parse(snapshotText));

      const contextWindowField = hostElement.querySelector<HTMLSelectElement>(
        '[data-neat-chat-context-window-token-count]',
      );
      synchronizeSessionSurfaceState(hostElement, currentSession);

      clearChatHistory(hostElement);
      appendSystemMessageToHistory(
        hostElement,
        'Imported a saved session snapshot.',
      );
      updateSessionStatsDisplay(hostElement, currentSession);
      updateSessionSnapshotStatus(
        hostElement,
        'Imported a saved session snapshot. Live chat now uses the imported weights and retained vocabulary.',
      );
    } catch (error) {
      updateSessionSnapshotStatus(
        hostElement,
        `Import failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      importSessionInput.value = '';
    }
  });

  const runAbButton = hostElement.querySelector<HTMLButtonElement>(
    '[data-neat-chat-run-ab]',
  );
  runAbButton?.addEventListener('click', () => {
    if (!currentSession) {
      return;
    }

    const abPromptField = hostElement.querySelector<HTMLInputElement>(
      '[data-neat-chat-ab-prompt]',
    );
    const comparisonPrompt = abPromptField?.value.trim() || 'hello there';
    const comparisonConversationLines =
      activeCorpusLines.length > 0 ? activeCorpusLines : sampleConversationLines;
    const comparisonLineSplit = splitNeatChatSeedAndValidationLines(
      comparisonConversationLines,
    );
    const comparisonRetainedTerms = resolveSessionRetainedTerms(
      currentSession,
      pretrainingPreview.retainedTerms,
    );
    const comparisonResult = createNeatChatAbComparison(comparisonPrompt, {
      corpusRetainedTerms: comparisonRetainedTerms,
      seedConversationLines: comparisonLineSplit.seedConversationLines,
      validationConversationLines:
        comparisonLineSplit.validationConversationLines,
      liveChatVocabLimit: Math.max(
        1,
        currentSession.vocabulary.size - NEATCHAT_SPECIAL_TOKEN_COUNT,
      ),
    });

    updateAbComparisonResults(hostElement, comparisonResult);
  });
}

function buildNeatChatMarkup(
  exampleContract: NeatChatExampleContract,
  corpusText: string,
  runtimeEstimate: NeatChatRuntimeEstimate,
  pretrainingPreview: NeatChatPretrainingPreview,
): string {
  const selectedContextWindow = runtimeEstimate.contextWindowTokenCount;
  const previewVocabulary = resolvePreviewVocabularyText(
    pretrainingPreview,
    exampleContract.pretraining.unknownToken,
  );
  const pretrainingStatus = pretrainingPreview.hasCorpus
    ? 'preseeded preview ready'
    : 'paste corpus text to prepare a preseeded preview';

  return `<section class="starter-demo-panel">
<p class="starter-demo-label" data-neat-chat-session-stats>Loading session…</p>
<article class="starter-demo-surface">
  <div class="starter-demo-header">
    <div>
      <h2>Live Chat</h2>
      <p>Type a message and press Send. The model trains on each exchange — send a few messages to observe adaptation.</p>
    </div>
  </div>
  <div data-neat-chat-history style="max-height:360px;overflow-y:auto;margin-bottom:0.75rem;"></div>
  <form data-neat-chat-live-form style="display:flex;gap:0.5rem;align-items:center;">
    <input type="text" data-neat-chat-live-input placeholder="Type a message…" style="flex:1;" autocomplete="off" />
    <button type="submit" class="starter-demo-button">Send</button>
  </form>
</article>
<article class="starter-demo-surface">
  <div class="starter-demo-header">
    <div>
      <h2>Training</h2>
      <p>Adjust the retained vocabulary cap, inspect the corpus report, then rebuild the live session from either pasted lines or the shipped warm start.</p>
    </div>
  </div>
  <div style="display:flex;gap:1rem;align-items:center;flex-wrap:wrap;margin-bottom:1rem;">
    <label for="neat-chat-top-word-limit">
      Vocabulary cap:
      <input id="neat-chat-top-word-limit" data-neat-chat-top-word-limit type="number" min="1" step="1" value="${runtimeEstimate.topWordLimit}" style="width:6.5rem;margin-left:0.25rem;" />
    </label>
    <label for="neat-chat-context-window-token-count">
      Context window:
      <select id="neat-chat-context-window-token-count" data-neat-chat-context-window-token-count style="margin-left:0.25rem;" data-tooltip-title="Context Window Size" title="${CONTEXT_WINDOW_EXPLANATION}">
        ${CONTEXT_WINDOW_OPTIONS.map((windowSize) => `<option value="${windowSize}"${windowSize === selectedContextWindow ? ' selected' : ''}>${windowSize} tokens</option>`).join('')}
      </select>
    </label>
    <button type="button" class="starter-demo-button" data-neat-chat-load-sample>Pre-train with sample (${SAMPLE_PRETRAIN_CHUNK_SIZE} lines)</button>
    <span class="starter-demo-label" data-neat-chat-sample-progress>Click to train the next ${SAMPLE_PRETRAIN_CHUNK_SIZE} lines.</span>
  </div>
  <label for="neat-chat-corpus" style="display:block;margin-bottom:0.5rem;">
    <span class="starter-demo-label">Optional pretraining corpus — paste conversation lines then click Apply</span>
    <textarea id="neat-chat-corpus" rows="6" data-neat-chat-corpus style="display:block;width:100%;margin-top:0.25rem;box-sizing:border-box;">${escapeHtml(corpusText)}</textarea>
  </label>
  <div style="display:flex;gap:0.75rem;align-items:center;flex-wrap:wrap;">
    <button type="button" class="starter-demo-button" data-neat-chat-rerender>Apply Corpus &amp; Rebuild Session</button>
    <button type="button" class="starter-demo-button" data-neat-chat-export-session>Export Session</button>
    <button type="button" class="starter-demo-button" data-neat-chat-import-session-button>Import Session</button>
    <input type="file" accept="application/json,.json" data-neat-chat-import-session-input style="display:none;" />
    <span class="starter-demo-label" data-neat-chat-session-snapshot-status>Session snapshots let you keep the best talking checkpoint.</span>
  </div>
  <p class="starter-demo-label">The corpus preview report explains the text you are about to train on. The live session stats above describe the checkpoint the chat box is currently using.</p>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:0.5rem;margin-top:1rem;">
    <span class="starter-demo-label">Characters: <strong data-neat-chat-corpus-character-count>${pretrainingPreview.corpusReport.characterCount}</strong></span>
    <span class="starter-demo-label">Tokens: <strong data-neat-chat-corpus-token-count>${pretrainingPreview.corpusReport.tokenCount}</strong></span>
    <span class="starter-demo-label">Unique terms: <strong data-neat-chat-corpus-unique-term-count>${pretrainingPreview.corpusReport.uniqueTermCount}</strong></span>
    <span class="starter-demo-label">Retained terms: <strong data-neat-chat-corpus-retained-term-count>${pretrainingPreview.corpusReport.retainedTermCount}</strong></span>
    <span class="starter-demo-label">Coverage: <strong data-neat-chat-corpus-coverage-percent>${pretrainingPreview.corpusReport.retainedTokenCoveragePercent}%</strong></span>
    <span class="starter-demo-label">Chunks: <strong data-neat-chat-corpus-chunk-count>${pretrainingPreview.processedChunkCount}</strong></span>
  </div>
  <div style="display:flex;gap:0.75rem;align-items:center;flex-wrap:wrap;margin-top:0.75rem;">
    <span class="starter-demo-label">Status: <strong data-neat-chat-pretraining-status>${pretrainingStatus}</strong></span>
    <span class="starter-demo-label">Preview vocabulary: <strong data-neat-chat-pretraining-preview-vocabulary>${escapeHtml(previewVocabulary)}</strong></span>
    <span class="starter-demo-label">Unknown-token fallback: ${exampleContract.pretraining.unknownToken}</span>
  </div>
</article>
<article class="starter-demo-surface">
  <div class="starter-demo-header">
    <div>
      <h2>A/B Evaluation</h2>
      <p>Run one prompt through blank-start and preseeded variants so the lightweight evaluation loop is visible in the flagship page, not only in tests.</p>
    </div>
  </div>
  <div style="display:flex;gap:0.75rem;align-items:center;flex-wrap:wrap;margin-bottom:0.75rem;">
    <input type="text" data-neat-chat-ab-prompt value="hello there" placeholder="Prompt for the A/B check" style="flex:1;min-width:220px;" autocomplete="off" />
    <button type="button" class="starter-demo-button" data-neat-chat-run-ab>Run A/B Check</button>
  </div>
  <div class="starter-demo-label" data-neat-chat-ab-results>Run a blank-start versus preseeded comparison to inspect held-out next-token accuracy, repetition rate, and response-length stability.</div>
</article>
</section>`;
}

function buildLoadingMarkup(): string {
  return '<section class="starter-demo-surface"><p class="starter-demo-loading">Loading the NEATchat flagship preview...</p></section>';
}

function buildErrorMarkup(errorMessage: string): string {
  return `<section class="starter-demo-surface"><h2>Preview failed</h2><p class="starter-demo-error">${escapeHtml(errorMessage)}</p><button type="button" class="starter-demo-button" data-neat-chat-rerender>Try Again</button></section>`;
}

function resolvePreviewVocabularyText(
  pretrainingPreview: NeatChatPretrainingPreview,
  unknownToken: string,
): string {
  return pretrainingPreview.previewTerms.length > 0
    ? pretrainingPreview.previewTerms.join(', ')
    : unknownToken;
}

/**
 * Appends one completed exchange row to the live chat history element.
 *
 * @param hostElement - Root host element containing the chat history.
 * @param userMessage - Raw user message text.
 * @param botResponse - Bot reply string.
 * @param trainedPairCount - Number of token pairs trained for this exchange.
 */
function appendExchangeToHistory(
  hostElement: HTMLElement,
  userMessage: string,
  botResponse: string,
  trainedPairCount: number,
  updatedSession: NeatChatSession,
): void {
  const historyEl = hostElement.querySelector<HTMLElement>(
    '[data-neat-chat-history]',
  );

  if (!historyEl) return;

  const exchangeEl = document.createElement('div');
  exchangeEl.innerHTML = `
    <p><strong>You</strong>: ${escapeHtml(userMessage)}</p>
    <p><strong>NEATchat</strong>: ${escapeHtml(botResponse)}</p>
    <p class="starter-demo-label">Trained on ${trainedPairCount} token pair${trainedPairCount !== 1 ? 's' : ''} this exchange | Total live learned: ${updatedSession.learnedTokenPairCount} | Seeded: ${updatedSession.seededTokenPairCount}</p>
    <hr style="border:none;border-top:1px solid rgba(100,200,255,0.2);margin:0.5rem 0;" />
  `;
  historyEl.appendChild(exchangeEl);
  historyEl.scrollTop = historyEl.scrollHeight;
}

function appendSystemMessageToHistory(
  hostElement: HTMLElement,
  systemMessage: string,
): void {
  const historyEl = hostElement.querySelector<HTMLElement>(
    '[data-neat-chat-history]',
  );

  if (!historyEl) return;

  const systemEl = document.createElement('div');
  systemEl.innerHTML = `
    <p class="starter-demo-label">${escapeHtml(systemMessage)}</p>
    <hr style="border:none;border-top:1px solid rgba(100,200,255,0.2);margin:0.5rem 0;" />
  `;
  historyEl.appendChild(systemEl);
  historyEl.scrollTop = historyEl.scrollHeight;
}

/**
 * Updates the session stats line below the live chat header.
 *
 * @param hostElement - Root host element containing the stats element.
 * @param session - Current live session with updated counts.
 */
function updateSessionStatsDisplay(
  hostElement: HTMLElement,
  session: NeatChatSession,
): void {
  const statsEl = hostElement.querySelector<HTMLElement>(
    '[data-neat-chat-session-stats]',
  );

  if (!statsEl) return;

  const vocabTermCount = session.vocabulary.size;
  const exchangeCount = session.learnedExchangeCount;
  const tokenPairCount = session.learnedTokenPairCount;
  const seededTokenPairCount = session.seededTokenPairCount;
  const contextWindowTokenCount = session.contextWindowTokenCount;
  const retainedTermCount = Math.max(
    0,
    session.vocabulary.size - NEATCHAT_SPECIAL_TOKEN_COUNT,
  );
  const replayBufferExchangeCount = session.replayBufferExchangeCount ?? 0;

  const cellStyle =
    'display:flex;align-items:stretch;flex:1;padding:0.45rem 1.1rem;text-align:center;color:#e0f8ff;font-weight:bold;font-size:0.95rem;border-right:1px solid rgba(0,212,255,0.25);';
  const lastCellStyle =
    'display:flex;align-items:stretch;flex:1;padding:0.45rem 1.1rem;text-align:center;color:#e0f8ff;font-weight:bold;font-size:0.95rem;';
  const headerCellStyle =
    'display:flex;flex-direction:column;align-items:stretch;flex:1;';
  const headStyle =
    'display:flex;align-items:stretch;flex:1;padding:0.3rem 1.1rem;text-align:center;color:#00d4ff;letter-spacing:0.08em;font-size:0.68rem;text-transform:uppercase;border-bottom:3px double rgba(0,212,255,0.6);border-right:1px solid rgba(0,212,255,0.25);';
  const lastHeadStyle =
    'display:flex;align-items:stretch;flex:1;padding:0.3rem 1.1rem;text-align:center;color:#00d4ff;letter-spacing:0.08em;font-size:0.68rem;text-transform:uppercase;border-bottom:3px double rgba(0,212,255,0.6);';
  const rowStyle = 'display:flex;align-items:stretch;flex:1;padding:1em;';
  statsEl.innerHTML = `<table style="display:flex;flex-direction:column;align-items:stretch;width:100%;border-collapse:separate;border-spacing:0;border:3px double rgba(0,212,255,0.7);background:rgba(0,16,36,0.55);font-family:'Courier New',monospace;">
    <thead style="${headerCellStyle}"><tr style="${rowStyle}">
      <th style="${headStyle}">Vocabulary</th>
      <th style="${headStyle}">Context Window</th>
      <th style="${headStyle}">Seed Token Pairs</th>
      <th style="${headStyle}">Exchanges</th>
      <th style="${headStyle}">Replay Buffer</th>
      <th style="${headStyle}">Live Retained Terms</th>
      <th style="${lastHeadStyle}">Token Pairs Learned</th>
    </tr></thead>
    <tbody><tr style="${rowStyle}">
      <td style="${cellStyle}">${vocabTermCount} terms</td>
      <td style="${cellStyle}" data-neat-chat-stats-context-window>${contextWindowTokenCount} tokens</td>
      <td style="${cellStyle}">${seededTokenPairCount}</td>
      <td style="${cellStyle}">${exchangeCount}</td>
      <td style="${cellStyle}">${replayBufferExchangeCount}</td>
      <td style="${cellStyle}" data-neat-chat-stats-retained-terms>${retainedTermCount}</td>
      <td style="${lastCellStyle}">${tokenPairCount}</td>
    </tr></tbody>
  </table>`;
}

function updateSamplePretrainProgressDisplay(
  hostElement: HTMLElement,
  chunkStartIndex: number,
  chunkEndIndex: number,
  totalLineCount: number,
): void {
  const progressEl = hostElement.querySelector<HTMLElement>(
    '[data-neat-chat-sample-progress]',
  );

  if (!progressEl) return;

  const remainingLineCount = Math.max(0, totalLineCount - chunkEndIndex);

  progressEl.textContent = `Sample pretraining progress: trained lines ${chunkStartIndex + 1}-${chunkEndIndex} of ${totalLineCount}. Remaining lines: ${remainingLineCount}.`;
}

function updateOptionalPretrainingSummaryDisplay(
  hostElement: HTMLElement,
  pretrainingPreview: NeatChatPretrainingPreview,
  topWordLimit: number,
  unknownToken: string,
): void {
  const characterCountEl = hostElement.querySelector<HTMLElement>(
    '[data-neat-chat-corpus-character-count]',
  );
  const tokenCountEl = hostElement.querySelector<HTMLElement>(
    '[data-neat-chat-corpus-token-count]',
  );
  const uniqueTermCountEl = hostElement.querySelector<HTMLElement>(
    '[data-neat-chat-corpus-unique-term-count]',
  );
  const retainedTermCountEl = hostElement.querySelector<HTMLElement>(
    '[data-neat-chat-corpus-retained-term-count]',
  );
  const coveragePercentEl = hostElement.querySelector<HTMLElement>(
    '[data-neat-chat-corpus-coverage-percent]',
  );
  const chunkCountEl = hostElement.querySelector<HTMLElement>(
    '[data-neat-chat-corpus-chunk-count]',
  );
  const statusEl = hostElement.querySelector<HTMLElement>(
    '[data-neat-chat-pretraining-status]',
  );
  const previewVocabularyEl = hostElement.querySelector<HTMLElement>(
    '[data-neat-chat-pretraining-preview-vocabulary]',
  );

  if (characterCountEl) {
    characterCountEl.textContent = String(
      pretrainingPreview.corpusReport.characterCount,
    );
  }
  if (tokenCountEl) {
    tokenCountEl.textContent = String(
      pretrainingPreview.corpusReport.tokenCount,
    );
  }
  if (uniqueTermCountEl) {
    uniqueTermCountEl.textContent = String(
      pretrainingPreview.corpusReport.uniqueTermCount,
    );
  }
  if (retainedTermCountEl) {
    retainedTermCountEl.textContent = String(
      pretrainingPreview.corpusReport.retainedTermCount,
    );
  }
  if (coveragePercentEl) {
    coveragePercentEl.textContent = `${pretrainingPreview.corpusReport.retainedTokenCoveragePercent}%`;
  }
  if (chunkCountEl) {
    chunkCountEl.textContent = String(pretrainingPreview.processedChunkCount);
  }
  if (statusEl) {
    statusEl.textContent = pretrainingPreview.hasCorpus
      ? 'preseeded preview ready'
      : 'paste corpus text to prepare a preseeded preview';
  }
  if (previewVocabularyEl) {
    previewVocabularyEl.textContent = resolvePreviewVocabularyText(
      pretrainingPreview,
      unknownToken,
    );
  }

  // Keep the top-word limit field synchronized in case users inspect after sample clicks.
  const topWordLimitField = hostElement.querySelector<HTMLInputElement>(
    '[data-neat-chat-top-word-limit]',
  );
  if (topWordLimitField) {
    topWordLimitField.value = String(topWordLimit);
  }
}

function updateSessionSnapshotStatus(
  hostElement: HTMLElement,
  statusText: string,
): void {
  const statusEl = hostElement.querySelector<HTMLElement>(
    '[data-neat-chat-session-snapshot-status]',
  );

  if (!statusEl) return;

  statusEl.textContent = statusText;
}

function updateAbComparisonResults(
  hostElement: HTMLElement,
  comparisonResult: NeatChatAbComparisonResult,
): void {
  const abResultsEl = hostElement.querySelector<HTMLElement>(
    '[data-neat-chat-ab-results]',
  );

  if (!abResultsEl) {
    return;
  }

  abResultsEl.innerHTML = comparisonResult.variants
    .map((variantResult) => {
      const variantLabel =
        variantResult.variant === 'blank-start'
          ? 'Blank-start'
          : 'Preseeded';

      return `<div style="margin-bottom:0.75rem;">
        <p><strong>${variantLabel}</strong>: ${escapeHtml(variantResult.response)}</p>
        <p class="starter-demo-label">Held-out next-token accuracy: ${variantResult.metrics.heldOutNextTokenAccuracy}% | Repetition rate: ${variantResult.metrics.repetitionRate} | Response-length stability: ${variantResult.metrics.responseLengthStability}</p>
      </div>`;
    })
    .join('');
}

function clearChatHistory(hostElement: HTMLElement): void {
  const historyEl = hostElement.querySelector<HTMLElement>(
    '[data-neat-chat-history]',
  );

  if (!historyEl) return;

  historyEl.innerHTML = '';
}

function downloadTextFile(fileName: string, fileContents: string): void {
  const downloadUrl = URL.createObjectURL(
    new Blob([fileContents], { type: 'application/json' }),
  );
  const downloadLink = document.createElement('a');

  downloadLink.href = downloadUrl;
  downloadLink.download = fileName;
  downloadLink.click();
  URL.revokeObjectURL(downloadUrl);
}

function resolveContextWindowTokenCount(
  contextWindowField: HTMLSelectElement | null,
  exampleContract: NeatChatExampleContract,
): number {
  const rawValue = Number(contextWindowField?.value ?? Number.NaN);

  if (!Number.isInteger(rawValue) || rawValue <= 0) {
    return exampleContract.pretraining.defaultContextWindowTokenCount;
  }

  return rawValue;
}

function resolveTopWordLimit(
  topWordLimitField: HTMLInputElement | null,
  exampleContract: NeatChatExampleContract,
): number {
  const rawValue = Number(topWordLimitField?.value ?? Number.NaN);

  if (!Number.isInteger(rawValue) || rawValue <= 0) {
    return exampleContract.pretraining.defaultTopWordLimit;
  }

  return rawValue;
}

function resolveSessionRetainedTerms(
  session: NeatChatSession,
  fallbackRetainedTerms: readonly string[],
): readonly string[] {
  const sessionVocabularyTerms = session.vocabulary as {
    indexToTerm?: readonly string[];
  };

  if (Array.isArray(sessionVocabularyTerms.indexToTerm)) {
    return sessionVocabularyTerms.indexToTerm.slice(NEATCHAT_SPECIAL_TOKEN_COUNT);
  }

  return fallbackRetainedTerms;
}

function synchronizeSessionSurfaceState(
  hostElement: HTMLElement,
  session: NeatChatSession,
): void {
  const contextWindowField = hostElement.querySelector<HTMLSelectElement>(
    '[data-neat-chat-context-window-token-count]',
  );

  if (!contextWindowField) {
    return;
  }

  ensureContextWindowOptionExists(
    contextWindowField,
    session.contextWindowTokenCount,
  );
  contextWindowField.value = String(session.contextWindowTokenCount);
}

function ensureContextWindowOptionExists(
  contextWindowField: HTMLSelectElement,
  contextWindowTokenCount: number,
): void {
  const optionValue = String(contextWindowTokenCount);

  if (
    Array.from(contextWindowField.options).some(
      (contextWindowOption) => contextWindowOption.value === optionValue,
    )
  ) {
    return;
  }

  const insertedOption = new Option(
    `${contextWindowTokenCount} tokens`,
    optionValue,
  );
  const sortedWindowSizes = [...CONTEXT_WINDOW_OPTIONS, contextWindowTokenCount]
    .filter(
      (windowSize, windowIndex, windowSizes) =>
        windowSizes.indexOf(windowSize) === windowIndex,
    )
    .toSorted((leftWindowSize, rightWindowSize) => leftWindowSize - rightWindowSize);
  const insertionIndex = sortedWindowSizes.indexOf(contextWindowTokenCount);

  contextWindowField.add(insertedOption, insertionIndex);
}

function resolveHostElement(container: BrowserHostContainer): HTMLElement {
  if (typeof container === 'string') {
    const resolvedElement = document.getElementById(container);

    if (!resolvedElement) {
      throw new Error(`NEATchat browser demo could not find #${container}.`);
    }

    return resolvedElement;
  }

  return container;
}

function waitForAnimationFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      resolve();
    });
  });
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

if (typeof window !== 'undefined') {
  window.neatChatStart = start;
  window.neatChat = { start };
}
