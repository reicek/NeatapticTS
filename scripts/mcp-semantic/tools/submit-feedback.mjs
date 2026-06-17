/**
 * @module submit-feedback
 * @description Explicit feedback submission MCP tool for the Repo Cortex server.
 *
 * Records reference, positive, or negative feedback events for a specific corpus
 * chunk and immediately recomputes the chunk's feedback boost score so that
 * subsequent searches can incorporate the new signal.
 */
import Database from 'better-sqlite3';

import { resolveDatabasePath } from './cortex-db.mjs';
import { recordFeedbackEvent, updateFeedbackScores } from './feedback-core.mjs';
import { ErrorCodes, cortexError } from './cortex-error.mjs';

/** Explicit feedback signal types accepted by the submit_feedback tool. */
const SUBMITTABLE_SIGNAL_TYPES = Object.freeze([
  'reference',
  'positive',
  'negative',
  'irrelevant',
]);

/**
 * Submit an explicit feedback signal for a corpus chunk.
 *
 * @param {object} [options={}] - Tool options.
 * @param {number} options.chunk_id - Target chunk ID.
 * @param {string} options.signal_type - One of 'reference', 'positive', 'negative', 'irrelevant'.
 * @param {number} [options.signal_strength] - Optional explicit signal strength override.
 * @param {string} [options.context] - Optional free-text context (truncated to 500 chars).
 * @param {string} [options.query] - Optional originating query (hashed before storage).
 * @param {string} [options.agent_id] - Optional agent identifier.
 * @param {string} [options.databasePath] - Override corpus database path.
 * @returns {Promise<object>} Submission summary.
 * @throws {Error} When chunk_id is not a positive integer, the chunk is missing, or signal_type is not allowed.
 */
export async function submitFeedback(options = {}) {
  const chunkId = Number(options.chunk_id);
  if (!Number.isInteger(chunkId) || chunkId < 1) {
    throw cortexError(
      ErrorCodes.MISSING_CHUNK_ID,
      'chunk_id must be a positive integer.',
    );
  }

  const signalType = options.signal_type;
  if (!SUBMITTABLE_SIGNAL_TYPES.includes(signalType)) {
    throw cortexError(
      ErrorCodes.INVALID_SIGNAL_TYPE,
      `signal_type must be one of: ${SUBMITTABLE_SIGNAL_TYPES.join(', ')}.`,
    );
  }

  const db = new Database(resolveDatabasePath(options.databasePath));
  try {
    const chunkExists = db
      .prepare('SELECT 1 FROM chunks WHERE chunk_id = ?')
      .get(chunkId);
    if (!chunkExists) {
      throw cortexError(
        ErrorCodes.MISSING_CHUNK_ID,
        `chunk_id ${chunkId} does not exist.`,
      );
    }

    const recorded = recordFeedbackEvent(db, {
      chunk_id: chunkId,
      signal_type: signalType,
      signal_strength: options.signal_strength,
      query: options.query,
      agent_id: options.agent_id,
      context: options.context,
    });

    const score = updateFeedbackScores(db, chunkId, Date.now());
    const totalSignals = db
      .prepare('SELECT COUNT(*) AS c FROM feedback_events WHERE chunk_id = ?')
      .get(chunkId).c;

    return {
      chunk_id: chunkId,
      signal_type: signalType,
      signal_strength: recorded.signal_strength,
      recorded: true,
      feedback_boost_after: score.feedback_boost,
      feedback_score: score.feedback_boost,
      feedback_boost: score.feedback_boost,
      total_signals: totalSignals,
    };
  } finally {
    db.close();
  }
}
