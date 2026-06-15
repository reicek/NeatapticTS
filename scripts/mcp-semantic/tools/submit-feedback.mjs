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

/** Signal types that callers may submit explicitly through this tool. */
const VALID_SIGNAL_TYPES = ['reference', 'positive', 'negative'];

/**
 * Submit an explicit feedback signal for a corpus chunk.
 *
 * @param {object} [options={}] - Tool options.
 * @param {number} options.chunk_id - Target chunk ID.
 * @param {string} options.signal_type - One of 'reference', 'positive', 'negative'.
 * @param {string} [options.context] - Optional free-text context (truncated to 500 chars).
 * @param {string} [options.query] - Optional originating query (hashed before storage).
 * @param {string} [options.agent_id] - Optional agent identifier.
 * @param {string} [options.databasePath] - Override corpus database path.
 * @returns {Promise<{chunk_id: number, signal_type: string, recorded: boolean, feedback_boost_after: number}>} Submission summary.
 * @throws {Error} When chunk_id is not a positive integer or signal_type is not allowed.
 */
export async function submitFeedback(options = {}) {
  const chunkId = Number(options.chunk_id);
  if (!Number.isInteger(chunkId) || chunkId < 1) {
    throw new Error('chunk_id must be a positive integer.');
  }

  const signalType = options.signal_type;
  if (!VALID_SIGNAL_TYPES.includes(signalType)) {
    throw new Error(
      `signal_type must be one of: ${VALID_SIGNAL_TYPES.join(', ')}.`,
    );
  }

  const db = new Database(resolveDatabasePath(options.databasePath));
  try {
    recordFeedbackEvent(db, {
      chunk_id: chunkId,
      signal_type: signalType,
      query: options.query,
      agent_id: options.agent_id,
      context: options.context,
    });

    const score = updateFeedbackScores(db, chunkId, Date.now());

    return {
      chunk_id: chunkId,
      signal_type: signalType,
      recorded: true,
      feedback_boost_after: score.feedback_boost,
    };
  } finally {
    db.close();
  }
}
