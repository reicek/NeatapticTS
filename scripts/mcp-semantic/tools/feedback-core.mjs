/**
 * @module feedback-core
 * @description Core feedback signal recording, boost computation, and score aggregation.
 *
 * Implements the relevance-feedback scoring described in
 * {@link ../../rag_architecture/cortex-relevance-feedback.md}.
 */

import { createHash, randomUUID } from 'node:crypto';

/** Maximum character length for free-text feedback context. */
const MAX_CONTEXT_LENGTH = 500;

/** Hex-character set used to detect pre-computed SHA-256 hashes. */
const HEX_RE = /^[0-9a-f]{64}$/i;

/** Half-life for feedback time decay (7 days in milliseconds). */
export const FEEDBACK_HALF_LIFE_MS = 7 * 24 * 60 * 60 * 1000;

/** Minimum impressions before impression-based decay applies. */
export const MIN_IMPRESSIONS_FOR_DECAY = 10;

/** Neutral click-through-rate threshold; above this no impression decay applies. */
export const MIN_CTR_FOR_NEUTRAL = 0.1;

/** Lower bound for explicit signal-strength overrides. */
const MIN_EXPLICIT_SIGNAL_STRENGTH = -1.0;

/** Upper bound for explicit signal-strength overrides. */
const MAX_EXPLICIT_SIGNAL_STRENGTH = 1.0;

/** Time window within which repeated same-session positive signals are normalized. */
const SAME_SESSION_WINDOW_MS = 60 * 60 * 1000;

const SIGNAL_STRENGTHS = {
  click: 0.3,
  impression: 0.1,
  negative: -1.0,
  positive: 1.0,
  reference: 0.6,
  irrelevant: -0.5,
};

/** Valid signal_type values accepted by the feedback pipeline. */
export const VALID_SIGNAL_TYPES = Object.freeze(Object.keys(SIGNAL_STRENGTHS));

/**
 * Hash a plaintext query with SHA-256.
 *
 * @param {string} query - Plaintext query.
 * @returns {string} Lower-case hex SHA-256 digest.
 */
function hashQuery(query) {
  return createHash('sha256').update(query).digest('hex');
}

/**
 * Normalize a query identifier.
 *
 * If the supplied value already looks like a SHA-256 hash it is returned as-is;
 * otherwise it is hashed so that plaintext queries are never persisted.
 *
 * @param {string} [value] - Caller-provided query or query hash.
 * @returns {string | null} A SHA-256 hash, or null when no value is given.
 */
function normalizeQueryHash(value) {
  if (value === undefined || value === null) {
    return null;
  }
  if (HEX_RE.test(value)) {
    return value.toLowerCase();
  }
  return hashQuery(value);
}

/**
 * Convert a feedback event created_at value to milliseconds since epoch.
 *
 * @param {number | string} createdAt - Stored timestamp.
 * @returns {number} Milliseconds since epoch.
 */
function parseCreatedAt(createdAt) {
  if (typeof createdAt === 'number') {
    return createdAt;
  }
  const parsed = Date.parse(createdAt);
  return Number.isNaN(parsed) ? 0 : parsed;
}

/**
 * Clamp an explicit signal strength to the designed feedback range.
 *
 * @param {number} strength - Raw signal strength.
 * @returns {number} Strength clamped to [MIN_EXPLICIT_SIGNAL_STRENGTH, MAX_EXPLICIT_SIGNAL_STRENGTH].
 */
function clampSignalStrength(strength) {
  if (typeof strength !== 'number' || Number.isNaN(strength)) {
    return strength;
  }
  return Math.max(
    MIN_EXPLICIT_SIGNAL_STRENGTH,
    Math.min(MAX_EXPLICIT_SIGNAL_STRENGTH, strength),
  );
}

/**
 * Check whether a positive signal for the same chunk and agent was already
 * recorded within the same-session window, regardless of query.
 *
 * @param {import('better-sqlite3').Database} db - SQLite database connection.
 * @param {number} chunkId - Target chunk ID.
 * @param {string | null} agentId - Agent identifier.
 * @param {number} now - Current time in milliseconds since epoch.
 * @returns {boolean} True when a recent same-session positive event exists.
 */
function hasRecentSameSessionPositive(db, chunkId, agentId, now) {
  if (!agentId) {
    return false;
  }
  const cutoff = new Date(now - SAME_SESSION_WINDOW_MS).toISOString();
  const row = db
    .prepare(
      `SELECT 1 FROM feedback_events
       WHERE chunk_id = ? AND signal_type = 'positive'
         AND agent_id = ? AND created_at >= ?
       LIMIT 1`,
    )
    .get(chunkId, agentId, cutoff);
  return Boolean(row);
}

/**
 * Record a feedback event in the feedback_events table.
 *
 * @param {import('better-sqlite3').Database} db - SQLite database connection.
 * @param {object} params - Event parameters.
 * @param {number} params.chunk_id - Target chunk ID.
 * @param {string} params.signal_type - One of impression, click, reference, positive, negative.
 * @param {string} [params.query_hash] - SHA-256 hash of the correlated query, or plaintext to be hashed.
 * @param {string} [params.query] - Plaintext query (hashed before storage).
 * @param {string} [params.agent_id] - Optional agent identifier.
 * @param {string} [params.context] - Optional free-text context (will be truncated to 500 chars).
 * @param {number | string} [params.created_at] - Optional timestamp; defaults to now.
 * @returns {object} The inserted feedback event row, or the existing same-session row when a duplicate positive signal is normalized.
 * @throws {Error} When signal_type is not recognized.
 */
export function recordFeedbackEvent(db, params) {
  const eventId = randomUUID();
  const defaultStrength = SIGNAL_STRENGTHS[params.signal_type];
  if (defaultStrength === undefined) {
    throw new Error(`Unknown signal_type: ${params.signal_type}`);
  }
  const signalStrength =
    params.signal_strength !== undefined && params.signal_strength !== null
      ? clampSignalStrength(Number(params.signal_strength))
      : defaultStrength;

  const rawQuery = params.query ?? params.query_hash;
  const queryHash = normalizeQueryHash(rawQuery);
  const context =
    params.context === undefined || params.context === null
      ? null
      : params.context.slice(0, MAX_CONTEXT_LENGTH);
  const agentId = params.agent_id ?? null;
  const createdAt =
    params.created_at === undefined || params.created_at === null
      ? new Date().toISOString()
      : params.created_at;

  // Normalize repeated same-session positive signals so they cannot inflate
  // the feedback score. Return the existing event without inserting a duplicate.
  const now = Date.now();
  if (
    params.signal_type === 'positive' &&
    hasRecentSameSessionPositive(db, params.chunk_id, agentId, now)
  ) {
    const cutoff = new Date(now - SAME_SESSION_WINDOW_MS).toISOString();
    const existing = db
      .prepare(
        `SELECT * FROM feedback_events
         WHERE chunk_id = ? AND signal_type = 'positive'
           AND agent_id = ? AND created_at >= ?
         ORDER BY created_at DESC
         LIMIT 1`,
      )
      .get(params.chunk_id, agentId, cutoff);
    return existing;
  }

  const insert = db.prepare(`
    INSERT INTO feedback_events
      (event_id, chunk_id, signal_type, signal_strength, query_hash, agent_id, context, created_at)
    VALUES
      (?, ?, ?, ?, ?, ?, ?, ?)
    RETURNING *
  `);

  return insert.get(
    eventId,
    params.chunk_id,
    params.signal_type,
    signalStrength,
    queryHash,
    agentId,
    context,
    createdAt,
  );
}

/**
 * Compute the feedback boost score using sigmoid dampening.
 *
 * Positive and negative signals are combined with click-through-rate and
 * reference bonuses, then an impression-based decay penalty is applied for
 * low-CTR chunks that have enough impressions. The result is dampened through
 * {@code 0.5 * Math.tanh(netFeedback * 2.0)}, which naturally clamps it to
 * [-0.5, +0.5].
 *
 * @param {object} scores - Aggregated score counters.
 * @param {number} [scores.total_positive=0] - Time-decayed sum of positive signal strengths.
 * @param {number} [scores.total_negative=0] - Time-decayed sum of negative signal magnitudes.
 * @param {number} [scores.total_impressions=0] - Total impression count.
 * @param {number} [scores.total_clicks=0] - Total click count.
 * @param {number} [scores.total_references=0] - Total reference count.
 * @returns {number} Feedback boost in [-0.5, +0.5].
 */
export function computeFeedbackBoost(scores) {
  const totalPositive = scores.total_positive ?? 0;
  const totalNegative = scores.total_negative ?? 0;
  const totalImpressions = scores.total_impressions ?? 0;
  const totalClicks = scores.total_clicks ?? 0;
  const totalReferences = scores.total_references ?? 0;

  const clickThroughRate = totalClicks / Math.max(1, totalImpressions);
  const referenceBonus = Math.min(1.0, totalReferences * 0.2);
  const combinedPositive =
    totalPositive + clickThroughRate * 0.3 + referenceBonus;
  let netFeedback = combinedPositive - totalNegative;

  const impressionDecayScore =
    totalImpressions >= MIN_IMPRESSIONS_FOR_DECAY &&
    clickThroughRate < MIN_CTR_FOR_NEUTRAL
      ? -0.2 * (1.0 - clickThroughRate / MIN_CTR_FOR_NEUTRAL)
      : 0.0;

  netFeedback += impressionDecayScore;

  return 0.5 * Math.tanh(netFeedback * 2.0);
}

/**
 * Build a feedback score aggregate for a single chunk.
 *
 * @param {import('better-sqlite3').Database} db - SQLite database connection.
 * @param {number} chunkId - Chunk to recompute.
 * @param {number} now - Current time in milliseconds since epoch.
 * @returns {object | null} Score counters and boost, or null when no events exist.
 */
function buildChunkAggregate(db, chunkId, now) {
  const events = db
    .prepare('SELECT * FROM feedback_events WHERE chunk_id = ?')
    .all(chunkId);
  if (events.length === 0) {
    return null;
  }

  let totalPositive = 0;
  let totalNegative = 0;
  let totalImpressions = 0;
  let totalClicks = 0;
  let totalReferences = 0;

  for (const event of events) {
    const age = now - parseCreatedAt(event.created_at);
    const decay = Math.pow(0.5, age / FEEDBACK_HALF_LIFE_MS);
    const strength = event.signal_strength * decay;

    switch (event.signal_type) {
      case 'impression':
        totalImpressions += 1;
        break;
      case 'click':
        totalClicks += 1;
        break;
      case 'reference':
        totalReferences += 1;
        break;
      case 'positive':
        totalPositive += strength;
        break;
      case 'negative':
      case 'irrelevant':
        totalNegative += Math.abs(strength);
        break;
    }
  }

  const feedbackBoost = computeFeedbackBoost({
    total_positive: totalPositive,
    total_negative: totalNegative,
    total_impressions: totalImpressions,
    total_clicks: totalClicks,
    total_references: totalReferences,
  });

  return {
    chunk_id: chunkId,
    total_positive: totalPositive,
    total_negative: totalNegative,
    total_impressions: totalImpressions,
    total_clicks: totalClicks,
    total_references: totalReferences,
    feedback_boost: feedbackBoost,
    last_feedback_at: new Date(now).toISOString(),
  };
}

/**
 * Upsert a feedback_scores row.
 *
 * @param {import('better-sqlite3').Database} db - SQLite database connection.
 * @param {object} score - Score aggregate returned by buildChunkAggregate.
 */
function upsertFeedbackScore(db, score) {
  const stmt = db.prepare(`
    INSERT INTO feedback_scores
      (chunk_id, total_positive, total_negative, total_impressions, total_clicks, total_references, last_feedback_at, feedback_boost)
    VALUES
      (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(chunk_id) DO UPDATE SET
      total_positive = excluded.total_positive,
      total_negative = excluded.total_negative,
      total_impressions = excluded.total_impressions,
      total_clicks = excluded.total_clicks,
      total_references = excluded.total_references,
      last_feedback_at = excluded.last_feedback_at,
      feedback_boost = excluded.feedback_boost
  `);
  stmt.run(
    score.chunk_id,
    score.total_positive,
    score.total_negative,
    score.total_impressions,
    score.total_clicks,
    score.total_references,
    score.last_feedback_at,
    score.feedback_boost,
  );
}

/**
 * Incrementally update feedback_scores for a single chunk after a new event.
 *
 * @param {import('better-sqlite3').Database} db - SQLite database connection.
 * @param {number} chunkId - Chunk to update.
 * @param {number} [now=Date.now()] - Current time in milliseconds since epoch.
 * @returns {object | null} The upserted score row, or null when no events exist.
 */
export function updateFeedbackScores(db, chunkId, now = Date.now()) {
  const score = buildChunkAggregate(db, chunkId, now);
  if (score === null) {
    return null;
  }
  upsertFeedbackScore(db, score);
  return score;
}

/**
 * Recompute all feedback scores from scratch, applying time decay and impression decay.
 *
 * @param {import('better-sqlite3').Database} db - SQLite database connection.
 * @param {number} [now=Date.now()] - Current time in milliseconds since epoch.
 * @returns {number} Number of chunks whose scores were recomputed.
 */
export function recomputeAllFeedbackScores(db, now = Date.now()) {
  const rows = db
    .prepare('SELECT DISTINCT chunk_id FROM feedback_events')
    .all();
  let count = 0;
  for (const { chunk_id: chunkId } of rows) {
    const score = buildChunkAggregate(db, chunkId, now);
    upsertFeedbackScore(db, score);
    count += 1;
  }
  return count;
}
