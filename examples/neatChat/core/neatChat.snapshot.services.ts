import { Network } from '../../../src/browser-entry.ts';
import {
  NEATCHAT_REPLAY_BUFFER_MAX_EXCHANGES,
  NEATCHAT_SPECIAL_TOKENS,
} from './neatChat.constants';
import {
  NeatChatNonNegativeIntegerValidationError,
  NeatChatSnapshotShapeError,
  NeatChatSnapshotVersionError,
} from './neatChat.errors';
import { buildNeatChatVocabulary } from './neatChat.session.services';
import { resolvePositiveInteger } from './neatChat.tokenization.utils';
import type {
  NeatChatSession,
  NeatChatSessionSnapshot,
} from './neatChat.types';

/**
 * Serializes a live NEATchat session into a JSON-safe snapshot.
 *
 * @param session - Current live session to persist.
 * @returns JSON-safe snapshot that can later restore the same weights and vocabulary.
 *
 * @example
 * ```ts
 * const session = createNeatChatSession();
 * const snapshot = exportNeatChatSession(session);
 * ```
 */
export function exportNeatChatSession(
  session: NeatChatSession,
): NeatChatSessionSnapshot {
  return {
    formatVersion: 1,
    retainedTerms: session.vocabulary.indexToTerm.slice(
      NEATCHAT_SPECIAL_TOKENS.length,
    ),
    networkJson: session.network.toJSON(),
    exchanges: session.exchanges.map((exchangeRecord) => ({
      userMessage: exchangeRecord.userMessage,
      response: exchangeRecord.response,
      trainedTokenPairCount: exchangeRecord.trainedTokenPairCount,
      userTokens: [...exchangeRecord.userTokens],
      responseTokens: [...exchangeRecord.responseTokens],
    })),
    learnedExchangeCount: session.learnedExchangeCount,
    learnedTokenPairCount: session.learnedTokenPairCount,
    seededTokenPairCount: session.seededTokenPairCount,
    contextWindowTokenCount: session.contextWindowTokenCount,
  };
}

/**
 * Restores a NEATchat session from a serialized snapshot.
 *
 * This import path is intentionally strict: it validates shape and counters
 * before rebuilding vocabulary and weights so sessions fail early with
 * boundary-specific typed errors instead of partial runtime corruption.
 *
 * @param snapshot - Parsed snapshot payload created by exportNeatChatSession.
 * @returns Restored live session with matching vocabulary, weights, and counts.
 * @throws {NeatChatSnapshotVersionError} When the snapshot version is unsupported.
 * @throws {NeatChatSnapshotShapeError} When required payload fields are invalid.
 * @throws {NeatChatPositiveIntegerValidationError} When a positive counter is invalid.
 * @throws {NeatChatNonNegativeIntegerValidationError} When a non-negative counter is invalid.
 *
 * @example
 * ```ts
 * const restoredSession = importNeatChatSession(snapshot);
 * ```
 */
export function importNeatChatSession(
  snapshot: NeatChatSessionSnapshot,
): NeatChatSession {
  const validatedSnapshot = validateNeatChatSessionSnapshot(snapshot);

  return {
    vocabulary: buildNeatChatVocabulary(validatedSnapshot.retainedTerms),
    network: Network.fromJSON(validatedSnapshot.networkJson),
    exchanges: validatedSnapshot.exchanges.map((exchangeRecord) => ({
      userMessage: exchangeRecord.userMessage,
      response: exchangeRecord.response,
      trainedTokenPairCount: exchangeRecord.trainedTokenPairCount,
      userTokens: [...exchangeRecord.userTokens],
      responseTokens: [...exchangeRecord.responseTokens],
    })),
    learnedExchangeCount: validatedSnapshot.learnedExchangeCount,
    learnedTokenPairCount: validatedSnapshot.learnedTokenPairCount,
    seededTokenPairCount: validatedSnapshot.seededTokenPairCount,
    contextWindowTokenCount: validatedSnapshot.contextWindowTokenCount,
    replayBufferExchangeCount: Math.min(
      validatedSnapshot.exchanges.length,
      NEATCHAT_REPLAY_BUFFER_MAX_EXCHANGES,
    ),
  };
}

/**
 * Validates and normalizes an imported session snapshot payload.
 *
 * @param snapshot - Parsed snapshot payload from external input.
 * @returns Normalized snapshot safe for rehydrating a live session.
 * @throws {NeatChatSnapshotVersionError} When formatVersion is unsupported.
 * @throws {NeatChatSnapshotShapeError} When required fields are missing or malformed.
 * @throws {NeatChatPositiveIntegerValidationError} When positive counters are invalid.
 * @throws {NeatChatNonNegativeIntegerValidationError} When non-negative counters are invalid.
 */
function validateNeatChatSessionSnapshot(
  snapshot: NeatChatSessionSnapshot,
): NeatChatSessionSnapshot {
  if (snapshot.formatVersion !== 1) {
    throw new NeatChatSnapshotVersionError(
      `Unsupported NEATchat session snapshot version: ${String(snapshot.formatVersion)}.`,
    );
  }

  if (!Array.isArray(snapshot.retainedTerms)) {
    throw new NeatChatSnapshotShapeError(
      'NEATchat session snapshot retainedTerms must be an array.',
    );
  }

  if (
    snapshot.networkJson === null ||
    typeof snapshot.networkJson !== 'object' ||
    Array.isArray(snapshot.networkJson)
  ) {
    throw new NeatChatSnapshotShapeError(
      'NEATchat session snapshot networkJson must be an object.',
    );
  }

  if (!Array.isArray(snapshot.exchanges)) {
    throw new NeatChatSnapshotShapeError(
      'NEATchat session snapshot exchanges must be an array.',
    );
  }

  return {
    ...snapshot,
    retainedTerms: snapshot.retainedTerms.map((term) => String(term)),
    exchanges: snapshot.exchanges.map((exchangeRecord) => ({
      userMessage: String(exchangeRecord.userMessage),
      response: String(exchangeRecord.response),
      trainedTokenPairCount: resolvePositiveInteger(
        exchangeRecord.trainedTokenPairCount,
        'trainedTokenPairCount',
      ),
      userTokens: [...exchangeRecord.userTokens].map((token) => String(token)),
      responseTokens: [...exchangeRecord.responseTokens].map((token) =>
        String(token),
      ),
    })),
    learnedExchangeCount: resolveNonNegativeInteger(
      snapshot.learnedExchangeCount,
      'learnedExchangeCount',
    ),
    learnedTokenPairCount: resolveNonNegativeInteger(
      snapshot.learnedTokenPairCount,
      'learnedTokenPairCount',
    ),
    seededTokenPairCount: resolveNonNegativeInteger(
      snapshot.seededTokenPairCount,
      'seededTokenPairCount',
    ),
    contextWindowTokenCount: resolvePositiveInteger(
      snapshot.contextWindowTokenCount,
      'contextWindowTokenCount',
    ),
  };
}

/**
 * Validates a non-negative integer counter used in snapshots.
 *
 * @param value - Candidate numeric counter value.
 * @param fieldName - Counter field name used in validation errors.
 * @returns Original value when valid.
 * @throws {NeatChatNonNegativeIntegerValidationError} When the value is not a non-negative integer.
 */
function resolveNonNegativeInteger(value: number, fieldName: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new NeatChatNonNegativeIntegerValidationError(
      `${fieldName} must be a non-negative integer.`,
    );
  }

  return value;
}
