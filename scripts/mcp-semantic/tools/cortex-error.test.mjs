/**
 * @module cortex-error.test
 * @description Coverage tests for cortex-error.mjs — shared error taxonomy.
 */
import { ErrorCodes, cortexError, isCortexError } from './cortex-error.mjs';

describe('cortex-error', () => {
  describe('ErrorCodes', () => {
    it('exports a frozen object with all known codes', () => {
      expect(Object.isFrozen(ErrorCodes)).toBe(true);
      expect(ErrorCodes.CORPUS_NOT_FOUND).toBe('CORPUS_NOT_FOUND');
      expect(ErrorCodes.CORTEX_TIMEOUT_PARTIAL).toBe('CORTEX_TIMEOUT_PARTIAL');
      expect(ErrorCodes.EMPTY_QUERY).toBe('EMPTY_QUERY');
      expect(ErrorCodes.INVALID_ALPHA).toBe('INVALID_ALPHA');
      expect(ErrorCodes.INVALID_BUDGET).toBe('INVALID_BUDGET');
      expect(ErrorCodes.INVALID_LIMIT).toBe('INVALID_LIMIT');
      expect(ErrorCodes.INVALID_MAX_HOPS).toBe('INVALID_MAX_HOPS');
      expect(ErrorCodes.INVALID_METADATA_FILTER).toBe(
        'INVALID_METADATA_FILTER',
      );
      expect(ErrorCodes.INVALID_QUERY_CLASS).toBe('INVALID_QUERY_CLASS');
      expect(ErrorCodes.INVALID_SIGNAL_TYPE).toBe('INVALID_SIGNAL_TYPE');
      expect(ErrorCodes.MISSING_CHUNK_ID).toBe('MISSING_CHUNK_ID');
      expect(ErrorCodes.SEED_REQUIRED).toBe('SEED_REQUIRED');
    });
  });

  describe('cortexError', () => {
    it('builds an Error with code prefix in message', () => {
      const err = cortexError(ErrorCodes.EMPTY_QUERY, 'query is required');
      expect(err).toBeInstanceOf(Error);
      expect(err.message).toBe('EMPTY_QUERY: query is required');
    });

    it('works with any code and message', () => {
      const err = cortexError('CUSTOM_CODE', 'something happened');
      expect(err.message).toBe('CUSTOM_CODE: something happened');
    });
  });

  describe('isCortexError', () => {
    it('returns true when error is an Error with matching code prefix', () => {
      const err = cortexError(ErrorCodes.EMPTY_QUERY, 'missing');
      expect(isCortexError(err, ErrorCodes.EMPTY_QUERY)).toBe(true);
    });

    it('returns false when error is an Error but code does not match', () => {
      const err = cortexError(ErrorCodes.INVALID_LIMIT, 'bad limit');
      expect(isCortexError(err, ErrorCodes.EMPTY_QUERY)).toBe(false);
    });

    it('returns false when error is not an Error instance', () => {
      expect(isCortexError('string', ErrorCodes.EMPTY_QUERY)).toBe(false);
      expect(isCortexError(null, ErrorCodes.EMPTY_QUERY)).toBe(false);
      expect(isCortexError(undefined, ErrorCodes.EMPTY_QUERY)).toBe(false);
      expect(isCortexError(42, ErrorCodes.EMPTY_QUERY)).toBe(false);
      expect(isCortexError({}, ErrorCodes.EMPTY_QUERY)).toBe(false);
    });

    it('returns false for a generic Error without code prefix', () => {
      const err = new Error('some generic error');
      expect(isCortexError(err, ErrorCodes.EMPTY_QUERY)).toBe(false);
    });

    it('returns true for an Error with code prefix but not from cortexError', () => {
      const err = new Error('EMPTY_QUERY: manually constructed');
      expect(isCortexError(err, ErrorCodes.EMPTY_QUERY)).toBe(true);
    });

    it('handles error with non-string message', () => {
      const err = { message: 123 };
      expect(isCortexError(err, ErrorCodes.EMPTY_QUERY)).toBe(false);
    });
  });
});