/**
 * @module tokenizer.test
 * @description Branch-coverage tests for sanitizeFtsQuery.
 *
 * Exercises every branch in tokenizer.mjs:
 * - Nullish coalescing for `raw` (null/undefined vs non-null)
 * - Empty/whitespace input (length === 0 true branch)
 * - Punctuation-only input (match returns null → ?? true, tokens.length === 0)
 * - Code identifiers: dotted, snake_case, camelCase (isCodeIdentifier true via
 *   each pattern alternative)
 * - Acronyms (isCodeIdentifier false, isAcronym true)
 * - Plain words and numeric tokens (both identifier and acronym false)
 * - Mixed queries combining all token types
 */

import { sanitizeFtsQuery } from '../tokenizer.mjs';

describe('sanitizeFtsQuery', () => {
  describe('null and undefined input', () => {
    it('returns empty string for null (raw ?? "" nullish branch)', () => {
      expect(sanitizeFtsQuery(null)).toBe('');
    });

    it('returns empty string for undefined (raw ?? "" nullish branch)', () => {
      expect(sanitizeFtsQuery(undefined)).toBe('');
    });
  });

  describe('empty and whitespace input', () => {
    it('returns empty string for empty string (input.length === 0 true)', () => {
      expect(sanitizeFtsQuery('')).toBe('');
    });

    it('returns empty string for whitespace-only input (trims to empty)', () => {
      expect(sanitizeFtsQuery('   \t\n  ')).toBe('');
    });
  });

  describe('punctuation-only input (no word tokens)', () => {
    it('returns empty string when match returns null (?? [] true, tokens.length === 0 true)', () => {
      expect(sanitizeFtsQuery('!!! ??? ...')).toBe('');
    });
  });

  describe('code identifiers — isCodeIdentifier true', () => {
    it('quotes dotted identifiers (DOTTED_OR_SNAKE true, || short-circuit)', () => {
      expect(sanitizeFtsQuery('network.activate')).toBe('"network.activate"');
    });

    it('quotes snake_case identifiers (DOTTED_OR_SNAKE true via underscore)', () => {
      expect(sanitizeFtsQuery('snake_case_function')).toBe('"snake_case_function"');
    });

    it('quotes camelCase identifiers (DOTTED false, CAMEL_CASE true)', () => {
      expect(sanitizeFtsQuery('findTheNEATSelectionCode')).toBe('"findTheNEATSelectionCode"');
    });
  });

  describe('acronyms — isAcronym true', () => {
    it('emits all-caps acronyms as exact terms without wildcard', () => {
      expect(sanitizeFtsQuery('NEAT')).toBe('NEAT');
    });

    it('emits acronyms with trailing digits as exact terms', () => {
      expect(sanitizeFtsQuery('API2')).toBe('API2');
    });
  });

  describe('plain words — isCodeIdentifier false, isAcronym false', () => {
    it('appends prefix wildcard to plain words', () => {
      expect(sanitizeFtsQuery('selection')).toBe('selection*');
    });

    it('appends prefix wildcard to numeric-only tokens', () => {
      expect(sanitizeFtsQuery('123')).toBe('123*');
    });
  });

  describe('mixed queries', () => {
    it('formats acronym, plain words together', () => {
      expect(sanitizeFtsQuery('NEAT selection code')).toBe('NEAT selection* code*');
    });

    it('formats dotted identifier, acronym, and plain word together', () => {
      expect(sanitizeFtsQuery('network.activate NEAT selection')).toBe('"network.activate" NEAT selection*');
    });
  });

  describe('non-string input (raw ?? "" non-nullish branch)', () => {
    it('converts numbers to strings and processes them', () => {
      expect(sanitizeFtsQuery(123)).toBe('123*');
    });
  });
});