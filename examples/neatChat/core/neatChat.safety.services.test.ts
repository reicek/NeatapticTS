import {
  checkSafety,
  isDegenerateResponse,
  isRepetitionCollapse,
  isUnknownToken,
} from './neatChat.safety.services';
import { createNeatChatSession } from './neatChat.session.services';
import type {
  SafetyCheckResult,
  SafetyViolation,
} from './neatChat.safety.types';

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

/** Terms wide enough to cover all test inputs. */
const SAFETY_TEST_RETAINED_TERMS = [
  'hello',
  'there',
  'how',
  'are',
  'you',
  'was',
  'your',
  'the',
  'quick',
  'brown',
  'fox',
  'jumps',
  'over',
  'lazy',
  'dog',
  'world',
  'good',
  'morning',
];

function makeSession() {
  return createNeatChatSession({
    corpusRetainedTerms: SAFETY_TEST_RETAINED_TERMS,
  });
}

// ---------------------------------------------------------------------------
// isDegenerateResponse
// ---------------------------------------------------------------------------

describe('isDegenerateResponse', () => {
  it('returns true for an empty string', () => {
    expect(isDegenerateResponse('')).toBe(true);
  });

  it('returns true for a whitespace-only string', () => {
    expect(isDegenerateResponse('   ')).toBe(true);
  });

  it('returns true for a single-token response', () => {
    expect(isDegenerateResponse('hello')).toBe(true);
  });

  it('returns false for a two-token response', () => {
    expect(isDegenerateResponse('hello there')).toBe(false);
  });

  it('returns false for a well-formed multi-token response', () => {
    expect(isDegenerateResponse('hello there how are you')).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// isRepetitionCollapse
// ---------------------------------------------------------------------------

describe('isRepetitionCollapse', () => {
  it('returns false for a non-repetitive response', () => {
    expect(isRepetitionCollapse('hello there how are you')).toBe(false);
  });

  it('returns true for a single-word repeat loop', () => {
    expect(isRepetitionCollapse('hello hello hello hello hello')).toBe(true);
  });

  it('returns true for a bigram repeat loop', () => {
    expect(
      isRepetitionCollapse('hello there hello there hello there hello there'),
    ).toBe(true);
  });

  it('returns false for a single-token input (no bigrams to check)', () => {
    expect(isRepetitionCollapse('hello')).toBe(false);
  });

  it('returns false for an empty string', () => {
    expect(isRepetitionCollapse('')).toBe(false);
  });

  it('accepts a custom threshold and returns false when repetition is below it', () => {
    // 'hello there hello world' — 1 repeated bigram out of 3 = 0.33 < 0.5
    expect(isRepetitionCollapse('hello there hello world', 0.5)).toBe(false);
  });

  it('accepts a custom threshold and returns true when repetition exceeds it', () => {
    // all bigrams repeat in 'hello hello hello hello'
    expect(isRepetitionCollapse('hello hello hello hello', 0.01)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// isUnknownToken
// ---------------------------------------------------------------------------

describe('isUnknownToken', () => {
  it('returns false for a token that is in the session vocabulary', () => {
    const session = makeSession();
    expect(isUnknownToken('hello', session)).toBe(false);
  });

  it('returns true for a token that is absent from the session vocabulary', () => {
    const session = makeSession();
    expect(isUnknownToken('xyzzy', session)).toBe(true);
  });

  it('returns true for an empty-string token', () => {
    const session = makeSession();
    expect(isUnknownToken('', session)).toBe(true);
  });

  it('is case-insensitive: lowercased form matches vocabulary', () => {
    const session = makeSession();
    expect(isUnknownToken('HELLO', session)).toBe(false);
  });

  it('returns false for the UNK special token itself (it is in the vocabulary)', () => {
    const session = makeSession();
    expect(isUnknownToken('UNK', session)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// checkSafety — degenerate-response
// ---------------------------------------------------------------------------

describe('checkSafety — degenerate-response', () => {
  it('returns ok:false with violation "degenerate-response" for an empty response', () => {
    const session = makeSession();
    const result: SafetyCheckResult = checkSafety(session, '');
    expect(result.ok).toBe(false);
  });

  it('sets violation to "degenerate-response" for a single-token response', () => {
    const session = makeSession();
    const result = checkSafety(session, 'hello');
    expect(result.violation).toBe<SafetyViolation>('degenerate-response');
  });

  it('includes a non-empty detail string for a degenerate response', () => {
    const session = makeSession();
    const result = checkSafety(session, '');
    expect(result.detail.length > 0).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// checkSafety — repetition-collapse
// ---------------------------------------------------------------------------

describe('checkSafety — repetition-collapse', () => {
  it('returns ok:false with violation "repetition-collapse" for a looping output', () => {
    const session = makeSession();
    const result: SafetyCheckResult = checkSafety(
      session,
      'hello hello hello hello hello hello',
    );
    expect(result.ok).toBe(false);
  });

  it('sets violation to "repetition-collapse" for bigram loops', () => {
    const session = makeSession();
    const result = checkSafety(
      session,
      'hello there hello there hello there hello there',
    );
    expect(result.violation).toBe<SafetyViolation>('repetition-collapse');
  });

  it('includes a non-empty detail string for repetition collapse', () => {
    const session = makeSession();
    const result = checkSafety(session, 'hello hello hello hello');
    expect(result.detail.length > 0).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// checkSafety — incomplete-fragment
// ---------------------------------------------------------------------------

describe('checkSafety — incomplete-fragment', () => {
  it('rejects a dangling prompt fragment that ends on a possessive token', () => {
    const session = makeSession();
    const result = checkSafety(session, 'how was your');
    expect(result).toMatchObject({
      ok: false,
      violation: 'incomplete-fragment',
    });
  });
});

// ---------------------------------------------------------------------------
// checkSafety — unknown-token
// ---------------------------------------------------------------------------

describe('checkSafety — unknown-token', () => {
  it('returns ok:false with violation "unknown-token" for an OOV token', () => {
    const session = makeSession();
    // 'xyzzy' is not in SAFETY_TEST_RETAINED_TERMS
    const result: SafetyCheckResult = checkSafety(session, 'hello xyzzy world');
    expect(result.ok).toBe(false);
  });

  it('sets violation to "unknown-token" when any response token is OOV', () => {
    const session = makeSession();
    const result = checkSafety(session, 'the quick outsider jumps');
    expect(result.violation).toBe<SafetyViolation>('unknown-token');
  });

  it('includes a non-empty detail string for unknown-token violation', () => {
    const session = makeSession();
    const result = checkSafety(session, 'hello zork world');
    expect(result.detail.length > 0).toBe(true);
  });

  it('does not throw — produces a SafetyCheckResult even for all-OOV input', () => {
    const session = makeSession();
    const result = checkSafety(session, 'zork blorb grue');
    expect(typeof result.ok).toBe('boolean');
  });
});

// ---------------------------------------------------------------------------
// checkSafety — passing responses
// ---------------------------------------------------------------------------

describe('checkSafety — passing responses', () => {
  it('returns ok:true and null violation for a well-formed in-vocabulary response', () => {
    const session = makeSession();
    const result: SafetyCheckResult = checkSafety(
      session,
      'hello there how are you',
    );
    expect(result.ok).toBe(true);
  });

  it('sets violation to null for a safe response', () => {
    const session = makeSession();
    const result = checkSafety(
      session,
      'the quick brown fox jumps over the lazy dog',
    );
    expect(result.violation).toBe(null);
  });

  it('sets detail to empty string for a passing response', () => {
    const session = makeSession();
    const result = checkSafety(session, 'good morning world');
    expect(result.detail).toBe('');
  });
});

// ---------------------------------------------------------------------------
// checkSafety — check priority order
// ---------------------------------------------------------------------------

describe('checkSafety — violation priority', () => {
  it('degenerate-response is checked before repetition-collapse (empty is degenerate first)', () => {
    const session = makeSession();
    // Empty string is degenerate; repetition-collapse also applies to single repeated token
    // but degenerate should win
    const result = checkSafety(session, '');
    expect(result.violation).toBe<SafetyViolation>('degenerate-response');
  });

  it('repetition-collapse is flagged before unknown-token when both apply', () => {
    const session = makeSession();
    // 'xyzzy xyzzy xyzzy xyzzy' is both repetition and OOV; plan orders repetition first
    const result = checkSafety(session, 'xyzzy xyzzy xyzzy xyzzy xyzzy');
    expect(result.violation).toBe<SafetyViolation>('repetition-collapse');
  });
});
