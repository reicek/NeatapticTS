/**
 * Narrow smoke checks for the three starter examples.
 *
 * These tests guard the public export surface and the minimal output shape of
 * each starter example so that renames or structural breakage are caught
 * independently of the deeper behavioral tests in each example's own test file.
 *
 * Speed contract: this file must stay fast (no evolution loops).
 * The evolveXor export checks intentionally stop at typeof — behavioral
 * coverage is owned by `evolveXor/evolveXor.test.ts`.
 */

import {
  formatHelloNetworkExampleResult,
  runHelloNetworkExample,
} from './helloNetwork/index';
import {
  formatEvolveXorExampleResult,
  runEvolveXorExample,
} from './evolveXor/index';
import {
  formatSequenceResetExampleResult,
  runSequenceResetExample,
} from './sequenceReset/index';

// ─── helloNetwork ────────────────────────────────────────────────────────────

describe('starter smoke: helloNetwork', () => {
  it('exports runHelloNetworkExample as a callable function', () => {
    expect(typeof runHelloNetworkExample).toBe('function');
  });

  it('exports formatHelloNetworkExampleResult as a callable function', () => {
    expect(typeof formatHelloNetworkExampleResult).toBe('function');
  });

  it('format output opens with the Hello Network starter title', () => {
    expect(
      formatHelloNetworkExampleResult(runHelloNetworkExample()).startsWith(
        'Hello Network',
      ),
    ).toBe(true);
  });
});

// ─── evolveXor ───────────────────────────────────────────────────────────────

describe('starter smoke: evolveXor', () => {
  it('exports runEvolveXorExample as a callable function', () => {
    expect(typeof runEvolveXorExample).toBe('function');
  });

  it('exports formatEvolveXorExampleResult as a callable function', () => {
    expect(typeof formatEvolveXorExampleResult).toBe('function');
  });
});

// ─── sequenceReset ───────────────────────────────────────────────────────────

describe('starter smoke: sequenceReset', () => {
  it('exports runSequenceResetExample as a callable function', () => {
    expect(typeof runSequenceResetExample).toBe('function');
  });

  it('exports formatSequenceResetExampleResult as a callable function', () => {
    expect(typeof formatSequenceResetExampleResult).toBe('function');
  });

  it('format output opens with the Sequence Reset starter title', () => {
    expect(
      formatSequenceResetExampleResult(runSequenceResetExample()).startsWith(
        'Sequence Reset',
      ),
    ).toBe(true);
  });
});
