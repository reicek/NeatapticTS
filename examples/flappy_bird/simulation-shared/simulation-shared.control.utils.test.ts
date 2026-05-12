import { resolveFlapDecision } from './simulation-shared.control.utils';

describe('resolveFlapDecision', () => {
  it('treats typed-array two-output policies as competitive flap versus no-flap outputs', () => {
    expect(resolveFlapDecision(new Float64Array([0.25, 0.75]))).toBe(true);
  });
});
