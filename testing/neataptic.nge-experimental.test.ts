import { nge } from 'neataptic';

describe('nge experimental namespace', () => {
  it('exports the nge namespace', () => {
    expect(nge).toBeDefined();
  });

  it('exposes the adult sub-namespace', () => {
    expect(nge?.adult).toBeDefined();
  });

  it('exposes advanceAdultState as a function under nge.adult', () => {
    expect(typeof nge?.adult?.advanceAdultState).toBe('function');
  });

  it('exposes the lifecycle sub-namespace', () => {
    expect(nge?.lifecycle).toBeDefined();
  });

  it('exposes runNgeLifecycle as a function under nge.lifecycle', () => {
    expect(typeof nge?.lifecycle?.runNgeLifecycle).toBe('function');
  });

  it('exposes the assimilation sub-namespace', () => {
    expect(nge?.assimilation).toBeDefined();
  });

  it('exposes assimilateEquilibriumCandidate as a function under nge.assimilation', () => {
    expect(typeof nge?.assimilation?.assimilateEquilibriumCandidate).toBe(
      'function',
    );
  });

  it('exposes the juvenile sub-namespace', () => {
    expect(nge?.juvenile).toBeDefined();
  });

  it('exposes NGE_JUVENILE_DEFAULT_FOCUS_WEIGHTS under nge.juvenile', () => {
    expect(nge?.juvenile?.NGE_JUVENILE_DEFAULT_FOCUS_WEIGHTS).toBeDefined();
  });
});
