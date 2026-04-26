import Network, { formatConstructSummary } from './network';

describe('architecture network root facade', () => {
  it('exports Network as a function', () => {
    expect(typeof Network).toBe('function');
  });

  it('exports formatConstructSummary as a function', () => {
    expect(typeof formatConstructSummary).toBe('function');
  });
});
