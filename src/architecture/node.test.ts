import Node, { resolvePrimitiveIntent } from './node';

describe('architecture node root facade', () => {
  it('exports Node as a function', () => {
    expect(typeof Node).toBe('function');
  });

  it('exports resolvePrimitiveIntent as a function', () => {
    expect(typeof resolvePrimitiveIntent).toBe('function');
  });
});
