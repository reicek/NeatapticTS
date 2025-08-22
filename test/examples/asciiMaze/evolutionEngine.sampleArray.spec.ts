import { EvolutionEngine } from './evolutionEngine';

describe('EvolutionEngine.#sampleArray', () => {
  it('returns k items and uses replacement semantics', () => {
    process.env.NODE_ENV = 'test';
    const api: any = (EvolutionEngine as any)._testExpose();
    const source = [1, 2, 3];
    const sample = api.sampleArray(source, 10);
    // Single expectation: correct length
    expect(sample.length).toBe(10);
  });
});
