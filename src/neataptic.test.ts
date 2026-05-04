import {
  Architect,
  config,
  Connection,
  formatConstructSummary,
  Group,
  Layer,
  methods,
  multi,
  Neat,
  Network,
  Node,
} from './neataptic';

describe('neataptic root facade', () => {
  it('exports Neat as a function', () => {
    expect(typeof Neat).toBe('function');
  });

  it('exports Network as a function', () => {
    expect(typeof Network).toBe('function');
  });

  it('exports formatConstructSummary as a function', () => {
    expect(typeof formatConstructSummary).toBe('function');
  });

  it('exports Node as a function', () => {
    expect(typeof Node).toBe('function');
  });

  it('exports Layer as a function', () => {
    expect(typeof Layer).toBe('function');
  });

  it('exports Group as a function', () => {
    expect(typeof Group).toBe('function');
  });

  it('exports Connection as a function', () => {
    expect(typeof Connection).toBe('function');
  });

  it('exports Architect as a function', () => {
    expect(typeof Architect).toBe('function');
  });

  it('exports methods as an object', () => {
    expect(typeof methods).toBe('object');
  });

  it('exports config as an object', () => {
    expect(typeof config).toBe('object');
  });

  it('exports multi as an object', () => {
    expect(typeof multi).toBe('object');
  });
});
