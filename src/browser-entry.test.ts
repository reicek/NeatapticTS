import {
  Architect,
  config,
  Connection,
  formatConstructSummary,
  Group,
  Layer,
  methods,
  Neat,
  Network,
  Node,
} from './browser-entry';
import * as browserEntry from './browser-entry';

describe('browser entry facade', () => {
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

  it('does not export multi', () => {
    expect('multi' in browserEntry).toBe(false);
  });
});
