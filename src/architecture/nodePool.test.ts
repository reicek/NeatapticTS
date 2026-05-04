import nodePoolFacade, {
  acquireNode,
  nodePoolStats,
  releaseNode,
  resetNodePool,
} from './nodePool';

describe('architecture nodePool root facade', () => {
  afterEach(() => {
    resetNodePool();
  });

  it('exports acquireNode as a function', () => {
    expect(typeof acquireNode).toBe('function');
  });

  it('exports releaseNode as a function', () => {
    expect(typeof releaseNode).toBe('function');
  });

  it('exports nodePoolStats as a function', () => {
    expect(typeof nodePoolStats).toBe('function');
  });

  it('exports resetNodePool as a function', () => {
    expect(typeof resetNodePool).toBe('function');
  });

  it('exports the default facade object with function members', () => {
    expect(
      Object.values(nodePoolFacade).every(
        (member) => typeof member === 'function',
      ),
    ).toBe(true);
  });
});
