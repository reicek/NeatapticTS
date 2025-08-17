/**
 * Tests covering objective events & lineage block in buildTelemetryEntry.
 * Single expectation per test.
 */
import { buildTelemetryEntry } from '../../src/neat/neat.telemetry';

/** Helper to construct a minimal Neat-like context */
function makeCtx() {
  /** dummy genomes */
  const population = [
    {
      score: 1,
      nodes: [{}, {}],
      connections: [],
      _moRank: 0,
      _depth: 2,
      _parents: [1, 2],
    },
    {
      score: 0.5,
      nodes: [{}, {}],
      connections: [],
      _moRank: 1,
      _depth: 1,
      _parents: [1],
    },
  ];
  /** context object mimicking internal Neat state */
  const ctx: any = {
    generation: 3,
    population,
    _species: [],
    _operatorStats: new Map(),
    _diversityStats: {},
    _getObjectives: () => [],
    _getRNG: () => () => 0.42,
    options: {
      multiObjective: { enabled: false },
      telemetry: { complexity: false },
    },
  };
  return ctx;
}

describe('Telemetry objective events & lineage', () => {
  test('buildTelemetryEntry consumes pending objective events', () => {
    // Arrange
    const ctx = makeCtx();
    ctx._pendingObjectiveAdds = ['a', 'b'];
    ctx._pendingObjectiveRemoves = ['c'];
    ctx._objectiveEvents = [];
    ctx._lineageEnabled = true;
    // Act
    const entry = buildTelemetryEntry.call(ctx, ctx.population[0]);
    // Assert
    expect(((entry as any).objEvents as any[]).length).toBe(3);
  });
});
