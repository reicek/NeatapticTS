import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import { selection as selectionMethods } from '../../src/methods/selection';

// Single expectation per test.

describe('ensureNoDeadEnds', () => {
  test('repairs missing connections for input/output/hidden', async () => {
    const neat = new Neat(2, 1, (n: Network) => 1, {
      popsize: 1,
      seed: 3,
      minHidden: 1,
      speciation: false,
    });
    await neat.evaluate();
    const net = neat.population[0];
    // Remove all connections
    [...net.connections].forEach((c) => net.disconnect(c.from, c.to));
    const hidden = net.nodes.find((n) => n.type === 'hidden');
    // Call private method to repair
    (neat as any).ensureNoDeadEnds(net);
    const hasAny =
      net.connections.length > 0 &&
      (!hidden ||
        (hidden.connections.in.length > 0 &&
          hidden.connections.out.length > 0));
    expect(hasAny).toBe(true);
  });
});

describe('Dynamic compatibility threshold controller', () => {
  test('clamps to minThreshold when target species far exceeds observed', async () => {
    const neat = new Neat(3, 1, (n: Network) => 1, {
      popsize: 10,
      seed: 11,
      speciation: true,
      targetSpecies: 50,
      compatibilityThreshold: 3,
      compatAdjust: { kp: 10, ki: 0, minThreshold: 1, maxThreshold: 10 },
    });
    await neat.evaluate();
    expect(neat.options.compatibilityThreshold).toBe(1);
  });
  test('clamps to maxThreshold when observed species exceed tiny target', async () => {
    const neat = new Neat(3, 1, (n: Network) => n.connections.length, {
      popsize: 12,
      seed: 12,
      speciation: true,
      targetSpecies: 1,
      compatibilityThreshold: 0.5,
      compatAdjust: { kp: 10, ki: 0, minThreshold: 0.1, maxThreshold: 2 },
    });
    await neat.evaluate();
    expect(neat.options.compatibilityThreshold).toBe(2);
  });
});

describe('Selection fallbacks and cases', () => {
  test('legacy POWER constant selection returns member', async () => {
    const neat = new Neat(2, 1, (n: Network) => n.connections.length, {
      popsize: 6,
      seed: 21,
      speciation: false,
      selection: selectionMethods.POWER,
    });
    await neat.evaluate();
    const parent = neat.getParent();
    expect(neat.population.includes(parent)).toBe(true);
  });
  test('tournament size exceeding population throws error', async () => {
    const neat = new Neat(2, 1, (n: Network) => n.connections.length, {
      popsize: 5,
      seed: 22,
      speciation: false,
      selection: { name: 'TOURNAMENT', size: 50, probability: 0.5 } as any,
    });
    await neat.evaluate();
    expect(() => neat.getParent()).toThrow();
  });
  test('fitness proportionate handles negative scores', async () => {
    const neat = new Neat(2, 1, (n: Network) => 1, {
      popsize: 5,
      seed: 23,
      speciation: false,
      selection: { name: 'FITNESS_PROPORTIONATE' } as any,
    });
    await neat.evaluate();
    // Assign mixed negative/positive
    neat.population.forEach((g, i) => (g.score = i === 0 ? -5 : i - 2));
    const p = neat.getParent();
    expect(neat.population.includes(p)).toBe(true);
  });
});

describe('State export/import', () => {
  test('exportState/importState preserves generation', async () => {
    const neat = new Neat(2, 1, (n: Network) => n.connections.length, {
      popsize: 4,
      seed: 31,
      speciation: false,
    });
    await neat.evaluate();
    await neat.evolve(); // advance generation
    const bundle = neat.exportState();
    const restored = (Neat as any).importState(
      bundle,
      (n: Network) => n.connections.length
    ) as Neat;
    expect(restored.generation).toBe(neat.generation);
  });
  test('toJSON/fromJSON preserves nextGlobalInnovation', async () => {
    const neat = new Neat(2, 1, (n: Network) => n.connections.length, {
      popsize: 3,
      seed: 32,
      speciation: false,
    });
    await neat.evaluate();
    await neat.evolve();
    const meta = neat.toJSON();
    const neat2 = (Neat as any).fromJSON(
      meta,
      (n: Network) => n.connections.length
    ) as Neat;
    expect(neat2.toJSON().nextGlobalInnovation).toBe(meta.nextGlobalInnovation);
  });
});
