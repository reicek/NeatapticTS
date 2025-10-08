import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import Node from '../../src/architecture/node';
import * as methods from '../../src/methods/methods';

type NodeSplitRecord = {
  inInnov: number;
  outInnov: number;
  newNodeGeneId: number;
};

// Tests focused on innovation reuse (ADD_NODE / ADD_CONN) and cycle safety.
// Each test has a single expectation per project testing guidelines.

describe('Innovation Reuse', () => {
  describe('ADD_NODE (connection split) innovation reuse', () => {
    const seed = 123;
    const fitness = () => 0;

    describe('initial split records innovations', () => {
      let neat: Neat;
      let nodeSplitMap: Map<string, NodeSplitRecord>;
      let registrySize: number;

      beforeAll(async () => {
        neat = new Neat(2, 1, fitness, {
          popsize: 1,
          seed,
          mutation: [methods.mutation.ADD_NODE],
          mutationRate: 1,
          mutationAmount: 1,
        });
        await neat.mutate(); // perform one ADD_NODE on the single genome
        nodeSplitMap = Reflect.get(neat, '_nodeSplitInnovations') as Map<
          string,
          NodeSplitRecord
        >;
        registrySize = nodeSplitMap.size;
      });

      test('registry has at least one entry', () => {
        expect(registrySize).toBeGreaterThan(0);
      });
    });

    describe('second split of same connection reuses innovation ids', () => {
      let neat: Neat;
      let firstRegistry: Map<string, NodeSplitRecord>;
      let secondRegistry: Map<string, NodeSplitRecord>;
      let base: Network;
      let reused: boolean;

      beforeAll(async () => {
        neat = new Neat(2, 1, fitness, {
          popsize: 1,
          seed: seed + 1, // different seed to avoid sharing RNG sequence with previous block
          mutation: [methods.mutation.ADD_NODE],
          mutationRate: 1,
          mutationAmount: 1,
        });
        // First mutation creates mapping
        await neat.mutate();
        firstRegistry = new Map(
          Reflect.get(neat, '_nodeSplitInnovations') as Map<
            string,
            NodeSplitRecord
          >,
        );
        // Reset genome to base network (same from->to gene ids stay constant across runs because geneId is global)
        base = new Network(2, 1, { minHidden: 0 });
        neat.population[0] = base;
        // Invoke internal reuse mutation directly to guarantee using same chosen connection (first enabled)
        const mutateAddNodeReuse = Reflect.get(neat, '_mutateAddNodeReuse') as (
          genome: Network,
        ) => void;
        mutateAddNodeReuse.call(neat, base);
        secondRegistry = Reflect.get(neat, '_nodeSplitInnovations') as Map<
          string,
          NodeSplitRecord
        >;
        // Compare one entry innovations equality
        reused = Array.from(firstRegistry.entries()).every(([key, record]) => {
          const reusedRecord = secondRegistry.get(key);
          return (
            !!reusedRecord &&
            record.inInnov === reusedRecord.inInnov &&
            record.outInnov === reusedRecord.outInnov &&
            record.newNodeGeneId === reusedRecord.newNodeGeneId
          );
        });
      });

      test('innovations reused for identical split', () => {
        expect(reused).toBe(true);
      });
    });
  });

  describe('ADD_CONN innovation reuse', () => {
    const fitness = () => 0;

    describe('adding a new connection assigns an innovation id', () => {
      const neat = new Neat(2, 1, fitness, { popsize: 1, seed: 50 });
      const net: Network = neat.population[0];
      // Create two hidden nodes with no connection between them so only one possible forward add
      const h1 = new Node('hidden');
      const h2 = new Node('hidden');
      net.nodes.splice(net.nodes.length - net.output, 0, h1, h2); // insert before outputs to preserve feedforward order
      // Remove any accidental connections between them if present (should not be)
      net.connections = net.connections.filter(
        (c) => !(c.from === h1 && c.to === h2),
      );
      const mutateAddConnReuse = Reflect.get(neat, '_mutateAddConnReuse') as (
        genome: Network,
      ) => void;
      mutateAddConnReuse.call(neat, net);
      const connInnovations = Reflect.get(neat, '_connInnovations') as Map<
        string,
        number
      >;
      const keyCount = connInnovations.size;
      test('connection innovation registry updated', () => {
        expect(keyCount).toBeGreaterThan(0);
      });
    });

    describe('re-adding same connection reuses innovation id', () => {
      const neat = new Neat(2, 1, fitness, { popsize: 1, seed: 77 });
      const net: Network = neat.population[0];
      const h1 = new Node('hidden');
      const h2 = new Node('hidden');
      net.nodes.splice(net.nodes.length - net.output, 0, h1, h2);
      const mutateAddConnReuse = Reflect.get(neat, '_mutateAddConnReuse') as (
        genome: Network,
      ) => void;
      mutateAddConnReuse.call(neat, net);
      const conn = net.connections.find(
        (c) => (c.from === h1 && c.to === h2) || (c.from === h2 && c.to === h1),
      );
      const innovBefore = conn?.innovation;
      if (conn) net.disconnect(conn.from, conn.to);
      mutateAddConnReuse.call(neat, net); // should recreate same pair (only viable)
      const conn2 = net.connections.find(
        (c) => (c.from === h1 && c.to === h2) || (c.from === h2 && c.to === h1),
      );
      const innovAfter = conn2?.innovation;
      const reused = innovBefore === innovAfter;
      test('innovation id reused after deletion and re-add', () => {
        expect(reused).toBe(true);
      });
    });
  });
});

describe('Cycle Safety', () => {
  describe('edge that would create a cycle is skipped when acyclic enforced', () => {
    const fitness = () => 0;
    const neat = new Neat(1, 1, fitness, { popsize: 1, seed: 5 });
    const net: Network = neat.population[0];
    // Add two hidden nodes H1, H2
    const h1 = new Node('hidden');
    const h2 = new Node('hidden');
    net.nodes.splice(net.nodes.length - net.output, 0, h1, h2);
    // Create path h2 -> input -> h1 (so adding h1->h2 would form cycle)
    const inputNode = net.nodes.find((n) => n.type === 'input')!;
    net.connect(h2, inputNode, 1); // backward edge allowed before acyclic enforcement
    net.connect(inputNode, h1, 1);
    // Now enforce acyclicity for subsequent mutation attempts
    net.setEnforceAcyclic(true);
    const before = net.connections.length;
    const mutateAddConnReuse = Reflect.get(neat, '_mutateAddConnReuse') as (
      genome: Network,
    ) => void;
    for (let i = 0; i < 10; i++) mutateAddConnReuse.call(neat, net);
    const after = net.connections.length;
    const prevented =
      after === before ||
      !net.connections.some((c) => c.from === h1 && c.to === h2);
    test('cycle-forming connection not added', () => {
      expect(prevented).toBe(true);
    });
  });
});
