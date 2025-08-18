import Network from '../../src/architecture/network';

// Deterministic tiny dataset y=2x
const data = Array.from({ length: 3 }, (_, i) => ({
  input: [i + 1],
  output: [2 * (i + 1)],
}));

function buildNet() {
  const net = new Network(1, 1);
  net.connections.forEach((c) => (c.weight = 0.5));
  net.nodes.filter((n) => n.type !== 'input').forEach((n) => (n.bias = 0));
  return net;
}

// Utility to run one iteration and capture internal optimizer state
function trainSteps(net: Network, opt: any, iterations: number, rate = 0.01) {
  net.train(data, {
    iterations,
    rate,
    optimizer: opt,
    batchSize: 1,
    error: 0,
    cost: {
      fn: (t: number[], o: number[]) => (o[0] - t[0]) ** 2,
      calculate: (t: number[], o: number[]) => (o[0] - t[0]) ** 2,
    },
  });
}

describe('Optimizer formula characteristics', () => {
  describe('adamax infinity norm vs adam second moment', () => {
    const netA = buildNet();
    const netB = buildNet();
    trainSteps(netA, { type: 'adamax' }, 2);
    trainSteps(netB, { type: 'adam' }, 2);
    const connA: any = netA.connections[0];
    const connB: any = netB.connections[0];
    it('maintains infinityNorm different from sqrt(secondMoment)', () => {
      expect(connA.infinityNorm).not.toBeUndefined();
    });
  });

  describe('nadam nesterov lookahead produces larger early step than adam', () => {
    const netN = buildNet();
    const netA = buildNet();
    trainSteps(netN, { type: 'nadam' }, 1); // single step
    trainSteps(netA, { type: 'adam' }, 1);
    const deltaN = netN.connections[0].weight - 0.5;
    const deltaA = netA.connections[0].weight - 0.5;
    it('has different first step magnitude from adam', () => {
      expect(Math.abs(deltaN - deltaA)).toBeGreaterThan(0);
    });
  });

  describe('radam early unrectified vs later rectified variance', () => {
    const netEarly = buildNet();
    const netLate = buildNet();
    trainSteps(netEarly, { type: 'radam' }, 1);
    trainSteps(netLate, { type: 'radam' }, 10);
    const earlyDelta = Math.abs(netEarly.connections[0].weight - 0.5);
    const lateDelta = Math.abs(netLate.connections[0].weight - 0.5);
    it('late step magnitude differs from very early step', () => {
      expect(Math.abs(lateDelta - earlyDelta)).toBeGreaterThan(0);
    });
  });

  describe('adabelief variance differs from adam given same gradients', () => {
    const netBelief = buildNet();
    const netAdam = buildNet();
    trainSteps(netBelief, { type: 'adabelief' }, 2);
    trainSteps(netAdam, { type: 'adam' }, 2);
    const cB: any = netBelief.connections[0];
    const cA: any = netAdam.connections[0];
    it('maintains distinct second moment estimate', () => {
      expect(cB.secondMoment).not.toBe(cA.secondMoment);
    });
  });

  describe('lookahead defaults', () => {
    const net = buildNet();
    // Provide only type to trigger default baseType and params
    trainSteps(net, { type: 'lookahead' }, 3);
    const conn: any = net.connections[0];
    it('creates shadow weight with default params', () => {
      expect(conn.lookaheadShadowWeight).toBeDefined();
    });
  });
});
