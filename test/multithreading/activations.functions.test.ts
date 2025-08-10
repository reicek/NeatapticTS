import Multi from '../../src/multithreading/multi';

describe('Multi activations coverage', () => {
  it('evaluates all compiled activation functions', () => {
    const xs = [-2, -1, 0, 1, 2];
    Multi.activations.forEach((fn) => {
      xs.forEach((x) => {
        const y = fn(x);
        expect(Number.isFinite(y)).toBe(true);
      });
    });
  });

  it('evaluates static activation helpers', () => {
    const x = 0.5;
    expect(typeof Multi.logistic(x)).toBe('number');
    expect(typeof Multi.tanh(x)).toBe('number');
    expect(typeof Multi.identity(x)).toBe('number');
    expect(typeof Multi.step(x)).toBe('number');
    expect(typeof Multi.relu(x)).toBe('number');
    expect(typeof Multi.softsign(x)).toBe('number');
    expect(typeof Multi.sinusoid(x)).toBe('number');
    expect(typeof Multi.gaussian(x)).toBe('number');
    expect(typeof Multi.bentIdentity(x)).toBe('number');
    expect(typeof Multi.bipolar(x)).toBe('number');
    expect(typeof Multi.bipolarSigmoid(x)).toBe('number');
    expect(typeof Multi.hardTanh(x)).toBe('number');
    expect(typeof Multi.absolute(x)).toBe('number');
    expect(typeof Multi.inverse(x)).toBe('number');
    expect(typeof Multi.selu(x)).toBe('number');
    expect(typeof Multi.softplus(x)).toBe('number');
  });

  it('activates a simple serialized network including a gated incoming connection', () => {
    // inputs=1, outputs=1
    const data: number[] = [1, 1];
    // Node 0: index=0, bias=0, squash=2(identity), selfweight=0, selfgater=-1
    data.push(0, 0, 2, 0, -1);
    // One incoming: from input index 0, weight 0.5, gater -1 (no gate)
    data.push(0, 0.5, -1);
    // Sentinel end of connections
    data.push(-2);

    const input = [2];
    const A = [0];
    const S = [0];
    const F = Multi.activations; // includes identity at index 2

    const out = Multi.activateSerializedNetwork(input, A, S, data, F as any);
    expect(out.length).toBe(1);
    // identity( bias + 0.5*input ) = 1.0
    expect(out[0]).toBeCloseTo(1.0, 10);
  });
});
