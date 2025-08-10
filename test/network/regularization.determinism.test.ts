import Architect from '../../src/architecture/architect';
import Network from '../../src/architecture/network';

describe('Deterministic stochastic regularization', () => {
  it('stochastic depth skip pattern reproducible with seed', () => {
    const net1 = Architect.perceptron(4,8,8,8,2); // 3 hidden layers
    const net2 = Architect.perceptron(4,8,8,8,2);
    net1.setStochasticDepth([0.9,0.5,0.1]);
    net2.setStochasticDepth([0.9,0.5,0.1]);
    net1.setSeed(1234);
    net2.setSeed(1234);
    const inp = [0.1,0.2,0.3,0.4];
    const patterns1:number[][] = [];
    const patterns2:number[][] = [];
    for (let i=0;i<20;i++) {
      net1.activate(inp,true);
      net2.activate(inp,true);
      patterns1.push((net1 as any)._lastSkippedLayers.slice());
      patterns2.push((net2 as any)._lastSkippedLayers.slice());
    }
    expect(patterns1).toEqual(patterns2);
  });

  it('dropout reproducible with seed', () => {
    const net1 = new Network(4,2);
    const net2 = new Network(4,2);
    net1.dropout = 0.5; net2.dropout = 0.5;
    net1.setSeed(99); net2.setSeed(99);
    const inp=[0.2,0.1,0.05,0.9];
    const masks1:number[][]=[]; const masks2:number[][]=[];
    for (let i=0;i<15;i++) {
      net1.activate(inp,true); net2.activate(inp,true);
      const h1 = net1.nodes.filter(n=>n.type==='hidden').map(n=>n.mask);
      const h2 = net2.nodes.filter(n=>n.type==='hidden').map(n=>n.mask);
      masks1.push(h1); masks2.push(h2);
    }
    expect(masks1).toEqual(masks2);
  });

  it('per-hidden-layer weight noise applied with correct stds (deterministic)', () => {
    const net = Architect.perceptron(3,5,7,9,2); // 3 hidden layers => stds array length 3
    net.enableWeightNoise({ perHiddenLayer:[0.0, 0.05, 0.1] });
    net.setSeed(7);
    const input=[0.01,0.02,0.03];
    // Run several times; check that connections originating from hidden layer 1 have zero noise, hidden2 moderate, hidden3 higher (by absolute mean)
    const noiseByLayer: number[][] = [[],[],[]];
    for (let i=0;i<30;i++) {
      net.activate(input,true);
      for (const c of net.connections) {
        // find from-layer
        let fromLayer=-1;
        if (net.layers) {
          for (let li=0; li<net.layers.length; li++) {
            if (net.layers[li].nodes.includes(c.from)) { fromLayer=li; break; }
          }
        }
        if (fromLayer>0 && fromLayer < (net.layers!.length-1)) { // hidden
          const hiddenIdx = fromLayer -1;
            noiseByLayer[hiddenIdx].push((c as any)._wnLast || 0);
        }
      }
    }
    const avgAbs = noiseByLayer.map(arr => arr.reduce((a,b)=>a+Math.abs(b),0)/(arr.length||1));
    expect(avgAbs[0]).toBeLessThan(1e-6); // ~0
    expect(avgAbs[1]).toBeGreaterThan(0.0001);
    expect(avgAbs[2]).toBeGreaterThan(avgAbs[1]);
  });
});
