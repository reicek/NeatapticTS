import Network from '../../src/architecture/network';
import Node from '../../src/architecture/node';
import * as methods from '../../src/methods/methods';
import { importFromONNX } from '../../src/architecture/onnx';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('ONNX Import', () => {
  it('imports a 1-1 input-output network from ONNX', () => {
    // Arrange: export a simple network
    const net = new Network(1, 1);
    net.nodes[1].squash = methods.Activation.tanh;
    const onnx = net.toONNX();
    // Act: import from ONNX
    const imported = importFromONNX(onnx);
    // Assert: structure and weights
    expect(imported.nodes.length).toBe(2);
    expect(imported.nodes[1].squash).toBe(methods.Activation.tanh);
    expect(imported.nodes[1].bias).toBeCloseTo(net.nodes[1].bias);
  });

  it('imports a 2-2-1 MLP from ONNX and preserves weights/biases', () => {
    // Arrange
    const net = Network.createMLP(2, [2], 1);
    net.nodes[2].bias = 0.5;
    net.nodes[3].bias = -0.5;
    net.nodes[4].bias = 1.0;
    net.nodes[2].connections.in[0].weight = 0.1;
    net.nodes[2].connections.in[1].weight = 0.2;
    net.nodes[3].connections.in[0].weight = 0.3;
    net.nodes[3].connections.in[1].weight = 0.4;
    net.nodes[4].connections.in[0].weight = 0.5;
    net.nodes[4].connections.in[1].weight = 0.6;
    net.nodes[2].squash = methods.Activation.relu;
    net.nodes[3].squash = methods.Activation.relu;
    net.nodes[4].squash = methods.Activation.sigmoid;
    const onnx = net.toONNX();
    // Act
    const imported = importFromONNX(onnx);
    // Assert
    expect(imported.nodes.length).toBe(5);
    expect(imported.nodes[2].bias).toBeCloseTo(0.5);
    expect(imported.nodes[3].bias).toBeCloseTo(-0.5);
    expect(imported.nodes[4].bias).toBeCloseTo(1.0);
    expect(imported.nodes[2].connections.in[0].weight).toBeCloseTo(0.1);
    expect(imported.nodes[2].connections.in[1].weight).toBeCloseTo(0.2);
    expect(imported.nodes[3].connections.in[0].weight).toBeCloseTo(0.3);
    expect(imported.nodes[3].connections.in[1].weight).toBeCloseTo(0.4);
    expect(imported.nodes[4].connections.in[0].weight).toBeCloseTo(0.5);
    expect(imported.nodes[4].connections.in[1].weight).toBeCloseTo(0.6);
    expect(imported.nodes[2].squash).toBe(methods.Activation.relu);
    expect(imported.nodes[3].squash).toBe(methods.Activation.relu);
    expect(imported.nodes[4].squash).toBe(methods.Activation.sigmoid);
  });
});