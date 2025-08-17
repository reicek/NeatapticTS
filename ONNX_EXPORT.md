# ONNX Export in NeatapticTS

NeatapticTS supports exporting trained neural networks to the ONNX (Open Neural Network Exchange) format, which enables interoperability with other machine learning frameworks and tools.

## Supported Network Types

ONNX export in NeatapticTS supports:

- **Strictly Layered, Fully Connected MLPs**:
  - Simple MLPs (input → output)
  - Single hidden layer MLPs (input → hidden → output)
  - Multi-hidden layer MLPs (input → hidden₁ → hidden₂ → ... → output)

Each layer must be fully connected to the next layer, with no skip or recurrent connections.

## Supported Activation Functions

The following activation functions are mapped to their ONNX equivalents:

- `Tanh`
- `Sigmoid` (or `Logistic`)
- `ReLU`
- `Identity` (or any unknown function)

Unsupported activation functions will be mapped to `Identity` with a warning.

## Usage Example

```typescript
import { Architect } from 'neataptic';

// Create and train a network
const network = new Architect.Perceptron(2, 3, 1);
network.train([
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
]);

// Export to ONNX format
const onnxModel = network.toONNX();

// Convert to JSON string for saving or transmitting
const onnxJson = JSON.stringify(onnxModel);
```

## Limitations

- The export functionality only supports strictly layered, fully connected MLPs.
- Networks with skip connections, recurrent connections, or non-MLP topologies are not supported.
- LSTM, GRU, and other advanced architectures are not supported.
- All nodes in the same layer must use the same activation function (enforced).

## Example Network Structures

### Simple MLP (input → output)

```
Input Layer (2 nodes) → Output Layer (1 node)
```

### Single Hidden Layer MLP

```
Input Layer (2 nodes) → Hidden Layer (3 nodes) → Output Layer (1 node)
```

### Multi-Hidden Layer MLP

```
Input Layer (2 nodes) → Hidden Layer 1 (4 nodes) → Hidden Layer 2 (3 nodes) → Output Layer (1 node)
```

## ONNX Model Structure

The exported ONNX model includes:

- Input and output tensor definitions with batch dimension
- Weight and bias initializers for each layer
- MatMul, Add, and Activation nodes for the computation graph
- Producer information and ONNX version metadata

## Future Enhancements

Future versions may support:

- Skip connections
- Recurrent connections
- Custom activation functions
- LSTM and GRU cell types
