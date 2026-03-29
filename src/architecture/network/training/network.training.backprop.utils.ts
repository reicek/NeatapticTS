import type Network from '../../network/network';
import type {
  CostDerivative,
  NetworkNode,
  OutputNodeWithCostDerivative,
  PropagationContext,
  RegularizationArgument,
} from './network.training.utils.types';
import { NetworkTrainingOutputTargetLengthError } from './network.training.errors';

/**
 * Propagate output and hidden errors backward through the network.
 *
 * @param this Bound network instance.
 * @param rate Learning rate.
 * @param momentum Momentum factor.
 * @param update Whether to apply updates immediately.
 * @param target Output target values.
 * @param regularization L2 regularization factor.
 * @param costDerivative Optional output-node derivative override.
 */
export function propagate(
  this: Network,
  rate: number,
  momentum: number,
  update: boolean,
  target: number[],
  regularization = 0,
  costDerivative?: CostDerivative,
): void {
  // Step 1: Validate orchestrator inputs.
  validateTargetLength(this, target);

  // Step 2: Build a shared immutable propagation context.
  const context = createPropagationContext(
    this,
    rate,
    momentum,
    update,
    regularization,
    costDerivative,
  );

  // Step 3: Propagate the output layer with target supervision.
  propagateOutputLayer(context, target);

  // Step 4: Propagate all hidden nodes.
  propagateHiddenLayer(context);
}

/**
 * Clear all node runtime traces and states.
 *
 * @param this Bound network instance.
 */
export function clearState(this: Network): void {
  for (const networkNode of this.nodes) {
    clearNodeState(networkNode);
  }
}

/**
 * Validate that target output count matches the network output width.
 *
 * @param network Network instance receiving backpropagation.
 * @param target Output target vector.
 */
function validateTargetLength(network: Network, target: number[]): void {
  if (!target || target.length !== network.output) {
    throw new NetworkTrainingOutputTargetLengthError(
      'Output target length should match network output length',
    );
  }
}

/**
 * Build the shared propagation context consumed by layer helpers.
 *
 * @param network Network instance receiving backpropagation.
 * @param rate Learning rate.
 * @param momentum Momentum factor.
 * @param update Whether updates are applied immediately.
 * @param regularization Regularization setting used by node propagation.
 * @param costDerivative Optional cost-derivative override for output nodes.
 * @returns Immutable context consumed by propagation helpers.
 */
function createPropagationContext(
  network: Network,
  rate: number,
  momentum: number,
  update: boolean,
  regularization: RegularizationArgument,
  costDerivative?: CostDerivative,
): PropagationContext {
  return {
    network,
    rate,
    momentum,
    update,
    regularization,
    costDerivative,
  };
}

/**
 * Propagate all output nodes with explicit targets.
 *
 * @param context Shared propagation context.
 * @param target Output target vector.
 */
function propagateOutputLayer(
  context: PropagationContext,
  target: number[],
): void {
  let targetIndex = target.length - 1;
  for (
    let outputNodeIndex = getLastNodeIndex(context.network);
    outputNodeIndex >= getOutputLayerStartIndex(context.network);
    outputNodeIndex--
  ) {
    propagateSingleOutputNode(
      context,
      context.network.nodes[outputNodeIndex],
      target[targetIndex],
    );
    targetIndex--;
  }
}

/**
 * Propagate all hidden nodes in reverse topological order.
 *
 * @param context Shared propagation context.
 */
function propagateHiddenLayer(context: PropagationContext): void {
  for (
    let hiddenNodeIndex = getOutputLayerStartIndex(context.network) - 1;
    hiddenNodeIndex >= context.network.input;
    hiddenNodeIndex--
  ) {
    propagateSingleHiddenNode(context, context.network.nodes[hiddenNodeIndex]);
  }
}

/**
 * Propagate a single output node with a target value.
 *
 * @param context Shared propagation context.
 * @param node Output node to propagate.
 * @param targetValue Expected output value for this node.
 */
function propagateSingleOutputNode(
  context: PropagationContext,
  node: NetworkNode,
  targetValue: number,
): void {
  if (context.costDerivative) {
    propagateOutputNodeWithCostDerivative(
      node,
      context,
      targetValue,
      context.costDerivative,
    );
    return;
  }

  node.propagate(
    context.rate,
    context.momentum,
    context.update,
    context.regularization,
    targetValue,
  );
}

/**
 * Propagate a single hidden node without a target value.
 *
 * @param context Shared propagation context.
 * @param node Hidden node to propagate.
 */
function propagateSingleHiddenNode(
  context: PropagationContext,
  node: NetworkNode,
): void {
  node.propagate(
    context.rate,
    context.momentum,
    context.update,
    context.regularization,
  );
}

/**
 * Propagate one output node using a custom cost derivative override.
 *
 * @param node Output node to propagate.
 * @param context Shared propagation context.
 * @param targetValue Expected output value for this node.
 * @param costDerivative Cost derivative callback.
 */
function propagateOutputNodeWithCostDerivative(
  node: NetworkNode,
  context: PropagationContext,
  targetValue: number,
  costDerivative: CostDerivative,
): void {
  (node as OutputNodeWithCostDerivative).propagate(
    context.rate,
    context.momentum,
    context.update,
    context.regularization,
    targetValue,
    costDerivative,
  );
}

/**
 * Resolve the first index of the output layer.
 *
 * @param network Network instance.
 * @returns Index at which output nodes begin.
 */
function getOutputLayerStartIndex(network: Network): number {
  return network.nodes.length - network.output;
}

/**
 * Resolve the last node index in the network.
 *
 * @param network Network instance.
 * @returns Last valid node index.
 */
function getLastNodeIndex(network: Network): number {
  return network.nodes.length - 1;
}

/**
 * Clear runtime state for a single node.
 *
 * @param node Node to clear.
 */
function clearNodeState(node: NetworkNode): void {
  node.clear();
}
