import type Network from '../../network/network';
import type {
  TrainingNetworkInternals as NetworkInternals,
  TrainingNodeInternals as NodeInternals,
} from '../network.types';
import type { GradientClipRuntimeConfig } from './network.training.utils.types';

const collectGradientClipGroups = (
  net: Network,
  internalNet: NetworkInternals,
  mode: GradientClipRuntimeConfig['mode'],
): number[][] => {
  const collected: number[][] = [];

  if (mode.startsWith('layerwise')) {
    if (internalNet.layers && internalNet.layers.length > 0) {
      for (
        let layerIndex = 0;
        layerIndex < internalNet.layers.length;
        layerIndex++
      ) {
        const layer = internalNet.layers[layerIndex];
        if (!layer || !layer.nodes) continue;
        const groupValues: number[] = [];
        layer.nodes.forEach((node) => {
          if (!node || node.type === 'input') return;
          node.connections.in.forEach((connection) => {
            if (typeof connection.totalDeltaWeight === 'number') {
              groupValues.push(connection.totalDeltaWeight);
            }
          });
          node.connections.self.forEach((connection) => {
            if (typeof connection.totalDeltaWeight === 'number') {
              groupValues.push(connection.totalDeltaWeight);
            }
          });
          if (typeof node.totalDeltaBias === 'number') {
            groupValues.push(node.totalDeltaBias);
          }
        });
        if (groupValues.length) collected.push(groupValues);
      }
      return collected;
    }

    net.nodes.forEach((node) => {
      if (node.type === 'input') return;
      const groupValues: number[] = [];
      const nodeInternal = node as unknown as NodeInternals;
      nodeInternal.connections.in.forEach((connection) => {
        if (typeof connection.totalDeltaWeight === 'number') {
          groupValues.push(connection.totalDeltaWeight);
        }
      });
      nodeInternal.connections.self.forEach((connection) => {
        if (typeof connection.totalDeltaWeight === 'number') {
          groupValues.push(connection.totalDeltaWeight);
        }
      });
      if (typeof nodeInternal.totalDeltaBias === 'number') {
        groupValues.push(nodeInternal.totalDeltaBias);
      }
      if (groupValues.length) collected.push(groupValues);
    });

    return collected;
  }

  const globalValues: number[] = [];
  net.nodes.forEach((node) => {
    const nodeInternal = node as unknown as NodeInternals;
    nodeInternal.connections.in.forEach((connection) => {
      if (typeof connection.totalDeltaWeight === 'number') {
        globalValues.push(connection.totalDeltaWeight);
      }
    });
    nodeInternal.connections.self.forEach((connection) => {
      if (typeof connection.totalDeltaWeight === 'number') {
        globalValues.push(connection.totalDeltaWeight);
      }
    });
    if (typeof nodeInternal.totalDeltaBias === 'number') {
      globalValues.push(nodeInternal.totalDeltaBias);
    }
  });
  if (globalValues.length) collected.push(globalValues);

  return collected;
};

const computeAbsolutePercentileThreshold = (
  values: number[],
  percentile: number,
): number => {
  if (!values.length) return 0;

  const sortedByAbsoluteValue = values.toSorted(
    (leftValue, rightValue) => Math.abs(leftValue) - Math.abs(rightValue),
  );
  const rank = Math.min(
    sortedByAbsoluteValue.length - 1,
    Math.max(
      0,
      Math.floor((percentile / 100) * sortedByAbsoluteValue.length - 1),
    ),
  );

  return Math.abs(sortedByAbsoluteValue[rank]);
};

const applyGradientScale = (
  net: Network,
  cfg: GradientClipRuntimeConfig,
  groups: number[][],
  scaleFn: (currentValue: number, owningGroup: number[]) => number,
): void => {
  let groupIndex = 0;

  net.nodes.forEach((node) => {
    if (cfg.mode.startsWith('layerwise') && node.type === 'input') return;
    const activeGroup = cfg.mode.startsWith('layerwise')
      ? groups[groupIndex++]
      : groups[0];
    const nodeInternal = node as unknown as NodeInternals;
    nodeInternal.connections.in.forEach((connection) => {
      if (typeof connection.totalDeltaWeight === 'number') {
        connection.totalDeltaWeight = scaleFn(
          connection.totalDeltaWeight,
          activeGroup,
        );
      }
    });
    nodeInternal.connections.self.forEach((connection) => {
      if (typeof connection.totalDeltaWeight === 'number') {
        connection.totalDeltaWeight = scaleFn(
          connection.totalDeltaWeight,
          activeGroup,
        );
      }
    });
    if (typeof nodeInternal.totalDeltaBias === 'number') {
      nodeInternal.totalDeltaBias = scaleFn(
        nodeInternal.totalDeltaBias,
        activeGroup,
      );
    }
  });
};

const applyNormClipping = (
  net: Network,
  cfg: GradientClipRuntimeConfig,
  groups: number[][],
  maxAllowedNorm: number,
): void => {
  groups.forEach((groupValues) => {
    const groupL2Norm = Math.sqrt(
      groupValues.reduce((sum, value) => sum + value * value, 0),
    );
    if (groupL2Norm > maxAllowedNorm && groupL2Norm > 0) {
      const normScaleFactor = maxAllowedNorm / groupL2Norm;
      applyGradientScale(net, cfg, groups, (currentValue, owningGroup) =>
        owningGroup === groupValues
          ? currentValue * normScaleFactor
          : currentValue,
      );
    }
  });
};

const applyPercentileClipping = (
  net: Network,
  cfg: GradientClipRuntimeConfig,
  groups: number[][],
  percentileSetting: number,
): void => {
  groups.forEach((groupValues) => {
    const percentileThreshold = computeAbsolutePercentileThreshold(
      groupValues,
      percentileSetting,
    );
    if (percentileThreshold <= 0) return;
    applyGradientScale(net, cfg, groups, (currentValue, owningGroup) =>
      owningGroup === groupValues &&
      Math.abs(currentValue) > percentileThreshold
        ? percentileThreshold * Math.sign(currentValue)
        : currentValue,
    );
  });
};

/**
 * Apply gradient clipping to accumulated connection and bias delta buffers.
 *
 * @param net - Network instance whose accumulated gradients are clipped.
 * @param cfg - Runtime clipping configuration.
 * @returns Nothing.
 */
export const applyGradientClippingCore = (
  net: Network,
  cfg: GradientClipRuntimeConfig,
): void => {
  const internalNet = net as unknown as NetworkInternals;
  const groups = collectGradientClipGroups(net, internalNet, cfg.mode);
  internalNet._lastGradClipGroupCount = groups.length;

  if (cfg.mode === 'norm' || cfg.mode === 'layerwiseNorm') {
    applyNormClipping(net, cfg, groups, cfg.maxNorm || 1);
  } else if (cfg.mode === 'percentile' || cfg.mode === 'layerwisePercentile') {
    applyPercentileClipping(net, cfg, groups, cfg.percentile || 99);
  }
};
