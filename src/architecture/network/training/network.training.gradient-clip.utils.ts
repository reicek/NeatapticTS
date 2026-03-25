import type Network from '../../network/network';
import type {
  TrainingNetworkInternals as NetworkInternals,
  TrainingNodeInternals as NodeInternals,
} from '../network.types';
import type { GradientClipRuntimeConfig } from './network.training.utils.types';

/**
 * Apply gradient clipping to accumulated connection and bias deltas.
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

  const collectGroups = (): number[][] => {
    const collected: number[][] = [];
    if (cfg.mode.startsWith('layerwise')) {
      if (internalNet.layers && internalNet.layers.length > 0) {
        for (
          let layerIndex = 0;
          layerIndex < internalNet.layers.length;
          layerIndex++
        ) {
          const layer = internalNet.layers[layerIndex];
          if (!layer || !layer.nodes) continue;
          const groupVals: number[] = [];
          layer.nodes.forEach((node) => {
            if (!node || node.type === 'input') return;
            node.connections.in.forEach((connection) => {
              if (typeof connection.totalDeltaWeight === 'number') {
                groupVals.push(connection.totalDeltaWeight);
              }
            });
            node.connections.self.forEach((connection) => {
              if (typeof connection.totalDeltaWeight === 'number') {
                groupVals.push(connection.totalDeltaWeight);
              }
            });
            if (typeof node.totalDeltaBias === 'number') {
              groupVals.push(node.totalDeltaBias);
            }
          });
          if (groupVals.length) collected.push(groupVals);
        }
      } else {
        net.nodes.forEach((node) => {
          if (node.type === 'input') return;
          const groupVals: number[] = [];
          const nodeInternal = node as unknown as NodeInternals;
          nodeInternal.connections.in.forEach((connection) => {
            if (typeof connection.totalDeltaWeight === 'number') {
              groupVals.push(connection.totalDeltaWeight);
            }
          });
          nodeInternal.connections.self.forEach((connection) => {
            if (typeof connection.totalDeltaWeight === 'number') {
              groupVals.push(connection.totalDeltaWeight);
            }
          });
          if (typeof nodeInternal.totalDeltaBias === 'number') {
            groupVals.push(nodeInternal.totalDeltaBias);
          }
          if (groupVals.length) collected.push(groupVals);
        });
      }
    } else {
      const globalVals: number[] = [];
      net.nodes.forEach((node) => {
        const nodeInternal = node as unknown as NodeInternals;
        nodeInternal.connections.in.forEach((connection) => {
          if (typeof connection.totalDeltaWeight === 'number') {
            globalVals.push(connection.totalDeltaWeight);
          }
        });
        nodeInternal.connections.self.forEach((connection) => {
          if (typeof connection.totalDeltaWeight === 'number') {
            globalVals.push(connection.totalDeltaWeight);
          }
        });
        if (typeof nodeInternal.totalDeltaBias === 'number') {
          globalVals.push(nodeInternal.totalDeltaBias);
        }
      });
      if (globalVals.length) collected.push(globalVals);
    }
    return collected;
  };

  const groups = collectGroups();
  internalNet._lastGradClipGroupCount = groups.length;

  const computeAbsolutePercentileThreshold = (
    values: number[],
    percentile: number,
  ): number => {
    if (!values.length) return 0;
    const sortedByAbs = [...values].sort((a, b) => Math.abs(a) - Math.abs(b));
    const rank = Math.min(
      sortedByAbs.length - 1,
      Math.max(0, Math.floor((percentile / 100) * sortedByAbs.length - 1)),
    );
    return Math.abs(sortedByAbs[rank]);
  };

  const applyScale = (
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

  if (cfg.mode === 'norm' || cfg.mode === 'layerwiseNorm') {
    const maxAllowedNorm = cfg.maxNorm || 1;
    groups.forEach((groupValues) => {
      const groupL2Norm = Math.sqrt(
        groupValues.reduce((sum, value) => sum + value * value, 0),
      );
      if (groupL2Norm > maxAllowedNorm && groupL2Norm > 0) {
        const normScaleFactor = maxAllowedNorm / groupL2Norm;
        applyScale((currentValue, owningGroup) =>
          owningGroup === groupValues
            ? currentValue * normScaleFactor
            : currentValue,
        );
      }
    });
  } else if (cfg.mode === 'percentile' || cfg.mode === 'layerwisePercentile') {
    const percentileSetting = cfg.percentile || 99;
    groups.forEach((groupValues) => {
      const percentileThreshold = computeAbsolutePercentileThreshold(
        groupValues,
        percentileSetting,
      );
      if (percentileThreshold <= 0) return;
      applyScale((currentValue, owningGroup) =>
        owningGroup === groupValues &&
        Math.abs(currentValue) > percentileThreshold
          ? percentileThreshold * Math.sign(currentValue)
          : currentValue,
      );
    });
  }
};
