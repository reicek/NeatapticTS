import Network from '../network';
import {
  applyModelMetadata,
  buildOnnxModel,
  collectRecurrentLayerIndices,
  createBaseModel,
  createGraphDimensions,
  deriveHiddenLayerSizes,
  emitFusedRecurrentHeuristics,
  emitLayerGraph,
  finalizeExportMetadata,
  inferLayerOrdering,
  rebuildConnectionsLocal,
  runOnnxExportFlow,
  runOnnxImportFlow,
  assignActivationFunctions,
  assignWeightsAndBiases,
  validateLayerHomogeneityAndConnectivity,
} from './network.onnx.utils';

describe('network onnx utils barrel', () => {
  describe('when the compatibility barrel is imported directly', () => {
    it('exposes the runtime helper getters', () => {
      const runtimeHelpers = [
        runOnnxExportFlow,
        runOnnxImportFlow,
        inferLayerOrdering,
        rebuildConnectionsLocal,
        validateLayerHomogeneityAndConnectivity,
        assignActivationFunctions,
        assignWeightsAndBiases,
        deriveHiddenLayerSizes,
        applyModelMetadata,
        collectRecurrentLayerIndices,
        createBaseModel,
        createGraphDimensions,
        emitLayerGraph,
        emitFusedRecurrentHeuristics,
        finalizeExportMetadata,
      ];

      expect(runtimeHelpers.every((runtimeHelper) => typeof runtimeHelper === 'function')).toBe(
        true,
      );
    });
  });

  describe('when buildOnnxModel() is called through the compatibility barrel', () => {
    it('forwards to the implementation and returns a populated model', () => {
      const sourceNetwork = Network.createMLP(2, [2], 1);
      const orderedLayers = inferLayerOrdering(sourceNetwork);
      const onnxModel = buildOnnxModel(sourceNetwork, orderedLayers, {
        includeMetadata: true,
      });

      expect(onnxModel.graph.node.length).toBeGreaterThan(0);
    });
  });

  describe('when the helper defaults are omitted', () => {
    it('builds and exports models through the default option path', () => {
      const buildSourceNetwork = Network.createMLP(2, [2], 1);
      const exportSourceNetwork = Network.createMLP(2, [2], 1);
      const orderedLayers = inferLayerOrdering(buildSourceNetwork);
      const defaultBuildModel = buildOnnxModel(buildSourceNetwork, orderedLayers);
      const defaultExportModel = runOnnxExportFlow(exportSourceNetwork);

      expect(
        defaultBuildModel.graph.node.length > 0 &&
          defaultExportModel.graph.node.length > 0,
      ).toBe(true);
    });
  });
});