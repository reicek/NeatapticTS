import architectureOnnxBundle from './onnx';
import networkOnnxBundle from './network/onnx/network.onnx';

describe('architecture onnx compatibility facade', () => {
  describe('when the root default export is imported', () => {
    it('re-exports the same ONNX bundle as the implementation chapter', () => {
      expect(architectureOnnxBundle).toBe(networkOnnxBundle);
    });
  });
});
