/**
 * Resolve runtime-loaded Network and Layer factories in CJS/ESM-compatible form.
 *
 * @returns Perceptron factory and layer module object.
 */
export function loadRuntimeFactories(): {
  perceptronFactory: (...sizes: number[]) => unknown;
  layerModule: unknown;
} {
  try {
    const networkModule = require('../network') as {
      default?: {
        architecture?: { Perceptron?: (...sizes: number[]) => unknown };
      };
      architecture?: { Perceptron?: (...sizes: number[]) => unknown };
    };
    const layerModule = require('../layer') as unknown;
    const defaultPerceptronFactory =
      networkModule.default?.architecture?.Perceptron;
    const directPerceptronFactory = networkModule.architecture?.Perceptron;
    const perceptronFactory =
      defaultPerceptronFactory || directPerceptronFactory;
    if (!perceptronFactory) {
      throw new Error('Network Perceptron factory not found in runtime module');
    }
    return { perceptronFactory, layerModule };
  } catch {
    throw new Error(
      'CommonJS runtime required for dynamic Network/Layer loading',
    );
  }
}
