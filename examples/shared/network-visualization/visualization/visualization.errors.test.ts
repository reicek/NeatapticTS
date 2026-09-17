import {
  assertFiniteLegendBound,
  NETWORK_VISUALIZATION_NON_FINITE_BOUND_ERROR_MESSAGE,
  VisualizationNonFiniteBoundError,
} from './visualization.errors';

describe('visualization.errors', () => {
  it('assertFiniteLegendBound does not throw for a finite number', () => {
    expect(() => assertFiniteLegendBound(1.5)).not.toThrow();
  });

  it('assertFiniteLegendBound does not throw for zero', () => {
    expect(() => assertFiniteLegendBound(0)).not.toThrow();
  });

  it('assertFiniteLegendBound throws VisualizationNonFiniteBoundError for NaN', () => {
    expect(() => assertFiniteLegendBound(NaN)).toThrow(
      VisualizationNonFiniteBoundError,
    );
  });

  it('assertFiniteLegendBound reports the expected message for NaN', () => {
    expect(() => assertFiniteLegendBound(NaN)).toThrow(
      NETWORK_VISUALIZATION_NON_FINITE_BOUND_ERROR_MESSAGE,
    );
  });

  it('assertFiniteLegendBound throws VisualizationNonFiniteBoundError for positive Infinity', () => {
    expect(() => assertFiniteLegendBound(Infinity)).toThrow(
      VisualizationNonFiniteBoundError,
    );
  });

  it('assertFiniteLegendBound throws VisualizationNonFiniteBoundError for negative Infinity', () => {
    expect(() => assertFiniteLegendBound(-Infinity)).toThrow(
      VisualizationNonFiniteBoundError,
    );
  });

  it('VisualizationNonFiniteBoundError uses the expected message', () => {
    const error = new VisualizationNonFiniteBoundError();

    expect(error.message).toBe(
      NETWORK_VISUALIZATION_NON_FINITE_BOUND_ERROR_MESSAGE,
    );
  });

  it('VisualizationNonFiniteBoundError is an instance of Error', () => {
    const error = new VisualizationNonFiniteBoundError();

    expect(error).toBeInstanceOf(Error);
  });
});
