jest.mock('../../../objectives/objectives', () => ({
  clearObjectives: jest.fn(),
  registerObjective: jest.fn(),
}));

import {
  clearObjectives,
  registerObjective,
} from '../../../objectives/objectives';
import {
  clearTelemetryObjectives,
  registerTelemetryObjective,
  type TelemetryFacadeObjectivesHost,
} from './telemetry.facade.objectives';

const mockedClearObjectives = jest.mocked(clearObjectives);
const mockedRegisterObjective = jest.mocked(registerObjective);

describe('neat telemetry facade objectives direct helpers', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('registerTelemetryObjective', () => {
    describe('given the facade registers one custom telemetry objective', () => {
      it('delegates the registration to the objective subsystem with the host as context', () => {
        // Arrange
        const host = {
          _getObjectives: () => [],
          options: {},
        } as TelemetryFacadeObjectivesHost;
        const accessor = () => 7;

        // Act
        registerTelemetryObjective(host, 'entropy', 'max', accessor);

        // Assert
        expect({
          args: mockedRegisterObjective.mock.calls[0],
          context: mockedRegisterObjective.mock.contexts[0],
        }).toEqual({
          args: ['entropy', 'max', accessor],
          context: host,
        });
      });
    });
  });

  describe('clearTelemetryObjectives', () => {
    describe('given the facade clears the custom telemetry objective registry', () => {
      it('delegates the clear operation to the objective subsystem with the host as context', () => {
        // Arrange
        const host = {
          _getObjectives: () => [],
          options: {},
        } as TelemetryFacadeObjectivesHost;

        // Act
        clearTelemetryObjectives(host);

        // Assert
        expect(mockedClearObjectives.mock.contexts[0]).toBe(host);
      });
    });
  });
});
