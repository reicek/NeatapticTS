import {
  clearTelemetryBuffer,
  getTelemetryBuffer,
} from '../../accessors/telemetry.accessors';
import {
  exportTelemetryCSV as exportTelemetryCsvImpl,
  exportTelemetryJSONL as exportTelemetryJsonlImpl,
} from '../../exports/telemetry.exports';
import {
  clearTelemetry,
  exportTelemetryCSV,
  exportTelemetryJSONL,
  getTelemetry,
  type TelemetryFacadeBufferHost,
} from './telemetry.facade.buffer';

jest.mock('../../accessors/telemetry.accessors', () => ({
  clearTelemetryBuffer: jest.fn(),
  getTelemetryBuffer: jest.fn(),
}));

jest.mock('../../exports/telemetry.exports', () => ({
  exportTelemetryCSV: jest.fn(),
  exportTelemetryJSONL: jest.fn(),
}));

const mockedClearTelemetryBuffer = jest.mocked(clearTelemetryBuffer);
const mockedGetTelemetryBuffer = jest.mocked(getTelemetryBuffer);
const mockedExportTelemetryCsvImpl = jest.mocked(exportTelemetryCsvImpl);
const mockedExportTelemetryJsonlImpl = jest.mocked(exportTelemetryJsonlImpl);

describe('neat telemetry facade buffer coverage chapter', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('getTelemetry()', () => {
    describe('when the facade reads the raw telemetry buffer directly', () => {
      it('returns the accessor result unchanged', () => {
        // Arrange
        const telemetryHost: TelemetryFacadeBufferHost = {
          _telemetry: [],
        };
        const telemetryEntries = [{ gen: 4 }];

        mockedGetTelemetryBuffer.mockReturnValue(
          telemetryEntries as ReturnType<typeof getTelemetry>,
        );

        // Act
        const returnedTelemetry = getTelemetry(telemetryHost);

        // Assert
        expect(returnedTelemetry).toBe(telemetryEntries);
      });
    });
  });

  describe('exportTelemetryJSONL()', () => {
    describe('when the facade exports telemetry as JSON lines', () => {
      it('delegates to the export helper with the host as call context', () => {
        // Arrange
        const telemetryHost: TelemetryFacadeBufferHost = {
          _telemetry: [{ gen: 2 } as never],
        };

        mockedExportTelemetryJsonlImpl.mockReturnValue('{"gen":2}');

        // Act
        const jsonlPayload = exportTelemetryJSONL(telemetryHost);

        // Assert
        expect({
          jsonlPayload,
          helperContext: mockedExportTelemetryJsonlImpl.mock.contexts[0],
        }).toEqual({
          jsonlPayload: '{"gen":2}',
          helperContext: telemetryHost,
        });
      });
    });
  });

  describe('exportTelemetryCSV()', () => {
    describe('when the caller omits the entry limit', () => {
      it('uses the default entry limit and delegates with the host as call context', () => {
        // Arrange
        const telemetryHost: TelemetryFacadeBufferHost = {
          _telemetry: [{ gen: 3 } as never],
        };

        mockedExportTelemetryCsvImpl.mockReturnValue('gen\n3');

        // Act
        const csvPayload = exportTelemetryCSV(telemetryHost);

        // Assert
        expect({
          csvPayload,
          helperContext: mockedExportTelemetryCsvImpl.mock.contexts[0],
          maxEntries: mockedExportTelemetryCsvImpl.mock.calls[0]?.[0],
        }).toEqual({
          csvPayload: 'gen\n3',
          helperContext: telemetryHost,
          maxEntries: 500,
        });
      });
    });
  });

  describe('clearTelemetry()', () => {
    describe('when the facade clears the telemetry observation window', () => {
      it('delegates to the telemetry-buffer clear helper with the host object', () => {
        // Arrange
        const telemetryHost: TelemetryFacadeBufferHost = {
          _telemetry: [{ gen: 5 } as never],
        };

        // Act
        clearTelemetry(telemetryHost);

        // Assert
        expect(mockedClearTelemetryBuffer).toHaveBeenCalledWith(telemetryHost);
      });
    });
  });
});
