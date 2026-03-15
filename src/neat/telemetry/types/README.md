# neat/telemetry/types

Local telemetry contracts shared by the internal telemetry chapters.

The public `TelemetryEntry` model lives in `src/neat/shared/neat.shared.types.ts`,
but the telemetry subtree also needs a few smaller controller-local helper
shapes for sampling, buffer management, and operator snapshots. Grouping
those contracts here keeps the telemetry split readable without promoting
every internal convenience type into the broader NEAT controller surface.

## neat/telemetry/types/telemetry.types.ts

### OperatorStatsMap

Operator stats map shape for telemetry extraction.

### TelemetryBufferContext

Minimal telemetry buffer context shape.

### TelemetryCoreFields

Core telemetry field keys used by selection helpers.

### TelemetryDiversityOptions

Diversity telemetry options for sampling and novelty defaults.

### TelemetryEntryRecord

Telemetry entry shape used for constructing snapshots.

### TelemetryGenome

Minimal genome shape used by telemetry helpers.

### TelemetrySelectContext

Minimal telemetry selection context shape.

### TelemetryStreamOptions

Minimal telemetry stream options for streaming helpers.
