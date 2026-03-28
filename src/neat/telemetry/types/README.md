# neat/telemetry/types

Local telemetry contracts shared by the internal telemetry chapters.

The public `TelemetryEntry` model lives in `src/neat/shared/neat.shared.types.ts`,
but the telemetry subtree also needs a few smaller controller-local helper
shapes for sampling, buffer management, and operator snapshots. Grouping
those contracts here keeps the telemetry split readable without promoting
every internal convenience type into the broader NEAT controller surface.

This chapter exists to keep the telemetry internals legible. The recorder,
runtime, exports, and facade helpers all need a handful of small host and
option shapes, but those shapes are teaching tools for the telemetry
subsystem rather than public promises for the whole library.
Keeping them here makes the internal contracts explicit: what a helper needs
from a genome, what a runtime buffer host must expose, and which option bags
are narrow enough to stay telemetry-local.

## neat/telemetry/types/telemetry.types.ts

### OperatorStatsMap

Operator stats map shape for telemetry extraction.

Adaptive mutation telemetry records success and attempt counts per operator.
This map type names that contract so extraction helpers can stay specific
about what they read without depending on a broader controller class.

### TelemetryBufferContext

Minimal telemetry buffer context shape.

Runtime helpers only need a host that can expose or receive the in-memory
telemetry buffer. This tiny shape keeps those helpers reusable and easy to
test.

### TelemetryCoreFields

Core telemetry field keys used by selection helpers.

Some telemetry fields remain mandatory even when callers request a narrower
export or inspection surface. This alias documents that those core keys are
treated as a fixed protected set rather than arbitrary strings.

### TelemetryDiversityOptions

Diversity telemetry options for sampling and novelty defaults.

This local option bag narrows the much broader controller configuration down
to the fields that matter for diversity-oriented telemetry decisions: whether
diversity metrics are enabled, how aggressively to sample in fast mode, and
which novelty-search defaults should influence telemetry output.

### TelemetryEntryRecord

Telemetry entry shape used for constructing snapshots.

Builders often need the strongly typed `TelemetryEntry` model plus temporary
access to additional dynamic fields while a snapshot is being assembled or
filtered. This alias keeps that construction-phase flexibility localized.

### TelemetryGenome

Minimal genome shape used by telemetry helpers.

Telemetry code rarely needs the entire `Network` surface. Most helpers only
care about structural size and a few optional lineage hints, so this type
captures the smallest genome contract needed for diversity, complexity, and
lineage-oriented telemetry work.

### TelemetrySelectContext

Minimal telemetry selection context shape.

Selection helpers narrow telemetry entries to a configured subset of fields.
This context type names the single host capability they need: an optional set
of selected telemetry keys.

### TelemetryStreamOptions

Minimal telemetry stream options for streaming helpers.

The runtime layer only needs to know whether streaming is enabled and which
callback should receive each entry. This keeps the callback contract explicit
without forcing runtime helpers to depend on the entire controller option bag.
