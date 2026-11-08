# voxel

`.bin` LOD binary codec for the Neat Racing voxel racer (spec section
7.3.2, normative; slice P1-S3-C, AC-010).

Format: a magic + version + dims + palette-with-glow header followed by a
greedy-meshed quad array (position, normal axis, width/height, palette
index). The layout is little-endian and deterministic:

| section    | encoding                                                                        |
| ---------- | ------------------------------------------------------------------------------- |
| magic      | `NXVB` as 4 ASCII bytes                                                          |
| version    | u32 LE, currently 1                                                              |
| dims       | u16 x 3 (`width`, `height`, `depth`)                                             |
| palette    | u16 entry count, then 5 bytes per entry (`r`, `g`, `b`, `a`, `glow`)             |
| quad count | u32 LE                                                                           |
| quads      | 12 bytes each: position u16 x 3, normalAxis u8, width u16, height u16, paletteIndex u8 |

Round-trip guarantee (AC-010): writer output re-parsed by the loader and
re-written is byte-identical, so the build script
(`scripts/build-voxel-bins.mjs`) and the runtime loader share one format
without drift. The loader rejects bad magic, unknown versions, truncated
sections, and incoherent fields with named errors that mirror the
`model.schema.ts` class pattern (a stable `code` per subclass).

`VoxelPaletteEntry` is imported as a type-only symbol from
`./model.schema`: the import is erased at runtime under native type
stripping, so no extensionless specifier survives and this module stays
self-contained for the Node build script.

## voxel/bin.codec.ts

### allocateBinBuffer

```ts
allocateBinBuffer(
  model: VoxelBinModel,
): Uint8Array<ArrayBufferLike>
```

Allocates the exact deterministic output size for a model:
header + palette + quad count + quad records.

Parameters:
- `model` - The validated model to size.

Returns: The zero-initialized output buffer.

### assertBinHeaderPresent

```ts
assertBinHeaderPresent(
  bytes: Uint8Array<ArrayBufferLike>,
): void
```

Rejects buffers too short for the fixed 16-byte header.

Parameters:
- `bytes` - The buffer to check.

### assertBinIntegerField

```ts
assertBinIntegerField(
  fieldName: string,
  value: number,
  min: number,
  max: number,
): void
```

Shared integer-field gate: throws one named field error covering range
and integer-ness for every binary field.

Parameters:
- `fieldName` - Field description used in the rejection message.
- `value` - The field value to validate.
- `min` - Inclusive lower bound of the field's binary encoding.
- `max` - Inclusive upper bound of the field's binary encoding.

### assertBinMagic

```ts
assertBinMagic(
  bytes: Uint8Array<ArrayBufferLike>,
): void
```

Rejects buffers whose leading bytes are not the `NXVB` magic.

Parameters:
- `bytes` - The buffer to check.

### assertBinPalettePresent

```ts
assertBinPalettePresent(
  bytes: Uint8Array<ArrayBufferLike>,
  paletteCount: number,
): void
```

Rejects buffers that end before the declared palette section completes.

Parameters:
- `bytes` - The buffer to check.
- `paletteCount` - Declared palette entry count.

### assertBinQuadCountPresent

```ts
assertBinQuadCountPresent(
  bytes: Uint8Array<ArrayBufferLike>,
  quadCountOffset: number,
): void
```

Rejects buffers that end before the u32 quad count field completes.

Parameters:
- `bytes` - The buffer to check.
- `quadCountOffset` - Offset of the quad count field.

### assertBinQuadPaletteIndex

```ts
assertBinQuadPaletteIndex(
  paletteIndex: number,
  paletteSize: number,
): void
```

Validates a quad's palette index: an integer byte below the palette size.

Parameters:
- `paletteIndex` - The quad's palette index.
- `paletteSize` - Palette entry count.

### assertBinVersion

```ts
assertBinVersion(
  view: DataView<ArrayBufferLike>,
): void
```

Rejects unknown format versions.

Parameters:
- `view` - DataView over the buffer.

### loadVoxelBin

```ts
loadVoxelBin(
  bytes: Uint8Array<ArrayBufferLike>,
): VoxelBinModel
```

Parses a voxel bin produced by {@link writeVoxelBin} back into its
in-memory content (spec 7.3.2 loader; AC-010).

The loader is the AC-010 gate: it rejects bad magic, unknown versions,
and any buffer that ends before a declared section completes. Re-writing
the parsed content is byte-identical to the input bytes.

Parameters:
- `bytes` - The encoded voxel bin (a `Uint8Array`; subarray views with
a non-zero `byteOffset` are supported).

Returns: The decoded dims, palette-with-glow, and quad records.

Example:

```ts
const model = loadVoxelBin(bytes);
console.log(model.quads.length);
```

### readBinDims

```ts
readBinDims(
  view: DataView<ArrayBufferLike>,
): [number, number, number]
```

Reads the u16 x 3 dims fields as a tuple.

Parameters:
- `view` - DataView over the buffer.

Returns: The decoded `[width, height, depth]` dims tuple.

### readBinPalette

```ts
readBinPalette(
  bytes: Uint8Array<ArrayBufferLike>,
  paletteCount: number,
): VoxelPaletteEntry[]
```

Reads the palette-with-glow entries (5 bytes each).

Parameters:
- `bytes` - The buffer holding the palette section.
- `paletteCount` - Declared palette entry count (bounds-checked).

Returns: The decoded palette entries.

### readBinQuad

```ts
readBinQuad(
  view: DataView<ArrayBufferLike>,
  offset: number,
  paletteSize: number,
): VoxelBinQuad
```

Reads one 12-byte quad record and re-validates the fields whose binary
encoding can carry incoherent values (normal axis, palette index).

Parameters:
- `view` - DataView over the buffer.
- `offset` - Offset of the quad record.
- `paletteSize` - Palette entry count for index coherence.

Returns: The decoded quad record.

### readBinQuads

```ts
readBinQuads(
  view: DataView<ArrayBufferLike>,
  bufferLength: number,
  startOffset: number,
  quadCount: number,
  paletteSize: number,
): VoxelBinQuad[]
```

Reads every 12-byte quad record, rejecting any buffer that ends before a
record completes or any record with an incoherent field.

Parameters:
- `view` - DataView over the buffer.
- `bufferLength` - Total buffer length (record bounds gate).
- `startOffset` - Offset of the first quad record.
- `quadCount` - Declared quad record count.
- `paletteSize` - Palette entry count for index coherence.

Returns: The decoded quad records.

### validateBinDims

```ts
validateBinDims(
  dims: readonly [number, number, number],
): void
```

Validates the dims tuple: three positive-integer u16 components.

Parameters:
- `dims` - The dims tuple to validate.

### validateBinModel

```ts
validateBinModel(
  model: VoxelBinModel,
): void
```

Validates a full voxel bin model against the binary layout caps.

Parameters:
- `model` - The model to validate.

### validateBinPalette

```ts
validateBinPalette(
  palette: readonly VoxelPaletteEntry[],
): void
```

Validates the palette: at least the EMPTY sentinel, at most u16 entries,
and every component an integer byte.

Parameters:
- `palette` - The palette to validate.

### validateBinPaletteEntry

```ts
validateBinPaletteEntry(
  entry: VoxelPaletteEntry,
  entryIndex: number,
): void
```

Validates one palette entry's five components as integer bytes.

Parameters:
- `entry` - The palette entry to validate.
- `entryIndex` - Index of the entry (for the rejection message).

### validateBinQuad

```ts
validateBinQuad(
  quad: VoxelBinQuad,
  paletteSize: number,
): void
```

Validates one quad record against the binary layout and palette size.

Parameters:
- `quad` - The quad to validate.
- `paletteSize` - Palette entry count; the quad's palette index must
reference an existing entry.

### validateBinQuadPosition

```ts
validateBinQuadPosition(
  position: readonly [number, number, number],
): void
```

Validates a quad position: three integer u16 components (non-negative).

Parameters:
- `position` - The quad origin to validate.

### VOXEL_BIN_DIMS_COMPONENT_MIN

Inclusive lower bound for every dims component (positive integers).

### VOXEL_BIN_DIMS_OFFSET

Byte offset of the u16 x 3 dims fields.

### VOXEL_BIN_HEADER_BYTES

Fixed header size: magic + version + dims + palette count.

### VOXEL_BIN_MAGIC

Magic bytes prefixing every voxel bin (spec 7.3.2).

### VOXEL_BIN_MAGIC_OFFSET

Byte offset of the magic section.

### VOXEL_BIN_NORMAL_AXIS_MAX

Inclusive upper bound for the normalAxis field (z axis).

### VOXEL_BIN_NORMAL_AXIS_MIN

Inclusive lower bound for the normalAxis field (x axis).

### VOXEL_BIN_PALETTE_COUNT_OFFSET

Byte offset of the u16 palette entry count field.

### VOXEL_BIN_PALETTE_ENTRY_BYTES

Bytes per palette entry (`r`, `g`, `b`, `a`, `glow`).

### VOXEL_BIN_PALETTE_ENTRY_MIN

Inclusive lower bound for the palette entry count (the EMPTY sentinel).

### VOXEL_BIN_POSITION_COMPONENTS

Components per quad position (u16 x 3).

### VOXEL_BIN_QUAD_BYTES

Bytes of one 12-byte quad record.

### VOXEL_BIN_QUAD_COUNT_BYTES

Bytes of the u32 quad count field.

### VOXEL_BIN_QUAD_EXTENT_MIN

Inclusive lower bound for quad width/height extents.

### VOXEL_BIN_QUAD_HEIGHT_OFFSET

Byte offset of the u16 height field inside one quad record.

### VOXEL_BIN_QUAD_NORMAL_AXIS_OFFSET

Byte offset of the u8 normalAxis field inside one quad record.

### VOXEL_BIN_QUAD_PALETTE_INDEX_OFFSET

Byte offset of the u8 paletteIndex field inside one quad record.

### VOXEL_BIN_QUAD_WIDTH_OFFSET

Byte offset of the u16 width field inside one quad record.

### VOXEL_BIN_U16_COMPONENT_BYTES

Bytes per u16 component (dims, positions, widths, heights).

### VOXEL_BIN_U16_MAX

Inclusive upper bound for u16 fields (dims, positions, extents, palette count).

### VOXEL_BIN_U8_MAX

Inclusive upper bound for u8 fields (palette components, paletteIndex).

### VOXEL_BIN_VERSION

Binary format version written by this codec (u32 LE).

### VOXEL_BIN_VERSION_OFFSET

Byte offset of the u32 version field.

### VoxelBinCodecError

Base class for every named voxel-bin error (AC-010).

Carries a stable machine-readable `code` so callers (the build script,
the runtime loader) can branch on rejection kinds without parsing
human-readable messages.

#### code

Stable rejection code, for example `voxel-bin/magic`.

### VoxelBinFieldError

Named rejection when a field value cannot be represented by, or is
incoherent with, the binary layout: dims/position/extent components
outside their u16 range, palette components outside the byte range, or a
quad referencing a normal axis or palette entry that does not exist.

### VoxelBinMagicError

Named rejection when the leading bytes are not the `NXVB` magic
(spec 7.3.2: loader rejects bad magic).

### VoxelBinModel

In-memory voxel bin content (spec 7.3.2): dims, palette-with-glow, and
the greedy-meshed quad array.

### VoxelBinQuad

One greedy-meshed quad record (spec 7.3.2 body; AC-009 quad shape).

### VoxelBinTruncatedError

Named rejection when the buffer ends before a declared section completes
(header, palette-with-glow, quad count, or quad records).

### VoxelBinVersionError

Named rejection when the version field is not the supported format
version (spec 7.3.2: loader rejects bad version).

### writeBinHeader

```ts
writeBinHeader(
  bytes: Uint8Array<ArrayBufferLike>,
  view: DataView<ArrayBufferLike>,
  model: VoxelBinModel,
): void
```

Writes the fixed header: `NXVB` magic, u32 version, u16 x 3 dims, and the
u16 palette entry count (all little-endian).

Parameters:
- `bytes` - The output buffer.
- `view` - DataView over the output buffer.
- `model` - The validated model providing dims and palette size.

### writeBinPalette

```ts
writeBinPalette(
  bytes: Uint8Array<ArrayBufferLike>,
  palette: readonly VoxelPaletteEntry[],
  startOffset: number,
): number
```

Writes the palette-with-glow entries (5 bytes each: `r`, `g`, `b`, `a`,
`glow`) starting at the header end.

Parameters:
- `bytes` - The output buffer.
- `palette` - The validated palette entries.
- `startOffset` - Offset of the first palette byte.

Returns: The offset just past the palette section.

### writeBinQuad

```ts
writeBinQuad(
  bytes: Uint8Array<ArrayBufferLike>,
  view: DataView<ArrayBufferLike>,
  startOffset: number,
  quad: VoxelBinQuad,
): number
```

Writes one 12-byte quad record: position u16 x 3, normalAxis u8, width
u16, height u16, paletteIndex u8 (all little-endian).

Parameters:
- `bytes` - The output buffer (u8 fields).
- `view` - DataView over the output buffer (u16 fields).
- `startOffset` - Offset of the quad record.
- `quad` - The validated quad to encode.

Returns: The offset just past the quad record.

### writeBinQuadCount

```ts
writeBinQuadCount(
  view: DataView<ArrayBufferLike>,
  quadCount: number,
  offset: number,
): number
```

Writes the u32 little-endian quad count field.

Parameters:
- `view` - DataView over the output buffer.
- `quadCount` - Number of quad records that follow.
- `offset` - Offset of the quad count field.

Returns: The offset just past the quad count field.

### writeVoxelBin

```ts
writeVoxelBin(
  model: VoxelBinModel,
): Uint8Array<ArrayBufferLike>
```

Encodes a voxel bin into the deterministic little-endian `NXVB` layout
(spec 7.3.2).

Every field is validated against the binary layout caps first, so the
output is guaranteed to be re-loadable by {@link loadVoxelBin} and
byte-identical across repeated calls with equivalent content.

Parameters:
- `model` - The voxel bin content: dims, palette-with-glow, and quads.

Returns: The encoded bytes sized exactly
`16 + 5 x paletteCount + 4 + 12 x quadCount`.

Example:

```ts
const bytes = writeVoxelBin({ dims: [4, 3, 2], palette, quads });
console.log(bytes.length); // 16 + 5 * palette.length + 4 + 12 * quads.length
```

## voxel/materials.ts

Materials semantics for the Neat Racing voxel racer (slice P1-S4-A,
AC-013 part 1; spec sections 7.2 / LD-23 in
`plans/neatracing-voxel-racer.spec.md`).

This module is the single shared owner of the palette-entry material
contract ("glass/glow/black semantics" per the layout comment in the
spec): it classifies every on-disk palette entry into one render class
and applies the LD-23 black exception, so mesher body shading, the LOD
derive, and the dedicated glass pass can share one lookup instead of
re-deriving the rule per consumer.

The spec rules encoded here (section 7.2, normative per LD-26):

1. Palette entries are `[r, g, b, a, glow]` bytes; palette index 0 is
   reserved as the EMPTY sentinel (air; never rendered).
2. `glow` 0-255 drives the bloom/shine pass (section 6.6).
3. A fully transparent entry (`a === 0`) with a color renders as
   shining glass of that color (glow only, no body quads).
4. Black (`r === g === b === 0`) is the exception: no color and no
   shine even if `glow` is set (LD-23 / AC-013). Black stays solid
   occluding body geometry at opaque alphas, and black cells yield
   zero emissive in *every* material lookup.

Inputs are trusted palette entries as validated by `model.schema.ts`
the same way the mesher and LOD builder trust decoded models; this
module adds no I/O and performs one pure pass per entry.

### buildVoxelMaterialTable

```ts
buildVoxelMaterialTable(
  palette: readonly VoxelPaletteEntry[],
): readonly VoxelMaterial[]
```

Builds an index-aligned material table over a model palette (AC-013
part 1): entry `i` resolves to the material semantics of palette
index `i`, so meshing, LOD, and glass consumers perform one
O(palette) resolve pass and then O(1) per-voxel table lookups
instead of re-classifying every voxel.

Parameters:
- `palette` - Palette entries as validated by `model.schema.ts`;
index 0 must hold the EMPTY sentinel per section 7.2.

Returns: One  {@link VoxelMaterial} per palette index, in palette
 *   order; the returned table is safe to share across passes.
 *

Example:

```ts
const table = buildVoxelMaterialTable(document.palette);
const glassy = table.at(glassPaletteIndex); // { kind: 'glass', ... }
```

### classifyMaterialKind

```ts
classifyMaterialKind(
  entry: VoxelPaletteEntry,
): VoxelMaterialKind
```

Classifies one palette entry into its {@link VoxelMaterialKind} by
the section 7.2 / LD-23 priority order: the exact EMPTY sentinel
first, then the black exception, then glass, then opaque solid body.

Parameters:
- `entry` - Palette entry bytes `[r, g, b, a, glow]`.

Returns: The palette entry's render class.

### isBlackColor

```ts
isBlackColor(
  r: number,
  g: number,
  b: number,
): boolean
```

Detects black palette entries by their color bytes (LD-23),
regardless of alpha or glow.

Parameters:
- `r` - Red byte.
- `g` - Green byte.
- `b` - Blue byte.

Returns: True when all three color bytes are 0.

### isEmptyPaletteEntry

```ts
isEmptyPaletteEntry(
  entry: VoxelPaletteEntry,
): boolean
```

Checks whether a palette entry is the exact EMPTY sentinel entry.

Parameters:
- `entry` - The palette entry to check.

Returns: True when every byte matches
 *    {@link VOXEL_EMPTY_PALETTE_ENTRY} .

### resolveVoxelMaterial

```ts
resolveVoxelMaterial(
  entry: VoxelPaletteEntry,
): VoxelMaterial
```

Resolves the material semantics of one palette entry, as shown in the
material table (AC-013 part 1, spec section 7.2 / LD-23). This is the
single canonical classification; every consumer shares it so black
cells contribute no glow through any consumer.

Parameters:
- `entry` - Palette entry as `[r, g, b, a, glow]` with integer
bytes 0-255, as validated by `model.schema.ts`.

Returns: The resolved material: verbatim color/alpha bytes, the
render class, and the emissive glow after the LD-23 black
exception (empty and black entries never emit shine).

Example:

```ts
resolveVoxelMaterial([200, 20, 20, 255, 40]); // { kind: 'solid', ..., emissive: 40 }
resolveVoxelMaterial([190, 225, 255, 0, 160]); // { kind: 'glass', ..., emissive: 160 }
resolveVoxelMaterial([0, 0, 0, 255, 200]); // { kind: 'black', ..., emissive: 0 }
resolveVoxelMaterial([0, 0, 0, 0, 0]); // { kind: 'empty', ..., emissive: 0 }
```

### VOXEL_MATERIAL_BLACK_EMISSIVE

Emissive contribution forced onto black palette entries (LD-23:
black never emits shine), matching the mesher's
`VOXEL_BLACK_EXCEPTION_EMISSIVE` body-pass rule and the LOD
builder's derived-black glow rule (spec section 7.3.3 rule 7).

### VOXEL_MATERIAL_GLASS_ALPHA_MAX

Fully transparent alpha at or below which a non-black palette entry
is glass (spec section 7.2: "a fully transparent entry (`a === 0`)"
renders as shining glass of its color, glow only, no body quads).

### VoxelMaterial

One resolved material-table record for a palette entry: verbatim
color and alpha bytes plus the emissive glow after the LD-23
black-exception rule, plus the render class consumers branch on.

### VoxelMaterialKind

Render class of one palette entry after section 7.2 / LD-23
material semantics:

- `empty`: the exact EMPTY sentinel entry (air; never rendered).
- `black`: `r === g === b === 0`; solid occluding body geometry at
  opaque alphas with emissive forced to 0 regardless of the glow
  byte; never glass and never shining even when transparent.
- `glass`: fully transparent colored entry rendered by the glow-only
  glass pass (no body quads).
- `solid`: opaque (`alpha >= 1`) colored body voxel carrying its
  glow byte for the bloom pass.

## voxel/team-remap.ts

Team-color remap for the Neat Racing voxel racer (slice P1-S4-B, AC-014;
spec sections 6.5 / LD-24 in `plans/neatracing-voxel-racer.spec.md`).

Neatenstein agent parity: every car model carries ONE designated
team-livery palette slot, and at draw time that slot's RGB channels are
swapped to the current team's color while its alpha and glow bytes stay
verbatim (spec section 6.5). The swap is a pure palette-level operation —
the remapped palette is a new array and the input is never mutated — so
cars of every team draw and shine from the same model files and bins
without rebuilding them (AC-014, including the shining-glass pass, which
consumes the remapped palette like any other consumer).

The module owns the four named team colors in a fixed deterministic table
with the LD-24 "few colors overall" neon look, in documented table order
id 0 = cyan, id 1 = red, id 2 = green, id 3 = blue. Applying one team's
remap touches only the designated slot: body colors, glass, black trim,
glowing decals, and the EMPTY sentinel are untouched by every team's remap
(no cross-team bleed), and a remap of the same palette to the same team
id is deterministic and repeatable.

Inputs are trusted palette entries as validated by `model.schema.ts`, the
same way `materials.ts` trusts decoded models; this module adds no input
or output and performs one pure pass per palette.

### assertValidTeamColorId

```ts
assertValidTeamColorId(
  team: VoxelTeamColorId,
): void
```

Reject team color ids outside the four-entry table (or non-integer ids)
with the named `VoxelTeamRemapError`, so a corrupt id fails loudly at
draw time instead of painting an undefined color.

### clonePaletteEntry

```ts
clonePaletteEntry(
  entry: VoxelPaletteEntry,
): VoxelPaletteEntry
```

Copy a palette entry byte-for-byte so the remapped palette owns its
entries and draw-time consumers cannot mutate the interpreted input
through the returned array.

### isEmptyPaletteEntry

```ts
isEmptyPaletteEntry(
  entry: VoxelPaletteEntry,
): boolean
```

Detect the EMPTY sentinel so it can never be remapped: a sentinel slot
carries no geometry, and forging `[teamR, teamG, teamB, 0, 0]` there would
fabricate a renderable shining-glass voxel out of air.

### isTeamRemapSlotIndex

```ts
isTeamRemapSlotIndex(
  index: number,
): boolean
```

Lookup of the designated slot position, shared by the remap pass.

### NEATRACING_TEAM_COLOR_BLUE

Team id 3 livery: neon blue.

### NEATRACING_TEAM_COLOR_COUNT

Number of team colors the remap table carries (the four Neat Racing teams).

### NEATRACING_TEAM_COLOR_CYAN

Team id 0 livery: neon cyan (matches the star-sky cyan `rgba(95, 255, 255, 1)`).

### NEATRACING_TEAM_COLOR_GREEN

Team id 2 livery: neon green.

### NEATRACING_TEAM_COLOR_RED

Team id 1 livery: neon red.

### NEATRACING_TEAM_COLORS

Deterministic team-color table in documented table order: index 0 = cyan,
1 = red, 2 = green, 3 = blue. The order is normative for team color ids,
so consumers resolve liveries by NEATRACING_TEAM_ID_* constants instead of
hard-coding positions.

### NEATRACING_TEAM_ID_BLUE

Team color id for the blue livery (table index 3).

### NEATRACING_TEAM_ID_CYAN

Team color id for the cyan livery (table index 0).

### NEATRACING_TEAM_ID_GREEN

Team color id for the green livery (table index 2).

### NEATRACING_TEAM_ID_RED

Team color id for the red livery (table index 1).

### NEATRACING_TEAM_REMAP_SLOT

Designated team-livery palette slot (spec section 6.5: "a designated
palette slot remaps to team colors at draw time").

Model palettes reserve index 0 for the EMPTY sentinel and paint index 1
with the authoring-time livery placeholder that every team remap swaps;
all other indices carry shared non-team colors. Car models must therefore
author their livery placeholder at this index, while team-less voxel
models (for example wheel bodies) simply reuse the palette entries the
table leaves untouched.

### remapPaletteEntryToTeam

```ts
remapPaletteEntryToTeam(
  entry: VoxelPaletteEntry,
  teamRgb: VoxelTeamColorRgb,
): VoxelPaletteEntry
```

Swap one palette entry's RGB channels to a team color while keeping its
alpha and glow bytes verbatim, so shine and transparency behavior is
preserved for every team livery.

### remapVoxelPaletteToTeam

```ts
remapVoxelPaletteToTeam(
  palette: readonly VoxelPaletteEntry[],
  team: VoxelTeamColorId,
): VoxelPaletteEntry[]
```

Remap a model palette's designated team-livery slot to one team's color.

Returns a fresh palette where the NEATRACING_TEAM_REMAP_SLOT entry keeps
its resolved alpha and glow bytes but paints the team color's RGB bytes,
exactly like the Neatenstein agent palette replacement. Every other entry
— body colors, glass, black trim, decals, the EMPTY sentinel — is copied
through unchanged, so the same model draw works for every team without
rebuilding bins (AC-014). The input palette is never mutated, and repeated
remaps of the same palette to the same team id produce identical output.

Palettes shorter than the designated slot (for example a single-entry
wheel palette) carry no team livery and pass through unchanged, and an
EMPTY sentinel occupying the designated slot is left untouched rather
than being remapped into a renderable color.

Parameters:
- `palette` - Trusted model palette as validated by `model.schema.ts`.
- `team` - Team color id within 0 to NEATRACING_TEAM_COLOR_COUNT - 1.

Returns: The per-team remapped palette (a new array).

Example:

```ts
const teamPalette = remapVoxelPaletteToTeam(decodedPalette, NEATRACING_TEAM_ID_RED);
// teamPalette[NEATRACING_TEAM_REMAP_SLOT] now paints neon red;
// every other entry is byte-identical to decodedPalette.
```

### resolveVoxelTeamColor

```ts
resolveVoxelTeamColor(
  team: VoxelTeamColorId,
): VoxelTeamColorRgb
```

Resolve one team color id to its neon RGB byte triple.

Deterministic table lookup: the same team id always resolves to the same
named entry from NEATRACING_TEAM_COLORS on every call, and out-of-range or
non-integer ids reject with a named `VoxelTeamRemapError` instead of
silently yielding an undefined color.

Parameters:
- `team` - Team color id within 0 to NEATRACING_TEAM_COLOR_COUNT - 1.

Returns: The team color's RGB byte triple from the team color table.

Example:

```ts
const teamRgb = resolveVoxelTeamColor(NEATRACING_TEAM_ID_RED);
// -> [255, 77, 77]
```

### VoxelTeamColor

One named team color in the deterministic remap table.

The `name` is exposed for HUD and telemetry surfaces that label the four
team liveries; the `rgb` bytes are what the draw-time palette remap swaps
into the designated slot.

### VoxelTeamColorId

Team color id: a NEATRACING_TEAM_COLORS table index between 0 and 3.

### VoxelTeamColorRgb

RGB byte triple of one team color (LD-24 neon look).

### VoxelTeamRemapError

Named rejection for team color ids outside the four-entry table (or
non-integer ids), so invalid draw-time ids cannot silently resolve to an
undefined team color.

## voxel/lod.builder.ts

LOD coarse-tier builder for the Neat Racing voxel pipeline (spec §7.3.3;
slice P1-S3-D, AC-011, green-only TDD).

Derives the coarse LOD tier from a decoded voxel model by merging
2x2x2 fine-grid blocks into single coarse cells ({@link buildLodModel}).
The normative car case is the 128-cubed grid (`lod128` per
lod.distance.utils.ts) reduced to the 64-cubed tier (`lod64`); the same
builder serves the 32-cubed wheel grids (→ 16-cubed) and future track
chunks, per spec §7.3.4 ("the LOD builder machinery — derived palettes,
merge rules, §7.3.3 — is scoped to serve those as well").

Tier reconciliation: the tracker's "lod32/lod8" AC shorthand is retired
drift. Spec §7.3.3 (LD-19, locked; G2 — closed edge cases) is normative
and wins: the only derived tier is the 64-cubed grid built by merging
2x2x2 blocks from the 128-cubed grid. A 32-cubed tier is default NOT
built (LD-19 gate: non-goal until a recorded quad-count estimate
justifies it); this module implements no lod32/lod8 tiers.

The builder implements every §7.3.3 rule:

1-2. Contributing cells are palette indices ≠ EMPTY; empties are ignored
     entirely; an all-empty block writes EMPTY (no geometry).
3.   Black cells (r === g === b === 0, LD-23) are solid geometry: they
     pass through verbatim under the single-index rule and are excluded
     ONLY from the color and glow averages, never from alpha.
4.   A block whose contributing cells share one palette index references
     that source index verbatim (color, alpha, and glow unchanged; glow
     passes through on disk — render-time shine suppression stays in the
     mesher's LD-23 shade lookup, greedy-mesher.ts).
5.   Multi-index color bytes average over non-black contributing cells,
     rounding integer half-up, and quantize onto the 5-bit grid.
6.   Glow averages over the same non-black contributors (LD-23: black
     glow never leaks into derived shine).
7.   All averages round half-up, and a derived (0, 0, 0) color forces
     glow to 0 (the black exception at derive time), covering black+black
     mixes and legitimate dark mixes.
8.   The derived alpha snaps to 0 (glass-glow entry: averaged color and
     glow kept, alpha 0) below the threshold, else 255 (opaque).
9.   Identical derived tuples dedup (first = lowest index wins) with
     color channels quantized BEFORE dedup, under a hard 256-entry cap
     whose overflow merges to the nearest entry by squared RGB distance
     (tie-break lowest index; EMPTY index 0 is never a merge target).

Output is a plain {@link DecodedVoxelModel} — derived palette plus new
slice grids — so the existing mesher and bin codec consume it unchanged.
`symmetric` is always false in the output: mirror-X is already
materialized in decoded grids, so the derived grid carries full width.

### buildLodModel

```ts
buildLodModel(
  model: DecodedVoxelModel,
): DecodedVoxelModel
```

Derives the coarse LOD tier from a decoded voxel model (spec §7.3.3).
Deterministic: identical inputs always produce identical outputs, so
build outputs are byte-stable for the bin codec.

The normative car case reduces the 128-cubed grid to the 64-cubed tier;
any even-dim grid is accepted, so the same builder also derives the
16-cubed tier from 32-cubed wheel grids and serves track chunks.

Parameters:
- `model` - Decoded source model (palette indices and slice grids
trusted as already validated by the decoder).

Returns: The derived coarse model as a plain  {@link DecodedVoxelModel} :
 *   dims halved per axis, `symmetric` false (mirror-X already materialized),
 *   a derived palette (source palette copied verbatim at indices 0..n, then
 *   appended or merged per rule 9), and new coarse slice grids so the
 *   existing mesher and bin codec consume it unchanged.

Example:

```ts
import { decodeVoxelModel } from './model.decode.utils';
const lod = buildLodModel(decodeVoxelModel(carDocument)); // 128³ → 64³
const quads = meshVoxelModel(lod).quads; // consumes DecodedVoxelModel as-is
```

### findNearestPaletteIndex

```ts
findNearestPaletteIndex(
  palette: readonly VoxelPaletteEntry[],
  red: number,
  green: number,
  blue: number,
): number
```

Finds the palette entry with the minimum squared RGB distance to a
derived color, the lowest index breaking ties (deterministic). The scan
starts at {@link LOD_MERGE_TARGET_INDEX_MIN} because index 0 is the EMPTY
sentinel and merging solid geometry into air would violate rule 3; the
merge path is only reachable at the {@link VOXEL_LOD_PALETTE_ENTRY_CAP}
cap, so an eligible index always exists.

Parameters:
- `palette` - Current derived palette (at the cap).
- `red` - Derived red byte.
- `green` - Derived green byte.
- `blue` - Derived blue byte.

Returns: Index of the nearest existing palette entry (never 0).

### isBlackEntry

```ts
isBlackEntry(
  entry: VoxelPaletteEntry,
): boolean
```

Detects a black palette entry by its color bytes (LD-23), regardless of
its index, alpha, or glow.

Parameters:
- `entry` - Palette entry as [r, g, b, a, glow].

Returns: True when the entry's color bytes are all zero.

### LodPaletteState

Working palette plus dedup state for one {@link buildLodModel} call.

### quantizeChannel

```ts
quantizeChannel(
  value: number,
): number
```

Quantizes one averaged color byte onto the 5-bit grid (rule 9; G2/P4-F6),
mirroring the spec example `Math.round(v / 255 * 31) / 31 * 255` with the
final byte rounded to its grid point. `Math.round` never hits a .5 tie on
the level step for an integer byte value. Glow and alpha are NOT
quantized (the spec quantizes color channels only).

Parameters:
- `value` - Averaged color byte in 0-255.

Returns: The quantized byte on the 5-bit grid.

### readModelCell

```ts
readModelCell(
  model: DecodedVoxelModel,
  x: number,
  y: number,
  z: number,
): number
```

Reads the palette index at source lattice coordinates (row-major
`x + y * dims[0]` per depth slice), mirroring the decoder's grid layout.

Parameters:
- `model` - Decoded source model.
- `x` - Lattice x in 0..dims[0] - 1.
- `y` - Lattice y in 0..dims[1] - 1.
- `z` - Lattice z in 0..dims[2] - 1.

Returns: Palette index at the cell.

### resolveDerivedPaletteIndex

```ts
resolveDerivedPaletteIndex(
  state: LodPaletteState,
  derived: VoxelPaletteEntry,
): number
```

Resolves a derived entry's palette index (rule 9): an identical tuple
reuses its cached index (first = lowest index wins); a new tuple appends
while the palette is under the cap; at the cap it merges to the nearest
existing entry and the merged index is cached so repeat tuples stay
stable.

Parameters:
- `state` - Palette + dedup state for this build call.
- `derived` - Fully derived `[r, g, b, a, glow]` entry (quantized).

Returns: The resolved palette index for the derived entry.

### resolveLodBlockIndex

```ts
resolveLodBlockIndex(
  model: DecodedVoxelModel,
  state: LodPaletteState,
  coarseX: number,
  coarseY: number,
  coarseZ: number,
): number
```

Resolves one coarse cell's palette index from its 2x2x2 source block in
two allocation-free passes over the eight cells: pass A classifies the
block (all-empty → EMPTY; single index → verbatim), pass B derives the
mixed entry (rules 3, 5-9).

Parameters:
- `model` - Decoded source model.
- `state` - Palette + dedup state for this build call.
- `coarseX` - Coarse cell x (block origin 2x).
- `coarseY` - Coarse cell y (block origin 2y).
- `coarseZ` - Coarse cell z (block origin 2z).

Returns: The resolved palette index for the coarse cell.

### roundHalfUp

```ts
roundHalfUp(
  sum: number,
  count: number,
): number
```

Rounds sum/count to the nearest integer with .5 ties rounding up
(spec §7.3.3: all averages round half-up). Pure integer arithmetic over
non-negative byte-component sums, so there are no float edge cases.

Parameters:
- `sum` - Non-negative integer sum of the averaged component.
- `count` - Positive contributor count (callers guard count 0).

Returns: The half-up rounded integer average.

### VOXEL_LOD_ALPHA_GLASS

Derived alpha for the glass-glow entry (averaged color and glow kept).

### VOXEL_LOD_ALPHA_OPAQUE

Derived alpha for the opaque entry.

### VOXEL_LOD_ALPHA_SNAP_THRESHOLD

Derived-alpha snap point (spec §7.3.3 rule 8): an averaged alpha below
this threshold snaps the entry to glass (alpha 0, color and glow kept);
at or above it the entry snaps to opaque (alpha 255).

### VOXEL_LOD_DERIVED_BLACK_GLOW

Glow forced onto derived entries whose quantized color is (0, 0, 0)
(spec §7.3.3 rule 7, the black exception at derive time, LD-23).

### VOXEL_LOD_MERGE_BLOCK_SIZE

Merge-block edge per axis for the LOD tier (spec §7.3.3 rule 1; LD-19
locked). The coarse tier merges exactly 2x2x2 fine-grid blocks into one
output cell, so output dims are input dims / 2 on every axis.

### VOXEL_LOD_PALETTE_ENTRY_CAP

Hard cap on derived palette entries (spec §7.3.3 rule 9): palette indices
live in the Uint8 index space 0-255 (index 0 is the EMPTY sentinel), so
a derived tuple arriving at the cap merges to the nearest existing entry
instead of appending.

### VOXEL_LOD_QUANTIZATION_LEVELS

Quantization levels per color channel of the derived palette (spec
§7.3.3 rule 9; G2/P4-F6): color channels are quantized onto the 5-bit
grid before dedup so near-identical mixes collapse to one entry.

### VOXEL_LOD_QUANTIZATION_MAX_LEVEL

Highest 5-bit quantization level (= {@link VOXEL_LOD_QUANTIZATION_LEVELS}
- 1; kept derived so the pair stays in sync).

### VoxelLodDimsError

Named rejection when any input dim is not divisible by
{@link VOXEL_LOD_MERGE_BLOCK_SIZE} (spec §7.3.3 rule 1: only even-dim
grids split into whole 2x2x2 blocks).

Example:

```ts
try {
  buildLodModel(decodedModel); // dims [3, 4, 4]
} catch (error) {
  // VoxelLodDimsError with code 'voxel-lod/dims'
}
```

## voxel/model.schema.ts

On-disk voxel model schema for the Neat Racing voxel racer (spec section
7.2, normative per LD-26 / section 2.4; slice P1-S3-A).

A model is a single JSON document (for example
`assets/models/car-f1.model.json`) with three top-level fields:

1. `meta` — `{ name, dims: [width, height, depth], symmetric }`; dims are
   positive integers declaring the grid (normative car grid 128-cubed;
   wheels declare 32-cubed grids, section 7.4/TP-14).
2. `palette` — `[r, g, b, a, glow]` byte entries. Index 0 is reserved as
   EMPTY (air; never rendered, never counted in averages) and must stay
   the all-zero sentinel. Black (`r === g === b === 0`) is a legal solid
   entry even with `glow > 0` (the black-is-solid exception, LD-23) — the
   schema accepts it, and shine suppression happens downstream in the
   mesher and LOD builder, not here.
3. `slices` — one front-view slice per declared depth index (slice index =
   depth along the car's forward axis), each holding one row per declared
   height row by `dims[0]` cols; row 0 is ground/down. Rows are RLE hex
   strings of `<runLen:2hex><paletteIdx:2hex>` pairs; they are expanded by
   `model.decode.utils.ts`, which also mirrors the stored half-width when
   `meta.symmetric` is true.

The module owns the format types, the named constants, and the named
validation errors. Validation is decoder-enforced and test-enforced
(AC-007/AC-008): malformed inputs are rejected with named errors.

### isDimsTuple

```ts
isDimsTuple(
  value: unknown,
): boolean
```

Narrows an untrusted value to a 3-number dims tuple.

Parameters:
- `value` - Untrusted value.

Returns: True when the value is an array of exactly three numbers.

### isEmptyPaletteEntry

```ts
isEmptyPaletteEntry(
  entry: VoxelPaletteEntry,
): boolean
```

Checks whether a palette entry is the EMPTY sentinel exactly.

Parameters:
- `entry` - The palette entry to check.

Returns: True when every component matches
 *    {@link VOXEL_EMPTY_PALETTE_ENTRY} .

### isPaletteEntry

```ts
isPaletteEntry(
  value: unknown,
): boolean
```

Narrows an untrusted value to a palette entry tuple of
{@link VOXEL_PALETTE_ENTRY_LENGTH} integer components in the 0-255 byte
range.

Parameters:
- `value` - Untrusted value.

Returns: True when the value is a well-formed palette entry.

### isRecord

```ts
isRecord(
  value: unknown,
): boolean
```

Narrows an untrusted value to a plain JSON object record.

Parameters:
- `value` - Untrusted value.

Returns: True when the value is a non-null, non-array object.

### validateEmptyPaletteEntry

```ts
validateEmptyPaletteEntry(
  palette: unknown[],
): void
```

Validates that palette index 0 holds the EMPTY sentinel exactly
(section 7.2: index 0 is reserved as EMPTY; index 0 voxels never render
and never count in averages).

Parameters:
- `palette` - The palette array to validate.

### validateMeta

```ts
validateMeta(
  candidate: unknown,
): VoxelModelMeta
```

Validates the `meta` header: object shape, non-empty `name`, boolean
`symmetric`, and the declared dims.

Parameters:
- `candidate` - Untrusted `meta` value.

Returns: The validated metadata.

### validateMetaDims

```ts
validateMetaDims(
  candidate: unknown,
): [number, number, number]
```

Validates `meta.dims` as a 3-number tuple of positive integers declaring
the document's grid (section 7.2: reject `dims` ≠ 128³ *or the declared
dims* — any positive-integer tuple is legal, e.g. the 32-cubed wheel
grids of section 7.4/TP-14; slice/row counts are checked separately
against these declared dims).

Parameters:
- `candidate` - Untrusted `dims` value.

Returns: The validated dims tuple.

### validatePalette

```ts
validatePalette(
  candidate: unknown,
): VoxelPaletteEntry[]
```

Validates the palette array: present, non-empty, entries shaped as
integer byte tuples, and index 0 holding the EMPTY sentinel.

Parameters:
- `candidate` - Untrusted `palette` value.

Returns: The validated palette entries.

### validatePaletteEntries

```ts
validatePaletteEntries(
  palette: unknown[],
): void
```

Validates every palette entry's shape and byte range.

Parameters:
- `palette` - The palette array to validate.

### validateSliceRows

```ts
validateSliceRows(
  candidate: unknown,
  rowCount: number,
): void
```

Validates one slice's rows: an array of exactly `rowCount` strings.

Parameters:
- `candidate` - Untrusted slice value.
- `rowCount` - Row count declared by `meta.dims[1]`.

### validateSlices

```ts
validateSlices(
  candidate: unknown,
  dims: readonly [number, number, number],
): string[][]
```

Validates the slices array: one slice per depth index, each slice holding
one row string per height row.

Parameters:
- `candidate` - Untrusted `slices` value.
- `dims` - The validated dims tuple.

Returns: The validated slice matrix of RLE hex row strings.

### validateSymmetricWidth

```ts
validateSymmetricWidth(
  dims: readonly [number, number, number],
  symmetric: boolean,
): void
```

Rejects odd widths on symmetric documents: mirror-X stores exactly half
the width (LD-28), so the declared width must split into two equal
integer halves.

Parameters:
- `dims` - The validated dims tuple.
- `symmetric` - The validated `meta.symmetric` flag.

### validateVoxelModelDocument

```ts
validateVoxelModelDocument(
  candidate: unknown,
): VoxelModelDocument
```

Validates an untrusted model document against the section 7.2 schema
rules and returns the same content, narrowed to {@link VoxelModelDocument}.

Row-level RLE rules (hex validity, run sums, palette index ranges) are
enforced by the decoder in `model.decode.utils.ts`; this validator owns
the document structure: metadata, dims, palette entries, and slice/row
counts.

Parameters:
- `candidate` - Untrusted parsed JSON value.

Returns: The validated document content as a  {@link VoxelModelDocument} .

Example:

```ts
const document = validateVoxelModelDocument(JSON.parse(rawJson));
console.log(document.meta.name);
```

### VOXEL_DIMS_AXIS_MIN

Inclusive lower bound for every declared grid axis (positive integers).

### VOXEL_EMPTY_PALETTE_ENTRY

The EMPTY sentinel palette entry that palette index 0 must hold exactly
(section 7.2: index 0 is reserved as EMPTY).

### VOXEL_EMPTY_PALETTE_INDEX

Palette index reserved as EMPTY (air; never rendered, section 7.2).

### VOXEL_GRID_DIMS

Normative car-grid dimension along every axis (section 7.2/LD-19). This
constant describes the standard car asset size; validation itself accepts
any declared dims (section 7.2: reject `dims` ≠ 128³ *or the declared
dims*), for example the 32-cubed wheel grids of section 7.4/TP-14.

### VOXEL_PALETTE_COMPONENT_MAX

Inclusive upper bound for every palette component (byte range 0-255).

### VOXEL_PALETTE_ENTRY_LENGTH

Number of components in one palette entry: `[r, g, b, a, glow]`.

### VOXEL_SYMMETRIC_WIDTH_MODULUS

Modulus the declared width must satisfy when `meta.symmetric` is true:
mirror-X stores exactly half the width (LD-28), so the width must split
into two equal halves. Must stay in sync with
`VOXEL_SYMMETRIC_HALF_DIVISOR` in `model.decode.utils.ts`.

### VoxelModelDimsError

Named rejection when `meta.dims` is not a positive-integer tuple, the
width is odd while `meta.symmetric` is true, or the slice/row counts
contradict the declared dims (section 7.2: "dims ≠ 128³ (or the declared
dims)" rejection category; section 7.4/TP-14 accepts 32-cubed wheels).

### VoxelModelDocument

A full on-disk model document (section 7.2).

### VoxelModelError

Base class for every named voxel-model error (AC-007/AC-008).

Carries a stable machine-readable `code` so callers (server endpoints, the
editor rebuild pipeline) can branch on rejection kinds without parsing
human-readable messages.

#### code

Stable rejection code, for example `voxel-model/dims`.

### VoxelModelMeta

Model metadata header (`meta`).

### VoxelModelMetadataError

Named rejection when required model metadata is missing or malformed
(section 7.2: "missing metadata" rejection category).

### VoxelModelPaletteError

Named rejection when palette entries are malformed (wrong shape, byte
range, or integer-ness) or palette index 0 is not the EMPTY sentinel.

### VoxelPaletteEntry

One on-disk palette entry: `[r, g, b, a, glow]`, each component an integer
in the 0-255 byte range.

- `glow` drives the bloom/shine pass (section 6.6).
- A fully transparent entry (`a === 0`) with `glow > 0` renders later as
  shining glass of that color (glow only, no body quads).
- Black (`r === g === b === 0`) carries no color and no shine regardless
  of `glow` (LD-23); the entry itself stays legal on disk.

### VoxelRowHexError

Named rejection when an RLE row holds invalid hex (non-hex digits, short
fields) or a zero-length run (section 7.2: run lengths are 1-255).

### VoxelRowPaletteIndexError

Named rejection when a row references a palette index outside the
document's palette (section 7.2: "out-of-range palette indices").

### VoxelRowSumError

Named rejection when a row's runs do not sum exactly to the declared row
width (`dims[0]` cols for full documents, `dims[0] / 2` stored cols for
symmetric documents).

## voxel/greedy-mesher.ts

Greedy mesher for the Neat Racing voxel pipeline (spec §7.3.1; slice
P1-S3-B, AC-009).

Consumes a decoded voxel model ({@link DecodedVoxelModel}, produced by
`decodeVoxelModel` in `model.decode.utils.ts`) and emits greedy-merged quad
records. The merge key is (palette index, normal axis, face orientation):
faces merge along a plane run only when all three match, so every merged
quad shares one flat palette color and renders identically to unmerged
faces under the per-fragment shading model (palette color, signed normal,
depth — the G3 merge-contract shading invariant).

Interior-face culling: a face is emitted only where exactly one side of the
face plane is a body voxel. Body voxels are palette entries with
alpha >= {@link VOXEL_OPAQUE_ALPHA_MIN}; fully transparent entries (alpha
0, e.g. glow glass) and the EMPTY sentinel are non-occluding air, so body
faces stay visible through glass. Per §7.2 glass is glow only with no body
quads — glass-pass geometry is owned by the AC-013 materials/glass slices,
not by this mesher.

The LD-23 black exception is enforced in the per-voxel shade lookup
({@link resolveVoxelShade}): black (r === g === b === 0) never emits shine
regardless of its glow byte, while remaining solid occluding geometry.

Output quad record is the normative §7.3.1 shape
`{ position, normalAxis, width, height, paletteIndex }`: `position` is the
quad's minimum lattice corner, `normalAxis` the face-plane axis, and
width/height the extents along the two tangent axes in ascending order
(axis 0 → tangents y/z, axis 1 → z/x, axis 2 → x/y). Faces merge only
within one plane, one axis, and one orientation. Emission order is
deterministic (ascending axis, ascending face plane, row-major mask scan
extending width before height) so build outputs are byte-stable for the
bin codec (AC-010).

The mesher is the single shared implementation for car bodies, wheels,
track slabs, and the editor preview (dogfooding, LD-28).

### buildFaceMask

```ts
buildFaceMask(
  model: DecodedVoxelModel,
  context: FacePlaneContext,
): Int16Array<ArrayBufferLike>
```

Builds the signed face mask for one face plane. A cell holds +paletteIndex
when exactly the behind-side voxel is a body voxel (+face), -paletteIndex
when exactly the ahead-side voxel is a body voxel (-face), and 0 when both
sides agree (both solid → interior face culled; both air → no face).

Parameters:
- `model` - Decoded model to probe.
- `context` - Face-plane sweep context (axis, plane, tangent axes).

Returns: Mask of dims[widthAxis] x dims[heightAxis] signed palette indices.

### canExtendMaskBlockRow

```ts
canExtendMaskBlockRow(
  mask: Int16Array<ArrayBufferLike>,
  consumed: Uint8Array<ArrayBufferLike>,
  value: number,
  row: number,
  u: number,
  maskWidth: number,
  width: number,
): boolean
```

Checks whether one candidate row fully extends a rectangle: every cell in
the row segment must carry the rectangle's signed value and be unconsumed.

Parameters:
- `mask` - Signed face mask being merged.
- `consumed` - Cells already claimed by earlier rectangles.
- `origin` - Flat mask index of the rectangle origin.
- `value` - Signed mask value to match.
- `row` - Candidate height tangent row index.
- `u` - Origin column (width tangent coordinate).
- `maskWidth` - Mask extent along the width tangent axis.
- `width` - Rectangle width to validate.

Returns: True when the row segment can join the rectangle.

### consumeMaskBlock

```ts
consumeMaskBlock(
  consumed: Uint8Array<ArrayBufferLike>,
  origin: number,
  width: number,
  height: number,
  maskWidth: number,
): void
```

Marks a merged rectangle's mask cells as consumed.

Parameters:
- `consumed` - Claimed-cell tracker to mutate.
- `origin` - Flat mask index of the rectangle origin.
- `width` - Rectangle width.
- `height` - Rectangle height.
- `maskWidth` - Mask extent along the width tangent axis.

### createQuadRecord

```ts
createQuadRecord(
  context: FacePlaneContext,
  u: number,
  v: number,
  width: number,
  height: number,
  paletteIndex: number,
): VoxelQuad
```

Assembles the normative quad record for one merged rectangle, mapping the
face-plane context and mask cell origin onto lattice coordinates: the
plane coordinate lands on `axis`, the mask origin on the tangent axes.

Parameters:
- `context` - Face-plane sweep context (axis, plane, tangent axes).
- `u` - Rectangle origin along the width tangent axis.
- `v` - Rectangle origin along the height tangent axis.
- `width` - Rectangle width along the first tangent axis.
- `height` - Rectangle height along the second tangent axis.
- `paletteIndex` - Shared palette entry of the merged cells.

Returns: The normative `{ position, normalAxis, width, height, paletteIndex }` record.

### faceMaskValue

```ts
faceMaskValue(
  model: DecodedVoxelModel,
  context: FacePlaneContext,
  u: number,
  v: number,
): number
```

Computes one mask cell's signed value from the two face-plane neighbors.

Parameters:
- `model` - Decoded model to probe.
- `context` - Face-plane sweep context.
- `u` - Mask cell coordinate along the width tangent axis.
- `v` - Mask cell coordinate along the height tangent axis.

Returns: Signed palette index for the emitted face, or  {@link VOXEL_MASK_EMPTY} .

### FacePlaneContext

Face-plane sweep context for one (axis, plane) mask pass.

### isVoxelBodyEntry

```ts
isVoxelBodyEntry(
  entry: VoxelPaletteEntry,
): boolean
```

Classifies a palette entry as body geometry by its alpha byte.

Parameters:
- `entry` - Palette entry as [r, g, b, a, glow].

Returns: True when the entry renders as an opaque body voxel.

### measureMaskBlockHeight

```ts
measureMaskBlockHeight(
  mask: Int16Array<ArrayBufferLike>,
  consumed: Uint8Array<ArrayBufferLike>,
  value: number,
  v: number,
  u: number,
  maskWidth: number,
  maskHeight: number,
  width: number,
): number
```

Measures how many additional full-width rows of equal-value unconsumed
cells can extend the rectangle along the height tangent axis.

Parameters:
- `mask` - Signed face mask being merged.
- `consumed` - Cells already claimed by earlier rectangles.
- `origin` - Flat mask index of the rectangle origin.
- `value` - Signed mask value to match.
- `v` - Origin row (height tangent coordinate).
- `u` - Origin column (width tangent coordinate).
- `maskWidth` - Mask extent along the width tangent axis.
- `maskHeight` - Mask extent along the height tangent axis.
- `width` - Already-measured rectangle width.

Returns: Rectangle height in rows, at least 1.

### measureMaskRunWidth

```ts
measureMaskRunWidth(
  mask: Int16Array<ArrayBufferLike>,
  consumed: Uint8Array<ArrayBufferLike>,
  origin: number,
  value: number,
  v: number,
  maskWidth: number,
): number
```

Measures the maximal run of equal-value unconsumed mask cells along the
width tangent axis starting at the rectangle origin (never crossing the
row end).

Parameters:
- `mask` - Signed face mask being merged.
- `consumed` - Cells already claimed by earlier rectangles.
- `origin` - Flat mask index of the rectangle origin.
- `value` - Signed mask value to match.
- `v` - Origin row (height tangent coordinate).
- `maskWidth` - Mask extent along the width tangent axis.

Returns: Run width in cells, at least 1.

### mergeMaskRectangles

```ts
mergeMaskRectangles(
  mask: Int16Array<ArrayBufferLike>,
  maskWidth: number,
  maskHeight: number,
  context: FacePlaneContext,
): VoxelQuad[]
```

Greedy-merges a face mask into maximal rectangles of equal signed value
(the merge key: palette index + orientation within one fixed axis/plane).
Scans row-major and extends width before height; consumed cells never
join a second rectangle.

Parameters:
- `mask` - Signed face mask from  {@link buildFaceMask} .
- `maskWidth` - Mask extent along the width tangent axis.
- `maskHeight` - Mask extent along the height tangent axis.
- `context` - Face-plane sweep context for record placement.

Returns: Merged quads in row-major origin order.

### meshAxisFaces

```ts
meshAxisFaces(
  model: DecodedVoxelModel,
  axis: VoxelNormalAxis,
): VoxelQuad[]
```

Meshes one face-axis family: sweeps every face plane perpendicular to
`axis`, builds the signed face mask per plane, and greedy-merges maximal
equal-key rectangles.

Parameters:
- `model` - Decoded model to mesh.
- `axis` - Face-plane normal axis for this sweep.

Returns: Quads for this axis in ascending-plane emission order.

### meshVoxelModel

```ts
meshVoxelModel(
  model: DecodedVoxelModel,
): VoxelMesh
```

Greedy-meshes a decoded voxel model into merged face quads with interior
faces culled (AC-009, spec §7.3.1). Body voxels are palette entries with
alpha >= {@link VOXEL_OPAQUE_ALPHA_MIN}; glass and EMPTY cells are
non-occluding air, so body faces stay exposed behind glass.

Parameters:
- `model` - Decoded voxel model from `decodeVoxelModel`; its palette
indices and slice grids are trusted as already validated by the decoder.

Returns: The merged quad mesh with deterministic emission order (ascending
axis, ascending face plane, row-major mask scan) and the model dims
echoed for bin-header encoding.

Example:

```ts
import { decodeVoxelModel } from './model.decode.utils';
const decoded = decodeVoxelModel(document);
const mesh = meshVoxelModel(decoded);
// mesh.quads[0] -> { position, normalAxis, width, height, paletteIndex }
```

### resolveVoxelShade

```ts
resolveVoxelShade(
  entry: VoxelPaletteEntry,
): VoxelShade
```

Resolves the per-voxel render shade for one palette entry (AC-009
per-voxel color/emissive lookup). The LD-23 black exception is enforced
here: black (r === g === b === 0) never shines regardless of its glow
byte, while remaining solid geometry in the body pass.

Parameters:
- `entry` - Palette entry as [r, g, b, a, glow] with bytes 0-255.

Returns: Color bytes verbatim, alpha verbatim, and the emissive glow
contribution (0 for black entries).

Example:

```ts
resolveVoxelShade([0, 0, 0, 255, 200]); // { color: [0,0,0], alpha: 255, emissive: 0 }
resolveVoxelShade([0, 0, 255, 255, 40]); // { color: [0,0,255], alpha: 255, emissive: 40 }
```

### VOXEL_BLACK_EXCEPTION_EMISSIVE

Emissive value forced onto black palette entries (LD-23: black never shines).

### VOXEL_OPAQUE_ALPHA_MIN

Alpha byte at or above which a palette entry renders as an opaque body
voxel; entries below it (including glass at alpha 0) are non-occluding air
for body meshing.

### voxelBodyIndex

```ts
voxelBodyIndex(
  model: DecodedVoxelModel,
  context: FacePlaneContext,
  planeCoord: number,
  u: number,
  v: number,
): number
```

Resolves the palette index of the body voxel at a face-plane neighbor, or
the no-body sentinel. Glass (alpha 0) and EMPTY cells are non-occluding
air, so they never return a body index.

Parameters:
- `model` - Decoded model to probe.
- `context` - Face-plane sweep context.
- `planeCoord` - Lattice coordinate along the normal axis (may lie just
outside the grid, which yields no body).
- `u` - Mask cell coordinate along the width tangent axis.
- `v` - Mask cell coordinate along the height tangent axis.

Returns: Body palette index, or  {@link VOXEL_NO_BODY_CELL} .

### voxelIndexAt

```ts
voxelIndexAt(
  model: DecodedVoxelModel,
  coords: readonly [number, number, number],
): number
```

Reads the palette index at lattice coordinates, treating anything outside
the grid as the EMPTY sentinel (faces at grid boundaries stay exposed).

Parameters:
- `model` - Decoded model to probe.
- `coords` - Lattice coordinates [x, y, z].

Returns: Palette index at the cell, or  {@link VOXEL_EMPTY_PALETTE_INDEX} .

### VoxelMesh

Greedy-meshed voxel mesh: merged quads plus the source model dimensions.

### VoxelNormalAxis

Axis a quad face lies normal to: 0 = x, 1 = y, 2 = z.

### VoxelQuad

Normative §7.3.1 output quad record. `position` is the quad's minimum
lattice corner, `normalAxis` the face-plane axis, `width`/`height` the
extents along the two tangent axes in ascending order, and
`paletteIndex` the shared palette entry of every merged cell.

### VoxelShade

Resolved per-voxel render shade for one palette entry: the color bytes
verbatim, the alpha byte, and the emissive glow after the LD-23 black
exception (black entries carry emissive 0 regardless of their glow byte).

## voxel/wheel.mount.utils.ts

Wheel mount contract for the Neat Racing voxel racer.

Implements the spec §7.4 "G9 — pinned mount contract" (LD-21, AC-015, TP-14):

1. Wheels are separate small rotating voxel models authored in the compact
   voxel JSON format at the wheel's own voxel world scale. The wheel grid is
   the TP-14 normative 32-cubed grid and shares
   {@link NEATRACING_WHEEL_VOXEL_WORLD_SIZE} (== `NEATRACING_VOXEL_WORLD_SIZE`
   from `./lod.distance.utils`, which stays the single owner of the voxel
   scale constant).
2. Hub height: a wheel is mounted at hub height equal to the wheel's world
   radius, so the contact patch sits exactly on the ground plane `y = 0`
   (spec §7.4: "mounted at hub height = the wheel's world radius so the
   contact patch sits at `y = 0`").
3. Pivot: the wheel's rotation pivot is the disc's geometric center
   (the normalized grid center, fractions `[0.5, 0.5, 0.5]`).
4. Authoring frame and mount: the disc plane contains the vertical (`y`)
   axis and the spin axis is the model's depth axis before mounting. The
   mount rotates the model by exactly +90° around the car's up axis, which
   maps the pre-mount depth axis onto the car's lateral (car-right) axis,
   so the spin axis becomes the lateral axle line shared by left and right
   wheels.
5. Spin: the true rotation advances with ground speed as `ω = v / r`. The
displayed* spin is capped at
   {@link NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S} revolutions per second
   (spec §7.4: near 320 km/h a ~0.33 m wheel spins near 270 rad/s and
   strobes; the cap keeps the display alias-free). The cap is applied after
   the true ω is computed, so physics is unaffected.

The car frame convention used by socket offsets is the car model frame:
`+x` = lateral (car-right), `+y` = up, `+z` = forward (car nose), with
offsets measured from the car's model center. Deterministic: every exported
function is a pure function of its inputs with safe no-spin fallbacks for
non-finite numbers, so browser and replay runs agree bit-for-bit.

Row-0-is-down, RLE row decoding and palette validation stay owned by
`./model.schema` (and the decoder); this module only derives the wheel's
world geometry from the declared (validated) document shape.

### clampVisualWheelSpinOmega

```ts
clampVisualWheelSpinOmega(
  trueOmegaRadPerS: number,
): number
```

Clamps the true wheel spin to the visual strobing cap (G9): the displayed
ω never exceeds {@link NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S}
(2 rev/s). Sign is preserved so a reversing wheel still visibly reverses,
inputs at or below the cap pass through unchanged, and non-finite inputs
yield 0. Deterministic: a pure function of the input ω. Never feed the
clamped value to physics — pass true ω there (the cap is display-only and
applied after the true ω is computed, per spec §7.4).

Parameters:
- `trueOmegaRadPerS` - Signed true spin ω in rad/s from
 *  {@link trueWheelSpinOmega} .

Returns: The display-safe spin ω in rad/s, capped at 2 rev/s.

### computeVisualWheelSpinDeltaRad

```ts
computeVisualWheelSpinDeltaRad(
  tickCountDelta: number,
  timestepSeconds: number,
  groundSpeedMPerS: number,
  wheelRadiusM: number,
): number
```

Advances the display wheel spin for one batch of stepped physics ticks:
the true ω (v / r) is clamped to the 2 rev/s visual strobing cap first,
then the clamped omega integrates over `tickCountDelta` timesteps of
`timestepSeconds` each. The frame composer accumulates the returned
deltas so the displayed spin stays smooth and never strobes past the
cap (AC-015), while physics keeps the true omega (spec §7.4).

Safe fallbacks (deterministic total function): a non-finite ground speed
or radius yields the clamped zero omega, so the delta collapses to 0;
a non-positive tick delta returns 0 without touching the accumulator.

Parameters:
- `tickCountDelta` - Number of physics ticks elapsed (non-negative).
- `timestepSeconds` - Fixed timestep in seconds (0.5³ world seconds per
tick; 1/60 for the 60 Hz race loop).
- `groundSpeedMPerS` - Signed ground speed in m/s (positive = forward).
- `wheelRadiusM` - Wheel radius in meters; must be positive and finite.

Returns: The display spin delta in radians for this tick batch.

### NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S

Displayed wheel spin cap expressed in radians per second
(2 rev/s × 2π ≈ 12.566 rad/s), the exact value returned by
{@link clampVisualWheelSpinOmega} for over-cap inputs.

### NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S

Displayed wheel spin cap in revolutions per second (G9 strobing cap): the
display never renders faster than 2 rev/s no matter the ground speed. The
cap is applied after the true ω, so physics is unaffected.

### NEATRACING_WHEEL_GRID_SIZE

Normative wheel grid side in voxels (TP-14: wheel models use the 32-cubed
grid while car bodies use 128).

### NEATRACING_WHEEL_HUB_HEIGHT_M

Hub height in meters above the ground plane. Spec-pinned to the wheel's
world radius (G9), which puts the contact patch exactly at `y = 0`:
0.5625 − 0.5625 = 0.

### NEATRACING_WHEEL_MOUNT_YAW_RAD

Mount yaw in radians around the car's up axis. Spec §7.4: the mount rotates
the model 90° so the pre-mount depth (spin) axis becomes the lateral axis.
+π/2 maps the model +z axis onto the car's +x (car-right) axis, so the same
axle line serves the left and right wheels (spin sign is carried by ω).

### NEATRACING_WHEEL_PIVOT_FRACTIONS

Normative wheel rotation pivot in normalized [0, 1] grid fractions: the
geometric center of the cubic grid on every axis (G9).

### NEATRACING_WHEEL_PIVOT_VOXEL

Normative wheel rotation pivot in voxel coordinates: the disc's geometric
center, i.e. half the grid on every axis (G9: pivot = disc center).

### NEATRACING_WHEEL_SOCKET_ORDER

Deterministic socket order: front-left, front-right, rear-left, rear-right.

### NEATRACING_WHEEL_TRACK_M

Track width in meters (distance between the left and right wheel center
lines). Spec-unpinned F1-style tuning point for a ~4.5 m car (≈ 44 % of the
car length); retune here, not in geometry code.

### NEATRACING_WHEEL_VOXEL_WORLD_SIZE

Wheel voxel world size in meters per voxel. Shared with the car scale:
spec §7.4 requires wheel models at the car's voxel world scale, and the
scale constant stays owned by `./lod.distance.utils`.

### NEATRACING_WHEEL_WHEELBASE_M

Wheelbase in meters (distance between the front and rear axle lines,
measured from the car's model center front-to-back). Spec-unpinned F1-style
tuning point for a ~4.5 m car (≈ 71 % of the car length); retune here, not
in geometry code.

### NEATRACING_WHEEL_WORLD_RADIUS_M

Wheel world radius in meters (half the disc's world diameter):
1.125 / 2 = 0.5625 m (spec §7.4: the wheel's world radius).

### NEATRACING_WHEEL_WORLD_SIZE_M

Wheel model world extent in meters (grid side × voxel world size), i.e. the
tire's outer diameter for a disc spanning the declared height axis:
32 × 4.5/128 = 1.125 m.

### resolveVoxelWheelMetrics

```ts
resolveVoxelWheelMetrics(
  wheelModel: VoxelModelDocument,
): VoxelWheelMetrics
```

Resolves the wheel's world geometry from a compact wheel model document.

Derives the disc radius from the declared height axis
(`height / 2 × voxel world size`), the hub height (G9: equal to the radius),
and the grid-center rotation pivot. Expects the normative TP-14 wheel grid
(see {@link NEATRACING_WHEEL_GRID_SIZE}); deeper row/RLE/palette validation
stays owned by `./model.schema` and the decoder.

Parameters:
- `wheelModel` - A compact voxel model document authored as a wheel.

Returns: The wheel's world-space metrics.

### resolveVoxelWheelMountTransform

```ts
resolveVoxelWheelMountTransform(
  socketId: VoxelWheelSocketId,
  hubHeightM: number,
): VoxelWheelMountTransform
```

Resolves the mount transform for one wheel socket (G9): position at the
socket's hub offset, rotation pivot at the disc's geometric center
(`NEATRACING_WHEEL_PIVOT_FRACTIONS`), and the spec +90° mount yaw that maps
the pre-mount depth (spin) axis onto the lateral (car-right) axle line.

Parameters:
- `socketId` - Socket to transform; signs come from the shared axis table.
- `hubHeightM` - Hub height in meters; defaults to the normative wheel hub.

Returns: The car-frame mount transform for the socket.

### resolveVoxelWheelSockets

```ts
resolveVoxelWheelSockets(
  hubHeightM: number,
): readonly VoxelWheelSocket[]
```

Resolves the four wheel sockets at spec-defined offsets (plan AC: wheels
mount at 4 sockets; G9: hub height so the contact patch sits at y = 0).

Deterministic order: front-left, front-right, rear-left, rear-right.
Longitudinal offsets are ±`NEATRACING_WHEEL_WHEELBASE_M / 2` and lateral
offsets are ±`NEATRACING_WHEEL_TRACK_M / 2` around the car's model center
(`+x` car-right, `+z` forward). Override `hubHeightM` only when mounting a
non-normative wheel radius; the default keeps the contact patch at y = 0.

Parameters:
- `hubHeightM` - Hub height in meters; defaults to the normative wheel hub.

Returns: The four sockets in the pinned deterministic order.

### trueWheelSpinOmega

```ts
trueWheelSpinOmega(
  groundSpeedMPerS: number,
  wheelRadiusM: number,
): number
```

Computes the true wheel spin ω = v / r in rad/s from the ground speed,
advancing with ground speed (spec §7.4: rotation advances with ground
speed, `ω = v / r`). Sign is preserved: positive speed rolls forward
(positive ω about the car-right axle under the pinned +π/2 mount), negative
speed rolls in reverse. This value is physical truth — feed it to physics;
feed only the clamped result of {@link clampVisualWheelSpinOmega} to the
display.

Safe fallbacks (deterministic total function): a non-finite speed, or a
non-finite / non-positive radius, yields 0 (a stopped wheel) instead of
NaN or Infinity.

Parameters:
- `groundSpeedMPerS` - Signed ground speed in m/s (positive = forward).
- `wheelRadiusM` - Wheel radius in meters; must be positive and finite.

Returns: The true spin ω in rad/s, or 0 for unsafe inputs.

### VoxelWheelFormatError

Named rejection for wheel documents that do not satisfy the wheel-scale
format contract (spec §7.4 / TP-14: normative 32-cubed wheel grid).

#### code

Stable error code shared by every wheel-format rejection.

### VoxelWheelMetrics

World-space geometry derived from a wheel model document.

### VoxelWheelMountTransform

Full mount transform for one wheel socket (G9: hub height + 90° yaw).

### VoxelWheelSocket

One wheel socket: identity plus its hub-mount offset in the car frame.

### VoxelWheelSocketId

The four wheel sockets of the voxel racer.

### VoxelWheelSpinAxisMounted

Mounted spin axis label (car frame): the lateral car-right axis.

### VoxelWheelSpinAxisPreMount

Pre-mount spin axis label (authoring frame): the model's depth axis.

## voxel/lod.distance.utils.ts

LOD distance swap for Neat Racing voxel car tiers (spec §7.3.4; slice
P1-S3-E, AC-012).

The car renders in two tiers: the fine `lod128` tier (full 128³ grid) and
the coarse `lod64` tier (2×2×2 merge, §7.3.3). The tier is selected from
the camera distance so that the worst-case geometric error of a
coarse-vs-fine surface projects below a fixed pixel budget: the coarse
voxel is 2× the fine voxel, so the worst-case surface error is one fine
voxel (`NEATRACING_VOXEL_WORLD_SIZE`), and the swap happens when that
error projects below 0.5 px at the fixed render height and 60° FOV
(TP-06). The swap distance is therefore **computed** from car dimensions,
the fixed render height, and the FOV — the derived ~44 m value is a
consequence, not an input (LD-20).

Hysteresis (P4): near and far thresholds sit ±10% (TP-06) around the
computed swap distance so overtaking cars do not rapid-toggle tiers at
the boundary. The tier only changes when the distance crosses the edge on
the opposite side of the current tier; distances inside the band hold the
current tier. The selector is a pure function with zero per-frame
allocation (P9) — it returns existing tier constants, never a fresh
object.

### LOD_TIER_COARSE

The coarse car voxel tier: the 2×2×2-merged 64³ grid (spec §7.3.3).

### LOD_TIER_FINE

The fine car voxel tier: the full 128³ grid (spec §7.3.3; tier names are
locked to the spec wording).

### LodTier

Car voxel LOD tiers, named per spec §7.3.3 (`lod128` fine, `lod64`
coarse).

### NEATRACING_LOD_FINE_GRID_SIZE

Voxel axis resolution of the fine (lod128) car tier.

### NEATRACING_LOD_HYSTERESIS_BAND_FRACTION

Hysteresis band around the computed swap distance as a fraction
(TP-06/P4: ±10% separate near/far thresholds so overtaking cars do not
rapid-toggle tiers).

### NEATRACING_LOD_SWAP_DISTANCE_FAR_M

Far hysteresis edge in meters (swap distance + 10%): a fine-tier car
downgrades to the coarse tier only when the distance reaches this edge or
beyond.

### NEATRACING_LOD_SWAP_DISTANCE_M

Camera distance in meters at which one fine voxel of coarse-vs-fine
geometric error projects exactly the 0.5 px budget:
`swapDist = errWorld × focalPx / 0.5 ≈ 44 m` (spec §7.3.4; the value is
derived, TP-06 locks only the formula and the pixel threshold).

### NEATRACING_LOD_SWAP_DISTANCE_NEAR_M

Near hysteresis edge in meters (swap distance − 10%): a coarse-tier car
returns to the fine tier only when the distance drops to this edge or
below.

### NEATRACING_LOD_SWAP_FOCAL_PX

Focal length in pixels of the §7.3.4 projection basis, computed from the
fixed render height and the swap FOV:
`focalPx = (height / 2) / tan(fov / 2) ≈ (720 / 2) / tan(30°) ≈ 623 px`.

### NEATRACING_LOD_SWAP_FOV_DEG

Vertical field of view in degrees of the projection basis used for the
swap decision (spec §7.3.4: 60° FOV at the fixed render height).

### NEATRACING_LOD_SWAP_PIXEL_ERROR_PX

Projected pixel error at which the coarse tier becomes indistinguishable
from the fine tier (TP-06: 0.5 px projected error).

### NEATRACING_VOXEL_CAR_LENGTH_M

Longest car dimension in meters, assumed to fill the 128 voxel axis
(spec §7.3.4 G1 cubic-voxel assumption: length ≈ 4.5 m).

### NEATRACING_VOXEL_WORLD_SIZE

World size of one fine voxel in meters: the car's longest dimension
fills the 128 axis, so `carLength / 128` is the per-axis voxel size
(spec §7.3.4 named computed constant, AC-012 — not a magic number).

### selectLodTier

```ts
selectLodTier(
  distanceM: number,
  currentTier: LodTier,
): LodTier
```

Selects the car voxel LOD tier for a camera distance with hysteresis
(spec §7.3.4, AC-012).

The swap decision depends on the tier currently rendered, which is what
produces the hysteresis band: a fine (`lod128`) car stays fine until the
distance reaches the far edge (+10%), and a coarse (`lod64`) car stays
coarse until the distance drops to the near edge (−10%). Inside the band
— and at the base swap distance itself — the current tier is held, so an
overtaking car oscillating around the boundary never rapid-toggles
(mitigation for risk R-03).

The function is total and deterministic for any finite `distanceM`
(expected range is ≥ 0): a non-finite distance matches neither edge and
returns the current tier unchanged, which is the safe no-op fallback.

Parameters:
- `distanceM` - Camera distance to the car in meters (expected ≥ 0).
- `currentTier` - The tier currently rendered for the car.

Returns: The tier to render after this frame: `lod128` or `lod64`.

Example:

```ts
let tier: LodTier = LOD_TIER_FINE;
// Per frame, at the observed camera distance:
tier = selectLodTier(cameraDistanceM, tier);
```

## voxel/model.decode.utils.ts

RLE hex-row and mirror-X decoding for Neat Racing voxel models (spec
section 7.2; slice P1-S3-A, AC-007 + AC-008).

Row encoding: `<runLen:2hex><paletteIdx:2hex>` pairs; the runs of a row
must sum exactly to the declared row width — `dims[0]` cols for full
documents, `dims[0] / 2` stored cols when `meta.symmetric` is true.
Symmetric rows store the left half and are mirrored col c to col
`dims[0] - 1 - c` onto the full width.

The decoder emits one row-major `Uint8Array` per slice sized exactly from
the schema header (`dims[0] x dims[1]` palette indices); rows keep their
storage order, so row 0 stays ground/down (section 7.2). Malformed rows
are rejected with the named errors from `model.schema.ts`.

### DecodedVoxelModel

Decoded voxel model — typed arrays recovered from a validated section 7.2
document.

### decodeVoxelModel

```ts
decodeVoxelModel(
  document: VoxelModelDocument,
): DecodedVoxelModel
```

Decodes an on-disk model document into typed per-slice grids.

The document is re-validated first so the decoder is the single normative
gate for section 7.2 conformance; then every RLE hex row is expanded into
palette indices. Symmetric documents have their stored half-rows mirrored
onto the full declared width (col c mirrors to col `dims[0] - 1 - c`,
LD-28).

Parameters:
- `document` - The on-disk model document (validated internally).

Returns: The decoded model: one row-major `Uint8Array` of
`dims[0] * dims[1]` palette indices per slice, sized exactly from the
schema header.

Example:

```ts
const decoded = decodeVoxelModel(document);
const slice = decoded.slices[0];
const groundRowCol0 = slice[0]; // row 0, col 0 (ground row, front-left)
```

### decodeVoxelRow

```ts
decodeVoxelRow(
  row: string,
  expectedWidth: number,
  paletteSize: number,
): Uint8Array<ArrayBufferLike>
```

Expands one RLE hex row into palette indices.

Each `<runLen:2hex><paletteIdx:2hex>` pair fills `runLen` consecutive
cells with `paletteIdx`. The runs must sum exactly to `expectedWidth`
(64 stored cols for symmetric documents, 128 otherwise).

Parameters:
- `row` - RLE hex row string from the document.
- `expectedWidth` - Number of cells the row must cover exactly.
- `paletteSize` - Palette length; referenced indices must stay below it.

Returns: A `Uint8Array` of `expectedWidth` palette indices.

Example:

```ts
const cells = decodeVoxelRow('0a034000', 74, 5);
// cells[0..9] === 3 (10 voxels of palette index 3), cells[10..73] === 0
```

### decodeVoxelSlice

```ts
decodeVoxelSlice(
  rows: readonly string[],
  fullWidth: number,
  storedWidth: number,
  symmetric: boolean,
  paletteSize: number,
): Uint8Array<ArrayBufferLike>
```

Decodes one slice's rows into a row-major typed grid sized
`fullWidth * rows.length`, mirroring half-rows for symmetric documents.

Parameters:
- `rows` - The slice's RLE hex row strings in storage order.
- `fullWidth` - Full decoded width (`dims[0]`).
- `storedWidth` - Width each row must cover exactly (`dims[0] / 2`
stored cols for symmetric documents, `dims[0]` otherwise).
- `symmetric` - Whether stored half-rows must be mirrored.
- `paletteSize` - Palette length; referenced indices must stay below it.

Returns: The row-major typed grid for the slice.

### mirrorHalfRow

```ts
mirrorHalfRow(
  half: Uint8Array<ArrayBufferLike>,
  fullWidth: number,
): Uint8Array<ArrayBufferLike>
```

Mirrors a stored half-row (cols 0-63) onto the full width, mapping col c
to col `fullWidth - 1 - c` (LD-28 mirror-X symmetry).

Parameters:
- `half` - The stored half-row of palette indices.
- `fullWidth` - Full decoded width (`dims[0]`).

Returns: The mirrored full-width row.

### parseHexField

```ts
parseHexField(
  row: string,
  offset: number,
): number
```

Parses one 2-hex field, rejecting anything that is not exactly two hex
digits (a bare `parseInt` would silently misparse values such as `1g`).

Parameters:
- `row` - RLE hex row string.
- `offset` - Hex offset of the field within the row.

Returns: The parsed field value.

### parseVoxelRuns

```ts
parseVoxelRuns(
  row: string,
  paletteSize: number,
): VoxelRun[]
```

Parses the run pairs of one row, rejecting invalid hex, zero-length runs,
and out-of-range palette indices.

Parameters:
- `row` - RLE hex row string.
- `paletteSize` - Palette length; referenced indices must stay below it.

Returns: The decoded run pairs in row order.

### VOXEL_RLE_FIELD_HEX_CHARS

Hex characters per RLE field; both `<runLen>` and `<paletteIdx>` use 2.

### VOXEL_RLE_PAIR_HEX_CHARS

Hex characters per full `<runLen><paletteIdx>` pair (2 fields).

### VOXEL_RUN_LENGTH_MAX

Longest legal RLE run (two hex digits cap at 255).

### VOXEL_RUN_LENGTH_MIN

Shortest legal RLE run (section 7.2: run lengths are 1-255).

### VOXEL_SYMMETRIC_HALF_DIVISOR

Divisor mapping the full width to the stored symmetric half (128 to 64).

### VoxelRun

One decoded `<runLen><paletteIdx>` pair.

## voxel/memory.budget.utils.ts

Sparse voxel memory accounting for the Neat Racing voxel racer (spec
section 13.3; slice P3-S1-B, AC-036, risk R-07, appendix P7).

AC-036 asks for two proofs about the 128³ voxel racer:

1. **Sparse retention.** The *game runtime* never holds a dense 128³ byte
   grid. `decodeVoxelModel` (model.decode.utils.ts) emits one row-major
   `Uint8Array` per slice — 128 slices of 16,384 B for the car — and the
   boot path releases those slices as soon as the greedy mesher has
   produced the expanded vertex data the render passes upload to GL. The
   retained footprint is therefore the GL-side vertex data plus the
   palette, never one contiguous dense grid.
2. **A named budget.** The peak model memory of the boot pipeline stays
   under one named constant,
   {@linkcode NEATRACING_VOXEL_RUNTIME_MODEL_MEMORY_BUDGET_BYTES}.

The editor is the *bounded exception* of appendix P7 (the editor tool is a
separate page, never the game runtime): it may hold one dense working
volume plus count-bounded dense undo snapshots. This module owns that
reconciliation because AC-036 names the editor allowance, and it imports
the AC-020 constants (`NEATRACING_EDITOR_VOLUME_LENGTH`,
`NEATRACING_EDITOR_UNDO_MAX_STEPS`, `NEATRACING_EDITOR_UNDO_BUDGET_BYTES`)
instead of restating them, so the editor figures can never drift.

Scope boundaries:

- **LOD tiers are out of scope.** `buildLodModel` has no runtime call
  site today (only its own test), so no LOD tier footprint is accounted
  here; a future runtime LOD path must extend this module rather than
  assume the budget already covers it.
- **Pure data.** No DOM, no WebGL, no RNG: the module runs under the
  plain Node Jest lane (`jest:base`) and is safe to import from the
  editor, the browser entry, or a build script.
- **Dependency direction.** The engine imports the voxel folder, never the
  reverse, so this module imports nothing from `engine/` or
  `browser-entry/`.

Measurement note (peak semantics): the shipped measurements quoted by the
slice brief (car 2,146,628 B, wheel 76,088 B) are the *decode + mesh*
subtotal — the dense-grid reference plus the quad records plus the
expanded vertex data — and therefore exclude the palette bytes and the
mesher plane-mask transients that `pipelinePeakBytes` also carries. Both
readings agree on the boot aggregate: the subtotals sum to 2,222,716 B and
the mask transients add 52,224 B, so the boot peak is 2,274,940 B plus the
two 25-byte palettes, i.e. 2,274,990 B either way.

### accountEditorMemory

```ts
accountEditorMemory(
  editorUndoBytesInUse: number,
  retainedUndoSteps: number,
): EditorMemoryAccount
```

Accounts the editor lane against the appendix P7 exception and
reconciles it with the AC-020 constants (AC-036 scope note).

The reconciliation is arithmetic, not restated: the undo budget is
`NEATRACING_EDITOR_UNDO_MAX_STEPS × NEATRACING_EDITOR_VOLUME_LENGTH`, and
the envelope (working volume + undo budget) is exactly the 8 MiB
allowance (2,097,152 + 6,291,456 = 8,388,608).

Parameters:
- `editorUndoBytesInUse` - Bytes the undo stack currently retains
(measured with `EditorUndo.bytesInUse()`, never assumed).
- `retainedUndoSteps` - Dense snapshots the undo stack currently
retains (measured with `EditorUndo.size()`).

Returns: The editor memory account.

Example:

```ts
const undo = createEditorUndo(volume);
undo.push();
const account = accountEditorMemory(undo.bytesInUse(), undo.size());
account.withinUndoBudget; // true — 2,097,152 B of 6,291,456 B
account.editorExceptionEnvelopeBytes; // 8,388,608 — exactly the allowance
```

### accountVoxelBootMemory

```ts
accountVoxelBootMemory(
  models: readonly VoxelModelMemoryInput[],
): VoxelBootMemoryAccount
```

Aggregates the memory accounts of the models the boot path decodes
simultaneously and evaluates the aggregate against the named runtime
budget (AC-036).

Parameters:
- `models` - Models decoded together during boot; must not be empty.

Returns: The aggregated boot account with the budget evaluation.

Example:

```ts
const boot = accountVoxelBootMemory([carInput, wheelInput]);
boot.aggregatePipelinePeakBytes; // 2,274,990 — the boot worst instant
boot.withinBudget; // true — the budget is 2,621,440 B
```

### accountVoxelModelMemory

```ts
accountVoxelModelMemory(
  model: VoxelModelMemoryInput,
): VoxelModelMemoryAccount
```

Accounts the memory footprint of one voxel model through the boot
pipeline (AC-036).

The account separates three byte groups:

- the **decoded sparse footprint** (`decodedSliceBytes`, one
  `dims[0] * dims[1]` buffer per slice) versus the **dense reference**
  (`denseGridBytes`) the runtime must never hold as one buffer;
- the **mesh transients** (quad records, expanded vertex data, the mesher
  plane mask), which exist only while the model is being meshed and
  uploaded;
- the **retained footprint** (`retainedAfterBootBytes` = vertex data +
  palette), which is what the GL passes keep alive after the boot.

Parameters:
- `model` - Pure-data summary of the model's boot-pipeline footprint.

Returns: The memory account of the model.

Example:

```ts
const account = accountVoxelModelMemory({
  label: 'car-f1',
  dims: [128, 128, 128],
  quadCount: 217,
  paletteEntryCount: 5,
});
account.decodedSliceBytes; // 2,097,152 across 128 slices of 16,384 B
account.maxSingleSliceBufferBytes; // 16,384 — never the dense 2,097,152
account.retainedAfterBootBytes; // 46,897 — the vertex data plus the palette
```

### assertEditorMemoryWithinBudget

```ts
assertEditorMemoryWithinBudget(
  account: EditorMemoryAccount,
): true
```

Asserts that the editor lane stays inside the appendix P7 exception
(the named undo budget and the 8 MiB allowance).

Parameters:
- `account` - The editor memory account to inspect.

Returns: `true` when the account is within both limits.

Example:

```ts
const account = accountEditorMemory(undo.bytesInUse(), undo.size());
assertEditorMemoryWithinBudget(account); // true
```

### assertSparseVoxelRetention

```ts
assertSparseVoxelRetention(
  account: VoxelModelMemoryAccount,
): true
```

Asserts that a model account retains no dense volume grid (the sparse
half of AC-036).

The check is per buffer, never per class total: the car's 128 slices add
up to the dense 2,097,152 B, but no single buffer is that large, so the
class total is allowed while a dense single buffer is not.

Parameters:
- `account` - The model account to inspect.

Returns: `true` when the manifest holds no dense volume grid.

Example:

```ts
assertSparseVoxelRetention(carAccount); // true — 128 slices of 16,384 B
```

### assertValidEditorUndoUsage

```ts
assertValidEditorUndoUsage(
  editorUndoBytesInUse: number,
  retainedUndoSteps: number,
): void
```

Rejects malformed editor undo usage before any accounting runs.

Parameters:
- `editorUndoBytesInUse` - Bytes the undo stack retains.
- `retainedUndoSteps` - Snapshots the undo stack retains.

### assertValidModelMemoryInput

```ts
assertValidModelMemoryInput(
  model: VoxelModelMemoryInput,
): void
```

Rejects malformed model inputs before any accounting runs.

Parameters:
- `model` - The model input to validate.

### assertVoxelRuntimeMemoryWithinBudget

```ts
assertVoxelRuntimeMemoryWithinBudget(
  account: VoxelModelMemoryAccount | VoxelBootMemoryAccount,
): true
```

Asserts that the runtime memory account is sparse and within the named
AC-036 budget (the `assertVoxelRuntimeMemoryWithinBudget` contract).

Parameters:
- `account` - A per-model account or an aggregated boot account.

Returns: `true` when the account is sparse and within budget.

Example:

```ts
const boot = accountVoxelBootMemory([carInput, wheelInput]);
assertVoxelRuntimeMemoryWithinBudget(boot); // true
```

### buildRetainedManifest

```ts
buildRetainedManifest(
  model: VoxelModelMemoryInput,
  sliceBufferBytes: number,
  vertexDataBytes: number,
): VoxelBufferManifestEntry[]
```

Builds the typed-buffer manifest of one model: the sparse per-slice
grids, the expanded vertex data, and the palette.

Parameters:
- `model` - The validated model input.
- `sliceBufferBytes` - Bytes of one decoded slice grid.
- `vertexDataBytes` - Bytes of the expanded vertex buffer.

Returns: The manifest entries in allocation order.

### countDenseGridBuffers

```ts
countDenseGridBuffers(
  manifest: readonly VoxelBufferManifestEntry[],
  denseGridBytes: number,
): number
```

Counts the manifest entries that are dense volume grids.

Parameters:
- `manifest` - The retention manifest to scan.
- `denseGridBytes` - Dense-equivalent footprint of the grid.

Returns: How many entries are dense volume grids (0 when sparse).

### EditorMemoryAccount

Memory account of the editor lane (appendix P7 exception).

The editor is allowed one dense working volume plus count-bounded dense
undo snapshots; every byte figure here comes from the imported AC-020
constants, never from a restated literal.

### isBootMemoryAccount

```ts
isBootMemoryAccount(
  account: VoxelModelMemoryAccount | VoxelBootMemoryAccount,
): boolean
```

Narrows a runtime memory account to the aggregated boot account.

Parameters:
- `account` - A per-model account or an aggregated boot account.

Returns: True when the account is an aggregated boot account.

### largestPlaneCellCount

```ts
largestPlaneCellCount(
  dims: readonly [number, number, number],
): number
```

Computes the largest face-plane cell count of a grid: the greedy mesher
allocates one plane mask at a time, and the biggest plane sets the peak
mask footprint.

Parameters:
- `dims` - Grid dims `[width, height, depth]`.

Returns: Cells in the largest face plane.

### NEATRACING_EDITOR_MEMORY_EXCEPTION_ALLOWANCE_BYTES

Appendix P7 allowance for the editor exception: 8 MiB = 8,388,608 B.

The editor may hold one dense working volume plus count-bounded dense
undo snapshots; the envelope of the two (2,097,152 B + 6,291,456 B) lands
exactly on this allowance, which
{@linkcode accountEditorMemory} asserts.

### NEATRACING_VOXEL_DENSE_REFERENCE_BYTES

Dense 128³ reference footprint (2,097,152 B): the single contiguous byte
grid the *game runtime* must never retain. The editor working volume has
exactly this size, which is why the two lanes reconcile against one
shared dense reference.

### NEATRACING_VOXEL_DIM_COUNT

Axis count of a voxel grid (`[width, height, depth]`).

### NEATRACING_VOXEL_FLOAT32_BYTES

Bytes per Float32 element of the expanded vertex buffers.

### NEATRACING_VOXEL_MESHER_MASK_CELL_BYTES

Bytes per mesher plane cell: the `Int16Array` face mask (2 B) plus the
`Uint8Array` consumed map (1 B). The greedy mesher keeps one plane in
flight at a time, so the mask is a transient, not a retained buffer.

### NEATRACING_VOXEL_QUAD_RECORD_BYTES

Bytes of one greedy-mesh quad record in the `.bin` pipeline
(`position` u16 × 3, `normalAxis` u8, `width`/`height` u16 × 2,
`paletteIndex` u8).

Imported from `bin.codec` instead of restated so the runtime accounting
cannot drift from the shipped binary layout (AC-010).

### NEATRACING_VOXEL_QUAD_VERTICES

Vertices emitted per greedy quad: two counter-clockwise triangles.

### NEATRACING_VOXEL_RUNTIME_MODEL_MEMORY_BUDGET_BYTES

Named AC-036 budget for the peak model memory of the game runtime:
2.5 MiB = 2,621,440 B.

Derivation from the shipped assets (measured with `decodeVoxelModel` +
`meshVoxelModel` on `assets/models/car-f1.model.json` and
`assets/models/wheel.model.json`):

| model  | dense grid | quad records | vertex data | mask transients |
| ------ | ---------- | ------------ | ----------- | --------------- |
| car-f1 | 2,097,152  | 2,604        | 46,872      | 49,152          |
| wheel  | 32,768     | 2,280        | 41,040      | 3,072           |

The decode + mesh subtotals are 2,146,628 B (car) and 76,088 B (wheel),
i.e. 2,222,716 B together; the plane-mask transients add 52,224 B and the
two 25-byte palettes add 50 B, so the boot instant that decodes both
models peaks at 2,274,990 B. 2.5 MiB keeps 346,450 B (≈13 % of the
budget) of headroom over that aggregate while staying far below the
8 MiB editor allowance, which covers the separate editor tool lane (P7)
and is reconciled by {@linkcode accountEditorMemory}.

### NEATRACING_VOXEL_VERTEX_EXPANSION_BYTES_PER_QUAD

Bytes of expanded vertex data per greedy quad:
6 vertices × 9 floats × 4 B = 216 B.

### NEATRACING_VOXEL_VERTEX_FLOATS

Floats per expanded vertex in the widest runtime layout (the car-lit pass
uses 9; the glass pass uses 7, so the car is the worst-case rate).

### sumAccounts

```ts
sumAccounts(
  accounts: readonly VoxelModelMemoryAccount[],
  select: (account: VoxelModelMemoryAccount) => number,
): number
```

Sums one byte figure across the per-model accounts.

Parameters:
- `accounts` - The per-model accounts to sum.
- `select` - Selects the byte figure to sum.

Returns: The sum of the selected byte figures.

### VoxelBootMemoryAccount

Aggregated memory account of the models the boot path decodes at once.

### VoxelBufferKind

Typed-buffer classes a voxel model can hold.

- `slice-grid` — one decoded per-slice palette-index grid.
- `vertex-data` — the expanded Float32 vertex buffer uploaded to GL.
- `palette` — the palette entries (r, g, b, a, glow).

### VoxelBufferManifestEntry

One typed-buffer class in a retention manifest.

### VoxelMemoryBudgetError

Named rejection for every memory-budget violation (AC-036).

Carries a stable machine-readable `code` so callers can branch on
rejection kinds without parsing human-readable messages.

#### code

Stable rejection code, for example `VOXEL_MEMORY_BUDGET_EXCEEDED`.

### VoxelModelMemoryAccount

Memory account of a single voxel model.

`pipelinePeakBytes` is the worst instant of the boot pipeline for this
model: dense-grid reference + palette + quad records + expanded vertex
data + the mesher plane-mask transient.
`retainedAfterBootBytes` is what survives the boot: the expanded vertex
data plus the palette (the slices and the quad records are released).

### VoxelModelMemoryInput

Pure-data summary of one model's boot-pipeline footprint.

Callers derive `quadCount` from `meshVoxelModel` output and
`paletteEntryCount` from the decoded palette.
