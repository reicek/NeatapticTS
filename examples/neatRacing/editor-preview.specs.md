# NeatRacing Editor Preview Data Flow

> Permanent trace for the served `/editor/` page preview lane (slice `P1-editor-preview`, AC-019). The editor page renders **only** the selected model preview; no game-scene race passes run here.

## Boot flow

```mermaid
flowchart TD
    A[<code>load</code> event] --> B[Append editor bundle script]
    B --> C[window.neatRacingEditorStart
previewHostId]
    C --> D[Mount preview canvas
100% host size]
    D --> E[initRenderingContext]
    E -- GL2 available --> F[createEditorCamera]
    F --> G[createEditorPreviewBinsDraw]
    G --> H[startEditorPreviewRenderLoop]
    H --> I[Best-effort fetch first model
/api/models]
    I --> J[Loop memoizes document -> bins]
    E -- fallback --> K[Show fallback notice]
```

## Per-frame preview loop

```mermaid
sequenceDiagram
    autonumber
    participant RAF as requestAnimationFrame
    participant PREVIEW as startEditorPreviewRenderLoop
    participant CAM as EditorCamera
    participant RES as resolvePreviewBins
    participant DEC as decodeVoxelModel
    participant MESH as meshVoxelModel
    participant DRAW as createEditorPreviewBinsDraw
    participant GL as WebGL2 context

    RAF->>PREVIEW: timestamp (ms)
    PREVIEW->>CAM: camera.tick(dt)
    PREVIEW->>PREVIEW: getSelectedModelDocument
    alt document != null
        PREVIEW->>RES: resolvePreviewBins(memo, document)
        alt document identity changed
            RES->>DEC: decodeVoxelModel(document)
            DEC-->>RES: decoded voxel model
            RES->>MESH: meshVoxelModel(decoded)
            MESH-->>RES: VoxelQuad[] + dims
            RES->>RES: build VoxelBinModel
        end
        RES-->>PREVIEW: bins or null
        alt bins != null
            PREVIEW->>CAM: buildViewMatrix
            CAM-->>PREVIEW: view matrix
            PREVIEW->>CAM: buildProjectionMatrix(aspect)
            CAM-->>PREVIEW: projection matrix
            PREVIEW->>PREVIEW: multiplyViewProjection(viewProjection, view, projection)
            Note over PREVIEW: projection times view, column-major, same contract as game composer
            PREVIEW->>PREVIEW: orbitCameraWorld(pose, cameraWorld)
            PREVIEW->>DRAW: draw({ viewProjection, cameraWorld, bins })
            DRAW->>GL: viewport + clear color/depth
            DRAW->>GL: carPass.draw(...)
        end
    end
    PREVIEW->>RAF: schedule next frame
```

## Bins draw adapter

The editor preview reuses the production `createCarPass` from `engine/passes/car.pass.ts` by wrapping the selected document's bins as the car `body` and an empty wheel bin as the `wheel`. This satisfies the AC-019 dogfooding requirement (spec section 8.3): the preview lane draws through the same decoder, mesher, and renderer used by the race scene.

## Critical invariants

1. **Preview-only composition**: The editor page does **not** run starfield, grid, pulse, track, glass, or bloom passes. It clears the default framebuffer each frame and draws only the selected model.
2. **Matrix contract parity**: The editor loop uses the same `projection * view` column-major product as the game composer, because it feeds the same `car.pass.ts` `uViewProjection` uniform.
3. **Memoized decode/mesh**: `resolvePreviewBins` caches by document reference so a steady selection decodes and meshes exactly once; decode failures are cached and logged once.
4. **RAF survival**: Any failure during `createCarPass` or document decode is caught and logged; the RAF loop keeps ticking and the preview stays blank rather than crashing the page.
