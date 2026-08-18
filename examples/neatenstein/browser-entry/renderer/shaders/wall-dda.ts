/**
 * GLSL ES 3.00 shader source for the Neatenstein wall-DDA pass.
 *
 * This module exports the fragment shader source as a string so a future GPU
 * (WebGL2/WebGPU) tier can compile it at runtime. The shader implements the
 * same DDA perpendicular wall distance formula as the CPU tier:
 *
 * ```glsl
 * perpWallDist = (mapX - posX + (1 - stepX) / 2) / dirX;
 * ```
 *
 * This is NOT Euclidean distance — using camera-space perpendicular distance
 * avoids fish-eye distortion.
 *
 * The shader source is a **forward-looking scaffold** for a future GPU tier.
 * It is not compiled or dispatched by the current CPU-tier renderer. The source
 * string is exported so that alignment tests can verify the GLSL matches the
 * CPU-tier algorithm.
 *
 * @module
 */

/**
 * Fragment shader source for the wall-DDA pass.
 *
 * Contains `highp` precision qualifiers and a full DDA ray-marching loop that
 * computes `perpWallDist` per-fragment. The loop mirrors the CPU-tier
 * `castRayDDAFromFlatMap` algorithm: step through grid cells, detect wall
 * hits, and compute the perpendicular distance using the canonical formula.
 */
export const NEATENSTEIN_WALL_DDA_SHADER_SOURCE = `#version 300 es
precision highp float;
precision highp int;

uniform vec4 uCameraPos;    // (posX, posY, 0, 0)
uniform vec4 uCameraDir;    // (dirX, dirY, 0, 0)
uniform vec4 uCameraPlane;  // (planeX, planeY, 0, 0)
uniform float uFocalLength;
uniform float uRenderDistanceCap;
uniform sampler2D uWallMap; // R = wall (1.0 = wall, 0.0 = empty)

out vec4 fragColor;

// Perpendicular wall distance — NOT Euclidean distance.
// perpWallDist = (mapX - posX + (1 - stepX) / 2) / dirX
// This avoids fish-eye distortion by using camera-space distance.
float computePerpWallDist(float mapX, float posX, float stepX, float dirX) {
  return (mapX - posX + (1.0 - stepX) * 0.5) / dirX;
}

void main() {
  // Map fragment X to a camera-plane offset in [-1, 1].
  float screenX = gl_FragCoord.x;
  float planeOffset = (screenX / uFocalLength) * 2.0 - 1.0;

  // Ray direction: forward + plane * offset
  vec2 rayDir = uCameraDir.xy + uCameraPlane.xy * planeOffset;
  vec2 pos = uCameraPos.xy;

  // DDA setup — deltaDist is 1/|rayDir| per axis.
  vec2 deltaDist = abs(rayDir) > 0.0 ? 1.0 / abs(rayDir) : vec2(1e30);
  ivec2 mapPos = ivec2(floor(pos));

  ivec2 step;
  vec2 sideDist;
  if (rayDir.x < 0.0) {
    step.x = -1;
    sideDist.x = (pos.x - float(mapPos.x)) * deltaDist.x;
  } else {
    step.x = 1;
    sideDist.x = (float(mapPos.x) + 1.0 - pos.x) * deltaDist.x;
  }
  if (rayDir.y < 0.0) {
    step.y = -1;
    sideDist.y = (pos.y - float(mapPos.y)) * deltaDist.y;
  } else {
    step.y = 1;
    sideDist.y = (float(mapPos.y) + 1.0 - pos.y) * deltaDist.y;
  }

  // DDA traversal loop.
  float perpWallDist = uRenderDistanceCap;
  int side = 0;
  bool hit = false;
  for (int i = 0; i < 64; i++) {
    if (sideDist.x < sideDist.y) {
      sideDist.x += deltaDist.x;
      mapPos.x += step.x;
      side = 0;
    } else {
      sideDist.y += deltaDist.y;
      mapPos.y += step.y;
      side = 1;
    }

    // Check bounds and sample wall map.
    if (mapPos.x < 0 || mapPos.x >= 64 || mapPos.y < 0 || mapPos.y >= 64) {
      break;
    }

    float cell = texelFetch(uWallMap, ivec2(mapPos.x, mapPos.y), 0).r;
    if (cell > 0.5) {
      hit = true;
      if (side == 0) {
        perpWallDist = computePerpWallDist(float(mapPos.x), pos.x, float(step.x), rayDir.x);
      } else {
        perpWallDist = (float(mapPos.y) - pos.y + (1.0 - float(step.y)) * 0.5) / rayDir.y;
      }
      break;
    }
  }

  if (!hit || perpWallDist >= uRenderDistanceCap) {
    // No wall hit — output background (fully fogged).
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  // Wall shading: side-based dimming (X-side brighter, Y-side darker).
  float dim = (side == 0) ? 1.0 : 0.75;
  fragColor = vec4(0.0, 0.72 * dim, 1.0 * dim, 1.0);
}
`;