// Presents the accumulation texture to the swap chain.
//
// Compute shaders cannot write the canvas texture directly, so a minimal render
// pass does the final tone map and transfer-function encode. The accumulation
// texture is exactly canvas-sized, so this is a 1:1 textureLoad — no sampler and
// no filtering ambiguity.

struct Uniforms {
  camPos: vec4f,
  camRight: vec4f,
  camUp: vec4f,
  camFwd: vec4f,
  params: vec4f,
  // frame index, resolution x, resolution y, exposure
  frame: vec4f,
  options: vec4f,
}

@group(0) @binding(0) var accum: texture_2d<f32>;
@group(0) @binding(1) var<uniform> U: Uniforms;

struct VertexOut {
  @builtin(position) position: vec4f,
}

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VertexOut {
  // Single oversized triangle covering the viewport — no vertex buffer needed.
  var corners = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0),
  );
  var out: VertexOut;
  out.position = vec4f(corners[index], 0.0, 1.0);
  return out;
}

/// Narkowicz's ACES filmic approximation.
fn acesFilmic(x: vec3f) -> vec3f {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4f {
  let radiance = textureLoad(accum, vec2i(in.position.xy), 0).rgb;
  let mapped = acesFilmic(radiance * U.frame.w);

  // The preferred canvas format is typically bgra8unorm (not -srgb), so the
  // transfer function is applied here rather than by the swap chain.
  return vec4f(pow(mapped, vec3f(1.0 / 2.2)), 1.0);
}
