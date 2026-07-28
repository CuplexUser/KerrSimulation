// Presents the accumulation texture to the swap chain.
//
// Compute shaders cannot write the canvas texture directly, so a minimal render
// pass does the final composite, tone map and transfer-function encode.
//
// The accumulation texture is not always canvas-sized: while the camera is being
// dragged the renderer traces at a coarser resolution to keep the interaction
// immediate. So this samples with linear filtering rather than loading 1:1.
// rgba16float is filterable in core WebGPU, so no optional feature is needed.

struct Uniforms {
  camPos: vec4f,
  camRight: vec4f,
  camUp: vec4f,
  camFwd: vec4f,
  params: vec4f,
  // frame index, trace resolution x, trace resolution y, exposure
  frame: vec4f,
  // disk enabled, max steps, canvas width, canvas height
  options: vec4f,
  band: vec4f,
  // bloom width, bloom height, threshold, strength
  bloom: vec4f,
}

@group(0) @binding(0) var accum: texture_2d<f32>;
@group(0) @binding(1) var<uniform> U: Uniforms;
@group(0) @binding(2) var accumSampler: sampler;
@group(0) @binding(3) var bloomTex: texture_2d<f32>;

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
  let uv = in.position.xy / vec2f(U.options.z, U.options.w);

  let radiance = textureSampleLevel(accum, accumSampler, uv, 0.0).rgb * U.frame.w;
  let glow = textureSampleLevel(bloomTex, accumSampler, uv, 0.0).rgb;

  // Additive: bloom is light that scattered off the bright regions, so it adds
  // to them rather than replacing anything.
  let mapped = acesFilmic(radiance + glow * U.bloom.w);

  // The preferred canvas format is typically bgra8unorm (not -srgb), so the
  // transfer function is applied here rather than by the swap chain.
  return vec4f(pow(mapped, vec3f(1.0 / 2.2)), 1.0);
}
