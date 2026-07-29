// Bloom.
//
// The accretion disk spans a huge dynamic range: the beamed inner edge is orders
// of magnitude brighter than the outer rim. Without bloom the only way to make it
// read as *luminous* is to raise the overall level until the core clips, which
// flattens the whole disk to white and throws away the color ramp with it.
//
// Bright-pass at quarter resolution, separable Gaussian blur, then additive
// composite in the present pass. Three cheap passes at 1/16 the pixel count —
// nothing next to the cost of integrating geodesics.

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

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<uniform> U: Uniforms;
@group(0) @binding(2) var samp: sampler;

struct VertexOut {
  @builtin(position) position: vec4f,
}

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VertexOut {
  var corners = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0),
  );
  var out: VertexOut;
  out.position = vec4f(corners[index], 0.0, 1.0);
  return out;
}

// Five-tap linear-sampled Gaussian: the offsets sit between texels so hardware
// filtering does half the work, giving a nine-tap kernel for five fetches.
// Written out rather than looped over an array — WGSL does not allow indexing a
// module-scope const array with a runtime index.
const W0: f32 = 0.2270270270;
const W1: f32 = 0.3162162162;
const W2: f32 = 0.0702702703;
const O1: f32 = 1.3846153846;
const O2: f32 = 3.2307692308;

/// Widens the blur beyond a single Gaussian pass. Kept fairly tight on purpose:
/// a wide spread smears the white core out over the orange body of the disk and
/// the whole band reads as washed-out white rather than incandescent.
const BLOOM_SPREAD: f32 = 1.7;

/// Isolates the part of the image bright enough to glow, with a soft knee so the
/// bloom fades in rather than switching on at a hard edge.
@fragment
fn fsBright(in: VertexOut) -> @location(0) vec4f {
  let uv = in.position.xy / U.bloom.xy;

  // One bloom texel spans several accumulation texels, so take a small box
  // rather than a single sample. Point-sampling here aliases badly against the
  // starfield, whose cells are laid out on a regular grid — the beat between the
  // two grids shows up as blocky squares once the bloom is upscaled again.
  let texel = 1.0 / vec2f(U.frame.y, U.frame.z);
  var color = textureSampleLevel(source, samp, uv + texel * vec2f(-1.0, -1.0), 0.0).rgb;
  color += textureSampleLevel(source, samp, uv + texel * vec2f(1.0, -1.0), 0.0).rgb;
  color += textureSampleLevel(source, samp, uv + texel * vec2f(-1.0, 1.0), 0.0).rgb;
  color += textureSampleLevel(source, samp, uv + texel * vec2f(1.0, 1.0), 0.0).rgb;
  color += textureSampleLevel(source, samp, uv, 0.0).rgb;
  color = color * (U.frame.w / 5.0);

  let threshold = U.bloom.z;
  let luminance = dot(color, vec3f(0.2126, 0.7152, 0.0722));
  let knee = max(luminance - threshold, 0.0);
  let contribution = knee / max(luminance, 1e-4);

  return vec4f(color * contribution, 1.0);
}

fn blur(uv: vec2f, direction: vec2f) -> vec4f {
  let texel = direction * BLOOM_SPREAD / U.bloom.xy;
  let d1 = texel * O1;
  let d2 = texel * O2;

  var total = textureSampleLevel(source, samp, uv, 0.0).rgb * W0;
  total += textureSampleLevel(source, samp, uv + d1, 0.0).rgb * W1;
  total += textureSampleLevel(source, samp, uv - d1, 0.0).rgb * W1;
  total += textureSampleLevel(source, samp, uv + d2, 0.0).rgb * W2;
  total += textureSampleLevel(source, samp, uv - d2, 0.0).rgb * W2;
  return vec4f(total, 1.0);
}

@fragment
fn fsBlurH(in: VertexOut) -> @location(0) vec4f {
  return blur(in.position.xy / U.bloom.xy, vec2f(1.0, 0.0));
}

@fragment
fn fsBlurV(in: VertexOut) -> @location(0) vec4f {
  return blur(in.position.xy / U.bloom.xy, vec2f(0.0, 1.0));
}
