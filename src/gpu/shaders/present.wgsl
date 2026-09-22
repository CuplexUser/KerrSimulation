// Presents the accumulation texture to the swap chain.
//
// Compute shaders cannot write the canvas texture directly, so a minimal render
// pass does the final composite, tone map and transfer-function encode.
//
// The accumulation texture is allocated once at full trace resolution, but the
// image in it is often smaller: while the camera moves the renderer traces a
// coarser region in its top-left corner, and the last complete image stays on
// screen while the next pass is in flight. view.xy says how much of the texture
// the presented image covers, and it is scaled up to the canvas from there.
//
// The accumulation is rgba32float, which is not filterable in core WebGPU, so
// the bilinear filter is done by hand from four loads.

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
  disk: vec4f,
  // presented width, presented height, dither seed, pixel angle
  view: vec4f,
}

@group(0) @binding(0) var accum: texture_2d<f32>;
@group(0) @binding(1) var<uniform> U: Uniforms;
@group(0) @binding(2) var bloomSampler: sampler;
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

/// Bilinear sample of the presented region of an unfilterable texture. `uv`
/// spans the presented image, not the whole texture.
fn sampleRegion(tex: texture_2d<f32>, uv: vec2f, region: vec2f) -> vec3f {
  let maxTexel = vec2i(region) - 1;
  let p = uv * region - 0.5;
  let base = floor(p);
  let f = p - base;
  let i = vec2i(base);
  let c00 = textureLoad(tex, clamp(i, vec2i(0), maxTexel), 0).rgb;
  let c10 = textureLoad(tex, clamp(i + vec2i(1, 0), vec2i(0), maxTexel), 0).rgb;
  let c01 = textureLoad(tex, clamp(i + vec2i(0, 1), vec2i(0), maxTexel), 0).rgb;
  let c11 = textureLoad(tex, clamp(i + vec2i(1, 1), vec2i(0), maxTexel), 0).rgb;
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
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

/// The exact sRGB transfer function. pow(1/2.2) is close in the midtones but
/// lifts the deep shadows, which is exactly where this image lives.
fn srgbEncode(linear: vec3f) -> vec3f {
  let low = linear * 12.92;
  let high = 1.055 * pow(linear, vec3f(1.0 / 2.4)) - 0.055;
  return select(high, low, linear <= vec3f(0.0031308));
}

fn hash12(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

/// Bloom levels are summed with geometric weights; this divides that back out.
/// Mirrors BLOOM_UPSAMPLE_WEIGHT and BLOOM_LEVELS in KerrRenderer.ts:
/// sum of 0.6^i for i = 0..5.
const BLOOM_NORMALIZE: f32 = 1.0 / 2.3834;

@fragment
fn fs(in: VertexOut) -> @location(0) vec4f {
  let uv = in.position.xy / vec2f(U.options.z, U.options.w);

  let radiance = sampleRegion(accum, uv, U.view.xy) * U.frame.w;
  let glow = textureSampleLevel(bloomTex, bloomSampler, uv, 0.0).rgb * BLOOM_NORMALIZE;

  // Additive: bloom is light that scattered off the bright regions, so it adds
  // to them rather than replacing anything.
  let mapped = acesFilmic(radiance + glow * U.bloom.w);

  // The preferred canvas format is typically bgra8unorm (not -srgb), so the
  // transfer function is applied here rather than by the swap chain.
  let encoded = srgbEncode(mapped);

  // Triangular-PDF dither of one 8-bit step. The sky is a near-black gradient
  // and the disk's outer tail fades into it, and both band visibly when
  // quantized straight to 8 bits. The seed moves every frame so the pattern
  // reads as fine grain rather than a fixed texture.
  let seed = in.position.xy + vec2f(U.view.z * 17.0, U.view.z * 31.0);
  let noise = hash12(seed) + hash12(seed + 71.7) - 1.0;
  return vec4f(encoded + noise / 255.0, 1.0);
}
