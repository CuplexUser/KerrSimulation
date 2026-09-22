// Bloom.
//
// The accretion disk spans a huge dynamic range: the beamed inner edge is orders
// of magnitude brighter than the outer rim. Without bloom the only way to make it
// read as *luminous* is to raise the overall level until the core clips, which
// flattens the whole disk to white and throws away the color ramp with it.
//
// Real glare is not one Gaussian: it has a tight core and long tails. This is
// the downsample/upsample chain from Jimenez's "Next Generation Post Processing
// in Call of Duty": a bright pass into half resolution, successive 13-tap
// downsamples to 1/64, then tent-filtered upsamples that add each level back
// into the one above it. The renderer weights deeper (wider) levels down
// geometrically, so the result is a core with a soft falloff rather than a haze
// — a bright ring beside a dark shadow bleeds inward, and the shadow must stay
// black.
//
// A single wide blur used to do this job, and its sparse taps could ring. Every
// level here is filtered before it is decimated, so nothing aliases.

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
  // background index, unused, unused, unused
  sky: vec4f,
}

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<uniform> U: Uniforms;
@group(0) @binding(2) var samp: sampler;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VertexOut {
  var corners = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0),
  );
  let corner = corners[index];
  var out: VertexOut;
  out.position = vec4f(corner, 0.0, 1.0);
  // From the vertex rather than the fragment position, so each pass needs no
  // uniform for the size of the level it writes.
  out.uv = vec2f(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
  return out;
}

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

/// Bilinear sample of the presented region of the (unfilterable, rgba32float)
/// accumulation texture. Same as sampleRegion in present.wgsl.
fn sampleAccum(uv: vec2f) -> vec3f {
  let region = U.view.xy;
  let maxTexel = vec2i(region) - 1;
  let p = uv * region - 0.5;
  let base = floor(p);
  let f = p - base;
  let i = vec2i(base);
  let c00 = textureLoad(source, clamp(i, vec2i(0), maxTexel), 0).rgb;
  let c10 = textureLoad(source, clamp(i + vec2i(1, 0), vec2i(0), maxTexel), 0).rgb;
  let c01 = textureLoad(source, clamp(i + vec2i(0, 1), vec2i(0), maxTexel), 0).rgb;
  let c11 = textureLoad(source, clamp(i + vec2i(1, 1), vec2i(0), maxTexel), 0).rgb;
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

/// Bright pass into the first level. Four taps across the destination texel,
/// each weighted by 1 / (1 + luma) — Karis's average — so a single firefly
/// sample in a still-converging image cannot flash a bloom-sized blob.
@fragment
fn fsDownsampleFirst(in: VertexOut) -> @location(0) vec4f {
  let texel = 0.5 / U.bloom.xy;
  var total = vec3f(0.0);
  var weight = 0.0;
  for (var i = 0; i < 4; i++) {
    let offset = vec2f(f32(i & 1) * 2.0 - 1.0, f32(i >> 1) * 2.0 - 1.0) * texel;
    let c = sampleAccum(in.uv + offset) * U.frame.w;
    let w = 1.0 / (1.0 + luminance(c));
    total += c * w;
    weight += w;
  }
  let color = total / weight;

  // Soft threshold, so the bloom fades in rather than switching on at an edge.
  let threshold = U.bloom.z;
  let l = luminance(color);
  let contribution = max(l - threshold, 0.0) / max(l, 1e-4);
  return vec4f(color * contribution, 1.0);
}

/// 13-tap downsample: a center box plus four overlapping corner boxes, which
/// together form a kernel wide enough that decimating by two cannot alias.
@fragment
fn fsDownsample(in: VertexOut) -> @location(0) vec4f {
  let t = 1.0 / vec2f(textureDimensions(source));
  let uv = in.uv;

  let a = textureSampleLevel(source, samp, uv + t * vec2f(-2.0, -2.0), 0.0).rgb;
  let b = textureSampleLevel(source, samp, uv + t * vec2f(0.0, -2.0), 0.0).rgb;
  let c = textureSampleLevel(source, samp, uv + t * vec2f(2.0, -2.0), 0.0).rgb;
  let d = textureSampleLevel(source, samp, uv + t * vec2f(-1.0, -1.0), 0.0).rgb;
  let e = textureSampleLevel(source, samp, uv + t * vec2f(1.0, -1.0), 0.0).rgb;
  let f = textureSampleLevel(source, samp, uv + t * vec2f(-2.0, 0.0), 0.0).rgb;
  let g = textureSampleLevel(source, samp, uv, 0.0).rgb;
  let h = textureSampleLevel(source, samp, uv + t * vec2f(2.0, 0.0), 0.0).rgb;
  let i = textureSampleLevel(source, samp, uv + t * vec2f(-1.0, 1.0), 0.0).rgb;
  let j = textureSampleLevel(source, samp, uv + t * vec2f(1.0, 1.0), 0.0).rgb;
  let k = textureSampleLevel(source, samp, uv + t * vec2f(-2.0, 2.0), 0.0).rgb;
  let l = textureSampleLevel(source, samp, uv + t * vec2f(0.0, 2.0), 0.0).rgb;
  let m = textureSampleLevel(source, samp, uv + t * vec2f(2.0, 2.0), 0.0).rgb;

  let center = (d + e + i + j) * (0.5 / 4.0);
  let corners = (a + b + f + g + b + c + g + h + f + g + k + l + g + h + l + m) * (0.125 / 4.0);
  return vec4f(center + corners, 1.0);
}

/// 3x3 tent upsample of the smaller level. The pipeline blends the result
/// additively into the level above, scaled by a blend constant.
@fragment
fn fsUpsample(in: VertexOut) -> @location(0) vec4f {
  let t = 1.0 / vec2f(textureDimensions(source));
  let uv = in.uv;
  var total = textureSampleLevel(source, samp, uv, 0.0).rgb * 4.0;
  total += textureSampleLevel(source, samp, uv + t * vec2f(-1.0, 0.0), 0.0).rgb * 2.0;
  total += textureSampleLevel(source, samp, uv + t * vec2f(1.0, 0.0), 0.0).rgb * 2.0;
  total += textureSampleLevel(source, samp, uv + t * vec2f(0.0, -1.0), 0.0).rgb * 2.0;
  total += textureSampleLevel(source, samp, uv + t * vec2f(0.0, 1.0), 0.0).rgb * 2.0;
  total += textureSampleLevel(source, samp, uv + t * vec2f(-1.0, -1.0), 0.0).rgb;
  total += textureSampleLevel(source, samp, uv + t * vec2f(1.0, -1.0), 0.0).rgb;
  total += textureSampleLevel(source, samp, uv + t * vec2f(-1.0, 1.0), 0.0).rgb;
  total += textureSampleLevel(source, samp, uv + t * vec2f(1.0, 1.0), 0.0).rgb;
  return vec4f(total / 16.0, 1.0);
}
