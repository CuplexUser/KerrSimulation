# Kerr

An interactive, physically-based Kerr black hole. Null geodesics are integrated
through the real Kerr metric in a WebGPU compute shader, and the image refines
progressively while the camera holds still.

![The render: an accretion disk lensed over and under the shadow](docs/render.png)

## Running it

```bash
pnpm install
pnpm dev
```

Needs WebGPU — Chrome or Edge 113+ with hardware acceleration on. If the device
is unavailable the app says so and explains what to check.

Drag to orbit, scroll to zoom. Sliders control spin, disk outer radius, render
resolution, and exposure.

## The physics

Geometric units, `G = c = M = 1`. Kerr-Schild coordinates, signature `-+++`:

```
g_uv = eta_uv + f k_u k_v          exact inverse: g^uv = eta^uv - f k^u k^v
```

The metric is static, so `p_t` is exactly conserved. It is fixed at `p_t = -1`
and is **not** integrated — the state is position and spatial momentum only:

```
H(x, p) = 0.5 (-1 + dot(p,p) - f S^2),   S = 1 + dot(k, p)
dx/dl   = p - f S k                       exact closed form
dp/dl   = -dH/dx                          central differences
```

RK4 with an adaptive step `dl = clamp(r * 0.045, 0.015, 0.9)`. A ray terminates
when it crosses `r < r_plus * 1.02` (captured, black), passes `r > 60` (escaped,
sample the background), or exhausts its step budget.

Two details worth knowing:

- **The gradient differences `W = -0.5 f S²`, not the full `H`.** The remaining
  `0.5(-1 + dot(p,p))` term has zero `x`-derivative, so the gradient is identical
  — but the GPU runs in f32, where subtracting two O(1) Hamiltonians to recover an
  O(1e-5) difference is catastrophic cancellation.
- **Rays launch with momentum along the view direction**, then the null constraint
  `H = 0` is solved exactly for its magnitude. This is the coordinate-camera
  convention; a static observer's orthonormal tetrad would be more correct, but the
  difference is a smooth FOV-level aberration that is negligible at `r >= 15`, and
  the minimum zoom radius sits well outside the ergosphere.

The accretion disk is deliberately stylized — a temperature falloff, the standard
Shakura-Sunyaev inner-boundary factor, and a crude Doppler beaming term. It is not
radiative transfer.

Colour comes from the *observed* temperature, `T_obs = g · T_emit`, where `g` is the
Doppler factor. That single term carries the disk from crimson on the receding rim
through amber to blue-white on the approaching limb, and it is why the two sides
look so different.

## Validating it

Sign errors in a geodesic RHS produce a plausible-looking but wrong picture, so the
physics is checked numerically in two places against two invariants: the null
condition `H = 0`, and the axial angular momentum `L = x·py − y·px`.

```bash
pnpm validate:physics
```

Runs the f64 reference in `src/physics/kerrReference.ts` under Node and asserts
conservation over thousands of RK4 steps, plus the characteristic radii, the
frame-dragging asymmetry (exactly zero at `a = 0`), and an epsilon/step-size sweep
that distinguishes finite-difference truncation from an actual sign error.

The second check runs the **shipped WGSL** on the GPU and compares it against f64
values computed live in the browser — no stored golden numbers, so it cannot go
stale. Open the app and press **Run** under "Physics check", or load `/?validate=1`.

`src/physics/kerrReference.ts` and `src/gpu/shaders/kerr_math.wgsl` are line-for-line
mirrors of each other. Edit one, edit the other.

## Layout

```
scripts/validate-physics.mts    Node assertion suite over the f64 reference
src/physics/kerrReference.ts    f64 reference implementation, shared by both harnesses
src/gpu/
  KerrRenderer.ts               device, textures, pipelines, frame loop, resize
  camera.ts  uniforms.ts        orbit camera; uniform packing
  validatePhysics.ts            GPU-vs-reference comparison
  shaders/
    kerr_math.wgsl              the physics — prepended to trace and validate
    trace.wgsl                  ray generation, integration, disk, accumulation
    bloom.wgsl                  bright pass and separable blur
    present.wgsl                composite, tone map onto the swap chain
    validate.wgsl               single-ray invariant tracking
src/ui/                         panel, orbit controls, radial scale
```

## How the rendering works

Two passes per frame. A compute pass traces one jittered sample per pixel and
blends it into a running average; a render pass tone maps that onto the canvas,
since compute shaders cannot write the swap chain directly.

Accumulation is **double-buffered** rather than a single read-write texture. That
is not a portability hedge: baseline WebGPU permits `read_write` storage-texture
access only on `r32float`/`r32uint`/`r32sint`, and `rgba16float` read-write sits
behind the `rw-storage-texture-tier-2` extension.

Sample jitter uses the R2 low-discrepancy sequence indexed by frame, with a
per-pixel hash applied as a Cranley-Patterson rotation. The blend weight is
`1/n`, so the first pass after a reset fully overwrites and no clear is needed.
Any camera or parameter change resets the count; accumulation stops at 1024
samples and simply re-presents.

**One sample is split across several dispatches**, a horizontal band per frame.
Integrating a whole image in a single submission can occupy a modest GPU for the
better part of a second, and a submission that long stalls compositing and input
handling for the entire tab — the page reads as frozen and slider drags queue up
behind it. The band count auto-tunes to hold each dispatch near a 16 ms budget;
the panel reports it as "Split". Measured main-thread task lateness while
rendering is ~5 ms.

While the camera is moving the trace also drops to 40% resolution and snaps back
once it has been still for ~180 ms. Each RK4 step evaluates the metric ~28 times,
so pixel count dominates the frame cost — the resolution slider is the lever to
pull if the interaction still feels heavy.

Finally, a three-pass bloom (bright pass, then a separable blur at quarter
resolution) is composited additively before tone mapping. Without it the only way
to make the disk read as luminous is to raise the level until the core clips,
which flattens the whole disk to white and discards the colour ramp.

## Scripts

| Command | Does |
|---|---|
| `pnpm dev` | Dev server |
| `pnpm build` | Typecheck and build |
| `pnpm lint` | oxlint, type-aware |
| `pnpm typecheck` | `tsc -b` |
| `pnpm validate:physics` | The Node conservation suite |
