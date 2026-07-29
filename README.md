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

Drag to orbit, scroll to zoom. Sliders control spin, disk outer radius, disk
thickness, Doppler beaming, render resolution, exposure and glow. The control
panel can be dragged, resized and collapsed — see [The panel](#the-panel).

## Deploying it

`.github/workflows/deploy.yml` builds and publishes to GitHub Pages on every push
to `main` or `master`. Enable it once under **Settings → Pages → Source → GitHub
Actions**. The workflow runs lint, typecheck and the physics suite before it will
publish anything.

`vite.config.ts` reads `GITHUB_REPOSITORY` to set Vite's `base`, so a project
site served from `/<repo>/` gets the right asset paths without the repository
name being written down anywhere. Locally the variable is unset and the base
stays `/`. Pages serves over HTTPS, which WebGPU requires — opening `dist/`
from the filesystem will not work.

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

Color comes from the *observed* temperature, `T_obs = g · T_emit`, where `g` is the
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

## Testing it

```bash
pnpm test
```

A regression suite on Node's built-in runner — no test framework, no browser, no
GPU. It covers what the physics harnesses do not:

- **Shader contracts.** Every mirrored constant in `kerr_math.wgsl` is checked
  against its `kerrReference.ts` counterpart, and the `Uniforms` struct in
  `trace.wgsl` against the one in `present.wgsl` and the slot order
  `packUniforms` writes. That drift is otherwise invisible: it fails at shader
  compilation in a browser, or not at all — a uniform slot swapped with its
  neighbour just renders the wrong picture at full speed.
- **Render scheduling.** `KerrRenderer` is driven against a recording stand-in
  device (`test/support/fakeWebGpu.ts`) with a fake clock, which pins down that
  the bands of a pass tile the image exactly, that the presented texture is
  always the last complete one, and that exposure and bloom do not throw away
  converged samples while spin and step count do.
- **Camera and input.** Basis orthonormality and handedness — a mirrored black
  hole still looks like a black hole — plus drag/zoom direction and listener
  teardown.
- **The f64 reference**, as individual assertions rather than one exit code,
  including the step-control helpers and the flat-space limits.

Shader imports use Vite's `?raw`, which plain Node cannot resolve, so
`test/support/wgslRaw.ts` installs a module hook that loads them. It goes in via
`--import` because ESM links the whole graph before running anything.

Two things stay out of reach headlessly, both by nature: the *bodies* of the WGSL
functions, which only the in-app harness above can run, and the React components.

## Layout

```
scripts/validate-physics.mts    Node assertion suite over the f64 reference
test/                           regression suite (pnpm test)
  support/wgslRaw.ts            module hook so Node can load `?raw` shaders
  support/fakeWebGpu.ts         recording stand-in device, for the frame loop
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
src/ui/
  Controls.tsx                  the panel
  usePanelLayout.ts             drag, resize, collapse, persistence
  orbitControls.ts              pointer and wheel to camera
  RadialScale.tsx               horizon / ISCO / disk / camera on one axis
```

## The panel

Drag the header to move it, the bottom-right grip to resize, and the chevron (or
a double-click on the header) to collapse it to a title bar that still shows the
sample count. Position, size and collapsed state persist across reloads, and it
is clamped to stay fully on screen — otherwise the resize grip can end up past
the bottom edge with no way to get it back.

None of that touches the renderer. The panel floats over the canvas rather than
sharing layout with it, so moving or resizing it does not resize the swap chain
and never discards accumulated samples; a panel drag costs no trace frames.
Gestures write geometry straight to the DOM and commit to React state once on
release, for the same reason the orbit camera lives in a ref — a 60 Hz pointer
stream must not reach the render path.

**Restore defaults** puts every control *and* the camera back to their starting
values. **Reset panel** restores the panel's own geometry, which is separate —
you can rearrange the workspace without disturbing the scene, and vice versa.

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
which flattens the whole disk to white and discards the color ramp.

### Lensing level of detail

Right against the shadow there is a razor-thin bright arc, and it used to render
ragged and stair-stepped. Working out why took several wrong guesses, so the
result is worth writing down.

The direct image of the disk maps screen position to disk radius smoothly. Every
*higher-order* image — light that wound around the hole one or more times before
reaching the camera — is compressed exponentially, and by the second or third
pass the entire radial profile is squeezed into a band thinner than one pixel.
Point sampling that is aliasing a signal with no band limit.

What settled it was rendering the disk in a flat color: the same arcs came out
perfectly smooth. So the geometry was never wrong — the raggedness was the
*shading* being sampled below its resolvable scale.

The fix is a level of detail keyed to the image order, which the tracer already
knows because it counts equatorial crossings. Past the first image the radial
detail is faded out: the striation goes to its phase average and the emission
profile settles to the bright inner ring that dominates the stack anyway. It is
the trade a mip level makes for a minified texture — the detail is not
resolvable, so don't pretend to resolve it.

Ruled out by measurement first, in case they look tempting: the integration step
budget (450 → 2000 changed nothing), the striation texture (disabling it entirely
changed nothing), and correlation in the sample sequence (permuting the R2 index
per pixel changed nothing). Giving the disk finite thickness also did not fix it,
though the control survives as a real feature.

One genuine bug did turn up next door. A ray grazing the equatorial plane could
enter and leave it inside a single RK4 step, and since the disk test only sees
one sign change per step, that crossing was dropped entirely. `planeLimitedStep`
now shortens any step that would jump the plane. Its derivative comes from the
RK4 stage-one evaluation, which the loop hoists and feeds back in through
`rk4StepFrom`, so the check costs no extra metric evaluations — it removed one.

## Scripts

| Command | Does |
|---|---|
| `pnpm dev` | Dev server |
| `pnpm build` | Typecheck and build |
| `pnpm lint` | oxlint, type-aware |
| `pnpm typecheck` | `tsc -b` |
| `pnpm test` | Regression suite, `node --test` |
| `pnpm validate:physics` | The Node conservation suite |
