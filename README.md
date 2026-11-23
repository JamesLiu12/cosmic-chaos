# CosmicChaos

*A tiny universe with a questionable life expectancy.*

[Live Demo](https://jamesliu12.github.io/cosmic-chaos/)

## Introduction

I love the visual chaos of celestial bodies pulling each other in, growing, and blowing apart. I also enjoy GPU programming, so this project is a chance to build a tiny universe while exploring parallel simulation and performance optimization!

## Goal

Build a 3D gravity sandbox in WebGPU where bodies attract each other, collide, merge, and eventually explode into smaller fragments. A tiny universe with plenty of opportunities for chaos!

## Inspiration / Reference

**[Collider — Robert Hodgin](https://roberthodgin.com/project/collider)** is my main visual and behavioral reference. I like its cycle of attraction, growth, ignition, and explosion, especially how glowing bodies stand out among smaller particles.

![Collider by Robert Hodgin](https://roberthodgin.com/v2/wp-content/uploads/2020/05/collider_screen_shot_02.png)

*Reference image: Robert Hodgin, Collider.*

**[Fast N-Body Simulation with CUDA — GPU Gems 3, Chapter 31](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda)** by Lars Nyland, Mark Harris, and Jan Prins is my main technical reference. I'll adapt its gravitational softening and tiled shared-memory computation to WGSL.

![Tiled N-body computation](https://developer.download.nvidia.com/books/gpugems3/31fig02.jpg)

*Reference diagram: GPU Gems 3, Figure 31-2.*

## Specification

The core simulation features are mutual gravity, collisions, merging, and explosions. I’m aiming for around 10k+ planets and stars, depending on performance. I’ll compare a naive all-pairs solver with a tiled solver that reuses data in workgroup memory.

Planets and stars will have procedurally generated appearances inspired by real celestial bodies, with varied colors, surface patterns, and glowing effects.

### Presentation

The final product will run in a web browser, allowing users to explore the tiny universe with a movable camera. Controls will adjust simulation speed, gravity strength, the explosion threshold, and the initial number of bodies.

If time allows, I’ll add interactions such as clicking a body to trigger an explosion or using the mouse to apply an external force.

### Visual Assets

No external visual assets are needed. Planets and stars will be generated procedurally using mass, color, and a random seed. Noise-based shaders will create surface patterns and variation.

## Techniques

- **Gravity and integration:** Use softened Newtonian gravity and fixed-timestep leapfrog integration, following the GPU Gems chapter linked above.
- **Parallel computation:** Compare naive all-pairs gravity with a tiled version that loads body data into WGSL workgroup memory for reuse.
- **Collisions and merging:** Detect overlapping spheres, select non-conflicting pairs, and apply merges in separate compute passes.
- **Explosions:** Split bodies above a mass threshold into fragments, distributing the parent's mass and reusing inactive slots in a fixed-capacity GPU pool.
- **Rendering:** Read body data directly from GPU buffers for instanced rendering. Noise-based surface patterns, emission, and bloom will give planets and stars their appearance.

## Design

Compute passes update the body state on the GPU. The rendering pipeline reads the updated buffers directly to draw the bodies and apply bloom.

![Program Structure](program-structure.svg)

## Timeline

This is a solo project, so I'll handle simulation, rendering, interaction, and performance testing.

- **Week 1 — by Nov 11:** Set up WebGPU, camera controls, and procedural body generation. Implement the naive gravity solver, motion integration, and basic collision detection.
- **Week 2 — by Nov 18:** Implement merging, explosions, and fragment recycling. Add the tiled gravity solver and compare its performance with the naive version.
- **Week 3 — by Nov 23:** Finish procedural surfaces, glow, and simulation controls. Test larger body counts, fix simulation issues, and finalize the core features.
- **Week 4 — by Nov 30:** Polish the visuals and scene presets, add optional interactions if time permits, and prepare the browser demo, video, and final writeup.

## Milestone #1

I hit all my Milestone #1 goals! The WebGPU sandbox has a fly-around camera, seeded body generation, and a first pass at rocky surfaces.

The simulation uses a fixed-step leapfrog integrator. Bodies move half a step, get a velocity update from softened all-pairs gravity, then move the remaining half step. 

A separate GPU pass checks for overlaps, which I can highlight in the debug view. 

The renderer draws straight from the simulation’s GPU buffer, so body positions never need to make a round trip through the CPU.

### Generator examples

<img width="1267" height="728" alt="Default scene with 500 rocky bodies" src="https://github.com/user-attachments/assets/e02f0d8e-2c34-47a5-9281-1387aea0e8e7" />

*Seed 0, 500 bodies. The default scene starts with a cloud of spinning rocks of different sizes.*

<img width="1266" height="725" alt="Sparse scene with 50 bodies in debug mode" src="https://github.com/user-attachments/assets/4ff5c361-02c3-4471-8d4e-a058e58f6ca3" />

*Seed 42, 50 bodies. The debug mode highlights collision candidates.*

Please check out the [live demo](https://jamesliu12.github.io/cosmic-chaos/) and try a seed of your own!

Next up are merging, explosions, and a faster gravity solver.

## Milestone #2

I hit my core Milestone #2 goals! The sandbox now starts with 2,000 rocky bodies. When gravity brings bodies together and they overlap, a larger one can gradually absorb mass from a smaller one. A body lights up when it reaches the star threshold. At the explosion threshold, it starts to collapse, pulling nearby bodies inward before sending out a shockwave and scattering fragments.

Each star gets a warm or cool color from its seed. Noise gives it patches of different brightness, and the hotter areas turn whiter as the star gains mass. Rocky planets have layered surface noise and cracks that glow brighter as they grow. Ambient light gives the rocks a little color in the dark, nearby stars cast colored light that fades with distance, and bloom gives bright stars and cracks a soft glow.

### Performance

I built baseline versions of the five main simulation stages first, then optimized each one. Tiled gravity reuses body data in workgroup memory, and a spatial grid limits collision checks to nearby bodies. Parallel scans pack surviving bodies, group neighbors and donors into contiguous lists, and make room for fragments from simultaneous explosions. Reductions help fit the grid to the scene and divide a star's mass among its fragments.

I compared both presets in Chrome 157 on an NVIDIA RTX 3090 at **2560 × 1440**. Each run started from seed 0 with 5,000 bodies and used the default scene settings and a fixed camera. I let the simulation run for 10 seconds of simulated time, then measured the next 900 frames, through simulated second 25. During that window, the active count ranged from 4,655 to 8,294 in Baseline and from 4,655 to 8,228 in All Optimizations as bodies were absorbed and stars broke apart.

I used WebGPU timestamps at each stage boundary. Every frame includes two fixed physics steps of 1/120 second each and one complete surface render.

| Stage | Baseline (mean ms/frame) | All Optimizations (mean ms/frame) |
| --- | ---: | ---: |
| Half-step drift | 0.017 | 0.017 |
| Gravity and integration | 2.854 | 1.839 |
| Collision detection | 2.710 | 0.673 |
| Absorption | 2.194 | 0.248 |
| Compaction | 6.972 | 0.129 |
| Explosion | 7.570 | 0.214 |
| Dispatch size and step counter | 0.004 | 0.004 |
| **Simulation subtotal** | **22.321** | **3.124** |
| Render setup and star light collection | 0.016 | 0.015 |
| Body drawing and surface shading | 1.907 | 2.477 |
| Bloom | 0.319 | 0.475 |
| Compositing and tone mapping | 0.046 | 0.076 |
| **Rendering subtotal** | **2.288** | **3.043** |
| **Whole frame** | **24.609** | **6.168** |

The median whole-frame GPU time dropped from **18.84 ms to 5.77 ms**, about a **3.3× speedup**. Compaction and explosions saved the most time. The demo uses the optimized preset by default, and I also tested it with 12,000 starting bodies.

I also changed **Star Mass** while replaying the same 60 snapshots at 2560 × 1440. More bodies lit up as stars, and every shaded rock pixel had more lights to check.

| Star Mass | Mean star lights per frame | Mean rendering time | 95th percentile |
| ---: | ---: | ---: | ---: |
| 80 | 21.2 | 2.47 ms | 3.93 ms |
| 40 | 75.3 | 4.28 ms | 7.60 ms |
| 20 | 453.6 | 14.86 ms | 18.74 ms |

### Controls

I can regenerate the scene with a new seed, body count, initial velocity range, or boundary box. Gravity and drag can be tuned while the simulation runs, and star and explosion thresholds take effect when I regenerate. I can also adjust bloom, exposure, and the camera, or switch to the collision debug view. Under **Performance → Simulation**, I can switch presets or try each algorithm on its own.

### Generator examples

![Red, orange, and blue stars among rocky bodies with glowing cracks](images/bodies-1.png)

*Warm and cool stars cast colored light across a field of rocky bodies.*

![A close-up of two bright stars lighting the surrounding rocks](images/bodies-2.png)

*A hot star lights up nearby rocky planets. The planets' glowing cracks are visible even on their darker sides.*

<p>
  <img src="images/controls-1.png" width="320" align="top" alt="Initial scene, boundary, physics, and simulation algorithm settings" />
  <img src="images/controls-2.png" width="320" align="top" alt="Simulation presets, rendering settings, and camera controls" />
</p>

*The control panel showing the scene, physics, simulation, rendering, and camera settings.*

### Before the final submission

I've met all my project goals! I'll do my best to tackle a few stretch goals before the final submission. Rendering is my biggest priority. I'd like to try clustered forward+ and deferred rendering to see how they handle scenes with lots of star lights. I'll also look for ways to speed up gravity and collision detection and try adding a little displacement to the rocky surfaces.
