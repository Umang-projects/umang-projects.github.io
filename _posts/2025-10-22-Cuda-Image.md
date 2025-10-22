---
title: "From Confusion to Colors: My Journey Learning Parallel Image Magic with CUDA Threads"
date: 2025-10-22 15:21:00 +05:30
layout: post
author: Umang Singh
excerpt: "A beginner's journey from confusion to creating fractal art using CUDA threads — intuitive steps, exercises, and code snippets."
categories:
  - CUDA
  - Parallel Computing
  - Maths
  - Image
tags:
  - cuda
  - gpu
  - optimization
  - performance
  - warp-divergence
  - parallel-computing
  - profiling
  - shared-memory
  - matrix-multiplication
  - learning-journey
---


Hey folks! I’m just a curious learner wandering into the GPU universe — and wow, it’s a ride. A few weeks ago, I cracked open Chapter 5 of *Programming Massively Parallel Processors* (yep, the Kirk & Hwu classic), and my brain nearly melted.  
Threads? Blocks? Thousands of tiny workers painting pixels at once? Total chaos in my head.

But instead of just copying examples or staring at code, I built a **mental picture** — piece by piece — on paper. This post is my story of how that messy start turned into fractal art on my screen.  
No heavy theory, just the thoughts and visuals that finally *clicked*.  

By the end, I went from *“What even is a thread?”* to *“Whoa, I just made this GPU draw!”*  
If you’ve ever been stuck in that same fog, grab a cup of coffee and come along. Let’s turn confusion into color.

---

## Why This Mattered: The "Aha" Spark

Think about this: On a CPU, making an image means looping over every pixel — one after another — painfully slow.  
A GPU, though, is like hiring a stadium full of painters where **each one handles a pixel** simultaneously.

Sounds simple, right? Not quite. I kept wondering:  
"How does each worker know *where* to paint without a manager shouting orders?"

Books showed vector additions and ripple effects, but I didn’t *feel* it.  
So I dropped the code editor, grabbed a notebook, and drew grids.  

That's when it clicked: if pixels are soldiers in formation, and every thread knows its rank, there’s no chaos — just math.  

Once I saw the image as a **grid of independent jobs**, parallelism stopped being magic and started making sense.

---

## Step 1: Picture the Battlefield – Pixels as a Grid Army

Start simple: every image is a 2D grid — say, **512×512** pixels. That's over **250,000 dots**, each needing an RGB color (values between 0–255).

Now forget the CPU loop. On the GPU, you **launch threads** — one per pixel.  
Each thread paints its own square.

But how does it know *where* to paint?

Think of threads grouped into **blocks** (like squads).  
Each block has a fixed size — say **16×16** threads.  
The GPU launches a bunch of these blocks to cover the whole image.

Each thread calculates its own global coordinates:

```cuda
x = threadIdx.x + blockIdx.x * blockDim.x
y = threadIdx.y + blockIdx.y * blockDim.y
```

Then, to store color in a 1D array:

```cuda
index = x + y * total_width
```

That’s like flattening your chessboard into a line — every soldier still knows where their square is.

Once I imagined this, everything made sense:  
CPU = one painter with a brush.  
GPU = thousands of painters with spray guns — fast and synchronized.

---

## Step 2: Daily Warm-Ups – Building "Thread Thinking" Without Code

Forget typing for a while. The best clarity came when I trained my brain on paper.

Here are a few mini-exercises that helped me “see” parallelism:

1. **Draw the Grid Game**  
   Sketch a 4×4 grid. Label x (->) and y (↓).  
   Pick a pixel (say 2,3).  
   If the rule is `color = red if x > y`, shade it.  
   Try different rules and feel how each cell behaves uniquely.

2. **Invent Silly Rules**  
   - Gradient: `color = x / width * 255`  
   - Circle: color based on `distance from center`  
   - Checkerboard: `(x + y) % 2 == 0 ? white : black`

   Each one builds intuition — every pixel follows the same rule, but results differ by position.

3. **Block Drill**  
   Imagine 2×2 thread blocks inside a 4×4 grid.  
   Ask: what’s the global position of thread (1,1) in block (1,0)?  
   Compute it by hand:  
   `x = 1 + 1*2 = 3`, `y = 1 + 0*2 = 1`.

4. **Edge and Error Thinking**  
   What happens if a thread’s `(x, y)` exceeds the image boundary?  
   It should simply `return`.  
   This mental debugging builds muscle memory before coding.

After a few days of this, I started *seeing grids in my sleep*.  
Parallelism stopped being lines of code — it became *motion in my head.*

---

## Step 3: Hands-On – Tweaking Code That Made It Click

Once the visuals made sense, I jumped into **Google Colab** (hello free GPU!) and started modifying small CUDA templates.

Here’s the minimal kernel I played with:

```cuda
__global__ void paint_pixels(unsigned char *colors, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;

    // --- Rule playground ---

    // 1. Solid blue
    unsigned char r = 0, g = 0, b = 255;

    // 2. Gradient fade (left to right)
    r = (unsigned char)(x * 255 / (width - 1));
    g = 0; b = 0;

    // 3. Checkerboard pattern
    int chunk = 16;
    int cx = x / chunk, cy = y / chunk;
    unsigned char shade = ((cx + cy) % 2 == 0) ? 255 : 0;
    r = g = b = shade;

    // Store RGBA
    colors[idx + 0] = r;
    colors[idx + 1] = g;
    colors[idx + 2] = b;
    colors[idx + 3] = 255;
}
```

Launch setup:

```cpp
dim3 threads(16, 16);
dim3 blocks(width / 16 + 1, height / 16 + 1);
paint_pixels<<<blocks, threads>>>(d_colors, width, height);
```

Then save it to an image using Python and `matplotlib`.

First run: boring solid blue.  
Next tweak: smooth red gradient — it *worked!*  
Then came waves, circles, and glowing rings.

The best part? I didn’t rewrite the logic — I just kept changing the rule section and re-ran.

Every run taught me something new about boundaries, memory layout, or symmetry.

---

## Step 4: From Waves to Fractals – The Real “GPU Moment”

Once I had the hang of pixels, I couldn’t resist trying a **Mandelbrot fractal**.  
Each thread represented one point on the complex plane, looping until the number "escaped."

CPU version: 10 seconds per frame.  
GPU version: real-time zooms.  
No contest.

Then I added zoom and pan controls to explore different regions.  
Seeing fractals render *live* was that "I made it!" moment.  
It proved what parallel thinking could do — visually and computationally.

Now, the next dream: real-time ray tracing.

---

## Wrapping Up – Your Turn to Paint

From paper sketches to pixel shaders, this journey flipped my understanding of parallelism.  
It’s not just a programming trick — it’s a **way of thinking**:  
break big jobs into countless tiny ones that don’t depend on each other.  
