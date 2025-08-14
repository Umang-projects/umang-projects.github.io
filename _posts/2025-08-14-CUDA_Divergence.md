---
title:"From Matrix Multiplication to Warp Optimizations — My Journey and Insights"
subtitle: "Work-in-progress: Controlling warp divergence, efficient reductions, and memory-efficient attention kernels"
categories: [CUDA, Maths]
tags: [cuda,gpu,optimization,performance,warp-divergence,parallel-computing,profiling,shared-memory,matrix-multiplication,learning-journey]
---

**Hey everyone —** welcome to my CUDA notes-turned-blog! I started confused about how GPU threads decide work in warps and ended up experimenting with matrix multiplication, reductions, branch divergence, and warp tricks. This is a single, friendly post that combines those learnings into a practical guide with intuition, small code examples, and follow-up ideas. Treat it as my learning journal made readable — candid, practical, and useful.

---

## 🔹 Quick roadmap (what you'll read)

- Row vs column parallelism in matrix multiplication
- How warps and SIMT actually run threads
- Parallel reductions: why `__syncthreads()` matters
- Two practical ways to fix branch divergence (arithmetic tricks and warp grouping)
- Even/odd sums example (step-by-step)
- Shuffle-based reduction snippet and a fused idea
- Practical checklist and next experiments

---

## Starting with the Basics: Row and Column Parallelism in Matrix Multiplication

When I first wrote CUDA matmul kernels I asked: *is inner-loop work serial per thread?* The short answer: **yes, inside a thread it's serial; across threads it's parallel**.

### Row-per-thread pattern

Each thread computes a whole output **row** (or part of it). Inside a thread, you loop across columns and the `k` dimension. But warps (groups of 32 threads) run the same instructions on different data — so a warp computes 32 rows in lockstep.

```cpp
__global__ void MatrixMul_RowPerThread(float* M, float* N, float* P, int Width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < Width) {
        for (int col = 0; col < Width; col++) {
            float Pvalue = 0.0f;
            for (int k = 0; k < Width; k++) {
                Pvalue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = Pvalue;
        }
    }
}
```

### Column-per-thread pattern

Flip the roles: each thread computes a column (or chunk of columns). The inner serial loop changes, but the warp still gives you parallelism across many rows/columns.

**Key intuition:** The inner loops are serial *inside a thread*, but total parallel work comes from many threads (warps) working concurrently.

---

## Warps and SIMT: How Threads “Decide” Their Batch

There’s no runtime decision by threads about warps — warps are fixed hardware groups of 32 consecutive thread IDs (tid 0..31 = warp 0, 32..63 = warp 1, etc.). If you need warp-aligned behavior, compute `group = tid / 32` and `lane = tid % 32`.

**SIMT** (Single Instruction Multiple Threads) means each lane executes the same instruction. If lanes diverge (some take `if` true, others false), the GPU serializes the branches and masks off inactive lanes — which wastes cycles.

---

## Parallel Reductions: Syncs, Races, and Why Barriers Matter

Reductions like summing a vector need synchronization. I learned the hard way: moving `__syncthreads()` out of the reduction loop caused incorrect sums due to race conditions.

**Why**: threads write partial results into shared memory. Without a sync, another thread may read before that write completes — producing stale reads and wrong sums.

Correct pattern (tree reduction):

1. Each thread writes a partial value into shared memory.
2. For stride = blockDim/2 down to 1: reduce pairs, `__syncthreads()` after each step.
3. Final result in `sdata[0]`.

This ensures correct reads and avoids races.

---

## Controlling Divergence — Two Practical Techniques

Divergence reduces performance because warps serialize branches. Two practical fixes that actually helped me:

### Technique 1 — Arithmetic-based branch elimination

Replace `if-else` with arithmetic expressions that every thread executes. Boolean expressions become `0/1` masks (no branching) and multiply results accordingly.

**Divergent version**:

```cpp
if (x > 0) y = x * 2; else y = 0;
```

**Arithmetic version (no branch)**:

```cpp
y = (x > 0) * (x * 2);  // (x > 0) is 1 or 0
```

All lanes run the same instruction — no warp split. This trick is especially useful for simple conditional computations that can be expressed mathematically.

### Technique 2 — Restructure so similar threads are in the same warp

If you have many different branches based on `tid % 2` or similar, remap so a whole warp executes the same path.

Bad (highly divergent):

```cpp
if (tid % 2 == 0) { /* even path */ } else { /* odd path */ }
```

Better (warp-aligned grouping):

```cpp
int group = tid / 32;  // maps warps to groups
int lane = tid % 32;
if (group == 0) { /* even-group path (all lanes in warp 0) */ }
else { /* odd-group path (warp 1) */ }
```

Now each warp follows one uniform path — no internal masking.

---

## Even/Odd Sum Example — Step-by-step

Let's compute sums for numbers `1..64` split into even and odd sums. Two approaches:

### Approach A — Arithmetic mask + shuffle-reduce

- Compute `evenVal = ((tid & 1) == 0) * value;` for each lane. Odds compute similarly.
- Use warp-level shuffle (`__shfl_down_sync`) to reduce across lanes. No branching, all lanes active.

### Approach B — Warp grouping

- Remap work so `warp 0` processes all even indices and `warp 1` processes all odd indices. Each warp reduces within itself using shuffles — no divergence.

Both give correct results and avoid costly divergence. The shuffle-based approach is compact and very efficient for reductions inside a warp.

---

## Shuffle-based reduction snippet (warp-level)

This is my favorite compact reduction inside a warp — no shared memory needed.

```cpp
// warp sum using shuffle down
float val = /* per-lane value */;
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
}
// lane 0 now has the sum for the warp
```

If you need the warp-sum at a particular lane, broadcast it with a `__shfl_sync` call. For block-level sums you can combine warp-shuffle results with a shared-memory final pass.

---

## Fusing Ideas (why FlashAttention-style fusion matters)

One big practical lesson for me: **reducing memory traffic often beats raw compute optimization**. For attention, materializing big `scores` matrices and then running softmax and another matmul causes heavy global memory traffic.

Fusing steps (matmul → scale → softmax → multiply-V) with stable-blocking techniques avoids storing full intermediate matrices and cuts memory traffic — this is the idea behind FlashAttention. It’s advanced, but the intuition is simple: **keep data in registers/shared memory where possible, and minimize global reads/writes**.

---

## Practical checklist — what to try next (small steps)

- Try arithmetic replacement for simple branches in your kernels. Measure.
- If branches depend on tid patterns (even/odd), consider remapping indices so warps are uniform.
- Use shuffle reductions for warp-level sums, combine with shared memory for block-level.
- Profile before and after: look for reduced warp stalls and lower runtime.
- If your algorithm materializes large temporaries, consider fusion as a next step.

---

## Visuals & Canva suggestions (how to turn this into a nice blog graphic)

If you want to create a Canva post/carousel, make these 3 slides:

1. **Problem + Hook** — short story + profiler pie (hotspot).
2. **Two Techniques** — left: arithmetic trick (with 1-line example). Right: warp grouping (with a diagram of warps).
3. **Results + Next Steps** — show before/after numbers or a checklist.

Use simple diagrams: warps as rows of 32 boxes, highlight active lanes, and show masks (0 vs 1). Add a small code snippet box for the shuffle reduction.

---

## Final thoughts (honest & humble)

I started confused and made many small mistakes — but focusing on intuition (bits, masks, barriers) and small experiments helped the most. 

