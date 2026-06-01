"""STUDENT FILE: implement the Triton kernels and pipeline drivers.

You implement:
  - Six @triton.jit kernels: f1_kernel, f2_kernel, transpose_kernel,
    f4_kernel_L2, dft_kernel, bailey_scale_kernel.
  - The f1_launch and f2_launch grid-choice wrappers around them.
  - The pipeline drivers: f3_launch, f5_launch, _f6_rec, _f7_rec.
  - f6_factor: the chunk-recipe for F6/F7.

You do NOT implement (left given below):
  - The thin launch wrappers _transpose, _fft_chunk, _scale, _lookup_tw.
    These are mechanical "pick the grid and launch one kernel" helpers.
  - The tuning constants F4_L2_BLOCK_B, DFT_BLOCK_B, SCALE_BLOCK,
    TRANSPOSE_BLOCK.

The signatures below are the ones the harness calls -- your job is to fill
the bodies. When your code passes sanity_check.py, you're done.
"""

import math

import torch
import triton
import triton.language as tl


# Tunings -- GIVEN.
F4_L2_BLOCK_B = 2
DFT_BLOCK_B = 16
SCALE_BLOCK = 32
TRANSPOSE_BLOCK = 32


# =============================================================================
# Device-function helper: complex matmul
# =============================================================================
# Implement this once -- f1_kernel, f4_kernel_L2, and dft_kernel all call it.


@triton.jit
def _cdot(a_re, a_im, b_re, b_im):
    """Complex matmul Y = A @ B as four real tl.dot calls.

    Returns (y_re, y_im) in fp32 (out_dtype=tl.float32). Caller is responsible
    for any fp16 down-cast on store. Works at any matmul shape tl.dot accepts.

    Used by f1_kernel, f4_kernel_L2, and dft_kernel. Don't reimplement the
    four-tl.dot expansion at each call site -- implement once here, call
    everywhere.

    TODO: implement.
    """
    y_re = tl.dot(a_re, b_re) - tl.dot(a_im, b_im)
    y_im = tl.dot(a_re, b_im) + tl.dot(a_im, b_re)
    return y_re, y_im


# =============================================================================
# Chunk factorization for F6 / F7
# =============================================================================

def f6_factor(N: int) -> list[int]:
    """Factor N = 2^k into FFT chunks.

    Recipe: prefer 256-length chunks (radix-256, handled by f4_kernel_L2), then
    16-length (handled by dft_kernel via the padded radix-16 path), then a
    small leftover in {2, 4, 8} for the remaining bits. chunks[0] is the
    innermost (fastest) input axis. Examples:
        256 -> [256]                4096 -> [256, 16]
        65536 -> [256, 256]         1048576 -> [256, 256, 16]
        64 -> [16, 4]               2 -> [2]
    """
    k = N.bit_length() - 1          # N = 2^k
    a, rem = divmod(k, 8)           # 256-chunks (2^8)
    b, r = divmod(rem, 4)           # 16-chunks (2^4)
    chunks = [256] * a + [16] * b
    if r > 0:
        chunks.append(1 << r)      # leftover in {2, 4, 8}
    return chunks


f7_factor = f6_factor   # F7 reuses F6's chunk recipe


# =============================================================================
# F1: DFT as one dense complex matmul (four tl.dot)
# =============================================================================

@triton.jit
def f1_kernel(
    x_re_ptr, x_im_ptr,    # (B, N) fp16
    W_re_ptr, W_im_ptr,    # (N, N) fp16; W[n, k]
    y_re_ptr, y_im_ptr,    # (B, N) fp32
    B,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Y = X @ W^T as four (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N) tl.dot calls.

    Y[b, n] = sum_k X[b, k] * W[n, k]. Load W in transposed access
    (W_T[k, n] = W[n, k]) so tl.dot reads it the way it wants.

    Use `_cdot(x_re, x_im, W_T_re, W_T_im)` for the per-block complex matmul;
    accumulate its fp32 output into `acc_re` / `acc_im`.

    Dtype contract (same as F4): loads are fp16, `tl.dot` runs with
    `out_dtype=tl.float32` (handled by `_cdot`), accumulator is fp32, store
    is fp32. Allocations in `f1_alloc` already match this -- x_re/x_im are
    fp16, y_re/y_im are fp32.

    TODO: implement.
    """
    
    pid_m = tl.program_id(0)  # batch tile
    pid_n = tl.program_id(1)  # frequency tile

    m_off = pid_m * BLOCK_M
    n_off = pid_n * BLOCK_N

    rows = m_off + tl.arange(0, BLOCK_M)    # (BLOCK_M,)
    cols_n = n_off + tl.arange(0, BLOCK_N)  # (BLOCK_N,)

    acc_re = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in tl.range(0, N, BLOCK_K):
        cols_k = k_off + tl.arange(0, BLOCK_K)  # (BLOCK_K,)

        # x tile: (BLOCK_M, BLOCK_K)
        x_idx = rows[:, None] * N + cols_k[None, :]
        x_mask = (rows[:, None] < B) & (cols_k[None, :] < N)
        x_re_tile = tl.load(x_re_ptr + x_idx, mask=x_mask, other=0.0)
        x_im_tile = tl.load(x_im_ptr + x_idx, mask=x_mask, other=0.0)

        # W_T tile: (BLOCK_K, BLOCK_N) where W_T[k, n] = W[n, k]
        wt_idx = cols_n[None, :] * N + cols_k[:, None]
        wt_mask = (cols_k[:, None] < N) & (cols_n[None, :] < N)
        W_T_re = tl.load(W_re_ptr + wt_idx, mask=wt_mask, other=0.0)
        W_T_im = tl.load(W_im_ptr + wt_idx, mask=wt_mask, other=0.0)

        d_re, d_im = _cdot(x_re_tile, x_im_tile, W_T_re, W_T_im)
        acc_re += d_re
        acc_im += d_im

    out_idx = rows[:, None] * N + cols_n[None, :]
    out_mask = (rows[:, None] < B) & (cols_n[None, :] < N)
    tl.store(y_re_ptr + out_idx, acc_re, mask=out_mask)
    tl.store(y_im_ptr + out_idx, acc_im, mask=out_mask)


def f1_launch(x_re, x_im, W_re, W_im, y_re, y_im):
    """Grid: (cdiv(B, BLOCK_M), cdiv(N, BLOCK_N)). One program tiles a
    (BLOCK_M, BLOCK_N) output square. tl.dot needs all three dims >=16, so B
    should be >= 16.
    """
    B, N = x_re.shape
    BLOCK_M, BLOCK_K, BLOCK_N = 16, 16, 16
    grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(N, BLOCK_N))
    f1_kernel[grid](
        x_re, x_im, W_re, W_im, y_re, y_im,
        B, N=N,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )


# =============================================================================
# F2: radix-2 Cooley-Tukey, single program per signal
# =============================================================================
# F3 reuses this kernel! For F2, only BAILEY_EPILOGUE=False, STRIDED_STORE=False need to be implemented.
#
# Call-site cheatsheet:
#   F2 vanilla:  pid -> one signal in (B, N). Grid: (B,).
#                BAILEY_EPILOGUE=False, STRIDED_STORE=False.
#                OUTER_DIM and N_TOTAL unused (pass 1 / 0).
#                bt_*_ptr: pass tw_*_ptr again (sentinel; never read).
#   F2-A (F3):   pid -> (b, n1). Grid: (B*N1,). FFT length N=N2.
#                BAILEY_EPILOGUE=True, STRIDED_STORE=False.
#                OUTER_DIM=N1 (n1 = pid % N1).
#                bt_*_ptr: real Bailey twiddles shape (N1, N2).
#   F2-B (F3):   pid -> (b, k2). Grid: (B*N2,). FFT length N=N1.
#                BAILEY_EPILOGUE=False, STRIDED_STORE=True.
#                OUTER_DIM=N2, N_TOTAL=N1*N2.
#                bt_*_ptr: sentinel.

@triton.jit
def f2_kernel(
    x_re_ptr, x_im_ptr,        # (B, N) fp32 input
    y_re_ptr, y_im_ptr,        # (B, N) fp32 output (layout depends on STRIDED_STORE)
    tw_re_ptr, tw_im_ptr,      # (N/2,) fp32 radix-2 twiddles
    perm_ptr,                   # (N,) int32 bit-reversal index
    bt_re_ptr, bt_im_ptr,       # (OUTER_DIM, N) fp32 Bailey twiddles (BAILEY_EPILOGUE only)
    OUTER_DIM, N_TOTAL,
    N: tl.constexpr,
    LOG2_N: tl.constexpr,
    BAILEY_EPILOGUE: tl.constexpr,
    STRIDED_STORE: tl.constexpr,
):
    """Radix-2 Cooley-Tukey FFT in registers, with optional Bailey epilogue and
    strided store. log2(N) butterfly stages via tl.gather for partner shuffle.
    """
    pid = tl.program_id(0)
    idx = tl.arange(0, N)

    # Load all N samples with bit-reversal permutation applied at load time
    perm = tl.load(perm_ptr + idx)
    reg_re = tl.load(x_re_ptr + pid * N + perm)
    reg_im = tl.load(x_im_ptr + pid * N + perm)

    # log2(N) radix-2 butterfly stages (loop unrolled at compile time)
    for s in range(LOG2_N):
        half = 1 << s
        partner = idx ^ half                                # partner index
        tw_idx = (idx & (half - 1)) * (N >> (s + 1))      # twiddle table index

        tw_re = tl.load(tw_re_ptr + tw_idx)
        tw_im = tl.load(tw_im_ptr + tw_idx)

        p_re = tl.gather(reg_re, partner, 0)
        p_im = tl.gather(reg_im, partner, 0)

        is_hi = (idx & half) != 0

        # Both lanes of a pair need the same butterfly term t = tw * v_hi, where
        # v_hi is the bit-s-set element of the pair (== self on the hi lane,
        # == partner on the lo lane). Computing t once halves the per-stage
        # complex multiplies (and live temporaries) vs. forming tw*self and
        # tw*partner separately -- the spill pressure that dominates large N.
        hi_re = tl.where(is_hi, reg_re, p_re)
        hi_im = tl.where(is_hi, reg_im, p_im)
        t_re = tw_re * hi_re - tw_im * hi_im
        t_im = tw_re * hi_im + tw_im * hi_re

        # new_lo = v_lo + t (lo lane: self + t), new_hi = v_lo - t (hi lane: partner - t)
        reg_re = tl.where(is_hi, p_re - t_re, reg_re + t_re)
        reg_im = tl.where(is_hi, p_im - t_im, reg_im + t_im)

    # Optional Bailey twiddle multiply (F2-A: fused into F3 step 2)
    if BAILEY_EPILOGUE:
        n1 = pid % OUTER_DIM
        bt_re = tl.load(bt_re_ptr + n1 * N + idx)
        bt_im = tl.load(bt_im_ptr + n1 * N + idx)
        new_re = reg_re * bt_re - reg_im * bt_im
        new_im = reg_re * bt_im + reg_im * bt_re
        reg_re = new_re
        reg_im = new_im

    # Store: natural layout or transposed (absorbs T3 for F2-B)
    if STRIDED_STORE:
        b  = pid // OUTER_DIM
        k2 = pid % OUTER_DIM
        tl.store(y_re_ptr + b * N_TOTAL + idx * OUTER_DIM + k2, reg_re)
        tl.store(y_im_ptr + b * N_TOTAL + idx * OUTER_DIM + k2, reg_im)
    else:
        tl.store(y_re_ptr + pid * N + idx, reg_re)
        tl.store(y_im_ptr + pid * N + idx, reg_im)


def f2_launch(x_re, x_im, y_re, y_im, tw_re, tw_im, perm):
    """Grid: (B,). One program per length-N signal. Vanilla mode."""
    B, N = x_re.shape
    LOG2_N = int(math.log2(N))
    # One program holds the whole length-N signal in registers, so each thread
    # owns N / (32 * num_warps) elements. The default 4 warps leaves ~128
    # elements/thread at large N, which spills badly (the kernel's design wall).
    # Scale warps with N (target ~32 elements/thread) to spread the footprint
    # across more threads. Small N stays at 4 warps (no benefit from more).
    num_warps = max(4, min(16, N // 1024))
    f2_kernel[(B,)](
        x_re, x_im, y_re, y_im,
        tw_re, tw_im,
        perm,
        tw_re, tw_im,   # sentinel bt (never read in vanilla mode)
        1, 0,           # OUTER_DIM, N_TOTAL (unused)
        N=N, LOG2_N=LOG2_N,
        BAILEY_EPILOGUE=False, STRIDED_STORE=False,
        num_warps=num_warps,
    )


# =========================================================================x====
# transpose_kernel: (B, R, C) -> (B, C, R), paired re/im
# =============================================================================

@triton.jit
def transpose_kernel(
    x_re_ptr, x_im_ptr,     # (B*R*C,) fp16 or fp32 input
    y_re_ptr, y_im_ptr,     # (B*R*C,) fp16 or fp32 output
    R, C,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Logical (B, R, C) -> (B, C, R) transpose. Grid: (cdiv(R, BLOCK_R),
    cdiv(C, BLOCK_C), B). Each program copies a (BLOCK_R, BLOCK_C) tile.

    TODO: implement.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)   # (BLOCK_R,)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)   # (BLOCK_C,)
    mask = (r[:, None] < R) & (c[None, :] < C)

    in_idx = pid_b * R * C + r[:, None] * C + c[None, :]
    re = tl.load(x_re_ptr + in_idx, mask=mask)
    im = tl.load(x_im_ptr + in_idx, mask=mask)

    out_idx = pid_b * C * R + c[None, :] * R + r[:, None]
    tl.store(y_re_ptr + out_idx, re, mask=mask)
    tl.store(y_im_ptr + out_idx, im, mask=mask)


# =============================================================================
# F4: tcFFT radix-16 single-program FFT (N = 256, L = 2)
# =============================================================================
# See the kernel docstring for the tl.permute tuple-literal gotcha.

@triton.jit
def f4_kernel_L2(
    x_re_ptr, x_im_ptr,    # (B, 256) fp16
    y_re_ptr, y_im_ptr,    # (B, 256) or (B//M, 256, M) fp16
    F_re_ptr, F_im_ptr,    # (16, 16) fp16 -- F_16 DFT matrix
    tw_re_ptr, tw_im_ptr,  # (L=2, 16, 16) fp16 stacked stage twiddles
    B, M,
    BLOCK_B: tl.constexpr,
    STAGE_STOP: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """tcFFT length-256 FFT as two stages of (permute + per-stage twiddle +
    length-16 DFT via four tl.dot). fp16 storage, fp32 matmul accumulators.

    `STAGE_STOP` and `M` are both degenerate in vanilla F4 (`STAGE_STOP=L=2`,
    `M=1`). They exist so the same kernel handles two extra uses:
      - `STAGE_STOP=1`: stop after the s=0 stage, for the sanity_check.py
        stage-1 isolation test (no twiddles, no second matmul).
      - `M>1` with `STORE_T=True`: F7's fused FFT-m_0+T3, writing the
        transposed (rows_outer, 256, M) layout the next level expects.

    STORE_T=False (M=1): natural (B, 256) row-major output.
    STORE_T=True  (M>1): transposed (B//M, 256, M) output for F7 fusion.

    Each stage's four-`tl.dot` is one `_cdot` call; cast its fp32 output to
    fp16 before the next stage.

    Dtype contract:
        Loads:           fp16
        Reshape/permute: fp16 (free)
        tl.dot inputs:   fp16, out_dtype=tl.float32  (use _cdot)
        Twiddle mul:     fp32 * fp16 -> fp32
        Inter-stage:     .to(tl.float16) before next iter's reshape
        Store:           fp16
    Forgetting the inter-stage cast doubles register pressure and passes the
    L=2 tolerance, but fails as soon as F6 stacks more stages.

    Triton 3.6 gotcha -- tl.permute requires LITERAL tuples:
        tl.permute(x, (1, 0, 2))                  # works
        perm = (1, 0, 2); tl.permute(x, perm)     # fails
    Inline each stage's permute tuple at the call site; don't store the
    schedule in a loop variable.

    TODO: implement.
    """
    pid = tl.program_id(0)
    b = pid * BLOCK_B + tl.arange(0, BLOCK_B)     # (BLOCK_B,) row indices
    d = tl.arange(0, 16)

    # Load tile (BLOCK_B, 16, 16): element [bi, d0, d1] = x[b, d0*16 + d1].
    idx = b[:, None, None] * 256 + d[None, :, None] * 16 + d[None, None, :]
    bmask = b[:, None, None] < B
    xr = tl.load(x_re_ptr + idx, mask=bmask, other=0.0)
    xi = tl.load(x_im_ptr + idx, mask=bmask, other=0.0)

    # F_16 DFT matrix (16, 16).
    f_idx = d[:, None] * 16 + d[None, :]
    Fr = tl.load(F_re_ptr + f_idx)
    Fi = tl.load(F_im_ptr + f_idx)

    # ---- Stage 0: transform d0 (no twiddle). ----
    # permute (BLOCK_B, d0, d1) -> (d0, BLOCK_B, d1)
    s0r = tl.permute(xr, (1, 0, 2))
    s0i = tl.permute(xi, (1, 0, 2))
    s0r = tl.reshape(s0r, (16, BLOCK_B * 16))
    s0i = tl.reshape(s0i, (16, BLOCK_B * 16))
    o0r, o0i = _cdot(Fr, Fi, s0r, s0i)            # (16, BLOCK_B*16) = (e1, b, d1)
    o0r = tl.reshape(o0r, (16, BLOCK_B, 16)).to(tl.float16)
    o0i = tl.reshape(o0i, (16, BLOCK_B, 16)).to(tl.float16)

    if STAGE_STOP == 1:
        # final tile (e1, b, d1) -> (b, e1, d1); k = e1*16 + d1
        fr = tl.permute(o0r, (1, 0, 2))
        fi = tl.permute(o0i, (1, 0, 2))
        acc_re = fr.to(tl.float32)
        acc_im = fi.to(tl.float32)
    else:
        # ---- Stage 1: transform d1 (twiddle tw[1]). ----
        # (e1, b, d1) -> (d1, b, e1)
        p1r = tl.permute(o0r, (2, 1, 0))
        p1i = tl.permute(o0i, (2, 1, 0))
        # twiddle tw[1][m=d1, c=e1], broadcast over batch axis (1).
        tw_off = 1 * 16 * 16
        tw_idx = d[:, None] * 16 + d[None, :]     # (16, 16) [d1, e1]
        twr = tl.load(tw_re_ptr + tw_off + tw_idx)
        twi = tl.load(tw_im_ptr + tw_off + tw_idx)
        twr = twr[:, None, :]                     # (16, 1, 16)
        twi = twi[:, None, :]
        p1r_f = p1r.to(tl.float32)
        p1i_f = p1i.to(tl.float32)
        twr_f = twr.to(tl.float32)
        twi_f = twi.to(tl.float32)
        m1r = (p1r_f * twr_f - p1i_f * twi_f).to(tl.float16)
        m1i = (p1r_f * twi_f + p1i_f * twr_f).to(tl.float16)

        s1r = tl.reshape(m1r, (16, BLOCK_B * 16))
        s1i = tl.reshape(m1i, (16, BLOCK_B * 16))
        o1r, o1i = _cdot(Fr, Fi, s1r, s1i)        # (16, BLOCK_B*16) = (e0, b, e1)
        o1r = tl.reshape(o1r, (16, BLOCK_B, 16)).to(tl.float16)
        o1i = tl.reshape(o1i, (16, BLOCK_B, 16)).to(tl.float16)

        # final tile (e0, b, e1) -> (b, e0, e1); k = e0*16 + e1
        fr = tl.permute(o1r, (1, 0, 2))
        fi = tl.permute(o1i, (1, 0, 2))
        acc_re = fr.to(tl.float32)
        acc_im = fi.to(tl.float32)

    # ---- Store: (b, hi, lo) with k = hi*16 + lo. ----
    hi = tl.arange(0, 16)
    lo = tl.arange(0, 16)
    k = hi[None, :, None] * 16 + lo[None, None, :]
    if STORE_T:
        outer = b // M
        mm = b % M
        out_idx = outer[:, None, None] * 256 * M + k * M + mm[:, None, None]
    else:
        out_idx = b[:, None, None] * 256 + k
    omask = b[:, None, None] < B
    tl.store(y_re_ptr + out_idx, acc_re.to(tl.float16), mask=omask)
    tl.store(y_im_ptr + out_idx, acc_im.to(tl.float16), mask=omask)


# =============================================================================
# dft_kernel: padded length-R DFT for the small chunks (R in {2, 4, 8, 16})
# =============================================================================

@triton.jit
def dft_kernel(
    x_re_ptr, x_im_ptr,     # (rows, R) fp16
    y_re_ptr, y_im_ptr,     # (rows, R) or (rows//M, R, M) fp16
    M_re_ptr, M_im_ptr,     # (16, 16) fp16 padded-R DFT matrix
    rows, M,
    R: tl.constexpr,
    BLOCK_B: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Padded length-R DFT via a (16, 16) tl.dot. STORE_T toggles natural
    vs transposed output (same pattern as f4_kernel_L2).

    One `_cdot(x_re, x_im, MT_re, MT_im)` call replaces the four `tl.dot`
    expansions; cast its fp32 result to fp16 on store.

    TODO: implement.
    """
    pid = tl.program_id(0)
    b = pid * BLOCK_B + tl.arange(0, BLOCK_B)     # (BLOCK_B,) row indices
    n = tl.arange(0, 16)
    k = tl.arange(0, 16)

    # Load x (BLOCK_B, 16); zero-pad columns >= R.
    x_idx = b[:, None] * R + n[None, :]
    x_mask = (b[:, None] < rows) & (n[None, :] < R)
    xr = tl.load(x_re_ptr + x_idx, mask=x_mask, other=0.0)
    xi = tl.load(x_im_ptr + x_idx, mask=x_mask, other=0.0)

    # MT[n, k] = M[k, n]  (transposed load so _cdot computes X @ M^T).
    mt_idx = k[None, :] * 16 + n[:, None]
    MTr = tl.load(M_re_ptr + mt_idx)
    MTi = tl.load(M_im_ptr + mt_idx)

    yr, yi = _cdot(xr, xi, MTr, MTi)              # (BLOCK_B, 16) fp32

    # Store first R output columns.
    if STORE_T:
        outer = b // M
        mm = b % M
        out_idx = outer[:, None] * R * M + k[None, :] * M + mm[:, None]
    else:
        out_idx = b[:, None] * R + k[None, :]
    out_mask = (b[:, None] < rows) & (k[None, :] < R)
    tl.store(y_re_ptr + out_idx, yr.to(tl.float16), mask=out_mask)
    tl.store(y_im_ptr + out_idx, yi.to(tl.float16), mask=out_mask)


# =============================================================================
# bailey_scale_kernel: elementwise w_N^{n1 kM} multiply with optional fused T2
# =============================================================================

@triton.jit
def bailey_scale_kernel(
    x_re_ptr, x_im_ptr,     # (rows*m0*M,) fp16 input (logical (rows, m0, M))
    y_re_ptr, y_im_ptr,     # (rows*m0*M,) fp16 output ((rows, m0, M) or (rows, M, m0))
    tw_re_ptr, tw_im_ptr,   # (m0, M) fp16
    m0, M,
    BLOCK_M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Elementwise complex multiply by bt[n1, kM] over the (rows, m0, M) view.
    fp32 arithmetic, fp16 result. STORE_T=True fuses with a transpose to
    produce (rows, M, m0).

    Grid: (cdiv(m0, BLOCK_M0), cdiv(M, BLOCK_M), rows).

    TODO: implement.
    """
    pid_m0 = tl.program_id(0)
    pid_M = tl.program_id(1)
    row = tl.program_id(2)

    i0 = pid_m0 * BLOCK_M0 + tl.arange(0, BLOCK_M0)   # n1 in [0, m0)
    iM = pid_M * BLOCK_M + tl.arange(0, BLOCK_M)      # kM in [0, M)
    mask = (i0[:, None] < m0) & (iM[None, :] < M)

    in_idx = row * m0 * M + i0[:, None] * M + iM[None, :]
    xr = tl.load(x_re_ptr + in_idx, mask=mask)
    xi = tl.load(x_im_ptr + in_idx, mask=mask)

    tw_idx = i0[:, None] * M + iM[None, :]
    tr = tl.load(tw_re_ptr + tw_idx, mask=mask)
    ti = tl.load(tw_im_ptr + tw_idx, mask=mask)

    xr_f = xr.to(tl.float32)
    xi_f = xi.to(tl.float32)
    tr_f = tr.to(tl.float32)
    ti_f = ti.to(tl.float32)
    yr = (xr_f * tr_f - xi_f * ti_f).to(tl.float16)
    yi = (xr_f * ti_f + xi_f * tr_f).to(tl.float16)

    if STORE_T:
        # fuse T2: (rows, m0, M) -> (rows, M, m0)
        out_idx = row * m0 * M + iM[None, :] * m0 + i0[:, None]
    else:
        out_idx = in_idx
    tl.store(y_re_ptr + out_idx, yr, mask=mask)
    tl.store(y_im_ptr + out_idx, yi, mask=mask)


# =============================================================================
# Thin launch wrappers -- GIVEN, do not edit
# =============================================================================

def _transpose(in_re, in_im, out_re, out_im, B, R, C):
    """Logical (B, R, C) -> (B, C, R) transpose, paired re/im."""
    grid = (triton.cdiv(R, TRANSPOSE_BLOCK), triton.cdiv(C, TRANSPOSE_BLOCK), B)
    transpose_kernel[grid](
        in_re, in_im, out_re, out_im, R, C,
        BLOCK_R=TRANSPOSE_BLOCK, BLOCK_C=TRANSPOSE_BLOCK,
    )


def _fft_chunk(in_re, in_im, out_re, out_im, rows, m, plan, M=1, store_t=False):
    """Length-m FFT over `rows` contiguous (rows, m) signals.

    M / store_t control the output layout:
      store_t=False, M=1: natural (rows, m) row-major (F6 leaf path)
      store_t=True,  M>1: transposed (rows//M, m, M) (F7 fused FFT-m0+T3)
    """
    if m == 256:
        f4_plan = plan['f4_plan']
        f4_kernel_L2[(triton.cdiv(rows, F4_L2_BLOCK_B),)](
            in_re.view(rows, 256), in_im.view(rows, 256),
            out_re.view(rows, 256), out_im.view(rows, 256),
            f4_plan['F_re'], f4_plan['F_im'],
            f4_plan['tw_re'], f4_plan['tw_im'],
            rows, M,
            BLOCK_B=F4_L2_BLOCK_B, STAGE_STOP=f4_plan['L'], STORE_T=store_t,
            num_warps=4, num_stages=1,
        )
    else:
        M_re, M_im = plan['dft_mats'][m]
        dft_kernel[(triton.cdiv(rows, DFT_BLOCK_B),)](
            in_re.view(rows, m), in_im.view(rows, m),
            out_re.view(rows, m), out_im.view(rows, m),
            M_re, M_im, rows, M,
            R=m, BLOCK_B=DFT_BLOCK_B, STORE_T=store_t,
        )


def _scale(in_re, in_im, out_re, out_im, rows, m0, M, twr, twi, store_t=False):
    """Bailey scale over logical (rows, m0, M)."""
    grid = (triton.cdiv(m0, SCALE_BLOCK), triton.cdiv(M, SCALE_BLOCK), rows)
    bailey_scale_kernel[grid](
        in_re, in_im, out_re, out_im, twr, twi,
        m0, M, BLOCK_M0=SCALE_BLOCK, BLOCK_M=SCALE_BLOCK, STORE_T=store_t,
    )


def _lookup_tw(plan, m0, M, N_i):
    """Find the precomputed Bailey twiddle table for (m0, M, N_i) in plan['tw']."""
    for (a, b, n, tr, ti) in plan['tw']:
        if a == m0 and b == M and n == N_i:
            return tr, ti
    raise KeyError(f"no twiddle table for (m0={m0}, M={M}, N={N_i})")


# =============================================================================
# F3 pipeline: 4-step Bailey six-step (T1 -> F2-A -> T2 -> F2-B)
# =============================================================================

def f3_launch(in_re, in_im, out_re, out_im, mid_re, mid_im, plan, B):
    """Run the 4-step F3 pipeline. Buffer ping-pong: in -> mid -> out -> mid
    -> out. The Bailey twiddle fuses into F2-A (BAILEY_EPILOGUE=True), and
    the would-be T3 is absorbed by F2-B (STRIDED_STORE=True).

    Steps:
      1. T1 (transpose): x[b, n2, n1] -> A[b, n1, n2]
      2. F2-A:           length-N2 FFT over (B*N1) signals with Bailey epilogue
      3. T2 (transpose): Z[b, n1, k2] -> Z'[b, k2, n1]
      4. F2-B:           length-N1 FFT over (B*N2) signals with strided store

    TODO: implement.
    """
    N1 = plan['N1']
    N2 = plan['N2']
    LOG2_N1 = plan['LOG2_N1']
    LOG2_N2 = plan['LOG2_N2']

    # Step 1: T1  x[b, n2, n1] -> A[b, n1, n2]   (in -> mid)
    _transpose(in_re, in_im, mid_re, mid_im, B, N2, N1)

    # Step 2: F2-A  length-N2 FFT over (B*N1) signals + Bailey epilogue (mid -> out)
    f2_kernel[(B * N1,)](
        mid_re, mid_im, out_re, out_im,
        plan['tw_re_n2'], plan['tw_im_n2'],
        plan['perm_n2'],
        plan['bt_re'], plan['bt_im'],
        N1, 0,
        N=N2, LOG2_N=LOG2_N2,
        BAILEY_EPILOGUE=True, STRIDED_STORE=False,
    )

    # Step 3: T2  Z[b, n1, k2] -> Z'[b, k2, n1]   (out -> mid)
    _transpose(out_re, out_im, mid_re, mid_im, B, N1, N2)

    # Step 4: F2-B  length-N1 FFT over (B*N2) signals + strided store (mid -> out)
    f2_kernel[(B * N2,)](
        mid_re, mid_im, out_re, out_im,
        plan['tw_re_n1'], plan['tw_im_n1'],
        plan['perm_n1'],
        plan['tw_re_n1'], plan['tw_im_n1'],   # sentinel bt (never read)
        N2, N1 * N2,
        N=N1, LOG2_N=LOG2_N1,
        BAILEY_EPILOGUE=False, STRIDED_STORE=True,
    )


# =============================================================================
# F5 pipeline: 6-step Bailey at N1=N2=256 with F4 as inner FFT
# =============================================================================

def f5_launch(in_re, in_im, b0_re, b0_im, b1_re, b1_im, b2_re, b2_im, plan, B):
    """Run the 6-step F5 pipeline at N = 65536 = 256 * 256.

    Buffer ping-pong: in -> b0 -> b1 -> b0 -> b1 -> b2 -> b0 (final).
    The Bailey twiddle is NOT fused into F4 (F4 stays unmodified), so this is
    6 launches; F7 generalizes the fusion idea recursively.

    Steps:
      1. T1:    x[b, n2, n1] -> A[b, n1, n2]
      2. FFT-A: length-256 FFT along last axis -> Y[b, n1, k2]
      3. Scale: Z[b, n1, k2] = Y[b, n1, k2] * bt[n1, k2]
      4. T2:    Z[b, n1, k2] -> Z'[b, k2, n1]
      5. FFT-B: length-256 FFT along last axis -> V[b, k2, k1]
      6. T3:    V[b, k2, k1] -> X[b, k1, k2]   (final in b0)

    TODO: implement.
    """
    N1 = plan['N1']   # 256
    N2 = plan['N2']   # 256

    # 1. T1: x[b, n2, n1] -> A[b, n1, n2]            (in -> b0)
    _transpose(in_re, in_im, b0_re, b0_im, B, N2, N1)
    # 2. FFT-A: length-256 FFT over (B*N1) rows       (b0 -> b1)
    _fft_chunk(b0_re, b0_im, b1_re, b1_im, B * N1, 256, plan)
    # 3. Scale: Z = Y * bt[n1, k2] over (B, N1, N2)   (b1 -> b0)
    _scale(b1_re, b1_im, b0_re, b0_im, B, N1, N2, plan['bt_re'], plan['bt_im'])
    # 4. T2: Z[b, n1, k2] -> Z'[b, k2, n1]            (b0 -> b1)
    _transpose(b0_re, b0_im, b1_re, b1_im, B, N1, N2)
    # 5. FFT-B: length-256 FFT over (B*N2) rows       (b1 -> b2)
    _fft_chunk(b1_re, b1_im, b2_re, b2_im, B * N2, 256, plan)
    # 6. T3: V[b, k2, k1] -> X[b, k1, k2]             (b2 -> b0, final)
    _transpose(b2_re, b2_im, b0_re, b0_im, B, N2, N1)


# =============================================================================
# F6 / F7 recursion
# =============================================================================
# Per level i with chunks = [m_0, m_1, ..., m_{p-1}], M = prod(chunks[1:]):
#   T1 :       (rows, M, m_0) -> (rows, m_0, M)
#   recurse:   length-M FFT over (rows*m_0, M)
#   Scale :    y *= w_{N_i}^{n_1 k_M}            (n_1 = the m_0 digit)
#   T2 :       (rows, m_0, M) -> (rows, M, m_0)
#   FFT-m_0 :  length-m_0 FFT over (rows*M, m_0)
#   T3 :       (rows, M, m_0) -> (rows, m_0, M)   [F6 only; F7 fuses]

def _f6_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Recursive 2-factor Bailey split. Leaf (len(chunks)==1) is one
    _fft_chunk call; non-leaf is the 6-step pipeline above.

    Returns the (re, im) cycler-managed buffers holding the (rows, prod(chunks))
    FFT result.

    TODO: implement.
    """
    if len(chunks) == 1:
        m0 = chunks[0]
        out_re, out_im = cyc.next()
        _fft_chunk(cur_re, cur_im, out_re, out_im, rows, m0, plan)
        return out_re, out_im

    m0 = chunks[0]
    M = math.prod(chunks[1:])
    Ni = m0 * M

    # T1: (rows, M, m0) -> (rows, m0, M)
    t1_re, t1_im = cyc.next()
    _transpose(cur_re, cur_im, t1_re, t1_im, rows, M, m0)

    # recurse: length-M FFT over (rows*m0, M)
    rec_re, rec_im = _f6_rec(t1_re, t1_im, rows * m0, chunks[1:], plan, cyc)

    # Scale: y *= w_{Ni}^{n1 kM} over (rows, m0, M)
    tw_re, tw_im = _lookup_tw(plan, m0, M, Ni)
    sc_re, sc_im = cyc.next()
    _scale(rec_re, rec_im, sc_re, sc_im, rows, m0, M, tw_re, tw_im)

    # T2: (rows, m0, M) -> (rows, M, m0)
    t2_re, t2_im = cyc.next()
    _transpose(sc_re, sc_im, t2_re, t2_im, rows, m0, M)

    # FFT-m0: length-m0 FFT over (rows*M, m0)
    f_re, f_im = cyc.next()
    _fft_chunk(t2_re, t2_im, f_re, f_im, rows * M, m0, plan)

    # T3: (rows, M, m0) -> (rows, m0, M)
    t3_re, t3_im = cyc.next()
    _transpose(f_re, f_im, t3_re, t3_im, rows, M, m0)
    return t3_re, t3_im


def _f7_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Same recursion as _f6_rec but with Scale+T2 fused (store_t=True on
    bailey_scale_kernel) and FFT-m_0+T3 fused (store_t=True, M=M on the inner
    FFT kernel). Output should be bitwise-equal to _f6_rec.

    TODO: implement.
    """
    if len(chunks) == 1:
        m0 = chunks[0]
        out_re, out_im = cyc.next()
        _fft_chunk(cur_re, cur_im, out_re, out_im, rows, m0, plan)
        return out_re, out_im

    m0 = chunks[0]
    M = math.prod(chunks[1:])
    Ni = m0 * M

    # T1: (rows, M, m0) -> (rows, m0, M)
    t1_re, t1_im = cyc.next()
    _transpose(cur_re, cur_im, t1_re, t1_im, rows, M, m0)

    # recurse: length-M FFT over (rows*m0, M)
    rec_re, rec_im = _f7_rec(t1_re, t1_im, rows * m0, chunks[1:], plan, cyc)

    # Scale + T2 fused: scale over (rows, m0, M), write transposed (rows, M, m0)
    tw_re, tw_im = _lookup_tw(plan, m0, M, Ni)
    sc_re, sc_im = cyc.next()
    _scale(rec_re, rec_im, sc_re, sc_im, rows, m0, M, tw_re, tw_im, store_t=True)

    # FFT-m0 + T3 fused: length-m0 FFT over (rows*M, m0), write transposed
    # (rows, m0, M) via STORE_T with grouping M.
    out_re, out_im = cyc.next()
    _fft_chunk(sc_re, sc_im, out_re, out_im, rows * M, m0, plan, M=M, store_t=True)
    return out_re, out_im
