"""Shared core: two-point Richardson-extrapolated central finite differences.

Used to gradient-check adjoint/autodiff gradients against finite differences
without gating on a single raw FD(h), which on some GPU architectures (on an
Ada card the FD error GREW from h to h/2, 2.343e-07 -> 4.439e-07) can sit
under its own float32 rounding noise floor rather than reflecting a real
truncation trend. FD(h) and FD(h/2) are both computed and Richardson-combined
as FD_R = (4*FD(h/2) - FD(h)) / 3, which cancels the leading O(h^2) truncation
term and leaves O(h^4); comparing FD_R to the adjoint is architecture-stable
because it no longer depends on which of two noisy numbers is smaller.

This module intentionally does NOT own the perturbation or the cast to the
working dtype — callers pass an `evaluate(sign, h) -> float` closure that
does exactly that, in exactly their own order, so extracting this shared core
cannot change any existing caller's numerics (float64-then-cast-to-float32 vs
float32-throughout give different rounding in general).
"""


def richardson_fd_check(evaluate, h, adjoint_val):
    """Two-point Richardson-extrapolated central finite difference vs an
    adjoint value along one direction/component.

    `evaluate(sign, hh)` must return the scalar loss/value at the point
    perturbed by `sign * hh` along whatever direction the caller is
    checking (a single design voxel, or a full random direction vector);
    it is called as evaluate(+1, h), evaluate(-1, h), evaluate(+1, h/2),
    evaluate(-1, h/2), in that order.

    Returns a dict:
      fd_h, fd_h2       -- the raw central-difference quotients at h and h/2
      fd                -- Richardson extrapolate (4*fd_h2 - fd_h) / 3
      adjoint            -- adjoint_val, echoed back for convenience
      rel_err            -- |fd - adjoint_val| / (|fd| + 1e-12)
      fd_consistency      -- |fd_h - fd_h2| / (|fd| + 1e-12); large values with
                            a small rel_err mean the single-h FD had not
                            converged, not that the adjoint is wrong.
    """
    fd_h = (evaluate(+1, h) - evaluate(-1, h)) / (2 * h)
    h2 = h / 2
    fd_h2 = (evaluate(+1, h2) - evaluate(-1, h2)) / (2 * h2)
    fd = (4 * fd_h2 - fd_h) / 3
    rel_err = abs(fd - adjoint_val) / (abs(fd) + 1e-12)
    fd_consistency = abs(fd_h - fd_h2) / (abs(fd) + 1e-12)
    return {
        "fd_h": float(fd_h),
        "fd_h2": float(fd_h2),
        "fd": float(fd),
        "adjoint": float(adjoint_val),
        "rel_err": float(rel_err),
        "fd_consistency": float(fd_consistency),
    }
