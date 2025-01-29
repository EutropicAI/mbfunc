from typing import Optional

import vapoursynth as vs

import mbfunc.mvsfunc as mvf


def mwlmask(
    clip: vs.VideoNode, l1: int = 80, h1: int = 96, h2: Optional[int] = None, l2: Optional[int] = None
) -> vs.VideoNode:
    # Constants values
    sbitPS = clip.format.bits_per_sample
    black = 0
    white = (1 << sbitPS) - 1
    l1_ = l1 << (sbitPS - 8)
    h1_ = h1 << (sbitPS - 8)

    if h2 is None:
        h2_ = white
    else:
        h2_ = h2 << (sbitPS - 8)

    if l2 is None:
        l2_ = white
    else:
        l2_ = l2 << (sbitPS - 8)

    # Base expression for the second ramp
    if h2_ >= white:
        expr2 = f"{white}"
    else:
        slope2 = white / (h2_ - l2_)
        expr2 = f"x {h2_} <= {white} x {l2_} < x {l2_} - {slope2} * {black} ? ?"

    # Expression for the first ramp
    slope1 = white / (h1_ - l1_)
    expr = f"x {l1_} <= {black} " f"x {h1_} < x {l1_} - {slope1} * {expr2} ? ?"

    # Process only luma
    plane_y = mvf.GetPlane(clip, 0)
    plane_y = plane_y.rgvs.RemoveGrain(4)
    mask = plane_y.std.Expr(expr=expr)
    return mask
