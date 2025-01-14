import mvsfunc as mvf
import vapoursynth as vs
from havsfunc import cround, m4, mt_expand_multi, mt_inpand_multi, scale
from vapoursynth import core


def DeHalo_alpha(clp, rx=2.0, ry=2.0, darkstr=1.0, brightstr=1.0, lowsens=50, highsens=50, ss=1.5):
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error("DeHalo_alpha: This is not a clip")

    if clp.format.color_family == vs.RGB:
        raise vs.Error("DeHalo_alpha: RGB format is not supported")

    peak = (1 << clp.format.bits_per_sample) - 1 if clp.format.sample_type == vs.INTEGER else 1.0

    if clp.format.color_family != vs.GRAY:
        clp_orig = clp
        clp = mvf.GetPlane(clp, 0)
    else:
        clp_orig = None

    ox = clp.width
    oy = clp.height

    halos = clp.resize.Bicubic(m4(ox / rx), m4(oy / ry), filter_param_a=1 / 3, filter_param_b=1 / 3).resize.Bicubic(
        ox, oy, filter_param_a=1, filter_param_b=0
    )
    are = core.std.Expr([clp.std.Maximum(), clp.std.Minimum()], expr=["x y -"])
    ugly = core.std.Expr([halos.std.Maximum(), halos.std.Minimum()], expr=["x y -"])
    expr = f"y x - y y 0 = + / {peak} * {scale(lowsens, peak)} - y {scale(256, peak)} + {scale(512, peak)} / {highsens / 100} + *"
    so = core.std.Expr([ugly, are], expr=[expr])
    lets = core.std.MaskedMerge(halos, clp, so)
    if ss <= 1:
        remove = core.rgvs.Repair(clp, lets, mode=[1])
    else:
        remove = core.std.Expr(
            [
                core.std.Expr(
                    [
                        clp.resize.Lanczos(m4(ox * ss), m4(oy * ss)),
                        lets.std.Maximum().resize.Bicubic(
                            m4(ox * ss), m4(oy * ss), filter_param_a=1 / 3, filter_param_b=1 / 3
                        ),
                    ],
                    expr=["x y min"],
                ),
                lets.std.Minimum().resize.Bicubic(m4(ox * ss), m4(oy * ss), filter_param_a=1 / 3, filter_param_b=1 / 3),
            ],
            expr=["x y max"],
        ).resize.Lanczos(ox, oy)
    them = core.std.Expr([clp, remove], expr=[f"x y < x x y - {darkstr} * - x x y - {brightstr} * - ?"])

    if clp_orig is not None:
        them = core.std.ShufflePlanes([them, clp_orig], planes=[0, 1, 2], colorfamily=clp_orig.format.color_family)
    return them


def FineDehalo(
    src,
    rx=2.0,
    ry=None,
    thmi=80,
    thma=128,
    thlimi=50,
    thlima=100,
    darkstr=1.0,
    brightstr=1.0,
    showmask=0,
    excl=True,
    edgeproc=0.0,
):
    if not isinstance(src, vs.VideoNode):
        raise vs.Error("FineDehalo: This is not a clip")

    if src.format.color_family == vs.RGB:
        raise vs.Error("FineDehalo: RGB format is not supported")

    isInteger = src.format.sample_type == vs.INTEGER

    peak = (1 << src.format.bits_per_sample) - 1 if isInteger else 1.0

    if src.format.color_family != vs.GRAY:
        src_orig = src
        src = mvf.GetPlane(src, 0)
    else:
        src_orig = None

    if ry is None:
        ry = rx

    rx_i = cround(rx)
    ry_i = cround(ry)

    dehaloed = DeHalo_alpha(src, rx=rx, ry=ry, darkstr=darkstr, brightstr=brightstr)
    edges = core.std.Prewitt(src)

    # Keeps only the sharpest edges (line edges)
    strong = edges.std.Expr(
        expr=[
            f"x {scale(thmi, peak)} - {thma - thmi} / 255 *"
            if isInteger
            else f"x {scale(thmi, peak)} - {thma - thmi} / 255 * 0 max 1 min"
        ]
    )

    # Extends them to include the potential halos
    large = mt_expand_multi(strong, sw=rx_i, sh=ry_i)

    ### Exclusion zones ###

    # When two edges are close from each other (both edges of a single line or multiple parallel color bands), the halo removal oversmoothes them or makes seriously bleed the bands,
    # producing annoying artifacts. Therefore we have to produce a mask to exclude these zones from the halo removal

    # Includes more edges than previously, but ignores simple details
    light = edges.std.Expr(
        expr=[
            f"x {scale(thlimi, peak)} - {thlima - thlimi} / 255 *"
            if isInteger
            else f"x {scale(thlimi, peak)} - {thlima - thlimi} / 255 * 0 max 1 min"
        ]
    )

    # To build the exclusion zone, we make grow the edge mask, then shrink it to its original shape
    # During the growing stage, close adjacent edge masks will join and merge, forming a solid area, which will remain solid even after the shrinking stage

    # Mask growing
    shrink = mt_expand_multi(light, mode="ellipse", sw=rx_i, sh=ry_i)

    # At this point, because the mask was made of a shades of grey, we may end up with large areas of dark grey after shrinking
    # To avoid this, we amplify and saturate the mask here (actually we could even binarize it)
    shrink = shrink.std.Expr(expr=["x 4 *" if isInteger else "x 4 * 1 min"])

    # Mask shrinking
    shrink = mt_inpand_multi(shrink, mode="ellipse", sw=rx_i, sh=ry_i)

    # This mask is almost binary, which will produce distinct discontinuities once applied. Then we have to smooth it
    shrink = shrink.std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1]).std.Convolution(
        matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1]
    )

    ### Final mask building ###

    # Previous mask may be a bit weak on the pure edge side, so we ensure that the main edges are really excluded. We do not want them to be smoothed by the halo removal
    if excl:
        shr_med = core.std.Expr([strong, shrink], expr=["x y max"])
    else:
        shr_med = strong

    # Substracts masks and amplifies the difference to be sure we get 255 on the areas to be processed
    outside = core.std.Expr([large, shr_med], expr=["x y - 2 *" if isInteger else "x y - 2 * 0 max 1 min"])

    # If edge processing is required, adds the edgemask
    if edgeproc > 0:
        outside = core.std.Expr(
            [outside, strong], expr=[f"x y {edgeproc * 0.66} * +" if isInteger else f"x y {edgeproc * 0.66} * + 1 min"]
        )

    # Smooth again and amplify to grow the mask a bit, otherwise the halo parts sticking to the edges could be missed
    outside = outside.std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1]).std.Expr(
        expr=["x 2 *" if isInteger else "x 2 * 1 min"]
    )

    ### Masking ###

    if showmask <= 0:
        last = core.std.MaskedMerge(src, dehaloed, outside)

    if src_orig is not None:
        if showmask <= 0:
            return core.std.ShufflePlanes([last, src_orig], planes=[0, 1, 2], colorfamily=src_orig.format.color_family)
        elif showmask == 1:
            return core.resize.Bicubic(outside, format=src_orig.format.id)
        elif showmask == 2:
            return core.resize.Bicubic(shrink, format=src_orig.format.id)
        elif showmask == 3:
            return core.resize.Bicubic(edges, format=src_orig.format.id)
        else:
            return core.resize.Bicubic(strong, format=src_orig.format.id)
    else:
        if showmask <= 0:
            return last
        elif showmask == 1:
            return outside
        elif showmask == 2:
            return shrink
        elif showmask == 3:
            return edges
        else:
            return strong
