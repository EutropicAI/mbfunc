import vapoursynth as vs

core = vs.core

"""
Enhance high-frequency cotent
(steal from mawen1250's script)
"""

def mwenhance(
    diffclip: vs.VideoNode,
    chroma: bool = False,
    strength: float = 2.0,
    szrp8: int = 8,
    spwr: float = 1/4,
    sdmp_lo: int = 4,
    sdmp_hi: int = 48,
    soft: float = 0
) -> vs.VideoNode:

    if not isinstance(diffclip, vs.VideoNode):
        raise TypeError('mwenhance: this is not a clip')

    if strength < 0.0:
        return diffclip
		
	# Constants values for sharping lut
    bits = diffclip.format.bits_per_sample
    bps_mul8 = 1 << (bits - 8) 
    floor_val = 0
    ceil_val = (1 << bits) - 1
    neutral_val = 1 << (bits - 1)
    szrp = szrp8 * bps_mul8
    szrp8_sqr = szrp8 ** 2
    szrp_mul_strength = szrp * strength
    szrp8_sqr_plus_sdmp_lo = szrp8_sqr + sdmp_lo
    if sdmp_hi == 0:
        szrp8_div_sdmp_hi_power4_plus_1 = 1.0
    else:
        szrp8_div_sdmp_hi_power4_plus_1 = (szrp8 / sdmp_hi) ** 4 + 1.0
	
    # Applying enhanced expr
    abs_diff_expr = f"x {neutral_val} - abs"
    abs_diff8_expr = f"x {neutral_val} - {bps_mul8} / abs"
    diff8_sqr_expr = f"x {neutral_val} - {bps_mul8} / x {neutral_val} - {bps_mul8} / *"
    sign_mul_expr = f"x {neutral_val} - 0 >= 1 -1 ?"

    res1_expr = ( f"{abs_diff_expr} {szrp} / {spwr} pow "
        f"{szrp_mul_strength} * {sign_mul_expr} *" )

    res2_expr = ( f"{diff8_sqr_expr} "
        f"{szrp8_sqr_plus_sdmp_lo} * "
        f"{diff8_sqr_expr} {sdmp_lo} + "
        f"{szrp8_sqr} * / " )
    
    if sdmp_hi == 0:
        res3_expr = "0"
    else:
        res3_expr = f"{abs_diff8_expr} {sdmp_hi} / 4 pow"

    enhanced_expr = ( f"x {neutral_val} = "
        f"x {ceil_val} {floor_val} {neutral_val} "
        f"{res1_expr} "
        f"{res2_expr} * "
        f"{szrp8_div_sdmp_hi_power4_plus_1} * "
        f"1 {res3_expr} + / + max min ?" )

    if diffclip.format.num_planes == 1:
        enhanced_clip = core.akarin.Expr(diffclip, [expr])
    else:
        enhanced_clip = core.akarin.Expr(diffclip,[expr, expr if chroma else "",expr if chroma else ""])
	
    # Optional softening
    if soft > 0:
        softened_clip = core.rgvs.RemoveGrain(enhanced_clip, [19] * enhanced_clip.format.num_planes)
        if soft < 1:
            result = core.std.Merge(enhanced_clip, softened_clip, [soft])
        else:
            result = softened_clip
        limit_expr = f"x {neutral_val} - abs y {neutral_val} - abs <= x y ?"
        if enhanced_clip.format.num_planes == 1:
            result = core.akarin.Expr([enhanced_clip, result], [limit_expr])
        else:
            result = core.akarin.Expr([enhanced_clip, result], [limit_expr, limit_expr if chroma else "", limit_expr if chroma else ""])
        return result
    else:
        return enhanced_clip
