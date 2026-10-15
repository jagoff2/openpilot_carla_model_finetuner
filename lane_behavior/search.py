"""Pure bounded candidate offsets; callers retain independent body checks."""

def _clearance_offsets(low, high, maximum_search_m, maximum_alternatives):
    """Bounded outward search, retaining the original two analytic seeds.

    Vertex projections and an axis-aligned ego width only estimate the needed
    displacement. On a curve the vehicle rotates alongside the obstacle, so
    the first estimate can fail even when another 5 cm provides ample room.
    This is a search proposal, never a relaxation of the swept-body verifier.
    """
    offsets = [float(low-.03), float(high+.03)]
    for seed, sign in ((offsets[0], -1.), (offsets[1], 1.)):
        for index in range(1, int(maximum_alternatives)):
            offset = seed+sign*.05*index
            if abs(offset) > maximum_search_m:
                break
            offsets.append(float(offset))
    return tuple(sorted(set(offsets), key=lambda value: (abs(value), value)))
