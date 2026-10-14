"""Define lane boundaries at the lane-facing stripe of double road markings."""
from copy import deepcopy
import numpy as np

DOUBLE_MARKINGS={'SolidSolid','SolidBroken','BrokenSolid','BrokenBroken'}


def lane_facing_marking_reference(route, markings):
    """Use the inner stripe center, keeping raw OpenDRIVE geometry for audit.

    CARLA double markings have stripe centers one declared marking width either
    side of the OpenDRIVE divider. The divider itself is unpainted. Choosing the
    stripe facing this lane avoids treating unpainted gap pixels as observations.
    This is a declared reference convention, not a fitted QA tolerance. Separate
    rendered validation must still establish <=5 cm agreement in every frame.
    """
    corrected=deepcopy(route)
    left=np.asarray(route['left_xyz'],float);right=np.asarray(route['right_xyz'],float)
    if len(markings)!=len(left):raise ValueError('one marking record per route cross-section required')
    normal=right-left;normal/=np.linalg.norm(normal,axis=1)[:,None]
    shifts=np.zeros((len(left),2))
    for i,record in enumerate(markings):
        for j,side in enumerate(('left','right')):
            mark=record[side]
            if mark['type'] in DOUBLE_MARKINGS:
                width=float(mark['width_m'])
                if not np.isfinite(width) or not 0<width<=.3:
                    raise ValueError('double marking width must be finite and at most 30 cm')
                shifts[i,j]=width
    corrected['nominal_left_xyz']=left.tolist();corrected['nominal_right_xyz']=right.tolist()
    corrected['nominal_center_xyz']=deepcopy(route['center_xyz'])
    left=left+shifts[:,0,None]*normal;right=right-shifts[:,1,None]*normal
    corrected['left_xyz']=left.tolist();corrected['right_xyz']=right.tolist()
    corrected['center_xyz']=((left+right)*.5).tolist()
    corrected['boundary_markings']=deepcopy(markings)
    corrected['lane_facing_insets_m']=shifts.tolist()
    corrected['boundary_reference_policy']='lane-facing stripe centers for double markings; nominal boundary for single markings or unpainted curb/road edge'
    corrected['arc_basis']='original OpenDRIVE traversal arc identities retained'
    return corrected
