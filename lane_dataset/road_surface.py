"""Survey physical road elevation independently of rendered depth sensors."""
from copy import deepcopy
import numpy as np


def survey_road_surface(world, route):
    """Preserve lane XY/arc and measure road elevation with collision rays.

    Curb boundaries use road-level toe elevation, not sidewalk tops. Extrapolate
    road hits 20/40 cm inward; both must hit road. Rendered XY agreement remains
    a separate 5 cm gate. Fixture samples are independently raycast interiors.
    """
    import carla
    corrected=deepcopy(route)
    rows=[]
    road_labels={carla.CityObjectLabel.Roads,carla.CityObjectLabel.RoadLines}

    def measure(point, identity):
        x,y,z=map(float,point)
        hits=world.cast_ray(carla.Location(x=x,y=y,z=z+1.),carla.Location(x=x,y=y,z=z-1.))
        eligible=[hit for hit in hits if hit.label in road_labels
                  and abs(hit.location.z-z)<=1. and np.isfinite(hit.location.z)]
        if not eligible:
            raise ValueError(f'no physical road surface at {identity}')
        hit=min(eligible,key=lambda value:abs(value.location.z-z))
        measured=[x,y,float(hit.location.z)]
        rows.append(dict(sample=identity,nominal_xyz=list(point),measured_xyz=measured,
            label=str(hit.label),vertical_correction_m=float(hit.location.z-z)))
        return np.asarray(measured)

    fixtures=[]
    for index,(left,right,center) in enumerate(zip(route['left_xyz'],route['right_xyz'],route['center_xyz'])):
        left,right,center=map(lambda v:np.asarray(v,float),(left,right,center))
        direction=right-left
        direction[2]=0.
        direction/=np.linalg.norm(direction)
        corrected['center_xyz'][index]=measure(center,f'center[{index}]').tolist()
        for name,point,inward in (('left_xyz',left,direction),('right_xyz',right,-direction)):
            near=measure(point+.2*inward,f'{name}[{index}]+20cm inward')
            far=measure(point+.4*inward,f'{name}[{index}]+40cm inward')
            corrected[name][index][2]=float(2*near[2]-far[2])
        for fraction in (.25,.75):
            fixtures.append(measure(left+fraction*(right-left),f'fixture[{index},{fraction}]').tolist())
    corrected['surface_fixture_xyz']=fixtures
    corrected['surface_elevation_source']='road-surface-survey.json'
    report=dict(method='Road/RoadLines collision rays; boundary toe elevation from 20/40 cm interior road samples',
        maximum_absolute_vertical_correction_m=max(abs(row['vertical_correction_m']) for row in rows),
        samples=rows,scope='elevation only; boundary XY unchanged and separately checked against rendered paint/edges')
    return corrected,report
