"""Independent collision geometry at exact depth-camera sample rays."""
import numpy as np

from .visibility import project_world_points, validate_depth_geometry


def collision_depth_fixture(images, preprocessor, reference, world, optical_from_carla):
    import carla
    report={}
    # Interior road points stay in the narrow camera even when both boundaries
    # are outside its source image. No depth residual selects fixture samples.
    left,right=reference.left_xyz,reference.right_xyz
    points=np.concatenate([left+f*(right-left) for f in (.2,.4,.6,.8)])
    road_labels={carla.CityObjectLabel.Roads,carla.CityObjectLabel.RoadLines}
    for name,profile in zip(('narrow','wide'),preprocessor.profiles):
        transform=optical_from_carla@images[name+'_rgb_camera_from_world']
        inverse=np.linalg.inv(transform)
        pixels,optical=project_world_points(points,transform,profile.K)
        selected=(optical[:,2]>5)&(optical[:,2]<25)&(np.abs(optical[:,0])>.6)
        selected&=(np.linalg.norm(optical,axis=1)-optical[:,2]>.20)
        selected&=(pixels[:,0]>=3)&(pixels[:,0]<profile.width-3)&(pixels[:,1]>=3)&(pixels[:,1]<profile.height-3)
        indices=np.flatnonzero(selected)
        if len(indices)>48:indices=indices[np.unique(np.linspace(0,len(indices)-1,48).astype(int))]
        measured=[];coordinates=[];samples=[];seen=set()
        origin=inverse[:3,3]
        for index in indices:
            x,y=np.rint(pixels[index]).astype(int)
            if (x,y) in seen:continue
            seen.add((x,y))
            ray=np.linalg.solve(profile.K,np.array([x,y,1.]))
            endpoint=(inverse@np.r_[ray*40.,1.])[:3]
            hits=world.cast_ray(carla.Location(*map(float,origin)),carla.Location(*map(float,endpoint)))
            if not hits:continue
            hit=min(hits,key=lambda h:np.linalg.norm(np.array([h.location.x,h.location.y,h.location.z])-origin))
            if hit.label not in road_labels:continue
            point=np.array([hit.location.x,hit.location.y,hit.location.z])
            coordinate=(transform@np.r_[point,1.])[:3]
            if not 4.<coordinate[2]<30.:continue
            measured.append(float(images[name+'_depth'][y,x]));coordinates.append(coordinate)
            samples.append(dict(world=point.tolist(),pixel=[int(x),int(y)],label=str(hit.label)))
        if len(measured)>=3:
            result=validate_depth_geometry(np.asarray(measured),np.asarray(coordinates),tolerance_m=.15)
        else:result=dict(qualified=False,reason='insufficient_off_axis_known_surface_samples',depth_mode=None)
        result.update(samples=samples,measured_m=measured,camera_points=np.asarray(coordinates).tolist(),
            fixture='collision-surveyed road interiors at exact sensor pixel rays',camera_from_world=transform.tolist())
        report[name]=result
    return report
