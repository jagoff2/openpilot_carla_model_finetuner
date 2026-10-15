"""Independent collision geometry at exact depth-camera sample rays."""
import numpy as np
from collections import Counter

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
        candidate_pixels=[tuple(np.rint(pixels[i]).astype(int)) for i in indices]
        tags=images.get(name+'_semantic_segmentation')
        if tags is not None:
            # Sensor calibration can use any independently surveyed visible
            # road, including roads outside this episode's planned lane.
            for y in np.linspace(profile.height*.52,profile.height*.98,16).astype(int):
                for x in np.linspace(profile.width*.12,profile.width*.88,13).astype(int):
                    if tags[y,x] in (int(carla.CityObjectLabel.Roads),int(carla.CityObjectLabel.RoadLines)):
                        candidate_pixels.append((x,y))
        measured=[];coordinates=[];samples=[];seen=set();diagnostics=Counter(candidate_pixels=len(candidate_pixels))
        origin=inverse[:3,3]
        # The fixed 3 m ray start must be above the road. Wide-camera pixels
        # near the bumper can hit visible road before that start; casting them
        # from underground instead finds a buried collision mesh. Preselect
        # rays with the independently surveyed local road plane, never with
        # measured depth or a favorable depth residual.
        progress,_=reference.project(origin[:2])
        ground=reference._center(progress)
        normal=np.cross(reference._center(progress,1),reference._right(progress)-reference._left(progress))
        normal/=np.linalg.norm(normal)
        camera_normal=transform[:3,:3]@normal
        camera_ground=(transform@np.r_[ground,1.])[:3]
        plane_distance=float(camera_normal@camera_ground)
        for x,y in candidate_pixels:
            if (x,y) in seen:continue
            seen.add((x,y))
            if tags is not None and tags[y,x] not in (int(carla.CityObjectLabel.Roads),int(carla.CityObjectLabel.RoadLines)):
                diagnostics['not_rendered_road']+=1
                continue
            ray=np.linalg.solve(profile.K,np.array([x+.5,y+.5,1.]))
            denominator=float(camera_normal@ray)
            plane_forward=plane_distance/denominator if abs(denominator)>1e-8 else float('inf')
            if not 4.<plane_forward<30.:
                diagnostics['local_road_plane_outside_fixture_range']+=1
                continue
            # The physical camera is inside the ego collision volume. Begin
            # ahead of that body; all fixture road hits must still be >4 m away.
            ray_start=(inverse@np.r_[ray*3.,1.])[:3]
            endpoint=(inverse@np.r_[ray*40.,1.])[:3]
            hits=world.cast_ray(carla.Location(*map(float,ray_start)),carla.Location(*map(float,endpoint)))
            # CARLA reports containing collision volumes as a zero-distance
            # NONE hit at the ray origin. It is not a surface intersection.
            hits=[hit for hit in hits if np.linalg.norm(np.array([hit.location.x,hit.location.y,hit.location.z])-ray_start)>1e-3]
            # Some maps contain invisible, unlabelled collision volumes above
            # the rendered road. A same-frame Road/RoadLines semantic pixel
            # lets us ignore NONE volumes, not named physical occluders. The
            # frontmost remaining road surface must still pass the unchanged
            # metric depth check; residuals never choose or discard a hit.
            if tags is not None:
                labelled=[hit for hit in hits if str(hit.label)!='NONE']
                diagnostics['unlabelled_collision_hits_ignored']+=len(hits)-len(labelled)
                hits=labelled
            if not hits:
                diagnostics['ray_missed']+=1
                continue
            hit=min(hits,key=lambda h:np.linalg.norm(np.array([h.location.x,h.location.y,h.location.z])-origin))
            if hit.label not in road_labels:
                diagnostics['occluder_'+str(hit.label)]+=1
                continue
            point=np.array([hit.location.x,hit.location.y,hit.location.z])
            coordinate=(transform@np.r_[point,1.])[:3]
            if not 4.<coordinate[2]<30.:
                diagnostics['outside_fixture_range']+=1
                continue
            if np.linalg.norm(coordinate)-coordinate[2]<=.18:
                diagnostics['insufficient_off_axis_separation']+=1
                continue
            measured.append(float(images[name+'_depth'][y,x]));coordinates.append(coordinate)
            samples.append(dict(world=point.tolist(),pixel=[int(x),int(y)],label=str(hit.label)))
            if len(samples)>=64:break
        if len(measured)>=3:
            result=validate_depth_geometry(np.asarray(measured),np.asarray(coordinates),tolerance_m=.15)
        else:result=dict(qualified=False,reason='insufficient_off_axis_known_surface_samples',depth_mode=None)
        result.update(samples=samples,measured_m=measured,camera_points=np.asarray(coordinates).tolist(),
            fixture='collision-surveyed road interiors at exact sensor pixel rays',ray_start_forward_m=3.,pixel_center_offset=.5,
            ray_selection='independent_local_road_plane_4_to_30m; same_frame_road_semantics_exclude_NONE_volumes',
            diagnostics=dict(diagnostics),camera_from_world=transform.tolist())
        report[name]=result
    return report
