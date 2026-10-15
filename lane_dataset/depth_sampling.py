"""Predict ground depth at the exact pixel selected by stock nearest sampling."""
from functools import lru_cache
import numpy as np
from .preprocessing import _project_plane_nearest


def surface_normals(reference):
    tangent=np.gradient(reference.center_xyz,reference.s,axis=0)
    across=reference.right_xyz-reference.left_xyz
    normals=np.cross(tangent,across)
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    return np.concatenate([normals,normals])


@lru_cache(maxsize=4)
def crop_source_pixels(width,height,warp):
    matrix=np.asarray(warp,np.float32).reshape(3,3)
    x=np.broadcast_to(np.arange(width,dtype=np.float32),(height,width))
    y=np.broadcast_to(np.arange(height,dtype=np.float32)[:,None],(height,width))
    return (_project_plane_nearest(x,matrix,(512,256)),_project_plane_nearest(y,matrix,(512,256)))


def plane_depth_predictor(observation,optical_points,*,crop=False):
    normals=np.asarray(observation.surface_normals_world,float)
    if normals.shape!=optical_points.shape or not np.isfinite(normals).all():
        raise ValueError('one finite surface normal per queried boundary point required')
    lengths=np.linalg.norm(normals,axis=1)
    if np.any(lengths<.99) or np.any(lengths>1.01):raise ValueError('surface normals must be unit vectors')
    normals=normals@np.asarray(observation.camera_from_world)[:3,:3].T
    distance=np.einsum('ij,ij->i',normals,optical_points)
    k=np.asarray(observation.intrinsics,float)
    pixel_center=float(observation.depth_pixel_center_offset)
    if pixel_center not in (0.,.5):raise ValueError('unsupported sensor pixel center convention')
    if crop:
        height,width=observation.depth_m.shape
        source_x,source_y=crop_source_pixels(width,height,tuple(np.asarray(observation.model_to_source,np.float32).ravel()))
    def predict(index,x,y):
        if crop:x,y=source_x[y,x],source_y[y,x]
        ray=np.array([(x+pixel_center-k[0,2])/k[0,0],(y+pixel_center-k[1,2])/k[1,1],1.])
        denominator=float(normals[index]@ray)
        if abs(denominator)<1e-8:return float('inf')
        forward=distance[index]/denominator
        if forward<=0:return float('inf')
        return float(forward if observation.depth_mode=='forward' else forward*np.linalg.norm(ray))
    return predict
